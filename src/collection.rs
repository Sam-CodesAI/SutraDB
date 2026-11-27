use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::bm25::BM25Index;
use crate::distance::bq::BQVector;
use crate::distance::pq::PQCodebook;
use crate::distance::simd::{compute_vector_score, normalize_copy, normalize_in_place};
use crate::distance::pq::PQConfig;
use crate::distance::{BQArena, Metric, PQArena, SQ8Arena, VectorArena};
use crate::error::SutraError;
use crate::filter::FilterEngine;
use crate::fusion::{linear_score_fusion, reciprocal_rank_fusion};
use crate::index::hnsw::{HNSWConfig, HNSWIndex};
use crate::index::ivf::{auto_n_lists, IVFConfig, IVFIndex};
use crate::storage::{
    validate_collection_name, BinaryStorage, IndexStorage, MmapVectorStorage, PersistedIndexes,
    WalRecord, WriteAheadLog, HEADER_MAGIC_V4,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NativeDocument {
    pub id: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default)]
    pub sparse_vector: Option<crate::sparse::SparseVector>,
    #[serde(default)]
    pub colbert_tokens: Option<Vec<Vec<f32>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeSearchResult {
    pub id: String,
    pub score: f32,
    pub text: String,
    pub metadata: Value,
    pub dense_score: Option<f32>,
    pub bm25_score: Option<f32>,
    #[serde(default)]
    pub sparse_score: Option<f32>,
    #[serde(default)]
    pub colbert_score: Option<f32>,
    #[serde(default)]
    pub namespace: Option<String>,
}

/// Industrial Multi-Tenant Collection engine supporting HNSW, SQ8, BM25, and Hybrid RRF.
pub struct NativeCollection {
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
    pub storage_path: Option<PathBuf>,
    pub documents: Vec<NativeDocument>,
    pub vectors: VectorArena,
    pub mmap_storage: Option<MmapVectorStorage>,
    pub vectors_evicted: bool,
    pub sq8_vectors: SQ8Arena,
    pub use_sq8: bool,
    pub bq_vectors: BQArena,
    pub bq_centroid: Option<Vec<f32>>,
    pub use_bq: bool,
    pub pq_codebook: Option<PQCodebook>,
    pub pq_vectors: PQArena,
    pub use_pq: bool,
    pub id_to_index: HashMap<String, usize>,
    pub tombstones: Vec<bool>,
    pub bm25: BM25Index,
    pub hnsw: Option<HNSWIndex>,
    pub ivf: Option<IVFIndex>,
    pub metadata_index: crate::filter::SecondaryMetadataIndex,
    pub sparse_index: crate::sparse::SparseIndex,
    pub colbert_index: crate::colbert::ColbertIndex,
    pub wal: Option<WriteAheadLog>,
    pub dirty_index: bool,
}

impl NativeCollection {
    pub fn new(
        name: String,
        dimension: usize,
        metric: Metric,
        storage_path: Option<PathBuf>,
        enable_wal: bool,
    ) -> Result<Self, SutraError> {
        Self::with_analyzer(
            name,
            dimension,
            metric,
            storage_path,
            enable_wal,
            crate::bm25::Analyzer::default(),
        )
    }

    pub fn with_analyzer(
        name: String,
        dimension: usize,
        metric: Metric,
        storage_path: Option<PathBuf>,
        enable_wal: bool,
        analyzer: crate::bm25::Analyzer,
    ) -> Result<Self, SutraError> {
        validate_collection_name(&name)?;
        let mut wal = None;
        if enable_wal {
            if let Some(ref sp) = storage_path {
                let wal_path = sp.with_extension("wal");
                wal = Some(WriteAheadLog::open(wal_path)?);
            }
        }

        let mut col = Self {
            name,
            dimension,
            metric,
            storage_path,
            documents: Vec::new(),
            vectors: VectorArena::new(dimension),
            mmap_storage: None,
            vectors_evicted: false,
            sq8_vectors: SQ8Arena::new(dimension),
            use_sq8: false,
            bq_vectors: BQArena::new(dimension),
            bq_centroid: None,
            use_bq: false,
            pq_codebook: None,
            pq_vectors: PQArena::new(0),
            use_pq: false,
            id_to_index: HashMap::new(),
            tombstones: Vec::new(),
            bm25: BM25Index::with_analyzer(1.5, 0.75, analyzer),
            hnsw: None,
            ivf: None,
            metadata_index: crate::filter::SecondaryMetadataIndex::new(),
            sparse_index: crate::sparse::SparseIndex::new(),
            colbert_index: crate::colbert::ColbertIndex::new(dimension),
            wal,
            dirty_index: false,
        };

        col.replay_wal_if_present()?;
        Ok(col)
    }

    /// Returns an FP32 vector slice from RAM if resident, zero-copy mmap if evicted, or SQ8 dequantized fallback.
    #[inline]
    pub fn get_vector(&self, idx: usize) -> Option<Cow<'_, [f32]>> {
        if idx >= self.documents.len() {
            return None;
        }
        if !self.vectors_evicted {
            return self.vectors.get(idx).map(Cow::Borrowed);
        }
        if let Some(ref mmap) = self.mmap_storage {
            if mmap.header().quant_mode == 0 {
                if let Ok(slice) = mmap.try_get_vector_f32(idx) {
                    return Some(Cow::Borrowed(slice));
                }
            }
        }
        if self.use_sq8 && idx < self.sq8_vectors.len() {
            return Some(Cow::Owned(self.sq8_vectors.dequantize(idx)));
        }
        None
    }

    /// Re-materializes evicted FP32 vectors back into the contiguous RAM arena before in-memory mutations.
    fn ensure_vectors_in_ram(&mut self) -> Result<(), SutraError> {
        if !self.vectors_evicted {
            return Ok(());
        }
        let n = self.documents.len();
        let mut restored = Vec::with_capacity(n * self.dimension);
        for i in 0..n {
            if let Some(ref mmap) = self.mmap_storage {
                if mmap.header().quant_mode == 0 {
                    restored.extend_from_slice(mmap.try_get_vector_f32(i)?);
                    continue;
                }
            }
            if self.use_sq8 && i < self.sq8_vectors.len() {
                restored.extend_from_slice(&self.sq8_vectors.dequantize(i));
            } else {
                restored.resize(restored.len() + self.dimension, 0.0);
            }
        }
        self.vectors = VectorArena {
            data: restored,
            dim: self.dimension,
        };
        self.vectors_evicted = false;
        self.mmap_storage = None;
        Ok(())
    }

    /// Evicts in-RAM FP32 arena when quantization is enabled over an active FP32 mmap snapshot.
    fn maybe_evict_fp32_arena(&mut self) {
        if let Some(ref mmap) = self.mmap_storage {
            if mmap.header().quant_mode == 0 && (self.use_sq8 || self.use_bq || self.use_pq) {
                self.vectors.clear();
                self.vectors.shrink_to_fit();
                self.vectors_evicted = true;
            }
        }
    }

    fn replay_wal_if_present(&mut self) -> Result<usize, SutraError> {
        let records = match self.wal.as_mut() {
            Some(w) => w.replay()?,
            None => return Ok(0),
        };
        let count = records.len();
        self.apply_wal_records(records)?;
        Ok(count)
    }

    pub fn replay_wal_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), SutraError> {
        let mut temp_wal = WriteAheadLog::open(path)?;
        let records = temp_wal.replay()?;
        self.apply_wal_records(records)
    }

    fn parse_wal_sparse(v: &Value) -> Result<Option<crate::sparse::SparseVector>, SutraError> {
        if let Some(map) = v.as_object() {
            let mut pairs: Vec<(u32, f32)> = map
                .iter()
                .filter_map(|(k, val)| {
                    let idx = k.parse::<u32>().ok()?;
                    let f = val.as_f64()? as f32;
                    Some((idx, f))
                })
                .collect();
            pairs.sort_by_key(|p| p.0);
            let idxs = pairs.iter().map(|p| p.0).collect();
            let vals = pairs.iter().map(|p| p.1).collect();
            Ok(Some(crate::sparse::SparseVector::try_new(idxs, vals)?))
        } else {
            Ok(None)
        }
    }

    fn parse_wal_colbert(v: &Value) -> Option<Vec<Vec<f32>>> {
        v.as_array().map(|arr| {
            arr.iter()
                .filter_map(|tok_arr| {
                    tok_arr.as_array().map(|vec_arr| {
                        vec_arr
                            .iter()
                            .filter_map(|x| x.as_f64().map(|f| f as f32))
                            .collect()
                    })
                })
                .collect()
        })
    }

    fn apply_wal_records(&mut self, records: Vec<WalRecord>) -> Result<(), SutraError> {
        let is_cosine = self.metric == Metric::Cosine;
        for r in records {
            match r.op.as_str() {
                "insert" => {
                    let mut docs = Vec::new();
                    let mut vecs = Vec::new();
                    for d_val in r.docs {
                        let id = d_val.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let text = d_val.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let namespace = d_val.get("namespace").and_then(|v| v.as_str()).map(|s| s.to_string());
                        let metadata = d_val.get("metadata").cloned().unwrap_or(Value::Object(serde_json::Map::new()));

                        let vec_arr = d_val.get("vector").and_then(|v| v.as_array());
                        let vec: Vec<f32> = match vec_arr {
                            Some(arr) => arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect(),
                            None => vec![0.0; self.dimension],
                        };

                        let sparse_vector = match d_val.get("sparse_vector") {
                            Some(v) => Self::parse_wal_sparse(v)?,
                            None => None,
                        };
                        let colbert_tokens = d_val.get("colbert_tokens").and_then(Self::parse_wal_colbert);

                        docs.push(NativeDocument { id, namespace, text, metadata, sparse_vector, colbert_tokens });
                        vecs.push(vec);
                    }
                    self.insert_internal(docs, vecs, false)?;
                }
                "update" => {
                    for d_val in r.docs {
                        if let Some(id) = d_val.get("id").and_then(|v| v.as_str()) {
                            if let Some(&idx) = self.id_to_index.get(id) {
                                if idx < self.tombstones.len() && !self.tombstones[idx] {
                                    if let Some(meta_val) = d_val.get("metadata").filter(|v| !v.is_null()) {
                                        if let Some(target) = self.documents.get_mut(idx) {
                                            target.metadata = meta_val.clone();
                                        }
                                        self.metadata_index.remove_document(idx as u32);
                                        self.metadata_index.index_document(idx as u32, meta_val);
                                    }
                                    if d_val.get("namespace").is_some() {
                                        let ns = d_val.get("namespace").and_then(|v| v.as_str()).map(|s| s.to_string());
                                        if let Some(target) = self.documents.get_mut(idx) {
                                            target.namespace = ns;
                                        }
                                    }
                                    if let Some(sp_val) = d_val.get("sparse_vector") {
                                        let parsed_sp = if sp_val.is_null() { None } else { Self::parse_wal_sparse(sp_val)? };
                                        self.sparse_index.remove_document(idx as u32);
                                        if let Some(ref sv) = parsed_sp {
                                            self.sparse_index.add_document(idx as u32, sv);
                                        }
                                        if let Some(target) = self.documents.get_mut(idx) {
                                            target.sparse_vector = parsed_sp;
                                        }
                                    }
                                    if let Some(cb_val) = d_val.get("colbert_tokens") {
                                        let parsed_cb = if cb_val.is_null() { None } else { Self::parse_wal_colbert(cb_val) };
                                        self.colbert_index.remove_document(idx as u32);
                                        if let Some(ref tokens) = parsed_cb {
                                            self.colbert_index.try_add_document(idx as u32, tokens.clone())?;
                                        }
                                        if let Some(target) = self.documents.get_mut(idx) {
                                            target.colbert_tokens = parsed_cb;
                                        }
                                    }
                                    if let Some(new_text) = d_val.get("text").and_then(|v| v.as_str()) {
                                        if let Some(target) = self.documents.get_mut(idx) {
                                            let old_text = target.text.clone();
                                            target.text = new_text.to_string();
                                            self.bm25.update_document(idx as u32, &old_text, new_text);
                                        }
                                    }
                                    if let Some(vec_arr) = d_val.get("vector").and_then(|v| v.as_array()) {
                                        let mut vec: Vec<f32> = vec_arr
                                            .iter()
                                            .filter_map(|x| x.as_f64().map(|f| f as f32))
                                            .collect();
                                        if vec.len() != self.dimension {
                                            return Err(SutraError::DimensionMismatch {
                                                expected: self.dimension,
                                                got: vec.len(),
                                            });
                                        }
                                        if is_cosine {
                                            normalize_in_place(&mut vec);
                                        }
                                        self.ensure_vectors_in_ram()?;
                                        self.mmap_storage = None;
                                        if idx < self.vectors.len() {
                                            self.vectors.set_slice(idx, &vec);
                                        }
                                        if self.use_sq8 && idx < self.sq8_vectors.len() {
                                            self.sq8_vectors.set_vec(idx, &vec);
                                        }
                                        if self.use_bq && idx < self.bq_vectors.len() {
                                            self.bq_vectors
                                                .set_vec(idx, &vec, self.bq_centroid.as_deref());
                                        }
                                        if self.use_pq {
                                            if let Some(ref cb) = self.pq_codebook {
                                                if idx < self.pq_vectors.len() {
                                                    let mut codes = vec![0u8; cb.m];
                                                    cb.try_encode_into(&vec, &mut codes)?;
                                                    self.pq_vectors.set_codes(idx, &codes);
                                                }
                                            }
                                        }
                                        if self.ivf.is_some() {
                                            self.dirty_index = true;
                                        }
                                        if let Some(ref mut hnsw) = self.hnsw {
                                            hnsw.delete_node(idx as u32, &self.vectors, self.metric, is_cosine);
                                            hnsw.insert(idx as u32, &vec, &self.vectors, self.metric, is_cosine)?;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                "delete" => {
                    self.delete_internal(&r.ids, false)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn insert(
        &mut self,
        documents: Vec<NativeDocument>,
        vectors: Vec<Vec<f32>>,
    ) -> Result<usize, SutraError> {
        self.insert_internal(documents, vectors, true)
    }

    /// Zero-copy batch insert from a single contiguous row-major flat `f32` buffer (`n * dimension`).
    pub fn insert_flat(
        &mut self,
        documents: Vec<NativeDocument>,
        flat_vectors: Vec<f32>,
    ) -> Result<usize, SutraError> {
        self.insert_flat_internal(documents, flat_vectors, true)
    }

    fn insert_internal(
        &mut self,
        documents: Vec<NativeDocument>,
        vectors: Vec<Vec<f32>>,
        log_to_wal: bool,
    ) -> Result<usize, SutraError> {
        if documents.len() != vectors.len() {
            return Err(SutraError::DimensionMismatch {
                expected: documents.len(),
                got: vectors.len(),
            });
        }
        let mut flat = Vec::with_capacity(documents.len() * self.dimension);
        for vec in &vectors {
            if vec.len() != self.dimension {
                return Err(SutraError::DimensionMismatch {
                    expected: self.dimension,
                    got: vec.len(),
                });
            }
            flat.extend_from_slice(vec);
        }
        self.insert_flat_internal(documents, flat, log_to_wal)
    }

    fn insert_flat_internal(
        &mut self,
        documents: Vec<NativeDocument>,
        mut flat_vectors: Vec<f32>,
        log_to_wal: bool,
    ) -> Result<usize, SutraError> {
        let expected_flat = documents.len().checked_mul(self.dimension).ok_or_else(|| {
            SutraError::DimensionMismatch {
                expected: usize::MAX,
                got: flat_vectors.len(),
            }
        })?;
        if flat_vectors.len() != expected_flat {
            return Err(SutraError::DimensionMismatch {
                expected: expected_flat,
                got: flat_vectors.len(),
            });
        }

        // Phase 1: Validate all metadata/sparse/colbert inputs upfront before mutating state or WAL
        for doc in &documents {
            if let Some(ref sv) = doc.sparse_vector {
                if sv.indices.len() != sv.values.len() {
                    return Err(SutraError::DimensionMismatch {
                        expected: sv.indices.len(),
                        got: sv.values.len(),
                    });
                }
            }
            if let Some(ref tokens) = doc.colbert_tokens {
                for tok in tokens {
                    if tok.len() != self.dimension {
                        return Err(SutraError::DimensionMismatch {
                            expected: self.dimension,
                            got: tok.len(),
                        });
                    }
                }
            }
        }

        self.ensure_vectors_in_ram()?;
        if !documents.is_empty() {
            self.mmap_storage = None;
        }

        // Phase 2: Normalize in place across the flat slab, pre-encode PQ, and prepare WAL
        let is_cosine = self.metric == Metric::Cosine;
        let dim = self.dimension;
        let n_docs = documents.len();
        if is_cosine && dim > 0 {
            for row in flat_vectors.chunks_exact_mut(dim) {
                normalize_in_place(row);
            }
        }

        let pq_flat = if self.use_pq {
            if let Some(ref cb) = self.pq_codebook {
                Some(cb.try_encode_flat(&flat_vectors, n_docs)?)
            } else {
                None
            }
        } else {
            None
        };
        let pq_m = self.pq_codebook.as_ref().map_or(0, |cb| cb.m);

        if log_to_wal && self.wal.is_some() && !documents.is_empty() {
            let mut wal_docs = Vec::with_capacity(n_docs);
            for (i, doc) in documents.iter().enumerate() {
                let vec_slice = &flat_vectors[i * dim..(i + 1) * dim];
                let doc_json = serde_json::json!({
                    "id": doc.id,
                    "namespace": doc.namespace,
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "vector": vec_slice,
                    "sparse_vector": doc.sparse_vector.as_ref().map(|sv| {
                        let mut m = serde_json::Map::new();
                        for (&idx, &val) in sv.indices.iter().zip(sv.values.iter()) {
                            m.insert(idx.to_string(), serde_json::Value::from(val));
                        }
                        serde_json::Value::Object(m)
                    }),
                    "colbert_tokens": doc.colbert_tokens
                });
                wal_docs.push(doc_json);
            }
            if let Some(ref mut wal) = self.wal {
                wal.append(&WalRecord {
                    op: "insert".to_string(),
                    docs: wal_docs,
                    ids: Vec::new(),
                })?;
            }
        }

        // Phase 3: Commit in-memory mutations atomically into contiguous arenas
        self.vectors.data.reserve(n_docs * dim);
        if self.use_sq8 {
            self.sq8_vectors.reserve(n_docs);
        }

        for (i, doc) in documents.into_iter().enumerate() {
            let vec_slice = &flat_vectors[i * dim..(i + 1) * dim];
            let pq_slice = pq_flat.as_ref().map(|c| &c[i * pq_m..(i + 1) * pq_m]);

            if let Some(&existing_idx) = self.id_to_index.get(&doc.id) {
                self.vectors.set_slice(existing_idx, vec_slice);
                if self.use_sq8 {
                    self.sq8_vectors.set_vec(existing_idx, vec_slice);
                }
                if self.use_bq {
                    self.bq_vectors
                        .set_vec(existing_idx, vec_slice, self.bq_centroid.as_deref());
                }
                if let Some(codes) = pq_slice {
                    self.pq_vectors.set_codes(existing_idx, codes);
                }
                self.metadata_index.remove_document(existing_idx as u32);
                self.metadata_index.index_document(existing_idx as u32, &doc.metadata);
                self.sparse_index.remove_document(existing_idx as u32);
                if let Some(ref sv) = doc.sparse_vector {
                    self.sparse_index.add_document(existing_idx as u32, sv);
                }
                self.colbert_index.remove_document(existing_idx as u32);
                if let Some(ref tokens) = doc.colbert_tokens {
                    self.colbert_index.try_add_document(existing_idx as u32, tokens.clone())?;
                }
                let old_text = &self.documents[existing_idx].text;
                self.bm25.update_document(existing_idx as u32, old_text, &doc.text);
                self.documents[existing_idx] = doc;
                if existing_idx < self.tombstones.len() {
                    self.tombstones[existing_idx] = false;
                }
                if let Some(ref mut hnsw) = self.hnsw {
                    hnsw.delete_node(existing_idx as u32, &self.vectors, self.metric, is_cosine);
                    hnsw.insert(existing_idx as u32, vec_slice, &self.vectors, self.metric, is_cosine)?;
                    if self.ivf.is_some() {
                        self.dirty_index = true;
                    }
                } else {
                    self.dirty_index = true;
                }
                continue;
            }

            let idx = self.documents.len();
            self.id_to_index.insert(doc.id.clone(), idx);

            if self.use_sq8 {
                self.sq8_vectors.push_vec(vec_slice);
            }
            if self.use_bq {
                self.bq_vectors
                    .push_vec(vec_slice, self.bq_centroid.as_deref());
            }
            if let Some(codes) = pq_slice {
                self.pq_vectors.push_codes(codes);
            }

            self.metadata_index.index_document(idx as u32, &doc.metadata);
            if let Some(ref sv) = doc.sparse_vector {
                self.sparse_index.add_document(idx as u32, sv);
            }
            if let Some(ref tokens) = doc.colbert_tokens {
                self.colbert_index.try_add_document(idx as u32, tokens.clone())?;
            }
            self.bm25.add_documents(std::slice::from_ref(&doc.text));
            self.vectors.push_slice(vec_slice);
            self.documents.push(doc);
            self.tombstones.push(false);
            if let Some(ref mut hnsw) = self.hnsw {
                hnsw.insert(idx as u32, vec_slice, &self.vectors, self.metric, is_cosine)?;
                if self.ivf.is_some() {
                    self.dirty_index = true;
                }
            } else {
                self.dirty_index = true;
            }
        }

        Ok(self.count())
    }

    /// Idempotent batch upsert.
    pub fn upsert(
        &mut self,
        documents: Vec<NativeDocument>,
        vectors: Vec<Vec<f32>>,
    ) -> Result<usize, SutraError> {
        self.insert(documents, vectors)
    }

    /// In-place partial metadata patch without reallocating or modifying dense vectors.
    pub fn update_metadata(
        &mut self,
        doc_id: &str,
        new_metadata: Value,
    ) -> Result<bool, SutraError> {
        let idx = match self.id_to_index.get(doc_id) {
            Some(&i) => i,
            None => return Ok(false),
        };

        if idx < self.tombstones.len() && self.tombstones[idx] {
            return Ok(false);
        }

        if let Some(target) = self.documents.get_mut(idx) {
            target.metadata = new_metadata.clone();
            self.metadata_index.remove_document(idx as u32);
            self.metadata_index.index_document(idx as u32, &new_metadata);
        }

        if let Some(ref mut wal) = self.wal {
            let record = WalRecord {
                op: "update".to_string(),
                docs: vec![serde_json::json!({
                    "id": doc_id,
                    "metadata": new_metadata
                })],
                ids: Vec::new(),
            };
            wal.append(&record)?;
        }

        Ok(true)
    }

    pub fn delete(&mut self, ids: &[String]) -> Result<usize, SutraError> {
        self.delete_internal(ids, true)
    }

    fn delete_internal(&mut self, ids: &[String], log_to_wal: bool) -> Result<usize, SutraError> {
        let mut deleted_count = 0;
        let mut deleted_ids = Vec::new();
        let is_cosine = self.metric == Metric::Cosine;

        for id in ids {
            if let Some(&idx) = self.id_to_index.get(id) {
                if idx < self.tombstones.len() && !self.tombstones[idx] {
                    self.tombstones[idx] = true;
                    deleted_count += 1;
                    deleted_ids.push(id.clone());

                    self.metadata_index.remove_document(idx as u32);
                    self.sparse_index.remove_document(idx as u32);
                    self.colbert_index.remove_document(idx as u32);
                    self.bm25.remove_document(idx as u32, &self.documents[idx].text);

                    // Dynamic online HNSW node deletion and edge healing
                    if self.hnsw.is_some() {
                        self.ensure_vectors_in_ram()?;
                    }
                    if let Some(ref mut hnsw) = self.hnsw {
                        hnsw.delete_node(idx as u32, &self.vectors, self.metric, is_cosine);
                    }
                }
                self.id_to_index.remove(id);
            }
        }

        if log_to_wal && !deleted_ids.is_empty() {
            if let Some(ref mut wal) = self.wal {
                wal.append(&WalRecord {
                    op: "delete".to_string(),
                    docs: Vec::new(),
                    ids: deleted_ids,
                })?;
            }
        }

        Ok(deleted_count)
    }

    pub fn count(&self) -> usize {
        if self.tombstones.is_empty() {
            self.documents.len()
        } else {
            self.tombstones.iter().filter(|&&t| !t).count()
        }
    }

    pub fn build_hnsw(&mut self, config: Option<HNSWConfig>) -> Result<(), SutraError> {
        self.ensure_vectors_in_ram()?;
        let cfg = config.unwrap_or_default();
        let mut index = HNSWIndex::new(cfg);
        let is_cosine = self.metric == Metric::Cosine;

        let active_ids: Vec<u32> = (0..self.documents.len())
            .filter(|&i| i >= self.tombstones.len() || !self.tombstones[i])
            .map(|i| i as u32)
            .collect();
        index.build_batch(&active_ids, &self.vectors, self.metric, is_cosine)?;

        self.hnsw = Some(index);
        self.dirty_index = false;
        Ok(())
    }

    pub fn build_ivf(&mut self, config: Option<IVFConfig>) -> Result<(), SutraError> {
        self.ensure_vectors_in_ram()?;
        let active_count = self.count();
        if active_count == 0 {
            return Err(SutraError::IndexNotTrained("Cannot build index on empty collection".into()));
        }

        let cfg = config.unwrap_or_else(|| IVFConfig {
            n_lists: auto_n_lists(active_count),
            nprobe: 16,
            kmeans_iters: 25,
            seed: 42,
        });

        let mut index = IVFIndex::with_metric(cfg, self.metric);
        let mut active_flat = VectorArena::new(self.dimension);
        active_flat.data.reserve(active_count * self.dimension);
        for (i, v) in self.vectors.iter().enumerate() {
            if i >= self.tombstones.len() || !self.tombstones[i] {
                active_flat.push_slice(v);
            }
        }

        index.train_store(&active_flat, self.dimension)?;

        for (i, v) in self.vectors.iter().enumerate() {
            if i < self.tombstones.len() && self.tombstones[i] {
                continue;
            }
            index.add(i as u32, v)?;
        }

        self.ivf = Some(index);
        self.dirty_index = false;
        Ok(())
    }

    /// Computes per-dimension mean over active vectors without requiring full RAM arena residency.
    fn compute_bq_centroid(&self) -> Option<Vec<f32>> {
        let active_count = self.count();
        if active_count == 0 || self.dimension == 0 {
            return None;
        }
        let mut mean = vec![0.0f64; self.dimension];
        for i in 0..self.documents.len() {
            if i < self.tombstones.len() && self.tombstones[i] {
                continue;
            }
            if let Some(vec) = self.get_vector(i) {
                for (m, &val) in mean.iter_mut().zip(vec.iter()) {
                    *m += val as f64;
                }
            }
        }
        let inv = 1.0 / active_count as f64;
        Some(mean.into_iter().map(|s| (s * inv) as f32).collect())
    }

    pub fn enable_sq8(&mut self) {
        if !self.use_sq8 {
            if !self.vectors_evicted {
                self.sq8_vectors = SQ8Arena::from_flat_f32(&self.vectors.data, self.dimension);
            } else {
                let mut arena = SQ8Arena::new(self.dimension);
                arena.reserve(self.documents.len());
                for i in 0..self.documents.len() {
                    if let Some(v) = self.get_vector(i) {
                        arena.push_vec(&v);
                    }
                }
                self.sq8_vectors = arena;
            }
            self.use_sq8 = true;
            self.maybe_evict_fp32_arena();
        }
    }

    /// Enables 1-bit Binary Quantization (BQ) centered at the corpus per-dimension mean.
    /// Note: Sign-bit BQ approximates angular/cosine similarity and is automatically bypassed
    /// in favor of SQ8/exact scoring when the collection metric is `Metric::Euclidean`.
    pub fn enable_bq(&mut self) {
        if !self.use_bq {
            let centroid = self.compute_bq_centroid();
            if !self.vectors_evicted {
                self.bq_vectors = BQArena::from_flat_f32(
                    &self.vectors.data,
                    self.dimension,
                    centroid.as_deref(),
                );
            } else {
                let mut arena = BQArena::new(self.dimension);
                for i in 0..self.documents.len() {
                    if let Some(v) = self.get_vector(i) {
                        arena.push_vec(&v, centroid.as_deref());
                    }
                }
                self.bq_vectors = arena;
            }
            self.bq_centroid = centroid;
            self.use_bq = true;
            self.maybe_evict_fp32_arena();
        }
    }

    pub fn train_pq(&mut self, m: Option<usize>, max_train_samples: Option<usize>) -> Result<(), SutraError> {
        let active_count = self.count();
        if active_count == 0 {
            return Err(SutraError::IndexNotTrained("Cannot train PQ on empty collection".into()));
        }

        let m_val = m.unwrap_or_else(|| PQCodebook::suggest_m(self.dimension));
        let k_target = 256.min(active_count);
        let sample_cap = max_train_samples.unwrap_or_else(|| 256usize.saturating_mul(k_target));
        let sample_n = active_count.min(sample_cap).max(1);

        let mut train_flat = Vec::with_capacity(sample_n * self.dimension);
        for i in 0..self.documents.len() {
            if i >= self.tombstones.len() || !self.tombstones[i] {
                if let Some(v) = self.get_vector(i) {
                    train_flat.extend_from_slice(&v);
                    if train_flat.len() >= sample_n * self.dimension {
                        break;
                    }
                }
            }
        }
        let actual_train_n = train_flat.len() / self.dimension.max(1);

        let cfg = PQConfig {
            m: m_val,
            k: 256.min(actual_train_n),
            kmeans_iters: 20,
            seed: 42,
        };

        let codebook = PQCodebook::train_flat(&train_flat, actual_train_n, self.dimension, cfg)?;
        let codes = if !self.vectors_evicted {
            codebook.try_encode_flat(&self.vectors.data, self.documents.len())?
        } else {
            let mut buf = vec![0u8; self.documents.len() * codebook.m];
            for i in 0..self.documents.len() {
                if let Some(v) = self.get_vector(i) {
                    codebook.try_encode_into(&v, &mut buf[i * codebook.m..(i + 1) * codebook.m])?;
                }
            }
            buf
        };
        self.pq_vectors = PQArena {
            codes,
            m: codebook.m,
        };
        self.pq_codebook = Some(codebook);
        self.use_pq = true;
        self.maybe_evict_fp32_arena();
        Ok(())
    }

    pub fn enable_pq(&mut self) -> Result<(), SutraError> {
        if !self.use_pq {
            if let Some(ref cb) = self.pq_codebook {
                let codes = if !self.vectors_evicted {
                    cb.try_encode_flat(&self.vectors.data, self.documents.len())?
                } else {
                    let mut buf = vec![0u8; self.documents.len() * cb.m];
                    for i in 0..self.documents.len() {
                        if let Some(v) = self.get_vector(i) {
                            cb.try_encode_into(&v, &mut buf[i * cb.m..(i + 1) * cb.m])?;
                        }
                    }
                    buf
                };
                self.pq_vectors = PQArena { codes, m: cb.m };
                self.use_pq = true;
                self.maybe_evict_fp32_arena();
            } else {
                self.train_pq(None, None)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn query(
        &self,
        vector: Option<&[f32]>,
        text: Option<&str>,
        sparse_vector: Option<&crate::sparse::SparseVector>,
        colbert_tokens: Option<&[Vec<f32>]>,
        namespace: Option<&str>,
        filter: Option<&Value>,
        top_k: usize,
        hybrid: bool,
        alpha: Option<f32>,
        rrf_constant: usize,
        dense_weight: f32,
        bm25_weight: f32,
        sparse_weight: f32,
        colbert_weight: f32,
        use_ann: bool,
    ) -> Result<Vec<NativeSearchResult>, SutraError> {
        if let Some(q_vec) = vector {
            if q_vec.len() != self.dimension {
                return Err(SutraError::DimensionMismatch {
                    expected: self.dimension,
                    got: q_vec.len(),
                });
            }
        }
        if let Some(sv) = sparse_vector {
            if sv.indices.len() != sv.values.len() {
                return Err(SutraError::DimensionMismatch {
                    expected: sv.indices.len(),
                    got: sv.values.len(),
                });
            }
        }
        if let Some(ct) = colbert_tokens {
            for tok in ct {
                if tok.len() != self.dimension {
                    return Err(SutraError::DimensionMismatch {
                        expected: self.dimension,
                        got: tok.len(),
                    });
                }
            }
        }

        let n = self.documents.len();
        if n == 0 || top_k == 0 {
            return Ok(Vec::new());
        }

        // 1. Build compound mask: metadata filter & tombstones & namespace isolation
        // Sub-microsecond filter evaluation via SecondaryMetadataIndex
        let mut mask = vec![true; n];
        let mut index_exact = true;
        let eval_res = self.metadata_index.evaluate_filter_checked(filter);
        let mut exact_filter_bitmap: Option<roaring::RoaringBitmap> = None;
        if let Some((bitmap, is_exact)) = eval_res {
            index_exact = is_exact;
            for (i, m) in mask.iter_mut().enumerate() {
                *m = bitmap.contains(i as u32);
            }
            if is_exact && !filter.is_some_and(crate::filter::has_complex_operators) && namespace.is_none() {
                exact_filter_bitmap = Some(bitmap);
            }
        }

        // If the filter has complex operators ($geo_within, $geo_box, $regex, etc.) or index returned a superset,
        // verify remaining matching candidates with the full AST FilterEngine.
        if !index_exact || filter.is_some_and(crate::filter::has_complex_operators) {
            for (i, m) in mask.iter_mut().enumerate() {
                if *m {
                    *m = FilterEngine::evaluate(Some(&self.documents[i].metadata), filter);
                }
            }
        }

        for (i, &t) in self.tombstones.iter().enumerate() {
            if t && i < mask.len() {
                mask[i] = false;
            }
        }

        if let Some(ns) = namespace {
            for (i, doc) in self.documents.iter().enumerate() {
                if i < mask.len() && mask[i] && doc.namespace.as_deref() != Some(ns) {
                    mask[i] = false;
                }
            }
        }

        let valid_count = mask.iter().filter(|&&m| m).count();
        if valid_count == 0 {
            return Ok(Vec::new());
        }

        let has_vector = vector.is_some();
        let has_text = text.map(|t| !t.trim().is_empty()).unwrap_or(false);
        let has_sparse = sparse_vector.map(|sv| !sv.is_empty()).unwrap_or(false);
        let has_colbert = colbert_tokens.map(|ct| !ct.is_empty()).unwrap_or(false);

        // 2. Cost-Based Query Planning:
        // Only bypass HNSW for tiny candidate pools (<= 32) where exact SIMD scan finishes in microseconds;
        // larger filtered queries (including 1% selectivity on large corpora) traverse HNSW with in-flight bitmap filtering.
        let should_bypass_hnsw = valid_count <= 32;

        let is_cosine = self.metric == Metric::Cosine;
        let mut dense_scores = if has_vector {
            vec![f32::NEG_INFINITY; n]
        } else {
            vec![0.0f32; n]
        };

        if let Some(q_vec) = vector {
            let normalized_q = if is_cosine { normalize_copy(q_vec) } else { q_vec.to_vec() };

            // ANN path: HNSW or IVF if pure vector query without BM25 and selectivity is high enough
            if use_ann
                && !has_text
                && !has_sparse
                && !has_colbert
                && !self.dirty_index
                && !should_bypass_hnsw
                && !self.vectors_evicted
            {
                if let Some(ref hnsw) = self.hnsw {
                    let ef = hnsw.config.ef_search.max(top_k);
                    let ann_res = if let Some(ref bm) = exact_filter_bitmap {
                        hnsw.search_with_bitmap(&normalized_q, &self.vectors, top_k, ef, self.metric, is_cosine, Some(bm))?
                    } else if filter.is_some() || namespace.is_some() {
                        let bm: roaring::RoaringBitmap = mask
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &ok)| if ok { Some(i as u32) } else { None })
                            .collect();
                        hnsw.search_with_bitmap(&normalized_q, &self.vectors, top_k, ef, self.metric, is_cosine, Some(&bm))?
                    } else {
                        hnsw.search_with_bitmap(&normalized_q, &self.vectors, top_k, ef, self.metric, is_cosine, None)?
                    };
                    return Ok(ann_res.into_iter().map(|(id, score)| {
                        let doc = &self.documents[id as usize];
                        NativeSearchResult {
                            id: doc.id.clone(),
                            score,
                            text: doc.text.clone(),
                            metadata: doc.metadata.clone(),
                            dense_score: Some(score),
                            bm25_score: None,
                            sparse_score: None,
                            colbert_score: None,
                            namespace: doc.namespace.clone(),
                        }
                    }).collect());
                } else if let Some(ref ivf) = self.ivf {
                    let ann_res = ivf.search(&normalized_q, &self.vectors, top_k, self.metric, is_cosine, Some(&mask))?;
                    return Ok(ann_res.into_iter().map(|(id, score)| {
                        let doc = &self.documents[id as usize];
                        NativeSearchResult {
                            id: doc.id.clone(),
                            score,
                            text: doc.text.clone(),
                            metadata: doc.metadata.clone(),
                            dense_score: Some(score),
                            bm25_score: None,
                            sparse_score: None,
                            colbert_score: None,
                            namespace: doc.namespace.clone(),
                        }
                    }).collect());
                }
            }

            // Low-selectivity bypass or Exact scoring / Quantized scoring
            if self.use_pq {
                if let Some(ref cb) = self.pq_codebook {
                    let lut = cb.try_compute_lut(&normalized_q, self.metric)?;
                    let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(valid_count);

                    for i in 0..n {
                        if mask[i] {
                            let score = cb.try_score_adc_slice(
                                self.pq_vectors.code_slice(i),
                                &lut,
                                self.metric,
                            )?;
                            candidates.push((i, score));
                        }
                    }

                    let rerank_count = (top_k * 20).max(200).min(candidates.len());
                    if candidates.len() > rerank_count && rerank_count > 0 {
                        candidates.select_nth_unstable_by(rerank_count - 1, |a, b| {
                            b.1.partial_cmp(&a.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| a.0.cmp(&b.0))
                        });
                    }
                    let q_sum = if self.use_sq8 && self.vectors_evicted {
                        normalized_q.iter().sum()
                    } else {
                        0.0
                    };
                    for &(idx, adc_score) in candidates.iter().take(rerank_count) {
                        if let Some(vec_ref) = self.get_vector(idx) {
                            dense_scores[idx] = compute_vector_score(
                                &normalized_q,
                                vec_ref.as_ref(),
                                self.metric,
                                is_cosine,
                            )?;
                        } else if self.use_sq8 && idx < self.sq8_vectors.len() {
                            dense_scores[idx] = self.sq8_vectors.try_score_idx(
                                idx,
                                &normalized_q,
                                q_sum,
                                self.metric,
                            )?;
                        } else {
                            dense_scores[idx] = adc_score;
                        }
                    }
                }
            } else if self.use_bq && self.metric != Metric::Euclidean {
                // Stage 1: Fast Hamming distance scan over centered 1-bit quantized bitmasks (angular/cosine & dot only)
                let q_bq = match self.bq_centroid {
                    Some(ref centroid) => BQVector::quantize_centered(&normalized_q, centroid),
                    None => BQVector::quantize(&normalized_q),
                };
                let mut candidates: Vec<(usize, u32)> = if n >= 256 {
                    use rayon::prelude::*;
                    (0..n)
                        .into_par_iter()
                        .filter_map(|i| {
                            if mask[i] {
                                Some(
                                    self.bq_vectors
                                        .try_hamming_idx(i, &q_bq.bits)
                                        .map(|d| (i, d)),
                                )
                            } else {
                                None
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    let mut out = Vec::with_capacity(valid_count);
                    for i in 0..n {
                        if mask[i] {
                            let d = self.bq_vectors.try_hamming_idx(i, &q_bq.bits)?;
                            out.push((i, d));
                        }
                    }
                    out
                };

                // Oversample Hamming candidates via O(N) partial selection before Stage-2 reranking.
                // When disk-backed FP32 vectors are evicted or dim >= 384, cap rerank at max(20*k, 200)
                // to bound mmap page faults; for compact in-RAM embeddings (dim < 384), oversample
                // deeper to sustain >= 90% Recall@10 against high-dimensional sign-bit variance.
                let base_rerank = if self.vectors_evicted || self.dimension >= 384 {
                    (top_k * 20).max(200)
                } else {
                    (top_k * 150).max(1500)
                };
                let rerank_count = base_rerank.min(candidates.len());
                if candidates.len() > rerank_count && rerank_count > 0 {
                    candidates.select_nth_unstable_by(rerank_count - 1, |&(ia, da), &(ib, db)| {
                        da.cmp(&db).then_with(|| ia.cmp(&ib))
                    });
                }

                let q_sum = if self.use_sq8 {
                    normalized_q.iter().sum()
                } else {
                    0.0
                };
                for &(idx, _) in candidates.iter().take(rerank_count) {
                    if self.use_sq8 && (self.vectors_evicted || idx < self.sq8_vectors.len()) {
                        dense_scores[idx] = self.sq8_vectors.try_score_idx(
                            idx,
                            &normalized_q,
                            q_sum,
                            self.metric,
                        )?;
                    } else if let Some(vec_ref) = self.get_vector(idx) {
                        dense_scores[idx] = compute_vector_score(
                            &normalized_q,
                            vec_ref.as_ref(),
                            self.metric,
                            is_cosine,
                        )?;
                    }
                }
            } else if self.use_sq8 {
                let q_sum: f32 = normalized_q.iter().sum();
                let sq_count = self.sq8_vectors.len();
                if sq_count >= 64 {
                    use rayon::prelude::*;
                    dense_scores = (0..sq_count)
                        .into_par_iter()
                        .map(|i| {
                            if mask[i] {
                                self.sq8_vectors
                                    .try_score_idx(i, &normalized_q, q_sum, self.metric)
                            } else {
                                Ok(f32::NEG_INFINITY)
                            }
                        })
                        .collect::<Result<Vec<f32>, SutraError>>()?;
                } else {
                    for i in 0..sq_count {
                        if mask[i] {
                            dense_scores[i] = self.sq8_vectors.try_score_idx(
                                i,
                                &normalized_q,
                                q_sum,
                                self.metric,
                            )?;
                        }
                    }
                }
            } else if !self.vectors_evicted && self.vectors.len() >= 64 {
                use rayon::prelude::*;
                dense_scores = self
                    .vectors
                    .par_iter()
                    .enumerate()
                    .map(|(i, v)| {
                        if mask[i] {
                            compute_vector_score(&normalized_q, v, self.metric, is_cosine)
                        } else {
                            Ok(f32::NEG_INFINITY)
                        }
                    })
                    .collect::<Result<Vec<f32>, SutraError>>()?;
            } else {
                for i in 0..n {
                    if mask[i] {
                        if let Some(v) = self.get_vector(i) {
                            dense_scores[i] = compute_vector_score(
                                &normalized_q,
                                v.as_ref(),
                                self.metric,
                                is_cosine,
                            )?;
                        }
                    }
                }
            }
        }

        let num_modalities = (has_vector as usize) + (has_text as usize) + (has_sparse as usize) + (has_colbert as usize);

        // Fast-path return for pure dense vector query without allocating BM25/Sparse/ColBERT arrays
        if has_vector && (!hybrid || num_modalities < 2) && !has_colbert && !has_sparse {
            let mut valid: Vec<(usize, f32)> = Vec::with_capacity(valid_count.min(n));
            for i in 0..n {
                if mask[i] && dense_scores[i].is_finite() {
                    valid.push((i, dense_scores[i]));
                }
            }
            if valid.len() > top_k {
                valid.select_nth_unstable_by(top_k - 1, |a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                valid.truncate(top_k);
            }
            valid.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });

            return Ok(valid.into_iter().map(|(idx, score)| {
                let doc = &self.documents[idx];
                NativeSearchResult {
                    id: doc.id.clone(),
                    score,
                    text: doc.text.clone(),
                    metadata: doc.metadata.clone(),
                    dense_score: Some(score),
                    bm25_score: None,
                    sparse_score: None,
                    colbert_score: None,
                    namespace: doc.namespace.clone(),
                }
            }).collect());
        }

        // Fast-path return for pure BM25 lexical query without allocating O(N) score buffers
        if has_text && (!hybrid || num_modalities < 2) && !has_vector && !has_colbert && !has_sparse {
            let valid = self.bm25.search_top_k(text.unwrap(), top_k, Some(&mask));
            return Ok(valid.into_iter().map(|(idx, score)| {
                let doc = &self.documents[idx];
                NativeSearchResult {
                    id: doc.id.clone(),
                    score,
                    text: doc.text.clone(),
                    metadata: doc.metadata.clone(),
                    dense_score: None,
                    bm25_score: Some(score),
                    sparse_score: None,
                    colbert_score: None,
                    namespace: doc.namespace.clone(),
                }
            }).collect());
        }

        // 3. Score BM25
        let bm25_scores = if has_text {
            self.bm25.score_query(text.unwrap(), Some(&mask))
        } else {
            vec![0.0f32; n]
        };

        // 4. Score Neural Sparse Vectors
        let mut sparse_scores = Vec::new();
        if let Some(sv) = sparse_vector {
            sparse_scores.resize(n, 0.0f32);
            let sparse_hits = self.sparse_index.search(sv, n, Some(&mask));
            for (id, sc) in sparse_hits {
                let idx = id as usize;
                if idx < n {
                    sparse_scores[idx] = sc;
                }
            }
        }

        // 5. Score ColBERT Multi-Vectors
        let mut colbert_scores = Vec::new();
        if let Some(ct) = colbert_tokens {
            colbert_scores.resize(n, 0.0f32);
            let colbert_hits = self.colbert_index.try_query_maxsim(ct, Some(&mask), n)?;
            for (id, sc) in colbert_hits {
                let idx = id as usize;
                if idx < n {
                    colbert_scores[idx] = sc;
                }
            }
        }

        // 6. Fusion or routing
        if hybrid && num_modalities >= 2 {
            let empty_scores;
            let sp_slice = if !sparse_scores.is_empty() {
                &sparse_scores
            } else {
                empty_scores = vec![0.0f32; n];
                &empty_scores
            };
            let cb_empty;
            let cb_slice = if !colbert_scores.is_empty() {
                &colbert_scores
            } else {
                cb_empty = vec![0.0f32; n];
                &cb_empty
            };
            let fused = if has_colbert {
                crate::fusion::quad_modal_reciprocal_rank_fusion(
                    &dense_scores,
                    &bm25_scores,
                    sp_slice,
                    cb_slice,
                    top_k,
                    rrf_constant,
                    dense_weight,
                    bm25_weight,
                    sparse_weight,
                    colbert_weight,
                    Some(&mask),
                )
            } else if has_sparse {
                crate::fusion::tri_modal_reciprocal_rank_fusion(
                    &dense_scores,
                    &bm25_scores,
                    sp_slice,
                    top_k,
                    rrf_constant,
                    dense_weight,
                    bm25_weight,
                    sparse_weight,
                    Some(&mask),
                )
            } else if let Some(a) = alpha {
                linear_score_fusion(&dense_scores, &bm25_scores, a, top_k, Some(&mask))
            } else {
                reciprocal_rank_fusion(
                    &dense_scores,
                    &bm25_scores,
                    top_k,
                    rrf_constant,
                    dense_weight,
                    bm25_weight,
                    Some(&mask),
                )
            };

            return Ok(fused.into_iter().map(|c| {
                let doc = &self.documents[c.doc_idx];
                NativeSearchResult {
                    id: doc.id.clone(),
                    score: c.score,
                    text: doc.text.clone(),
                    metadata: doc.metadata.clone(),
                    dense_score: Some(c.dense_score),
                    bm25_score: Some(c.bm25_score),
                    sparse_score: Some(c.sparse_score),
                    colbert_score: Some(c.colbert_score),
                    namespace: doc.namespace.clone(),
                }
            }).collect());
        }

        if has_colbert && !has_vector && !has_text && !has_sparse {
            let mut valid: Vec<(usize, f32)> = (0..n).filter(|&i| mask[i] && colbert_scores[i] > 0.0).map(|i| (i, colbert_scores[i])).collect();
            valid.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            valid.truncate(top_k);

            return Ok(valid.into_iter().map(|(idx, score)| {
                let doc = &self.documents[idx];
                NativeSearchResult {
                    id: doc.id.clone(),
                    score,
                    text: doc.text.clone(),
                    metadata: doc.metadata.clone(),
                    dense_score: None,
                    bm25_score: None,
                    sparse_score: None,
                    colbert_score: Some(score),
                    namespace: doc.namespace.clone(),
                }
            }).collect());
        }

        if has_sparse && !has_vector && !has_text {
            let mut valid: Vec<(usize, f32)> = (0..n).filter(|&i| mask[i] && sparse_scores[i] > 0.0).map(|i| (i, sparse_scores[i])).collect();
            valid.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            valid.truncate(top_k);

            return Ok(valid.into_iter().map(|(idx, score)| {
                let doc = &self.documents[idx];
                NativeSearchResult {
                    id: doc.id.clone(),
                    score,
                    text: doc.text.clone(),
                    metadata: doc.metadata.clone(),
                    dense_score: None,
                    bm25_score: None,
                    sparse_score: Some(score),
                    colbert_score: None,
                    namespace: doc.namespace.clone(),
                }
            }).collect());
        }

        if has_vector {
            let mut valid: Vec<(usize, f32)> = (0..n)
                .filter(|&i| mask[i] && dense_scores[i].is_finite())
                .map(|i| (i, dense_scores[i]))
                .collect();
            valid.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            valid.truncate(top_k);

            return Ok(valid.into_iter().map(|(idx, score)| {
                let doc = &self.documents[idx];
                NativeSearchResult {
                    id: doc.id.clone(),
                    score,
                    text: doc.text.clone(),
                    metadata: doc.metadata.clone(),
                    dense_score: Some(score),
                    bm25_score: None,
                    sparse_score: None,
                    colbert_score: None,
                    namespace: doc.namespace.clone(),
                }
            }).collect());
        }

        if has_text {
            let mut valid: Vec<(usize, f32)> = (0..n).filter(|&i| mask[i] && bm25_scores[i] > 0.0).map(|i| (i, bm25_scores[i])).collect();
            valid.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            valid.truncate(top_k);

            return Ok(valid.into_iter().map(|(idx, score)| {
                let doc = &self.documents[idx];
                NativeSearchResult {
                    id: doc.id.clone(),
                    score,
                    text: doc.text.clone(),
                    metadata: doc.metadata.clone(),
                    dense_score: None,
                    bm25_score: Some(score),
                    sparse_score: None,
                    colbert_score: None,
                    namespace: doc.namespace.clone(),
                }
            }).collect());
        }

        Err(SutraError::InvalidQuery("Query must specify a vector, text, sparse vector, or combination".into()))
    }

    pub fn query_colbert(
        &self,
        query_tokens: &[Vec<f32>],
        namespace: Option<&str>,
        filter: Option<&Value>,
        top_k: usize,
    ) -> Result<Vec<NativeSearchResult>, SutraError> {
        for tok in query_tokens {
            if tok.len() != self.dimension {
                return Err(SutraError::DimensionMismatch {
                    expected: self.dimension,
                    got: tok.len(),
                });
            }
        }

        let n = self.documents.len();
        if n == 0 || top_k == 0 {
            return Ok(Vec::new());
        }

        let mut mask = vec![true; n];
        let mut index_exact = true;
        if let Some((bitmap, is_exact)) = self.metadata_index.evaluate_filter_checked(filter) {
            index_exact = is_exact;
            for (i, m) in mask.iter_mut().enumerate() {
                *m = bitmap.contains(i as u32);
            }
        }
        if !index_exact || filter.is_some_and(crate::filter::has_complex_operators) {
            for (i, m) in mask.iter_mut().enumerate() {
                if *m {
                    *m = FilterEngine::evaluate(Some(&self.documents[i].metadata), filter);
                }
            }
        }
        for (i, &t) in self.tombstones.iter().enumerate() {
            if t && i < mask.len() {
                mask[i] = false;
            }
        }
        if let Some(ns) = namespace {
            for (i, doc) in self.documents.iter().enumerate() {
                if i < mask.len() && mask[i] && doc.namespace.as_deref() != Some(ns) {
                    mask[i] = false;
                }
            }
        }

        let hits = self.colbert_index.try_query_maxsim(query_tokens, Some(&mask), top_k)?;
        Ok(hits.into_iter().map(|(idx, score)| {
            let doc = &self.documents[idx as usize];
            NativeSearchResult {
                id: doc.id.clone(),
                score,
                text: doc.text.clone(),
                metadata: doc.metadata.clone(),
                dense_score: None,
                bm25_score: None,
                sparse_score: None,
                colbert_score: Some(score),
                namespace: doc.namespace.clone(),
            }
        }).collect())
    }

    pub fn save<P: AsRef<Path>>(&mut self, path: Option<P>) -> Result<u64, SutraError> {
        validate_collection_name(&self.name)?;
        let p = match path {
            Some(ref path_ref) => path_ref.as_ref().to_path_buf(),
            None => match self.storage_path {
                Some(ref sp) => sp.clone(),
                None => return Err(SutraError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "No storage path"))),
            },
        };

        // Compact tombstones before saving
        self.compact()?;

        let doc_count = u32::try_from(self.documents.len())
            .map_err(|_| SutraError::InvalidHeader("Document count exceeds u32::MAX".into()))?;
        let dim = u32::try_from(self.dimension)
            .map_err(|_| SutraError::InvalidHeader("Dimension exceeds u32::MAX".into()))?;

        let doc_payloads: Vec<Value> = self.documents.iter().map(|d| {
            serde_json::json!({
                "id": d.id,
                "namespace": d.namespace,
                "text": d.text,
                "metadata": d.metadata,
                "sparse_vector": d.sparse_vector.as_ref().map(|sv| {
                    let mut m = serde_json::Map::new();
                    for (&idx, &val) in sv.indices.iter().zip(sv.values.iter()) {
                        m.insert(idx.to_string(), serde_json::Value::from(val));
                    }
                    serde_json::Value::Object(m)
                }),
                "colbert_tokens": d.colbert_tokens
            })
        }).collect();

        let meta_bytes = serde_json::to_vec(&doc_payloads)
            .map_err(|e| SutraError::Serialization(e.to_string()))?;

        let written = if self.use_sq8 {
            if self.sq8_vectors.len() != self.documents.len() {
                return Err(SutraError::DimensionMismatch {
                    expected: self.documents.len(),
                    got: self.sq8_vectors.len(),
                });
            }
            BinaryStorage::write_sq8_v4(
                &p,
                &self.sq8_vectors.params,
                &self.sq8_vectors.data,
                doc_count,
                dim,
                self.metric.as_str(),
                &meta_bytes,
            )?
        } else {
            self.ensure_vectors_in_ram()?;
            let written = BinaryStorage::write_f32(
                &p,
                &self.vectors.data,
                doc_count,
                dim,
                self.metric.as_str(),
                &meta_bytes,
            )?;
            self.mmap_storage = MmapVectorStorage::open_lazy(&p).ok();
            written
        };

        if let Some(ref mut wal) = self.wal {
            wal.clear()?;
        }

        let parent_crc32 = if let Some(ref mmap) = self.mmap_storage {
            mmap.header().crc32
        } else {
            MmapVectorStorage::open_lazy(&p)?.header().crc32
        };

        let persisted = PersistedIndexes {
            hnsw: self.hnsw.as_ref(),
            ivf: self.ivf.as_ref(),
            bq: if self.use_bq {
                Some((&self.bq_vectors, self.bq_centroid.as_deref()))
            } else {
                None
            },
            pq: if self.use_pq {
                self.pq_codebook
                    .as_ref()
                    .map(|cb| (&self.pq_vectors, cb))
            } else {
                None
            },
            colbert_centroids: self
                .colbert_index
                .centroids
                .as_ref()
                .map(|cc| (self.colbert_index.token_dim, cc)),
            dirty_index: self.dirty_index,
        };
        let _idx_written = IndexStorage::write_for_parent(
            &p,
            doc_count,
            dim,
            parent_crc32,
            &persisted,
        )?;

        Ok(written)
    }

    pub fn load<P: AsRef<Path>>(&mut self, path: Option<P>) -> Result<usize, SutraError> {
        validate_collection_name(&self.name)?;
        let p = match path {
            Some(ref path_ref) => path_ref.as_ref().to_path_buf(),
            None => match self.storage_path {
                Some(ref sp) => sp.clone(),
                None => return Err(SutraError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "No storage path"))),
            },
        };

        let storage = MmapVectorStorage::open_lazy(&p)?;
        let header = storage.header().clone();
        self.dimension = header.dimension as usize;
        self.metric = Metric::from_str(&header.metric)?;

        let mut restored_idx = IndexStorage::read_for_parent(
            &p,
            header.doc_count,
            header.dimension,
            header.crc32,
        )?;

        let doc_count = header.doc_count as usize;
        let doc_vals: Vec<Value> = serde_json::from_slice(storage.metadata_bytes())
            .map_err(|e| SutraError::Serialization(e.to_string()))?;
        if doc_vals.len() != doc_count {
            return Err(SutraError::InvalidHeader(format!(
                "Metadata document count mismatch: header={}, metadata={}",
                doc_count,
                doc_vals.len()
            )));
        }

        self.documents.clear();
        self.vectors = VectorArena::new(self.dimension);
        self.vectors.data.reserve(doc_count * self.dimension);
        self.mmap_storage = None;
        self.vectors_evicted = false;
        self.id_to_index.clear();
        self.tombstones.clear();
        self.bm25 = BM25Index::with_analyzer(self.bm25.k1, self.bm25.b, self.bm25.analyzer.clone());
        self.metadata_index.clear();
        self.sparse_index = crate::sparse::SparseIndex::new();
        self.colbert_index = crate::colbert::ColbertIndex::new(self.dimension);
        self.sq8_vectors = SQ8Arena::new(self.dimension);
        self.bq_vectors = BQArena::new(self.dimension);
        self.bq_centroid = None;
        self.pq_vectors = PQArena::new(0);
        self.pq_codebook = None;
        self.use_sq8 = false;
        self.use_bq = false;
        self.use_pq = false;
        self.hnsw = None;
        self.ivf = None;
        self.dirty_index = false;

        let mut has_persisted_colbert_centroids = false;
        if let Some(ref mut sidecar) = restored_idx {
            if let Some((token_dim, cc)) = sidecar.colbert_centroids.take() {
                if token_dim == self.dimension {
                    self.colbert_index.centroids = Some(cc);
                    self.colbert_index.is_trained = true;
                    has_persisted_colbert_centroids = true;
                }
            }
        }

        let is_v4_sq8 = header.magic == *HEADER_MAGIC_V4 && header.quant_mode == 1;
        if is_v4_sq8 {
            self.sq8_vectors.reserve(doc_count);
        }

        for (i, d_val) in doc_vals.iter().enumerate() {
            if is_v4_sq8 {
                let (min_val, scale) = storage.get_sq8_params(i)?;
                let data = storage.try_get_vector_sq8(i)?;
                self.sq8_vectors.push_raw(data, min_val, scale);
                let mut dequant = Vec::with_capacity(self.dimension);
                for &b in data {
                    dequant.push(min_val + (b as f32) * scale);
                }
                self.vectors.push_slice(&dequant);
            } else if header.quant_mode == 1 {
                let sq8_slice = storage.try_get_vector_sq8(i)?;
                let vec: Vec<f32> = sq8_slice.iter().map(|&b| (b as f32) / 255.0).collect();
                self.vectors.push_slice(&vec);
            } else {
                self.vectors.push_slice(storage.try_get_vector_f32(i)?);
            }

            let id = d_val.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let namespace = d_val.get("namespace").and_then(|v| v.as_str()).map(|s| s.to_string());
            let text = d_val.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let metadata = d_val.get("metadata").cloned().unwrap_or(Value::Object(serde_json::Map::new()));

            let sparse_vec = if let Some(v) = d_val.get("sparse_vector") {
                if let Some(map) = v.as_object() {
                    let mut pairs: Vec<(u32, f32)> = map.iter().filter_map(|(k, val)| {
                        let idx = k.parse::<u32>().ok()?;
                        let f = val.as_f64()? as f32;
                        Some((idx, f))
                    }).collect();
                    pairs.sort_by_key(|p| p.0);
                    let idxs = pairs.iter().map(|p| p.0).collect();
                    let vals = pairs.iter().map(|p| p.1).collect();
                    Some(crate::sparse::SparseVector::try_new(idxs, vals)?)
                } else {
                    None
                }
            } else {
                None
            };

            let colbert_tokens: Option<Vec<Vec<f32>>> = d_val.get("colbert_tokens").and_then(|v| {
                v.as_array().map(|arr| {
                    arr.iter().filter_map(|tok_arr| {
                        tok_arr.as_array().map(|vec_arr| {
                            vec_arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect()
                        })
                    }).collect()
                })
            });

            if let Some(ref sv) = sparse_vec {
                self.sparse_index.add_document(i as u32, sv);
            }
            if let Some(ref tokens) = colbert_tokens {
                if has_persisted_colbert_centroids {
                    self.colbert_index.restore_document_tokens(i as u32, tokens.clone())?;
                } else {
                    self.colbert_index.try_add_document(i as u32, tokens.clone())?;
                }
            }

            self.id_to_index.insert(id.clone(), i);
            self.metadata_index.index_document(i as u32, &metadata);
            self.documents.push(NativeDocument { id, namespace, text, metadata, sparse_vector: sparse_vec, colbert_tokens });
            self.tombstones.push(false);
        }

        if is_v4_sq8 {
            self.use_sq8 = true;
        } else if header.quant_mode == 1 {
            self.enable_sq8();
        }

        if let Some(sidecar) = restored_idx {
            self.hnsw = sidecar.hnsw;
            self.ivf = sidecar.ivf;
            if let Some((bq_arena, centroid)) = sidecar.bq {
                self.bq_vectors = bq_arena;
                self.bq_centroid = centroid;
                self.use_bq = true;
            }
            if let Some((pq_arena, pq_codebook)) = sidecar.pq {
                self.pq_vectors = pq_arena;
                self.pq_codebook = Some(pq_codebook);
                self.use_pq = true;
            }
            self.dirty_index = sidecar.dirty_index;
        }

        let corpus: Vec<String> = self.documents.iter().map(|d| d.text.clone()).collect();
        self.bm25.add_documents(&corpus);

        let wal_applied = self.replay_wal_if_present()?;
        if header.quant_mode == 0 && wal_applied == 0 {
            self.mmap_storage = Some(storage);
        }
        Ok(self.count())
    }

    pub fn compact(&mut self) -> Result<(), SutraError> {
        let has_tombstones = self.tombstones.iter().any(|&t| t);
        if !has_tombstones {
            if let Some(ref mut hnsw) = self.hnsw {
                hnsw.compact_tombstones();
            }
            self.bm25.compact_postings();
            return Ok(());
        }

        self.ensure_vectors_in_ram()?;
        self.mmap_storage = None;

        let prev_hnsw_cfg = self.hnsw.as_ref().map(|h| h.config.clone());
        let prev_ivf_cfg = self.ivf.as_ref().map(|ivf| ivf.config.clone());

        let mut new_docs = Vec::new();
        let mut new_vecs = VectorArena::new(self.dimension);
        let mut new_id_map = HashMap::new();
        let mut new_bm25 = BM25Index::with_analyzer(self.bm25.k1, self.bm25.b, self.bm25.analyzer.clone());
        let mut new_meta_idx = crate::filter::SecondaryMetadataIndex::new();
        let mut new_sparse_idx = crate::sparse::SparseIndex::new();
        let mut new_colbert_idx = crate::colbert::ColbertIndex::new(self.colbert_index.token_dim);
        if let Some(ref old_centroids) = self.colbert_index.centroids {
            new_colbert_idx.centroids = Some(crate::colbert::ColbertCentroids {
                centroids: old_centroids.centroids.clone(),
                postings: HashMap::new(),
            });
            new_colbert_idx.is_trained = true;
        }

        for (i, doc) in self.documents.drain(..).enumerate() {
            if i < self.tombstones.len() && self.tombstones[i] {
                continue;
            }
            let idx = new_docs.len();
            new_id_map.insert(doc.id.clone(), idx);
            new_meta_idx.index_document(idx as u32, &doc.metadata);
            if let Some(ref sv) = doc.sparse_vector {
                new_sparse_idx.add_document(idx as u32, sv);
            }
            if let Some(ref tokens) = doc.colbert_tokens {
                new_colbert_idx.try_add_document(idx as u32, tokens.clone())?;
            }
            new_vecs.push_slice(&self.vectors[i]);
            new_docs.push(doc);
        }

        let corpus: Vec<String> = new_docs.iter().map(|d| d.text.clone()).collect();
        new_bm25.add_documents(&corpus);

        self.documents = new_docs;
        self.vectors = new_vecs;
        self.id_to_index = new_id_map;
        self.tombstones = vec![false; self.documents.len()];
        self.bm25 = new_bm25;
        self.metadata_index = new_meta_idx;
        self.sparse_index = new_sparse_idx;
        self.colbert_index = new_colbert_idx;
        if self.use_sq8 {
            self.sq8_vectors = SQ8Arena::from_flat_f32(&self.vectors.data, self.dimension);
        }
        if self.use_bq {
            self.bq_centroid = self.compute_bq_centroid();
            self.bq_vectors = BQArena::from_flat_f32(
                &self.vectors.data,
                self.dimension,
                self.bq_centroid.as_deref(),
            );
        }
        if self.use_pq {
            if let Some(ref cb) = self.pq_codebook {
                let codes = cb.try_encode_flat(&self.vectors.data, self.documents.len())?;
                self.pq_vectors = PQArena { codes, m: cb.m };
            }
        }
        self.hnsw = None;
        self.ivf = None;
        self.dirty_index = false;
        if let Some(hnsw_cfg) = prev_hnsw_cfg {
            self.build_hnsw(Some(hnsw_cfg))?;
        }
        if let Some(ivf_cfg) = prev_ivf_cfg {
            if !self.vectors.is_empty() {
                self.build_ivf(Some(ivf_cfg))?;
            }
        }
        Ok(())
    }

    pub fn stats(&self) -> Value {
        let active_count = self.count();
        let vector_bytes = self.vectors.data.len() * 4;
        let sq8_bytes = if self.use_sq8 { self.sq8_vectors.data.len() } else { 0 };
        let bq_bytes = if self.use_bq { self.bq_vectors.bits.len() * 8 } else { 0 };
        let pq_bytes = if self.use_pq { self.pq_vectors.codes.len() } else { 0 };
        let total_vec_bytes = vector_bytes + sq8_bytes + bq_bytes + pq_bytes;

        let mut stats = serde_json::json!({
            "name": self.name,
            "count": active_count,
            "total_records": self.documents.len(),
            "tombstones": self.documents.len() - active_count,
            "dimension": self.dimension,
            "metric": self.metric.as_str(),
            "enable_wal": self.wal.is_some(),
            "vector_memory_bytes": total_vec_bytes,
            "vector_memory_mb": ((total_vec_bytes) as f64) / (1024.0 * 1024.0),
            "quantized_sq8": self.use_sq8,
            "quantized_bq": self.use_bq,
            "quantized_pq": self.use_pq,
            "pq_m": self.pq_codebook.as_ref().map(|cb| cb.m),
            "has_hnsw": self.hnsw.is_some(),
            "has_ivf": self.ivf.is_some(),
            "simd_level": crate::distance::simd::detect_simd_capabilities().to_string(),
            "analyzer": {
                "language": self.bm25.analyzer.config.language,
                "stemmer": self.bm25.analyzer.config.stemmer,
                "stopwords": self.bm25.analyzer.config.stopwords,
                "split_identifiers": self.bm25.analyzer.config.split_identifiers,
                "cjk_bigrams": self.bm25.analyzer.config.cjk_bigrams,
            }
        });
        if let Some(ref ivf) = self.ivf {
            if let Some(obj) = stats.as_object_mut() {
                obj.insert(
                    "ann".to_string(),
                    serde_json::json!({
                        "trained": ivf.is_trained,
                        "n_lists": ivf.centroids.len(),
                        "nprobe": ivf.config.nprobe,
                        "n_vectors": ivf.n_vectors,
                        "dirty": self.dirty_index,
                    }),
                );
            }
        }
        if let Some(ref hnsw) = self.hnsw {
            if let Some(obj) = stats.as_object_mut() {
                let hnsw_bytes = hnsw.memory_bytes();
                obj.insert(
                    "hnsw".to_string(),
                    serde_json::json!({
                        "count": active_count,
                        "layers": hnsw.layers.len(),
                        "m": hnsw.config.m,
                        "ef_construction": hnsw.config.ef_construction,
                        "ef_search": hnsw.config.ef_search,
                        "dirty": self.dirty_index,
                        "memory_bytes": hnsw_bytes,
                        "memory_mb": (hnsw_bytes as f64) / (1024.0 * 1024.0),
                    }),
                );
            }
        }
        stats
    }
}
