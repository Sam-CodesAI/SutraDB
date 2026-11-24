use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::bm25::BM25Index;
use crate::distance::bq::BQVector;
use crate::distance::pq::{PQCodebook, PQConfig, PQVector};
use crate::distance::simd::{compute_vector_score, normalize_copy};
use crate::distance::sq8::SQ8Vector;
use crate::distance::Metric;
use crate::error::SutraError;
use crate::filter::{FilterEngine, SecondaryIndex};
use crate::fusion::{linear_score_fusion, reciprocal_rank_fusion};
use crate::index::hnsw::{HNSWConfig, HNSWIndex};
use crate::index::ivf::{auto_n_lists, IVFConfig, IVFIndex};
use crate::storage::{BinaryStorage, MmapVectorStorage, WalRecord, WriteAheadLog};

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
    pub vectors: Vec<Vec<f32>>,
    pub sq8_vectors: Vec<SQ8Vector>,
    pub use_sq8: bool,
    pub bq_vectors: Vec<BQVector>,
    pub use_bq: bool,
    pub pq_codebook: Option<PQCodebook>,
    pub pq_vectors: Vec<PQVector>,
    pub use_pq: bool,
    pub id_to_index: HashMap<String, usize>,
    pub tombstones: Vec<bool>,
    pub bm25: BM25Index,
    pub hnsw: Option<HNSWIndex>,
    pub ivf: Option<IVFIndex>,
    pub secondary_index: SecondaryIndex,
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
            vectors: Vec::new(),
            sq8_vectors: Vec::new(),
            use_sq8: false,
            bq_vectors: Vec::new(),
            use_bq: false,
            pq_codebook: None,
            pq_vectors: Vec::new(),
            use_pq: false,
            id_to_index: HashMap::new(),
            tombstones: Vec::new(),
            bm25: BM25Index::with_analyzer(1.5, 0.75, analyzer),
            hnsw: None,
            ivf: None,
            secondary_index: SecondaryIndex::new(),
            metadata_index: crate::filter::SecondaryMetadataIndex::new(),
            sparse_index: crate::sparse::SparseIndex::new(),
            colbert_index: crate::colbert::ColbertIndex::new(dimension),
            wal,
            dirty_index: false,
        };

        col.replay_wal_if_present()?;
        Ok(col)
    }

    fn replay_wal_if_present(&mut self) -> Result<(), SutraError> {
        let wal = match self.wal.as_mut() {
            Some(w) => w,
            None => return Ok(()),
        };

        let records = wal.replay()?;
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

                        let sparse_vector = d_val.get("sparse_vector").and_then(|v| {
                            if let Some(map) = v.as_object() {
                                let mut pairs: Vec<(u32, f32)> = map.iter().filter_map(|(k, val)| {
                                    let idx = k.parse::<u32>().ok()?;
                                    let f = val.as_f64()? as f32;
                                    Some((idx, f))
                                }).collect();
                                pairs.sort_by_key(|p| p.0);
                                let idxs = pairs.iter().map(|p| p.0).collect();
                                let vals = pairs.iter().map(|p| p.1).collect();
                                Some(crate::sparse::SparseVector::new(idxs, vals))
                            } else {
                                None
                            }
                        });

                        let colbert_tokens: Option<Vec<Vec<f32>>> = d_val.get("colbert_tokens").and_then(|v| {
                            v.as_array().map(|arr| {
                                arr.iter().filter_map(|tok_arr| {
                                    tok_arr.as_array().map(|vec_arr| {
                                        vec_arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect()
                                    })
                                }).collect()
                            })
                        });

                        docs.push(NativeDocument { id, namespace, text, metadata, sparse_vector, colbert_tokens });
                        vecs.push(vec);
                    }
                    self.insert_internal(docs, vecs, false)?;
                }
                "update" => {
                    for d_val in r.docs {
                        if let Some(id) = d_val.get("id").and_then(|v| v.as_str()) {
                            let metadata = d_val.get("metadata").cloned().unwrap_or(Value::Object(serde_json::Map::new()));
                            if let Some(&idx) = self.id_to_index.get(id) {
                                if idx < self.tombstones.len() && !self.tombstones[idx] {
                                    if let Some(target) = self.documents.get_mut(idx) {
                                        target.metadata = metadata.clone();
                                        self.secondary_index.remove_document(idx as u32);
                                        self.secondary_index.index_document(idx as u32, &metadata);
                                        self.metadata_index.remove_document(idx as u32);
                                        self.metadata_index.index_document(idx as u32, &metadata);
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

        let mut wal_docs = Vec::new();
        let is_cosine = self.metric == Metric::Cosine;

        for (doc, mut vec) in documents.into_iter().zip(vectors.into_iter()) {
            if vec.len() != self.dimension {
                return Err(SutraError::DimensionMismatch {
                    expected: self.dimension,
                    got: vec.len(),
                });
            }

            if is_cosine {
                crate::distance::simd::normalize_in_place(&mut vec);
            }

            if log_to_wal && self.wal.is_some() {
                let doc_json = serde_json::json!({
                    "id": doc.id,
                    "namespace": doc.namespace,
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "vector": vec,
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

            if let Some(&existing_idx) = self.id_to_index.get(&doc.id) {
                // In-place update
                self.vectors[existing_idx] = vec.clone();
                if self.use_sq8 {
                    self.sq8_vectors[existing_idx] = SQ8Vector::quantize(&vec);
                }
                if self.use_bq {
                    self.bq_vectors[existing_idx] = BQVector::quantize(&vec);
                }
                if self.use_pq {
                    if let Some(ref cb) = self.pq_codebook {
                        self.pq_vectors[existing_idx] = cb.encode(&vec);
                    }
                }
                self.secondary_index.remove_document(existing_idx as u32);
                self.secondary_index.index_document(existing_idx as u32, &doc.metadata);
                self.metadata_index.remove_document(existing_idx as u32);
                self.metadata_index.index_document(existing_idx as u32, &doc.metadata);
                self.sparse_index.remove_document(existing_idx as u32);
                if let Some(ref sv) = doc.sparse_vector {
                    self.sparse_index.add_document(existing_idx as u32, sv);
                }
                self.colbert_index.remove_document(existing_idx as u32);
                if let Some(ref tokens) = doc.colbert_tokens {
                    self.colbert_index.add_document(existing_idx as u32, tokens.clone());
                }
                let old_text = &self.documents[existing_idx].text;
                self.bm25.update_document(existing_idx as u32, old_text, &doc.text);
                self.documents[existing_idx] = doc;
                if existing_idx < self.tombstones.len() {
                    self.tombstones[existing_idx] = false;
                }
                if let Some(ref mut hnsw) = self.hnsw {
                    hnsw.delete_node(existing_idx as u32, &self.vectors, self.metric, is_cosine);
                    let _ = hnsw.insert(existing_idx as u32, &vec, &self.vectors, self.metric, is_cosine);
                    if self.ivf.is_some() {
                        self.dirty_index = true;
                    }
                } else {
                    self.dirty_index = true;
                }
                continue;
            }

            // New insertion
            let idx = self.documents.len();
            self.id_to_index.insert(doc.id.clone(), idx);

            if self.use_sq8 {
                self.sq8_vectors.push(SQ8Vector::quantize(&vec));
            }
            if self.use_bq {
                self.bq_vectors.push(BQVector::quantize(&vec));
            }
            if self.use_pq {
                if let Some(ref cb) = self.pq_codebook {
                    self.pq_vectors.push(cb.encode(&vec));
                }
            }

            self.secondary_index.index_document(idx as u32, &doc.metadata);
            self.metadata_index.index_document(idx as u32, &doc.metadata);
            if let Some(ref sv) = doc.sparse_vector {
                self.sparse_index.add_document(idx as u32, sv);
            }
            if let Some(ref tokens) = doc.colbert_tokens {
                self.colbert_index.add_document(idx as u32, tokens.clone());
            }
            self.bm25.add_documents(&[doc.text.clone()]);
            self.vectors.push(vec.clone());
            self.documents.push(doc);
            self.tombstones.push(false);
            if let Some(ref mut hnsw) = self.hnsw {
                let _ = hnsw.insert(idx as u32, &vec, &self.vectors, self.metric, is_cosine);
                if self.ivf.is_some() {
                    self.dirty_index = true;
                }
            } else {
                self.dirty_index = true;
            }
        }

        if log_to_wal && !wal_docs.is_empty() {
            if let Some(ref mut wal) = self.wal {
                wal.append(&WalRecord {
                    op: "insert".to_string(),
                    docs: wal_docs,
                    ids: Vec::new(),
                })?;
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
            self.secondary_index.remove_document(idx as u32);
            self.secondary_index.index_document(idx as u32, &new_metadata);
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

                    self.secondary_index.remove_document(idx as u32);
                    self.metadata_index.remove_document(idx as u32);
                    self.sparse_index.remove_document(idx as u32);
                    self.colbert_index.remove_document(idx as u32);

                    // Dynamic online HNSW node deletion and edge healing
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
        let cfg = config.unwrap_or_default();
        let mut index = HNSWIndex::new(cfg);
        let is_cosine = self.metric == Metric::Cosine;

        for (i, v) in self.vectors.iter().enumerate() {
            if i < self.tombstones.len() && self.tombstones[i] {
                continue;
            }
            index.insert(i as u32, v, &self.vectors, self.metric, is_cosine)?;
        }

        self.hnsw = Some(index);
        self.dirty_index = false;
        Ok(())
    }

    pub fn build_ivf(&mut self, config: Option<IVFConfig>) -> Result<(), SutraError> {
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

        let mut index = IVFIndex::new(cfg);
        let active_vectors: Vec<Vec<f32>> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| *i >= self.tombstones.len() || !self.tombstones[*i])
            .map(|(_, v)| v.clone())
            .collect();

        index.train(&active_vectors, self.dimension)?;

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

    pub fn enable_sq8(&mut self) {
        if !self.use_sq8 {
            self.sq8_vectors = self.vectors.iter().map(|v| SQ8Vector::quantize(v)).collect();
            self.use_sq8 = true;
        }
    }

    pub fn enable_bq(&mut self) {
        if !self.use_bq {
            self.bq_vectors = self.vectors.iter().map(|v| BQVector::quantize(v)).collect();
            self.use_bq = true;
        }
    }

    pub fn train_pq(&mut self, m: Option<usize>, max_train_samples: Option<usize>) -> Result<(), SutraError> {
        let active_count = self.count();
        if active_count == 0 {
            return Err(SutraError::IndexNotTrained("Cannot train PQ on empty collection".into()));
        }

        let m_val = m.unwrap_or_else(|| PQCodebook::suggest_m(self.dimension));
        let active_vectors: Vec<Vec<f32>> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| *i >= self.tombstones.len() || !self.tombstones[*i])
            .map(|(_, v)| v.clone())
            .collect();

        let train_samples = if let Some(limit) = max_train_samples {
            active_vectors.iter().take(limit).cloned().collect()
        } else {
            active_vectors.clone()
        };

        let cfg = PQConfig {
            m: m_val,
            k: 256.min(train_samples.len()),
            kmeans_iters: 20,
            seed: 42,
        };

        let codebook = PQCodebook::train(&train_samples, self.dimension, cfg)?;
        self.pq_vectors = self.vectors.iter().map(|v| codebook.encode(v)).collect();
        self.pq_codebook = Some(codebook);
        self.use_pq = true;
        Ok(())
    }

    pub fn enable_pq(&mut self) -> Result<(), SutraError> {
        if !self.use_pq {
            if self.pq_codebook.is_none() {
                self.train_pq(None, None)?;
            } else {
                let cb = self.pq_codebook.as_ref().unwrap();
                self.pq_vectors = self.vectors.iter().map(|v| cb.encode(v)).collect();
                self.use_pq = true;
            }
        }
        Ok(())
    }

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
        let n = self.documents.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // 1. Build compound mask: metadata filter & tombstones & namespace isolation
        // Sub-microsecond filter evaluation via SecondaryMetadataIndex
        let mut mask = vec![true; n];
        if let Some(bitmap) = self.metadata_index.evaluate_filter(filter) {
            for i in 0..n {
                mask[i] = bitmap.contains(i as u32);
            }
        }

        // If the filter has complex operators ($geo_within, $geo_box, $regex, etc.), check remaining matching candidates
        if filter.map(|f| crate::filter::has_complex_operators(f)).unwrap_or(false) {
            for i in 0..n {
                if mask[i] {
                    mask[i] = FilterEngine::evaluate(Some(&self.documents[i].metadata), filter);
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
                if i < mask.len() && mask[i] {
                    if doc.namespace.as_deref() != Some(ns) {
                        mask[i] = false;
                    }
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
        // If valid_count is small (<= 512 or < 2% of collection),
        // HNSW graph walking suffers from filter collapse (hitting dead ends).
        // Bypassing HNSW to direct SIMD scan over the small candidate pool guarantees 100% recall in microseconds!
        let valid_ratio = valid_count as f32 / n as f32;
        let should_bypass_hnsw = valid_count <= 512 || valid_ratio < 0.02;

        let is_cosine = self.metric == Metric::Cosine;
        let mut dense_scores = vec![0.0f32; n];

        if let Some(q_vec) = vector {
            let normalized_q = if is_cosine { normalize_copy(q_vec) } else { q_vec.to_vec() };

            // ANN path: HNSW or IVF if pure vector query without BM25 and selectivity is high enough
            if use_ann && !has_text && !has_sparse && !has_colbert && !self.dirty_index && !should_bypass_hnsw {
                if let Some(ref hnsw) = self.hnsw {
                    let ann_res = hnsw.search(&normalized_q, &self.vectors, top_k, 32, self.metric, is_cosine, Some(&mask))?;
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
            if self.use_pq && self.pq_codebook.is_some() {
                let cb = self.pq_codebook.as_ref().unwrap();
                let lut = cb.compute_lut(&normalized_q, self.metric);
                let mut candidates: Vec<(usize, f32)> = Vec::new();

                for (i, pqv) in self.pq_vectors.iter().enumerate() {
                    if mask[i] {
                        let score = cb.score_adc(pqv, &lut, self.metric);
                        candidates.push((i, score));
                    }
                }

                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let rerank_count = (top_k * 4).max(32).min(candidates.len());
                for &(idx, _) in candidates.iter().take(rerank_count) {
                    dense_scores[idx] = compute_vector_score(&normalized_q, &self.vectors[idx], self.metric, is_cosine)?;
                }
            } else if self.use_bq {
                // Stage 1: Fast Hamming distance scan over 1-bit quantized bitmasks
                let q_bq = BQVector::quantize(&normalized_q);
                let mut candidates: Vec<(usize, u32)> = Vec::new();
                for (i, bqv) in self.bq_vectors.iter().enumerate() {
                    if mask[i] {
                        let d = q_bq.hamming_distance(bqv);
                        candidates.push((i, d));
                    }
                }
                candidates.sort_by_key(|&(_, d)| d);

                // Stage 2: Rerank top M candidates with full precision / SQ8
                let rerank_count = (top_k * 4).max(32).min(candidates.len());
                let q_sum = if self.use_sq8 { normalized_q.iter().sum() } else { 0.0 };
                for &(idx, _) in candidates.iter().take(rerank_count) {
                    if self.use_sq8 {
                        dense_scores[idx] = self.sq8_vectors[idx].score(&normalized_q, q_sum, self.metric);
                    } else {
                        dense_scores[idx] = compute_vector_score(&normalized_q, &self.vectors[idx], self.metric, is_cosine)?;
                    }
                }
            } else if self.use_sq8 {
                let q_sum: f32 = normalized_q.iter().sum();
                if self.sq8_vectors.len() >= 64 {
                    use rayon::prelude::*;
                    dense_scores = self.sq8_vectors.par_iter().enumerate().map(|(i, sq)| {
                        if mask[i] {
                            sq.score(&normalized_q, q_sum, self.metric)
                        } else {
                            0.0
                        }
                    }).collect();
                } else {
                    for (i, sq) in self.sq8_vectors.iter().enumerate() {
                        if mask[i] {
                            dense_scores[i] = sq.score(&normalized_q, q_sum, self.metric);
                        }
                    }
                }
            } else {
                if self.vectors.len() >= 64 {
                    use rayon::prelude::*;
                    dense_scores = self.vectors.par_iter().enumerate().map(|(i, v)| {
                        if mask[i] {
                            compute_vector_score(&normalized_q, v, self.metric, is_cosine).unwrap_or(0.0)
                        } else {
                            0.0
                        }
                    }).collect();
                } else {
                    for (i, v) in self.vectors.iter().enumerate() {
                        if mask[i] {
                            dense_scores[i] = compute_vector_score(&normalized_q, v, self.metric, is_cosine)?;
                        }
                    }
                }
            }
        }

        // 3. Score BM25
        let mut bm25_scores = vec![0.0f32; n];
        if has_text {
            bm25_scores = self.bm25.score_query(text.unwrap(), Some(&mask));
        }

        // 4. Score Neural Sparse Vectors
        let mut sparse_scores = vec![0.0f32; n];
        if let Some(sv) = sparse_vector {
            let sparse_hits = self.sparse_index.search(sv, n, Some(&mask));
            for (id, sc) in sparse_hits {
                sparse_scores[id as usize] = sc;
            }
        }

        // 5. Score ColBERT Multi-Vectors
        let mut colbert_scores = vec![0.0f32; n];
        if let Some(ct) = colbert_tokens {
            let colbert_hits = self.colbert_index.query_maxsim(ct, Some(&mask), n);
            for (id, sc) in colbert_hits {
                let idx = id as usize;
                if idx < n {
                    colbert_scores[idx] = sc;
                }
            }
        }

        // 6. Fusion or routing
        let num_modalities = (has_vector as usize) + (has_text as usize) + (has_sparse as usize) + (has_colbert as usize);
        if hybrid && num_modalities >= 2 {
            let fused = if has_colbert {
                crate::fusion::quad_modal_reciprocal_rank_fusion(
                    &dense_scores,
                    &bm25_scores,
                    &sparse_scores,
                    &colbert_scores,
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
                    &sparse_scores,
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
            let mut valid: Vec<(usize, f32)> = (0..n).filter(|&i| mask[i]).map(|i| (i, dense_scores[i])).collect();
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
        let n = self.documents.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let mut mask = vec![true; n];
        if let Some(bitmap) = self.metadata_index.evaluate_filter(filter) {
            for i in 0..n {
                mask[i] = bitmap.contains(i as u32);
            }
        }
        if filter.map(|f| crate::filter::has_complex_operators(f)).unwrap_or(false) {
            for i in 0..n {
                if mask[i] {
                    mask[i] = FilterEngine::evaluate(Some(&self.documents[i].metadata), filter);
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
                if i < mask.len() && mask[i] {
                    if doc.namespace.as_deref() != Some(ns) {
                        mask[i] = false;
                    }
                }
            }
        }

        let hits = self.colbert_index.query_maxsim(query_tokens, Some(&mask), top_k);
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
        let p = match path {
            Some(ref path_ref) => path_ref.as_ref().to_path_buf(),
            None => match self.storage_path {
                Some(ref sp) => sp.clone(),
                None => return Err(SutraError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "No storage path"))),
            },
        };

        // Compact tombstones before saving
        self.compact()?;

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
            let mut flattened_sq8 = Vec::with_capacity(self.documents.len() * self.dimension);
            for sq in &self.sq8_vectors {
                flattened_sq8.extend_from_slice(&sq.data);
            }
            BinaryStorage::write_sq8(
                &p,
                &flattened_sq8,
                self.documents.len() as u32,
                self.dimension as u32,
                self.metric.as_str(),
                &meta_bytes,
            )?
        } else {
            let mut flattened_vectors = Vec::with_capacity(self.documents.len() * self.dimension);
            for v in &self.vectors {
                flattened_vectors.extend_from_slice(v);
            }
            BinaryStorage::write_f32(
                &p,
                &flattened_vectors,
                self.documents.len() as u32,
                self.dimension as u32,
                self.metric.as_str(),
                &meta_bytes,
            )?
        };

        if let Some(ref mut wal) = self.wal {
            wal.clear()?;
        }

        Ok(written)
    }

    pub fn load<P: AsRef<Path>>(&mut self, path: Option<P>) -> Result<usize, SutraError> {
        let p = match path {
            Some(ref path_ref) => path_ref.as_ref().to_path_buf(),
            None => match self.storage_path {
                Some(ref sp) => sp.clone(),
                None => return Err(SutraError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "No storage path"))),
            },
        };

        let storage = MmapVectorStorage::open(&p)?;
        let header = storage.header();
        self.dimension = header.dimension as usize;
        self.metric = Metric::from_str(&header.metric).unwrap_or(Metric::Cosine);

        let doc_count = header.doc_count as usize;
        self.documents.clear();
        self.vectors.clear();
        self.id_to_index.clear();
        self.tombstones.clear();
        self.bm25 = BM25Index::with_analyzer(self.bm25.k1, self.bm25.b, self.bm25.analyzer.clone());
        self.secondary_index.clear();
        self.metadata_index.clear();
        self.sparse_index = crate::sparse::SparseIndex::new();
        self.colbert_index = crate::colbert::ColbertIndex::new(self.colbert_index.token_dim);
        self.sq8_vectors.clear();
        self.bq_vectors.clear();
        self.pq_vectors.clear();
        self.use_sq8 = false;
        self.use_bq = false;
        self.use_pq = false;
        self.hnsw = None;
        self.ivf = None;

        let doc_vals: Vec<Value> = serde_json::from_slice(storage.metadata_bytes())
            .map_err(|e| SutraError::Serialization(e.to_string()))?;

        for i in 0..doc_count {
            let vec: Vec<f32> = if header.quant_mode == 1 {
                let sq8_slice = storage.get_vector_sq8(i);
                sq8_slice.iter().map(|&b| (b as f32) / 255.0).collect()
            } else {
                storage.get_vector_f32(i).to_vec()
            };
            let d_val = &doc_vals[i];
            let id = d_val.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let namespace = d_val.get("namespace").and_then(|v| v.as_str()).map(|s| s.to_string());
            let text = d_val.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let metadata = d_val.get("metadata").cloned().unwrap_or(Value::Object(serde_json::Map::new()));

            let sparse_vec = d_val.get("sparse_vector").and_then(|v| {
                if let Some(map) = v.as_object() {
                    let mut pairs: Vec<(u32, f32)> = map.iter().filter_map(|(k, val)| {
                        let idx = k.parse::<u32>().ok()?;
                        let f = val.as_f64()? as f32;
                        Some((idx, f))
                    }).collect();
                    pairs.sort_by_key(|p| p.0);
                    let idxs = pairs.iter().map(|p| p.0).collect();
                    let vals = pairs.iter().map(|p| p.1).collect();
                    Some(crate::sparse::SparseVector::new(idxs, vals))
                } else {
                    None
                }
            });

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
                self.colbert_index.add_document(i as u32, tokens.clone());
            }

            self.id_to_index.insert(id.clone(), i);
            self.secondary_index.index_document(i as u32, &metadata);
            self.metadata_index.index_document(i as u32, &metadata);
            self.vectors.push(vec);
            self.documents.push(NativeDocument { id, namespace, text, metadata, sparse_vector: sparse_vec, colbert_tokens });
            self.tombstones.push(false);
        }

        if header.quant_mode == 1 {
            self.enable_sq8();
        }

        let corpus: Vec<String> = self.documents.iter().map(|d| d.text.clone()).collect();
        self.bm25.add_documents(&corpus);

        self.replay_wal_if_present()?;
        Ok(self.count())
    }

    pub fn compact(&mut self) -> Result<(), SutraError> {
        let has_tombstones = self.tombstones.iter().any(|&t| t);
        if !has_tombstones {
            return Ok(());
        }

        let mut new_docs = Vec::new();
        let mut new_vecs = Vec::new();
        let mut new_id_map = HashMap::new();
        let mut new_bm25 = BM25Index::with_analyzer(self.bm25.k1, self.bm25.b, self.bm25.analyzer.clone());
        let mut new_sec = SecondaryIndex::new();
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
            new_sec.index_document(idx as u32, &doc.metadata);
            new_meta_idx.index_document(idx as u32, &doc.metadata);
            if let Some(ref sv) = doc.sparse_vector {
                new_sparse_idx.add_document(idx as u32, sv);
            }
            if let Some(ref tokens) = doc.colbert_tokens {
                new_colbert_idx.add_document(idx as u32, tokens.clone());
            }
            new_vecs.push(self.vectors[i].clone());
            new_docs.push(doc);
        }

        let corpus: Vec<String> = new_docs.iter().map(|d| d.text.clone()).collect();
        new_bm25.add_documents(&corpus);

        self.documents = new_docs;
        self.vectors = new_vecs;
        self.id_to_index = new_id_map;
        self.tombstones = vec![false; self.documents.len()];
        self.bm25 = new_bm25;
        self.secondary_index = new_sec;
        self.metadata_index = new_meta_idx;
        self.sparse_index = new_sparse_idx;
        self.colbert_index = new_colbert_idx;
        if self.use_sq8 {
            self.sq8_vectors = self.vectors.iter().map(|v| SQ8Vector::quantize(v)).collect();
        }
        if self.use_bq {
            self.bq_vectors = self.vectors.iter().map(|v| BQVector::quantize(v)).collect();
        }
        if self.use_pq {
            if let Some(ref cb) = self.pq_codebook {
                self.pq_vectors = self.vectors.iter().map(|v| cb.encode(v)).collect();
            }
        }
        self.hnsw = None;
        self.ivf = None;
        self.dirty_index = true;
        Ok(())
    }

    pub fn stats(&self) -> Value {
        let active_count = self.count();
        let vector_bytes = self.vectors.len() * self.dimension * 4;
        let sq8_bytes = if self.use_sq8 { self.sq8_vectors.len() * self.dimension } else { 0 };
        let bq_bytes = if self.use_bq { self.bq_vectors.len() * ((self.dimension + 63) / 64) * 8 } else { 0 };
        let pq_bytes = if self.use_pq && self.pq_codebook.is_some() {
            self.pq_vectors.len() * self.pq_codebook.as_ref().unwrap().m
        } else {
            0
        };
        let total_vec_bytes = vector_bytes + sq8_bytes + bq_bytes + pq_bytes;

        serde_json::json!({
            "name": self.name,
            "count": active_count,
            "total_records": self.documents.len(),
            "tombstones": self.documents.len() - active_count,
            "dimension": self.dimension,
            "metric": self.metric.as_str(),
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
        })
    }
}
