use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::bm25::BM25Index;
use crate::distance::simd::{compute_vector_score, normalize_copy};
use crate::distance::sq8::SQ8Vector;
use crate::distance::Metric;
use crate::error::SutraError;
use crate::filter::{FilterEngine, SecondaryIndex};
use crate::fusion::{linear_score_fusion, reciprocal_rank_fusion};
use crate::index::hnsw::{HNSWConfig, HNSWIndex};
use crate::index::ivf::{auto_n_lists, IVFConfig, IVFIndex};
use crate::storage::{BinaryStorage, MmapVectorStorage, WalRecord, WriteAheadLog};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeDocument {
    pub id: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub metadata: Value,
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
    pub id_to_index: HashMap<String, usize>,
    pub tombstones: Vec<bool>,
    pub bm25: BM25Index,
    pub hnsw: Option<HNSWIndex>,
    pub ivf: Option<IVFIndex>,
    pub secondary_index: SecondaryIndex,
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
            id_to_index: HashMap::new(),
            tombstones: Vec::new(),
            bm25: BM25Index::default(),
            hnsw: None,
            ivf: None,
            secondary_index: SecondaryIndex::new(),
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
                "insert" | "update" => {
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

                        docs.push(NativeDocument { id, namespace, text, metadata });
                        vecs.push(vec);
                    }
                    self.insert_internal(docs, vecs, false)?;
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
                    "vector": vec
                });
                wal_docs.push(doc_json);
            }

            if let Some(&existing_idx) = self.id_to_index.get(&doc.id) {
                // In-place update
                self.vectors[existing_idx] = vec.clone();
                if self.use_sq8 {
                    self.sq8_vectors[existing_idx] = SQ8Vector::quantize(&vec);
                }
                self.documents[existing_idx] = doc;
                if existing_idx < self.tombstones.len() {
                    self.tombstones[existing_idx] = false;
                }
                self.dirty_index = true;
                continue;
            }

            // New insertion
            let idx = self.documents.len();
            self.id_to_index.insert(doc.id.clone(), idx);

            if self.use_sq8 {
                self.sq8_vectors.push(SQ8Vector::quantize(&vec));
            }

            self.secondary_index.index_document(idx as u32, &doc.metadata);
            self.bm25.add_documents(&[doc.text.clone()]);
            self.vectors.push(vec);
            self.documents.push(doc);
            self.tombstones.push(false);
            self.dirty_index = true;
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
            self.secondary_index.index_document(idx as u32, &new_metadata);
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

        for id in ids {
            if let Some(&idx) = self.id_to_index.get(id) {
                if idx < self.tombstones.len() && !self.tombstones[idx] {
                    self.tombstones[idx] = true;
                    deleted_count += 1;
                    deleted_ids.push(id.clone());
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

    pub fn query(
        &mut self,
        vector: Option<&[f32]>,
        text: Option<&str>,
        namespace: Option<&str>,
        filter: Option<&Value>,
        top_k: usize,
        hybrid: bool,
        alpha: Option<f32>,
        rrf_constant: usize,
        dense_weight: f32,
        bm25_weight: f32,
        use_ann: bool,
    ) -> Result<Vec<NativeSearchResult>, SutraError> {
        let n = self.documents.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // 1. Build compound mask: metadata filter & tombstones & namespace isolation
        let metadata_list: Vec<Value> = self.documents.iter().map(|d| d.metadata.clone()).collect();
        let mut mask = FilterEngine::build_mask(&metadata_list, filter);

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

        // 2. Score Dense
        let is_cosine = self.metric == Metric::Cosine;
        let mut dense_scores = vec![0.0f32; n];

        if let Some(q_vec) = vector {
            let normalized_q = if is_cosine { normalize_copy(q_vec) } else { q_vec.to_vec() };

            // ANN path: HNSW or IVF if pure vector query without BM25
            if use_ann && !has_text && !self.dirty_index {
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
                            namespace: doc.namespace.clone(),
                        }
                    }).collect());
                }
            }

            // Brute force / exact scoring
            if self.use_sq8 {
                let q_sum: f32 = normalized_q.iter().sum();
                for (i, sq) in self.sq8_vectors.iter().enumerate() {
                    if mask[i] {
                        dense_scores[i] = sq.asymmetric_dot_product(&normalized_q, q_sum);
                    }
                }
            } else {
                for (i, v) in self.vectors.iter().enumerate() {
                    if mask[i] {
                        dense_scores[i] = compute_vector_score(&normalized_q, v, self.metric, is_cosine)?;
                    }
                }
            }
        }

        // 3. Score BM25
        let mut bm25_scores = vec![0.0f32; n];
        if has_text {
            bm25_scores = self.bm25.score_query(text.unwrap(), Some(&mask));
        }

        // 4. Fusion or routing
        if hybrid && has_vector && has_text {
            let fused = if let Some(a) = alpha {
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
                    namespace: doc.namespace.clone(),
                }
            }).collect());
        }

        Err(SutraError::InvalidQuery("Query must specify a vector, text, or both".into()))
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

        let mut flattened_vectors = Vec::with_capacity(self.documents.len() * self.dimension);
        for v in &self.vectors {
            flattened_vectors.extend_from_slice(v);
        }

        let doc_payloads: Vec<Value> = self.documents.iter().map(|d| {
            serde_json::json!({
                "id": d.id,
                "namespace": d.namespace,
                "text": d.text,
                "metadata": d.metadata
            })
        }).collect();

        let meta_bytes = serde_json::to_vec(&doc_payloads)
            .map_err(|e| SutraError::Serialization(e.to_string()))?;

        let written = BinaryStorage::write_f32(
            &p,
            &flattened_vectors,
            self.documents.len() as u32,
            self.dimension as u32,
            self.metric.as_str(),
            &meta_bytes,
        )?;

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
        self.metric = Metric::from_u8(header.quant_mode).unwrap_or(Metric::Cosine);

        let doc_count = header.doc_count as usize;
        self.documents.clear();
        self.vectors.clear();
        self.id_to_index.clear();
        self.tombstones.clear();
        self.bm25 = BM25Index::default();

        let doc_vals: Vec<Value> = serde_json::from_slice(storage.metadata_bytes())
            .map_err(|e| SutraError::Serialization(e.to_string()))?;

        for i in 0..doc_count {
            let vec_slice = storage.get_vector_f32(i);
            let d_val = &doc_vals[i];
            let id = d_val.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let namespace = d_val.get("namespace").and_then(|v| v.as_str()).map(|s| s.to_string());
            let text = d_val.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let metadata = d_val.get("metadata").cloned().unwrap_or(Value::Object(serde_json::Map::new()));

            self.id_to_index.insert(id.clone(), i);
            self.secondary_index.index_document(i as u32, &metadata);
            self.vectors.push(vec_slice.to_vec());
            self.documents.push(NativeDocument { id, namespace, text, metadata });
            self.tombstones.push(false);
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
        let mut new_bm25 = BM25Index::default();
        let mut new_sec = SecondaryIndex::new();

        for (i, doc) in self.documents.drain(..).enumerate() {
            if i < self.tombstones.len() && self.tombstones[i] {
                continue;
            }
            let idx = new_docs.len();
            new_id_map.insert(doc.id.clone(), idx);
            new_sec.index_document(idx as u32, &doc.metadata);
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
        self.hnsw = None;
        self.ivf = None;
        self.dirty_index = true;
        Ok(())
    }

    pub fn stats(&self) -> Value {
        let active_count = self.count();
        let vector_bytes = self.vectors.len() * self.dimension * 4;
        let sq8_bytes = if self.use_sq8 { self.sq8_vectors.len() * self.dimension } else { 0 };

        serde_json::json!({
            "name": self.name,
            "count": active_count,
            "total_records": self.documents.len(),
            "tombstones": self.documents.len() - active_count,
            "dimension": self.dimension,
            "metric": self.metric.as_str(),
            "vector_memory_bytes": vector_bytes + sq8_bytes,
            "vector_memory_mb": ((vector_bytes + sq8_bytes) as f64) / (1024.0 * 1024.0),
            "quantized_sq8": self.use_sq8,
            "has_hnsw": self.hnsw.is_some(),
            "has_ivf": self.ivf.is_some()
        })
    }
}
