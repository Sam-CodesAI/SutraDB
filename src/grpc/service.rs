use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

use crate::collection::{NativeCollection, NativeDocument};
use crate::distance::Metric;
use crate::grpc::proto::sutra_service_server::SutraService;
use crate::grpc::proto::*;

#[derive(Clone)]
pub struct SutraGrpcService {
    collections: Arc<RwLock<HashMap<String, Arc<RwLock<NativeCollection>>>>>,
    start_time: Instant,
    default_data_dir: Option<PathBuf>,
}

impl SutraGrpcService {
    pub fn new(default_data_dir: Option<PathBuf>) -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
            default_data_dir,
        }
    }

    pub fn with_collection(name: String, collection: NativeCollection) -> Self {
        let mut map = HashMap::new();
        map.insert(name, Arc::new(RwLock::new(collection)));
        Self {
            collections: Arc::new(RwLock::new(map)),
            start_time: Instant::now(),
            default_data_dir: None,
        }
    }

    pub async fn add_collection(&self, name: String, collection: NativeCollection) {
        let mut map = self.collections.write().await;
        map.insert(name, Arc::new(RwLock::new(collection)));
    }
}

#[tonic::async_trait]
impl SutraService for SutraGrpcService {
    async fn health(&self, _request: Request<HealthRequest>) -> Result<Response<HealthResponse>, Status> {
        let count = self.collections.read().await.len() as u32;
        let uptime = self.start_time.elapsed().as_secs();
        Ok(Response::new(HealthResponse {
            status: "SERVING".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            collection_count: count,
            uptime_seconds: uptime,
        }))
    }

    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        if req.collection_name.trim().is_empty() {
            return Err(Status::invalid_argument("collection_name cannot be empty"));
        }
        if req.dimension == 0 {
            return Err(Status::invalid_argument("dimension must be greater than 0"));
        }

        let metric = Metric::from_str(&req.metric).map_err(|e| Status::invalid_argument(e.to_string()))?;

        let storage_path = req.data_dir.map(PathBuf::from).or_else(|| {
            self.default_data_dir
                .as_ref()
                .map(|dir| dir.join(&req.collection_name))
        });

        let mut map = self.collections.write().await;
        if map.contains_key(&req.collection_name) {
            return Ok(Response::new(CreateCollectionResponse {
                success: false,
                message: format!("Collection '{}' already exists", req.collection_name),
            }));
        }

        let col = NativeCollection::new(
            req.collection_name.clone(),
            req.dimension as usize,
            metric,
            storage_path,
            true,
        )
        .map_err(|e| Status::internal(e.to_string()))?;

        map.insert(req.collection_name.clone(), Arc::new(RwLock::new(col)));

        Ok(Response::new(CreateCollectionResponse {
            success: true,
            message: format!("Collection '{}' created successfully", req.collection_name),
        }))
    }

    async fn drop_collection(
        &self,
        request: Request<DropCollectionRequest>,
    ) -> Result<Response<DropCollectionResponse>, Status> {
        let req = request.into_inner();
        let mut map = self.collections.write().await;
        let removed = map.remove(&req.collection_name).is_some();
        Ok(Response::new(DropCollectionResponse { success: removed }))
    }

    async fn list_collections(
        &self,
        _request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        let map = self.collections.read().await;
        let collection_names = map.keys().cloned().collect();
        Ok(Response::new(ListCollectionsResponse { collection_names }))
    }

    async fn get_stats(
        &self,
        request: Request<CollectionStatsRequest>,
    ) -> Result<Response<CollectionStatsResponse>, Status> {
        let req = request.into_inner();
        let col_arc = {
            let map = self.collections.read().await;
            map.get(&req.collection_name).cloned()
        };
        let col_arc = match col_arc {
            Some(c) => c,
            None => {
                return Err(Status::not_found(format!(
                    "Collection '{}' not found",
                    req.collection_name
                )))
            }
        };

        let col = col_arc.read().await;
        Ok(Response::new(CollectionStatsResponse {
            name: col.name.clone(),
            count: col.count() as u64,
            dimension: col.dimension as u32,
            metric: col.metric.as_str().to_string(),
            is_dirty: col.dirty_index,
            wal_records: col.documents.len() as u64,
        }))
    }

    async fn insert(&self, request: Request<InsertRequest>) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();
        let col_arc = {
            let map = self.collections.read().await;
            map.get(&req.collection_name).cloned()
        };
        let col_arc = match col_arc {
            Some(c) => c,
            None => {
                return Err(Status::not_found(format!(
                    "Collection '{}' not found",
                    req.collection_name
                )))
            }
        };

        let mut docs = Vec::with_capacity(req.documents.len());
        let mut vecs = Vec::with_capacity(req.documents.len());

        for d in req.documents {
            let metadata = match d.metadata_json {
                Some(ref json_str) if !json_str.trim().is_empty() => serde_json::from_str(json_str)
                    .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
                _ => serde_json::Value::Object(serde_json::Map::new()),
            };

            let sparse_vector = d
                .sparse_vector
                .map(|sv| crate::sparse::SparseVector::new(sv.indices, sv.values));

            let colbert_tokens = d.colbert_tokens.map(|mv| {
                mv.vectors.into_iter().map(|v| v.values).collect()
            });

            docs.push(NativeDocument {
                id: d.id,
                namespace: d.namespace,
                text: d.text.unwrap_or_default(),
                metadata,
                sparse_vector,
                colbert_tokens,
            });
            vecs.push(d.vector);
        }

        let mut col = col_arc.write().await;
        let inserted = col
            .insert(docs, vecs)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(InsertResponse {
            inserted_count: inserted as u64,
            success: true,
            message: format!("Successfully inserted {} documents", inserted),
        }))
    }

    async fn delete(&self, request: Request<DeleteRequest>) -> Result<Response<DeleteResponse>, Status> {
        let req = request.into_inner();
        let col_arc = {
            let map = self.collections.read().await;
            map.get(&req.collection_name).cloned()
        };
        let col_arc = match col_arc {
            Some(c) => c,
            None => {
                return Err(Status::not_found(format!(
                    "Collection '{}' not found",
                    req.collection_name
                )))
            }
        };

        let mut col = col_arc.write().await;
        let deleted = col
            .delete(&req.ids)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(DeleteResponse {
            deleted_count: deleted as u64,
            success: true,
        }))
    }

    async fn query(&self, request: Request<QueryRequest>) -> Result<Response<QueryResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        let col_arc = {
            let map = self.collections.read().await;
            map.get(&req.collection_name).cloned()
        };
        let col_arc = match col_arc {
            Some(c) => c,
            None => {
                return Err(Status::not_found(format!(
                    "Collection '{}' not found",
                    req.collection_name
                )))
            }
        };

        let filter_val = match req.filter_json {
            Some(ref s) if !s.trim().is_empty() => {
                let parsed: serde_json::Value = serde_json::from_str(s)
                    .map_err(|e| Status::invalid_argument(format!("Invalid filter JSON: {}", e)))?;
                Some(parsed)
            }
            _ => None,
        };

        let sparse_vec = req
            .sparse_vector
            .map(|sv| crate::sparse::SparseVector::new(sv.indices, sv.values));
        let colbert_tokens: Option<Vec<Vec<f32>>> = req.colbert_tokens.map(|mv| {
            mv.vectors.into_iter().map(|v| v.values).collect()
        });
        let vector_opt = if !req.vector.is_empty() {
            Some(req.vector.as_slice())
        } else {
            None
        };
        let text_opt = req.text.as_deref().filter(|s| !s.is_empty());
        let top_k = if req.top_k > 0 { req.top_k as usize } else { 10 };
        let d_w = if req.dense_weight > 0.0 {
            req.dense_weight
        } else {
            1.0
        };
        let b_w = if req.bm25_weight > 0.0 {
            req.bm25_weight
        } else {
            1.0
        };
        let s_w = if req.sparse_weight > 0.0 {
            req.sparse_weight
        } else {
            1.0
        };
        let c_w = if req.colbert_weight > 0.0 {
            req.colbert_weight
        } else {
            1.0
        };

        let col = col_arc.read().await;
        let search_results = col
            .query(
                vector_opt,
                text_opt,
                sparse_vec.as_ref(),
                colbert_tokens.as_deref(),
                req.namespace.as_deref().filter(|s| !s.is_empty()),
                filter_val.as_ref(),
                top_k,
                req.hybrid,
                None,
                60,
                d_w,
                b_w,
                s_w,
                c_w,
                req.use_ann,
            )
            .map_err(|e| Status::internal(e.to_string()))?;

        let results = search_results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                score: r.score,
                text: if r.text.is_empty() { None } else { Some(r.text) },
                metadata_json: serde_json::to_string(&r.metadata).ok(),
                dense_score: r.dense_score,
                bm25_score: r.bm25_score,
                sparse_score: r.sparse_score,
                colbert_score: r.colbert_score,
                namespace: r.namespace,
            })
            .collect();

        let latency_micros = start.elapsed().as_micros() as u64;
        Ok(Response::new(QueryResponse {
            results,
            latency_micros,
        }))
    }

    async fn batch_query(
        &self,
        request: Request<BatchQueryRequest>,
    ) -> Result<Response<BatchQueryResponse>, Status> {
        let req = request.into_inner();
        let mut responses = Vec::with_capacity(req.queries.len());

        for query_req in req.queries {
            let res = self.query(Request::new(query_req)).await?;
            responses.push(res.into_inner());
        }

        Ok(Response::new(BatchQueryResponse { responses }))
    }

    async fn batch_distances(
        &self,
        request: Request<BatchDistanceRequest>,
    ) -> Result<Response<BatchDistanceResponse>, Status> {
        let req = request.into_inner();
        let dim = req.dimension as usize;
        if dim == 0 {
            return Err(Status::invalid_argument("Dimension must be greater than 0"));
        }
        if req.query_vector.len() != dim {
            return Err(Status::invalid_argument(format!(
                "Query vector length ({}) does not match dimension ({})",
                req.query_vector.len(),
                dim
            )));
        }
        if req.target_vectors.len() % dim != 0 {
            return Err(Status::invalid_argument(format!(
                "Target vectors length ({}) is not a multiple of dimension ({})",
                req.target_vectors.len(),
                dim
            )));
        }

        let metric = Metric::from_str(&req.metric).map_err(|e| Status::invalid_argument(e.to_string()))?;
        let num_targets = req.target_vectors.len() / dim;
        let mut scores = Vec::with_capacity(num_targets);

        let is_cosine = metric == Metric::Cosine;
        let q_norm = if is_cosine {
            crate::distance::simd::normalize_copy(&req.query_vector)
        } else {
            req.query_vector.clone()
        };

        for chunk in req.target_vectors.chunks_exact(dim) {
            let score = crate::distance::simd::compute_vector_score(&q_norm, chunk, metric, is_cosine)
                .map_err(|e| Status::internal(e.to_string()))?;
            scores.push(score);
        }

        Ok(Response::new(BatchDistanceResponse { scores }))
    }
}
