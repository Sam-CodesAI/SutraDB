use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use serde_json::Value;

use crate::collection::{NativeCollection, NativeDocument, NativeSearchResult};
use crate::cluster::hash_ring::ConsistentHashRing;
use crate::error::SutraError;

/// Shard descriptor holding local collection instance and replica metadata.
pub struct Shard {
    pub shard_id: String,
    pub collection_name: String,
    pub collection: Arc<Mutex<NativeCollection>>,
}

/// Coordinates distributed multi-shard writes and scatter-gather hybrid queries.
pub struct ClusterCoordinator {
    pub node_id: String,
    pub hash_ring: Arc<RwLock<ConsistentHashRing>>,
    pub shards: Arc<RwLock<HashMap<String, Shard>>>,
}

impl ClusterCoordinator {
    pub fn new(node_id: &str, vnodes_per_node: usize) -> Self {
        // Do NOT add the coordinator's own node_id to the hash ring.
        // Only actual shard nodes should be ring members so that routing
        // never lands on the coordinator itself (which holds no data).
        let ring = ConsistentHashRing::new(vnodes_per_node);

        Self {
            node_id: node_id.to_string(),
            hash_ring: Arc::new(RwLock::new(ring)),
            shards: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Registers a new logical shard on this coordinator.
    pub fn register_shard(&self, shard_id: &str, collection: NativeCollection) {
        let col_name = collection.name.clone();
        let shard = Shard {
            shard_id: shard_id.to_string(),
            collection_name: col_name,
            collection: Arc::new(Mutex::new(collection)),
        };

        let mut shards = self.shards.write().unwrap_or_else(|p| p.into_inner());
        shards.insert(shard_id.to_string(), shard);

        let mut ring = self.hash_ring.write().unwrap_or_else(|p| p.into_inner());
        ring.add_node(shard_id);
    }

    /// Partitions an incoming batch of documents and vectors by hash ring key and inserts into target shards.
    pub fn insert_partitioned(
        &self,
        collection_name: &str,
        documents: Vec<NativeDocument>,
        vectors: Vec<Vec<f32>>,
    ) -> Result<usize, SutraError> {
        if documents.len() != vectors.len() {
            return Err(SutraError::DimensionMismatch {
                expected: documents.len(),
                got: vectors.len(),
            });
        }

        // Group documents by target shard using consistent hashing
        let mut shard_batches: HashMap<String, (Vec<NativeDocument>, Vec<Vec<f32>>)> = HashMap::new();

        {
            let ring = self.hash_ring.read().unwrap_or_else(|p| p.into_inner());
            for (doc, vec) in documents.into_iter().zip(vectors) {
                // Route by namespace if present (tenant-affinity), otherwise doc id
                let routing_key = doc.namespace.as_deref().unwrap_or(&doc.id);
                let target_shard = ring.get_node(routing_key).unwrap_or(&self.node_id).to_string();

                let batch = shard_batches.entry(target_shard).or_insert_with(|| (Vec::new(), Vec::new()));
                batch.0.push(doc);
                batch.1.push(vec);
            }
        }

        let mut total_inserted = 0;
        let shards = self.shards.read().unwrap_or_else(|p| p.into_inner());

        for (shard_id, (docs, vecs)) in shard_batches {
            let batch_len = docs.len();
            if let Some(shard) = shards.get(&shard_id) {
                if shard.collection_name == collection_name {
                    let mut col = shard.collection.lock().unwrap_or_else(|p| p.into_inner());
                    col.insert(docs, vecs)?;
                    total_inserted += batch_len;
                }
            } else {
                // In a multi-node deployment, remote shards receive routed gRPC call.
                // For local coordinator, fallback to any available shard matching collection name.
                for s in shards.values() {
                    if s.collection_name == collection_name {
                        let mut col = s.collection.lock().unwrap_or_else(|p| p.into_inner());
                        col.insert(docs, vecs)?;
                        total_inserted += batch_len;
                        break;
                    }
                }
            }
        }

        Ok(total_inserted)
    }

    /// Executes a scatter-gather query across all shards and merges partial results into a global top-K ranking.
    #[allow(clippy::too_many_arguments)]
    pub fn scatter_gather_query(
        &self,
        collection_name: &str,
        vector: Option<&[f32]>,
        text: Option<&str>,
        sparse_vector: Option<&crate::sparse::SparseVector>,
        colbert_tokens: Option<&[Vec<f32>]>,
        namespace: Option<&str>,
        filter: Option<&Value>,
        top_k: usize,
        hybrid: bool,
    ) -> Result<Vec<NativeSearchResult>, SutraError> {
        let matching_shards: Vec<Arc<Mutex<NativeCollection>>> = {
            let shards = self.shards.read().unwrap_or_else(|p| p.into_inner());
            shards.values()
                .filter(|s| s.collection_name == collection_name)
                .map(|s| Arc::clone(&s.collection))
                .collect()
        };

        if matching_shards.is_empty() {
            return Ok(Vec::new());
        }

        // Parallel query scattering across all shards using Rayon
        use rayon::prelude::*;
        let partial_results: Vec<Vec<NativeSearchResult>> = matching_shards.par_iter().map(|shard_arc| {
            let col = shard_arc.lock().unwrap_or_else(|p| p.into_inner());
            col.query(
                vector,
                text,
                sparse_vector,
                colbert_tokens,
                namespace,
                filter,
                top_k,
                hybrid,
                None, // alpha
                60,   // rrf_constant
                1.0,  // dense_weight
                1.0,  // bm25_weight
                1.0,  // sparse_weight
                1.0,  // colbert_weight
                false // use_ann
            ).unwrap_or_default()
        }).collect();

        // Gather & merge stage: candidate deduplication and score fusion
        let mut merged: Vec<NativeSearchResult> = Vec::new();
        let mut seen_ids: HashSet<String> = HashSet::new();

        for shard_res in partial_results {
            for item in shard_res {
                if seen_ids.insert(item.id.clone()) {
                    merged.push(item);
                }
            }
        }

        // Sort descending by merged score
        merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        merged.truncate(top_k);

        Ok(merged)
    }

    /// Returns collection total document count aggregated across all shards.
    pub fn cluster_count(&self, collection_name: &str) -> usize {
        let shards = self.shards.read().unwrap_or_else(|p| p.into_inner());
        shards.values()
            .filter(|s| s.collection_name == collection_name)
            .map(|s| s.collection.lock().unwrap_or_else(|p| p.into_inner()).count())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    #[test]
    fn test_cluster_coordinator_sharded_write_and_scatter_gather() {
        let coord = ClusterCoordinator::new("coordinator-0", 64);

        // Register 3 shards for the "wiki" collection
        for i in 0..3 {
            let shard_id = format!("shard-{}", i);
            let col = NativeCollection::new("wiki".to_string(), 4, Metric::Cosine, None, false).unwrap();
            coord.register_shard(&shard_id, col);
        }

        // Partition 6 documents across shards
        let docs = vec![
            NativeDocument { id: "d1".to_string(), namespace: Some("tenant_1".to_string()), text: "Kubernetes clustering".to_string(), metadata: serde_json::json!({"tag": "k8s"}), sparse_vector: None, colbert_tokens: None },
            NativeDocument { id: "d2".to_string(), namespace: Some("tenant_2".to_string()), text: "PostgreSQL streaming replication".to_string(), metadata: serde_json::json!({"tag": "db"}), sparse_vector: None, colbert_tokens: None },
            NativeDocument { id: "d3".to_string(), namespace: Some("tenant_1".to_string()), text: "Raft consensus protocol".to_string(), metadata: serde_json::json!({"tag": "k8s"}), sparse_vector: None, colbert_tokens: None },
            NativeDocument { id: "d4".to_string(), namespace: Some("tenant_3".to_string()), text: "Vector database SIMD speed".to_string(), metadata: serde_json::json!({"tag": "ai"}), sparse_vector: None, colbert_tokens: None },
            NativeDocument { id: "d5".to_string(), namespace: Some("tenant_2".to_string()), text: "Distributed transaction logs".to_string(), metadata: serde_json::json!({"tag": "db"}), sparse_vector: None, colbert_tokens: None },
            NativeDocument { id: "d6".to_string(), namespace: Some("tenant_3".to_string()), text: "ColBERT multi-vector retrieval".to_string(), metadata: serde_json::json!({"tag": "ai"}), sparse_vector: None, colbert_tokens: None },
        ];
        let vecs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0, 0.0],
            vec![0.0, 0.0, 0.9, 0.1],
        ];

        let count = coord.insert_partitioned("wiki", docs, vecs).unwrap();
        assert_eq!(count, 6);
        assert_eq!(coord.cluster_count("wiki"), 6);

        // Scatter-gather query across all shards
        let q_vec = [1.0, 0.0, 0.0, 0.0];
        let results = coord.scatter_gather_query(
            "wiki",
            Some(&q_vec),
            Some("Kubernetes Raft"),
            None,
            None,
            None,
            None,
            3,
            true,
        ).unwrap();

        assert_eq!(results.len(), 3);
        // d1 and d3 match best on vector and text
        let top_ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
        assert!(top_ids.contains(&"d1".to_string()));
        assert!(top_ids.contains(&"d3".to_string()));

        // Tenant-isolated scatter-gather query
        let tenant_1_res = coord.scatter_gather_query(
            "wiki",
            Some(&q_vec),
            None,
            None,
            None,
            Some("tenant_1"),
            None,
            5,
            false,
        ).unwrap();

        for r in &tenant_1_res {
            assert_eq!(r.namespace.as_deref(), Some("tenant_1"));
        }
        assert_eq!(tenant_1_res.len(), 2);
    }
}
