use std::net::SocketAddr;
use std::time::Duration;
use tempfile::tempdir;
use tokio::time::sleep;

use _sutradb::grpc::proto::sutra_service_client::SutraServiceClient;
use _sutradb::grpc::proto::*;
use _sutradb::grpc::{SutraGrpcService, SutraServiceServer};

async fn spawn_test_server() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let tmp = tempdir().unwrap();
    let service = SutraGrpcService::new(Some(tmp.path().to_path_buf()));
    let server = SutraServiceServer::new(service);

    let handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(server)
            .serve(addr)
            .await
            .unwrap();
    });

    // Wait briefly for server socket bind
    sleep(Duration::from_millis(100)).await;
    (addr, handle)
}

#[tokio::test]
async fn test_grpc_server_full_lifecycle() {
    let (addr, _handle) = spawn_test_server().await;
    let endpoint = format!("http://{}", addr);

    let mut client = SutraServiceClient::connect(endpoint)
        .await
        .expect("Failed to connect to gRPC server");

    // 1. Health check
    let health_res = client
        .health(HealthRequest {})
        .await
        .expect("Health check failed")
        .into_inner();
    assert_eq!(health_res.status, "SERVING");
    assert_eq!(health_res.collection_count, 0);

    // 2. Create Collection
    let create_res = client
        .create_collection(CreateCollectionRequest {
            collection_name: "grpc_test_col".to_string(),
            dimension: 4,
            metric: "cosine".to_string(),
            data_dir: None,
        })
        .await
        .expect("CreateCollection failed")
        .into_inner();
    assert!(create_res.success);

    // 3. List Collections
    let list_res = client
        .list_collections(ListCollectionsRequest {})
        .await
        .expect("ListCollections failed")
        .into_inner();
    assert_eq!(list_res.collection_names, vec!["grpc_test_col"]);

    // 4. Insert Documents
    let docs = vec![
        Document {
            id: "doc-1".to_string(),
            vector: vec![1.0, 0.0, 0.0, 0.0],
            text: Some("Rust vector database engine".to_string()),
            sparse_vector: Some(SparseVector {
                indices: vec![1, 5],
                values: vec![0.8, 0.4],
            }),
            metadata_json: Some(r#"{"category": "db", "price": 10.0}"#.to_string()),
            namespace: Some("prod".to_string()),
            colbert_tokens: None,
        },
        Document {
            id: "doc-2".to_string(),
            vector: vec![0.0, 1.0, 0.0, 0.0],
            text: Some("High throughput gRPC binary streaming".to_string()),
            sparse_vector: Some(SparseVector {
                indices: vec![2, 5],
                values: vec![0.9, 0.2],
            }),
            metadata_json: Some(r#"{"category": "network", "price": 50.0}"#.to_string()),
            namespace: Some("prod".to_string()),
            colbert_tokens: None,
        },
        Document {
            id: "doc-3".to_string(),
            vector: vec![0.7071, 0.7071, 0.0, 0.0],
            text: Some("Hybrid search and secondary metadata indexing".to_string()),
            sparse_vector: Some(SparseVector {
                indices: vec![1, 2],
                values: vec![0.5, 0.5],
            }),
            metadata_json: Some(r#"{"category": "db", "price": 100.0}"#.to_string()),
            namespace: Some("prod".to_string()),
            colbert_tokens: None,
        },
    ];

    let insert_res = client
        .insert(InsertRequest {
            collection_name: "grpc_test_col".to_string(),
            documents: docs,
        })
        .await
        .expect("Insert failed")
        .into_inner();
    assert!(insert_res.success);
    assert_eq!(insert_res.inserted_count, 3);

    // 5. Get Stats
    let stats_res = client
        .get_stats(CollectionStatsRequest {
            collection_name: "grpc_test_col".to_string(),
        })
        .await
        .expect("GetStats failed")
        .into_inner();
    assert_eq!(stats_res.name, "grpc_test_col");
    assert_eq!(stats_res.count, 3);
    assert_eq!(stats_res.dimension, 4);
    assert_eq!(stats_res.metric, "cosine");

    // 6. Vector Query (Dense search)
    let query_dense = client
        .query(QueryRequest {
            collection_name: "grpc_test_col".to_string(),
            vector: vec![1.0, 0.0, 0.0, 0.0],
            text: None,
            sparse_vector: None,
            namespace: Some("prod".to_string()),
            filter_json: None,
            top_k: 2,
            hybrid: false,
            dense_weight: 1.0,
            bm25_weight: 0.0,
            sparse_weight: 0.0,
            use_ann: false,
            colbert_tokens: None,
            colbert_weight: 1.0,
        })
        .await
        .expect("Dense query failed")
        .into_inner();
    assert_eq!(query_dense.results.len(), 2);
    assert_eq!(query_dense.results[0].id, "doc-1");
    assert!((query_dense.results[0].score - 1.0).abs() < 1e-4);

    // 7. Metadata Filtered Query
    let query_filter = client
        .query(QueryRequest {
            collection_name: "grpc_test_col".to_string(),
            vector: vec![1.0, 0.0, 0.0, 0.0],
            text: None,
            sparse_vector: None,
            namespace: None,
            filter_json: Some(r#"{"category": {"$eq": "db"}}"#.to_string()),
            top_k: 10,
            hybrid: false,
            dense_weight: 1.0,
            bm25_weight: 0.0,
            sparse_weight: 0.0,
            use_ann: false,
            colbert_tokens: None,
            colbert_weight: 1.0,
        })
        .await
        .expect("Filtered query failed")
        .into_inner();
    assert_eq!(query_filter.results.len(), 2);
    let ids: Vec<&str> = query_filter.results.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"doc-1"));
    assert!(ids.contains(&"doc-3"));
    assert!(!ids.contains(&"doc-2"));

    // 8. Hybrid Tri-Modal Query (Dense + Lexical + Sparse)
    let query_hybrid = client
        .query(QueryRequest {
            collection_name: "grpc_test_col".to_string(),
            vector: vec![0.0, 1.0, 0.0, 0.0],
            text: Some("gRPC binary streaming".to_string()),
            sparse_vector: Some(SparseVector {
                indices: vec![2, 5],
                values: vec![1.0, 1.0],
            }),
            namespace: None,
            filter_json: None,
            top_k: 3,
            hybrid: true,
            dense_weight: 1.0,
            bm25_weight: 1.0,
            sparse_weight: 1.0,
            use_ann: false,
            colbert_tokens: None,
            colbert_weight: 1.0,
        })
        .await
        .expect("Hybrid query failed")
        .into_inner();
    assert!(!query_hybrid.results.is_empty());
    // doc-2 matches dense vector exactly, BM25 text, and sparse vector
    assert_eq!(query_hybrid.results[0].id, "doc-2");
    assert!(query_hybrid.results[0].sparse_score.is_some());

    // 9. Batch Distance SIMD computation
    let batch_dist = client
        .batch_distances(BatchDistanceRequest {
            query_vector: vec![1.0, 0.0, 0.0, 0.0],
            target_vectors: vec![
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
            ],
            dimension: 4,
            metric: "cosine".to_string(),
        })
        .await
        .expect("BatchDistances failed")
        .into_inner();
    assert_eq!(batch_dist.scores.len(), 2);
    assert!((batch_dist.scores[0] - 1.0).abs() < 1e-4);
    assert!(batch_dist.scores[1].abs() < 1e-4);

    // 10. Delete Document
    let del_res = client
        .delete(DeleteRequest {
            collection_name: "grpc_test_col".to_string(),
            ids: vec!["doc-2".to_string()],
        })
        .await
        .expect("Delete failed")
        .into_inner();
    assert!(del_res.success);
    assert_eq!(del_res.deleted_count, 1);

    // Verify doc-2 is deleted from stats and queries
    let stats_after = client
        .get_stats(CollectionStatsRequest {
            collection_name: "grpc_test_col".to_string(),
        })
        .await
        .expect("GetStats after delete failed")
        .into_inner();
    assert_eq!(stats_after.count, 2);

    // 11. Drop Collection
    let drop_res = client
        .drop_collection(DropCollectionRequest {
            collection_name: "grpc_test_col".to_string(),
        })
        .await
        .expect("DropCollection failed")
        .into_inner();
    assert!(drop_res.success);

    let list_after = client
        .list_collections(ListCollectionsRequest {})
        .await
        .expect("ListCollections after drop failed")
        .into_inner();
    assert!(list_after.collection_names.is_empty());
}
