use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::{tempdir, TempDir};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio::time::sleep;
use tonic::Code;

use _sutradb::grpc::http_bridge;
use _sutradb::grpc::proto::sutra_service_client::SutraServiceClient;
use _sutradb::grpc::proto::*;
use _sutradb::grpc::{SutraGrpcService, SutraServiceServer};
use _sutradb::storage::wal::{WalRecord, WriteAheadLog};

async fn spawn_test_server_with_dir(
    data_dir: Option<PathBuf>,
) -> (
    SocketAddr,
    SocketAddr,
    watch::Sender<bool>,
    tokio::task::JoinHandle<()>,
    tokio::task::JoinHandle<()>,
) {
    let grpc_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let grpc_addr = grpc_listener.local_addr().unwrap();
    drop(grpc_listener);

    let bridge_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bridge_addr = bridge_listener.local_addr().unwrap();

    let service = SutraGrpcService::new(data_dir);
    let server = SutraServiceServer::new(service.clone());
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let mut grpc_rx = shutdown_rx.clone();
    let grpc_handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(server)
            .serve_with_shutdown(grpc_addr, async move {
                while !*grpc_rx.borrow() {
                    if grpc_rx.changed().await.is_err() {
                        break;
                    }
                }
            })
            .await
            .unwrap();
    });

    let bridge_handle = tokio::spawn(async move {
        http_bridge::serve_http_bridge(bridge_listener, service, shutdown_rx)
            .await
            .unwrap();
    });

    sleep(Duration::from_millis(80)).await;
    (grpc_addr, bridge_addr, shutdown_tx, grpc_handle, bridge_handle)
}

async fn spawn_test_server() -> (SocketAddr, tokio::task::JoinHandle<()>, TempDir) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
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
    (addr, handle, tmp)
}

async fn raw_http_request(
    stream: &mut TcpStream,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    extra_headers: &[(&str, &str)],
) -> (u16, String, Vec<u8>) {
    let mut req = format!("{} {} HTTP/1.1\r\nHost: 127.0.0.1\r\n", method, path);
    for (k, v) in extra_headers {
        req.push_str(&format!("{}: {}\r\n", k, v));
    }
    if let Some(b) = body {
        if !extra_headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("Content-Length"))
        {
            req.push_str(&format!("Content-Length: {}\r\n", b.len()));
        }
        if !extra_headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("Content-Type"))
        {
            req.push_str("Content-Type: application/json\r\n");
        }
    }
    req.push_str("\r\n");
    stream.write_all(req.as_bytes()).await.unwrap();
    if let Some(b) = body {
        stream.write_all(b).await.unwrap();
    }
    stream.flush().await.unwrap();

    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos;
        }
        let n = stream.read(&mut tmp).await.unwrap();
        assert!(n > 0, "Unexpected EOF reading HTTP response headers");
        buf.extend_from_slice(&tmp[..n]);
    };

    let head_str = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head_str.split("\r\n");
    let status_line = lines.next().unwrap();
    let status_code: u16 = status_line.split_whitespace().nth(1).unwrap().parse().unwrap();

    let mut content_length: usize = 0;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("Content-Length") {
                content_length = v.trim().parse().unwrap();
            }
        }
    }

    let mut resp_body = buf[header_end + 4..].to_vec();
    while resp_body.len() < content_length {
        let n = stream.read(&mut tmp).await.unwrap();
        assert!(n > 0, "Unexpected EOF reading HTTP response body");
        resp_body.extend_from_slice(&tmp[..n]);
    }
    resp_body.truncate(content_length);

    (status_code, head_str, resp_body)
}

async fn http_json(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<serde_json::Value>,
) -> (u16, serde_json::Value) {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let body_bytes = body.map(|v| serde_json::to_vec(&v).unwrap());
    let (code, _head, raw) = raw_http_request(
        &mut stream,
        method,
        path,
        body_bytes.as_deref(),
        &[("Connection", "close")],
    )
    .await;
    let json = if raw.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&raw).unwrap_or_else(|e| {
            panic!(
                "Invalid JSON response (status {}): {}: {:?}",
                code,
                e,
                String::from_utf8_lossy(&raw)
            )
        })
    };
    (code, json)
}

#[tokio::test]
async fn test_grpc_server_full_lifecycle() {
    let (addr, _handle, _tmp) = spawn_test_server().await;
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
            vector: vec![
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
                0.0,
                0.0,
            ],
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

#[tokio::test]
async fn test_grpc_startup_recovery_and_corrupt_tolerance() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();

    // Instance 1: create saved_col (with post-save WAL delta), wal_only_col, and empty_col
    {
        let (grpc_addr, _, tx, g_handle, b_handle) =
            spawn_test_server_with_dir(Some(dir.clone())).await;
        let mut client = SutraServiceClient::connect(format!("http://{}", grpc_addr))
            .await
            .unwrap();

        client
            .create_collection(CreateCollectionRequest {
                collection_name: "saved_col".into(),
                dimension: 3,
                metric: "euclidean".into(),
                data_dir: None,
            })
            .await
            .unwrap();
        client
            .insert(InsertRequest {
                collection_name: "saved_col".into(),
                documents: vec![Document {
                    id: "s1".into(),
                    vector: vec![1.0, 2.0, 3.0],
                    text: Some("snapshot doc".into()),
                    sparse_vector: None,
                    metadata_json: None,
                    namespace: None,
                    colbert_tokens: None,
                }],
            })
            .await
            .unwrap();

        let save_res = client
            .save(SaveRequest {
                collection_name: "saved_col".into(),
            })
            .await
            .unwrap()
            .into_inner();
        assert!(save_res.success);
        assert!(save_res.bytes_written > 0);
        assert!(save_res.path.ends_with("saved_col.sutra"));
        assert!(dir.join("saved_col.sutra").is_file());
        assert!(dir.join("saved_col.meta.json").is_file());

        // WAL delta after save
        client
            .insert(InsertRequest {
                collection_name: "saved_col".into(),
                documents: vec![Document {
                    id: "s2_delta".into(),
                    vector: vec![4.0, 5.0, 6.0],
                    text: Some("post save wal delta".into()),
                    sparse_vector: None,
                    metadata_json: None,
                    namespace: None,
                    colbert_tokens: None,
                }],
            })
            .await
            .unwrap();

        // WAL-only collection (never saved)
        client
            .create_collection(CreateCollectionRequest {
                collection_name: "wal_only_col".into(),
                dimension: 2,
                metric: "dot_product".into(),
                data_dir: None,
            })
            .await
            .unwrap();
        client
            .insert(InsertRequest {
                collection_name: "wal_only_col".into(),
                documents: vec![Document {
                    id: "w1".into(),
                    vector: vec![2.0, 0.5],
                    text: Some("wal only doc".into()),
                    sparse_vector: None,
                    metadata_json: Some(r#"{"source":"wal"}"#.into()),
                    namespace: None,
                    colbert_tokens: None,
                }],
            })
            .await
            .unwrap();
        assert!(!dir.join("wal_only_col.sutra").exists());
        assert!(dir.join("wal_only_col.meta.json").is_file());
        assert!(dir.join("wal_only_col.wal").is_file());

        // Empty collection (no docs yet)
        client
            .create_collection(CreateCollectionRequest {
                collection_name: "empty_col".into(),
                dimension: 4,
                metric: "cosine".into(),
                data_dir: None,
            })
            .await
            .unwrap();

        let _ = tx.send(true);
        let _ = g_handle.await;
        let _ = b_handle.await;
    }

    // Also plant an orphan WAL (no .meta.json, no .sutra) and a corrupt .sutra file
    {
        let mut orphan_wal = WriteAheadLog::open(dir.join("orphan_wal_col.wal")).unwrap();
        orphan_wal
            .append(&WalRecord {
                op: "insert".into(),
                docs: vec![serde_json::json!({
                    "id": "orp-1",
                    "text": "orphan recovered",
                    "metadata": {},
                    "vector": [0.0, 1.0, 0.0]
                })],
                ids: vec![],
            })
            .unwrap();
        std::fs::write(dir.join("corrupt_col.sutra"), b"NOT_A_VALID_SUTRA_HEADER").unwrap();
    }

    // Instance 2: restart against same directory
    {
        let (grpc_addr, _, tx, g_handle, b_handle) =
            spawn_test_server_with_dir(Some(dir.clone())).await;
        let mut client = SutraServiceClient::connect(format!("http://{}", grpc_addr))
            .await
            .unwrap();

        let listed = client
            .list_collections(ListCollectionsRequest {})
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            listed.collection_names,
            vec!["empty_col", "orphan_wal_col", "saved_col", "wal_only_col"]
        );

        let saved_stats = client
            .get_stats(CollectionStatsRequest {
                collection_name: "saved_col".into(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(saved_stats.count, 2);
        assert_eq!(saved_stats.dimension, 3);
        assert_eq!(saved_stats.metric, "euclidean");
        assert!(saved_stats.wal_records >= 1);

        let wal_only_stats = client
            .get_stats(CollectionStatsRequest {
                collection_name: "wal_only_col".into(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(wal_only_stats.count, 1);
        assert_eq!(wal_only_stats.dimension, 2);
        assert_eq!(wal_only_stats.metric, "dot_product");
        assert!(wal_only_stats.wal_records >= 1);

        let orphan_stats = client
            .get_stats(CollectionStatsRequest {
                collection_name: "orphan_wal_col".into(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(orphan_stats.count, 1);
        assert_eq!(orphan_stats.dimension, 3);

        let empty_stats = client
            .get_stats(CollectionStatsRequest {
                collection_name: "empty_col".into(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(empty_stats.count, 0);
        assert_eq!(empty_stats.dimension, 4);

        let _ = tx.send(true);
        let _ = g_handle.await;
        let _ = b_handle.await;
    }
}

#[tokio::test]
async fn test_grpc_drop_cleans_disk_and_save_compact_wal_records() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let (grpc_addr, _, tx, g_handle, b_handle) =
        spawn_test_server_with_dir(Some(dir.clone())).await;
    let mut client = SutraServiceClient::connect(format!("http://{}", grpc_addr))
        .await
        .unwrap();

    client
        .create_collection(CreateCollectionRequest {
            collection_name: "ghost".into(),
            dimension: 2,
            metric: "cosine".into(),
            data_dir: None,
        })
        .await
        .unwrap();

    client
        .insert(InsertRequest {
            collection_name: "ghost".into(),
            documents: vec![
                Document {
                    id: "d1".into(),
                    vector: vec![1.0, 0.0],
                    text: Some("first".into()),
                    sparse_vector: None,
                    metadata_json: None,
                    namespace: None,
                    colbert_tokens: None,
                },
                Document {
                    id: "d2".into(),
                    vector: vec![0.0, 1.0],
                    text: Some("second".into()),
                    sparse_vector: None,
                    metadata_json: None,
                    namespace: None,
                    colbert_tokens: None,
                },
            ],
        })
        .await
        .unwrap();

    let st_before = client
        .get_stats(CollectionStatsRequest {
            collection_name: "ghost".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(st_before.wal_records, 1);

    client
        .delete(DeleteRequest {
            collection_name: "ghost".into(),
            ids: vec!["d2".into()],
        })
        .await
        .unwrap();

    let st_after_del = client
        .get_stats(CollectionStatsRequest {
            collection_name: "ghost".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(st_after_del.wal_records, 2);

    let compact_res = client
        .compact(CompactRequest {
            collection_name: "ghost".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(compact_res.success);
    assert_eq!(compact_res.count, 1);

    let save_res = client
        .save(SaveRequest {
            collection_name: "ghost".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(save_res.success);
    assert!(save_res.bytes_written > 0);
    assert!(dir.join("ghost.sutra").is_file());

    let st_after_save = client
        .get_stats(CollectionStatsRequest {
            collection_name: "ghost".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(st_after_save.wal_records, 0);

    // Drop must remove all artifacts so recreate starts clean with count == 0
    let drop_res = client
        .drop_collection(DropCollectionRequest {
            collection_name: "ghost".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(drop_res.success);
    assert!(!dir.join("ghost.sutra").exists());
    assert!(!dir.join("ghost.wal").exists());
    assert!(!dir.join("ghost.tmp").exists());
    assert!(!dir.join("ghost.meta.json").exists());
    assert!(!dir.join("ghost.meta.tmp").exists());

    client
        .create_collection(CreateCollectionRequest {
            collection_name: "ghost".into(),
            dimension: 2,
            metric: "cosine".into(),
            data_dir: None,
        })
        .await
        .unwrap();
    let st_recreated = client
        .get_stats(CollectionStatsRequest {
            collection_name: "ghost".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(st_recreated.count, 0);
    assert_eq!(st_recreated.wal_records, 0);

    let _ = tx.send(true);
    let _ = g_handle.await;
    let _ = b_handle.await;
}

#[tokio::test]
async fn test_grpc_name_validation_and_path_confinement() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let (grpc_addr, _, tx, g_handle, b_handle) =
        spawn_test_server_with_dir(Some(dir.clone())).await;
    let mut client = SutraServiceClient::connect(format!("http://{}", grpc_addr))
        .await
        .unwrap();

    let bad_names = [
        "".to_string(),
        "../evil".to_string(),
        "a/b".to_string(),
        "a\\b".to_string(),
        "bad name".to_string(),
        "x".repeat(129),
    ];

    for bad in &bad_names {
        let err = client
            .create_collection(CreateCollectionRequest {
                collection_name: bad.clone(),
                dimension: 2,
                metric: "cosine".into(),
                data_dir: None,
            })
            .await
            .unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument, "create {:?}", bad);

        assert_eq!(
            client
                .drop_collection(DropCollectionRequest {
                    collection_name: bad.clone()
                })
                .await
                .unwrap_err()
                .code(),
            Code::InvalidArgument
        );
        assert_eq!(
            client
                .get_stats(CollectionStatsRequest {
                    collection_name: bad.clone()
                })
                .await
                .unwrap_err()
                .code(),
            Code::InvalidArgument
        );
        assert_eq!(
            client
                .save(SaveRequest {
                    collection_name: bad.clone()
                })
                .await
                .unwrap_err()
                .code(),
            Code::InvalidArgument
        );
        assert_eq!(
            client
                .compact(CompactRequest {
                    collection_name: bad.clone()
                })
                .await
                .unwrap_err()
                .code(),
            Code::InvalidArgument
        );
    }

    // Path traversal / escape in req.data_dir
    for bad_dir in ["../outside", "/etc/sutradb_escape", "sub/../../escape"] {
        let err = client
            .create_collection(CreateCollectionRequest {
                collection_name: "safe_name".into(),
                dimension: 2,
                metric: "cosine".into(),
                data_dir: Some(bad_dir.into()),
            })
            .await
            .unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument, "data_dir {:?}", bad_dir);
    }

    // Valid confined subdir inside default_data_dir works
    let ok_res = client
        .create_collection(CreateCollectionRequest {
            collection_name: "confined_col".into(),
            dimension: 2,
            metric: "cosine".into(),
            data_dir: Some("tenant_a".into()),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(ok_res.success);
    assert!(dir.join("tenant_a").join("confined_col.meta.json").is_file());

    // Custom data_dir when server has default_data_dir == None is rejected
    let stateless_svc = SutraGrpcService::new(None);
    let err = _sutradb::grpc::SutraService::create_collection(
        &stateless_svc,
        tonic::Request::new(CreateCollectionRequest {
            collection_name: "x".into(),
            dimension: 2,
            metric: "cosine".into(),
            data_dir: Some("sub".into()),
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);

    let _ = tx.send(true);
    let _ = g_handle.await;
    let _ = b_handle.await;
}

#[tokio::test]
async fn test_grpc_malformed_inputs_and_concurrent_batch_query() {
    let tmp = tempdir().unwrap();
    let (grpc_addr, _, tx, g_handle, b_handle) =
        spawn_test_server_with_dir(Some(tmp.path().to_path_buf())).await;
    let mut client = SutraServiceClient::connect(format!("http://{}", grpc_addr))
        .await
        .unwrap();

    client
        .create_collection(CreateCollectionRequest {
            collection_name: "guard_col".into(),
            dimension: 3,
            metric: "cosine".into(),
            data_dir: None,
        })
        .await
        .unwrap();

    // Malformed SparseVector on insert -> InvalidArgument (no panic)
    let err_sp_ins = client
        .insert(InsertRequest {
            collection_name: "guard_col".into(),
            documents: vec![Document {
                id: "bad_sp".into(),
                vector: vec![1.0, 0.0, 0.0],
                text: None,
                sparse_vector: Some(SparseVector {
                    indices: vec![1, 2],
                    values: vec![0.5],
                }),
                metadata_json: None,
                namespace: None,
                colbert_tokens: None,
            }],
        })
        .await
        .unwrap_err();
    assert_eq!(err_sp_ins.code(), Code::InvalidArgument);

    // Malformed metadata_json on insert -> InvalidArgument (not swallowed)
    let err_meta = client
        .insert(InsertRequest {
            collection_name: "guard_col".into(),
            documents: vec![Document {
                id: "bad_meta".into(),
                vector: vec![1.0, 0.0, 0.0],
                text: None,
                sparse_vector: None,
                metadata_json: Some("{not-valid-json".into()),
                namespace: None,
                colbert_tokens: None,
            }],
        })
        .await
        .unwrap_err();
    assert_eq!(err_meta.code(), Code::InvalidArgument);

    // Empty doc id -> InvalidArgument
    let err_empty_id = client
        .insert(InsertRequest {
            collection_name: "guard_col".into(),
            documents: vec![Document {
                id: "   ".into(),
                vector: vec![1.0, 0.0, 0.0],
                text: None,
                sparse_vector: None,
                metadata_json: None,
                namespace: None,
                colbert_tokens: None,
            }],
        })
        .await
        .unwrap_err();
    assert_eq!(err_empty_id.code(), Code::InvalidArgument);

    // Dimension mismatch on insert -> InvalidArgument
    let err_dim = client
        .insert(InsertRequest {
            collection_name: "guard_col".into(),
            documents: vec![Document {
                id: "bad_dim".into(),
                vector: vec![1.0, 0.0],
                text: None,
                sparse_vector: None,
                metadata_json: None,
                namespace: None,
                colbert_tokens: None,
            }],
        })
        .await
        .unwrap_err();
    assert_eq!(err_dim.code(), Code::InvalidArgument);

    // Malformed SparseVector on query -> InvalidArgument
    let err_sp_q = client
        .query(QueryRequest {
            collection_name: "guard_col".into(),
            vector: vec![1.0, 0.0, 0.0],
            text: None,
            sparse_vector: Some(SparseVector {
                indices: vec![1],
                values: vec![0.1, 0.2],
            }),
            namespace: None,
            filter_json: None,
            top_k: 5,
            hybrid: false,
            dense_weight: 1.0,
            bm25_weight: 0.0,
            sparse_weight: 0.0,
            use_ann: false,
            colbert_tokens: None,
            colbert_weight: 1.0,
        })
        .await
        .unwrap_err();
    assert_eq!(err_sp_q.code(), Code::InvalidArgument);

    // Insert 3 orthogonal vectors and verify concurrent batch_query order + empty collection_name fallback
    client
        .insert(InsertRequest {
            collection_name: "guard_col".into(),
            documents: vec![
                Document {
                    id: "x".into(),
                    vector: vec![1.0, 0.0, 0.0],
                    text: None,
                    sparse_vector: None,
                    metadata_json: None,
                    namespace: None,
                    colbert_tokens: None,
                },
                Document {
                    id: "y".into(),
                    vector: vec![0.0, 1.0, 0.0],
                    text: None,
                    sparse_vector: None,
                    metadata_json: None,
                    namespace: None,
                    colbert_tokens: None,
                },
                Document {
                    id: "z".into(),
                    vector: vec![0.0, 0.0, 1.0],
                    text: None,
                    sparse_vector: None,
                    metadata_json: None,
                    namespace: None,
                    colbert_tokens: None,
                },
            ],
        })
        .await
        .unwrap();

    let make_q = |v: Vec<f32>| QueryRequest {
        collection_name: String::new(), // empty -> must fallback to BatchQueryRequest.collection_name
        vector: v,
        text: None,
        sparse_vector: None,
        namespace: None,
        filter_json: None,
        top_k: 1,
        hybrid: false,
        dense_weight: 1.0,
        bm25_weight: 0.0,
        sparse_weight: 0.0,
        use_ann: false,
        colbert_tokens: None,
        colbert_weight: 1.0,
    };

    let batch_res = client
        .batch_query(BatchQueryRequest {
            collection_name: "guard_col".into(),
            queries: vec![
                make_q(vec![0.0, 0.0, 1.0]),
                make_q(vec![1.0, 0.0, 0.0]),
                make_q(vec![0.0, 1.0, 0.0]),
            ],
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(batch_res.responses.len(), 3);
    assert_eq!(batch_res.responses[0].results[0].id, "z");
    assert_eq!(batch_res.responses[1].results[0].id, "x");
    assert_eq!(batch_res.responses[2].results[0].id, "y");

    let _ = tx.send(true);
    let _ = g_handle.await;
    let _ = b_handle.await;
}

#[tokio::test]
async fn test_http_bridge_endpoints_keepalive_chunked_and_grpc_parity() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let (grpc_addr, bridge_addr, tx, g_handle, b_handle) =
        spawn_test_server_with_dir(Some(dir.clone())).await;
    let mut client = SutraServiceClient::connect(format!("http://{}", grpc_addr))
        .await
        .unwrap();

    // 1. Keep-alive reuse across GET /health and GET /
    {
        let mut ka_stream = TcpStream::connect(bridge_addr).await.unwrap();
        let (code1, _h1, b1) = raw_http_request(&mut ka_stream, "GET", "/health", None, &[]).await;
        assert_eq!(code1, 200);
        let j1: serde_json::Value = serde_json::from_slice(&b1).unwrap();
        assert_eq!(j1["status"], "healthy");
        assert_eq!(j1["version"], "5.1.0");

        let (code2, _h2, b2) =
            raw_http_request(&mut ka_stream, "GET", "/", None, &[("Connection", "close")]).await;
        assert_eq!(code2, 200);
        let j2: serde_json::Value = serde_json::from_slice(&b2).unwrap();
        assert_eq!(j2["status"], "healthy");
    }

    // 2. Create via gRPC -> visible via HTTP /collections and /stats
    client
        .create_collection(CreateCollectionRequest {
            collection_name: "parity_col".into(),
            dimension: 4,
            metric: "cosine".into(),
            data_dir: None,
        })
        .await
        .unwrap();

    let (code, cols_json) = http_json(bridge_addr, "GET", "/collections", None).await;
    assert_eq!(code, 200);
    assert_eq!(cols_json["collections"][0]["name"], "parity_col");

    // 3. Duplicate POST /collections -> 409; invalid name -> 400
    let (dup_code, _) = http_json(
        bridge_addr,
        "POST",
        "/collections",
        Some(serde_json::json!({
            "name": "parity_col",
            "dimension": 4,
            "metric": "cosine"
        })),
    )
    .await;
    assert_eq!(dup_code, 409);

    let (bad_name_code, _) = http_json(
        bridge_addr,
        "POST",
        "/collections",
        Some(serde_json::json!({
            "name": "../bad",
            "dimension": 4
        })),
    )
    .await;
    assert_eq!(bad_name_code, 400);

    // 4. Insert via gRPC (with {indices, values} sparse) AND via HTTP chunked ({dim: weight} sparse + >100-char UTF-8)
    client
        .insert(InsertRequest {
            collection_name: "parity_col".into(),
            documents: vec![Document {
                id: "grpc-doc".into(),
                vector: vec![1.0, 0.0, 0.0, 0.0],
                text: Some("created over grpc".into()),
                sparse_vector: Some(SparseVector {
                    indices: vec![1, 3],
                    values: vec![0.8, 0.4],
                }),
                metadata_json: Some(r#"{"origin":"grpc"}"#.into()),
                namespace: None,
                colbert_tokens: None,
            }],
        })
        .await
        .unwrap();

    let long_utf8 = "🚀".repeat(120);
    let http_doc_payload = serde_json::json!({
        "documents": [{
            "id": "http-doc",
            "vector": [0.0, 1.0, 0.0, 0.0],
            "text": long_utf8,
            "metadata": {"origin": "http", "rank": 2},
            "sparse_vector": {"2": 0.9, "5": 0.3}
        }]
    });
    let chunk_bytes = serde_json::to_vec(&http_doc_payload).unwrap();
    {
        let mut s = TcpStream::connect(bridge_addr).await.unwrap();
        let req = format!(
            "POST /collections/parity_col/insert HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nContent-Type: application/json\r\n\r\n{:x}\r\n",
            chunk_bytes.len()
        );
        s.write_all(req.as_bytes()).await.unwrap();
        s.write_all(&chunk_bytes).await.unwrap();
        s.write_all(b"\r\n0\r\n\r\n").await.unwrap();
        s.flush().await.unwrap();
        let mut resp = String::new();
        s.read_to_string(&mut resp).await.unwrap();
        assert!(resp.starts_with("HTTP/1.1 200 OK"), "chunked insert failed: {}", resp);
    }

    // 5. Check /documents pagination, /documents/{id}, /sample UTF-8 safety, /metrics
    let (code, doc_list) = http_json(
        bridge_addr,
        "GET",
        "/collections/parity_col/documents?limit=1&offset=1",
        None,
    )
    .await;
    assert_eq!(code, 200);
    assert_eq!(doc_list["total"], 2);
    assert_eq!(doc_list["documents"].as_array().unwrap().len(), 1);
    assert!(doc_list["documents"][0].get("vector").is_none());

    let (code, single_doc) =
        http_json(bridge_addr, "GET", "/collections/parity_col/documents/http-doc", None).await;
    assert_eq!(code, 200);
    assert_eq!(single_doc["vector"].as_array().unwrap().len(), 4);

    let (code, sample_json) =
        http_json(bridge_addr, "GET", "/collections/parity_col/sample?limit=10", None).await;
    assert_eq!(code, 200);
    assert_eq!(sample_json["count"], 2);
    let sampled_http = sample_json["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["id"] == "http-doc")
        .unwrap();
    assert_eq!(sampled_http["text"].as_str().unwrap().chars().count(), 100);

    let (alias_code, alias_sample) = http_json(
        bridge_addr,
        "GET",
        "/collections/parity_col/sample_vectors?limit=1",
        None,
    )
    .await;
    assert_eq!(alias_code, 200);
    assert_eq!(alias_sample["count"], 1);

    // Malformed sparse_vector and dim mismatch on HTTP bridge return 400 without crashing
    assert_eq!(
        http_json(
            bridge_addr,
            "POST",
            "/collections/parity_col/documents",
            Some(serde_json::json!({
                "documents": [{"id": "bad_dim", "vector": [1.0, 0.0]}]
            }))
        )
        .await
        .0,
        400
    );
    assert_eq!(
        http_json(
            bridge_addr,
            "POST",
            "/collections/parity_col/insert",
            Some(serde_json::json!({
                "documents": [{
                    "id": "bad_sp",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "sparse_vector": {"indices": [1, 2], "values": [1.0]}
                }]
            }))
        )
        .await
        .0,
        400
    );
    assert_eq!(
        http_json(bridge_addr, "GET", "/collections/does_not_exist/stats", None)
            .await
            .0,
        404
    );

    {
        let mut s = TcpStream::connect(bridge_addr).await.unwrap();
        let (code, _, raw) = raw_http_request(
            &mut s,
            "GET",
            "/metrics",
            None,
            &[("Connection", "close")],
        )
        .await;
        assert_eq!(code, 200);
        let metrics_text = String::from_utf8_lossy(&raw);
        assert!(metrics_text.contains("sutradb_collections_total 1"));
        assert!(metrics_text.contains("sutradb_documents_count{collection=\"parity_col\"} 2"));
    }

    // 6. PATCH metadata via HTTP -> immediately filterable via gRPC query!
    let (code, _) = http_json(
        bridge_addr,
        "PATCH",
        "/collections/parity_col/documents/http-doc/metadata",
        Some(serde_json::json!({
            "metadata": {"origin": "http_patched", "verified": true}
        })),
    )
    .await;
    assert_eq!(code, 200);

    let grpc_filtered = client
        .query(QueryRequest {
            collection_name: "parity_col".into(),
            vector: vec![0.0, 1.0, 0.0, 0.0],
            text: None,
            sparse_vector: None,
            namespace: None,
            filter_json: Some(r#"{"origin":{"$eq":"http_patched"}}"#.into()),
            top_k: 5,
            hybrid: false,
            dense_weight: 1.0,
            bm25_weight: 0.0,
            sparse_weight: 0.0,
            use_ann: false,
            colbert_tokens: None,
            colbert_weight: 1.0,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(grpc_filtered.results.len(), 1);
    assert_eq!(grpc_filtered.results[0].id, "http-doc");

    // 7. Upsert, query, batch_query, hnsw, sq8, bq, pq/train, pq, compact, save
    let (code, _) = http_json(
        bridge_addr,
        "POST",
        "/collections/parity_col/upsert",
        Some(serde_json::json!({
            "documents": [{
                "id": "http-doc-2",
                "vector": [0.0, 0.0, 1.0, 0.0],
                "text": "third document for quantization",
                "metadata": {"origin": "upsert"}
            }]
        })),
    )
    .await;
    assert_eq!(code, 200);

    let (code, q_resp) = http_json(
        bridge_addr,
        "POST",
        "/collections/parity_col/query",
        Some(serde_json::json!({
            "vector": [1.0, 0.0, 0.0, 0.0],
            "top_k": 2
        })),
    )
    .await;
    assert_eq!(code, 200);
    assert_eq!(q_resp["results"][0]["id"], "grpc-doc");

    let (code, bq_resp) = http_json(
        bridge_addr,
        "POST",
        "/collections/parity_col/batch_query",
        Some(serde_json::json!({
            "queries": [
                {"vector": [0.0, 0.0, 1.0, 0.0], "top_k": 1},
                {"vector": [1.0, 0.0, 0.0, 0.0], "top_k": 1}
            ]
        })),
    )
    .await;
    assert_eq!(code, 200);
    assert_eq!(bq_resp["responses"][0]["results"][0]["id"], "http-doc-2");
    assert_eq!(bq_resp["responses"][1]["results"][0]["id"], "grpc-doc");

    assert_eq!(
        http_json(bridge_addr, "POST", "/collections/parity_col/hnsw", None)
            .await
            .0,
        200
    );
    assert_eq!(
        http_json(bridge_addr, "POST", "/collections/parity_col/sq8", None)
            .await
            .0,
        200
    );
    assert_eq!(
        http_json(bridge_addr, "POST", "/collections/parity_col/bq", None)
            .await
            .0,
        200
    );
    assert_eq!(
        http_json(
            bridge_addr,
            "POST",
            "/collections/parity_col/pq/train",
            Some(serde_json::json!({"m": 2}))
        )
        .await
        .0,
        200
    );
    assert_eq!(
        http_json(bridge_addr, "POST", "/collections/parity_col/pq", None)
            .await
            .0,
        200
    );

    // Delete single doc via HTTP, compact, save -> verify wal_records == 0 in BOTH HTTP and gRPC
    assert_eq!(
        http_json(
            bridge_addr,
            "DELETE",
            "/collections/parity_col/documents/grpc-doc",
            None
        )
        .await
        .0,
        200
    );
    assert_eq!(
        http_json(
            bridge_addr,
            "DELETE",
            "/collections/parity_col/documents/grpc-doc",
            None
        )
        .await
        .0,
        404
    );
    assert_eq!(
        http_json(bridge_addr, "POST", "/collections/parity_col/compact", None)
            .await
            .0,
        200
    );
    let (save_code, save_json) =
        http_json(bridge_addr, "POST", "/collections/parity_col/save", None).await;
    assert_eq!(save_code, 200);
    assert!(save_json["bytes"].as_u64().unwrap() > 0);
    assert!(dir.join("parity_col.sutra").is_file());

    let (st_code, st_json) =
        http_json(bridge_addr, "GET", "/collections/parity_col/stats", None).await;
    assert_eq!(st_code, 200);
    assert_eq!(st_json["wal_records"], 0);
    assert_eq!(st_json["has_hnsw"], true);
    assert_eq!(st_json["quantized_sq8"], true);
    assert_eq!(st_json["quantized_bq"], true);
    assert_eq!(st_json["quantized_pq"], true);

    let grpc_st = client
        .get_stats(CollectionStatsRequest {
            collection_name: "parity_col".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(grpc_st.wal_records, 0);
    assert_eq!(grpc_st.count, 2);

    // 8. 413 Payload Too Large enforcement
    {
        let mut s = TcpStream::connect(bridge_addr).await.unwrap();
        let (code, _, _) = raw_http_request(
            &mut s,
            "POST",
            "/collections/parity_col/insert",
            None,
            &[
                ("Content-Length", "10485761"),
                ("Connection", "close"),
            ],
        )
        .await;
        assert_eq!(code, 413);
    }

    // Batch delete via HTTP DELETE /collections/{name}/documents
    let (bdel_code, bdel_json) = http_json(
        bridge_addr,
        "DELETE",
        "/collections/parity_col/documents",
        Some(serde_json::json!({"ids": ["http-doc-2"]})),
    )
    .await;
    assert_eq!(bdel_code, 200);
    assert_eq!(bdel_json["deleted"], 1);

    // 9. Drop via HTTP -> gone in gRPC and disk cleaned
    let (drop_code, _) =
        http_json(bridge_addr, "DELETE", "/collections/parity_col", None).await;
    assert_eq!(drop_code, 200);
    assert!(!dir.join("parity_col.sutra").exists());
    assert!(!dir.join("parity_col.wal").exists());
    assert!(!dir.join("parity_col.meta.json").exists());

    let listed_after = client
        .list_collections(ListCollectionsRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(listed_after.collection_names.is_empty());

    let _ = tx.send(true);
    let _ = g_handle.await;
    let _ = b_handle.await;
}

#[tokio::test]
async fn test_grpc_batch_distances_unnormalized_cosine() {
    let (addr, _handle, _tmp) = spawn_test_server().await;
    let mut client = SutraServiceClient::connect(format!("http://{}", addr))
        .await
        .expect("Failed to connect to test server");

    let resp = client
        .batch_distances(BatchDistanceRequest {
            query_vector: vec![3.0, 0.0, 0.0, 0.0],
            target_vectors: vec![
                2.0, 0.0, 0.0, 0.0,
                0.0, 5.0, 0.0, 0.0,
                3.0, 4.0, 0.0, 0.0,
            ],
            dimension: 4,
            metric: "cosine".to_string(),
        })
        .await
        .expect("BatchDistances failed")
        .into_inner();

    assert_eq!(resp.scores.len(), 3);
    assert!(
        (resp.scores[0] - 1.0).abs() < 1e-4,
        "Expected cosine 1.0 for collinear unnormalized target, got {}",
        resp.scores[0]
    );
    assert!(resp.scores[1].abs() < 1e-4);
    assert!((resp.scores[2] - 0.6).abs() < 1e-4);
}
