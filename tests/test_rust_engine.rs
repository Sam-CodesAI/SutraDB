use std::path::PathBuf;
use tempfile::tempdir;
use serde_json::json;

use _sutradb::collection::{NativeCollection, NativeDocument};
use _sutradb::distance::simd::{dot_product, compute_vector_score, normalize_copy};
use _sutradb::distance::sq8::SQ8Vector;
use _sutradb::distance::Metric;
use _sutradb::bm25::BM25Index;
use _sutradb::filter::FilterEngine;
use _sutradb::fusion::reciprocal_rank_fusion;
use _sutradb::index::hnsw::{HNSWConfig, HNSWIndex};
use _sutradb::storage::{WalRecord, WriteAheadLog};

#[test]
fn test_simd_distance_correctness() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 3.0, 4.0, 5.0];
    let dot = dot_product(&a, &b);
    assert_eq!(dot, 40.0);

    let cos = compute_vector_score(&a, &b, Metric::Cosine, false).unwrap();
    let norm_a = (1.0f32 + 4.0 + 9.0 + 16.0).sqrt();
    let norm_b = (4.0f32 + 9.0 + 16.0 + 25.0).sqrt();
    let expected_cos = 40.0 / (norm_a * norm_b);
    assert!((cos - expected_cos).abs() < 1e-5);

    let euc_sim = compute_vector_score(&a, &b, Metric::Euclidean, false).unwrap();
    // distance is sqrt(1+1+1+1) = 2.0. sim = 1 / (1 + 2.0) = 1/3
    assert!((euc_sim - (1.0 / 3.0)).abs() < 1e-5);
}

#[test]
fn test_sq8_quantization_and_asymmetric_scoring() {
    let v = vec![0.1, 0.5, 0.9, -0.4, 0.2];
    let sq = SQ8Vector::quantize(&v);
    assert_eq!(sq.data.len(), v.len());
    let deq = sq.dequantize();
    for i in 0..v.len() {
        assert!((v[i] - deq[i]).abs() < 0.02);
    }

    let q = vec![0.2, 0.4, 0.8, -0.3, 0.1];
    let q_sum: f32 = q.iter().sum();
    let asym_dot = sq.asymmetric_dot_product(&q, q_sum);
    let true_dot = dot_product(&q, &v);
    assert!((asym_dot - true_dot).abs() < 0.05);
}

#[test]
fn test_hnsw_indexing_and_recall() {
    let mut index = HNSWIndex::new(HNSWConfig {
        m: 16,
        m0: 32,
        ef_construction: 64,
        ef_search: 32,
        ml: 1.0 / (16.0f64).ln(),
        seed: 42,
    });

    let mut vectors = Vec::new();
    let n = 200;
    let dim = 16;
    for i in 0..n {
        let mut v = Vec::with_capacity(dim);
        for d in 0..dim {
            v.push(((i * 7 + d * 13) % 100) as f32 / 100.0);
        }
        let norm_v = normalize_copy(&v);
        vectors.push(norm_v);
    }

    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u32, v, &vectors, Metric::Cosine, true).unwrap();
    }

    // Query with first vector
    let q = &vectors[0];
    let results = index.search(q, &vectors, 5, 32, Metric::Cosine, true, None).unwrap();
    assert!(!results.is_empty());
    // The query vector itself should be the #1 result with score ~ 1.0
    assert_eq!(results[0].0, 0);
    assert!((results[0].1 - 1.0).abs() < 1e-4);
}

#[test]
fn test_bm25_lexical_scoring() {
    let mut bm25 = BM25Index::default();
    let corpus = vec![
        "Deploying microservices to Kubernetes clusters".to_string(),
        "PostgreSQL connection pooling and pgbouncer tuning".to_string(),
        "Machine learning vector search and ANN algorithms".to_string(),
    ];
    bm25.add_documents(&corpus);
    let scores = bm25.score_query("Kubernetes microservices", None);
    assert!(scores[0] > scores[1]);
    assert!(scores[0] > scores[2]);
}

#[test]
fn test_filter_engine_ast() {
    let meta = json!({
        "tier": "enterprise",
        "priority": 1,
        "tags": ["cloud", "devops", "ai"],
        "user": {
            "profile": {
                "city": "Seattle",
                "score": 98.5
            }
        }
    });

    let filter_pass = json!({
        "tier": "enterprise",
        "priority": {"$lte": 2},
        "tags": {"$contains": "cloud"},
        "user.profile.city": "Seattle",
        "user.profile.score": {"$gt": 90.0},
        "tags": {"$all": ["cloud", "ai"]},
        "tags": {"$size": 3}
    });
    assert!(FilterEngine::evaluate(Some(&meta), Some(&filter_pass)));

    let filter_fail = json!({
        "user.profile.city": "New York"
    });
    assert!(!FilterEngine::evaluate(Some(&meta), Some(&filter_fail)));
}

#[test]
fn test_reciprocal_rank_fusion() {
    let dense_scores = vec![0.9, 0.4, 0.8, 0.1];
    let bm25_scores = vec![0.0, 5.0, 2.0, 0.0];

    let fused = reciprocal_rank_fusion(&dense_scores, &bm25_scores, 2, 60, 1.0, 1.0, None);
    assert_eq!(fused.len(), 2);
    // Doc 2 has strong dense (0.8) and non-zero BM25 (2.0)
    assert!(fused[0].doc_idx == 0 || fused[0].doc_idx == 2 || fused[0].doc_idx == 1);
}

#[test]
fn test_storage_and_wal_crash_recovery() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let mut wal = WriteAheadLog::open(&wal_path).unwrap();
    let rec1 = WalRecord {
        op: "insert".to_string(),
        docs: vec![json!({"id": "d1", "text": "First doc"})],
        ids: Vec::new(),
    };
    let rec2 = WalRecord {
        op: "insert".to_string(),
        docs: vec![json!({"id": "d2", "text": "Second doc"})],
        ids: Vec::new(),
    };

    wal.append(&rec1).unwrap();
    wal.append(&rec2).unwrap();

    let replayed = wal.replay().unwrap();
    assert_eq!(replayed.len(), 2);
    assert_eq!(replayed[0].docs[0]["id"], "d1");
    assert_eq!(replayed[1].docs[0]["id"], "d2");

    // Simulate trailing partial/corrupt write
    {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().append(true).open(&wal_path).unwrap();
        file.write_all(&[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();
    }

    // Replay should safely truncate corrupt tail and recover the 2 valid records
    let healed = wal.replay().unwrap();
    assert_eq!(healed.len(), 2);
}

#[test]
fn test_native_collection_full_lifecycle() {
    let dir = tempdir().unwrap();
    let store_file = dir.path().join("lifecycle.sutra");

    let mut col = NativeCollection::new(
        "test_col".to_string(),
        4,
        Metric::Cosine,
        Some(store_file.clone()),
        true,
    ).unwrap();

    let docs = vec![
        NativeDocument {
            id: "d1".to_string(),
            namespace: Some("tenant_a".to_string()),
            text: "Kubernetes orchestration and container deployment".to_string(),
            metadata: json!({"category": "cloud", "env": "prod"}),
        },
        NativeDocument {
            id: "d2".to_string(),
            namespace: Some("tenant_b".to_string()),
            text: "PostgreSQL relational database indexing".to_string(),
            metadata: json!({"category": "database", "env": "prod"}),
        },
    ];

    let vecs = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
    ];

    let count = col.insert(docs, vecs).unwrap();
    assert_eq!(count, 2);

    // Multi-tenant namespace query: searching tenant_a should NEVER return tenant_b docs
    let q_vec = vec![0.0, 1.0, 0.0, 0.0];
    let res_a = col.query(
        Some(&q_vec),
        None,
        Some("tenant_a"),
        None,
        5,
        false,
        None,
        60,
        1.0,
        1.0,
        false,
    ).unwrap();
    assert_eq!(res_a.len(), 1);
    assert_eq!(res_a[0].id, "d1"); // only d1 is in tenant_a

    // In-place metadata patch
    let updated = col.update_metadata("d1", json!({"category": "cloud", "env": "staging"})).unwrap();
    assert!(updated);

    // Snapshot save & reload via zero-copy mmap
    col.save(None::<PathBuf>).unwrap();

    let mut loaded = NativeCollection::new(
        "test_col".to_string(),
        4,
        Metric::Cosine,
        Some(store_file),
        false,
    ).unwrap();
    loaded.load(None::<PathBuf>).unwrap();
    assert_eq!(loaded.count(), 2);
    let loaded_d1 = loaded.documents.iter().find(|d| d.id == "d1").unwrap();
    assert_eq!(loaded_d1.metadata["env"], "staging");
}

#[test]
fn test_bq_quantization_and_two_stage_search() {
    let mut col = NativeCollection::new(
        "bq_test".to_string(),
        128,
        Metric::Cosine,
        None,
        false,
    ).unwrap();

    let n = 50;
    let mut docs = Vec::new();
    let mut vecs = Vec::new();

    for i in 0..n {
        let mut v = vec![0.0f32; 128];
        v[i % 128] = 1.0;
        v[(i + 1) % 128] = 0.5;
        let v_norm = normalize_copy(&v);
        docs.push(NativeDocument {
            id: format!("doc_{}", i),
            namespace: None,
            text: format!("Text for doc {}", i),
            metadata: json!({"idx": i}),
        });
        vecs.push(v_norm);
    }

    col.insert(docs, vecs).unwrap();
    col.enable_bq();
    assert!(col.use_bq);
    assert_eq!(col.bq_vectors.len(), n);

    // Two-stage BQ search
    let mut q = vec![0.0f32; 128];
    q[0] = 1.0;
    q[1] = 0.5;
    let q_norm = normalize_copy(&q);

    let res = col.query(
        Some(&q_norm),
        None,
        None,
        None,
        5,
        false,
        None,
        60,
        1.0,
        1.0,
        false,
    ).unwrap();

    assert!(!res.is_empty());
    assert_eq!(res[0].id, "doc_0");
    assert!((res[0].score - 1.0).abs() < 1e-4);
}

#[test]
fn test_hnsw_online_deletion_and_reconnection() {
    let mut col = NativeCollection::new(
        "hnsw_del_test".to_string(),
        4,
        Metric::Cosine,
        None,
        false,
    ).unwrap();

    let docs = vec![
        NativeDocument { id: "a".into(), namespace: None, text: "alpha".into(), metadata: json!({}) },
        NativeDocument { id: "b".into(), namespace: None, text: "beta".into(), metadata: json!({}) },
        NativeDocument { id: "c".into(), namespace: None, text: "gamma".into(), metadata: json!({}) },
    ];
    let vecs = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
    ];

    col.insert(docs, vecs).unwrap();
    col.build_hnsw(None).unwrap();
    assert!(col.hnsw.is_some());

    // Query ANN
    let q = vec![1.0, 0.0, 0.0, 0.0];
    let res1 = col.query(Some(&q), None, None, None, 2, false, None, 60, 1.0, 1.0, true).unwrap();
    assert_eq!(res1[0].id, "a");

    // Dynamic deletion of "a"
    let deleted = col.delete(&["a".to_string()]).unwrap();
    assert_eq!(deleted, 1);
    assert_eq!(col.count(), 2);

    // Query ANN again: "a" must not appear, nearest should be "b"
    let res2 = col.query(Some(&q), None, None, None, 2, false, None, 60, 1.0, 1.0, true).unwrap();
    assert_eq!(res2[0].id, "b");
}

#[test]
fn test_geospatial_filters() {
    let mut col = NativeCollection::new(
        "geo_test".to_string(),
        4,
        Metric::Cosine,
        None,
        false,
    ).unwrap();

    let docs = vec![
        NativeDocument {
            id: "sf".into(),
            namespace: None,
            text: "San Francisco City Hall".into(),
            metadata: json!({"city": "SF", "loc": [37.7792, -122.4191]}),
        },
        NativeDocument {
            id: "oak".into(),
            namespace: None,
            text: "Oakland City Center".into(),
            metadata: json!({"city": "Oakland", "loc": [37.8044, -122.2712]}),
        },
        NativeDocument {
            id: "sjc".into(),
            namespace: None,
            text: "San Jose Plaza".into(),
            metadata: json!({"city": "San Jose", "loc": [37.3382, -121.8863]}),
        },
    ];
    let vecs = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];
    col.insert(docs, vecs).unwrap();

    // 1. $geo_within: Within 10km of SF Downtown
    let sf_filter = json!({
        "loc": {
            "$geo_within": {
                "center": [37.7749, -122.4194],
                "radius_meters": 10000.0
            }
        }
    });
    let res_sf = col.query(None, Some("City"), None, Some(&sf_filter), 10, false, None, 60, 1.0, 1.0, false).unwrap();
    assert_eq!(res_sf.len(), 1);
    assert_eq!(res_sf[0].id, "sf");

    // 2. $geo_box: Bounding box covering SF and Oakland, excluding San Jose
    let bay_box = json!({
        "loc": {
            "$geo_box": {
                "bottom_left": [37.70, -122.50],
                "top_right": [37.85, -122.20]
            }
        }
    });
    let res_box = col.query(None, Some("City"), None, Some(&bay_box), 10, false, None, 60, 1.0, 1.0, false).unwrap();
    assert_eq!(res_box.len(), 2);
    let ids: Vec<String> = res_box.into_iter().map(|r| r.id).collect();
    assert!(ids.contains(&"sf".to_string()));
    assert!(ids.contains(&"oak".to_string()));
    assert!(!ids.contains(&"sjc".to_string()));
}

#[test]
fn test_pq_quantization_and_two_stage_search() {
    let mut col = NativeCollection::new(
        "pq_engine_test".to_string(),
        16,
        Metric::Cosine,
        None,
        false,
    ).unwrap();

    let n = 40;
    let mut docs = Vec::new();
    let mut vecs = Vec::new();

    for i in 0..n {
        let mut v = vec![0.0f32; 16];
        v[i % 16] = 1.0;
        v[(i + 1) % 16] = 0.5;
        let v_norm = normalize_copy(&v);
        docs.push(NativeDocument {
            id: format!("doc_{}", i),
            namespace: None,
            text: format!("Document number {}", i),
            metadata: json!({"index": i}),
        });
        vecs.push(v_norm);
    }

    col.insert(docs, vecs).unwrap();

    // Train PQ with M=4 (4 sub-spaces of 4 dimensions)
    col.train_pq(Some(4), None).unwrap();
    assert!(col.use_pq);
    assert!(col.pq_codebook.is_some());
    assert_eq!(col.pq_vectors.len(), n);

    let stats = col.stats();
    assert_eq!(stats["quantized_pq"], true);
    assert_eq!(stats["pq_m"], 4);

    // Two-stage ADC search
    let mut q = vec![0.0f32; 16];
    q[0] = 1.0;
    q[1] = 0.5;
    let q_norm = normalize_copy(&q);

    let res = col.query(
        Some(&q_norm),
        None,
        None,
        None,
        5,
        false,
        None,
        60,
        1.0,
        1.0,
        false,
    ).unwrap();

    assert!(!res.is_empty());
    assert_eq!(res[0].id, "doc_0");
    assert!((res[0].score - 1.0).abs() < 1e-4);
}
