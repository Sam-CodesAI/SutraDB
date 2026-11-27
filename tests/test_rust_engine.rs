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
            sparse_vector: None, colbert_tokens: None,
        },
        NativeDocument {
            id: "d2".to_string(),
            namespace: Some("tenant_b".to_string()),
            text: "PostgreSQL relational database indexing".to_string(),
            metadata: json!({"category": "database", "env": "prod"}),
            sparse_vector: None, colbert_tokens: None,
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
        None,
        None,
        Some("tenant_a"),
        None,
        5,
        false,
        None,
        60,
        1.0,
        1.0,
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
            sparse_vector: None, colbert_tokens: None,
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
        None,
        None,
        5,
        false,
        None,
        60,
        1.0,
        1.0,
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
        NativeDocument { id: "a".into(), namespace: None, text: "alpha".into(), metadata: json!({}), sparse_vector: None, colbert_tokens: None },
        NativeDocument { id: "b".into(), namespace: None, text: "beta".into(), metadata: json!({}), sparse_vector: None, colbert_tokens: None },
        NativeDocument { id: "c".into(), namespace: None, text: "gamma".into(), metadata: json!({}), sparse_vector: None, colbert_tokens: None },
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
    let res1 = col.query(Some(&q), None, None, None, None, None, 2, false, None, 60, 1.0, 1.0, 1.0, 1.0, true).unwrap();
    assert_eq!(res1[0].id, "a");

    // Dynamic deletion of "a"
    let deleted = col.delete(&["a".to_string()]).unwrap();
    assert_eq!(deleted, 1);
    assert_eq!(col.count(), 2);

    // Query ANN again: "a" must not appear, nearest should be "b"
    let res2 = col.query(Some(&q), None, None, None, None, None, 2, false, None, 60, 1.0, 1.0, 1.0, 1.0, true).unwrap();
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
            sparse_vector: None, colbert_tokens: None,
        },
        NativeDocument {
            id: "oak".into(),
            namespace: None,
            text: "Oakland City Center".into(),
            metadata: json!({"city": "Oakland", "loc": [37.8044, -122.2712]}),
            sparse_vector: None, colbert_tokens: None,
        },
        NativeDocument {
            id: "sjc".into(),
            namespace: None,
            text: "San Jose Plaza".into(),
            metadata: json!({"city": "San Jose", "loc": [37.3382, -121.8863]}),
            sparse_vector: None, colbert_tokens: None,
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
    let res_sf = col.query(None, Some("City"), None, None, None, Some(&sf_filter), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
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
    let res_box = col.query(None, Some("City"), None, None, None, Some(&bay_box), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
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
            sparse_vector: None, colbert_tokens: None,
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
        None,
        None,
        5,
        false,
        None,
        60,
        1.0,
        1.0,
        1.0,
        1.0,
        false,
    ).unwrap();

    assert!(!res.is_empty());
    assert_eq!(res[0].id, "doc_0");
    assert!((res[0].score - 1.0).abs() < 1e-4);
}

#[test]
fn test_nlp_analyzer_stemming_and_stopwords() {
    use _sutradb::bm25::analyzer::Analyzer;

    let analyzer = Analyzer::default();
    let text = "Deploying containerized microservices and connection pools in Kubernetes";
    let tokens = analyzer.analyze(text);

    // Stopwords "and" & "in" should be filtered out
    assert!(!tokens.contains(&"and".to_string()));
    assert!(!tokens.contains(&"in".to_string()));

    // Words should be stemmed
    assert!(tokens.contains(&"deploi".to_string()));
    assert!(tokens.contains(&"microservic".to_string()));
    assert!(tokens.contains(&"connect".to_string()));
    assert!(tokens.contains(&"pool".to_string()));
    assert!(tokens.contains(&"kubernet".to_string()));

    // Test code identifier splitting
    let code_tokens = analyzer.analyze("getUserProfile and fetch_jwt_token");
    assert!(code_tokens.contains(&"get".to_string()));
    assert!(code_tokens.contains(&"user".to_string()));
    assert!(code_tokens.contains(&"profil".to_string()));
    assert!(code_tokens.contains(&"fetch".to_string()));
    assert!(code_tokens.contains(&"jwt".to_string()));
    assert!(code_tokens.contains(&"token".to_string()));
}

#[test]
fn test_nlp_multilingual_and_cjk_support() {
    use _sutradb::bm25::analyzer::{Analyzer, AnalyzerConfig};

    // Spanish
    let es_config = AnalyzerConfig {
        language: "spanish".to_string(),
        ..Default::default()
    };
    let es_analyzer = Analyzer::new(es_config);
    let es_tokens = es_analyzer.analyze("Optimización y despliegue de servidores");
    // "y" and "de" are stopwords
    assert!(!es_tokens.contains(&"y".to_string()));
    assert!(!es_tokens.contains(&"de".to_string()));
    assert!(es_tokens.contains(&"optim".to_string()));

    // CJK bigrams
    let cjk_analyzer = Analyzer::default();
    let cjk_tokens = cjk_analyzer.analyze("自然语言处理");
    assert!(cjk_tokens.contains(&"自然".to_string()));
    assert!(cjk_tokens.contains(&"语言".to_string()));
    assert!(cjk_tokens.contains(&"处理".to_string()));
}

#[test]
fn test_simd_auto_dispatch_and_batch_distances() {
    use _sutradb::distance::simd::{detect_simd_capabilities, batch_distances};

    let simd_level = detect_simd_capabilities();
    println!("Detected CPU SIMD Level: {}", simd_level);

    let query = vec![1.0f32, 0.0, 0.0, 0.0];
    let mut vectors = Vec::new();
    for i in 0..100 {
        let mut v = vec![0.0f32; 4];
        v[i % 4] = 1.0;
        vectors.push(v);
    }

    let distances = batch_distances(&query, &vectors, Metric::Cosine, true);
    assert_eq!(distances.len(), 100);
    assert_eq!(distances[0], 1.0);
    assert_eq!(distances[1], 0.0);
    assert_eq!(distances[4], 1.0);
}

#[test]
fn test_secondary_metadata_roaring_and_range_filtering() {
    let mut col = NativeCollection::new(
        "sec_meta_test".to_string(),
        4,
        Metric::Cosine,
        None,
        false,
    ).unwrap();

    let docs = vec![
        NativeDocument {
            id: "doc_1".into(),
            namespace: None,
            text: "Fast database search".into(),
            metadata: json!({"tenant": "acme", "tier": "gold", "price": 99.5, "tags": ["db", "fast"]}),
            sparse_vector: None, colbert_tokens: None,
        },
        NativeDocument {
            id: "doc_2".into(),
            namespace: None,
            text: "Cloud container deployment".into(),
            metadata: json!({"tenant": "acme", "tier": "silver", "price": 49.0, "tags": ["cloud"]}),
            sparse_vector: None, colbert_tokens: None,
        },
        NativeDocument {
            id: "doc_3".into(),
            namespace: None,
            text: "Machine learning model serving".into(),
            metadata: json!({"tenant": "globex", "tier": "gold", "price": 150.0, "tags": ["ai", "fast"]}),
            sparse_vector: None, colbert_tokens: None,
        },
    ];

    let vecs = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];

    col.insert(docs, vecs).unwrap();

    // 1. Compound categorical filter: tenant == acme AND tier == gold
    let filter_compound = json!({"tenant": "acme", "tier": "gold"});
    let res = col.query(None, Some("search"), None, None, None, Some(&filter_compound), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].id, "doc_1");

    // 2. Numerical range filter: price >= 50.0
    let filter_price = json!({"price": {"$gte": 50.0}});
    let q_dummy = vec![1.0, 0.0, 0.0, 0.0];
    let res_price = col.query(Some(&q_dummy), None, None, None, None, Some(&filter_price), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(res_price.len(), 2);
    let ids: Vec<String> = res_price.into_iter().map(|r| r.id).collect();
    assert!(ids.contains(&"doc_1".to_string()));
    assert!(ids.contains(&"doc_3".to_string()));

    // 3. Array tag filter
    let filter_tags = json!({"tags": "cloud"});
    let res_tag = col.query(Some(&q_dummy), None, None, None, None, Some(&filter_tags), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(res_tag.len(), 1);
    assert_eq!(res_tag[0].id, "doc_2");
}

#[test]
fn test_cost_based_acorn_query_planning_low_selectivity() {
    let mut col = NativeCollection::new(
        "acorn_plan_test".to_string(),
        4,
        Metric::Cosine,
        None,
        false,
    ).unwrap();

    let n = 1000;
    let mut docs = Vec::new();
    let mut vecs = Vec::new();

    for i in 0..n {
        let is_needle = i == 42 || i == 777;
        docs.push(NativeDocument {
            id: format!("doc_{}", i),
            namespace: None,
            text: format!("Document number {}", i),
            metadata: json!({
                "is_needle": is_needle,
                "domain": if is_needle { "secret_vault" } else { "general_public" }
            }),
            sparse_vector: None, colbert_tokens: None,
        });
        vecs.push(vec![1.0, 0.0, 0.0, 0.0]);
    }

    col.insert(docs, vecs).unwrap();
    col.build_hnsw(None).unwrap();
    assert!(col.hnsw.is_some());

    // Filter selectivity is 2 out of 1000 (0.2%).
    // In naive HNSW, 0.2% selectivity leads to HNSW Filter Collapse.
    // In SutraDB v4, Cost-Based Planning automatically bypasses HNSW and dispatches direct SIMD scan over the matching 2 docs.
    let rare_filter = json!({"domain": "secret_vault"});
    let q = vec![1.0, 0.0, 0.0, 0.0];
    let res = col.query(
        Some(&q),
        None,
        None,
        None,
        None,
        Some(&rare_filter),
        10,
        false,
        None,
        60,
        1.0,
        1.0,
        1.0,
        1.0,
        true, // use_ann = true, but planner bypasses to guarantee 100% recall
    ).unwrap();

    assert_eq!(res.len(), 2);
    let result_ids: Vec<String> = res.into_iter().map(|r| r.id).collect();
    assert!(result_ids.contains(&"doc_42".to_string()));
    assert!(result_ids.contains(&"doc_777".to_string()));
}

#[test]
fn test_neural_sparse_splade_and_trimodal_rrf() {
    use _sutradb::sparse::SparseVector;

    let mut col = NativeCollection::new(
        "sparse_rrf_test".to_string(),
        4,
        Metric::Cosine,
        None,
        false,
    ).unwrap();

    let docs = vec![
        NativeDocument {
            id: "doc_sparse_1".into(),
            namespace: None,
            text: "Rust memory safety and concurrency".into(),
            metadata: json!({"tag": "systems"}),
            sparse_vector: Some(SparseVector::new(vec![100, 200, 300], vec![2.0, 1.5, 3.0])),
            colbert_tokens: None,
        },
        NativeDocument {
            id: "doc_sparse_2".into(),
            namespace: None,
            text: "Python asynchronous web development".into(),
            metadata: json!({"tag": "web"}),
            sparse_vector: Some(SparseVector::new(vec![200, 400], vec![3.0, 2.5])),
            colbert_tokens: None,
        },
    ];

    let vecs = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
    ];

    col.insert(docs, vecs).unwrap();

    // 1. Pure sparse vector query
    let query_sparse = SparseVector::new(vec![200, 400], vec![1.0, 2.0]);
    let sparse_res = col.query(
        None,
        None,
        Some(&query_sparse),
        None,
        None,
        None,
        5,
        false,
        None,
        60,
        1.0,
        1.0,
        1.0,
        1.0,
        false,
    ).unwrap();

    assert!(!sparse_res.is_empty());
    assert_eq!(sparse_res[0].id, "doc_sparse_2");
    assert!(sparse_res[0].sparse_score.is_some());

    // 2. Tri-Modal Hybrid Query (Dense + BM25 + Neural Sparse fused via RRF)
    let q_dense = vec![1.0, 0.0, 0.0, 0.0];
    let tri_res = col.query(
        Some(&q_dense),
        Some("memory safety"),
        Some(&query_sparse),
        None,
        None,
        None,
        5,
        true, // hybrid = true
        None,
        60,
        1.0,
        1.0,
        1.0,
        1.0,
        false,
    ).unwrap();

    assert_eq!(tri_res.len(), 2);
    assert_eq!(tri_res[0].id, "doc_sparse_1");
    assert!(tri_res[0].dense_score.is_some());
    assert!(tri_res[0].bm25_score.is_some());
    assert!(tri_res[0].sparse_score.is_some());
}

#[test]
fn test_colbert_multivector_maxsim_and_quadmodal_rrf() {
    let mut col = NativeCollection::new(
        "colbert_engine_test".to_string(),
        4,
        Metric::Cosine,
        None,
        false,
    ).unwrap();

    // Doc 1: Has token embeddings strongly aligned with query tokens
    let doc1_tokens = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
    ];
    // Doc 2: Has token embeddings orthogonal to query
    let doc2_tokens = vec![
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];

    let docs = vec![
        NativeDocument {
            id: "doc_colbert_1".into(),
            namespace: None,
            text: "Contextual late interaction multi-vector search".into(),
            metadata: json!({"type": "deep_learning"}),
            sparse_vector: None,
            colbert_tokens: Some(doc1_tokens),
        },
        NativeDocument {
            id: "doc_colbert_2".into(),
            namespace: None,
            text: "Standard database disk storage".into(),
            metadata: json!({"type": "storage"}),
            sparse_vector: None,
            colbert_tokens: Some(doc2_tokens),
        },
    ];

    let vecs = vec![
        vec![std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0],
        vec![0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2],
    ];

    col.insert(docs, vecs).unwrap();

    // 1. Direct ColBERT MaxSim Query
    let query_tokens = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
    ];

    let colbert_results = col.query_colbert(&query_tokens, None, None, 2).unwrap();
    assert_eq!(colbert_results.len(), 2);
    assert_eq!(colbert_results[0].id, "doc_colbert_1");
    // MaxSim score: max(1.0*1.0, 1.0*0.0) + max(0.0*1.0, 1.0*1.0) = 1.0 + 1.0 = 2.0
    assert!((colbert_results[0].score - 2.0).abs() < 1e-4);
    assert_eq!(colbert_results[0].colbert_score, Some(2.0));

    // 2. Quad-Modal Hybrid Query (Dense + BM25 + Sparse + ColBERT)
    let q_dense = vec![std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0];
    let quad_res = col.query(
        Some(&q_dense),
        Some("interaction multi-vector"),
        None,
        Some(&query_tokens),
        None,
        None,
        2,
        true, // hybrid = true
        None,
        60,
        1.0,
        1.0,
        1.0,
        1.0,
        false,
    ).unwrap();

    assert_eq!(quad_res.len(), 2);
    assert_eq!(quad_res[0].id, "doc_colbert_1");
    assert!(quad_res[0].colbert_score.is_some());
    assert!(quad_res[0].dense_score.is_some());
    assert!(quad_res[0].bm25_score.is_some());
}

#[test]
fn test_metric_preservation_on_reload() {
    let tmp = tempdir().unwrap();
    let euc_path = tmp.path().join("euc_test.sutra");

    // 1. Euclidean collection Float32
    let mut col_euc = NativeCollection::new(
        "euc_test".to_string(),
        4,
        Metric::Euclidean,
        Some(euc_path.clone()),
        false,
    ).unwrap();

    let docs = vec![
        NativeDocument {
            id: "d1".to_string(),
            namespace: None,
            text: "doc 1".to_string(),
            metadata: json!({"tag": "a"}),
            sparse_vector: None,
            colbert_tokens: None,
        },
    ];
    let vecs = vec![vec![1.0, 2.0, 3.0, 4.0]];
    col_euc.insert(docs, vecs).unwrap();
    col_euc.save::<PathBuf>(None).unwrap();

    // Reload Euclidean collection
    let mut col_reloaded = NativeCollection::new(
        "euc_test".to_string(),
        4,
        Metric::Cosine, // initially set to Cosine before load
        Some(euc_path),
        false,
    ).unwrap();
    col_reloaded.load::<PathBuf>(None).unwrap();
    assert_eq!(col_reloaded.metric, Metric::Euclidean, "Euclidean metric must be preserved on reload");

    // 2. SQ8 Cosine collection
    let sq8_path = tmp.path().join("sq8_test.sutra");
    let mut col_sq8 = NativeCollection::new(
        "sq8_test".to_string(),
        4,
        Metric::Cosine,
        Some(sq8_path.clone()),
        false,
    ).unwrap();

    let docs_sq8 = vec![
        NativeDocument {
            id: "s1".to_string(),
            namespace: None,
            text: "sq8 doc".to_string(),
            metadata: json!({"tag": "b"}),
            sparse_vector: None,
            colbert_tokens: None,
        },
    ];
    col_sq8.insert(docs_sq8, vec![vec![0.1, 0.2, 0.3, 0.4]]).unwrap();
    col_sq8.enable_sq8();
    col_sq8.save::<PathBuf>(None).unwrap();

    let mut col_sq8_reloaded = NativeCollection::new(
        "sq8_test".to_string(),
        4,
        Metric::Euclidean, // initially set to Euclidean before load
        Some(sq8_path),
        false,
    ).unwrap();
    col_sq8_reloaded.load::<PathBuf>(None).unwrap();
    assert_eq!(col_sq8_reloaded.metric, Metric::Cosine, "Cosine SQ8 metric must not be corrupted to Euclidean");
    assert!(col_sq8_reloaded.use_sq8, "SQ8 mode must be enabled on reload of SQ8 snapshot");
}

#[test]
fn test_wal_replay_preserves_vectors_on_update_metadata() {
    let tmp = tempdir().unwrap();
    let col_path = tmp.path().join("wal_update_test.sutra");

    let original_vec = vec![0.5, 0.5, 0.5, 0.5];
    let original_text = "persisted neural text";

    // Create collection and insert
    {
        let mut col = NativeCollection::new(
            "wal_update".to_string(),
            4,
            Metric::Cosine,
            Some(col_path.clone()),
            true, // enable WAL
        ).unwrap();

        col.insert(
            vec![NativeDocument {
                id: "target_doc".to_string(),
                namespace: None,
                text: original_text.to_string(),
                metadata: json!({"version": 1, "status": "draft"}),
                sparse_vector: None,
                colbert_tokens: None,
            }],
            vec![original_vec.clone()],
        ).unwrap();

        // Update metadata via WAL
        let updated = col.update_metadata("target_doc", json!({"version": 2, "status": "published", "author": "sam"})).unwrap();
        assert!(updated);
        // Do NOT call save(), so reload must replay WAL
    }

    // Reopen from WAL
    {
        let col_recovered = NativeCollection::new(
            "wal_update".to_string(),
            4,
            Metric::Cosine,
            Some(col_path),
            true,
        ).unwrap();

        assert_eq!(col_recovered.count(), 1);
        let doc = &col_recovered.documents[0];
        assert_eq!(doc.id, "target_doc");
        assert_eq!(doc.text, original_text, "Document text must not be wiped by WAL update");
        assert_eq!(doc.metadata["version"], 2);
        assert_eq!(doc.metadata["status"], "published");
        assert_eq!(doc.metadata["author"], "sam");

        let vec = &col_recovered.vectors[0];
        assert_eq!(vec, original_vec.as_slice(), "Document vector must not be wiped with zeros on WAL replay");

        // Verify secondary index reflects updated metadata and removed stale keys
        let old_hits = col_recovered.metadata_index.evaluate_filter(Some(&json!({"status": "draft"}))).unwrap();
        assert_eq!(old_hits.len(), 0, "Stale metadata must not match in secondary index after WAL update");

        let new_hits = col_recovered.metadata_index.evaluate_filter(Some(&json!({"status": "published"}))).unwrap();
        assert_eq!(new_hits.len(), 1, "Updated metadata must match in secondary index after WAL update");
    }
}

#[test]
fn test_hnsw_level_disconnect_healing() {
    let config = HNSWConfig {
        m: 4,
        m0: 8,
        ef_construction: 16,
        ef_search: 16,
        ml: 1.0 / (4.0f64).ln(),
        seed: 12345,
    };
    let mut hnsw = HNSWIndex::new(config);

    // Insert 50 vectors with known levels
    let mut all_vectors = Vec::new();
    for i in 0..50 {
        let v = vec![(i as f32) * 0.02, 1.0 - (i as f32) * 0.02, 0.5, 0.2];
        all_vectors.push(v);
    }

    for (i, vec) in all_vectors.iter().enumerate() {
        hnsw.insert(i as u32, vec, &all_vectors, Metric::Cosine, false).unwrap();
    }

    // Check entry point and layers
    let ep = hnsw.entry_point.expect("Entry point must exist");
    let max_lvl = hnsw.max_level;
    let ep_level = hnsw.node_level(ep).expect("Entry point must have active level");
    assert_eq!(ep_level, max_lvl);

    // Verify all layers up to max_lvl contain the entry point
    for l in 0..=max_lvl {
        assert!(
            hnsw.contains_node(l, ep),
            "Layer {} must contain entry point {}",
            l,
            ep
        );
    }

    // Search must succeed and return non-empty hits
    let query = vec![0.1, 0.9, 0.5, 0.2];
    let hits = hnsw.search(&query, &all_vectors, 5, 16, Metric::Cosine, false, None).unwrap();
    assert!(!hits.is_empty(), "HNSW search must find nearest neighbors without level disconnect");
}

#[test]
fn test_bm25_score_query_concurrent_immutable_readers() {
    use std::sync::Arc;
    use std::thread;

    let mut bm25 = BM25Index::default();
    let corpus = vec![
        "high performance vector search engine".to_string(),
        "zero dependency rust memory mapping persistence".to_string(),
        "hybrid reciprocal rank fusion with bm25".to_string(),
    ];
    bm25.add_documents(&corpus);

    let bm25_arc = Arc::new(bm25);
    let mut handles = Vec::new();

    for _ in 0..8 {
        let b = Arc::clone(&bm25_arc);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let scores = b.score_query("vector search", None);
                assert_eq!(scores.len(), 3);
                assert!(scores[0] > 0.0);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_bm25_idf_cache_poison_recovery() {
    let mut bm25 = BM25Index::default();
    bm25.add_documents(&["high performance vector search".to_string(), "database engine".to_string()]);

    // Poison the lock intentionally via catch_unwind
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _g = bm25.idf_cache.write().unwrap();
        panic!("simulated panic holding idf_cache write lock");
    }));

    assert!(bm25.idf_cache.is_poisoned(), "Lock must be poisoned");

    // BM25 query must recover gracefully without crashing or panicking
    let scores = bm25.score_query("vector search", None);
    assert_eq!(scores.len(), 2);
    assert!(scores[0] > 0.0);
}

#[test]
fn test_colbert_and_sparse_index_compaction_remap() {
    let mut col = NativeCollection::new("colbert_compact".to_string(), 4, Metric::Cosine, None, false).unwrap();

    let docs = vec![
        NativeDocument {
            id: "doc-0".to_string(),
            namespace: None,
            text: "first document to delete".to_string(),
            metadata: json!({"idx": 0}),
            sparse_vector: Some(_sutradb::sparse::SparseVector::new(vec![1, 2], vec![1.0, 2.0])),
            colbert_tokens: Some(vec![vec![1.0, 0.0, 0.0, 0.0]]),
        },
        NativeDocument {
            id: "doc-1".to_string(),
            namespace: None,
            text: "second document to keep".to_string(),
            metadata: json!({"idx": 1}),
            sparse_vector: Some(_sutradb::sparse::SparseVector::new(vec![3, 4], vec![3.0, 4.0])),
            colbert_tokens: Some(vec![vec![0.0, 1.0, 0.0, 0.0]]),
        },
        NativeDocument {
            id: "doc-2".to_string(),
            namespace: None,
            text: "third document to delete".to_string(),
            metadata: json!({"idx": 2}),
            sparse_vector: Some(_sutradb::sparse::SparseVector::new(vec![5], vec![5.0])),
            colbert_tokens: Some(vec![vec![0.0, 0.0, 1.0, 0.0]]),
        },
        NativeDocument {
            id: "doc-3".to_string(),
            namespace: None,
            text: "fourth document to keep".to_string(),
            metadata: json!({"idx": 3}),
            sparse_vector: Some(_sutradb::sparse::SparseVector::new(vec![6, 7], vec![6.0, 7.0])),
            colbert_tokens: Some(vec![vec![0.0, 0.0, 0.0, 1.0]]),
        },
    ];

    let vecs = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];

    col.insert(docs, vecs).unwrap();
    assert_eq!(col.count(), 4);

    // Delete doc-0 and doc-2
    col.delete(&["doc-0".to_string()]).unwrap();
    col.delete(&["doc-2".to_string()]).unwrap();
    assert_eq!(col.count(), 2);

    // Compact collection
    col.compact().unwrap();
    assert_eq!(col.count(), 2);
    assert_eq!(col.documents.len(), 2);
    assert_eq!(col.documents[0].id, "doc-1");
    assert_eq!(col.documents[1].id, "doc-3");

    // Verify ColBERT index reflects compacted IDs (doc-1 is now index 0)
    let q_tokens = vec![vec![0.0, 1.0, 0.0, 0.0]];
    let colbert_res = col.query_colbert(&q_tokens, None, None, 2).unwrap();
    assert!(!colbert_res.is_empty());
    assert_eq!(colbert_res[0].id, "doc-1", "Compact doc ID 0 must match doc-1's tokens");

    // Verify Sparse index reflects compacted IDs (doc-3 is now index 1)
    let q_sparse = _sutradb::sparse::SparseVector::new(vec![6], vec![1.0]);
    let sparse_res = col.query(None, None, Some(&q_sparse), None, None, None, 2, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert!(!sparse_res.is_empty());
    assert_eq!(sparse_res[0].id, "doc-3", "Compact doc ID 1 must match doc-3's sparse vector");
}

#[test]
fn test_dynamic_hnsw_incremental_ingestion() {
    let mut col = NativeCollection::new("hnsw_incremental".to_string(), 4, Metric::Cosine, None, false).unwrap();

    let mut docs = Vec::new();
    let mut vecs = Vec::new();
    for i in 0..50 {
        docs.push(NativeDocument {
            id: format!("doc-{}", i),
            namespace: None,
            text: format!("document {}", i),
            metadata: json!({"num": i}),
            sparse_vector: None,
            colbert_tokens: None,
        });
        vecs.push(vec![(i as f32) * 0.02, 1.0 - (i as f32) * 0.02, 0.5, 0.2]);
    }
    col.insert(docs, vecs).unwrap();

    // Build HNSW
    col.build_hnsw(None).unwrap();
    assert!(!col.dirty_index, "Index must not be dirty after build_hnsw");

    // Insert 10 more documents incrementally
    let mut new_docs = Vec::new();
    let mut new_vecs = Vec::new();
    for i in 50..60 {
        new_docs.push(NativeDocument {
            id: format!("doc-{}", i),
            namespace: None,
            text: format!("document {}", i),
            metadata: json!({"num": i}),
            sparse_vector: None,
            colbert_tokens: None,
        });
        new_vecs.push(vec![0.99, 0.01, 0.01, 0.01]);
    }
    col.insert(new_docs, new_vecs).unwrap();

    // HNSW index must NOT be dirty because incremental ingestion updated HNSW
    assert!(!col.dirty_index, "Dynamic HNSW ingestion must keep dirty_index false");

    // Query with vector matching new docs
    let q = vec![1.0, 0.0, 0.0, 0.0];
    let res = col.query(Some(&q), None, None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, true).unwrap();
    assert!(!res.is_empty());
    let top_id = &res[0].id;
    assert!(top_id.starts_with("doc-5"), "Top hit should be one of the dynamically inserted docs (50..60)");
}

#[test]
fn test_bm25_update_document_stale_term_purge() {
    let mut col = NativeCollection::new("bm25_update".to_string(), 2, Metric::Cosine, None, false).unwrap();

    col.insert(
        vec![NativeDocument {
            id: "doc-1".to_string(),
            namespace: None,
            text: "database indexing engine".to_string(),
            metadata: json!({}),
            sparse_vector: None,
            colbert_tokens: None,
        }],
        vec![vec![1.0, 0.0]],
    ).unwrap();

    // Initial search matches "indexing"
    let res1 = col.query(None, Some("indexing"), None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(res1.len(), 1);
    assert_eq!(res1[0].id, "doc-1");

    // In-place update with new text
    col.insert(
        vec![NativeDocument {
            id: "doc-1".to_string(),
            namespace: None,
            text: "machine learning vision model".to_string(),
            metadata: json!({}),
            sparse_vector: None,
            colbert_tokens: None,
        }],
        vec![vec![0.0, 1.0]],
    ).unwrap();

    // Stale keyword "indexing" must no longer match
    let res_stale = col.query(None, Some("indexing"), None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(res_stale.len(), 0, "Stale lexical terms must be purged upon re-insertion");

    // New keyword "learning" must match
    let res_new = col.query(None, Some("learning"), None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(res_new.len(), 1);
    assert_eq!(res_new[0].id, "doc-1");
}

#[test]
fn test_sq8_euclidean_metric_scoring() {
    let mut col = NativeCollection::new("sq8_euclidean".to_string(), 4, Metric::Euclidean, None, false).unwrap();

    col.insert(
        vec![
            NativeDocument {
                id: "d1".to_string(),
                namespace: None,
                text: "doc 1".to_string(),
                metadata: json!({}),
                sparse_vector: None,
                colbert_tokens: None,
            },
            NativeDocument {
                id: "d2".to_string(),
                namespace: None,
                text: "doc 2".to_string(),
                metadata: json!({}),
                sparse_vector: None,
                colbert_tokens: None,
            },
        ],
        vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.9, 0.8, 0.7, 0.6],
        ],
    ).unwrap();

    col.enable_sq8();

    let q = vec![0.1, 0.2, 0.3, 0.4];
    let res = col.query(Some(&q), None, None, None, None, None, 2, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();

    assert_eq!(res.len(), 2);
    // Euclidean similarity score is 1.0 / (1.0 + dist) which must be in (0, 1]
    assert!(res[0].score > 0.0 && res[0].score <= 1.0, "SQ8 Euclidean score must be a valid similarity score: got {}", res[0].score);
    assert_eq!(res[0].id, "d1");
    assert!(res[0].score > res[1].score);
}

#[test]
fn test_secondary_index_null_filter_eq_and_ne() {
    let mut idx = _sutradb::filter::SecondaryMetadataIndex::new();
    idx.index_document(0, &json!({"status": null, "tenant": "acme"}));
    idx.index_document(1, &json!({"tenant": "globex"})); // missing "status"
    idx.index_document(2, &json!({"status": "active", "tenant": "acme"}));
    idx.index_document(3, &json!({"status": "null", "tenant": "acme"})); // string "null", NOT null value

    // $eq: null matches doc 0 (null) and doc 1 (missing), but NOT doc 3 (string "null")
    let res_eq_null = idx.evaluate_filter(Some(&json!({"status": {"$eq": null}}))).unwrap();
    let hits_eq: Vec<u32> = res_eq_null.iter().collect();
    assert_eq!(hits_eq, vec![0, 1], "{{$eq: null}} must match null and missing fields, not string 'null'");

    // direct null matches doc 0 and doc 1
    let res_null = idx.evaluate_filter(Some(&json!({"status": null}))).unwrap();
    let hits_null: Vec<u32> = res_null.iter().collect();
    assert_eq!(hits_null, vec![0, 1], "{{field: null}} must match null and missing fields, not string 'null'");

    // $ne: null matches doc 2 (where status is "active") and doc 3 (where status is "null" string)
    let res_ne_null = idx.evaluate_filter(Some(&json!({"status": {"$ne": null}}))).unwrap();
    let hits_ne: Vec<u32> = res_ne_null.iter().collect();
    assert_eq!(hits_ne, vec![2, 3], "{{$ne: null}} must match non-null fields including string 'null'");

    // Query for literal string "null" matches ONLY doc 3, not doc 0 or 1
    let res_str_null = idx.evaluate_filter(Some(&json!({"status": "null"}))).unwrap();
    let hits_str: Vec<u32> = res_str_null.iter().collect();
    assert_eq!(hits_str, vec![3], "{{field: 'null'}} must match string 'null' only");
}

#[test]
fn test_float_to_sortable_u64_negative_zero() {
    use _sutradb::filter::index::float_to_sortable_u64;
    let neg_zero = -0.0f64;
    let pos_zero = 0.0f64;
    assert_eq!(float_to_sortable_u64(neg_zero), float_to_sortable_u64(pos_zero));
}

#[test]
#[should_panic(expected = "Dimension mismatch in BQVector hamming_distance")]
fn test_bq_dimension_mismatch_panics() {
    let bq1 = _sutradb::distance::bq::BQVector::quantize(&[1.0, -1.0, 0.5]);
    let bq2 = _sutradb::distance::bq::BQVector::quantize(&[1.0, -1.0, 0.5, -0.5]);
    let _ = bq1.hamming_distance(&bq2);
}

#[test]
fn test_mmap_crc32_corruption_detection() {
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom, Write};
    let tmp = tempdir().unwrap();
    let file_path = tmp.path().join("corrupted.sutra");

    let vecs = vec![0.1f32, 0.2, 0.3, 0.4];
    let docs = b"[{\"id\":\"1\",\"text\":\"hello\"}]";

    _sutradb::storage::BinaryStorage::write_f32(
        &file_path,
        &vecs,
        1,
        4,
        "cosine",
        docs,
    ).unwrap();

    // Verify valid open works
    assert!(_sutradb::storage::MmapVectorStorage::open(&file_path).is_ok());

    // Corrupt a byte in the vector payload (after the 64-byte header)
    {
        let mut file = OpenOptions::new().write(true).open(&file_path).unwrap();
        file.seek(SeekFrom::Start(68)).unwrap();
        file.write_all(&[0xFF]).unwrap();
    }

    // Opening corrupted file must fail with InvalidHeader containing CRC32 mismatch
    let err = match _sutradb::storage::MmapVectorStorage::open(&file_path) {
        Err(e) => e,
        Ok(_) => panic!("Expected corrupted file to fail MmapVectorStorage::open"),
    };
    let err_msg = format!("{:?}", err);
    assert!(err_msg.contains("CRC32 checksum mismatch"), "Expected CRC32 error, got: {}", err_msg);
}

#[test]
fn test_wal_replay_preserves_sparse_and_colbert_modalities() {
    let tmp = tempdir().unwrap();
    let col_path = tmp.path().join("wal_multimodal.sutra");

    let sparse_v = _sutradb::sparse::SparseVector::new(vec![10, 20], vec![1.5, 3.0]);
    let colbert_toks = vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7, 0.8]];

    // 1. Create collection with WAL enabled and insert multimodal document
    {
        let mut col = NativeCollection::new(
            "wal_mm".to_string(),
            4,
            Metric::Cosine,
            Some(col_path.clone()),
            true, // enable WAL
        ).unwrap();

        col.insert(
            vec![NativeDocument {
                id: "mm_doc".to_string(),
                namespace: None,
                text: "multimodal retrieval document".to_string(),
                metadata: json!({"tag": "ai"}),
                sparse_vector: Some(sparse_v.clone()),
                colbert_tokens: Some(colbert_toks.clone()),
            }],
            vec![vec![0.5, 0.5, 0.5, 0.5]],
        ).unwrap();
        // Do NOT call save() - simulate sudden power loss / crash
    }

    // 2. Reopen from WAL replay
    {
        let col_recovered = NativeCollection::new(
            "wal_mm".to_string(),
            4,
            Metric::Cosine,
            Some(col_path),
            true,
        ).unwrap();

        assert_eq!(col_recovered.count(), 1);
        let doc = &col_recovered.documents[0];
        assert_eq!(doc.id, "mm_doc");
        assert_eq!(doc.sparse_vector.as_ref().unwrap(), &sparse_v);
        assert_eq!(doc.colbert_tokens.as_ref().unwrap(), &colbert_toks);

        // Verify that sparse index can retrieve the recovered document
        let query_sp = _sutradb::sparse::SparseVector::new(vec![10], vec![1.0]);
        let hits = col_recovered.sparse_index.search(&query_sp, 5, None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 0);

        // Verify that ColBERT index can query the recovered document
        let hits_colbert = col_recovered.colbert_index.query_maxsim(&colbert_toks, None, 5);
        assert_eq!(hits_colbert.len(), 1);
        assert_eq!(hits_colbert[0].0, 0);
    }
}

#[test]
fn test_bm25_analyzer_preservation_on_compaction_and_load() {
    let tmp = tempdir().unwrap();
    let col_path = tmp.path().join("analyzer_preserve.sutra");

    let custom_cfg = _sutradb::bm25::AnalyzerConfig {
        lowercase: true,
        language: "german".to_string(),
        stemmer: true,
        stopwords: false,
        split_identifiers: true,
        cjk_bigrams: true,
    };

    let mut col = NativeCollection::new(
        "analyzer_test".to_string(),
        4,
        Metric::Cosine,
        Some(col_path.clone()),
        false,
    ).unwrap();

    // Configure custom analyzer
    col.bm25.analyzer = _sutradb::bm25::Analyzer::new(custom_cfg.clone());

    col.insert(
        vec![
            NativeDocument {
                id: "d1".to_string(),
                namespace: None,
                text: "text one".to_string(),
                metadata: json!({}),
                sparse_vector: None,
                colbert_tokens: None,
            },
            NativeDocument {
                id: "d2".to_string(),
                namespace: None,
                text: "text two".to_string(),
                metadata: json!({}),
                sparse_vector: None,
                colbert_tokens: None,
            },
        ],
        vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7, 0.8]],
    ).unwrap();

    // Delete d1 to create a tombstone and compact
    col.delete(&["d1".to_string()]).unwrap();
    col.compact().unwrap();

    // Verify analyzer config was preserved across compact()
    assert_eq!(col.bm25.analyzer.config.language, "german");
    assert!(col.bm25.analyzer.config.split_identifiers);
    assert!(col.bm25.analyzer.config.cjk_bigrams);

    // Save and load
    col.save::<PathBuf>(None).unwrap();
    col.load::<PathBuf>(None).unwrap();

    // Verify analyzer config was preserved across load()
    assert_eq!(col.bm25.analyzer.config.language, "german");
    assert!(col.bm25.analyzer.config.split_identifiers);
    assert!(col.bm25.analyzer.config.cjk_bigrams);
}

// ============================================================================
// M1 Audit Remediation Regression Tests
// ============================================================================

#[test]
fn test_sq8_v4_negative_and_scaled_roundtrip_and_v3_backward_read() {
    use _sutradb::storage::{BinaryStorage, MmapVectorStorage, HEADER_MAGIC_V3, HEADER_MAGIC_V4};

    let tmp = tempdir().unwrap();
    let v4_path = tmp.path().join("sq8_v4_cos.sutra");

    // Negative and scaled vectors under Cosine
    let mut col = NativeCollection::new(
        "sq8_v4_test".to_string(),
        4,
        Metric::Cosine,
        Some(v4_path.clone()),
        false,
    )
    .unwrap();

    let v0 = vec![-std::f32::consts::FRAC_1_SQRT_2, 0.0, std::f32::consts::FRAC_1_SQRT_2, 0.0];
    let v1 = vec![0.0, -std::f32::consts::FRAC_1_SQRT_2, 0.0, std::f32::consts::FRAC_1_SQRT_2];
    col.insert(
        vec![
            NativeDocument {
                id: "neg0".into(),
                namespace: None,
                text: "negative vec".into(),
                metadata: json!({}),
                sparse_vector: None,
                colbert_tokens: None,
            },
            NativeDocument {
                id: "neg1".into(),
                namespace: None,
                text: "other vec".into(),
                metadata: json!({}),
                sparse_vector: None,
                colbert_tokens: None,
            },
        ],
        vec![v0.clone(), v1.clone()],
    )
    .unwrap();
    col.enable_sq8();
    let (orig_min_val, orig_scale) = col.sq8_vectors.params[0];
    col.save::<PathBuf>(None).unwrap();

    // Verify file header is V4
    let mmap = MmapVectorStorage::open(&v4_path).unwrap();
    assert_eq!(mmap.header().magic, *HEADER_MAGIC_V4);
    assert_eq!(mmap.header().quant_mode, 1);
    let (min_val, scale) = mmap.get_sq8_params(0).unwrap();
    assert!((min_val - orig_min_val).abs() < 1e-6);
    assert!((scale - orig_scale).abs() < 1e-6);

    // Reload into fresh collection and check exact SQ8 params & cosine score ~ 1.0
    let mut reloaded = NativeCollection::new(
        "sq8_v4_test".to_string(),
        4,
        Metric::Cosine,
        Some(v4_path),
        false,
    )
    .unwrap();
    reloaded.load::<PathBuf>(None).unwrap();
    assert!(reloaded.use_sq8);
    assert_eq!(reloaded.sq8_vectors.params[0].0, orig_min_val);
    assert_eq!(reloaded.sq8_vectors.params[0].1, orig_scale);
    for (a, b) in reloaded.vectors[0].iter().zip(v0.iter()) {
        assert!((a - b).abs() < 0.02, "Dequantized mismatch: {} vs {}", a, b);
    }
    let hits = reloaded
        .query(Some(&v0), None, None, None, None, None, 2, false, None, 60, 1.0, 1.0, 1.0, 1.0, false)
        .unwrap();
    assert_eq!(hits[0].id, "neg0");
    assert!((hits[0].score - 1.0).abs() < 0.03, "Expected cosine ~ 1.0, got {}", hits[0].score);

    // Scaled Euclidean SQ8 V4 roundtrip
    let euc_path = tmp.path().join("sq8_v4_euc.sutra");
    let mut col_euc = NativeCollection::new(
        "sq8_v4_euc".to_string(),
        4,
        Metric::Euclidean,
        Some(euc_path.clone()),
        false,
    )
    .unwrap();
    let scaled = vec![10.0, -20.0, 30.0, -40.0];
    col_euc
        .insert(
            vec![NativeDocument {
                id: "s0".into(),
                namespace: None,
                text: "scaled".into(),
                metadata: json!({}),
                sparse_vector: None,
                colbert_tokens: None,
            }],
            vec![scaled.clone()],
        )
        .unwrap();
    col_euc.enable_sq8();
    col_euc.save::<PathBuf>(None).unwrap();

    let mut reloaded_euc = NativeCollection::new(
        "sq8_v4_euc".to_string(),
        4,
        Metric::Euclidean,
        Some(euc_path),
        false,
    )
    .unwrap();
    reloaded_euc.load::<PathBuf>(None).unwrap();
    for (a, b) in reloaded_euc.vectors[0].iter().zip(scaled.iter()) {
        assert!((a - b).abs() < 0.5, "Scaled dequantized mismatch: {} vs {}", a, b);
    }

    // Legacy V3 SQ8 backward compatibility read
    let v3_path = tmp.path().join("legacy_v3_sq8.sutra");
    let raw_sq8 = vec![0u8, 128, 255, 64];
    let meta = br#"[{"id":"v3doc","text":"legacy","metadata":{}}]"#;
    BinaryStorage::write_sq8(&v3_path, &raw_sq8, 1, 4, "cosine", meta).unwrap();
    let mmap_v3 = MmapVectorStorage::open(&v3_path).unwrap();
    assert_eq!(mmap_v3.header().magic, *HEADER_MAGIC_V3);
    assert!(mmap_v3.get_sq8_params(0).is_err());

    let mut col_v3 = NativeCollection::new(
        "legacy_v3".to_string(),
        4,
        Metric::Cosine,
        Some(v3_path),
        false,
    )
    .unwrap();
    col_v3.load::<PathBuf>(None).unwrap();
    assert_eq!(col_v3.count(), 1);
    assert!(col_v3.use_sq8);
    assert!((col_v3.vectors[0][2] - 1.0).abs() < 1e-5);
}

#[test]
fn test_wal_clear_append_replay_and_name_validation() {
    use _sutradb::validate_collection_name;

    let tmp = tempdir().unwrap();
    let wal_path = tmp.path().join("clear_test.wal");
    let mut wal = WriteAheadLog::open(&wal_path).unwrap();

    wal.append(&WalRecord {
        op: "insert".into(),
        docs: vec![json!({"id": "old", "text": "before clear"})],
        ids: Vec::new(),
    })
    .unwrap();
    wal.clear().unwrap();
    assert_eq!(std::fs::metadata(&wal_path).unwrap().len(), 0);

    // Subsequent append after clear must append cleanly at offset 0 (no NUL hole)
    wal.append(&WalRecord {
        op: "insert".into(),
        docs: vec![json!({"id": "new", "text": "after clear"})],
        ids: Vec::new(),
    })
    .unwrap();
    let replayed = wal.replay().unwrap();
    assert_eq!(replayed.len(), 1);
    assert_eq!(replayed[0].docs[0]["id"], "new");

    // Collection name validation
    assert!(validate_collection_name("valid-col_123").is_ok());
    for bad in ["", "../evil", "a/b", "a\\b", "bad name", "a\0b"] {
        assert!(validate_collection_name(bad).is_err(), "Should reject {:?}", bad);
        assert!(
            NativeCollection::new(bad.to_string(), 4, Metric::Cosine, None, false).is_err(),
            "NativeCollection::new should reject {:?}",
            bad
        );
    }
    let too_long = "a".repeat(129);
    assert!(validate_collection_name(&too_long).is_err());
}

#[test]
fn test_save_and_compact_preserve_hnsw_and_ivf() {
    let tmp = tempdir().unwrap();
    let hnsw_path = tmp.path().join("hnsw_compact.sutra");
    let mut col = NativeCollection::new(
        "hnsw_compact".to_string(),
        4,
        Metric::Cosine,
        Some(hnsw_path),
        false,
    )
    .unwrap();

    let mut docs = Vec::new();
    let mut vecs = Vec::new();
    for i in 0..20 {
        docs.push(NativeDocument {
            id: format!("d{}", i),
            namespace: None,
            text: format!("doc {}", i),
            metadata: json!({}),
            sparse_vector: None,
            colbert_tokens: None,
        });
        vecs.push(vec![(i as f32) + 1.0, 0.5, 0.2, 0.1]);
    }
    col.insert(docs, vecs).unwrap();
    col.build_hnsw(Some(HNSWConfig {
        m: 8,
        m0: 16,
        ef_construction: 32,
        ef_search: 24,
        ml: 1.0 / (8.0f64).ln(),
        seed: 42,
    }))
    .unwrap();

    col.delete(&["d0".to_string()]).unwrap();
    col.save::<PathBuf>(None).unwrap();

    assert!(col.hnsw.is_some());
    assert!(!col.dirty_index);
    assert_eq!(col.stats()["has_hnsw"], true);
    let hits = col
        .query(Some(&[20.0, 0.5, 0.2, 0.1]), None, None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, true)
        .unwrap();
    assert!(!hits.is_empty());
    assert!(!hits.iter().any(|h| h.id == "d0"));

    // Repeat for IVF
    let mut col_ivf = NativeCollection::new("ivf_compact".to_string(), 4, Metric::Cosine, None, false).unwrap();
    let mut docs2 = Vec::new();
    let mut vecs2 = Vec::new();
    for i in 0..20 {
        docs2.push(NativeDocument {
            id: format!("d{}", i),
            namespace: None,
            text: format!("doc {}", i),
            metadata: json!({}),
            sparse_vector: None,
            colbert_tokens: None,
        });
        vecs2.push(vec![(i as f32) + 1.0, 0.5, 0.2, 0.1]);
    }
    col_ivf.insert(docs2, vecs2).unwrap();
    col_ivf
        .build_ivf(Some(_sutradb::index::ivf::IVFConfig {
            n_lists: 4,
            nprobe: 2,
            kmeans_iters: 10,
            seed: 42,
        }))
        .unwrap();
    col_ivf.delete(&["d0".to_string()]).unwrap();
    col_ivf.compact().unwrap();

    assert!(col_ivf.ivf.is_some());
    assert!(!col_ivf.dirty_index);
    assert_eq!(col_ivf.stats()["has_ivf"], true);
}

#[test]
fn test_bq_pq_negative_dense_scores_and_linear_fusion_finite() {
    // 40 docs all pointing into negative hemisphere relative to query [1, 0, 0, 0, ...]
    let dim = 8;
    let mut docs = Vec::new();
    let mut vecs = Vec::new();
    for i in 0..40 {
        let x = if i == 0 { -0.1 } else { -0.9 };
        let y = (1.0f32 - x * x).sqrt();
        let mut v = vec![0.0f32; dim];
        v[0] = x;
        v[1] = y;
        docs.push(NativeDocument {
            id: format!("d{}", i),
            namespace: None,
            text: if i == 0 { "match keyword".into() } else { "other text".into() },
            metadata: json!({}),
            sparse_vector: None,
            colbert_tokens: None,
        });
        vecs.push(v);
    }

    let mut q = vec![0.0f32; dim];
    q[0] = 1.0;

    // 1. BQ negative dense scores & linear fusion
    let mut col_bq = NativeCollection::new("bq_neg".to_string(), dim, Metric::Cosine, None, false).unwrap();
    col_bq.insert(docs.clone(), vecs.clone()).unwrap();
    col_bq.enable_bq();
    let res_bq = col_bq
        .query(Some(&q), None, None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, false)
        .unwrap();
    assert_eq!(res_bq.len(), 5);
    assert_eq!(res_bq[0].id, "d0", "Un-reranked docs must not beat -0.1 score");
    assert!(res_bq.iter().all(|r| r.score.is_finite() && r.score < 0.0));

    let res_bq_lin = col_bq
        .query(Some(&q), Some("match keyword"), None, None, None, None, 5, true, Some(0.5), 60, 1.0, 1.0, 1.0, 1.0, false)
        .unwrap();
    assert!(res_bq_lin.iter().all(|r| r.score.is_finite()), "Linear fusion must not emit NaN");

    // 2. PQ negative dense scores & linear fusion
    let mut col_pq = NativeCollection::new("pq_neg".to_string(), dim, Metric::Cosine, None, false).unwrap();
    col_pq.insert(docs, vecs).unwrap();
    col_pq.train_pq(Some(4), None).unwrap();
    let res_pq = col_pq
        .query(Some(&q), None, None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, false)
        .unwrap();
    assert_eq!(res_pq.len(), 5);
    assert_eq!(res_pq[0].id, "d0", "Un-reranked PQ docs must not beat -0.1 score");
    assert!(res_pq.iter().all(|r| r.score.is_finite() && r.score < 0.0));

    let res_pq_lin = col_pq
        .query(Some(&q), Some("match keyword"), None, None, None, None, 5, true, Some(0.5), 60, 1.0, 1.0, 1.0, 1.0, false)
        .unwrap();
    assert!(res_pq_lin.iter().all(|r| r.score.is_finite()), "PQ linear fusion must not emit NaN");
}

#[test]
fn test_rayon_64_plus_query_and_batch_dimension_mismatch() {
    use _sutradb::distance::simd::try_batch_distances;
    use _sutradb::error::SutraError;

    let mut col = NativeCollection::new("rayon_dim".to_string(), 4, Metric::Cosine, None, false).unwrap();
    let mut docs = Vec::new();
    let mut vecs = Vec::new();
    for i in 0..70 {
        docs.push(NativeDocument {
            id: format!("d{}", i),
            namespace: None,
            text: "doc".into(),
            metadata: json!({}),
            sparse_vector: None,
            colbert_tokens: None,
        });
        vecs.push(vec![1.0, 0.0, 0.0, 0.0]);
    }
    col.insert(docs, vecs.clone()).unwrap();

    let bad_q = vec![1.0, 0.0, 0.0];
    let err = col.query(Some(&bad_q), None, None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap_err();
    assert!(matches!(err, SutraError::DimensionMismatch { expected: 4, got: 3 }));

    col.enable_sq8();
    assert!(matches!(
        col.query(Some(&bad_q), None, None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, false),
        Err(SutraError::DimensionMismatch { expected: 4, got: 3 })
    ));

    col.enable_bq();
    assert!(matches!(
        col.query(Some(&bad_q), None, None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, false),
        Err(SutraError::DimensionMismatch { expected: 4, got: 3 })
    ));

    assert!(matches!(
        try_batch_distances(&bad_q, &vecs, Metric::Cosine, false),
        Err(SutraError::DimensionMismatch { expected: 3, got: 4 })
    ));
}

#[test]
fn test_h2_no_panics_on_malformed_inputs_and_atomic_batch() {
    use _sutradb::distance::pq::{PQCodebook, PQConfig};
    use _sutradb::error::SutraError;
    use _sutradb::sparse::SparseVector;
    use _sutradb::storage::{MmapVectorStorage, SutraHeader, HEADER_MAGIC_V4};

    // 1. PQCodebook::train validation
    let vecs = vec![vec![1.0, 2.0, 3.0, 4.0]];
    assert!(PQCodebook::train(&vecs, 4, PQConfig { m: 0, ..Default::default() }).is_err());
    assert!(PQCodebook::train(&vecs, 4, PQConfig { m: 8, ..Default::default() }).is_err());
    assert!(PQCodebook::train(&vecs, 4, PQConfig { m: 3, ..Default::default() }).is_err());
    assert!(PQCodebook::train(&[vec![1.0, 2.0]], 4, PQConfig { m: 2, ..Default::default() }).is_err());

    // 2. SparseVector::try_new
    assert!(matches!(
        SparseVector::try_new(vec![1, 2], vec![1.0]),
        Err(SutraError::DimensionMismatch { expected: 2, got: 1 })
    ));

    // 3. Atomic batch insert_internal on bad vector or bad colbert token dim
    let mut col = NativeCollection::new("atomic_batch".to_string(), 4, Metric::Cosine, None, false).unwrap();
    let err_vec = col.insert(
        vec![
            NativeDocument { id: "ok".into(), namespace: None, text: "ok".into(), metadata: json!({}), sparse_vector: None, colbert_tokens: None },
            NativeDocument { id: "bad".into(), namespace: None, text: "bad".into(), metadata: json!({}), sparse_vector: None, colbert_tokens: None },
        ],
        vec![vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 0.0]],
    );
    assert!(matches!(err_vec, Err(SutraError::DimensionMismatch { expected: 4, got: 2 })));
    assert_eq!(col.count(), 0);
    assert!(col.documents.is_empty());

    let err_cb = col.insert(
        vec![
            NativeDocument { id: "ok".into(), namespace: None, text: "ok".into(), metadata: json!({}), sparse_vector: None, colbert_tokens: None },
            NativeDocument { id: "bad_cb".into(), namespace: None, text: "bad".into(), metadata: json!({}), sparse_vector: None, colbert_tokens: Some(vec![vec![1.0, 2.0]]) },
        ],
        vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
    );
    assert!(matches!(err_cb, Err(SutraError::DimensionMismatch { expected: 4, got: 2 })));
    assert_eq!(col.count(), 0);

    // 4. ColBERT query dimension checks
    assert!(matches!(
        col.query_colbert(&[vec![1.0, 2.0]], None, None, 5),
        Err(SutraError::DimensionMismatch { expected: 4, got: 2 })
    ));
    assert!(matches!(
        col.query(None, None, None, Some(&[vec![1.0, 2.0]]), None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, false),
        Err(SutraError::DimensionMismatch { expected: 4, got: 2 })
    ));

    // 5. MmapVectorStorage overflow / invalid quant_mode / metadata mismatch
    let tmp = tempdir().unwrap();
    let bad_qm_path = tmp.path().join("bad_qm.sutra");
    let mut hdr_bytes = SutraHeader {
        magic: *HEADER_MAGIC_V4,
        doc_count: 1,
        dimension: 4,
        metric: "cosine".into(),
        metadata_length: 2,
        crc32: 0,
        quant_mode: 9,
    }
    .to_bytes();
    let mut file_bytes = hdr_bytes.to_vec();
    file_bytes.extend_from_slice(&[0u8; 32]);
    std::fs::write(&bad_qm_path, &file_bytes).unwrap();
    assert!(matches!(MmapVectorStorage::open(&bad_qm_path), Err(SutraError::InvalidHeader(_))));

    // Overflow u32::MAX * u32::MAX
    let overflow_path = tmp.path().join("overflow.sutra");
    hdr_bytes = SutraHeader::new(u32::MAX, u32::MAX, "cosine", 0, 0, 0).to_bytes();
    std::fs::write(&overflow_path, hdr_bytes).unwrap();
    assert!(matches!(MmapVectorStorage::open(&overflow_path), Err(SutraError::InvalidHeader(_))));

    // Metadata count mismatch with valid CRC32
    let meta_mismatch_path = tmp.path().join("meta_mismatch.sutra");
    let vecs_raw = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let bad_meta = br#"[{"id":"only_one","text":"a","metadata":{}}]"#;
    _sutradb::storage::BinaryStorage::write_f32(&meta_mismatch_path, &vecs_raw, 2, 4, "cosine", bad_meta).unwrap();
    let mut col_load = NativeCollection::new("meta_mismatch".to_string(), 4, Metric::Cosine, Some(meta_mismatch_path), false).unwrap();
    assert!(matches!(col_load.load::<PathBuf>(None), Err(SutraError::InvalidHeader(_))));
}

#[test]
fn test_hnsw_filtered_recall_above_acorn_threshold() {
    let mut col = NativeCollection::new("hnsw_filtered".to_string(), 4, Metric::Cosine, None, false).unwrap();
    let mut docs = Vec::new();
    let mut vecs = Vec::new();
    for i in 0..1000 {
        let is_rare = i >= 400; // 60% selectivity (> 2% ACORN threshold)
        docs.push(NativeDocument {
            id: format!("doc_{}", i),
            namespace: None,
            text: format!("doc {}", i),
            metadata: json!({"tag": if is_rare { "rare_cluster" } else { "blocked" }}),
            sparse_vector: None,
            colbert_tokens: None,
        });
        if is_rare {
            vecs.push(vec![0.01 * ((i % 10) as f32), 1.0, 0.0, 0.0]);
        } else {
            vecs.push(vec![1.0, 0.01 * ((i % 10) as f32), 0.0, 0.0]);
        }
    }
    col.insert(docs, vecs).unwrap();
    col.build_hnsw(Some(HNSWConfig {
        m: 16,
        m0: 32,
        ef_construction: 64,
        ef_search: 32,
        ml: 1.0 / (16.0f64).ln(),
        seed: 42,
    }))
    .unwrap();

    // Query near blocked cluster [1, 0, 0, 0] with filter tag=rare_cluster
    let filter = json!({"tag": "rare_cluster"});
    let hits = col
        .query(Some(&[1.0, 0.0, 0.0, 0.0]), None, None, None, None, Some(&filter), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, true)
        .unwrap();
    assert_eq!(hits.len(), 10, "Filtered HNSW must traverse through masked-out nodes to return top_k=10 results");
    assert!(hits.iter().all(|h| h.metadata["tag"] == "rare_cluster"));
}

#[test]
fn test_hnsw_delete_cleans_asymmetric_incoming_edges() {
    let mut hnsw = HNSWIndex::new(HNSWConfig {
        m: 2,
        m0: 2,
        ef_construction: 16,
        ef_search: 16,
        ml: 1.0 / (2.0f64).ln(),
        seed: 7,
    });
    let mut vectors = Vec::new();
    for i in 0..35 {
        vectors.push(normalize_copy(&[((i % 7) + 1) as f32, ((i / 7) + 1) as f32, 0.3, 0.1]));
    }
    for (i, v) in vectors.iter().enumerate() {
        hnsw.insert(i as u32, v, &vectors, Metric::Cosine, true).unwrap();
    }

    let deleted_ids = [0u32, 5, 10, 15, 20];
    for &del_id in &deleted_ids {
        assert!(hnsw.delete_node(del_id, &vectors, Metric::Cosine, true));
        assert!(hnsw.tombstones.contains(del_id));
        assert!(hnsw.node_level(del_id).is_none());
    }

    let hits = hnsw.search(&vectors[1], &vectors, 10, 16, Metric::Cosine, true, None).unwrap();
    assert!(!hits.is_empty());
    for (id, _) in hits {
        assert!(!deleted_ids.contains(&id), "Deleted node {} returned by search", id);
    }

    // Compacting HNSW tombstones scrubs all dangling edges across flat layers
    hnsw.compact_tombstones();
    for layer in &hnsw.layers {
        for &del_id in &deleted_ids {
            assert!(layer.get(del_id as usize).is_none_or(|v| v.is_empty()));
            for nbrs in layer {
                assert!(!nbrs.contains(&del_id), "Dangling edge to {} remained after compaction", del_id);
            }
        }
    }
}

#[test]
fn test_filter_regex_not_complex_exists_and_type_parity() {
    let mut col = NativeCollection::new("filter_parity".to_string(), 4, Metric::Cosine, None, false).unwrap();
    let docs = vec![
        NativeDocument {
            id: "d0".into(),
            namespace: None,
            text: "alpha".into(),
            metadata: json!({
                "name": "foo-123",
                "tags": ["rust", "db"],
                "scores": [10, 100],
                "count": 1.0,
                "user": {"name": "alice"},
                "empty_arr": []
            }),
            sparse_vector: None,
            colbert_tokens: None,
        },
        NativeDocument {
            id: "d1".into(),
            namespace: None,
            text: "beta".into(),
            metadata: json!({
                "name": "bar-456",
                "tags": ["python"],
                "scores": [10, 20],
                "count": 2,
                "user": null,
                "empty_arr": [1]
            }),
            sparse_vector: None,
            colbert_tokens: None,
        },
    ];
    col.insert(docs, vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]]).unwrap();

    // $regex match and invalid regex
    let q = vec![1.0, 0.0, 0.0, 0.0];
    let rx = col.query(Some(&q), None, None, None, None, Some(&json!({"name": {"$regex": "^foo-\\d+$"}})), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(rx.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), vec!["d0"]);

    let rx_bad = col.query(Some(&q), None, None, None, None, Some(&json!({"name": {"$regex": "[unclosed"}})), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert!(rx_bad.is_empty());

    // $not over complex operators ($contains and $regex)
    let not_contains = col.query(Some(&q), None, None, None, None, Some(&json!({"$not": {"tags": {"$contains": "rust"}}})), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(not_contains.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), vec!["d1"]);

    let not_regex = col.query(Some(&q), None, None, None, None, Some(&json!({"$not": {"name": {"$regex": "^foo"}}})), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(not_regex.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), vec!["d1"]);

    // $exists on nested object & empty array
    let exists_user = col.query(Some(&q), None, None, None, None, Some(&json!({"user": {"$exists": true}})), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(exists_user.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), vec!["d0"]);

    let exists_empty = col.query(Some(&q), None, None, None, None, Some(&json!({"empty_arr": {"$exists": true}})), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(exists_empty.len(), 2);

    // Numeric 1 == 1.0 and array-element comparison parity with/without complex ops
    let eq_int = col.query(Some(&q), None, None, None, None, Some(&json!({"count": 1})), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(eq_int.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), vec!["d0"]);

    let eq_int_complex = col.query(Some(&q), None, None, None, None, Some(&json!({"count": 1, "name": {"$regex": "foo"}})), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(eq_int_complex.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), vec!["d0"]);

    let arr_gt = col.query(Some(&q), None, None, None, None, Some(&json!({"scores": {"$gt": 50}, "name": {"$regex": ".*"}})), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(arr_gt.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), vec!["d0"]);

    // Dotted array index path ("scores.1" == 100)
    let dotted_arr = col.query(Some(&q), None, None, None, None, Some(&json!({"scores.1": 100})), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(dotted_arr.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), vec!["d0"]);
}

#[test]
fn test_bm25_delete_updates_idf_and_avdl_without_oob() {
    let mut col = NativeCollection::new("bm25_del".to_string(), 2, Metric::Cosine, None, false).unwrap();
    col.insert(
        vec![
            NativeDocument { id: "d0".into(), namespace: None, text: "rareterm alpha".into(), metadata: json!({}), sparse_vector: None, colbert_tokens: None },
            NativeDocument { id: "d1".into(), namespace: None, text: "common beta".into(), metadata: json!({}), sparse_vector: None, colbert_tokens: None },
            NativeDocument { id: "d2".into(), namespace: None, text: "common gamma".into(), metadata: json!({}), sparse_vector: None, colbert_tokens: None },
        ],
        vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]],
    )
    .unwrap();
    assert_eq!(col.bm25.doc_count, 3);

    col.delete(&["d0".to_string()]).unwrap();
    assert_eq!(col.bm25.doc_count, 2);
    // Deleted term must not match
    let res_deleted = col.query(None, Some("rareterm"), None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert!(res_deleted.is_empty());

    // Surviving doc 2 must still score without OOB panic
    let res_surv = col.query(None, Some("gamma"), None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(res_surv.len(), 1);
    assert_eq!(res_surv[0].id, "d2");

    // Inserting a new doc after deletion must assign unique slot index without collision
    col.insert(
        vec![NativeDocument { id: "d3".into(), namespace: None, text: "delta unique".into(), metadata: json!({}), sparse_vector: None, colbert_tokens: None }],
        vec![vec![1.0, 0.0]],
    )
    .unwrap();
    assert_eq!(col.bm25.doc_count, 3);
    let res_new = col.query(None, Some("delta"), None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, false).unwrap();
    assert_eq!(res_new.len(), 1);
    assert_eq!(res_new[0].id, "d3");
}

#[test]
fn test_ivf_cosine_normalized_centroids_and_bq_euclidean_guard() {
    let mut col = NativeCollection::new("ivf_cos".to_string(), 4, Metric::Cosine, None, false).unwrap();
    let mut docs = Vec::new();
    let mut vecs = Vec::new();
    for i in 0..32 {
        docs.push(NativeDocument {
            id: format!("d{}", i),
            namespace: None,
            text: format!("doc {}", i),
            metadata: json!({}),
            sparse_vector: None,
            colbert_tokens: None,
        });
        let angle = (i as f32) * 0.1;
        vecs.push(vec![angle.cos(), angle.sin(), 0.1, 0.0]);
    }
    col.insert(docs, vecs).unwrap();
    col.build_ivf(Some(_sutradb::index::ivf::IVFConfig {
        n_lists: 4,
        nprobe: 2,
        kmeans_iters: 15,
        seed: 42,
    }))
    .unwrap();

    let ivf = col.ivf.as_ref().unwrap();
    assert_eq!(ivf.metric, Metric::Cosine);
    for c in &ivf.centroids {
        let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "Cosine IVF centroid must be unit-normalized, got {}", norm);
    }

    let q = vec![1.0, 0.0, 0.1, 0.0];
    let hits = col.query(Some(&q), None, None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, true).unwrap();
    assert_eq!(hits[0].id, "d0");
    assert!(ivf.search(&[1.0, 0.0], &col.vectors, 5, Metric::Cosine, true, None).is_err());

    // BQ + Euclidean guard: distinct magnitudes in same positive orthant must rank by exact Euclidean distance
    let mut col_euc = NativeCollection::new("bq_euc_guard".to_string(), 4, Metric::Euclidean, None, false).unwrap();
    col_euc
        .insert(
            vec![
                NativeDocument { id: "far".into(), namespace: None, text: "far".into(), metadata: json!({}), sparse_vector: None, colbert_tokens: None },
                NativeDocument { id: "near".into(), namespace: None, text: "near".into(), metadata: json!({}), sparse_vector: None, colbert_tokens: None },
            ],
            vec![vec![100.0, 100.0, 100.0, 100.0], vec![1.0, 1.0, 1.0, 1.0]],
        )
        .unwrap();
    col_euc.enable_bq();
    let euc_hits = col_euc
        .query(Some(&[1.0, 1.0, 1.0, 1.0]), None, None, None, None, None, 2, false, None, 60, 1.0, 1.0, 1.0, 1.0, false)
        .unwrap();
    assert_eq!(euc_hits[0].id, "near");
}

#[test]
fn test_secondary_index_type_separation_and_query_parity() {
    let mut idx = _sutradb::filter::SecondaryMetadataIndex::new();
    let metas = [
        json!({"text": "a", "flag": "true", "code": "1", "arr": ["true", "1"]}),
        json!({"text": "b", "flag": true, "code": 1, "arr": [true, 1.0]}),
        json!({"text": "c", "flag": false, "code": 1.0, "arr": [false, -0.0]}),
        json!({"text": "d", "flag": "false", "code": 0, "arr": ["false", 0]}),
        json!({"text": "e", "status": "null"}),
    ];
    for (i, m) in metas.iter().enumerate() {
        idx.index_document(i as u32, m);
    }

    // 1. Direct SecondaryMetadataIndex exactness and type separation
    let cases = [
        (json!({"flag": true}), vec![1]),
        (json!({"flag": "true"}), vec![0]),
        (json!({"flag": false}), vec![2]),
        (json!({"flag": "false"}), vec![3]),
        (json!({"code": "1"}), vec![0]),
        (json!({"code": 1}), vec![1, 2]),
        (json!({"code": 1.0}), vec![1, 2]),
        (json!({"code": -0.0}), vec![3]),
        (json!({"flag": {"$eq": true}}), vec![1]),
        (json!({"flag": {"$ne": "true"}}), vec![1, 2, 3, 4]),
        (json!({"flag": {"$in": [true]}}), vec![1]),
        (json!({"flag": {"$nin": ["true"]}}), vec![1, 2, 3, 4]),
        (json!({"code": {"$in": [1.0]}}), vec![1, 2]),
        (json!({"code": {"$nin": ["1"]}}), vec![1, 2, 3, 4]),
        (json!({"arr": true}), vec![1]),
        (json!({"arr": "true"}), vec![0]),
        (json!({"arr": 1}), vec![1]),
        (json!({"arr": "1"}), vec![0]),
        (json!({"arr": 0.0}), vec![2, 3]),
    ];

    for (filter, expected) in &cases {
        let (bm, is_exact) = idx.evaluate_filter_checked(Some(filter)).unwrap();
        assert!(is_exact, "Expected exact index eval for {}", filter);
        let got: Vec<u32> = bm.iter().collect();
        assert_eq!(&got, expected, "SecondaryMetadataIndex mismatch for {}", filter);
    }

    // Non-object elements in $and / $or must mark exact = false (Superset)
    let (_, and_non_obj_exact) = idx.evaluate_filter_checked(Some(&json!({"$and": [123]}))).unwrap();
    assert!(!and_non_obj_exact, "$and with non-object element must return Superset");
    let (_, or_non_obj_exact) = idx
        .evaluate_filter_checked(Some(&json!({"$or": [{"flag": true}, 123]})))
        .unwrap();
    assert!(!or_non_obj_exact, "$or with non-object element must return Superset");
    let (_, not_and_non_obj_exact) = idx
        .evaluate_filter_checked(Some(&json!({"$not": {"$and": [123]}})))
        .unwrap();
    assert!(!not_and_non_obj_exact, "$not over $and with non-object must return Superset");

    // Removal must only remove the deleted document's typed key without touching other types
    idx.remove_document(0);
    let (bm_after_del, _) = idx.evaluate_filter_checked(Some(&json!({"flag": true}))).unwrap();
    assert_eq!(bm_after_del.iter().collect::<Vec<_>>(), vec![1]);
    let (bm_str_after_del, _) = idx.evaluate_filter_checked(Some(&json!({"flag": "true"}))).unwrap();
    assert!(bm_str_after_del.is_empty());

    // 2. NativeCollection::query vs FilterEngine bypass parity across all cases
    let mut col = NativeCollection::new("type_parity".to_string(), 2, Metric::Cosine, None, false).unwrap();
    let docs: Vec<NativeDocument> = metas
        .iter()
        .enumerate()
        .map(|(i, m)| NativeDocument {
            id: format!("d{}", i),
            namespace: None,
            text: format!("doc {}", i),
            metadata: m.clone(),
            sparse_vector: None,
            colbert_tokens: None,
        })
        .collect();
    let vecs = vec![vec![1.0, 0.0]; docs.len()];
    col.insert(docs, vecs).unwrap();

    let q = vec![1.0, 0.0];
    for (filter, expected_indices) in &cases {
        let mut direct_ids: Vec<String> = col
            .query(Some(&q), None, None, None, None, Some(filter), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false)
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        direct_ids.sort();

        let bypass_filter = json!({"$and": [filter, {"text": {"$regex": ".*"}}]});
        let mut bypass_ids: Vec<String> = col
            .query(Some(&q), None, None, None, None, Some(&bypass_filter), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false)
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        bypass_ids.sort();

        let expected_ids: Vec<String> = expected_indices.iter().map(|i| format!("d{}", i)).collect();
        assert_eq!(direct_ids, expected_ids, "Collection direct query mismatch for {}", filter);
        assert_eq!(direct_ids, bypass_ids, "Collection direct vs bypass mismatch for {}", filter);
    }
}

#[test]
fn test_native_stats_includes_enable_wal_flag() {
    let tmp = tempdir().unwrap();
    let wal_col = NativeCollection::new(
        "wal_on".to_string(),
        4,
        Metric::Cosine,
        Some(tmp.path().join("wal_on.sutra")),
        true,
    )
    .unwrap();
    assert_eq!(wal_col.stats()["enable_wal"], true);

    let mem_col = NativeCollection::new(
        "wal_off".to_string(),
        4,
        Metric::Cosine,
        None,
        false,
    )
    .unwrap();
    assert_eq!(mem_col.stats()["enable_wal"], false);
}

#[test]
fn test_native_load_rejects_invalid_metric_header() {
    use _sutradb::storage::BinaryStorage;

    let tmp = tempdir().unwrap();
    let bad_path = tmp.path().join("bad_metric.sutra");
    let meta = serde_json::to_vec(&vec![json!({"id": "d1", "text": "x", "metadata": {}})]).unwrap();
    BinaryStorage::write_f32(&bad_path, &[1.0, 0.0], 1, 2, "bogus_metric", &meta).unwrap();

    let mut col = NativeCollection::new("bad_metric".to_string(), 2, Metric::Cosine, None, false).unwrap();
    let err = col.load(Some(&bad_path)).unwrap_err();
    assert!(
        err.to_string().contains("Unsupported metric"),
        "Expected Unsupported metric error, got: {}",
        err
    );
}

#[test]
fn test_wal_replay_ignores_corrupt_oversized_length_without_oom() {
    use std::io::Write;
    use _sutradb::storage::wal::{WalRecord, WriteAheadLog};

    let tmp = tempdir().unwrap();
    let wal_path = tmp.path().join("corrupt_len.wal");
    let mut wal = WriteAheadLog::open(&wal_path).unwrap();
    wal.append(&WalRecord {
        op: "delete".to_string(),
        docs: Vec::new(),
        ids: vec!["d1".to_string()],
    })
    .unwrap();
    drop(wal);

    // Append a corrupt header claiming 500MB payload, with only 10 bytes present
    let mut f = std::fs::OpenOptions::new().append(true).open(&wal_path).unwrap();
    f.write_all(&500_000_000u32.to_le_bytes()).unwrap();
    f.write_all(&0u32.to_le_bytes()).unwrap();
    f.write_all(b"0123456789").unwrap();
    drop(f);

    let mut wal2 = WriteAheadLog::open(&wal_path).unwrap();
    let recs = wal2.replay().unwrap();
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].ids, vec!["d1"]);
}

#[test]
fn test_simd_unaligned_and_sq8_parity() {
    use _sutradb::distance::simd::{dot_product, euclidean_distance_squared};
    use _sutradb::distance::sq8::{sq8_asymmetric_dot_slice, SQ8Vector};

    for &dim in &[0usize, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 128, 131] {
        let a: Vec<f32> = (0..dim).map(|i| ((i * 17 + 3) % 41) as f32 * 0.1 - 2.0).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((i * 29 + 5) % 37) as f32 * 0.1 - 1.5).collect();

        let ref_dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let ref_l2: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();

        assert!(
            (dot_product(&a, &b) - ref_dot).abs() < 1e-3,
            "dot parity mismatch at dim={}",
            dim
        );
        assert!(
            (euclidean_distance_squared(&a, &b) - ref_l2).abs() < 1e-3,
            "l2 parity mismatch at dim={}",
            dim
        );

        let sq = SQ8Vector::quantize(&b);
        let q_sum: f32 = a.iter().sum();
        let ref_sq8: f32 = sq.min_val * q_sum
            + sq.scale
                * a.iter()
                    .zip(sq.data.iter())
                    .map(|(q, &byte)| q * (byte as f32))
                    .sum::<f32>();
        let simd_sq8 = sq8_asymmetric_dot_slice(&sq.data, sq.min_val, sq.scale, &a, q_sum).unwrap();
        assert!(
            (simd_sq8 - ref_sq8).abs() < 1e-3,
            "sq8 parity mismatch at dim={}",
            dim
        );
    }
}

#[test]
fn test_mmap_fp32_ram_eviction_on_quantization() {
    let tmp = tempdir().unwrap();
    let sutra_path = tmp.path().join("evict_test.sutra");
    let dim = 16;
    let n = 64;

    let mut docs = Vec::with_capacity(n);
    let mut vecs = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = vec![0.0f32; dim];
        v[i % dim] = 1.0;
        v[(i + 3) % dim] = 0.25 * ((i / dim + 1) as f32);
        docs.push(NativeDocument {
            id: format!("ev_{}", i),
            namespace: None,
            text: format!("doc {}", i),
            metadata: json!({"i": i}),
            sparse_vector: None,
            colbert_tokens: None,
        });
        vecs.push(normalize_copy(&v));
    }

    let mut col = NativeCollection::new(
        "evict_col".to_string(),
        dim,
        Metric::Cosine,
        Some(sutra_path.clone()),
        false,
    )
    .unwrap();
    col.insert(docs, vecs.clone()).unwrap();
    col.save::<PathBuf>(None).unwrap();

    let mut reloaded = NativeCollection::new(
        "evict_col".to_string(),
        dim,
        Metric::Cosine,
        Some(sutra_path),
        false,
    )
    .unwrap();
    reloaded.load::<PathBuf>(None).unwrap();
    assert!(reloaded.mmap_storage.is_some());
    assert!(!reloaded.vectors_evicted);
    assert_eq!(reloaded.vectors.len(), n);

    reloaded.enable_bq();
    assert!(reloaded.vectors_evicted, "FP32 arena must be evicted when BQ enabled over mmap");
    assert!(reloaded.vectors.is_empty(), "Evicted FP32 arena must hold 0 bytes");
    assert_eq!(reloaded.stats()["vector_memory_bytes"].as_u64().unwrap(), (n * 8) as u64);

    let hits = reloaded
        .query(
            Some(&vecs[0]),
            None,
            None,
            None,
            None,
            None,
            3,
            false,
            None,
            60,
            1.0,
            1.0,
            1.0,
            1.0,
            false,
        )
        .unwrap();
    assert_eq!(hits[0].id, "ev_0");
}

#[test]
fn test_hnsw_filtered_recall_1pct_selectivity() {
    let n = 5_000usize;
    let dim = 32usize;
    let mut col = NativeCollection::new("hnsw_1pct".to_string(), dim, Metric::Cosine, None, false).unwrap();

    let mut docs = Vec::with_capacity(n);
    let mut vecs = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = vec![0.0f32; dim];
        for d in 0..dim {
            let phase = ((i * 13 + d * 29) % 200) as f32 / 100.0 - 1.0;
            v[d] = phase;
        }
        // Make every 100th doc (50 docs total = 1% selectivity) match `rare = true`
        let is_rare = i % 100 == 0;
        if is_rare {
            v[0] += 0.35;
        }
        docs.push(NativeDocument {
            id: format!("d_{}", i),
            namespace: None,
            text: format!("record {}", i),
            metadata: json!({"rare": is_rare, "idx": i}),
            sparse_vector: None,
            colbert_tokens: None,
        });
        vecs.push(normalize_copy(&v));
    }

    col.insert(docs, vecs.clone()).unwrap();
    col.build_hnsw(Some(HNSWConfig {
        m: 16,
        m0: 32,
        ef_construction: 200,
        ef_search: 64,
        ml: 1.0 / (16.0f64).ln(),
        seed: 42,
    }))
    .unwrap();

    let filter = json!({"rare": true});
    let hnsw = col.hnsw.as_ref().unwrap();
    let filter_bm = col.metadata_index.evaluate_filter(Some(&filter)).unwrap();
    assert_eq!(filter_bm.len(), 50, "Expected exact 1% selectivity (50 docs)");

    let mut matched_top10 = 0usize;
    let num_queries = 20usize;
    for q_idx in 0..num_queries {
        let mut q = vec![0.0f32; dim];
        for d in 0..dim {
            q[d] = ((q_idx * 37 + d * 19) % 200) as f32 / 100.0 - 1.0;
        }
        q[0] += 0.4;
        let q_norm = normalize_copy(&q);

        // Exact ground truth among rare docs
        let exact_hits = col
            .query(Some(&q_norm), None, None, None, None, Some(&filter), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false)
            .unwrap();
        let exact_set: std::collections::HashSet<&str> = exact_hits.iter().map(|h| h.id.as_str()).collect();

        // Direct HNSW filtered bitmap search
        let hnsw_hits = hnsw
            .search_with_bitmap(&q_norm, &col.vectors, 10, 64, Metric::Cosine, true, Some(&filter_bm))
            .unwrap();
        assert_eq!(hnsw_hits.len(), 10);
        for (id, _) in &hnsw_hits {
            assert!(filter_bm.contains(*id));
            if exact_set.contains(col.documents[*id as usize].id.as_str()) {
                matched_top10 += 1;
            }
        }

        // Collection-level query with use_ann=true also matches >= 95%
        let col_hits = col
            .query(Some(&q_norm), None, None, None, None, Some(&filter), 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, true)
            .unwrap();
        assert_eq!(col_hits.len(), 10);
        assert!(col_hits.iter().all(|h| h.metadata["rare"] == true));
    }

    let recall = matched_top10 as f64 / (num_queries * 10) as f64;
    assert!(recall >= 0.95, "Expected >= 95% HNSW Recall@10 at 1% selectivity, got {:.2}%", recall * 100.0);
}

#[test]
fn test_wal_batch_group_commit_and_partial_tail_recovery() {
    use std::io::Write;
    use _sutradb::storage::wal::{WalRecord, WriteAheadLog};

    let tmp = tempdir().unwrap();
    let wal_path = tmp.path().join("group_commit.wal");
    let mut wal = WriteAheadLog::open_with_sync(&wal_path, false).unwrap();

    let batch = vec![
        WalRecord {
            op: "insert".to_string(),
            docs: vec![json!({
                "id": "b1",
                "text": "batched one",
                "metadata": {"k": 1},
                "vector": [1.0, 0.0, 0.5, 0.25]
            })],
            ids: Vec::new(),
        },
        WalRecord {
            op: "insert".to_string(),
            docs: vec![json!({
                "id": "b2",
                "text": "batched two",
                "metadata": {"k": 2},
                "vector": [0.0, 1.0, 0.25, 0.5]
            })],
            ids: Vec::new(),
        },
        WalRecord {
            op: "delete".to_string(),
            docs: Vec::new(),
            ids: vec!["b1".to_string()],
        },
    ];

    wal.append_batch(&batch).unwrap();
    // Mix legacy JSON record into the same WAL stream to confirm dual decoding
    wal.append_value(&json!({"op": "update", "doc": {"id": "b2", "text": "json updated"}}))
        .unwrap();
    wal.flush().unwrap();
    drop(wal);

    let valid_len = std::fs::metadata(&wal_path).unwrap().len();

    // Simulate crash leaving partial trailing frame after valid batch
    let mut raw = std::fs::OpenOptions::new().append(true).open(&wal_path).unwrap();
    raw.write_all(&64u32.to_le_bytes()).unwrap();
    raw.write_all(&0xDEAD_BEEFu32.to_le_bytes()).unwrap();
    raw.write_all(&[0xABu8; 19]).unwrap();
    drop(raw);

    assert!(std::fs::metadata(&wal_path).unwrap().len() > valid_len);

    let mut wal_recover = WriteAheadLog::open(&wal_path).unwrap();
    let replayed = wal_recover.replay().unwrap();
    assert_eq!(replayed.len(), 4);
    assert_eq!(replayed[0].op, "insert");
    assert_eq!(replayed[0].docs[0]["id"], "b1");
    assert_eq!(replayed[1].docs[0]["id"], "b2");
    assert_eq!(replayed[2].op, "delete");
    assert_eq!(replayed[2].ids, vec!["b1"]);
    assert_eq!(replayed[3].op, "update");
    assert_eq!(replayed[3].docs[0]["text"], "json updated");
    assert_eq!(
        std::fs::metadata(&wal_path).unwrap().len(),
        valid_len,
        "Partial trailing bytes after batch write must be truncated back to valid frame boundary"
    );
}

#[test]
fn test_instant_restart_preserves_hnsw_bq_pq() {
    use _sutradb::distance::simd::normalize_copy;
    use _sutradb::storage::index_sidecar_path;

    let tmp = tempdir().unwrap();
    let col_path = tmp.path().join("instant_restart.sutra");
    let dim = 32;
    let n = 128;

    let mut col = NativeCollection::new(
        "instant_restart".to_string(),
        dim,
        Metric::Cosine,
        Some(col_path.clone()),
        false,
    )
    .unwrap();

    let mut docs = Vec::with_capacity(n);
    let mut vecs = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = vec![0.0f32; dim];
        for d in 0..dim {
            v[d] = (((i * 31 + d * 17) % 100) as f32 / 50.0) - 1.0;
        }
        docs.push(NativeDocument {
            id: format!("doc_{:03}", i),
            namespace: None,
            text: format!("instant restart document {}", i),
            metadata: json!({"idx": i, "bucket": i % 4}),
            sparse_vector: None,
            colbert_tokens: None,
        });
        vecs.push(normalize_copy(&v));
    }

    col.insert(docs, vecs.clone()).unwrap();
    col.build_hnsw(Some(HNSWConfig {
        m: 16,
        m0: 32,
        ef_construction: 64,
        ef_search: 48,
        ..HNSWConfig::default()
    }))
    .unwrap();
    col.enable_bq();
    col.train_pq(Some(4), Some(128)).unwrap();
    col.enable_pq().unwrap();

    let mut q_raw = vec![0.0f32; dim];
    for d in 0..dim {
        q_raw[d] = (((7 * 43 + d * 29) % 100) as f32 / 50.0) - 1.0;
    }
    let query_vec = normalize_copy(&q_raw);

    let hnsw_before = col
        .query(Some(&query_vec), None, None, None, None, None, 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, true)
        .unwrap();
    let pq_before = col
        .query(Some(&query_vec), None, None, None, None, None, 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false)
        .unwrap();

    col.save::<PathBuf>(None).unwrap();
    assert!(index_sidecar_path(&col_path).exists());

    // Cold reload into a brand-new NativeCollection without rebuilding HNSW, BQ, or PQ
    let mut reloaded = NativeCollection::new(
        "instant_restart".to_string(),
        dim,
        Metric::Cosine,
        Some(col_path.clone()),
        false,
    )
    .unwrap();
    reloaded.load::<PathBuf>(None).unwrap();

    assert!(reloaded.hnsw.is_some());
    assert!(reloaded.use_bq);
    assert!(reloaded.bq_centroid.is_some());
    assert_eq!(reloaded.bq_vectors.len(), n);
    assert!(reloaded.use_pq);
    assert!(reloaded.pq_codebook.is_some());
    assert_eq!(reloaded.pq_vectors.len(), n);

    let hnsw_after = reloaded
        .query(Some(&query_vec), None, None, None, None, None, 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, true)
        .unwrap();
    let pq_after = reloaded
        .query(Some(&query_vec), None, None, None, None, None, 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false)
        .unwrap();

    let hnsw_ids_before: Vec<&str> = hnsw_before.iter().map(|r| r.id.as_str()).collect();
    let hnsw_ids_after: Vec<&str> = hnsw_after.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(hnsw_ids_before, hnsw_ids_after);

    let pq_ids_before: Vec<&str> = pq_before.iter().map(|r| r.id.as_str()).collect();
    let pq_ids_after: Vec<&str> = pq_after.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(pq_ids_before, pq_ids_after);

    // Also verify BQ-only save -> load top-10 exact parity without enable_bq()
    let bq_path = tmp.path().join("bq_only.sutra");
    let mut bq_col = NativeCollection::new(
        "bq_only".to_string(),
        dim,
        Metric::Cosine,
        Some(bq_path.clone()),
        false,
    )
    .unwrap();
    let bq_docs: Vec<NativeDocument> = (0..n)
        .map(|i| NativeDocument {
            id: format!("bq_{:03}", i),
            namespace: None,
            text: format!("bq {}", i),
            metadata: json!({}),
            sparse_vector: None,
            colbert_tokens: None,
        })
        .collect();
    bq_col.insert(bq_docs, vecs).unwrap();
    bq_col.enable_bq();
    let bq_before = bq_col
        .query(Some(&query_vec), None, None, None, None, None, 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false)
        .unwrap();
    bq_col.save::<PathBuf>(None).unwrap();

    let mut bq_reloaded = NativeCollection::new(
        "bq_only".to_string(),
        dim,
        Metric::Cosine,
        Some(bq_path),
        false,
    )
    .unwrap();
    bq_reloaded.load::<PathBuf>(None).unwrap();
    assert!(bq_reloaded.use_bq);
    let bq_after = bq_reloaded
        .query(Some(&query_vec), None, None, None, None, None, 10, false, None, 60, 1.0, 1.0, 1.0, 1.0, false)
        .unwrap();
    let bq_ids_before: Vec<&str> = bq_before.iter().map(|r| r.id.as_str()).collect();
    let bq_ids_after: Vec<&str> = bq_after.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(bq_ids_before, bq_ids_after);
}

#[test]
fn test_sidecar_ivf_colbert_corruption_and_legacy_fallback() {
    use _sutradb::index::ivf::IVFConfig;
    use _sutradb::storage::index_sidecar_path;

    let tmp = tempdir().unwrap();
    let col_path = tmp.path().join("ivf_colbert.sutra");
    let dim = 8;

    let mut col = NativeCollection::new(
        "ivf_colbert".to_string(),
        dim,
        Metric::Cosine,
        Some(col_path.clone()),
        false,
    )
    .unwrap();

    let mut docs = Vec::new();
    let mut vecs = Vec::new();
    for i in 0..32 {
        let mut v = vec![0.1f32; dim];
        v[i % dim] = 1.0;
        let tok = v.clone();
        docs.push(NativeDocument {
            id: format!("ic_{}", i),
            namespace: None,
            text: format!("text {}", i),
            metadata: json!({}),
            sparse_vector: None,
            colbert_tokens: Some(vec![tok]),
        });
        vecs.push(v);
    }
    col.insert(docs, vecs).unwrap();
    col.build_ivf(Some(IVFConfig {
        n_lists: 4,
        nprobe: 2,
        kmeans_iters: 10,
        seed: 7,
    }))
    .unwrap();
    col.colbert_index.train_centroids(4, 10).unwrap();

    let q = vec![1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
    let ivf_before = col
        .query(Some(&q), None, None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, true)
        .unwrap();
    col.save::<PathBuf>(None).unwrap();

    let idx_file = index_sidecar_path(&col_path);
    assert!(idx_file.exists());

    // Reload cleanly -> IVF and ColBERT centroids preserved
    let mut reloaded = NativeCollection::new(
        "ivf_colbert".to_string(),
        dim,
        Metric::Cosine,
        Some(col_path.clone()),
        false,
    )
    .unwrap();
    reloaded.load::<PathBuf>(None).unwrap();
    assert!(reloaded.ivf.is_some());
    assert!(reloaded.colbert_index.is_trained);
    assert!(reloaded.colbert_index.centroids.is_some());
    let ivf_after = reloaded
        .query(Some(&q), None, None, None, None, None, 5, false, None, 60, 1.0, 1.0, 1.0, 1.0, true)
        .unwrap();
    assert_eq!(
        ivf_before.iter().map(|r| &r.id).collect::<Vec<_>>(),
        ivf_after.iter().map(|r| &r.id).collect::<Vec<_>>()
    );

    // Corrupt a payload byte inside .sutra.idx -> must error on section CRC32 mismatch
    let mut raw_idx = std::fs::read(&idx_file).unwrap();
    let last = raw_idx.len() - 1;
    raw_idx[last] ^= 0xFF;
    std::fs::write(&idx_file, &raw_idx).unwrap();
    let mut corrupt_loader = NativeCollection::new(
        "ivf_colbert".to_string(),
        dim,
        Metric::Cosine,
        Some(col_path.clone()),
        false,
    )
    .unwrap();
    assert!(corrupt_loader.load::<PathBuf>(None).is_err());

    // Remove .sutra.idx -> graceful legacy V2/V3/V4 load fallback still loads all docs
    std::fs::remove_file(&idx_file).unwrap();
    let mut legacy_loader = NativeCollection::new(
        "ivf_colbert".to_string(),
        dim,
        Metric::Cosine,
        Some(col_path),
        false,
    )
    .unwrap();
    let loaded_count = legacy_loader.load::<PathBuf>(None).unwrap();
    assert_eq!(loaded_count, 32);
    assert!(legacy_loader.ivf.is_none());
}

#[test]
fn test_parallel_hnsw_batch_recall() {
    use _sutradb::distance::simd::normalize_copy;
    use _sutradb::index::hnsw::{HNSWConfig, HNSWIndex};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let n = 2000;
    let dim = 64;
    let mut rng = StdRng::seed_from_u64(42);
    // Clustered embeddings (standard ANN geometry; modulo-periodic sequences lack local neighborhood structure)
    let centers: Vec<Vec<f32>> = (0..20)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect();
    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        let c = &centers[i % centers.len()];
        let v: Vec<f32> = c.iter().map(|&x| x + rng.gen_range(-0.35f32..0.35f32)).collect();
        vecs.push(normalize_copy(&v));
    }

    let cfg = HNSWConfig {
        m: 32,
        m0: 64,
        ef_construction: 200,
        ef_search: 128,
        ml: 1.0 / (32.0f64).ln(),
        seed: 42,
    };

    let mut seq_index = HNSWIndex::new(cfg.clone());
    for (i, v) in vecs.iter().enumerate() {
        seq_index.insert(i as u32, v, &vecs, Metric::Cosine, true).unwrap();
    }

    let mut par_index = HNSWIndex::new(cfg);
    let doc_ids: Vec<u32> = (0..n as u32).collect();
    par_index
        .build_batch(&doc_ids, &vecs, Metric::Cosine, true)
        .unwrap();
    assert_eq!(par_index.active_count(), n);

    let mut seq_hits = 0usize;
    let mut par_hits = 0usize;
    let num_queries = 40;
    for q_idx in 0..num_queries {
        let c = &centers[q_idx % centers.len()];
        let q: Vec<f32> = c.iter().map(|&x| x + rng.gen_range(-0.35f32..0.35f32)).collect();
        let q_norm = normalize_copy(&q);

        let mut exact: Vec<(u32, f32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let s: f32 = q_norm.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                (i as u32, s)
            })
            .collect();
        exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let truth: std::collections::HashSet<u32> = exact.iter().take(10).map(|&(id, _)| id).collect();

        for (id, _) in seq_index
            .search(&q_norm, &vecs, 10, 128, Metric::Cosine, true, None)
            .unwrap()
        {
            if truth.contains(&id) {
                seq_hits += 1;
            }
        }

        for (id, _) in par_index
            .search(&q_norm, &vecs, 10, 128, Metric::Cosine, true, None)
            .unwrap()
        {
            if truth.contains(&id) {
                par_hits += 1;
            }
        }
    }

    let seq_recall = seq_hits as f64 / (num_queries * 10) as f64;
    let par_recall = par_hits as f64 / (num_queries * 10) as f64;
    eprintln!(
        "HNSW Recall@10 (2K x 64d): sequential={:.2}%, parallel={:.2}%",
        seq_recall * 100.0,
        par_recall * 100.0
    );
    assert!(
        par_recall >= 0.98,
        "Expected >= 98% parallel HNSW Recall@10, got {:.2}% (seq={:.2}%)",
        par_recall * 100.0,
        seq_recall * 100.0
    );
}
