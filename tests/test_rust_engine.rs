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
    let mut es_config = AnalyzerConfig::default();
    es_config.language = "spanish".to_string();
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
        vec![0.7071, 0.7071, 0.0, 0.0],
        vec![0.0, 0.0, 0.7071, 0.7071],
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
    let q_dense = vec![0.7071, 0.7071, 0.0, 0.0];
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
        assert_eq!(vec, &original_vec, "Document vector must not be wiped with zeros on WAL replay");

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
    let ep_level = hnsw.node_levels[&ep];
    assert_eq!(ep_level, max_lvl);

    // Verify all layers up to max_lvl have entry point in their layers map
    for l in 0..=max_lvl {
        assert!(
            hnsw.layers[l].contains_key(&ep),
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
    assert_eq!(col.bm25.analyzer.config.split_identifiers, true);
    assert_eq!(col.bm25.analyzer.config.cjk_bigrams, true);

    // Save and load
    col.save::<PathBuf>(None).unwrap();
    col.load::<PathBuf>(None).unwrap();

    // Verify analyzer config was preserved across load()
    assert_eq!(col.bm25.analyzer.config.language, "german");
    assert_eq!(col.bm25.analyzer.config.split_identifiers, true);
    assert_eq!(col.bm25.analyzer.config.cjk_bigrams, true);
}
