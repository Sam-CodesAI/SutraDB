"""
Unit and integration tests for NLP linguistic analyzer, multi-lingual stemming,
stopwords, identifier splitting, CJK n-grams, and hardware SIMD auto-dispatch.
"""

import pytest
from sutradb import SutraDB, Document, Analyzer, AnalyzerConfig, PorterStemmer


def test_porter_stemmer_direct():
    assert PorterStemmer.stem("connecting") == "connect"
    assert PorterStemmer.stem("connections") == "connect"
    assert PorterStemmer.stem("microservices") == "microservic"
    assert PorterStemmer.stem("deploying") == "deploi"
    assert PorterStemmer.stem("relational") == "relat"
    assert PorterStemmer.stem("optimization") == "optim"
    assert PorterStemmer.stem("happy") == "happi"
    assert PorterStemmer.stem("cats") == "cat"


def test_analyzer_tokenization_and_stopwords():
    analyzer = Analyzer(AnalyzerConfig(language="english", stopwords=True, stemmer=True))
    tokens = analyzer.analyze("Deploying containerized microservices and connection pools in Kubernetes")

    # Stopwords pruned
    assert "and" not in tokens
    assert "in" not in tokens

    # Stems present
    assert "deploi" in tokens
    assert "microservic" in tokens
    assert "connect" in tokens
    assert "pool" in tokens
    assert "kubernet" in tokens


def test_analyzer_code_identifier_splitting():
    analyzer = Analyzer(AnalyzerConfig(split_identifiers=True, stemmer=True, stopwords=True))
    tokens = analyzer.analyze("getUserProfile and fetch_jwt_token")

    assert "get" in tokens
    assert "user" in tokens
    assert "profil" in tokens
    assert "fetch" in tokens
    assert "jwt" in tokens
    assert "token" in tokens


def test_analyzer_cjk_bigrams():
    analyzer = Analyzer(AnalyzerConfig(cjk_bigrams=True))
    tokens = analyzer.analyze("自然语言处理")

    assert "自然" in tokens
    assert "语言" in tokens
    assert "处理" in tokens


def test_collection_nlp_analyzer_integration():
    db = SutraDB()
    col = db.create_collection(
        name="nlp_kb",
        dimension=4,
        analyzer={"language": "english", "stemmer": True, "stopwords": True, "split_identifiers": True}
    )

    col.insert([
        Document(
            id="doc_k8s",
            vector=[1.0, 0.0, 0.0, 0.0],
            text="Deploying containerized microservices on Kubernetes clusters",
            metadata={"domain": "cloud"}
        ),
        Document(
            id="doc_db",
            vector=[0.0, 1.0, 0.0, 0.0],
            text="PostgreSQL connection pooling and query optimization",
            metadata={"domain": "database"}
        )
    ])

    # Query with uninflected form "deploy microservice"
    results = col.query(text="deploy microservice", top_k=2)
    assert len(results) >= 1
    assert results[0].id == "doc_k8s"

    # Query with code identifier "queryOptimization"
    results_code = col.query(text="queryOptimization", top_k=2)
    assert len(results_code) >= 1
    assert results_code[0].id == "doc_db"


def test_native_simd_detection_and_batch_distances():
    try:
        from sutradb._sutradb import detect_simd, batch_vector_distances, analyze_text
    except ImportError:
        pytest.skip("Rust extension _sutradb not built in test environment")

    simd_str = detect_simd()
    assert isinstance(simd_str, str)
    assert len(simd_str) > 0
    print(f"Detected Host CPU SIMD: {simd_str}")

    tokens = analyze_text("Deploying microservices in Kubernetes", "english")
    assert "deploi" in tokens
    assert "microservic" in tokens
    assert "in" not in tokens

    # Test batch distances
    q = [1.0, 0.0, 0.0, 0.0]
    vecs = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.707, 0.707, 0.0, 0.0]]
    dists = batch_vector_distances(q, vecs, "cosine", True)
    assert len(dists) == 3
    assert abs(dists[0] - 1.0) < 1e-4
    assert abs(dists[1] - 0.0) < 1e-4
    assert dists[2] > 0.5
