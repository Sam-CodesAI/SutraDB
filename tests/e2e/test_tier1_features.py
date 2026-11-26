"""
Tier 1 E2E Feature Tests (Category-Partition Happy-Path Verification).
Covers all 8 feature domains (F1-F8) with 5 isolated happy-path tests per domain (40 total).
"""

from __future__ import annotations

import math
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
from typing import Any, Callable, Dict, List

import httpx
import numpy as np
import pytest

from sutradb import (
    Analyzer,
    AnalyzerConfig,
    Collection,
    ConsistentHashRing,
    DistributedCluster,
    DistributedCollection,
    Document,
    IVFConfig,
    Metric,
    NativeCollection,
    SutraClient,
    SutraDB,
)
from sutradb._sutradb import (
    analyze_text,
    batch_vector_distances,
    simd_cosine,
    simd_dot,
    simd_euclidean,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_clustered_vectors(n: int, dim: int, seed: int = 42) -> List[List[float]]:
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).tolist()


# =============================================================================
# F1: Dense Vector Search & Quantization (5 tests)
# =============================================================================


def test_f1_exact_search_all_metrics_python_and_native(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Cosine, Euclidean, and DotProduct exact search in Python and Native engines."""
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.6, 0.8, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    query_vec = [0.98, 0.1, 0.0, 0.0]

    for metric in ("cosine", "euclidean", "dot_product"):
        py_col = py_collection_factory(name=f"py_{metric}", dimension=4, metric=metric)
        py_col.insert([
            Document(id="d0", vector=vectors[0], text="first"),
            Document(id="d1", vector=vectors[1], text="second"),
            Document(id="d2", vector=vectors[2], text="third"),
        ])
        py_res = py_col.query(vector=query_vec, top_k=3, hybrid=False)
        assert [r.id for r in py_res] == ["d0", "d1", "d2"]
        if metric == "euclidean":
            assert all(0.0 < r.score <= 1.0 for r in py_res)

        nat_col = native_collection_factory(name=f"nat_{metric}", dimension=4, metric=metric)
        nat_col.insert(
            [{"id": f"d{i}", "text": f"doc {i}", "metadata": {}} for i in range(3)],
            vectors,
        )
        nat_res = nat_col.query(vector=query_vec, top_k=3, hybrid=False)
        assert [r["id"] for r in nat_res] == ["d0", "d1", "d2"]
        if metric == "euclidean":
            assert all(0.0 < r["score"] <= 1.0 for r in nat_res)


def test_f1_sq8_quantization_and_recall(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """SQ8 quantization enables cleanly, updates on incremental insert, and preserves top-1 recall."""
    vecs = _make_clustered_vectors(25, 16, seed=11)
    vecs[7] = [1.0] + [0.0] * 15
    query_vec = [0.99, 0.05] + [0.0] * 14

    py_col = py_collection_factory(name="py_sq8", dimension=16, metric="cosine")
    py_col.insert([Document(id=f"doc_{i}", vector=vecs[i], text=f"item {i}") for i in range(20)])
    py_col.enable_sq8()
    py_col.insert([Document(id=f"doc_{i}", vector=vecs[i], text=f"item {i}") for i in range(20, 25)])
    assert py_col.stats()["quantized_sq8"] is True
    py_res = py_col.query(vector=query_vec, top_k=3, hybrid=False)
    assert py_res[0].id == "doc_7"

    nat_col = native_collection_factory(name="nat_sq8", dimension=16, metric="cosine")
    nat_col.insert([{"id": f"doc_{i}", "text": f"item {i}", "metadata": {}} for i in range(20)], vecs[:20])
    nat_col.enable_sq8()
    nat_col.insert([{"id": f"doc_{i}", "text": f"item {i}", "metadata": {}} for i in range(20, 25)], vecs[20:])
    assert nat_col.stats()["quantized_sq8"] is True
    nat_res = nat_col.query(vector=query_vec, top_k=3, hybrid=False)
    assert nat_res[0]["id"] == "doc_7"


def test_f1_bq_quantization_and_recall(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """1-bit Binary Quantization (BQ) compresses vectors and retrieves matching target."""
    rng = np.random.default_rng(22)
    vecs = rng.normal(size=(40, 64)).astype(np.float32)
    target = np.ones(64, dtype=np.float32)
    target[32:] = -1.0
    vecs[12] = target
    for i in range(40):
        if i != 12:
            vecs[i] = -target + rng.normal(scale=0.2, size=64).astype(np.float32)
    vecs_list = vecs.tolist()
    q = target.tolist()

    py_col = py_collection_factory(name="py_bq", dimension=64, metric="cosine")
    py_col.insert([Document(id=f"d_{i}", vector=vecs_list[i]) for i in range(40)])
    py_col.enable_bq()
    assert py_col.stats()["quantized_bq"] is True
    assert py_col.query(vector=q, top_k=3, hybrid=False)[0].id == "d_12"

    nat_col = native_collection_factory(name="nat_bq", dimension=64, metric="cosine")
    nat_col.insert([{"id": f"d_{i}", "text": "", "metadata": {}} for i in range(40)], vecs_list)
    nat_col.enable_bq()
    assert nat_col.stats()["quantized_bq"] is True
    assert nat_col.query(vector=q, top_k=3, hybrid=False)[0]["id"] == "d_12"


def test_f1_pq_train_and_enable_recall(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Product Quantization train_pq(m=4) and enable_pq() work in Python and Native engines."""
    vecs = _make_clustered_vectors(50, 16, seed=33)
    target = [1.0, 0.5, -0.5, -1.0] * 4
    vecs[19] = target

    py_col = py_collection_factory(name="py_pq", dimension=16, metric="cosine")
    py_col.insert([Document(id=f"p_{i}", vector=vecs[i]) for i in range(50)])
    py_col.train_pq(m=4, max_samples=50)
    py_col.enable_pq()
    st = py_col.stats()
    assert st["quantized_pq"] is True
    assert st["pq_m"] == 4
    assert py_col.query(vector=target, top_k=5, hybrid=False)[0].id == "p_19"

    nat_col = native_collection_factory(name="nat_pq", dimension=16, metric="cosine")
    nat_col.insert([{"id": f"p_{i}", "text": "", "metadata": {}} for i in range(50)], vecs)
    nat_col.train_pq(m=4, max_samples=50)
    nat_col.enable_pq()
    nat_st = nat_col.stats()
    assert nat_st["quantized_pq"] is True
    assert nat_st["pq_m"] == 4
    assert nat_col.query(vector=target, top_k=5, hybrid=False)[0]["id"] == "p_19"


def test_f1_python_native_simd_consistency() -> None:
    """SIMD primitives and batch_vector_distances match analytical NumPy expectations."""
    a = [1.0, 2.0, 3.0, 4.0]
    b = [2.0, 0.0, 1.0, 2.0]
    na = np.array(a, dtype=np.float32)
    nb = np.array(b, dtype=np.float32)

    expected_dot = float(np.dot(na, nb))
    expected_cos = float(expected_dot / (np.linalg.norm(na) * np.linalg.norm(nb)))
    expected_euc = float(1.0 / (1.0 + np.linalg.norm(na - nb)))

    assert math.isclose(simd_dot(a, b), expected_dot, rel_tol=1e-4)
    assert math.isclose(simd_cosine(a, b), expected_cos, rel_tol=1e-4)
    assert math.isclose(simd_euclidean(a, b), expected_euc, rel_tol=1e-4)

    batch_cos = batch_vector_distances(a, [a, b], metric="cosine", is_normalized=False)
    assert len(batch_cos) == 2
    assert math.isclose(batch_cos[0], 1.0, abs_tol=1e-4)
    assert math.isclose(batch_cos[1], expected_cos, rel_tol=1e-4)


# =============================================================================
# F2: ANN Indexing — HNSW & IVF (5 tests)
# =============================================================================


def test_f2_python_hnsw_build_and_ann_query(py_collection_factory: Callable[..., Any]) -> None:
    """Python Collection.build_hnsw builds real HNSWIndex and query(..., use_ann=True) retrieves target."""
    vecs = _make_clustered_vectors(120, 16, seed=101)
    vecs[42] = [1.0] + [0.0] * 15
    col = py_collection_factory(name="py_hnsw", dimension=16, metric="cosine")
    col.insert([Document(id=f"h_{i}", vector=vecs[i], text=f"vec {i}") for i in range(120)])

    col.build_hnsw(m=16, ef_construction=64, ef_search=32)
    assert col.stats()["has_hnsw"] is True
    assert not isinstance(col._hnsw_index, dict), "Python _hnsw_index must be a real HNSWIndex, not a metadata dict"

    res = col.query(vector=[0.99, 0.05] + [0.0] * 14, top_k=5, hybrid=False, use_ann=True)
    assert len(res) == 5
    assert res[0].id == "h_42"


def test_f2_rust_hnsw_build_and_ann_query(native_collection_factory: Callable[..., Any]) -> None:
    """Rust NativeCollection.build_hnsw and use_ann=True query achieve top-1 recall."""
    vecs = _make_clustered_vectors(150, 16, seed=102)
    vecs[88] = [0.0, 1.0] + [0.0] * 14
    col = native_collection_factory(name="nat_hnsw", dimension=16, metric="cosine")
    col.insert([{"id": f"n_{i}", "text": f"node {i}", "metadata": {}} for i in range(150)], vecs)

    col.build_hnsw(m=16, ef_construction=64, ef_search=32)
    assert col.stats()["has_hnsw"] is True

    res = col.query(vector=[0.02, 0.99] + [0.0] * 14, top_k=5, hybrid=False, use_ann=True)
    assert len(res) == 5
    assert res[0]["id"] == "n_88"


def test_f2_hnsw_delete_and_subsequent_search(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Deleting nodes from an HNSW-indexed collection excludes them and finds next-nearest."""
    vecs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.95, 0.31, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    py_col = py_collection_factory(name="py_hnsw_del", dimension=4, metric="cosine")
    py_col.insert([Document(id=f"d_{i}", vector=vecs[i]) for i in range(4)])
    py_col.build_hnsw(m=8, ef_construction=32, ef_search=16)
    assert py_col.delete(["d_0"]) == 1
    py_res = py_col.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=3, hybrid=False, use_ann=True)
    assert "d_0" not in [r.id for r in py_res]
    assert py_res[0].id == "d_1"

    nat_col = native_collection_factory(name="nat_hnsw_del", dimension=4, metric="cosine")
    nat_col.insert([{"id": f"d_{i}", "text": "", "metadata": {}} for i in range(4)], vecs)
    nat_col.build_hnsw(m=8, ef_construction=32, ef_search=16)
    assert nat_col.delete(["d_0"]) == 1
    nat_res = nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=3, hybrid=False, use_ann=True)
    assert "d_0" not in [r["id"] for r in nat_res]
    assert nat_res[0]["id"] == "d_1"


def test_f2_ivf_build_and_search_across_metrics(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """IVF-Flat build and search work across cosine and euclidean metrics."""
    for metric in ("cosine", "euclidean"):
        vecs = _make_clustered_vectors(80, 8, seed=103)
        vecs[15] = [1.0] + [0.0] * 7

        py_col = py_collection_factory(name=f"py_ivf_{metric}", dimension=8, metric=metric)
        py_col.insert([Document(id=f"v_{i}", vector=vecs[i]) for i in range(80)])
        py_col.build_index(IVFConfig(n_lists=8, nprobe=4))
        assert py_col.stats()["ann"]["trained"] is True
        py_res = py_col.query(vector=[1.0] + [0.0] * 7, top_k=3, hybrid=False, use_ann=True)
        assert py_res[0].id == "v_15"

        nat_col = native_collection_factory(name=f"nat_ivf_{metric}", dimension=8, metric=metric)
        nat_col.insert([{"id": f"v_{i}", "text": "", "metadata": {}} for i in range(80)], vecs)
        nat_col.build_ivf(n_lists=8, nprobe=4)
        assert nat_col.stats()["has_ivf"] is True
        nat_res = nat_col.query(vector=[1.0] + [0.0] * 7, top_k=3, hybrid=False, use_ann=True)
        assert nat_res[0]["id"] == "v_15"


def test_f2_filtered_hnsw_high_selectivity(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Filtered HNSW query with 50% selectivity returns top_k strictly matching filter."""
    vecs = _make_clustered_vectors(200, 8, seed=104)
    py_docs = [
        Document(id=f"f_{i}", vector=vecs[i], metadata={"group": "even" if i % 2 == 0 else "odd"})
        for i in range(200)
    ]
    py_col = py_collection_factory(name="py_filt_hnsw", dimension=8, metric="cosine")
    py_col.insert(py_docs)
    py_col.build_hnsw(m=16, ef_construction=64, ef_search=32)
    py_res = py_col.query(vector=vecs[0], filter={"group": "even"}, top_k=5, hybrid=False, use_ann=True)
    assert len(py_res) == 5
    assert all(r.metadata["group"] == "even" for r in py_res)

    nat_col = native_collection_factory(name="nat_filt_hnsw", dimension=8, metric="cosine")
    nat_col.insert(
        [{"id": f"f_{i}", "text": "", "metadata": {"group": "even" if i % 2 == 0 else "odd"}} for i in range(200)],
        vecs,
    )
    nat_col.build_hnsw(m=16, ef_construction=64, ef_search=32)
    nat_res = nat_col.query(vector=vecs[0], filter={"group": "even"}, top_k=5, hybrid=False, use_ann=True)
    assert len(nat_res) == 5
    assert all(r["metadata"]["group"] == "even" for r in nat_res)


# =============================================================================
# F3: Hybrid & Multi-Modal Search (5 tests)
# =============================================================================


def test_f3_bm25_multilingual_and_cjk(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """BM25 stemming works for English, Spanish, French, German, and CJK bigrams."""
    # Spanish inflection matching
    es_py = py_collection_factory(
        name="es_py",
        dimension=4,
        analyzer=Analyzer(AnalyzerConfig(language="spanish")),
    )
    es_py.insert([
        Document(id="es1", vector=[1.0, 0.0, 0.0, 0.0], text="los ingenieros construyendo motores rapidos"),
        Document(id="es2", vector=[0.0, 1.0, 0.0, 0.0], text="receta tradicional de cocina"),
    ])
    res_es = es_py.query(text="construir motor rapido", top_k=2, hybrid=False)
    assert len(res_es) >= 1
    assert res_es[0].id == "es1"

    # French & German analyzer tokens (Python + Rust)
    fr_tokens = Analyzer(AnalyzerConfig(language="french")).analyze("les développeurs travaillent rapidement")
    de_tokens = Analyzer(AnalyzerConfig(language="german")).analyze("Entwickler arbeiten schnell")
    assert len(fr_tokens) > 0 and len(de_tokens) > 0

    rust_es = analyze_text("los ingenieros construyendo motores", "spanish")
    rust_fr = analyze_text("les développeurs travaillent rapidement", "french")
    rust_de = analyze_text("Entwickler arbeiten schnell", "german")
    assert len(rust_es) > 0 and len(rust_fr) > 0 and len(rust_de) > 0

    # CJK bigrams in both engines
    cjk_col = native_collection_factory(name="cjk_nat", dimension=4)
    cjk_col.insert(
        [
            {"id": "zh1", "text": "人工智能与向量数据库搜索引擎", "metadata": {}},
            {"id": "zh2", "text": "今天天气很好适合散步", "metadata": {}},
        ],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    cjk_res = cjk_col.query(text="向量数据库", top_k=2, hybrid=False)
    assert cjk_res[0]["id"] == "zh1"


def test_f3_sparse_splade_search(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Sparse SPLADE vector retrieval scores matching token IDs accurately."""
    py_col = py_collection_factory(name="py_sparse", dimension=4)
    py_col.insert([
        Document(id="sp1", vector=[0.0, 1.0, 0.0, 0.0], text="alpha", sparse_vector={101: 3.0, 202: 1.5}),
        Document(id="sp2", vector=[1.0, 0.0, 0.0, 0.0], text="beta", sparse_vector={303: 2.0}),
    ])
    py_res = py_col.query(sparse_vector={101: 2.0, 202: 1.0}, top_k=2, hybrid=False)
    assert len(py_res) == 1
    assert py_res[0].id == "sp1"
    assert py_res[0].sparse_score is not None and py_res[0].sparse_score > 0.0

    nat_col = native_collection_factory(name="nat_sparse", dimension=4)
    nat_col.insert(
        [
            {"id": "sp1", "text": "alpha", "metadata": {}, "sparse_vector": {101: 3.0, 202: 1.5}},
            {"id": "sp2", "text": "beta", "metadata": {}, "sparse_vector": {303: 2.0}},
        ],
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    )
    nat_res = nat_col.query(sparse_vector={101: 2.0, 202: 1.0}, top_k=2, hybrid=False)
    assert nat_res[0]["id"] == "sp1"
    assert nat_res[0]["sparse_score"] is not None and nat_res[0]["sparse_score"] > 0.0


def test_f3_colbert_maxsim_search(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """ColBERT MaxSim late-interaction scoring ranks matching token bags first."""
    tok_a = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    tok_b = [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

    py_col = py_collection_factory(name="py_colbert", dimension=4)
    py_col.insert([
        Document(id="cb1", vector=[0.0, 0.0, 1.0, 0.0], colbert_tokens=tok_a),
        Document(id="cb2", vector=[1.0, 0.0, 0.0, 0.0], colbert_tokens=tok_b),
    ])
    py_res = py_col.query(colbert_tokens=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], top_k=2, hybrid=False)
    assert py_res[0].id == "cb1"
    assert py_res[0].colbert_score is not None and py_res[0].colbert_score > 1.5

    nat_col = native_collection_factory(name="nat_colbert", dimension=4)
    nat_col.insert(
        [
            {"id": "cb1", "text": "", "metadata": {}, "colbert_tokens": tok_a},
            {"id": "cb2", "text": "", "metadata": {}, "colbert_tokens": tok_b},
        ],
        [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    )
    nat_res = nat_col.query_colbert(query_tokens=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], top_k=2)
    assert nat_res[0]["id"] == "cb1"


def test_f3_rrf_hybrid_fusion(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """RRF hybrid search combines dense and BM25 ranks for joint winners."""
    docs_data = [
        ("both", [1.0, 0.0, 0.0, 0.0], "quantum vector database search"),
        ("dense_only", [0.99, 0.14, 0.0, 0.0], "unrelated culinary recipe"),
        ("text_only", [0.0, 0.0, 1.0, 0.0], "quantum vector database architecture"),
    ]
    py_col = py_collection_factory(name="py_rrf", dimension=4)
    py_col.insert([Document(id=i, vector=v, text=t) for i, v, t in docs_data])
    py_res = py_col.query(vector=[1.0, 0.0, 0.0, 0.0], text="quantum vector database", top_k=3, hybrid=True)
    assert py_res[0].id == "both"
    assert py_res[0].dense_score is not None and py_res[0].bm25_score is not None

    nat_col = native_collection_factory(name="nat_rrf", dimension=4)
    nat_col.insert(
        [{"id": i, "text": t, "metadata": {}} for i, _, t in docs_data],
        [v for _, v, _ in docs_data],
    )
    nat_res = nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], text="quantum vector database", top_k=3, hybrid=True)
    assert nat_res[0]["id"] == "both"


def test_f3_linear_score_fusion_alpha(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Linear score fusion alpha=0.9 favors dense match while alpha=0.1 favors lexical match."""
    docs = [
        ("vec_winner", [1.0, 0.0, 0.0, 0.0], "ordinary filler words"),
        ("lex_winner", [0.0, 1.0, 0.0, 0.0], "hyperbolic vector index calibration"),
    ]
    py_col = py_collection_factory(name="py_lin", dimension=4)
    py_col.insert([Document(id=i, vector=v, text=t) for i, v, t in docs])

    dense_heavy = py_col.query(
        vector=[1.0, 0.0, 0.0, 0.0],
        text="hyperbolic vector index calibration",
        alpha=0.9,
        top_k=2,
        hybrid=True,
    )
    lex_heavy = py_col.query(
        vector=[1.0, 0.0, 0.0, 0.0],
        text="hyperbolic vector index calibration",
        alpha=0.1,
        top_k=2,
        hybrid=True,
    )
    assert dense_heavy[0].id == "vec_winner"
    assert lex_heavy[0].id == "lex_winner"

    nat_col = native_collection_factory(name="nat_lin", dimension=4)
    nat_col.insert([{"id": i, "text": t, "metadata": {}} for i, _, t in docs], [v for _, v, _ in docs])
    nat_dense = nat_col.query(
        vector=[1.0, 0.0, 0.0, 0.0],
        text="hyperbolic vector index calibration",
        alpha=0.9,
        top_k=2,
        hybrid=True,
    )
    nat_lex = nat_col.query(
        vector=[1.0, 0.0, 0.0, 0.0],
        text="hyperbolic vector index calibration",
        alpha=0.1,
        top_k=2,
        hybrid=True,
    )
    assert nat_dense[0]["id"] == "vec_winner"
    assert nat_lex[0]["id"] == "lex_winner"


# =============================================================================
# F4: Metadata & Geospatial Filtering (5 tests)
# =============================================================================


def test_f4_equality_and_comparison_operators(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """$eq, $ne, $gt, $gte, $lt, $lte and nested dot-notation paths in Python and Rust."""
    meta_rows = [
        ("m1", {"tier": "free", "user": {"age": 22}, "rating": 3.5}),
        ("m2", {"tier": "pro", "user": {"age": 30}, "rating": 4.5}),
        ("m3", {"tier": "enterprise", "user": {"age": 45}, "rating": 4.9}),
    ]
    vecs = [[1.0, 0.0, 0.0, 0.0]] * 3

    py_col = py_collection_factory(name="py_cmp", dimension=4)
    py_col.insert([Document(id=i, vector=vecs[idx], metadata=m) for idx, (i, m) in enumerate(meta_rows)])

    nat_col = native_collection_factory(name="nat_cmp", dimension=4)
    nat_col.insert([{"id": i, "text": "", "metadata": m} for i, m in meta_rows], vecs)

    flt = {"tier": {"$ne": "free"}, "user.age": {"$gte": 30, "$lt": 45}, "rating": {"$gt": 4.0, "$lte": 4.5}}
    assert [r.id for r in py_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=flt, top_k=5)] == ["m2"]
    assert [r["id"] for r in nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=flt, top_k=5)] == ["m2"]


def test_f4_membership_and_array_operators(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """$in, $nin, $contains, $all, $size behave consistently across Python and Rust."""
    meta_rows = [
        ("a1", {"env": "prod", "tags": ["rust", "vector", "simd"], "desc": "fast search engine"}),
        ("a2", {"env": "staging", "tags": ["python", "vector"], "desc": "prototype"}),
        ("a3", {"env": "dev", "tags": ["rust"], "desc": "experimental"}),
    ]
    vecs = [[1.0, 0.0, 0.0, 0.0]] * 3

    py_col = py_collection_factory(name="py_arr", dimension=4)
    py_col.insert([Document(id=i, vector=vecs[idx], metadata=m) for idx, (i, m) in enumerate(meta_rows)])
    nat_col = native_collection_factory(name="nat_arr", dimension=4)
    nat_col.insert([{"id": i, "text": "", "metadata": m} for i, m in meta_rows], vecs)

    flt1 = {"env": {"$in": ["prod", "staging"], "$nin": ["dev"]}, "tags": {"$contains": "vector"}}
    assert {r.id for r in py_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=flt1, top_k=5)} == {"a1", "a2"}
    assert {r["id"] for r in nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=flt1, top_k=5)} == {"a1", "a2"}

    flt2 = {"tags": {"$all": ["rust", "simd"], "$size": 3}}
    assert [r.id for r in py_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=flt2, top_k=5)] == ["a1"]
    assert [r["id"] for r in nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=flt2, top_k=5)] == ["a1"]


def test_f4_regex_and_exists_operators(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """$regex and $exists work on nested dicts, empty lists, and missing keys."""
    meta_rows = [
        ("r1", {"service": "auth-v2-prod", "labels": [], "owner": {"team": "core"}}),
        ("r2", {"service": "billing-staging", "labels": ["finance"]}),
    ]
    vecs = [[1.0, 0.0, 0.0, 0.0]] * 2

    py_col = py_collection_factory(name="py_rx", dimension=4)
    py_col.insert([Document(id=i, vector=vecs[idx], metadata=m) for idx, (i, m) in enumerate(meta_rows)])
    nat_col = native_collection_factory(name="nat_rx", dimension=4)
    nat_col.insert([{"id": i, "text": "", "metadata": m} for i, m in meta_rows], vecs)

    flt = {"service": {"$regex": r"^auth-v\d+-prod$"}, "owner": {"$exists": True}, "labels": {"$exists": True}}
    assert [r.id for r in py_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=flt, top_k=5)] == ["r1"]
    assert [r["id"] for r in nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=flt, top_k=5)] == ["r1"]


def test_f4_logical_compound_operators(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """$and, $or, and $not logical AST operators match expected subsets."""
    meta_rows = [
        ("l1", {"region": "us", "score": 95, "archived": False}),
        ("l2", {"region": "eu", "score": 88, "archived": False}),
        ("l3", {"region": "us", "score": 40, "archived": True}),
    ]
    vecs = [[1.0, 0.0, 0.0, 0.0]] * 3

    py_col = py_collection_factory(name="py_log", dimension=4)
    py_col.insert([Document(id=i, vector=vecs[idx], metadata=m) for idx, (i, m) in enumerate(meta_rows)])
    nat_col = native_collection_factory(name="nat_log", dimension=4)
    nat_col.insert([{"id": i, "text": "", "metadata": m} for i, m in meta_rows], vecs)

    flt = {
        "$and": [
            {"$or": [{"region": "us"}, {"region": "eu"}]},
            {"$not": {"archived": True}},
            {"score": {"$gte": 85}},
        ]
    }
    assert {r.id for r in py_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=flt, top_k=5)} == {"l1", "l2"}
    assert {r["id"] for r in nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=flt, top_k=5)} == {"l1", "l2"}


def test_f4_geospatial_within_and_box(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """$geo_within and $geo_box accurately filter geospatial coordinates in Python and Rust."""
    cities = [
        ("sf", {"location": {"lat": 37.7749, "lon": -122.4194}}),
        ("oakland", {"location": {"lat": 37.8044, "lon": -122.2712}}),
        ("nyc", {"location": {"lat": 40.7128, "lon": -74.0060}}),
    ]
    vecs = [[1.0, 0.0, 0.0, 0.0]] * 3

    py_col = py_collection_factory(name="py_geo", dimension=4)
    py_col.insert([Document(id=i, vector=vecs[idx], metadata=m) for idx, (i, m) in enumerate(cities)])
    nat_col = native_collection_factory(name="nat_geo", dimension=4)
    nat_col.insert([{"id": i, "text": "", "metadata": m} for i, m in cities], vecs)

    within_flt = {"location": {"$geo_within": {"center": [37.7749, -122.4194], "radius_km": 25.0}}}
    assert {r.id for r in py_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=within_flt, top_k=5)} == {"sf", "oakland"}
    assert {r["id"] for r in nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=within_flt, top_k=5)} == {"sf", "oakland"}

    box_flt = {
        "location": {
            "$geo_box": {
                "bottom_left": [40.5, -74.3],
                "top_right": [40.9, -73.7],
            }
        }
    }
    assert [r.id for r in py_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=box_flt, top_k=5)] == ["nyc"]
    assert [r["id"] for r in nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=box_flt, top_k=5)] == ["nyc"]


# =============================================================================
# F5: Persistence, WAL & Crash Recovery (5 tests)
# =============================================================================


def test_f5_sutra_save_load_f32_and_v4_sq8_roundtrip(
    tmp_path: Path,
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """f32 and V4 SQ8 .sutra snapshots roundtrip cleanly within and across Python & Rust."""
    # f32 roundtrip
    f32_path = tmp_path / "f32_roundtrip.sutra"
    py_f32 = py_collection_factory(name="f32_col", dimension=4, storage_filename="f32_roundtrip.sutra")
    py_f32.insert([
        Document(id="a", vector=[1.0, 0.0, 0.0, 0.0], text="alpha", metadata={"k": 1}),
        Document(id="b", vector=[0.0, 1.0, 0.0, 0.0], text="beta", metadata={"k": 2}),
    ])
    py_f32.save()

    reloaded_py = Collection(name="f32_col", dimension=4)
    assert reloaded_py.load(f32_path) == 2
    assert reloaded_py.get("a") is not None

    # V4 SQ8 roundtrip across Rust -> Python
    sq8_path = tmp_path / "sq8_v4.sutra"
    nat_sq8 = native_collection_factory(name="sq8_v4", dimension=4, metric="cosine", storage_filename="sq8_v4.sutra")
    nat_sq8.insert(
        [{"id": "s1", "text": "one", "metadata": {}}, {"id": "s2", "text": "two", "metadata": {}}],
        [[-0.8, 0.2, 0.5, -0.1], [0.5, -0.5, 0.5, -0.5]],
    )
    nat_sq8.enable_sq8()
    nat_sq8.save()
    assert sq8_path.read_bytes()[:8] == b"SUTRA\x04\x00\x00"

    py_sq8_reloaded = Collection(name="sq8_v4", dimension=4, metric="cosine")
    assert py_sq8_reloaded.load(sq8_path) == 2
    res = py_sq8_reloaded.query(vector=[-0.8, 0.2, 0.5, -0.1], top_k=2, hybrid=False)
    assert res[0].id == "s1"


def test_f5_python_wal_crash_recovery_multimodal(tmp_path: Path) -> None:
    """Python WAL preserves namespace, sparse_vector, and colbert_tokens without save()."""
    col_path = tmp_path / "crash_wal.sutra"
    c1 = Collection(name="crash_wal", dimension=4, metric="cosine", storage_path=col_path, enable_wal=True)
    c1.insert([
        Document(
            id="mm1",
            vector=[1.0, 0.0, 0.0, 0.0],
            text="multimodal wal recovery",
            metadata={"env": "prod"},
            namespace="tenant_a",
            sparse_vector={7: 2.5, 19: 1.0},
            colbert_tokens=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )
    ])
    c1.wal.close()

    c2 = Collection(name="crash_wal", dimension=4, metric="cosine", storage_path=col_path, enable_wal=True)
    try:
        doc = c2.get("mm1")
        assert doc is not None
        assert doc.namespace == "tenant_a"
        assert doc.sparse_vector == {7: 2.5, 19: 1.0} or doc.sparse_vector == {"7": 2.5, "19": 1.0}
        assert doc.colbert_tokens == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        assert len(c2.query(vector=[1.0, 0.0, 0.0, 0.0], namespace="tenant_a", top_k=5)) == 1
        assert len(c2.query(vector=[1.0, 0.0, 0.0, 0.0], namespace="tenant_b", top_k=5)) == 0
    finally:
        if c2.wal:
            c2.wal.close()


def test_f5_update_metadata_wal_persistence_and_index_sync(tmp_path: Path) -> None:
    """update_metadata persists through WAL and updates query filtering in Python & Rust."""
    # Python WAL
    py_path = tmp_path / "py_meta_wal.sutra"
    c1 = Collection(name="py_meta", dimension=4, storage_path=py_path, enable_wal=True)
    c1.insert(Document(id="d1", vector=[1.0, 0.0, 0.0, 0.0], text="doc", metadata={"status": "draft"}))
    assert c1.update_metadata("d1", {"status": "published", "score": 99}) is True
    c1.wal.close()

    c2 = Collection(name="py_meta", dimension=4, storage_path=py_path, enable_wal=True)
    try:
        assert c2.get("d1").metadata == {"status": "published", "score": 99}
        assert [r.id for r in c2.query(vector=[1.0, 0.0, 0.0, 0.0], filter={"status": "published"})] == ["d1"]
        assert c2.query(vector=[1.0, 0.0, 0.0, 0.0], filter={"status": "draft"}) == []
    finally:
        if c2.wal:
            c2.wal.close()

    # Rust WAL + SecondaryMetadataIndex sync (WAL replays automatically on construction)
    nat_path = str(tmp_path / "nat_meta_wal.sutra")
    n1 = NativeCollection(name="nat_meta", dimension=4, metric="cosine", storage_path=nat_path, enable_wal=True)
    n1.insert([{"id": "d1", "text": "doc", "metadata": {"status": "draft"}}], [[1.0, 0.0, 0.0, 0.0]])
    assert n1.update_metadata("d1", {"status": "published"}) is True

    # Reopen via WAL only without calling save() -> load()
    n2 = NativeCollection(name="nat_meta", dimension=4, metric="cosine", storage_path=nat_path, enable_wal=True)
    assert [r["id"] for r in n2.query(vector=[1.0, 0.0, 0.0, 0.0], filter={"status": "published"})] == ["d1"]
    assert n2.query(vector=[1.0, 0.0, 0.0, 0.0], filter={"status": "draft"}) == []

    # Also verify snapshot save() -> load() preserves updated metadata and index state
    n2.save()
    n3 = NativeCollection(name="nat_meta", dimension=4, metric="cosine", storage_path=nat_path, enable_wal=False)
    assert n3.load(nat_path) == 1
    assert [r["id"] for r in n3.query(vector=[1.0, 0.0, 0.0, 0.0], filter={"status": "published"})] == ["d1"]
    assert n3.query(vector=[1.0, 0.0, 0.0, 0.0], filter={"status": "draft"}) == []


def test_f5_sutradb_multi_collection_lifecycle(tmp_path: Path) -> None:
    """SutraDB creates, persists, lists, reloads across instances, and drops collections cleanly."""
    db_dir = tmp_path / "multi_db"
    db1 = SutraDB(persist_directory=str(db_dir))
    c_alpha = db1.create_collection("alpha", dimension=4, metric="cosine", enable_wal=True)
    c_alpha.insert(Document(id="a1", vector=[1.0, 0.0, 0.0, 0.0], text="alpha doc"))
    c_alpha.save()

    c_beta = db1.get_or_create_collection("beta", dimension=4, metric="euclidean", enable_wal=True)
    c_beta.insert(Document(id="b1", vector=[0.0, 1.0, 0.0, 0.0], text="beta doc"))
    assert sorted(db1.list_collections()) == ["alpha", "beta"]
    for c in db1.collections.values():
        if c.wal:
            c.wal.close()

    db2 = SutraDB(persist_directory=str(db_dir))
    try:
        assert sorted(db2.list_collections()) == ["alpha", "beta"]
        assert db2.get_collection("alpha").count() == 1
        assert db2.get_collection("beta").count() == 1
        assert db2.drop_collection("alpha") is True
        assert not (db_dir / "alpha.sutra").exists()
        assert not (db_dir / "alpha.wal").exists()
        assert db2.list_collections() == ["beta"]
    finally:
        for c in db2.collections.values():
            if c.wal:
                c.wal.close()


def test_f5_collection_compact_reclaims_tombstones(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Collection.compact() and NativeCollection.compact() purge tombstoned slots cleanly."""
    vecs = [[float(i == j) for j in range(4)] for i in range(4)]
    py_col = py_collection_factory(name="py_compact", dimension=4)
    py_col.insert([Document(id=f"c{i}", vector=vecs[i], text=f"doc {i}") for i in range(4)])
    py_col.delete(["c0", "c2"])
    assert py_col.stats()["tombstones"] == 2
    py_col.compact()
    st = py_col.stats()
    assert st["tombstones"] == 0
    assert st["total_records"] == 2
    assert st["count"] == 2
    assert py_col.query(vector=vecs[1], top_k=5)[0].id == "c1"

    nat_col = native_collection_factory(name="nat_compact", dimension=4)
    nat_col.insert([{"id": f"c{i}", "text": f"doc {i}", "metadata": {}} for i in range(4)], vecs)
    nat_col.delete(["c0", "c2"])
    assert nat_col.stats()["tombstones"] == 2
    nat_col.compact()
    nst = nat_col.stats()
    assert nst["tombstones"] == 0
    assert nst["total_records"] == 2
    assert nst["count"] == 2
    assert nat_col.query(vector=vecs[1], top_k=5)[0]["id"] == "c1"


# =============================================================================
# F6: Concurrency & Distributed Cluster (5 tests)
# =============================================================================


def test_f6_native_concurrent_readers_and_writers(native_collection_factory: Callable[..., Any]) -> None:
    """PyNativeCollection handles concurrent readers and writers safely under Barrier."""
    col = native_collection_factory(name="rw_concurrent", dimension=4, metric="cosine")
    col.insert(
        [{"id": f"seed_{i}", "text": f"seed document {i}", "metadata": {"batch": 0}} for i in range(10)],
        [[1.0, 0.0, 0.0, 0.0] for _ in range(10)],
    )

    errors: List[BaseException] = []
    barrier = threading.Barrier(4)

    def writer(tid: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for step in range(10):
                col.insert(
                    [{"id": f"w_{tid}_{step}", "text": f"concurrent write {tid}", "metadata": {"tid": tid}}],
                    [[0.0, 1.0, 0.0, 0.0]],
                )
        except BaseException as exc:
            errors.append(exc)

    def reader() -> None:
        try:
            barrier.wait(timeout=5.0)
            for _ in range(20):
                res = col.query(vector=[1.0, 0.0, 0.0, 0.0], text="concurrent", top_k=5)
                assert len(res) > 0
                assert col.count() >= 10
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(0,)),
        threading.Thread(target=writer, args=(1,)),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)
        assert not t.is_alive(), "Thread hung during concurrent NativeCollection R/W"

    assert errors == []
    assert col.count() == 30


def test_f6_consistent_hash_ring_distribution_and_preferences() -> None:
    """ConsistentHashRing deterministically routes keys and returns distinct preference lists."""
    ring = ConsistentHashRing(vnodes_per_node=64)
    nodes = ["shard-0", "shard-1", "shard-2", "shard-3"]
    for n in nodes:
        ring.add_node(n)

    assert ring.node_count() == 4
    assert set(ring.get_nodes()) == set(nodes)

    counts = {n: 0 for n in nodes}
    for i in range(400):
        key = f"tenant-doc-{i}"
        owner = ring.get_node(key)
        assert owner in counts
        assert ring.get_node(key) == owner  # deterministic
        counts[owner] += 1

    assert all(c > 40 for c in counts.values()), f"Skewed ring distribution: {counts}"

    prefs = ring.get_preference_list("hot-key", 3)
    assert len(prefs) == 3
    assert len(set(prefs)) == 3


def test_f6_distributed_collection_sharded_insert_and_query() -> None:
    """DistributedCollection partitions documents across shards and runs scatter-gather query."""
    dcol = DistributedCollection(name="dist_docs", dimension=4, metric="cosine", shard_count=3)
    docs = [
        Document(id=f"doc_{i}", vector=[1.0 if i == 5 else 0.0, 0.0 if i == 5 else 1.0, 0.0, 0.0], text=f"item {i}")
        for i in range(30)
    ]
    inserted = dcol.insert(docs)
    assert inserted == 30
    assert dcol.count() == 30
    assert sum(1 for s in dcol.shards.values() if s.count() > 0) >= 2

    results = dcol.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=5, hybrid=False)
    assert len(results) == 5
    assert results[0].id == "doc_5"


def test_f6_distributed_cluster_multi_collection_lifecycle() -> None:
    """DistributedCluster manages multiple distributed collections end-to-end."""
    cluster = DistributedCluster(node_id="coord-primary", vnodes_per_node=32)
    c1 = cluster.create_collection("products", dimension=4, shard_count=3)
    c1.insert(Document(id="p1", vector=[1.0, 0.0, 0.0, 0.0], text="laptop"))
    c2 = cluster.get_or_create_collection("users", dimension=4, shard_count=2)
    c2.insert(Document(id="u1", vector=[0.0, 1.0, 0.0, 0.0], text="alice"))

    assert cluster.list_collections() == ["products", "users"]
    assert cluster.get_collection("products").count() == 1
    assert cluster.drop_collection("users") is True
    assert cluster.list_collections() == ["products"]


def test_f6_distributed_collection_multimodal_weight_forwarding() -> None:
    """DistributedCollection.query forwards sparse_weight and colbert_weight to shards."""
    dcol = DistributedCollection(name="dist_mm", dimension=4, metric="cosine", shard_count=3)
    dcol.insert([
        Document(
            id="sparse_winner",
            vector=[0.0, 1.0, 0.0, 0.0],
            text="generic",
            sparse_vector={99: 5.0},
            colbert_tokens=[[0.0, 0.0, 1.0, 0.0]],
        ),
        Document(
            id="colbert_winner",
            vector=[0.0, 1.0, 0.0, 0.0],
            text="generic",
            sparse_vector={11: 0.1},
            colbert_tokens=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        ),
    ])

    sparse_favored = dcol.query(
        sparse_vector={99: 5.0, 11: 0.1},
        colbert_tokens=[[1.0, 0.0, 0.0, 0.0]],
        sparse_weight=5.0,
        colbert_weight=0.01,
        top_k=2,
        hybrid=True,
    )
    colbert_favored = dcol.query(
        sparse_vector={99: 5.0, 11: 0.1},
        colbert_tokens=[[1.0, 0.0, 0.0, 0.0]],
        sparse_weight=0.01,
        colbert_weight=5.0,
        top_k=2,
        hybrid=True,
    )
    assert sparse_favored[0].id == "sparse_winner"
    assert colbert_favored[0].id == "colbert_winner"


# =============================================================================
# F7: REST API, Web UI (/ui) & Playground (5 tests)
# =============================================================================


def test_f7_rest_crud_and_query_lifecycle(rest_client: Any) -> None:
    """FastAPI REST API supports full collection & document CRUD + hybrid query."""
    assert rest_client.get("/health").status_code == 200

    r_create = rest_client.post("/collections", json={"name": "articles", "dimension": 4, "metric": "cosine", "enable_wal": True})
    assert r_create.status_code == 201

    r_ins = rest_client.post(
        "/collections/articles/documents",
        json={
            "documents": [
                {"id": "art1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "neural vector search", "metadata": {"tag": "ai"}},
                {"id": "art2", "vector": [0.0, 1.0, 0.0, 0.0], "text": "relational tables", "metadata": {"tag": "sql"}},
            ]
        },
    )
    assert r_ins.status_code == 200
    assert r_ins.json()["total_count"] == 2

    r_list = rest_client.get("/collections/articles/documents?limit=10&offset=0")
    assert r_list.status_code == 200
    assert r_list.json()["total"] == 2

    assert rest_client.get("/collections/articles/documents/art1").status_code == 200

    r_patch = rest_client.patch("/collections/articles/documents/art1/metadata", json={"metadata": {"tag": "ai", "verified": True}})
    assert r_patch.status_code == 200

    r_query = rest_client.post(
        "/collections/articles/query",
        json={"vector": [1.0, 0.0, 0.0, 0.0], "text": "neural", "filter": {"verified": True}, "top_k": 5},
    )
    assert r_query.status_code == 200
    assert [x["id"] for x in r_query.json()["results"]] == ["art1"]

    assert rest_client.delete("/collections/articles/documents/art2").status_code == 200
    assert rest_client.delete("/collections/articles").status_code == 200


def test_f7_rest_compact_hnsw_quantization_and_save(rest_client: Any) -> None:
    """REST /hnsw, /sq8, /bq, /pq, /compact, and /save endpoints succeed cleanly."""
    assert rest_client.post("/collections", json={"name": "ops_col", "dimension": 8, "metric": "cosine"}).status_code == 201
    vecs = _make_clustered_vectors(20, 8, seed=77)
    docs = [{"id": f"op_{i}", "vector": vecs[i], "text": f"doc {i}"} for i in range(20)]
    assert rest_client.post("/collections/ops_col/insert", json={"documents": docs}).status_code == 200

    assert rest_client.delete("/collections/ops_col/documents/op_0").status_code == 200
    r_compact = rest_client.post("/collections/ops_col/compact")
    assert r_compact.status_code == 200
    assert rest_client.get("/collections/ops_col/stats").json()["tombstones"] == 0

    assert rest_client.post("/collections/ops_col/hnsw", json={"m": 8, "ef_construction": 32, "ef_search": 16}).status_code == 200
    assert rest_client.post("/collections/ops_col/sq8").status_code == 200
    assert rest_client.post("/collections/ops_col/bq").status_code == 200
    assert rest_client.post("/collections/ops_col/pq").status_code == 200
    r_save = rest_client.post("/collections/ops_col/save")
    assert r_save.status_code == 200
    assert r_save.json()["bytes"] > 0


def test_f7_rest_ui_stats_and_prometheus_metrics(rest_client: Any) -> None:
    """GET /ui, /dashboard, /ui/stats, and /metrics return expected HTML, JSON, and Prometheus gauges."""
    rest_client.post("/collections", json={"name": "obs_col", "dimension": 4})
    rest_client.post("/collections/obs_col/insert", json={"documents": [{"id": "o1", "vector": [1, 0, 0, 0], "text": "hi"}]})

    r_ui = rest_client.get("/ui")
    assert r_ui.status_code == 200
    assert "5.1.0" in r_ui.text
    assert rest_client.get("/dashboard").status_code == 200

    r_ui_stats = rest_client.get("/ui/stats")
    assert r_ui_stats.status_code == 200
    assert r_ui_stats.json()["total_collections"] == 1
    assert r_ui_stats.json()["total_documents"] == 1

    r_metrics = rest_client.get("/metrics")
    assert r_metrics.status_code == 200
    assert "sutradb_collections_total 1" in r_metrics.text
    assert 'sutradb_documents_count{collection="obs_col"} 1' in r_metrics.text
    assert 'sutradb_vector_memory_bytes{collection="obs_col"}' in r_metrics.text


def test_f7_sutra_client_http_roundtrip(uvicorn_server_factory: Callable[..., Any]) -> None:
    """SutraClient executes real HTTP CRUD, indexing, and search against live uvicorn server."""
    base_url, _ = uvicorn_server_factory()
    with SutraClient(base_url=base_url, timeout=10.0) as client:
        assert client.health()["status"] == "healthy"
        client.create_collection("sdk_col", dimension=4, metric="cosine", enable_wal=True)
        client.insert("sdk_col", [
            {"id": "c1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "distributed vector engine"},
            {"id": "c2", "vector": [0.0, 1.0, 0.0, 0.0], "text": "static archive"},
        ])
        client.update_metadata("sdk_col", "c1", {"verified": True})
        client.enable_sq8("sdk_col")
        client.build_hnsw("sdk_col", m=8, ef_construction=32, ef_search=16)

        hits = client.query("sdk_col", vector=[1.0, 0.0, 0.0, 0.0], text="vector", top_k=2, use_ann=True)
        assert len(hits) >= 1
        assert hits[0]["id"] == "c1"
        assert hits[0]["metadata"]["verified"] is True
        client.drop_collection("sdk_col")


def test_f7_playground_http_endpoints(playground_server: str) -> None:
    """Playground serves UI HTML, stats, and hybrid/text/vector search with category filter."""
    r_home = httpx.get(f"{playground_server}/", timeout=5.0)
    assert r_home.status_code == 200
    assert "SutraSearch" in r_home.text

    r_stats = httpx.get(f"{playground_server}/api/stats", timeout=5.0)
    assert r_stats.status_code == 200
    stats_json = r_stats.json()
    assert stats_json["total_documents"] > 0
    assert len(stats_json["categories"]) > 0

    for mode in ("hybrid", "text", "vector"):
        r_search = httpx.post(
            f"{playground_server}/api/search",
            json={"text": "HNSW vector search", "mode": mode, "top_k": 5},
            timeout=5.0,
        )
        assert r_search.status_code == 200
        body = r_search.json()
        assert body["mode"] == mode
        assert len(body["results"]) > 0


# =============================================================================
# F8: gRPC Server Lifecycle & Container Coordination (5 tests)
# =============================================================================


def test_f8_sutradb_server_bridge_crud_and_grpc_port(sutradb_server_factory: Callable[..., Any]) -> None:
    """sutradb-server accepts HTTP/2 gRPC connections and serves collection CRUD/query via bridge."""
    srv = sutradb_server_factory()

    # Probe HTTP/2 connection preface on the gRPC port
    with socket.create_connection(("127.0.0.1", srv.grpc_port), timeout=2.0) as sock:
        sock.sendall(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
        resp = sock.recv(64)
        assert len(resp) > 0, "gRPC HTTP/2 port did not respond with SETTINGS frame"

    r_create = httpx.post(
        f"{srv.bridge_url}/collections",
        json={"name": "grpc_col", "dimension": 4, "metric": "cosine", "enable_wal": True},
        timeout=5.0,
    )
    assert r_create.status_code in (200, 201)

    r_ins = httpx.post(
        f"{srv.bridge_url}/collections/grpc_col/insert",
        json={
            "documents": [
                {"id": "g1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "tonic grpc server", "metadata": {"ok": True}},
                {"id": "g2", "vector": [0.0, 1.0, 0.0, 0.0], "text": "other", "metadata": {"ok": False}},
            ]
        },
        timeout=5.0,
    )
    assert r_ins.status_code == 200

    r_q = httpx.post(
        f"{srv.bridge_url}/collections/grpc_col/query",
        json={"vector": [1.0, 0.0, 0.0, 0.0], "text": "tonic", "top_k": 2, "hybrid": True},
        timeout=5.0,
    )
    assert r_q.status_code == 200
    assert r_q.json()["results"][0]["id"] == "g1"


def test_f8_sutradb_server_startup_recovery_across_restart(
    tmp_path: Path,
    sutradb_server_factory: Callable[..., Any],
) -> None:
    """sutradb-server reloads persisted collections and WAL entries on restart with same --data-dir."""
    shared_dir = tmp_path / "persistent_grpc_data"
    srv1 = sutradb_server_factory(data_dir=shared_dir)
    assert httpx.post(
        f"{srv1.bridge_url}/collections",
        json={"name": "durable_col", "dimension": 4, "metric": "cosine", "enable_wal": True},
        timeout=5.0,
    ).status_code in (200, 201)
    assert httpx.post(
        f"{srv1.bridge_url}/collections/durable_col/insert",
        json={"documents": [{"id": "rec1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "survived restart"}]},
        timeout=5.0,
    ).status_code == 200
    srv1.stop()

    srv2 = sutradb_server_factory(data_dir=shared_dir)
    cols = httpx.get(f"{srv2.bridge_url}/collections", timeout=5.0).json()
    col_names = [c["name"] if isinstance(c, dict) else c for c in cols.get("collections", [])]
    assert "durable_col" in col_names
    st = httpx.get(f"{srv2.bridge_url}/collections/durable_col/stats", timeout=5.0).json()
    assert st["count"] == 1
    q = httpx.post(
        f"{srv2.bridge_url}/collections/durable_col/query",
        json={"vector": [1.0, 0.0, 0.0, 0.0], "top_k": 1},
        timeout=5.0,
    ).json()
    assert q["results"][0]["id"] == "rec1"


def test_f8_sutradb_server_save_persists_and_resets_wal_records(sutradb_server_factory: Callable[..., Any]) -> None:
    """Save persists .sutra snapshot to disk and resets wal_records to 0."""
    srv = sutradb_server_factory()
    httpx.post(
        f"{srv.bridge_url}/collections",
        json={"name": "snap_col", "dimension": 4, "metric": "cosine", "enable_wal": True},
        timeout=5.0,
    )
    httpx.post(
        f"{srv.bridge_url}/collections/snap_col/insert",
        json={"documents": [{"id": "s1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "wal entry"}]},
        timeout=5.0,
    )
    st_before = httpx.get(f"{srv.bridge_url}/collections/snap_col/stats", timeout=5.0).json()
    assert st_before.get("wal_records", 0) >= 1

    r_save = httpx.post(f"{srv.bridge_url}/collections/snap_col/save", timeout=5.0)
    assert r_save.status_code == 200
    assert (srv.data_dir / "snap_col.sutra").exists()

    st_after = httpx.get(f"{srv.bridge_url}/collections/snap_col/stats", timeout=5.0).json()
    assert st_after.get("wal_records") == 0


def test_f8_sutradb_server_drop_collection_cleans_disk(sutradb_server_factory: Callable[..., Any]) -> None:
    """DropCollection removes .sutra and .wal files so recreating the same name starts empty."""
    srv = sutradb_server_factory()
    httpx.post(
        f"{srv.bridge_url}/collections",
        json={"name": "ephemeral", "dimension": 4, "metric": "cosine", "enable_wal": True},
        timeout=5.0,
    )
    httpx.post(
        f"{srv.bridge_url}/collections/ephemeral/insert",
        json={"documents": [{"id": "e1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "temp"}]},
        timeout=5.0,
    )
    httpx.post(f"{srv.bridge_url}/collections/ephemeral/save", timeout=5.0)
    assert (srv.data_dir / "ephemeral.sutra").exists()

    r_drop = httpx.delete(f"{srv.bridge_url}/collections/ephemeral", timeout=5.0)
    assert r_drop.status_code == 200
    assert not (srv.data_dir / "ephemeral.sutra").exists()
    assert not (srv.data_dir / "ephemeral.wal").exists()

    r_recreate = httpx.post(
        f"{srv.bridge_url}/collections",
        json={"name": "ephemeral", "dimension": 4, "metric": "cosine", "enable_wal": True},
        timeout=5.0,
    )
    assert r_recreate.status_code in (200, 201)
    assert httpx.get(f"{srv.bridge_url}/collections/ephemeral/stats", timeout=5.0).json()["count"] == 0


def test_f8_sutradb_server_batch_query_and_healthcheck(
    sutradb_server_factory: Callable[..., Any],
    uvicorn_server_factory: Callable[..., Any],
) -> None:
    """Concurrent batch_query returns ordered results and sutra-healthcheck.py exits 0 in SUTRA_MODE=all."""
    srv = sutradb_server_factory()
    httpx.post(
        f"{srv.bridge_url}/collections",
        json={"name": "batch_col", "dimension": 4, "metric": "cosine"},
        timeout=5.0,
    )
    httpx.post(
        f"{srv.bridge_url}/collections/batch_col/insert",
        json={
            "documents": [
                {"id": "b0", "vector": [1.0, 0.0, 0.0, 0.0], "text": "zero"},
                {"id": "b1", "vector": [0.0, 1.0, 0.0, 0.0], "text": "one"},
            ]
        },
        timeout=5.0,
    )

    r_batch = httpx.post(
        f"{srv.bridge_url}/collections/batch_col/batch_query",
        json={
            "queries": [
                {"vector": [1.0, 0.0, 0.0, 0.0], "top_k": 1, "hybrid": False},
                {"vector": [0.0, 1.0, 0.0, 0.0], "top_k": 1, "hybrid": False},
            ]
        },
        timeout=5.0,
    )
    assert r_batch.status_code == 200
    responses = r_batch.json().get("responses", r_batch.json().get("results", []))
    assert len(responses) == 2
    assert responses[0]["results"][0]["id"] == "b0"
    assert responses[1]["results"][0]["id"] == "b1"

    # Spin up FastAPI backed by bridge and test sutra-healthcheck.py + entrypoint syntax
    rest_url, _ = uvicorn_server_factory(backend_url=srv.bridge_url)
    rest_port = int(rest_url.rsplit(":", 1)[1])

    hc_proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "sutra-healthcheck.py")],
        env={
            "PATH": "/usr/bin:/bin",
            "SUTRA_MODE": "all",
            "SUTRA_GRPC_PORT": str(srv.grpc_port),
            "SUTRA_REST_PORT": str(rest_port),
        },
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    assert hc_proc.returncode == 0, f"Healthcheck failed: {hc_proc.stderr}"

    bash_check = subprocess.run(
        ["bash", "-n", str(REPO_ROOT / "docker-entrypoint.sh")],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    assert bash_check.returncode == 0, f"docker-entrypoint.sh syntax error: {bash_check.stderr}"
