"""
Integration tests for SutraDB v4.0 Industrial features:
1. Secondary Inverted & Numerical Range Metadata Indexing (sub-microsecond bitset intersections)
2. Cost-Based ACORN Query Planning (eliminating HNSW filter collapse on low selectivity)
3. Neural Sparse Vector Retrieval (SPLADE / BGE-M3 / WAND dynamic pruning)
4. Tri-Modal Reciprocal Rank Fusion (Dense + BM25 + Sparse)
"""

import pytest
import numpy as np
from sutradb import NativeCollection, Document, SutraDB


def test_secondary_metadata_fast_filtering(tmp_path):
    store_file = str(tmp_path / "v4_meta.sutra")
    col = NativeCollection("v4_meta", 4, "cosine", store_file, True)

    docs = [
        {
            "id": "doc_1",
            "namespace": "acme",
            "text": "PostgreSQL database performance optimization",
            "metadata": {"tier": "gold", "price": 99.0, "user": {"role": "admin", "age": 35}},
        },
        {
            "id": "doc_2",
            "namespace": "acme",
            "text": "Kubernetes container cluster scaling",
            "metadata": {"tier": "silver", "price": 45.0, "user": {"role": "developer", "age": 28}},
        },
        {
            "id": "doc_3",
            "namespace": "globex",
            "text": "Deep learning transformer models",
            "metadata": {"tier": "gold", "price": 150.0, "user": {"role": "researcher", "age": 42}},
        },
    ]

    vecs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    col.insert(docs, vecs)

    # 1. Test compound metadata filter (tier == gold AND price >= 50.0)
    results = col.query(
        vector=[1.0, 0.0, 0.0, 0.0],
        filter={"tier": "gold", "price": {"$gte": 50.0}},
        top_k=5,
    )
    assert len(results) == 2
    ids = [r["id"] for r in results]
    assert "doc_1" in ids
    assert "doc_3" in ids
    assert "doc_2" not in ids

    # 2. Test nested path filtering (user.role == developer)
    dev_results = col.query(
        vector=[0.0, 1.0, 0.0, 0.0],
        filter={"user.role": "developer"},
        top_k=5,
    )
    assert len(dev_results) == 1
    assert dev_results[0]["id"] == "doc_2"

    # 3. Test nested numerical range (user.age > 40)
    senior_results = col.query(
        vector=[0.0, 0.0, 1.0, 0.0],
        filter={"user.age": {"$gt": 40}},
        top_k=5,
    )
    assert len(senior_results) == 1
    assert senior_results[0]["id"] == "doc_3"


def test_cost_based_acorn_hnsw_bypass(tmp_path):
    """
    Verifies that the Cost-Based Query Planner prevents 'HNSW Filter Collapse':
    When a filter matches only 0.2% of documents, standard HNSW collapses.
    SutraDB v4's planner detects low selectivity, bypasses HNSW, and delivers 100% exact recall.
    """
    store_file = str(tmp_path / "acorn.sutra")
    col = NativeCollection("acorn_test", 4, "cosine", store_file, False)

    n = 500
    docs = []
    vecs = []

    for i in range(n):
        is_target = (i == 42 or i == 314)
        docs.append({
            "id": f"doc_{i}",
            "text": f"Document content for index {i}",
            "metadata": {
                "secret_target": is_target,
                "tier": "classified" if is_target else "public"
            }
        })
        # Orthogonal noise vectors except target
        if is_target:
            vecs.append([1.0, 0.0, 0.0, 0.0])
        else:
            v = [0.0, 0.0, 0.0, 0.0]
            v[i % 4] = 1.0
            vecs.append(v)

    col.insert(docs, vecs)
    col.build_hnsw(m=16, ef_construction=32, ef_search=16)

    # Query with use_ann=True on needle-in-a-haystack filter (2 out of 500 = 0.4% selectivity)
    results = col.query(
        vector=[1.0, 0.0, 0.0, 0.0],
        filter={"secret_target": True},
        top_k=10,
        use_ann=True  # Optimizer will bypass HNSW to prevent filter collapse
    )

    assert len(results) == 2
    res_ids = {r["id"] for r in results}
    assert "doc_42" in res_ids
    assert "doc_314" in res_ids
    assert results[0]["score"] > 0.99


def test_neural_sparse_vector_splade_and_trimodal_rrf(tmp_path):
    """
    Verifies SPLADE / BGE-M3 neural sparse vector indexing,
    WAND search, and Tri-Modal RRF fusion (Dense + BM25 + Sparse).
    """
    store_file = str(tmp_path / "sparse.sutra")
    col = NativeCollection("sparse_test", 4, "cosine", store_file, False)

    docs = [
        {
            "id": "doc_systems",
            "text": "Rust systems programming with zero-cost abstractions",
            "metadata": {"domain": "systems"},
            "sparse_vector": {101: 2.5, 202: 1.8, 303: 3.2}
        },
        {
            "id": "doc_web",
            "text": "Python web frameworks and async event loops",
            "metadata": {"domain": "web"},
            "sparse_vector": {202: 3.0, 404: 2.1}
        },
        {
            "id": "doc_ai",
            "text": "Neural network architectures and deep learning models",
            "metadata": {"domain": "ai"},
            "sparse_vector": {303: 1.5, 505: 4.0}
        }
    ]

    vecs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    col.insert(docs, vecs)

    # 1. Pure sparse vector query
    # Matches doc_web strongly (202: 2.0 * 3.0 = 6.0)
    sparse_query = {202: 2.0, 404: 1.5}
    sparse_results = col.query(
        sparse_vector=sparse_query,
        top_k=2,
        hybrid=False
    )
    assert len(sparse_results) >= 1
    assert sparse_results[0]["id"] == "doc_web"
    assert sparse_results[0]["sparse_score"] is not None
    assert sparse_results[0]["sparse_score"] > 0.0

    # 2. Tri-Modal Hybrid Query (Dense + BM25 + Sparse fused via RRF)
    tri_results = col.query(
        vector=[1.0, 0.0, 0.0, 0.0],      # Matches doc_systems on dense
        text="zero-cost abstractions",     # Matches doc_systems on BM25
        sparse_vector={101: 2.0, 303: 2.0},# Matches doc_systems on sparse
        top_k=3,
        hybrid=True,
        dense_weight=1.0,
        bm25_weight=1.0,
        sparse_weight=1.0
    )

    assert len(tri_results) == 3
    assert tri_results[0]["id"] == "doc_systems"
    assert tri_results[0]["dense_score"] is not None
    assert tri_results[0]["bm25_score"] is not None
    assert tri_results[0]["sparse_score"] is not None
