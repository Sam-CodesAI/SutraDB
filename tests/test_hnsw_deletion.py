"""
Tests for Dynamic Online HNSW Node Deletion and Edge Healing.
"""

import pytest
import numpy as np
from sutradb import Collection, Document, Metric, NativeCollection


def test_native_hnsw_online_deletion_and_reconnection():
    if NativeCollection is None:
        pytest.skip("Native Rust engine not available")

    col = NativeCollection(name="hnsw_healing", dimension=4, metric="cosine")

    # Insert 4 documents
    docs = [
        {"id": "doc_a", "text": "alpha", "metadata": {"tag": "a"}},
        {"id": "doc_b", "text": "beta", "metadata": {"tag": "b"}},
        {"id": "doc_c", "text": "gamma", "metadata": {"tag": "c"}},
        {"id": "doc_d", "text": "delta", "metadata": {"tag": "d"}},
    ]
    vecs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.9, 0.1, 0.0],
    ]

    col.insert(docs, vecs)
    col.build_hnsw(m=16, ef_construction=64, ef_search=32)

    # Search ANN
    q = [1.0, 0.0, 0.0, 0.0]
    res = col.query(vector=q, top_k=2, hybrid=False, use_ann=True)
    assert len(res) == 2
    assert res[0]["id"] == "doc_a"

    # Dynamic delete of top node "doc_a"
    del_count = col.delete(["doc_a"])
    assert del_count == 1
    assert col.count() == 3

    # Query ANN again: graph must heal and find doc_b
    res_after = col.query(vector=q, top_k=2, hybrid=False, use_ann=True)
    assert len(res_after) >= 1
    assert res_after[0]["id"] == "doc_b"
    assert "doc_a" not in [r["id"] for r in res_after]


def test_python_collection_deletion_integrity():
    col = Collection(name="py_del_test", dimension=4, metric=Metric.COSINE)
    docs = [
        Document(id="d1", vector=[1.0, 0.0, 0.0, 0.0], text="doc 1"),
        Document(id="d2", vector=[0.0, 1.0, 0.0, 0.0], text="doc 2"),
        Document(id="d3", vector=[0.0, 0.0, 1.0, 0.0], text="doc 3"),
    ]
    col.insert(docs)
    assert col.count() == 3

    col.delete(["d2"])
    assert col.count() == 2
    assert col.get("d2") is None
    assert col.get("d1") is not None

    res = col.query(vector=[0.0, 1.0, 0.0, 0.0], top_k=5)
    retrieved_ids = [r.id for r in res]
    assert "d2" not in retrieved_ids
