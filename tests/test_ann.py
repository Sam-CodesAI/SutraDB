"""
Tests for IVF-Flat Approximate Nearest Neighbor indexing and Collection integration in SutraDB.
"""

import threading
import numpy as np
import pytest

from sutradb import (
    Collection,
    Document,
    Metric,
    IVFIndex,
    IVFConfig,
    auto_n_lists,
    __version__
)


def test_ann_exports():
    """Verify that IVFIndex, IVFConfig, and auto_n_lists are properly exported."""
    assert IVFIndex is not None
    assert IVFConfig is not None
    assert callable(auto_n_lists)
    assert auto_n_lists(100) == 16
    assert auto_n_lists(10_000) == 100
    assert __version__ in ("1.0.0", "2.0.0", "2.2.0")


def test_ivf_index_direct():
    """Verify standalone IVFIndex training, insertion, and candidate search."""
    np.random.seed(42)
    dim = 16
    n = 200
    vectors = np.random.randn(n, dim).astype(np.float32)
    # L2 normalize
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    config = IVFConfig(n_lists=16, nprobe=4, seed=42)
    index = IVFIndex(config)
    assert not index.is_trained

    index.train(vectors)
    assert index.is_trained
    assert index.centroids.shape == (16, dim)

    index.add(vectors, start_idx=0)
    stats = index.stats()
    assert stats["trained"] is True
    assert stats["n_vectors"] == n

    # Search with a query vector close to vector 0
    query = vectors[0] + np.random.randn(dim).astype(np.float32) * 0.01
    query /= np.linalg.norm(query)

    indices, scores = index.search(query, vectors, k=5, metric=Metric.COSINE, is_normalized=True)
    assert len(indices) == 5
    assert indices[0] == 0  # Closest vector should be vector 0


def test_collection_ann_integration():
    """Verify Collection.build_index() and Collection.query(use_ann=True)."""
    np.random.seed(123)
    dim = 8
    n = 100
    col = Collection("test_ann_col", dimension=dim, metric=Metric.COSINE)

    docs = []
    for i in range(n):
        v = np.random.randn(dim).astype(np.float32)
        docs.append(Document(
            id=f"doc_{i}",
            vector=v,
            text=f"Document {i} indexing test",
            metadata={"category": "test", "num": i}
        ))

    col.insert(docs)
    assert col.count() == n

    # Build ANN index
    ann_index = col.build_index(IVFConfig(n_lists=10, nprobe=4))
    assert ann_index.is_trained

    col_stats = col.stats()
    assert "ann" in col_stats
    assert col_stats["ann"]["trained"] is True
    assert col_stats["ann"]["dirty"] is False

    # Execute query with use_ann=True
    q_vec = docs[10].vector
    ann_results = col.query(vector=q_vec, top_k=5, use_ann=True)
    assert len(ann_results) == 5
    assert ann_results[0].id == "doc_10"
    assert ann_results[0].score > 0.99

    # Filtered ANN query
    filtered_results = col.query(
        vector=q_vec,
        filter={"num": {"$lte": 5}},
        top_k=5,
        use_ann=True
    )
    for r in filtered_results:
        assert r.metadata["num"] <= 5


def test_collection_concurrent_thread_safety():
    """Verify concurrent reads and writes using multiple threads under load."""
    dim = 4
    col = Collection("test_concurrency", dimension=dim, metric=Metric.COSINE)

    # Seed initial documents
    initial_docs = [
        Document(id=f"seed_{i}", vector=np.ones(dim, dtype=np.float32), text=f"seed {i}")
        for i in range(20)
    ]
    col.insert(initial_docs)

    errors = []

    def writer_task(thread_id: int):
        try:
            for j in range(15):
                v = np.random.randn(dim).astype(np.float32)
                col.insert(Document(id=f"t{thread_id}_{j}", vector=v, text=f"text {thread_id} {j}"))
        except Exception as e:
            errors.append(e)

    def reader_task():
        try:
            for _ in range(25):
                q = np.ones(dim, dtype=np.float32)
                res = col.query(vector=q, top_k=3)
                assert len(res) <= 3
                assert col.count() >= 20
        except Exception as e:
            errors.append(e)

    threads = []
    # Launch 3 writer threads and 3 reader threads concurrently
    for t_id in range(3):
        threads.append(threading.Thread(target=writer_task, args=(t_id,)))
        threads.append(threading.Thread(target=reader_task))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent thread errors encountered: {errors}"
    assert col.count() == 20 + (3 * 15)
