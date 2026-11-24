"""
Tests for 1-Bit Binary Quantization (BQ) and Two-Stage Search in SutraDB.
"""

import pytest
import numpy as np
from sutradb import Collection, Document, Metric
from sutradb.quantization import BQVector


def test_bq_quantization_and_hamming():
    # 6-dim vector
    v1 = np.array([1.0, -1.0, 0.5, -0.2, 0.0, 2.0], dtype=np.float32)
    v2 = np.array([1.0, -1.0, 0.5, -0.2, 0.0, 2.0], dtype=np.float32)
    v3 = np.array([-1.0, 1.0, -0.5, 0.2, -0.1, -2.0], dtype=np.float32)

    bq1 = BQVector.quantize(v1)
    bq2 = BQVector.quantize(v2)
    bq3 = BQVector.quantize(v3)

    assert bq1.hamming_distance(bq2) == 0
    assert pytest.approx(bq1.cosine_similarity(bq2)) == 1.0

    assert bq1.hamming_distance(bq3) == 6
    assert pytest.approx(bq1.cosine_similarity(bq3)) == -1.0


def test_bq_high_dimension_compression():
    dim = 1536
    v = np.random.randn(dim).astype(np.float32)
    bq = BQVector.quantize(v)

    # 1536 float32 = 6144 bytes
    # 1536 bits = 24 words of 64-bit = 192 bytes -> exactly 32x reduction!
    assert len(bq.bits) == 24
    assert bq.dimension == 1536


def test_bq_two_stage_collection_search():
    dim = 64
    col = Collection(name="bq_test_col", dimension=dim, metric=Metric.COSINE)

    # Insert 100 documents
    docs = []
    np.random.seed(42)
    for i in range(100):
        vec = np.random.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        docs.append(Document(id=f"doc_{i}", vector=vec, text=f"doc {i}", metadata={"idx": i}))

    col.insert(docs)
    col.enable_bq()

    assert col._bq_enabled is True
    assert len(col._bq_vectors) == 100

    # Query with target close to doc_0
    target_vec = col.vectors[0].copy()
    results = col.query(vector=target_vec, top_k=5)

    assert len(results) > 0
    assert results[0].id == "doc_0"
    assert pytest.approx(results[0].score, abs=1e-3) == 1.0

    stats = col.stats()
    assert stats["quantized_bq"] is True
