"""
Tests for Product Quantization (PQ) and Asymmetric Distance Computation (ADC) in SutraDB.
"""

import pytest
import numpy as np
from sutradb import Collection, Document, Metric, PQCodebook, PQVector


def test_pq_suggest_m():
    assert PQCodebook.suggest_m(1536) in (64, 48, 32, 24, 16)
    assert PQCodebook.suggest_m(128) in (64, 32, 16, 8)
    assert PQCodebook.suggest_m(16) in (16, 8, 4)
    assert PQCodebook.suggest_m(7) == 1


def test_pq_training_and_reconstruction():
    dim = 16
    n = 60
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    # Train PQ with M=4 (4 sub-spaces of 4 dimensions)
    codebook = PQCodebook.train(vectors, m=4, k=16, iters=15)
    assert codebook.m == 4
    assert codebook.sub_dim == 4
    assert codebook.k == 16

    # Encode vector
    pq_vec = codebook.encode(vectors[0])
    assert len(pq_vec.codes) == 4
    assert all(0 <= c < 16 for c in pq_vec.codes)

    # Decode vector (reconstruction)
    recon = codebook.decode(pq_vec)
    assert recon.shape == (dim,)
    # Reconstruction should be reasonably close to original
    cos_sim = float(np.dot(vectors[0], recon) / (np.linalg.norm(vectors[0]) * np.linalg.norm(recon)))
    assert cos_sim > 0.70


def test_pq_asymmetric_distance_computation():
    dim = 8
    n = 30
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)

    codebook = PQCodebook.train(vectors, m=2, k=8, iters=10)
    query = vectors[0]

    lut = codebook.compute_lut(query, metric="cosine")
    assert lut.shape == (2, 8)

    encoded = codebook.encode(query)
    adc_score = codebook.score_adc(encoded, lut, metric="cosine")
    # Score against self should be positive
    assert adc_score > 0.0


def test_collection_pq_integration_and_search():
    dim = 16
    col = Collection(name="pq_col_test", dimension=dim, metric=Metric.COSINE)

    docs = []
    np.random.seed(42)
    for i in range(50):
        vec = np.random.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        docs.append(Document(id=f"doc_{i}", vector=vec, text=f"Document text {i}", metadata={"idx": i}))

    col.insert(docs)

    # Train and enable PQ
    col.train_pq(m=4)
    assert col._pq_enabled is True
    assert len(col._pq_vectors) == 50

    stats = col.stats()
    assert stats["quantized_pq"] is True
    assert stats["pq_m"] == 4

    # Query with target close to doc_0
    target_vec = col.vectors[0].copy()
    results = col.query(vector=target_vec, top_k=5)

    assert len(results) > 0
    assert results[0].id == "doc_0"
    assert pytest.approx(results[0].score, abs=1e-3) == 1.0
