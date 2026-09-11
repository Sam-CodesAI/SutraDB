"""
Tests for distance and similarity metrics.
"""

import numpy as np
import pytest
from sutradb.distance import Metric, compute_distances, normalize_matrix, normalize_vector


def test_normalize_vector():
    v = np.array([3.0, 4.0], dtype=np.float32)
    normed = normalize_vector(v)
    assert np.isclose(np.linalg.norm(normed), 1.0)
    assert np.allclose(normed, np.array([0.6, 0.8], dtype=np.float32))

    # Zero vector should not raise ZeroDivisionError
    zero_v = np.array([0.0, 0.0], dtype=np.float32)
    normed_zero = normalize_vector(zero_v)
    assert np.allclose(normed_zero, zero_v)


def test_cosine_similarity():
    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32)
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    scores = compute_distances(query, matrix, metric=Metric.COSINE)
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.0)
    assert np.isclose(scores[2], 1.0 / np.sqrt(2.0))


def test_dot_product():
    matrix = np.array([
        [2.0, 3.0],
        [-1.0, 4.0]
    ], dtype=np.float32)
    query = np.array([3.0, 2.0], dtype=np.float32)

    scores = compute_distances(query, matrix, metric=Metric.DOT_PRODUCT)
    # 2*3 + 3*2 = 12
    assert np.isclose(scores[0], 12.0)
    # -1*3 + 4*2 = 5
    assert np.isclose(scores[1], 5.0)


def test_euclidean_similarity():
    matrix = np.array([
        [0.0, 0.0],
        [0.0, 3.0],
        [0.0, 4.0]
    ], dtype=np.float32)
    query = np.array([0.0, 0.0], dtype=np.float32)

    scores = compute_distances(query, matrix, metric=Metric.EUCLIDEAN)
    # Distances: 0 -> 1/(1+0) = 1.0
    # 3 -> 1/(1+3) = 0.25
    # 4 -> 1/(1+4) = 0.20
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.25)
    assert np.isclose(scores[2], 0.20)


def test_dimension_mismatch():
    matrix = np.zeros((5, 4), dtype=np.float32)
    query = np.zeros(3, dtype=np.float32)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        compute_distances(query, matrix, metric=Metric.COSINE)
