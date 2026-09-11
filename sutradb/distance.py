"""
Distance and similarity metrics for SutraDB.
Optimized for SIMD vectorization via NumPy C-BLAS routines.
"""

from enum import Enum
from typing import Union
import numpy as np


class Metric(str, Enum):
    """Supported distance and similarity metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


def normalize_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalizes a vector to unit length (L2 norm = 1.0).
    Safely handles zero-norm vectors by returning zeroes.
    """
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm < eps:
        return v
    return v / norm


def normalize_matrix(m: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalizes a 2D matrix of row vectors to unit length.
    Safely handles zero-norm vectors without division-by-zero warnings.
    """
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms < eps] = 1.0
    return m / norms


def compute_distances(
    query: np.ndarray,
    matrix: np.ndarray,
    metric: Union[Metric, str] = Metric.COSINE,
    is_normalized: bool = False
) -> np.ndarray:
    """
    Computes distance/similarity scores between a single query vector (D,)
    and an (N, D) matrix of stored vectors.
    
    Returns an array of shape (N,) containing similarity scores:
    - For COSINE: higher is more similar (range [-1.0, 1.0])
    - For DOT_PRODUCT: higher is more similar
    - For EUCLIDEAN: transformed to similarity via 1 / (1 + distance), so higher is more similar
    """
    metric_str = metric.value if isinstance(metric, Metric) else str(metric).lower()
    q = np.asarray(query, dtype=np.float32).ravel()
    M = np.asarray(matrix, dtype=np.float32)

    if M.ndim == 1:
        M = M.reshape(1, -1)

    if q.shape[0] != M.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query has dimension {q.shape[0]}, "
            f"matrix vectors have dimension {M.shape[1]}"
        )

    if metric_str == Metric.COSINE:
        if is_normalized:
            # When pre-normalized, Cosine Similarity simplifies to matrix-vector dot product
            q_norm = normalize_vector(q)
            return np.dot(M, q_norm)
        else:
            q_norm = normalize_vector(q)
            m_norm = normalize_matrix(M)
            return np.dot(m_norm, q_norm)

    elif metric_str == Metric.DOT_PRODUCT:
        return np.dot(M, q)

    elif metric_str == Metric.EUCLIDEAN:
        # Euclidean distance = ||M - q||
        diff = M - q
        raw_distances = np.linalg.norm(diff, axis=1)
        # Convert distance to a bounded similarity score in (0, 1]
        return 1.0 / (1.0 + raw_distances)

    else:
        raise ValueError(
            f"Unsupported metric: {metric}. Supported metrics are 'cosine', 'euclidean', 'dot_product'."
        )
