"""
1-Bit Binary Quantization (BQ), 8-Bit Scalar Quantization (SQ8), and Product Quantization (PQ) for SutraDB.
Provides up to 64x RAM reduction, Asymmetric Distance Computation (ADC), and sub-microsecond vector search.
"""

from typing import List, Tuple, Union, Optional
import numpy as np

try:
    from sutradb._sutradb import bq_quantize as _native_bq_quantize
    from sutradb._sutradb import bq_hamming as _native_bq_hamming
    from sutradb._sutradb import sq8_quantize as _native_sq8_quantize
except ImportError:
    _native_bq_quantize = None
    _native_bq_hamming = None
    _native_sq8_quantize = None


class BQVector:
    """
    1-Bit Binary Quantization representation.
    Compresses 64 single-precision float32 dimensions into a single 64-bit word.
    """

    def __init__(self, bits: List[int], dimension: int):
        self.bits = bits
        self.dimension = dimension

    @classmethod
    def quantize(cls, vector: Union[List[float], np.ndarray]) -> "BQVector":
        """Quantizes float vector into 1-bit binary representation (v_i >= 0.0 -> 1)."""
        vec = np.asarray(vector, dtype=np.float32)
        dim = len(vec)
        if _native_bq_quantize is not None:
            bits, d = _native_bq_quantize(vec.tolist())
            return cls(bits, d)

        # Pure NumPy fallback
        num_words = (dim + 63) // 64
        words = []
        for w in range(num_words):
            word_val = 0
            start = w * 64
            end = min(start + 64, dim)
            for bit_idx, i in enumerate(range(start, end)):
                if vec[i] >= 0.0:
                    word_val |= (1 << bit_idx)
            words.append(word_val)
        return cls(words, dim)

    def hamming_distance(self, other: "BQVector") -> int:
        """Computes bitwise Hamming distance between two binary quantized vectors."""
        if _native_bq_hamming is not None:
            return _native_bq_hamming(self.bits, other.bits)

        dist = 0
        n = min(len(self.bits), len(other.bits))
        for i in range(n):
            dist += bin(self.bits[i] ^ other.bits[i]).count("1")
        return dist

    def cosine_similarity(self, other: "BQVector") -> float:
        """Computes linear cosine similarity approximation: (dim - 2 * d_H) / dim."""
        if self.dimension == 0:
            return 0.0
        dist = self.hamming_distance(other)
        return float(self.dimension - 2 * dist) / float(self.dimension)


class SQ8Vector:
    """
    8-Bit Scalar Quantization representation.
    Compresses 32-bit floats into 8-bit integers (4x RAM reduction).
    """

    def __init__(self, data: np.ndarray, min_val: float, scale: float):
        self.data = np.asarray(data, dtype=np.uint8)
        self.min_val = min_val
        self.scale = scale

    @classmethod
    def quantize(cls, vector: Union[List[float], np.ndarray]) -> "SQ8Vector":
        vec = np.asarray(vector, dtype=np.float32)
        if _native_sq8_quantize is not None:
            data, min_val, scale = _native_sq8_quantize(vec.tolist())
            return cls(np.array(data, dtype=np.uint8), min_val, scale)

        min_val = float(np.min(vec))
        max_val = float(np.max(vec))
        diff = max_val - min_val
        scale = diff / 255.0 if diff > 1e-8 else 1.0
        quantized = np.clip(np.round((vec - min_val) / scale), 0, 255).astype(np.uint8)
        return cls(quantized, min_val, scale)

    def dequantize(self) -> np.ndarray:
        return self.data.astype(np.float32) * self.scale + self.min_val


class PQVector:
    """Product Quantized Vector storing M bytes of centroid indices."""

    def __init__(self, codes: Union[List[int], np.ndarray]):
        self.codes = np.asarray(codes, dtype=np.uint8)

    def to_bytes(self) -> bytes:
        return self.codes.tobytes()


class PQCodebook:
    """
    Product Quantization Codebook.
    Decomposes D-dimensional space into M orthogonal sub-spaces of dimension d* = D // M.
    """

    def __init__(self, dimension: int, m: int, k: int, centroids: np.ndarray):
        self.dimension = dimension
        self.m = m
        self.k = k
        self.sub_dim = dimension // m
        self.centroids = np.asarray(centroids, dtype=np.float32)

    @classmethod
    def suggest_m(cls, dimension: int) -> int:
        """Suggests optimal M dividing dimension evenly."""
        for cand in [64, 48, 32, 24, 16, 8, 4, 2]:
            if cand <= dimension and dimension % cand == 0:
                return cand
        return 1

    @classmethod
    def train(cls, vectors: Union[List[List[float]], np.ndarray], m: Optional[int] = None, k: int = 256, iters: int = 20, seed: int = 42) -> "PQCodebook":
        vecs = np.asarray(vectors, dtype=np.float32)
        n, dim = vecs.shape
        effective_m = m or cls.suggest_m(dim)
        if dim % effective_m != 0:
            raise ValueError(f"Vector dimension {dim} must be divisible by M={effective_m}")

        sub_dim = dim // effective_m
        k = min(k, n)
        np.random.seed(seed)

        all_centroids = np.empty((effective_m, k, sub_dim), dtype=np.float32)
        for m_idx in range(effective_m):
            sub_vecs = vecs[:, m_idx * sub_dim : (m_idx + 1) * sub_dim]
            init_indices = np.random.choice(n, size=k, replace=False)
            centroids = sub_vecs[init_indices].copy()

            for _ in range(iters):
                dists = np.sum((sub_vecs[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
                assignments = np.argmin(dists, axis=1)
                new_centroids = np.array([
                    sub_vecs[assignments == c].mean(axis=0) if np.any(assignments == c) else centroids[c]
                    for c in range(k)
                ])
                centroids = new_centroids

            all_centroids[m_idx] = centroids

        return cls(dim, effective_m, k, all_centroids)

    def encode(self, vector: Union[List[float], np.ndarray]) -> PQVector:
        vec = np.asarray(vector, dtype=np.float32)
        codes = np.empty(self.m, dtype=np.uint8)
        for m_idx in range(self.m):
            sub_v = vec[m_idx * self.sub_dim : (m_idx + 1) * self.sub_dim]
            dists = np.sum((self.centroids[m_idx] - sub_v[None, :]) ** 2, axis=1)
            codes[m_idx] = int(np.argmin(dists))
        return PQVector(codes)

    def decode(self, pq_vec: PQVector) -> np.ndarray:
        out = np.empty(self.dimension, dtype=np.float32)
        for m_idx in range(self.m):
            c_idx = pq_vec.codes[m_idx]
            out[m_idx * self.sub_dim : (m_idx + 1) * self.sub_dim] = self.centroids[m_idx, c_idx]
        return out

    def compute_lut(self, query: Union[List[float], np.ndarray], metric: str = "cosine") -> np.ndarray:
        q = np.asarray(query, dtype=np.float32)
        lut = np.empty((self.m, self.k), dtype=np.float32)
        for m_idx in range(self.m):
            q_sub = q[m_idx * self.sub_dim : (m_idx + 1) * self.sub_dim]
            if metric in ("cosine", "dot_product"):
                lut[m_idx] = np.dot(self.centroids[m_idx], q_sub)
            else:
                lut[m_idx] = np.sum((self.centroids[m_idx] - q_sub[None, :]) ** 2, axis=1)
        return lut

    def score_adc(self, pq_vec: PQVector, lut: np.ndarray, metric: str = "cosine") -> float:
        total = float(np.sum(lut[np.arange(self.m), pq_vec.codes]))
        if metric in ("cosine", "dot_product"):
            return total
        return float(1.0 / (1.0 + np.sqrt(max(0.0, total))))
