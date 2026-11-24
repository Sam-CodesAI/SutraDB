"""
1-Bit Binary Quantization (BQ) and 8-Bit Scalar Quantization (SQ8) for SutraDB.
Provides 32x RAM reduction and sub-microsecond Hamming distance vector evaluation.
"""

from typing import List, Tuple, Union
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
