use serde::{Deserialize, Serialize};

/// 1-Bit Binary Quantization (BQ) vector representation.
/// Compresses 64 single-precision float32 dimensions into a single `u64` bitmask,
/// achieving a 32x RAM reduction and sub-microsecond Hamming distance via hardware POPCNT.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BQVector {
    pub bits: Vec<u64>,
    pub dimension: usize,
}

impl BQVector {
    /// Quantizes a full-precision float slice into 1-bit binary representation.
    /// Bit is set to 1 if v_i >= 0.0, else 0.
    pub fn quantize(v: &[f32]) -> Self {
        let dimension = v.len();
        let num_words = (dimension + 63) / 64;
        let mut bits = vec![0u64; num_words];

        for (i, &val) in v.iter().enumerate() {
            if val >= 0.0 {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                bits[word_idx] |= 1u64 << bit_idx;
            }
        }

        Self { bits, dimension }
    }

    /// Computes the bitwise Hamming distance between two binary quantized vectors using POPCNT.
    #[inline]
    pub fn hamming_distance(&self, other: &BQVector) -> u32 {
        let mut dist = 0u32;
        let n = self.bits.len().min(other.bits.len());
        for i in 0..n {
            dist += (self.bits[i] ^ other.bits[i]).count_ones();
        }
        dist
    }

    /// Computes Hamming distance directly between raw u64 slices.
    #[inline]
    pub fn hamming_distance_words(a: &[u64], b: &[u64]) -> u32 {
        let mut dist = 0u32;
        let n = a.len().min(b.len());
        for i in 0..n {
            dist += (a[i] ^ b[i]).count_ones();
        }
        dist
    }

    /// Linear cosine similarity approximation: (dim - 2 * d_H) / dim.
    /// Range: [-1.0, 1.0]. 1.0 = identical bits, -1.0 = completely inverted bits.
    #[inline]
    pub fn cosine_similarity(&self, other: &BQVector) -> f32 {
        if self.dimension == 0 {
            return 0.0;
        }
        let dist = self.hamming_distance(other);
        (self.dimension as f32 - 2.0 * dist as f32) / (self.dimension as f32)
    }

    /// Angular cosine similarity based on Charikar's Theorem: cos(pi * d_H / dim).
    #[inline]
    pub fn angular_cosine_similarity(&self, other: &BQVector) -> f32 {
        if self.dimension == 0 {
            return 0.0;
        }
        let dist = self.hamming_distance(other);
        let theta = (std::f32::consts::PI * dist as f32) / (self.dimension as f32);
        theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bq_quantization_and_hamming() {
        let v1 = vec![1.0, -1.0, 0.5, -0.2, 0.0, 2.0];
        let v2 = vec![1.0, -1.0, 0.5, -0.2, 0.0, 2.0]; // identical
        let v3 = vec![-1.0, 1.0, -0.5, 0.2, -0.1, -2.0]; // completely inverted

        let bq1 = BQVector::quantize(&v1);
        let bq2 = BQVector::quantize(&v2);
        let bq3 = BQVector::quantize(&v3);

        assert_eq!(bq1.hamming_distance(&bq2), 0);
        assert_eq!(bq1.cosine_similarity(&bq2), 1.0);

        assert_eq!(bq1.hamming_distance(&bq3), 6);
        assert_eq!(bq1.cosine_similarity(&bq3), -1.0);
    }

    #[test]
    fn test_bq_large_dimension() {
        let dim = 1536;
        let v1 = vec![1.0; dim];
        let mut v2 = vec![1.0; dim];
        // Invert first 64 dimensions
        for i in 0..64 {
            v2[i] = -1.0;
        }

        let bq1 = BQVector::quantize(&v1);
        let bq2 = BQVector::quantize(&v2);

        assert_eq!(bq1.hamming_distance(&bq2), 64);
        let sim = bq1.cosine_similarity(&bq2);
        let expected = (1536.0 - 128.0) / 1536.0;
        assert!((sim - expected).abs() < 1e-4);
    }
}
