use serde::{Deserialize, Serialize};
use crate::error::SutraError;

/// 1-Bit Binary Quantization (BQ) vector representation.
/// Compresses 64 single-precision float32 dimensions into a single `u64` bitmask,
/// achieving a 32x RAM reduction and sub-microsecond Hamming distance via hardware POPCNT.
/// Note: Sign-bit BQ preserves angular/cosine direction and is not suitable for unnormalized Euclidean distance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BQVector {
    pub bits: Vec<u64>,
    pub dimension: usize,
}

/// Computes bitwise Hamming distance between two raw `u64` word slices using POPCNT.
#[inline]
pub fn try_hamming_distance_words(a: &[u64], b: &[u64]) -> Result<u32, SutraError> {
    if a.len() != b.len() {
        return Err(SutraError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let mut dist = 0u32;
    for (&wa, &wb) in a.iter().zip(b.iter()) {
        dist += (wa ^ wb).count_ones();
    }
    Ok(dist)
}

/// Encodes `v` into an existing `out_words` bitmask slice, thresholding against
/// `v[i] - centroid[i] >= 0.0` (when `centroid` is provided) or `v[i] >= 0.0`.
#[inline]
pub fn quantize_bq_into_words(v: &[f32], centroid: Option<&[f32]>, out_words: &mut [u64]) {
    out_words.fill(0u64);
    match centroid {
        Some(mean) => {
            for (i, (&val, &mu)) in v.iter().zip(mean.iter()).enumerate() {
                if val - mu >= 0.0 {
                    out_words[i / 64] |= 1u64 << (i % 64);
                }
            }
        }
        None => {
            for (i, &val) in v.iter().enumerate() {
                if val >= 0.0 {
                    out_words[i / 64] |= 1u64 << (i % 64);
                }
            }
        }
    }
}

impl BQVector {
    /// Quantizes a full-precision float slice into 1-bit binary representation.
    /// Bit is set to 1 if v_i >= 0.0, else 0.
    pub fn quantize(v: &[f32]) -> Self {
        let dimension = v.len();
        let num_words = dimension.div_ceil(64);
        let mut bits = vec![0u64; num_words];
        quantize_bq_into_words(v, None, &mut bits);
        Self { bits, dimension }
    }

    /// Quantizes `v` centered by a per-dimension corpus mean vector (`v_i - mean_i >= 0.0`).
    pub fn quantize_centered(v: &[f32], centroid: &[f32]) -> Self {
        let dimension = v.len();
        let num_words = dimension.div_ceil(64);
        let mut bits = vec![0u64; num_words];
        quantize_bq_into_words(v, Some(centroid), &mut bits);
        Self { bits, dimension }
    }

    /// Fallible bitwise Hamming distance returning DimensionMismatch instead of panicking.
    #[inline]
    pub fn try_hamming_distance(&self, other: &BQVector) -> Result<u32, SutraError> {
        if self.dimension != other.dimension {
            return Err(SutraError::DimensionMismatch {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        try_hamming_distance_words(&self.bits, &other.bits)
    }

    /// Computes the bitwise Hamming distance between two binary quantized vectors using POPCNT.
    #[inline]
    pub fn hamming_distance(&self, other: &BQVector) -> u32 {
        assert_eq!(
            self.dimension, other.dimension,
            "Dimension mismatch in BQVector hamming_distance: {} vs {}",
            self.dimension, other.dimension
        );
        self.try_hamming_distance(other).unwrap()
    }

    /// Computes Hamming distance directly between raw u64 slices.
    #[inline]
    pub fn hamming_distance_words(a: &[u64], b: &[u64]) -> u32 {
        assert_eq!(
            a.len(),
            b.len(),
            "Slice length mismatch in hamming_distance_words: {} vs {}",
            a.len(),
            b.len()
        );
        try_hamming_distance_words(a, b).unwrap()
    }

    /// Linear cosine similarity approximation: (dim - 2 * d_H) / dim.
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
        let v2 = vec![1.0, -1.0, 0.5, -0.2, 0.0, 2.0];
        let v3 = vec![-1.0, 1.0, -0.5, 0.2, -0.1, -2.0];

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
        v2[..64].fill(-1.0);

        let bq1 = BQVector::quantize(&v1);
        let bq2 = BQVector::quantize(&v2);

        assert_eq!(bq1.hamming_distance(&bq2), 64);
        let sim = bq1.cosine_similarity(&bq2);
        let expected = (1536.0 - 128.0) / 1536.0;
        assert!((sim - expected).abs() < 1e-4);
    }
}
