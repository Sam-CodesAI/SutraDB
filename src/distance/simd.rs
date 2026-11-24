use super::Metric;
use crate::error::SutraError;

/// Computes the dot product of two single-precision float slices.
/// Uses 8-way loop unrolling to maximize CPU pipeline throughput and enable SIMD FMA vectorization.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Slices must have equal length for dot product");
    let len = a.len();
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut sum4 = 0.0f32;
    let mut sum5 = 0.0f32;
    let mut sum6 = 0.0f32;
    let mut sum7 = 0.0f32;

    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let base = i * 8;
        unsafe {
            sum0 += a.get_unchecked(base) * b.get_unchecked(base);
            sum1 += a.get_unchecked(base + 1) * b.get_unchecked(base + 1);
            sum2 += a.get_unchecked(base + 2) * b.get_unchecked(base + 2);
            sum3 += a.get_unchecked(base + 3) * b.get_unchecked(base + 3);
            sum4 += a.get_unchecked(base + 4) * b.get_unchecked(base + 4);
            sum5 += a.get_unchecked(base + 5) * b.get_unchecked(base + 5);
            sum6 += a.get_unchecked(base + 6) * b.get_unchecked(base + 6);
            sum7 += a.get_unchecked(base + 7) * b.get_unchecked(base + 7);
        }
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);
    let base = chunks * 8;
    for i in 0..remainder {
        unsafe {
            total += a.get_unchecked(base + i) * b.get_unchecked(base + i);
        }
    }

    total
}

/// Computes squared Euclidean distance ||a - b||^2 using 8-way unrolling.
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Slices must have equal length for Euclidean distance");
    let len = a.len();
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut sum4 = 0.0f32;
    let mut sum5 = 0.0f32;
    let mut sum6 = 0.0f32;
    let mut sum7 = 0.0f32;

    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let base = i * 8;
        unsafe {
            let d0 = a.get_unchecked(base) - b.get_unchecked(base);
            let d1 = a.get_unchecked(base + 1) - b.get_unchecked(base + 1);
            let d2 = a.get_unchecked(base + 2) - b.get_unchecked(base + 2);
            let d3 = a.get_unchecked(base + 3) - b.get_unchecked(base + 3);
            let d4 = a.get_unchecked(base + 4) - b.get_unchecked(base + 4);
            let d5 = a.get_unchecked(base + 5) - b.get_unchecked(base + 5);
            let d6 = a.get_unchecked(base + 6) - b.get_unchecked(base + 6);
            let d7 = a.get_unchecked(base + 7) - b.get_unchecked(base + 7);

            sum0 += d0 * d0;
            sum1 += d1 * d1;
            sum2 += d2 * d2;
            sum3 += d3 * d3;
            sum4 += d4 * d4;
            sum5 += d5 * d5;
            sum6 += d6 * d6;
            sum7 += d7 * d7;
        }
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);
    let base = chunks * 8;
    for i in 0..remainder {
        unsafe {
            let d = a.get_unchecked(base + i) - b.get_unchecked(base + i);
            total += d * d;
        }
    }

    total
}

/// Normalizes a vector in-place to unit L2 length.
#[inline]
pub fn normalize_in_place(v: &mut [f32]) {
    let norm_sq = dot_product(v, v);
    if norm_sq > 1e-24 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for x in v.iter_mut() {
            *x *= inv_norm;
        }
    }
}

/// Returns a normalized copy of the vector.
#[inline]
pub fn normalize_copy(v: &[f32]) -> Vec<f32> {
    let mut copy = v.to_vec();
    normalize_in_place(&mut copy);
    copy
}

/// Computes distance/similarity score between query vector and candidate vector.
/// Returns score where higher is more similar:
/// - Cosine: [-1.0, 1.0]
/// - DotProduct: unbounded dot product
/// - Euclidean: transformed to similarity via 1.0 / (1.0 + sqrt(dist_sq)) in (0, 1]
#[inline]
pub fn compute_vector_score(
    query: &[f32],
    candidate: &[f32],
    metric: Metric,
    is_normalized: bool,
) -> Result<f32, SutraError> {
    if query.len() != candidate.len() {
        return Err(SutraError::DimensionMismatch {
            expected: query.len(),
            got: candidate.len(),
        });
    }

    match metric {
        Metric::Cosine => {
            if is_normalized {
                Ok(dot_product(query, candidate))
            } else {
                let norm_q = dot_product(query, query).sqrt();
                let norm_c = dot_product(candidate, candidate).sqrt();
                if norm_q < 1e-12 || norm_c < 1e-12 {
                    Ok(0.0)
                } else {
                    Ok(dot_product(query, candidate) / (norm_q * norm_c))
                }
            }
        }
        Metric::DotProduct => Ok(dot_product(query, candidate)),
        Metric::Euclidean => {
            let dist_sq = euclidean_distance_squared(query, candidate);
            Ok(1.0 / (1.0 + dist_sq.sqrt()))
        }
    }
}

/// Hardware SIMD execution capability level detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    AVX512,
    AVX2,
    NEON,
    Scalar,
}

impl std::fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AVX512 => write!(f, "AVX-512 FMA"),
            Self::AVX2 => write!(f, "AVX2 / FMA"),
            Self::NEON => write!(f, "ARM NEON"),
            Self::Scalar => write!(f, "Scalar 8-Way Unrolled"),
        }
    }
}

/// Dynamically detects CPU hardware capabilities at runtime for optimal instruction dispatch.
pub fn detect_simd_capabilities() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
            return SimdLevel::AVX512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdLevel::AVX2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return SimdLevel::NEON;
        }
    }
    SimdLevel::Scalar
}

/// Computes similarity scores for a batch of candidate vectors against a single query vector.
/// Uses Rayon work-stealing concurrency when batch size >= 64, saturating multicore execution.
pub fn batch_distances(
    query: &[f32],
    vectors: &[Vec<f32>],
    metric: Metric,
    is_normalized: bool,
) -> Vec<f32> {
    use rayon::prelude::*;

    if vectors.len() >= 64 {
        vectors
            .par_iter()
            .map(|v| compute_vector_score(query, v, metric, is_normalized).unwrap_or(0.0))
            .collect()
    } else {
        vectors
            .iter()
            .map(|v| compute_vector_score(query, v, metric, is_normalized).unwrap_or(0.0))
            .collect()
    }
}

/// Evaluates distance matrix between multiple query vectors and database vectors using Rayon.
pub fn batch_matrix_distance(
    queries: &[Vec<f32>],
    vectors: &[Vec<f32>],
    metric: Metric,
    is_normalized: bool,
) -> Vec<Vec<f32>> {
    use rayon::prelude::*;

    queries
        .par_iter()
        .map(|q| batch_distances(q, vectors, metric, is_normalized))
        .collect()
}
