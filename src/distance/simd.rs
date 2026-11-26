use super::Metric;
use crate::error::SutraError;

#[inline(always)]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
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

#[inline(always)]
fn euclidean_distance_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hsum_m256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 0x55);
    _mm_cvtss_f32(_mm_add_ss(sum64, hi32))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 32 <= len {
        let a0 = _mm256_loadu_ps(pa.add(i));
        let b0 = _mm256_loadu_ps(pb.add(i));
        let a1 = _mm256_loadu_ps(pa.add(i + 8));
        let b1 = _mm256_loadu_ps(pb.add(i + 8));
        let a2 = _mm256_loadu_ps(pa.add(i + 16));
        let b2 = _mm256_loadu_ps(pb.add(i + 16));
        let a3 = _mm256_loadu_ps(pa.add(i + 24));
        let b3 = _mm256_loadu_ps(pb.add(i + 24));

        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        acc2 = _mm256_fmadd_ps(a2, b2, acc2);
        acc3 = _mm256_fmadd_ps(a3, b3, acc3);
        i += 32;
    }

    while i + 8 <= len {
        let va = _mm256_loadu_ps(pa.add(i));
        let vb = _mm256_loadu_ps(pb.add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }

    let combined = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut total = hsum_m256(combined);
    while i < len {
        total += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();

    let mut i = 0usize;
    while i + 64 <= len {
        let a0 = _mm512_loadu_ps(pa.add(i));
        let b0 = _mm512_loadu_ps(pb.add(i));
        let a1 = _mm512_loadu_ps(pa.add(i + 16));
        let b1 = _mm512_loadu_ps(pb.add(i + 16));
        let a2 = _mm512_loadu_ps(pa.add(i + 32));
        let b2 = _mm512_loadu_ps(pb.add(i + 32));
        let a3 = _mm512_loadu_ps(pa.add(i + 48));
        let b3 = _mm512_loadu_ps(pb.add(i + 48));

        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        acc1 = _mm512_fmadd_ps(a1, b1, acc1);
        acc2 = _mm512_fmadd_ps(a2, b2, acc2);
        acc3 = _mm512_fmadd_ps(a3, b3, acc3);
        i += 64;
    }

    while i + 16 <= len {
        let va = _mm512_loadu_ps(pa.add(i));
        let vb = _mm512_loadu_ps(pb.add(i));
        acc0 = _mm512_fmadd_ps(va, vb, acc0);
        i += 16;
    }

    let combined = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    let mut total = _mm512_reduce_add_ps(combined);
    while i < len {
        total += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn euclidean_distance_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 32 <= len {
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
        let d1 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)));
        let d2 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 16)), _mm256_loadu_ps(pb.add(i + 16)));
        let d3 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 24)), _mm256_loadu_ps(pb.add(i + 24)));

        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
        i += 32;
    }

    while i + 8 <= len {
        let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
        acc0 = _mm256_fmadd_ps(d, d, acc0);
        i += 8;
    }

    let combined = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut total = hsum_m256(combined);
    while i < len {
        let d = *pa.add(i) - *pb.add(i);
        total += d * d;
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn euclidean_distance_squared_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();

    let mut i = 0usize;
    while i + 64 <= len {
        let d0 = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)));
        let d1 = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i + 16)), _mm512_loadu_ps(pb.add(i + 16)));
        let d2 = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i + 32)), _mm512_loadu_ps(pb.add(i + 32)));
        let d3 = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i + 48)), _mm512_loadu_ps(pb.add(i + 48)));

        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
        acc2 = _mm512_fmadd_ps(d2, d2, acc2);
        acc3 = _mm512_fmadd_ps(d3, d3, acc3);
        i += 64;
    }

    while i + 16 <= len {
        let d = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)));
        acc0 = _mm512_fmadd_ps(d, d, acc0);
        i += 16;
    }

    let combined = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    let mut total = _mm512_reduce_add_ps(combined);
    while i < len {
        let d = *pa.add(i) - *pb.add(i);
        total += d * d;
        i += 1;
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let len = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let mut i = 0usize;
    while i + 16 <= len {
        acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
        acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
        acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
        i += 16;
    }
    while i + 4 <= len {
        acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        i += 4;
    }

    let combined = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let mut total = vaddvq_f32(combined);
    while i < len {
        total += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn euclidean_distance_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let len = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let mut i = 0usize;
    while i + 16 <= len {
        let d0 = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        let d1 = vsubq_f32(vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
        let d2 = vsubq_f32(vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
        let d3 = vsubq_f32(vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
        acc2 = vfmaq_f32(acc2, d2, d2);
        acc3 = vfmaq_f32(acc3, d3, d3);
        i += 16;
    }
    while i + 4 <= len {
        let d = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        acc0 = vfmaq_f32(acc0, d, d);
        i += 4;
    }

    let combined = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let mut total = vaddvq_f32(combined);
    while i < len {
        let d = *pa.add(i) - *pb.add(i);
        total += d * d;
        i += 1;
    }
    total
}

/// Computes the dot product of two single-precision float slices using runtime-dispatched
/// hardware SIMD intrinsics (AVX-512F, AVX2/FMA, ARM NEON, or 8-way scalar fallback).
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Slices must have equal length for dot product");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { dot_product_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { dot_product_neon(a, b) };
        }
    }
    dot_product_scalar(a, b)
}

/// Computes squared Euclidean distance ||a - b||^2 using runtime-dispatched hardware SIMD intrinsics.
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Slices must have equal length for Euclidean distance");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { euclidean_distance_squared_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { euclidean_distance_squared_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { euclidean_distance_squared_neon(a, b) };
        }
    }
    euclidean_distance_squared_scalar(a, b)
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

/// Computes similarity scores over a contiguous flat `f32` vector arena (`n * dim`).
pub fn try_batch_distances_flat(
    query: &[f32],
    flat_vectors: &[f32],
    dim: usize,
    metric: Metric,
    is_normalized: bool,
) -> Result<Vec<f32>, SutraError> {
    use rayon::prelude::*;

    if dim == 0 || flat_vectors.is_empty() {
        return Ok(Vec::new());
    }
    if query.len() != dim {
        return Err(SutraError::DimensionMismatch {
            expected: dim,
            got: query.len(),
        });
    }
    let n = flat_vectors.len() / dim;
    if n >= 64 {
        flat_vectors
            .par_chunks_exact(dim)
            .map(|v| compute_vector_score(query, v, metric, is_normalized))
            .collect()
    } else {
        flat_vectors
            .chunks_exact(dim)
            .map(|v| compute_vector_score(query, v, metric, is_normalized))
            .collect()
    }
}

/// Fallible batch distance computation propagating dimension mismatch errors across Rayon threads.
pub fn try_batch_distances(
    query: &[f32],
    vectors: &[Vec<f32>],
    metric: Metric,
    is_normalized: bool,
) -> Result<Vec<f32>, SutraError> {
    use rayon::prelude::*;

    if vectors.len() >= 64 {
        vectors
            .par_iter()
            .map(|v| compute_vector_score(query, v, metric, is_normalized))
            .collect()
    } else {
        vectors
            .iter()
            .map(|v| compute_vector_score(query, v, metric, is_normalized))
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
