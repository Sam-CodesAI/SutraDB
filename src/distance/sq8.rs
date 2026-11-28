use serde::{Deserialize, Serialize};
use crate::distance::Metric;
use crate::error::SutraError;

/// 8-bit Scalar Quantized Vector (SQ8).
/// Compresses 32-bit floats into 8-bit unsigned integers:
/// original ~= min_val + (byte as f32) * scale
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SQ8Vector {
    pub data: Vec<u8>,
    pub min_val: f32,
    pub scale: f32,
}

#[inline(always)]
fn sq8_asymmetric_dot_scalar(data: &[u8], query: &[f32]) -> f32 {
    let len = query.len();
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
            sum0 += *query.get_unchecked(base) * (*data.get_unchecked(base) as f32);
            sum1 += *query.get_unchecked(base + 1) * (*data.get_unchecked(base + 1) as f32);
            sum2 += *query.get_unchecked(base + 2) * (*data.get_unchecked(base + 2) as f32);
            sum3 += *query.get_unchecked(base + 3) * (*data.get_unchecked(base + 3) as f32);
            sum4 += *query.get_unchecked(base + 4) * (*data.get_unchecked(base + 4) as f32);
            sum5 += *query.get_unchecked(base + 5) * (*data.get_unchecked(base + 5) as f32);
            sum6 += *query.get_unchecked(base + 6) * (*data.get_unchecked(base + 6) as f32);
            sum7 += *query.get_unchecked(base + 7) * (*data.get_unchecked(base + 7) as f32);
        }
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);
    let base = chunks * 8;
    for i in 0..remainder {
        unsafe {
            total += *query.get_unchecked(base + i) * (*data.get_unchecked(base + i) as f32);
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
unsafe fn sq8_asymmetric_dot_avx2(data: &[u8], query: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = query.len();
    let pq = query.as_ptr();
    let pd = data.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 32 <= len {
        let u0 = _mm_loadl_epi64(pd.add(i) as *const __m128i);
        let f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u0));
        let q0 = _mm256_loadu_ps(pq.add(i));

        let u1 = _mm_loadl_epi64(pd.add(i + 8) as *const __m128i);
        let f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u1));
        let q1 = _mm256_loadu_ps(pq.add(i + 8));

        let u2 = _mm_loadl_epi64(pd.add(i + 16) as *const __m128i);
        let f2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u2));
        let q2 = _mm256_loadu_ps(pq.add(i + 16));

        let u3 = _mm_loadl_epi64(pd.add(i + 24) as *const __m128i);
        let f3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u3));
        let q3 = _mm256_loadu_ps(pq.add(i + 24));

        acc0 = _mm256_fmadd_ps(q0, f0, acc0);
        acc1 = _mm256_fmadd_ps(q1, f1, acc1);
        acc2 = _mm256_fmadd_ps(q2, f2, acc2);
        acc3 = _mm256_fmadd_ps(q3, f3, acc3);
        i += 32;
    }

    while i + 8 <= len {
        let u = _mm_loadl_epi64(pd.add(i) as *const __m128i);
        let f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u));
        let q = _mm256_loadu_ps(pq.add(i));
        acc0 = _mm256_fmadd_ps(q, f, acc0);
        i += 8;
    }

    let combined = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut total = hsum_m256(combined);
    while i < len {
        total += *pq.add(i) * (*pd.add(i) as f32);
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sq8_asymmetric_dot_avx512(data: &[u8], query: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = query.len();
    let pq = query.as_ptr();
    let pd = data.as_ptr();

    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();

    let mut i = 0usize;
    while i + 64 <= len {
        let u0 = _mm_loadu_si128(pd.add(i) as *const __m128i);
        let f0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u0));
        let q0 = _mm512_loadu_ps(pq.add(i));

        let u1 = _mm_loadu_si128(pd.add(i + 16) as *const __m128i);
        let f1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u1));
        let q1 = _mm512_loadu_ps(pq.add(i + 16));

        let u2 = _mm_loadu_si128(pd.add(i + 32) as *const __m128i);
        let f2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u2));
        let q2 = _mm512_loadu_ps(pq.add(i + 32));

        let u3 = _mm_loadu_si128(pd.add(i + 48) as *const __m128i);
        let f3 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u3));
        let q3 = _mm512_loadu_ps(pq.add(i + 48));

        acc0 = _mm512_fmadd_ps(q0, f0, acc0);
        acc1 = _mm512_fmadd_ps(q1, f1, acc1);
        acc2 = _mm512_fmadd_ps(q2, f2, acc2);
        acc3 = _mm512_fmadd_ps(q3, f3, acc3);
        i += 64;
    }

    while i + 16 <= len {
        let u = _mm_loadu_si128(pd.add(i) as *const __m128i);
        let f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u));
        let q = _mm512_loadu_ps(pq.add(i));
        acc0 = _mm512_fmadd_ps(q, f, acc0);
        i += 16;
    }

    let combined = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    let mut total = _mm512_reduce_add_ps(combined);
    while i < len {
        total += *pq.add(i) * (*pd.add(i) as f32);
        i += 1;
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sq8_asymmetric_dot_neon(data: &[u8], query: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let len = query.len();
    let pq = query.as_ptr();
    let pd = data.as_ptr();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let mut i = 0usize;
    while i + 16 <= len {
        let u8x16 = vld1q_u8(pd.add(i));
        let lo16 = vmovl_u8(vget_low_u8(u8x16));
        let hi16 = vmovl_u8(vget_high_u8(u8x16));

        let f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
        let f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
        let f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
        let f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));

        acc0 = vfmaq_f32(acc0, vld1q_f32(pq.add(i)), f0);
        acc1 = vfmaq_f32(acc1, vld1q_f32(pq.add(i + 4)), f1);
        acc2 = vfmaq_f32(acc2, vld1q_f32(pq.add(i + 8)), f2);
        acc3 = vfmaq_f32(acc3, vld1q_f32(pq.add(i + 12)), f3);
        i += 16;
    }

    let combined = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let mut total = vaddvq_f32(combined);
    while i < len {
        total += *pq.add(i) * (*pd.add(i) as f32);
        i += 1;
    }
    total
}

#[inline]
pub fn sq8_asymmetric_dot_slice(
    data: &[u8],
    min_val: f32,
    scale: f32,
    query: &[f32],
    query_sum: f32,
) -> Result<f32, SutraError> {
    if query.len() != data.len() {
        return Err(SutraError::DimensionMismatch {
            expected: data.len(),
            got: query.len(),
        });
    }
    let sum_qb = {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe { sq8_asymmetric_dot_avx512(data, query) }
            } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { sq8_asymmetric_dot_avx2(data, query) }
            } else {
                sq8_asymmetric_dot_scalar(data, query)
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe { sq8_asymmetric_dot_neon(data, query) }
            } else {
                sq8_asymmetric_dot_scalar(data, query)
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            sq8_asymmetric_dot_scalar(data, query)
        }
    };
    Ok(scale * sum_qb + min_val * query_sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sq8_asymmetric_l2_sq_avx2(
    data: &[u8],
    min_val: f32,
    scale: f32,
    query: &[f32],
) -> f32 {
    use std::arch::x86_64::*;
    let len = query.len();
    let pq = query.as_ptr();
    let pd = data.as_ptr();
    let vmin = _mm256_set1_ps(min_val);
    let vscale = _mm256_set1_ps(scale);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 16 <= len {
        let u0 = _mm_loadl_epi64(pd.add(i) as *const __m128i);
        let f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u0));
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(pq.add(i)), _mm256_fmadd_ps(f0, vscale, vmin));

        let u1 = _mm_loadl_epi64(pd.add(i + 8) as *const __m128i);
        let f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u1));
        let d1 = _mm256_sub_ps(_mm256_loadu_ps(pq.add(i + 8)), _mm256_fmadd_ps(f1, vscale, vmin));

        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        i += 16;
    }

    let mut total = hsum_m256(_mm256_add_ps(acc0, acc1));
    while i < len {
        let orig = min_val + (*pd.add(i) as f32) * scale;
        let d = *pq.add(i) - orig;
        total += d * d;
        i += 1;
    }
    total
}

#[inline]
pub fn sq8_asymmetric_euclidean_squared_slice(
    data: &[u8],
    min_val: f32,
    scale: f32,
    query: &[f32],
) -> Result<f32, SutraError> {
    if query.len() != data.len() {
        return Err(SutraError::DimensionMismatch {
            expected: data.len(),
            got: query.len(),
        });
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Ok(unsafe { sq8_asymmetric_l2_sq_avx2(data, min_val, scale, query) });
        }
    }
    let mut dist_sq = 0.0f32;
    for (i, &q) in query.iter().enumerate() {
        let orig = min_val + (data[i] as f32) * scale;
        let d = q - orig;
        dist_sq += d * d;
    }
    Ok(dist_sq)
}

#[inline]
pub fn sq8_score_slice(
    data: &[u8],
    min_val: f32,
    scale: f32,
    query: &[f32],
    query_sum: f32,
    metric: Metric,
) -> Result<f32, SutraError> {
    match metric {
        Metric::Cosine | Metric::DotProduct => {
            sq8_asymmetric_dot_slice(data, min_val, scale, query, query_sum)
        }
        Metric::Euclidean => {
            let dist_sq = sq8_asymmetric_euclidean_squared_slice(data, min_val, scale, query)?;
            Ok(1.0 / (1.0 + dist_sq.sqrt()))
        }
    }
}

/// Quantizes `v` directly into `out_data`, returning `(min_val, scale)`.
/// Non-finite (`NaN`/`Inf`) components are ignored when computing bounds and quantized to `0`.
#[inline]
pub fn quantize_sq8_into(v: &[f32], out_data: &mut Vec<u8>) -> (f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &x in v {
        if !x.is_finite() {
            continue;
        }
        if x < min_val {
            min_val = x;
        }
        if x > max_val {
            max_val = x;
        }
    }
    if !min_val.is_finite() || !max_val.is_finite() {
        out_data.resize(out_data.len() + v.len(), 0u8);
        return (0.0, 0.0);
    }
    let diff = max_val - min_val;
    let scale = if diff > 1e-12 && diff.is_finite() {
        diff / 255.0
    } else {
        0.0
    };
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

    out_data.reserve(v.len());
    for &x in v {
        let q = if scale > 0.0 && x.is_finite() {
            ((x - min_val) * inv_scale).round().clamp(0.0, 255.0) as u8
        } else {
            0u8
        };
        out_data.push(q);
    }
    (min_val, scale)
}

impl SQ8Vector {
    /// Quantizes a float32 vector slice into an 8-bit unsigned integer vector.
    pub fn quantize(v: &[f32]) -> Self {
        let mut data = Vec::with_capacity(v.len());
        let (min_val, scale) = quantize_sq8_into(v, &mut data);
        Self {
            data,
            min_val,
            scale,
        }
    }

    /// Dequantizes back to an allocated float32 vector.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.data.len());
        for &b in &self.data {
            out.push(self.min_val + (b as f32) * self.scale);
        }
        out
    }

    /// Fallible asymmetric dot product returning DimensionMismatch instead of panicking.
    #[inline]
    pub fn try_asymmetric_dot_product(&self, query: &[f32], query_sum: f32) -> Result<f32, SutraError> {
        sq8_asymmetric_dot_slice(&self.data, self.min_val, self.scale, query, query_sum)
    }

    /// Computes asymmetric dot product between a full-precision query vector (f32)
    /// and this quantized vector (u8).
    #[inline]
    pub fn asymmetric_dot_product(&self, query: &[f32], query_sum: f32) -> f32 {
        self.try_asymmetric_dot_product(query, query_sum)
            .expect("Dimension mismatch in SQ8Vector::asymmetric_dot_product")
    }

    /// Fallible asymmetric squared Euclidean distance returning DimensionMismatch instead of panicking.
    #[inline]
    pub fn try_asymmetric_euclidean_squared(&self, query: &[f32]) -> Result<f32, SutraError> {
        sq8_asymmetric_euclidean_squared_slice(&self.data, self.min_val, self.scale, query)
    }

    /// Computes asymmetric squared Euclidean distance ||query - dequantized||^2
    #[inline]
    pub fn asymmetric_euclidean_squared(&self, query: &[f32]) -> f32 {
        self.try_asymmetric_euclidean_squared(query)
            .expect("Dimension mismatch in SQ8Vector::asymmetric_euclidean_squared")
    }

    /// Fallible asymmetric similarity score according to the specified Metric.
    #[inline]
    pub fn try_score(&self, query: &[f32], query_sum: f32, metric: Metric) -> Result<f32, SutraError> {
        sq8_score_slice(&self.data, self.min_val, self.scale, query, query_sum, metric)
    }

    /// Computes asymmetric similarity score according to the specified Metric.
    #[inline]
    pub fn score(&self, query: &[f32], query_sum: f32, metric: Metric) -> f32 {
        self.try_score(query, query_sum, metric)
            .expect("Dimension mismatch in SQ8Vector::score")
    }
}
