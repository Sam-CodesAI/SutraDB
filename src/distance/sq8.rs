use serde::{Deserialize, Serialize};

/// 8-bit Scalar Quantized Vector (SQ8).
/// Compresses 32-bit floats into 8-bit unsigned integers:
/// original ~= min_val + (byte as f32) * scale
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SQ8Vector {
    pub data: Vec<u8>,
    pub min_val: f32,
    pub scale: f32,
}

impl SQ8Vector {
    /// Quantizes a float32 vector slice into an 8-bit unsigned integer vector.
    pub fn quantize(v: &[f32]) -> Self {
        if v.is_empty() {
            return Self {
                data: Vec::new(),
                min_val: 0.0,
                scale: 0.0,
            };
        }

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for &x in v {
            if x < min_val {
                min_val = x;
            }
            if x > max_val {
                max_val = x;
            }
        }

        let diff = max_val - min_val;
        let scale = if diff > 1e-12 { diff / 255.0 } else { 0.0 };
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let mut data = Vec::with_capacity(v.len());
        for &x in v {
            let q = if scale > 0.0 {
                let val = ((x - min_val) * inv_scale).round();
                val.clamp(0.0, 255.0) as u8
            } else {
                0u8
            };
            data.push(q);
        }

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

    /// Computes asymmetric dot product between a full-precision query vector (f32)
    /// and this quantized vector (u8).
    /// Formula: sum(q_i * (min + b_i * scale)) = scale * sum(q_i * b_i) + min * sum(q_i)
    #[inline]
    pub fn asymmetric_dot_product(&self, query: &[f32], query_sum: f32) -> f32 {
        assert_eq!(query.len(), self.data.len());
        let len = query.len();
        let mut sum_qb = 0.0f32;

        let chunks = len / 8;
        let remainder = len % 8;

        for i in 0..chunks {
            let base = i * 8;
            unsafe {
                sum_qb += query.get_unchecked(base) * (*self.data.get_unchecked(base) as f32);
                sum_qb += query.get_unchecked(base + 1) * (*self.data.get_unchecked(base + 1) as f32);
                sum_qb += query.get_unchecked(base + 2) * (*self.data.get_unchecked(base + 2) as f32);
                sum_qb += query.get_unchecked(base + 3) * (*self.data.get_unchecked(base + 3) as f32);
                sum_qb += query.get_unchecked(base + 4) * (*self.data.get_unchecked(base + 4) as f32);
                sum_qb += query.get_unchecked(base + 5) * (*self.data.get_unchecked(base + 5) as f32);
                sum_qb += query.get_unchecked(base + 6) * (*self.data.get_unchecked(base + 6) as f32);
                sum_qb += query.get_unchecked(base + 7) * (*self.data.get_unchecked(base + 7) as f32);
            }
        }

        let base = chunks * 8;
        for i in 0..remainder {
            unsafe {
                sum_qb += query.get_unchecked(base + i) * (*self.data.get_unchecked(base + i) as f32);
            }
        }

        self.scale * sum_qb + self.min_val * query_sum
    }

    /// Computes asymmetric squared Euclidean distance ||query - dequantized||^2
    #[inline]
    pub fn asymmetric_euclidean_squared(&self, query: &[f32]) -> f32 {
        assert_eq!(query.len(), self.data.len());
        let mut dist_sq = 0.0f32;
        for (i, &q) in query.iter().enumerate() {
            let orig = self.min_val + (self.data[i] as f32) * self.scale;
            let d = q - orig;
            dist_sq += d * d;
        }
        dist_sq
    }
}
