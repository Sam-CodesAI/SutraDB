use serde::{Deserialize, Serialize};
use rand::prelude::*;
use rayon::prelude::*;
use crate::distance::simd::{dot_product, euclidean_distance_squared};
use crate::distance::Metric;
use crate::error::SutraError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    pub m: usize,            // Number of sub-vectors/sub-spaces
    pub k: usize,            // Centroids per sub-space (default: 256 for 1-byte encoding)
    pub kmeans_iters: usize, // Iterations for training
    pub seed: u64,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            m: 8,
            k: 256,
            kmeans_iters: 20,
            seed: 42,
        }
    }
}

/// Product Quantization (PQ) Codebook.
/// Decomposes D-dimensional vector space into M sub-spaces with K centroids each.
/// Reduces 32-bit float vectors to M bytes (up to 64x compression) and accelerates
/// distance calculation via Asymmetric Distance Computation (ADC) lookup tables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQCodebook {
    pub dimension: usize,
    pub m: usize,
    pub k: usize,
    pub sub_dim: usize,
    // centroids[m] has length (k * sub_dim)
    pub centroids: Vec<Vec<f32>>,
}

/// Product Quantized Vector containing M bytes of centroid indices.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PQVector {
    pub codes: Vec<u8>,
}

#[inline(always)]
fn dot_inline(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

#[inline(always)]
fn l2_sq_inline(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Trains one subspace codebook over a contiguous `sample_n * sub_dim` slab using incremental
/// O(k * n) k-means++ initialization followed by Lloyd iterations.
fn train_subspace(
    sub_flat: &[f32],
    sample_n: usize,
    sub_dim: usize,
    k_clusters: usize,
    kmeans_iters: usize,
    seed: u64,
) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids_flat = vec![0.0f32; k_clusters * sub_dim];

    let first_idx = rng.gen_range(0..sample_n);
    centroids_flat[0..sub_dim]
        .copy_from_slice(&sub_flat[first_idx * sub_dim..(first_idx + 1) * sub_dim]);

    // Maintain incremental min squared distances to selected centroids: O(k * n) instead of O(k^2 * n).
    let mut min_dists = vec![f32::INFINITY; sample_n];
    for c_count in 1..k_clusters {
        let newest = &centroids_flat[(c_count - 1) * sub_dim..c_count * sub_dim];
        let mut sum_dists = 0.0f64;
        for (i, d_slot) in min_dists.iter_mut().enumerate() {
            let sv = &sub_flat[i * sub_dim..(i + 1) * sub_dim];
            let d = l2_sq_inline(sv, newest);
            if d < *d_slot {
                *d_slot = d;
            }
            sum_dists += (*d_slot as f64).max(1e-12);
        }

        let threshold = rng.gen::<f64>() * sum_dists;
        let mut acc = 0.0f64;
        let mut chosen = 0;
        for (idx, &d) in min_dists.iter().enumerate() {
            acc += (d as f64).max(1e-12);
            if acc >= threshold {
                chosen = idx;
                break;
            }
        }
        centroids_flat[c_count * sub_dim..(c_count + 1) * sub_dim]
            .copy_from_slice(&sub_flat[chosen * sub_dim..(chosen + 1) * sub_dim]);
    }

    // Lloyd k-means iterations using argmin_c (0.5 * ||c||^2 - dot(v, c)) and <0.5% churn early stop
    let mut assignments = vec![0usize; sample_n];
    let mut counts = vec![0usize; k_clusters];
    let mut new_centroids = vec![0.0f32; k_clusters * sub_dim];
    let mut half_norms_sq = vec![0.0f32; k_clusters];
    let min_changes = if sample_n >= 200 { sample_n / 200 } else { 1 };

    for iter in 0..kmeans_iters {
        for (c_idx, norm_slot) in half_norms_sq.iter_mut().enumerate() {
            let c = &centroids_flat[c_idx * sub_dim..(c_idx + 1) * sub_dim];
            *norm_slot = 0.5 * dot_inline(c, c);
        }

        let mut changed_count = 0usize;
        for (i, assign_slot) in assignments.iter_mut().enumerate() {
            let sv = &sub_flat[i * sub_dim..(i + 1) * sub_dim];
            let mut min_obj = f32::INFINITY;
            let mut nearest = 0;
            for (c_idx, &half_norm) in half_norms_sq.iter().enumerate() {
                let c = &centroids_flat[c_idx * sub_dim..(c_idx + 1) * sub_dim];
                let obj = half_norm - dot_inline(sv, c);
                if obj < min_obj {
                    min_obj = obj;
                    nearest = c_idx;
                }
            }
            if iter == 0 || *assign_slot != nearest {
                *assign_slot = nearest;
                changed_count += 1;
            }
        }

        if iter > 0 && changed_count < min_changes {
            break;
        }

        counts.fill(0);
        new_centroids.fill(0.0);

        for (i, &cluster) in assignments.iter().enumerate() {
            counts[cluster] += 1;
            let sv = &sub_flat[i * sub_dim..(i + 1) * sub_dim];
            let dst = &mut new_centroids[cluster * sub_dim..(cluster + 1) * sub_dim];
            for (d, &val) in dst.iter_mut().zip(sv.iter()) {
                *d += val;
            }
        }

        for (c, &cnt) in counts.iter().enumerate() {
            if cnt > 0 {
                let inv_cnt = 1.0 / cnt as f32;
                let src = &new_centroids[c * sub_dim..(c + 1) * sub_dim];
                let dst = &mut centroids_flat[c * sub_dim..(c + 1) * sub_dim];
                for (out_v, &in_v) in dst.iter_mut().zip(src.iter()) {
                    *out_v = in_v * inv_cnt;
                }
            }
        }
    }

    centroids_flat
}

impl PQCodebook {
    /// Suggests an optimal sub-space count M that evenly divides the vector dimension.
    pub fn suggest_m(dimension: usize) -> usize {
        for cand in [64, 48, 32, 24, 16, 8, 4, 2] {
            if cand <= dimension && dimension.is_multiple_of(cand) {
                return cand;
            }
        }
        1
    }

    /// Trains a Product Quantization codebook over a contiguous `n * dimension` flat buffer.
    /// Subsamples up to `min(n, 256 * k)` vectors and trains all `m` subspaces in parallel via Rayon.
    pub fn train_flat(
        flat_vectors: &[f32],
        n: usize,
        dimension: usize,
        config: PQConfig,
    ) -> Result<Self, SutraError> {
        if n == 0 || flat_vectors.is_empty() {
            return Err(SutraError::IndexNotTrained(
                "Cannot train PQ on empty vectors".into(),
            ));
        }
        if flat_vectors.len() != n * dimension {
            return Err(SutraError::DimensionMismatch {
                expected: n * dimension,
                got: flat_vectors.len(),
            });
        }

        let m = config.m;
        if m == 0 {
            return Err(SutraError::InvalidQuery(
                "PQ sub-spaces M must be greater than 0".into(),
            ));
        }
        if dimension == 0 || m > dimension || !dimension.is_multiple_of(m) {
            return Err(SutraError::InvalidQuery(format!(
                "Vector dimension {} must be non-zero, >= M, and divisible by PQ sub-spaces M={}",
                dimension, m
            )));
        }

        let sub_dim = dimension / m;
        let k_clusters = config.k.min(n).clamp(1, 256);
        let sample_cap = 256usize.saturating_mul(k_clusters);
        let sample_n = n.min(sample_cap).max(1);

        // Build evenly strided training sample indices cleanly in [0, n)
        let sample_indices: Vec<usize> = if sample_n == n {
            (0..n).collect()
        } else {
            (0..sample_n).map(|i| (i * n) / sample_n).collect()
        };

        let all_sub_centroids: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|m_idx| {
                let start = m_idx * sub_dim;
                let mut sub_flat = Vec::with_capacity(sample_n * sub_dim);
                for &idx in &sample_indices {
                    let row_start = idx * dimension + start;
                    sub_flat.extend_from_slice(&flat_vectors[row_start..row_start + sub_dim]);
                }
                train_subspace(
                    &sub_flat,
                    sample_n,
                    sub_dim,
                    k_clusters,
                    config.kmeans_iters,
                    config.seed.wrapping_add(m_idx as u64),
                )
            })
            .collect();

        Ok(Self {
            dimension,
            m,
            k: k_clusters,
            sub_dim,
            centroids: all_sub_centroids,
        })
    }

    /// Trains a Product Quantization codebook using k-means++ across M sub-spaces.
    pub fn train(vectors: &[Vec<f32>], dimension: usize, config: PQConfig) -> Result<Self, SutraError> {
        let n = vectors.len();
        if n == 0 {
            return Err(SutraError::IndexNotTrained(
                "Cannot train PQ on empty vectors".into(),
            ));
        }
        for v in vectors {
            if v.len() != dimension {
                return Err(SutraError::DimensionMismatch {
                    expected: dimension,
                    got: v.len(),
                });
            }
        }
        let mut flat = Vec::with_capacity(n * dimension);
        for v in vectors {
            flat.extend_from_slice(v);
        }
        Self::train_flat(&flat, n, dimension, config)
    }

    /// Encodes `vector` directly into an existing `out_codes` slice of length `self.m`.
    #[inline]
    pub fn try_encode_into(&self, vector: &[f32], out_codes: &mut [u8]) -> Result<(), SutraError> {
        if vector.len() != self.dimension {
            return Err(SutraError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }
        if out_codes.len() != self.m {
            return Err(SutraError::DimensionMismatch {
                expected: self.m,
                got: out_codes.len(),
            });
        }

        for (m_idx, code_slot) in out_codes.iter_mut().enumerate() {
            let start = m_idx * self.sub_dim;
            let sub_v = &vector[start..start + self.sub_dim];
            let c_flat = &self.centroids[m_idx];

            let mut min_d = f32::INFINITY;
            let mut best_c = 0u8;

            for c in 0..self.k {
                let c_start = c * self.sub_dim;
                let centroid = &c_flat[c_start..c_start + self.sub_dim];
                let d = l2_sq_inline(sub_v, centroid);
                if d < min_d {
                    min_d = d;
                    best_c = c as u8;
                }
            }
            *code_slot = best_c;
        }
        Ok(())
    }

    /// Parallel batch-encodes a contiguous FP32 buffer into a contiguous `n * m` `Vec<u8>` slab.
    pub fn try_encode_flat(&self, flat_vectors: &[f32], n: usize) -> Result<Vec<u8>, SutraError> {
        if flat_vectors.len() != n * self.dimension {
            return Err(SutraError::DimensionMismatch {
                expected: n * self.dimension,
                got: flat_vectors.len(),
            });
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let half_norms: Vec<Vec<f32>> = (0..self.m)
            .map(|m_idx| {
                let c_flat = &self.centroids[m_idx];
                (0..self.k)
                    .map(|c| {
                        let c_slice = &c_flat[c * self.sub_dim..(c + 1) * self.sub_dim];
                        0.5 * dot_inline(c_slice, c_slice)
                    })
                    .collect()
            })
            .collect();

        let mut codes = vec![0u8; n * self.m];
        codes
            .par_chunks_exact_mut(self.m)
            .zip(flat_vectors.par_chunks_exact(self.dimension))
            .for_each(|(out_codes, vec_slice)| {
                for (m_idx, code_slot) in out_codes.iter_mut().enumerate() {
                    let start = m_idx * self.sub_dim;
                    let sub_v = &vec_slice[start..start + self.sub_dim];
                    let c_flat = &self.centroids[m_idx];
                    let hn = &half_norms[m_idx];
                    let mut min_obj = f32::INFINITY;
                    let mut best_c = 0u8;
                    for (c, &half_norm) in hn.iter().enumerate() {
                        let centroid = &c_flat[c * self.sub_dim..(c + 1) * self.sub_dim];
                        let obj = half_norm - dot_inline(sub_v, centroid);
                        if obj < min_obj {
                            min_obj = obj;
                            best_c = c as u8;
                        }
                    }
                    *code_slot = best_c;
                }
            });
        Ok(codes)
    }

    /// Fallible encoding of a full-precision vector into M bytes of centroid indices.
    pub fn try_encode(&self, vector: &[f32]) -> Result<PQVector, SutraError> {
        let mut codes = vec![0u8; self.m];
        self.try_encode_into(vector, &mut codes)?;
        Ok(PQVector { codes })
    }

    /// Encodes a full-precision vector into M bytes of centroid indices.
    pub fn encode(&self, vector: &[f32]) -> PQVector {
        self.try_encode(vector)
            .expect("Dimension mismatch in PQCodebook::encode")
    }

    /// Decodes (reconstructs) an approximate vector from its M-byte codes.
    pub fn decode(&self, pq_vec: &PQVector) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.dimension);
        for (m_idx, &code) in pq_vec.codes.iter().enumerate() {
            if m_idx >= self.m {
                break;
            }
            let c = (code as usize).min(self.k - 1);
            let c_start = c * self.sub_dim;
            let c_end = c_start + self.sub_dim;
            out.extend_from_slice(&self.centroids[m_idx][c_start..c_end]);
        }
        out
    }

    /// Fallible computation of the Asymmetric Distance Computation (ADC) Lookup Table (LUT).
    pub fn try_compute_lut(&self, query: &[f32], metric: Metric) -> Result<Vec<f32>, SutraError> {
        if query.len() != self.dimension {
            return Err(SutraError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }
        let mut lut = vec![0.0f32; self.m * self.k];

        for m_idx in 0..self.m {
            let start = m_idx * self.sub_dim;
            let q_sub = &query[start..start + self.sub_dim];

            for c in 0..self.k {
                let c_start = c * self.sub_dim;
                let centroid = &self.centroids[m_idx][c_start..c_start + self.sub_dim];

                let val = match metric {
                    Metric::Cosine | Metric::DotProduct => dot_product(q_sub, centroid),
                    Metric::Euclidean => euclidean_distance_squared(q_sub, centroid),
                };

                lut[m_idx * self.k + c] = val;
            }
        }

        Ok(lut)
    }

    /// Computes the Asymmetric Distance Computation (ADC) Lookup Table (LUT) for a query vector.
    /// Shape: [M * K] floats.
    pub fn compute_lut(&self, query: &[f32], metric: Metric) -> Vec<f32> {
        self.try_compute_lut(query, metric)
            .expect("Dimension mismatch in PQCodebook::compute_lut")
    }

    /// Fallible evaluation of asymmetric distance/score between a query LUT and a raw `codes` slice.
    #[inline]
    pub fn try_score_adc_slice(
        &self,
        codes: &[u8],
        lut: &[f32],
        metric: Metric,
    ) -> Result<f32, SutraError> {
        if codes.len() != self.m {
            return Err(SutraError::DimensionMismatch {
                expected: self.m,
                got: codes.len(),
            });
        }
        let required_lut = self.m * self.k;
        if lut.len() < required_lut {
            return Err(SutraError::DimensionMismatch {
                expected: required_lut,
                got: lut.len(),
            });
        }

        let mut sum = 0.0f32;
        for (m_idx, &code) in codes.iter().enumerate() {
            let c = (code as usize).min(self.k - 1);
            sum += unsafe { *lut.get_unchecked(m_idx * self.k + c) };
        }

        Ok(match metric {
            Metric::Cosine | Metric::DotProduct => sum,
            Metric::Euclidean => 1.0 / (1.0 + sum.max(0.0).sqrt()),
        })
    }

    /// Fallible evaluation of asymmetric distance/score between a query LUT and an encoded PQ vector.
    #[inline]
    pub fn try_score_adc(&self, pq_vec: &PQVector, lut: &[f32], metric: Metric) -> Result<f32, SutraError> {
        self.try_score_adc_slice(&pq_vec.codes, lut, metric)
    }

    /// Evaluates the asymmetric distance/score between a query LUT and an encoded PQ vector.
    #[inline]
    pub fn score_adc(&self, pq_vec: &PQVector, lut: &[f32], metric: Metric) -> f32 {
        self.try_score_adc(pq_vec, lut, metric)
            .expect("Dimension mismatch in PQCodebook::score_adc")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_training_and_encoding() {
        let dim = 8;
        let n = 40;
        let mut vectors = Vec::new();

        for i in 0..n {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            v[(i + 1) % dim] = 0.5;
            vectors.push(v);
        }

        let config = PQConfig {
            m: 4,
            k: 16,
            kmeans_iters: 10,
            seed: 42,
        };

        let codebook = PQCodebook::train(&vectors, dim, config).unwrap();
        assert_eq!(codebook.m, 4);
        assert_eq!(codebook.sub_dim, 2);
        assert_eq!(codebook.k, 16);

        let encoded = codebook.encode(&vectors[0]);
        assert_eq!(encoded.codes.len(), 4);

        let decoded = codebook.decode(&encoded);
        assert_eq!(decoded.len(), dim);

        let lut = codebook.compute_lut(&vectors[0], Metric::DotProduct);
        assert_eq!(lut.len(), 4 * 16);

        let adc_score = codebook.score_adc(&encoded, &lut, Metric::DotProduct);
        assert!(adc_score > 0.5);
    }
}
