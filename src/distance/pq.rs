use serde::{Deserialize, Serialize};
use rand::prelude::*;
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

impl PQCodebook {
    /// Suggests an optimal sub-space count M that evenly divides the vector dimension.
    pub fn suggest_m(dimension: usize) -> usize {
        for cand in [64, 48, 32, 24, 16, 8, 4, 2] {
            if cand <= dimension && dimension % cand == 0 {
                return cand;
            }
        }
        1
    }

    /// Trains a Product Quantization codebook using k-means++ across M sub-spaces.
    pub fn train(vectors: &[Vec<f32>], dimension: usize, config: PQConfig) -> Result<Self, SutraError> {
        let n = vectors.len();
        if n == 0 {
            return Err(SutraError::IndexNotTrained("Cannot train PQ on empty vectors".into()));
        }

        let m = config.m;
        if dimension % m != 0 {
            return Err(SutraError::InvalidQuery(format!(
                "Vector dimension {} must be divisible by PQ sub-spaces M={}",
                dimension, m
            )));
        }

        let sub_dim = dimension / m;
        let k_clusters = config.k.min(n).max(1);
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut all_sub_centroids = Vec::with_capacity(m);

        for m_idx in 0..m {
            let start = m_idx * sub_dim;
            let end = start + sub_dim;

            // Extract m-th sub-vectors
            let sub_vecs: Vec<Vec<f32>> = vectors.iter().map(|v| v[start..end].to_vec()).collect();

            // K-Means++ initialization for this sub-space
            let mut sub_centroids: Vec<Vec<f32>> = Vec::with_capacity(k_clusters);
            let first_idx = rng.gen_range(0..n);
            sub_centroids.push(sub_vecs[first_idx].clone());

            for _ in 1..k_clusters {
                let mut dists = Vec::with_capacity(n);
                let mut sum_dists = 0.0f64;

                for sv in &sub_vecs {
                    let mut min_d = f32::INFINITY;
                    for c in &sub_centroids {
                        let d = euclidean_distance_squared(sv, c);
                        if d < min_d {
                            min_d = d;
                        }
                    }
                    let d_f64 = (min_d as f64).max(1e-12);
                    dists.push(d_f64);
                    sum_dists += d_f64;
                }

                let threshold = rng.gen::<f64>() * sum_dists;
                let mut acc = 0.0f64;
                let mut chosen = 0;
                for (idx, &d) in dists.iter().enumerate() {
                    acc += d;
                    if acc >= threshold {
                        chosen = idx;
                        break;
                    }
                }
                sub_centroids.push(sub_vecs[chosen].clone());
            }

            // Lloyd k-means iterations
            let mut assignments = vec![0usize; n];
            for _ in 0..config.kmeans_iters {
                let mut changed = false;
                for (i, sv) in sub_vecs.iter().enumerate() {
                    let mut min_d = f32::INFINITY;
                    let mut nearest = 0;
                    for (c_idx, c) in sub_centroids.iter().enumerate() {
                        let d = euclidean_distance_squared(sv, c);
                        if d < min_d {
                            min_d = d;
                            nearest = c_idx;
                        }
                    }
                    if assignments[i] != nearest {
                        assignments[i] = nearest;
                        changed = true;
                    }
                }

                if !changed {
                    break;
                }

                // Update centroids
                let mut counts = vec![0usize; k_clusters];
                let mut new_centroids = vec![vec![0.0f32; sub_dim]; k_clusters];

                for (i, &cluster) in assignments.iter().enumerate() {
                    counts[cluster] += 1;
                    for d in 0..sub_dim {
                        new_centroids[cluster][d] += sub_vecs[i][d];
                    }
                }

                for c in 0..k_clusters {
                    if counts[c] > 0 {
                        let inv_cnt = 1.0 / counts[c] as f32;
                        for d in 0..sub_dim {
                            new_centroids[c][d] *= inv_cnt;
                        }
                        sub_centroids[c] = new_centroids[c].clone();
                    }
                }
            }

            // Flatten sub_centroids into contiguous slice
            let mut flat = Vec::with_capacity(k_clusters * sub_dim);
            for c in sub_centroids {
                flat.extend_from_slice(&c);
            }
            all_sub_centroids.push(flat);
        }

        Ok(Self {
            dimension,
            m,
            k: k_clusters,
            sub_dim,
            centroids: all_sub_centroids,
        })
    }

    /// Encodes a full-precision vector into M bytes of centroid indices.
    pub fn encode(&self, vector: &[f32]) -> PQVector {
        let mut codes = Vec::with_capacity(self.m);

        for m_idx in 0..self.m {
            let start = m_idx * self.sub_dim;
            let end = start + self.sub_dim;
            let sub_v = &vector[start..end];

            let mut min_d = f32::INFINITY;
            let mut best_c = 0u8;

            for c in 0..self.k {
                let c_start = c * self.sub_dim;
                let c_end = c_start + self.sub_dim;
                let centroid = &self.centroids[m_idx][c_start..c_end];

                let d = euclidean_distance_squared(sub_v, centroid);
                if d < min_d {
                    min_d = d;
                    best_c = c as u8;
                }
            }

            codes.push(best_c);
        }

        PQVector { codes }
    }

    /// Decodes (reconstructs) an approximate vector from its M-byte codes.
    pub fn decode(&self, pq_vec: &PQVector) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.dimension);
        for (m_idx, &code) in pq_vec.codes.iter().enumerate() {
            let c = (code as usize).min(self.k - 1);
            let c_start = c * self.sub_dim;
            let c_end = c_start + self.sub_dim;
            out.extend_from_slice(&self.centroids[m_idx][c_start..c_end]);
        }
        out
    }

    /// Computes the Asymmetric Distance Computation (ADC) Lookup Table (LUT) for a query vector.
    /// Shape: [M * K] floats.
    pub fn compute_lut(&self, query: &[f32], metric: Metric) -> Vec<f32> {
        let mut lut = vec![0.0f32; self.m * self.k];

        for m_idx in 0..self.m {
            let start = m_idx * self.sub_dim;
            let end = start + self.sub_dim;
            let q_sub = &query[start..end];

            for c in 0..self.k {
                let c_start = c * self.sub_dim;
                let c_end = c_start + self.sub_dim;
                let centroid = &self.centroids[m_idx][c_start..c_end];

                let val = match metric {
                    Metric::Cosine | Metric::DotProduct => dot_product(q_sub, centroid),
                    Metric::Euclidean => euclidean_distance_squared(q_sub, centroid),
                };

                lut[m_idx * self.k + c] = val;
            }
        }

        lut
    }

    /// Evaluates the asymmetric distance/score between a query LUT and an encoded PQ vector.
    #[inline]
    pub fn score_adc(&self, pq_vec: &PQVector, lut: &[f32], metric: Metric) -> f32 {
        let mut sum = 0.0f32;
        for (m_idx, &code) in pq_vec.codes.iter().enumerate() {
            let c = (code as usize).min(self.k - 1);
            sum += lut[m_idx * self.k + c];
        }

        match metric {
            Metric::Cosine | Metric::DotProduct => sum,
            Metric::Euclidean => 1.0 / (1.0 + sum.max(0.0).sqrt()),
        }
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
            m: 4, // 4 sub-spaces of 2 dimensions each
            k: 16,
            kmeans_iters: 10,
            seed: 42,
        };

        let codebook = PQCodebook::train(&vectors, dim, config).unwrap();
        assert_eq!(codebook.m, 4);
        assert_eq!(codebook.sub_dim, 2);
        assert_eq!(codebook.k, 16);

        // Encode vector 0
        let encoded = codebook.encode(&vectors[0]);
        assert_eq!(encoded.codes.len(), 4);

        // Decode vector 0
        let decoded = codebook.decode(&encoded);
        assert_eq!(decoded.len(), dim);

        // ADC LUT
        let lut = codebook.compute_lut(&vectors[0], Metric::DotProduct);
        assert_eq!(lut.len(), 4 * 16);

        let adc_score = codebook.score_adc(&encoded, &lut, Metric::DotProduct);
        assert!(adc_score > 0.5);
    }
}
