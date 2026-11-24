use std::collections::HashMap;
use rand::prelude::*;
use crate::distance::simd::{compute_vector_score, euclidean_distance_squared};
use crate::distance::Metric;
use crate::error::SutraError;

#[derive(Debug, Clone)]
pub struct IVFConfig {
    pub n_lists: usize,
    pub nprobe: usize,
    pub kmeans_iters: usize,
    pub seed: u64,
}

impl Default for IVFConfig {
    fn default() -> Self {
        Self {
            n_lists: 256,
            nprobe: 16,
            kmeans_iters: 25,
            seed: 42,
        }
    }
}

pub fn auto_n_lists(n_vectors: usize) -> usize {
    if n_vectors < 16 {
        return 1;
    }
    let sqrt_n = (n_vectors as f64).sqrt() as usize;
    sqrt_n.clamp(4, 1024)
}

/// Inverted File Flat (IVF-Flat) approximate nearest neighbor index.
#[derive(Debug, Clone)]
pub struct IVFIndex {
    pub config: IVFConfig,
    pub centroids: Vec<Vec<f32>>,
    pub inverted_lists: HashMap<usize, Vec<u32>>,
    pub is_trained: bool,
    pub n_vectors: usize,
}

impl IVFIndex {
    pub fn new(config: IVFConfig) -> Self {
        Self {
            config,
            centroids: Vec::new(),
            inverted_lists: HashMap::new(),
            is_trained: false,
            n_vectors: 0,
        }
    }

    pub fn train(&mut self, vectors: &[Vec<f32>], dimension: usize) -> Result<(), SutraError> {
        let n = vectors.len();
        if n == 0 {
            return Err(SutraError::IndexNotTrained("Cannot train on empty vectors".into()));
        }

        let k_clusters = self.config.n_lists.min(n);
        let mut rng = StdRng::seed_from_u64(self.config.seed);

        // K-Means++ initialization
        let mut centroids = Vec::with_capacity(k_clusters);
        let first_idx = rng.gen_range(0..n);
        centroids.push(vectors[first_idx].clone());

        for _ in 1..k_clusters {
            let mut dists = Vec::with_capacity(n);
            let mut sum_dists = 0.0f64;

            for v in vectors {
                let mut min_d = f32::INFINITY;
                for c in &centroids {
                    let d = euclidean_distance_squared(v, c);
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
            centroids.push(vectors[chosen].clone());
        }

        // Lloyd iterations
        let mut assignments = vec![0usize; n];
        for _ in 0..self.config.kmeans_iters {
            // Assign step
            let mut changed = false;
            for (i, v) in vectors.iter().enumerate() {
                let mut min_d = f32::INFINITY;
                let mut nearest = 0;
                for (c_idx, c) in centroids.iter().enumerate() {
                    let d = euclidean_distance_squared(v, c);
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

            // Update step
            let mut counts = vec![0usize; k_clusters];
            let mut sums = vec![vec![0.0f32; dimension]; k_clusters];

            for (i, v) in vectors.iter().enumerate() {
                let cluster = assignments[i];
                counts[cluster] += 1;
                for d in 0..dimension {
                    sums[cluster][d] += v[d];
                }
            }

            for c_idx in 0..k_clusters {
                if counts[c_idx] > 0 {
                    let count_f = counts[c_idx] as f32;
                    for d in 0..dimension {
                        centroids[c_idx][d] = sums[c_idx][d] / count_f;
                    }
                }
            }
        }

        self.centroids = centroids;
        self.inverted_lists.clear();
        for i in 0..k_clusters {
            self.inverted_lists.insert(i, Vec::new());
        }

        self.is_trained = true;
        self.n_vectors = 0;
        Ok(())
    }

    pub fn add(&mut self, doc_idx: u32, vector: &[f32]) -> Result<(), SutraError> {
        if !self.is_trained || self.centroids.is_empty() {
            return Err(SutraError::IndexNotTrained("IVF index is not trained".into()));
        }

        let mut min_d = f32::INFINITY;
        let mut nearest = 0;
        for (c_idx, c) in self.centroids.iter().enumerate() {
            let d = euclidean_distance_squared(vector, c);
            if d < min_d {
                min_d = d;
                nearest = c_idx;
            }
        }

        self.inverted_lists.entry(nearest).or_default().push(doc_idx);
        self.n_vectors += 1;
        Ok(())
    }

    pub fn search(
        &self,
        query: &[f32],
        all_vectors: &[Vec<f32>],
        k: usize,
        metric: Metric,
        is_normalized: bool,
        mask: Option<&[bool]>,
    ) -> Result<Vec<(u32, f32)>, SutraError> {
        if !self.is_trained || self.centroids.is_empty() {
            return Err(SutraError::IndexNotTrained("IVF index is not trained".into()));
        }

        // 1. Find nprobe nearest centroids
        let mut centroid_scores: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(c_idx, c)| {
                let score = compute_vector_score(query, c, metric, false).unwrap_or(0.0);
                (c_idx, score)
            })
            .collect();

        centroid_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let nprobe = self.config.nprobe.min(centroid_scores.len());

        // 2. Collect candidate doc indices
        let mut candidates = Vec::new();
        for &(c_idx, _) in &centroid_scores[..nprobe] {
            if let Some(doc_list) = self.inverted_lists.get(&c_idx) {
                for &doc_idx in doc_list {
                    let idx = doc_idx as usize;
                    if let Some(m) = mask {
                        if idx >= m.len() || !m[idx] {
                            continue;
                        }
                    }
                    candidates.push(doc_idx);
                }
            }
        }

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // 3. Exact score candidates
        let mut scored: Vec<(u32, f32)> = Vec::with_capacity(candidates.len());
        for &doc_idx in &candidates {
            let idx = doc_idx as usize;
            if idx < all_vectors.len() {
                let s = compute_vector_score(query, &all_vectors[idx], metric, is_normalized)?;
                scored.push((doc_idx, s));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }
}
