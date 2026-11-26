use std::collections::HashMap;
use rand::prelude::*;
use crate::distance::simd::{compute_vector_score, euclidean_distance_squared, normalize_in_place};
use crate::distance::{Metric, VectorStore};
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
/// Uses spherical k-means (normalized centroids + cosine similarity) for `Metric::Cosine`
/// and Euclidean k-means Voronoi partitioning for `Metric::Euclidean` / `Metric::DotProduct`.
#[derive(Debug, Clone)]
pub struct IVFIndex {
    pub config: IVFConfig,
    pub metric: Metric,
    pub centroids: Vec<Vec<f32>>,
    pub inverted_lists: HashMap<usize, Vec<u32>>,
    pub is_trained: bool,
    pub n_vectors: usize,
}

impl IVFIndex {
    pub fn new(config: IVFConfig) -> Self {
        Self::with_metric(config, Metric::Euclidean)
    }

    pub fn with_metric(config: IVFConfig, metric: Metric) -> Self {
        Self {
            config,
            metric,
            centroids: Vec::new(),
            inverted_lists: HashMap::new(),
            is_trained: false,
            n_vectors: 0,
        }
    }

    pub fn train_store<V: VectorStore + ?Sized>(
        &mut self,
        vectors: &V,
        dimension: usize,
    ) -> Result<(), SutraError> {
        let n = vectors.vec_len();
        if n == 0 {
            return Err(SutraError::IndexNotTrained("Cannot train on empty vectors".into()));
        }
        if dimension == 0 {
            return Err(SutraError::InvalidQuery("Dimension must be greater than 0".into()));
        }
        for i in 0..n {
            let v = vectors.vec_slice(i);
            if v.len() != dimension {
                return Err(SutraError::DimensionMismatch {
                    expected: dimension,
                    got: v.len(),
                });
            }
        }

        let k_clusters = self.config.n_lists.min(n).max(1);
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let use_cosine = self.metric == Metric::Cosine;

        // K-Means++ initialization with incremental min_dists (O(k * n))
        let mut centroids = Vec::with_capacity(k_clusters);
        let first_idx = rng.gen_range(0..n);
        let mut first_c = vectors.vec_slice(first_idx).to_vec();
        if use_cosine {
            normalize_in_place(&mut first_c);
        }
        centroids.push(first_c);

        let mut min_dists = vec![f32::INFINITY; n];
        for _ in 1..k_clusters {
            let newest = centroids.last().unwrap();
            let mut sum_dists = 0.0f64;

            for (i, d_slot) in min_dists.iter_mut().enumerate() {
                let v = vectors.vec_slice(i);
                let d = if use_cosine {
                    let sim = compute_vector_score(v, newest, Metric::Cosine, false)?;
                    (1.0 - sim).max(0.0)
                } else {
                    euclidean_distance_squared(v, newest)
                };
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
            let mut next_c = vectors.vec_slice(chosen).to_vec();
            if use_cosine {
                normalize_in_place(&mut next_c);
            }
            centroids.push(next_c);
        }

        // Lloyd iterations
        let mut assignments = vec![0usize; n];
        let mut counts = vec![0usize; k_clusters];
        let mut sums = vec![vec![0.0f32; dimension]; k_clusters];
        for _ in 0..self.config.kmeans_iters {
            let mut changed = false;
            for (i, assign_slot) in assignments.iter_mut().enumerate() {
                let v = vectors.vec_slice(i);
                let mut min_d = f32::INFINITY;
                let mut nearest = 0;
                for (c_idx, c) in centroids.iter().enumerate() {
                    let d = if use_cosine {
                        let sim = compute_vector_score(v, c, Metric::Cosine, false)?;
                        (1.0 - sim).max(0.0)
                    } else {
                        euclidean_distance_squared(v, c)
                    };
                    if d < min_d {
                        min_d = d;
                        nearest = c_idx;
                    }
                }
                if *assign_slot != nearest {
                    *assign_slot = nearest;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            counts.fill(0);
            for s in &mut sums {
                s.fill(0.0);
            }

            for (i, &cluster) in assignments.iter().enumerate() {
                let v = vectors.vec_slice(i);
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
                    if use_cosine {
                        normalize_in_place(&mut centroids[c_idx]);
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

    pub fn train(&mut self, vectors: &[Vec<f32>], dimension: usize) -> Result<(), SutraError> {
        self.train_store(vectors, dimension)
    }

    pub fn add(&mut self, doc_idx: u32, vector: &[f32]) -> Result<(), SutraError> {
        if !self.is_trained || self.centroids.is_empty() {
            return Err(SutraError::IndexNotTrained("IVF index is not trained".into()));
        }
        let dim = self.centroids[0].len();
        if vector.len() != dim {
            return Err(SutraError::DimensionMismatch {
                expected: dim,
                got: vector.len(),
            });
        }

        let use_cosine = self.metric == Metric::Cosine;
        let mut min_d = f32::INFINITY;
        let mut nearest = 0;
        for (c_idx, c) in self.centroids.iter().enumerate() {
            let d = if use_cosine {
                let sim = compute_vector_score(vector, c, Metric::Cosine, false)?;
                (1.0 - sim).max(0.0)
            } else {
                euclidean_distance_squared(vector, c)
            };
            if d < min_d {
                min_d = d;
                nearest = c_idx;
            }
        }

        self.inverted_lists.entry(nearest).or_default().push(doc_idx);
        self.n_vectors += 1;
        Ok(())
    }

    pub fn search<V: VectorStore + ?Sized>(
        &self,
        query: &[f32],
        all_vectors: &V,
        k: usize,
        metric: Metric,
        is_normalized: bool,
        mask: Option<&[bool]>,
    ) -> Result<Vec<(u32, f32)>, SutraError> {
        if !self.is_trained || self.centroids.is_empty() {
            return Err(SutraError::IndexNotTrained("IVF index is not trained".into()));
        }
        let dim = self.centroids[0].len();
        if query.len() != dim {
            return Err(SutraError::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut centroid_scores: Vec<(usize, f32)> = Vec::with_capacity(self.centroids.len());
        for (c_idx, c) in self.centroids.iter().enumerate() {
            let score = compute_vector_score(query, c, metric, false)?;
            centroid_scores.push((c_idx, score));
        }

        centroid_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let nprobe = self.config.nprobe.min(centroid_scores.len());

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

        let total_vecs = all_vectors.vec_len();
        let mut scored: Vec<(u32, f32)> = Vec::with_capacity(candidates.len());
        for &doc_idx in &candidates {
            let idx = doc_idx as usize;
            if idx < total_vecs {
                let s = compute_vector_score(query, all_vectors.vec_slice(idx), metric, is_normalized)?;
                scored.push((doc_idx, s));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }
}
