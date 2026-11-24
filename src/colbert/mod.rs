use std::collections::{HashMap, HashSet};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::distance::simd::{dot_product, normalize_in_place};
use crate::error::SutraError;

/// Computes the exact ColBERT Late Interaction MaxSim score between a query token matrix
/// and a document token matrix:
///
/// MaxSim(Q, D) = sum_{q in Q} max_{d in D} (q . d)
pub fn compute_maxsim(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }

    let mut total_score = 0.0f32;
    for q in query_tokens {
        let mut max_sim = f32::NEG_INFINITY;
        for d in doc_tokens {
            let sim = dot_product(q, d);
            if sim > max_sim {
                max_sim = sim;
            }
        }
        if max_sim != f32::NEG_INFINITY {
            total_score += max_sim;
        }
    }

    total_score
}

/// Token inverted index and spherical K-Means clustering for ColBERT PLAID-style retrieval.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColbertCentroids {
    pub centroids: Vec<Vec<f32>>,
    pub postings: HashMap<usize, Vec<u32>>,
}

/// Industrial ColBERT Multi-Vector late interaction storage and indexing engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColbertIndex {
    pub token_dim: usize,
    pub doc_tokens: HashMap<u32, Vec<Vec<f32>>>,
    pub centroids: Option<ColbertCentroids>,
    pub is_trained: bool,
}

impl ColbertIndex {
    pub fn new(token_dim: usize) -> Self {
        Self {
            token_dim,
            doc_tokens: HashMap::new(),
            centroids: None,
            is_trained: false,
        }
    }

    pub fn len(&self) -> usize {
        self.doc_tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_tokens.is_empty()
    }

    /// Adds or updates a document's token embeddings.
    /// Normalizes each token in-place to unit sphere for accurate cosine dot products.
    pub fn add_document(&mut self, doc_id: u32, mut tokens: Vec<Vec<f32>>) {
        for t in &mut tokens {
            if t.len() == self.token_dim {
                normalize_in_place(t);
            }
        }

        // If centroids exist, map new document's tokens into centroid postings
        if let Some(ref mut c) = self.centroids {
            let mut matched_centroids = HashSet::new();
            for t in &tokens {
                if let Some(c_idx) = find_nearest_centroid(t, &c.centroids) {
                    matched_centroids.insert(c_idx);
                }
            }
            for c_idx in matched_centroids {
                c.postings.entry(c_idx).or_default().push(doc_id);
            }
        }

        self.doc_tokens.insert(doc_id, tokens);
    }

    /// Removes a document from the token storage and centroid index.
    pub fn remove_document(&mut self, doc_id: u32) {
        self.doc_tokens.remove(&doc_id);
        if let Some(ref mut c) = self.centroids {
            for list in c.postings.values_mut() {
                list.retain(|&id| id != doc_id);
            }
        }
    }

    /// Trains spherical K-Means centroids across sampled token embeddings (PLAID acceleration).
    pub fn train_centroids(
        &mut self,
        num_centroids: usize,
        max_iters: usize,
    ) -> Result<(), SutraError> {
        let total_docs = self.doc_tokens.len();
        if total_docs == 0 {
            return Err(SutraError::IndexNotTrained(
                "Cannot train ColBERT centroids on empty index".into(),
            ));
        }

        // Gather all token vectors for training
        let mut sample_tokens: Vec<Vec<f32>> = Vec::new();
        for tokens in self.doc_tokens.values() {
            for t in tokens {
                sample_tokens.push(t.clone());
            }
        }

        if sample_tokens.is_empty() {
            return Err(SutraError::IndexNotTrained(
                "No token vectors found for centroid training".into(),
            ));
        }

        let k = num_centroids.min(sample_tokens.len()).max(1);

        // Deterministic k-means++ initialization
        let mut centroids = Vec::with_capacity(k);
        let step = sample_tokens.len() / k;
        for i in 0..k {
            let idx = (i * step).min(sample_tokens.len() - 1);
            let mut c = sample_tokens[idx].clone();
            normalize_in_place(&mut c);
            centroids.push(c);
        }

        // Iterative spherical k-means refinement
        for _ in 0..max_iters {
            let mut clusters: Vec<Vec<Vec<f32>>> = vec![Vec::new(); k];

            for t in &sample_tokens {
                let mut best_sim = f32::NEG_INFINITY;
                let mut best_c = 0;
                for (c_idx, c) in centroids.iter().enumerate() {
                    let sim = dot_product(t, c);
                    if sim > best_sim {
                        best_sim = sim;
                        best_c = c_idx;
                    }
                }
                clusters[best_c].push(t.clone());
            }

            let mut converged = true;
            for (c_idx, cluster) in clusters.iter().enumerate() {
                if cluster.is_empty() {
                    continue;
                }
                let mut new_c = vec![0.0f32; self.token_dim];
                for vec in cluster {
                    for (i, val) in vec.iter().enumerate() {
                        new_c[i] += val;
                    }
                }
                normalize_in_place(&mut new_c);

                if dot_product(&centroids[c_idx], &new_c) < 0.9999 {
                    converged = false;
                }
                centroids[c_idx] = new_c;
            }

            if converged {
                break;
            }
        }

        // Build inverted index for all documents against trained centroids
        let mut postings: HashMap<usize, Vec<u32>> = HashMap::new();
        for (&doc_id, tokens) in &self.doc_tokens {
            let mut matched = HashSet::new();
            for t in tokens {
                if let Some(c_idx) = find_nearest_centroid(t, &centroids) {
                    matched.insert(c_idx);
                }
            }
            for c_idx in matched {
                postings.entry(c_idx).or_default().push(doc_id);
            }
        }

        self.centroids = Some(ColbertCentroids {
            centroids,
            postings,
        });
        self.is_trained = true;
        Ok(())
    }

    /// Queries the ColBERT index using Late Interaction MaxSim.
    /// If centroids are trained and document pool > 512, performs sub-linear candidate pruning.
    /// Otherwise, evaluates parallel Rayon SIMD MaxSim across all active documents.
    pub fn query_maxsim(
        &self,
        query_tokens: &[Vec<f32>],
        mask: Option<&[bool]>,
        top_k: usize,
    ) -> Vec<(u32, f32)> {
        if query_tokens.is_empty() || self.doc_tokens.is_empty() {
            return Vec::new();
        }

        // Normalize query tokens copy
        let mut norm_queries = query_tokens.to_vec();
        for q in &mut norm_queries {
            if q.len() == self.token_dim {
                normalize_in_place(q);
            }
        }

        // Step 1: Candidate identification
        let candidate_doc_ids: Vec<u32> = if let Some(ref c) = self.centroids {
            if self.doc_tokens.len() > 512 {
                // PLAID Centroid Candidate Pruning:
                // Find top-2 centroids for each query token and gather candidate doc_ids
                let mut candidate_set = HashSet::new();
                for q in &norm_queries {
                    let top_centroids = find_top_centroids(q, &c.centroids, 2);
                    for c_idx in top_centroids {
                        if let Some(docs) = c.postings.get(&c_idx) {
                            for &d_id in docs {
                                candidate_set.insert(d_id);
                            }
                        }
                    }
                }

                candidate_set
                    .into_iter()
                    .filter(|&id| {
                        mask.map(|m| {
                            let idx = id as usize;
                            idx < m.len() && m[idx]
                        })
                        .unwrap_or(true)
                    })
                    .collect()
            } else {
                self.gather_active_candidates(mask)
            }
        } else {
            self.gather_active_candidates(mask)
        };

        if candidate_doc_ids.is_empty() {
            return Vec::new();
        }

        // Step 2: Parallel Rayon SIMD MaxSim evaluation over candidate pool
        let mut scores: Vec<(u32, f32)> = candidate_doc_ids
            .par_iter()
            .filter_map(|&doc_id| {
                self.doc_tokens.get(&doc_id).map(|tokens| {
                    let score = compute_maxsim(&norm_queries, tokens);
                    (doc_id, score)
                })
            })
            .collect();

        // Sort descending by MaxSim score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    fn gather_active_candidates(&self, mask: Option<&[bool]>) -> Vec<u32> {
        self.doc_tokens
            .keys()
            .copied()
            .filter(|&id| {
                mask.map(|m| {
                    let idx = id as usize;
                    idx < m.len() && m[idx]
                })
                .unwrap_or(true)
            })
            .collect()
    }
}

fn find_nearest_centroid(token: &[f32], centroids: &[Vec<f32>]) -> Option<usize> {
    if centroids.is_empty() {
        return None;
    }
    let mut best_sim = f32::NEG_INFINITY;
    let mut best_idx = 0;
    for (i, c) in centroids.iter().enumerate() {
        let sim = dot_product(token, c);
        if sim > best_sim {
            best_sim = sim;
            best_idx = i;
        }
    }
    Some(best_idx)
}

fn find_top_centroids(token: &[f32], centroids: &[Vec<f32>], top_n: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, dot_product(token, c)))
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(top_n).map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_maxsim_exact() {
        // Query has 2 tokens of 4 dimensions
        let q = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];

        // Doc A has 2 tokens matching Q perfectly
        let doc_a = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];

        // Doc B has 2 tokens orthogonal
        let doc_b = vec![
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let score_a = compute_maxsim(&q, &doc_a);
        let score_b = compute_maxsim(&q, &doc_b);

        // MaxSim for doc_a should be 1.0 + 1.0 = 2.0
        assert!((score_a - 2.0).abs() < 1e-5);
        // MaxSim for doc_b should be 0.0 + 0.0 = 0.0
        assert!(score_b.abs() < 1e-5);
    }

    #[test]
    fn test_colbert_index_lifecycle_and_pruning() {
        let mut index = ColbertIndex::new(4);

        let doc1 = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.7071, 0.7071, 0.0, 0.0],
        ];
        let doc2 = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let doc3 = vec![
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        index.add_document(0, doc1);
        index.add_document(1, doc2);
        index.add_document(2, doc3);

        assert_eq!(index.len(), 3);

        let query = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let results = index.query_maxsim(&query, None, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // doc 0 has exact match [1, 0, 0, 0]
        assert!((results[0].1 - 1.0).abs() < 1e-4);

        // Test centroid training
        index.train_centroids(2, 10).unwrap();
        assert!(index.is_trained);

        let query_multi = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        let results_multi = index.query_maxsim(&query_multi, None, 3);
        assert_eq!(results_multi[0].0, 0); // doc 0 matches [1,0,0,0] with 1.0 + [0.7071, 0.7071] with 0.7071 = 1.7071
    }
}
