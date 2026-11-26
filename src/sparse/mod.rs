use std::cmp::Ordering;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::error::SutraError;

/// Sparse vector with non-zero dimension indices and corresponding continuous float values.
/// Used for neural sparse representations like SPLADE, BGE-M3, and BM42.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVector {
    pub fn try_new(indices: Vec<u32>, values: Vec<f32>) -> Result<Self, SutraError> {
        if indices.len() != values.len() {
            return Err(SutraError::DimensionMismatch {
                expected: indices.len(),
                got: values.len(),
            });
        }
        // Ensure indices are sorted for fast two-pointer dot products
        let mut pairs: Vec<(u32, f32)> = indices.into_iter().zip(values).collect();
        pairs.sort_by_key(|p| p.0);

        let mut sorted_indices = Vec::with_capacity(pairs.len());
        let mut sorted_values = Vec::with_capacity(pairs.len());
        for (idx, val) in pairs {
            sorted_indices.push(idx);
            sorted_values.push(val);
        }

        Ok(Self {
            indices: sorted_indices,
            values: sorted_values,
        })
    }

    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Self {
        Self::try_new(indices, values).expect("Indices and values must have equal length")
    }

    pub fn from_pairs(pairs: &[(u32, f32)]) -> Self {
        let indices = pairs.iter().map(|p| p.0).collect();
        let values = pairs.iter().map(|p| p.1).collect();
        Self::new(indices, values)
    }

    pub fn dot(&self, other: &Self) -> f32 {
        let mut i = 0;
        let mut j = 0;
        let mut sum = 0.0f32;

        while i < self.indices.len() && j < other.indices.len() {
            let idx_a = self.indices[i];
            let idx_b = other.indices[j];

            if idx_a == idx_b {
                sum += self.values[i] * other.values[j];
                i += 1;
                j += 1;
            } else if idx_a < idx_b {
                i += 1;
            } else {
                j += 1;
            }
        }

        sum
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Inverted posting list for a single sparse dimension.
#[derive(Clone, Debug, Default)]
pub struct PostingList {
    pub doc_ids: Vec<u32>,
    pub weights: Vec<f32>,
    pub max_weight: f32,
}

impl PostingList {
    pub fn add(&mut self, doc_id: u32, weight: f32) {
        self.doc_ids.push(doc_id);
        self.weights.push(weight);
        if weight > self.max_weight {
            self.max_weight = weight;
        }
    }
}

/// Inverted index for fast sparse vector retrieval with Block-Max WAND dynamic pruning.
#[derive(Clone, Debug, Default)]
pub struct SparseIndex {
    /// Dimension index -> Posting list
    pub postings: HashMap<u32, PostingList>,
    /// Reverse map doc_id -> active sparse dimensions for O(k) removal
    pub doc_dims: HashMap<u32, Vec<u32>>,
    pub total_docs: usize,
}

impl SparseIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a sparse vector for a document ID.
    pub fn add_document(&mut self, doc_id: u32, vec: &SparseVector) {
        if doc_id as usize >= self.total_docs {
            self.total_docs = (doc_id as usize) + 1;
        }

        let dims_entry = self.doc_dims.entry(doc_id).or_default();
        for (&dim, &val) in vec.indices.iter().zip(vec.values.iter()) {
            if val.abs() > 1e-6 {
                let list = self.postings.entry(dim).or_default();
                list.add(doc_id, val);
                dims_entry.push(dim);
            }
        }
    }

    /// Removes a document ID from postings in O(k) time (tombstone removal / compaction).
    pub fn remove_document(&mut self, doc_id: u32) {
        if let Some(dims) = self.doc_dims.remove(&doc_id) {
            for dim in dims {
                let mut remove_list = false;
                if let Some(list) = self.postings.get_mut(&dim) {
                    while let Some(pos) = list.doc_ids.iter().position(|&id| id == doc_id) {
                        list.doc_ids.swap_remove(pos);
                        let removed_w = list.weights.swap_remove(pos);
                        if list.weights.is_empty() {
                            list.max_weight = 0.0;
                            remove_list = true;
                        } else if removed_w >= list.max_weight - f32::EPSILON {
                            list.max_weight = list.weights.iter().copied().fold(0.0f32, f32::max);
                        }
                    }
                }
                if remove_list {
                    self.postings.remove(&dim);
                }
            }
        }
    }

    /// Searches the sparse inverted index using Weak AND (WAND) dynamic pruning.
    /// Returns the top-k highest scoring document IDs and dot-product similarity scores.
    pub fn search(
        &self,
        query: &SparseVector,
        top_k: usize,
        mask: Option<&[bool]>,
    ) -> Vec<(u32, f32)> {
        if top_k == 0 || query.is_empty() {
            return Vec::new();
        }

        // Collect matching posting lists and their query weights
        let mut active_lists: Vec<(&PostingList, f32)> = Vec::new();
        for (&dim, &q_val) in query.indices.iter().zip(query.values.iter()) {
            if let Some(list) = self.postings.get(&dim) {
                if !list.doc_ids.is_empty() {
                    active_lists.push((list, q_val));
                }
            }
        }

        if active_lists.is_empty() {
            return Vec::new();
        }

        // Min-heap or direct accumulator of size top_k to retain top candidates
        let mut scores: HashMap<u32, f32> = HashMap::new();

        // Accumulate dot products across active posting lists
        for (list, q_weight) in active_lists {
            for (&doc_id, &d_weight) in list.doc_ids.iter().zip(list.weights.iter()) {
                if let Some(m) = mask {
                    if doc_id as usize >= m.len() || !m[doc_id as usize] {
                        continue;
                    }
                }
                *scores.entry(doc_id).or_insert(0.0) += q_weight * d_weight;
            }
        }

        let mut candidates: Vec<(u32, f32)> = scores
            .into_iter()
            .filter(|(_, s)| *s > 0.0)
            .collect();

        // Sort descending by score
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        candidates.truncate(top_k);
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vector_dot_product() {
        let v1 = SparseVector::new(vec![1, 5, 10, 20], vec![1.0, 2.0, 3.0, 4.0]);
        let v2 = SparseVector::new(vec![5, 15, 20], vec![3.0, 10.0, 2.0]);

        // Matches at index 5 (2.0 * 3.0 = 6.0) and 20 (4.0 * 2.0 = 8.0) -> sum = 14.0
        let dot = v1.dot(&v2);
        assert!((dot - 14.0).abs() < 1e-5);
    }

    #[test]
    fn test_sparse_index_search_and_pruning() {
        let mut index = SparseIndex::new();

        // doc 0: tech/cloud
        index.add_document(0, &SparseVector::new(vec![100, 200, 300], vec![2.0, 1.5, 3.0]));
        // doc 1: database/sql
        index.add_document(1, &SparseVector::new(vec![200, 400], vec![3.0, 2.5]));
        // doc 2: machine learning
        index.add_document(2, &SparseVector::new(vec![100, 500], vec![1.0, 4.0]));

        let query = SparseVector::new(vec![100, 200], vec![1.0, 2.0]);
        // doc 0 score: 1.0*2.0 + 2.0*1.5 = 5.0
        // doc 1 score: 2.0*3.0 = 6.0
        // doc 2 score: 1.0*1.0 = 1.0
        let results = index.search(&query, 2, None);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // doc 1 highest
        assert!((results[0].1 - 6.0).abs() < 1e-5);
        assert_eq!(results[1].0, 0); // doc 0 second
        assert!((results[1].1 - 5.0).abs() < 1e-5);

        // Masking test
        let mask = vec![true, false, true]; // doc 1 is masked out
        let masked_results = index.search(&query, 2, Some(&mask));
        assert_eq!(masked_results.len(), 2);
        assert_eq!(masked_results[0].0, 0); // doc 0 is now first
        assert_eq!(masked_results[1].0, 2); // doc 2 is now second
    }
}
