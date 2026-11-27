use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
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

/// Inverted posting list for a single sparse dimension with ascending `doc_ids`
/// and precomputed `max_weight` for WAND dynamic pruning.
#[derive(Clone, Debug, Default)]
pub struct PostingList {
    pub doc_ids: Vec<u32>,
    pub weights: Vec<f32>,
    pub max_weight: f32,
}

impl PostingList {
    /// Appends or inserts `(doc_id, weight)` while keeping `doc_ids` sorted ascending.
    pub fn add(&mut self, doc_id: u32, weight: f32) {
        if self.doc_ids.last().is_none_or(|&last| doc_id > last) {
            self.doc_ids.push(doc_id);
            self.weights.push(weight);
        } else {
            match self.doc_ids.binary_search(&doc_id) {
                Ok(pos) => {
                    self.weights[pos] = weight;
                    if weight > self.max_weight {
                        self.max_weight = weight;
                    } else {
                        self.max_weight = self.weights.iter().copied().fold(0.0f32, f32::max);
                    }
                    return;
                }
                Err(pos) => {
                    self.doc_ids.insert(pos, doc_id);
                    self.weights.insert(pos, weight);
                }
            }
        }
        if weight > self.max_weight {
            self.max_weight = weight;
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
struct HeapEntry {
    doc_id: u32,
    score: f32,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.doc_id.cmp(&self.doc_id))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct WandCursor<'a> {
    list: &'a PostingList,
    pos: usize,
    current_doc: u32,
    q_weight: f32,
    upper_bound: f32,
}

impl<'a> WandCursor<'a> {
    #[inline]
    fn advance_valid(&mut self, mask: Option<&[bool]>) -> bool {
        let len = self.list.doc_ids.len();
        while self.pos < len {
            let doc_id = self.list.doc_ids[self.pos];
            if let Some(m) = mask {
                let idx = doc_id as usize;
                if idx >= m.len() || !m[idx] {
                    self.pos += 1;
                    continue;
                }
            }
            self.current_doc = doc_id;
            return true;
        }
        false
    }

    #[inline]
    fn seek_to(&mut self, target_doc: u32, mask: Option<&[bool]>) -> bool {
        if self.pos < self.list.doc_ids.len() && self.current_doc >= target_doc {
            return true;
        }
        let remaining = &self.list.doc_ids[self.pos..];
        let step = remaining.partition_point(|&id| id < target_doc);
        self.pos += step;
        self.advance_valid(mask)
    }
}

/// Inverted index for fast sparse vector retrieval with WAND dynamic pruning.
#[derive(Clone, Debug, Default)]
pub struct SparseIndex {
    /// Dimension index -> Posting list (kept sorted ascending by `doc_id`)
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

    /// Removes a document ID from postings while preserving sorted `doc_ids` order for WAND.
    pub fn remove_document(&mut self, doc_id: u32) {
        if let Some(dims) = self.doc_dims.remove(&doc_id) {
            for dim in dims {
                let mut remove_list = false;
                if let Some(list) = self.postings.get_mut(&dim) {
                    if let Ok(pos) = list.doc_ids.binary_search(&doc_id) {
                        list.doc_ids.remove(pos);
                        let removed_w = list.weights.remove(pos);
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

    fn search_exhaustive(
        active_lists: &[(&PostingList, f32)],
        top_k: usize,
        mask: Option<&[bool]>,
    ) -> Vec<(u32, f32)> {
        let mut scores: HashMap<u32, f32> = HashMap::new();
        for &(list, q_weight) in active_lists {
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
        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        candidates.truncate(top_k);
        candidates
    }

    /// Searches the sparse inverted index using lossless Weak AND (WAND) dynamic pruning.
    /// Falls back to exact accumulator only if any posting or query weight is negative.
    pub fn search(
        &self,
        query: &SparseVector,
        top_k: usize,
        mask: Option<&[bool]>,
    ) -> Vec<(u32, f32)> {
        if top_k == 0 || query.is_empty() {
            return Vec::new();
        }

        let mut active_lists: Vec<(&PostingList, f32)> = Vec::new();
        let mut has_negative = false;
        for (&dim, &q_val) in query.indices.iter().zip(query.values.iter()) {
            if q_val == 0.0 {
                continue;
            }
            if let Some(list) = self.postings.get(&dim) {
                if !list.doc_ids.is_empty() {
                    if q_val < 0.0 || list.weights.iter().any(|&w| w < 0.0) {
                        has_negative = true;
                    }
                    active_lists.push((list, q_val));
                }
            }
        }

        if active_lists.is_empty() {
            return Vec::new();
        }
        if has_negative {
            return Self::search_exhaustive(&active_lists, top_k, mask);
        }

        let mut cursors: Vec<WandCursor<'_>> = Vec::with_capacity(active_lists.len());
        for (list, q_weight) in active_lists {
            let ub = q_weight * list.max_weight;
            if ub <= 0.0 {
                continue;
            }
            let mut cur = WandCursor {
                list,
                pos: 0,
                current_doc: 0,
                q_weight,
                upper_bound: ub,
            };
            if cur.advance_valid(mask) {
                cursors.push(cur);
            }
        }

        if cursors.is_empty() {
            return Vec::new();
        }

        let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::with_capacity(top_k + 1);
        let mut threshold = 0.0f32;

        loop {
            if cursors.is_empty() {
                break;
            }
            cursors.sort_unstable_by_key(|c| c.current_doc);

            // Find pivot index where cumulative upper bound exceeds current top-K threshold
            let mut ub_sum = 0.0f32;
            let mut pivot_idx = None;
            for (idx, c) in cursors.iter().enumerate() {
                ub_sum += c.upper_bound;
                if ub_sum > threshold {
                    pivot_idx = Some(idx);
                    break;
                }
            }

            let Some(p_idx) = pivot_idx else {
                // Remaining candidates cannot beat `threshold`
                break;
            };

            let pivot_doc = cursors[p_idx].current_doc;
            if cursors[0].current_doc == pivot_doc {
                // Pivot doc has sufficient upper bound — score all cursors positioned at pivot_doc
                let mut score = 0.0f32;
                let mut idx = 0;
                while idx < cursors.len() {
                    if cursors[idx].current_doc == pivot_doc {
                        score += cursors[idx].q_weight * cursors[idx].list.weights[cursors[idx].pos];
                        cursors[idx].pos += 1;
                        if !cursors[idx].advance_valid(mask) {
                            cursors.swap_remove(idx);
                            continue;
                        }
                    }
                    idx += 1;
                }

                if score > 0.0 {
                    if heap.len() < top_k {
                        heap.push(Reverse(HeapEntry {
                            doc_id: pivot_doc,
                            score,
                        }));
                        if heap.len() == top_k {
                            threshold = heap.peek().unwrap().0.score;
                        }
                    } else if score > threshold {
                        heap.pop();
                        heap.push(Reverse(HeapEntry {
                            doc_id: pivot_doc,
                            score,
                        }));
                        threshold = heap.peek().unwrap().0.score;
                    }
                }
            } else {
                // Pick the preceding cursor with the longest remaining list to jump past non-candidates
                let mut jump_idx = 0usize;
                let mut max_rem = 0usize;
                for (i, c) in cursors.iter().enumerate().take(p_idx) {
                    let rem = c.list.doc_ids.len().saturating_sub(c.pos);
                    if rem >= max_rem {
                        max_rem = rem;
                        jump_idx = i;
                    }
                }
                if !cursors[jump_idx].seek_to(pivot_doc, mask) {
                    cursors.swap_remove(jump_idx);
                }
            }
        }

        let mut results: Vec<(u32, f32)> = heap
            .into_iter()
            .map(|Reverse(e)| (e.doc_id, e.score))
            .collect();
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vector_dot_product() {
        let v1 = SparseVector::new(vec![1, 5, 10, 20], vec![1.0, 2.0, 3.0, 4.0]);
        let v2 = SparseVector::new(vec![5, 15, 20], vec![3.0, 10.0, 2.0]);

        let dot = v1.dot(&v2);
        assert!((dot - 14.0).abs() < 1e-5);
    }

    #[test]
    fn test_sparse_index_search_and_pruning() {
        let mut index = SparseIndex::new();

        index.add_document(0, &SparseVector::new(vec![100, 200, 300], vec![2.0, 1.5, 3.0]));
        index.add_document(1, &SparseVector::new(vec![200, 400], vec![3.0, 2.5]));
        index.add_document(2, &SparseVector::new(vec![100, 500], vec![1.0, 4.0]));

        let query = SparseVector::new(vec![100, 200], vec![1.0, 2.0]);
        let results = index.search(&query, 2, None);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 6.0).abs() < 1e-5);
        assert_eq!(results[1].0, 0);
        assert!((results[1].1 - 5.0).abs() < 1e-5);

        let mask = vec![true, false, true];
        let masked_results = index.search(&query, 2, Some(&mask));
        assert_eq!(masked_results.len(), 2);
        assert_eq!(masked_results[0].0, 0);
        assert_eq!(masked_results[1].0, 2);
    }

    #[test]
    fn test_wand_matches_exhaustive_top_k() {
        let mut index = SparseIndex::new();
        for doc_id in 0..400u32 {
            let indices = vec![doc_id % 10, 20 + (doc_id % 15), 50 + (doc_id % 7)];
            let values = vec![
                ((doc_id * 17) % 100) as f32 * 0.05 + 0.1,
                ((doc_id * 31) % 80) as f32 * 0.03 + 0.2,
                ((doc_id * 7) % 50) as f32 * 0.04 + 0.15,
            ];
            index.add_document(doc_id, &SparseVector::new(indices, values));
        }

        // Exercise sorted removal invariance as well
        index.remove_document(15);
        index.remove_document(102);

        let query = SparseVector::new(vec![2, 24, 53], vec![1.5, 0.8, 2.2]);
        let mask: Vec<bool> = (0..400).map(|i| i % 4 != 0).collect();

        for use_mask in [None, Some(mask.as_slice())] {
            let wand_hits = index.search(&query, 10, use_mask);
            let active: Vec<(&PostingList, f32)> = query
                .indices
                .iter()
                .zip(query.values.iter())
                .filter_map(|(dim, &q)| index.postings.get(dim).map(|l| (l, q)))
                .collect();
            let exact_hits = SparseIndex::search_exhaustive(&active, 10, use_mask);

            assert_eq!(wand_hits.len(), exact_hits.len());
            for ((wd, ws), (ed, es)) in wand_hits.iter().zip(exact_hits.iter()) {
                assert_eq!(wd, ed, "WAND doc mismatch vs exhaustive");
                assert!((ws - es).abs() < 1e-5, "WAND score mismatch: {} vs {}", ws, es);
            }
        }
    }
}
