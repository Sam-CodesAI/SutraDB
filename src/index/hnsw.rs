use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use rand::prelude::*;
use crate::distance::simd::compute_vector_score;
use crate::distance::{Metric, VectorStore};
use crate::error::SutraError;

#[derive(Debug, Clone)]
pub struct HNSWConfig {
    pub m: usize,               // Max links per node (e.g. 16)
    pub m0: usize,              // Max links on layer 0 (e.g. 32)
    pub ef_construction: usize, // Beam width during construction (e.g. 200)
    pub ef_search: usize,       // Beam width during search (e.g. 64)
    pub ml: f64,                // Normalization factor 1 / ln(m)
    pub seed: u64,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: m * 2,
            ef_construction: 200,
            ef_search: 64,
            ml: 1.0 / (m as f64).ln(),
            seed: 42,
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
struct Candidate {
    id: u32,
    score: f32, // Similarity score (higher = closer)
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Fast visited tracker backed by a compact bitset for dense vector IDs.
struct VisitedTracker {
    bits: Vec<u64>,
    overflow: HashSet<u32>,
}

impl VisitedTracker {
    fn new(capacity: usize) -> Self {
        Self {
            bits: vec![0u64; capacity.div_ceil(64)],
            overflow: HashSet::new(),
        }
    }

    #[inline]
    fn insert(&mut self, id: u32) -> bool {
        let idx = id as usize;
        let word = idx >> 6;
        if word < self.bits.len() {
            let mask = 1u64 << (idx & 63);
            let prev = self.bits[word];
            if (prev & mask) != 0 {
                return false;
            }
            self.bits[word] = prev | mask;
            true
        } else {
            self.overflow.insert(id)
        }
    }
}

/// Hierarchical Navigable Small World (HNSW) graph index.
/// Provides O(log N) approximate nearest neighbor vector search with dynamic insertions.
#[derive(Debug, Clone)]
pub struct HNSWIndex {
    pub config: HNSWConfig,
    pub entry_point: Option<u32>,
    pub max_level: usize,
    // layers[level][node_id] -> neighbors
    pub layers: Vec<HashMap<u32, Vec<u32>>>,
    pub node_levels: HashMap<u32, usize>,
    rng: StdRng,
}

impl HNSWIndex {
    pub fn new(config: HNSWConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            config,
            entry_point: None,
            max_level: 0,
            layers: Vec::new(),
            node_levels: HashMap::new(),
            rng,
        }
    }

    /// Generates a random maximum level for a newly inserted node.
    fn random_level(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        let level = (-r.ln() * self.config.ml).floor() as usize;
        level.min(16)
    }

    /// Maps monotonic similarity score (higher = closer) to true non-negative distance
    /// (lower = closer) so HNSW Algorithm 4 distance comparisons are metric-consistent.
    #[inline]
    fn score_to_distance(score: f32, metric: Metric) -> f32 {
        match metric {
            Metric::Cosine | Metric::DotProduct => 1.0 - score,
            Metric::Euclidean => {
                if score > 0.0 {
                    (1.0 / score) - 1.0
                } else {
                    f32::INFINITY
                }
            }
        }
    }

    /// HNSW Algorithm 4 diversity heuristic in distance space:
    /// prunes candidate `e` only if it is strictly closer to an already-selected
    /// neighbor `r` (`dist(e, r) < dist(q, e)`) than to the base node `q`.
    fn select_neighbors_heuristic<V: VectorStore + ?Sized>(
        candidates: &[Candidate],
        m_max: usize,
        all_vectors: &V,
        metric: Metric,
        is_normalized: bool,
    ) -> Vec<u32> {
        if candidates.len() <= m_max {
            return candidates.iter().map(|c| c.id).collect();
        }
        let mut selected: Vec<u32> = Vec::with_capacity(m_max);
        let mut discarded: Vec<u32> = Vec::new();
        for cand in candidates {
            if selected.len() >= m_max {
                break;
            }
            if selected.is_empty() {
                selected.push(cand.id);
                continue;
            }
            let dist_q = Self::score_to_distance(cand.score, metric);
            let cand_vec = all_vectors.vec_slice(cand.id as usize);
            let mut closer_to_existing = false;
            for &sel_id in &selected {
                let sim_r = compute_vector_score(
                    cand_vec,
                    all_vectors.vec_slice(sel_id as usize),
                    metric,
                    is_normalized,
                )
                .unwrap_or(f32::NEG_INFINITY);
                let dist_r = Self::score_to_distance(sim_r, metric);
                if dist_r < dist_q {
                    closer_to_existing = true;
                    break;
                }
            }
            if !closer_to_existing {
                selected.push(cand.id);
            } else {
                discarded.push(cand.id);
            }
        }
        for id in discarded {
            if selected.len() >= m_max {
                break;
            }
            selected.push(id);
        }
        selected
    }

    /// Dynamically inserts a document vector into the HNSW graph.
    pub fn insert<V: VectorStore + ?Sized>(
        &mut self,
        doc_id: u32,
        vector: &[f32],
        all_vectors: &V,
        metric: Metric,
        is_normalized: bool,
    ) -> Result<(), SutraError> {
        let node_level = self.random_level();
        self.node_levels.insert(doc_id, node_level);

        while self.layers.len() <= node_level {
            self.layers.push(HashMap::new());
        }

        let curr_ep = match self.entry_point {
            Some(ep) => ep,
            None => {
                for layer in self.layers.iter_mut().take(node_level + 1) {
                    layer.insert(doc_id, Vec::new());
                }
                self.entry_point = Some(doc_id);
                self.max_level = node_level;
                return Ok(());
            }
        };

        let mut curr_obj = curr_ep;
        let mut curr_score = compute_vector_score(
            vector,
            all_vectors.vec_slice(curr_obj as usize),
            metric,
            is_normalized,
        )?;

        // Top-down greedy routing to find entry point at level `node_level`
        for l in (node_level + 1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if let Some(neighbors) = self.layers[l].get(&curr_obj) {
                    for &nbr in neighbors {
                        let score = compute_vector_score(
                            vector,
                            all_vectors.vec_slice(nbr as usize),
                            metric,
                            is_normalized,
                        )?;
                        if score > curr_score {
                            curr_score = score;
                            curr_obj = nbr;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Insert at layers min(node_level, max_level) down to 0
        let mut enter_pts = vec![curr_obj];
        let insert_top = node_level.min(self.max_level);

        for l in (0..=insert_top).rev() {
            let m_max = if l == 0 { self.config.m0 } else { self.config.m };
            let candidates = self.search_layer(
                vector,
                &enter_pts,
                self.config.ef_construction,
                l,
                all_vectors,
                metric,
                is_normalized,
                None,
            )?;

            let filtered_cands: Vec<Candidate> =
                candidates.iter().copied().filter(|c| c.id != doc_id).collect();
            let neighbors = Self::select_neighbors_heuristic(
                &filtered_cands,
                m_max,
                all_vectors,
                metric,
                is_normalized,
            );
            self.layers[l].insert(doc_id, neighbors.clone());

            for &nbr in &neighbors {
                let nbr_neighbors = self.layers[l].entry(nbr).or_default();
                if !nbr_neighbors.contains(&doc_id) {
                    nbr_neighbors.push(doc_id);
                    if nbr_neighbors.len() > m_max {
                        let nbr_vec = all_vectors.vec_slice(nbr as usize);
                        let mut scored_nbrs: Vec<Candidate> = nbr_neighbors
                            .iter()
                            .map(|&n| {
                                let s = compute_vector_score(
                                    nbr_vec,
                                    all_vectors.vec_slice(n as usize),
                                    metric,
                                    is_normalized,
                                )
                                .unwrap_or(f32::NEG_INFINITY);
                                Candidate { id: n, score: s }
                            })
                            .collect();
                        scored_nbrs.sort_by(|a, b| b.cmp(a));
                        *nbr_neighbors = Self::select_neighbors_heuristic(
                            &scored_nbrs,
                            m_max,
                            all_vectors,
                            metric,
                            is_normalized,
                        );
                    }
                }
            }

            if !candidates.is_empty() {
                enter_pts = candidates.into_iter().map(|c| c.id).collect();
            }
        }

        if node_level > self.max_level {
            for l in (self.max_level + 1)..=node_level {
                self.layers[l].insert(doc_id, Vec::new());
            }
            self.max_level = node_level;
            self.entry_point = Some(doc_id);
        }

        Ok(())
    }

    /// Dynamically deletes a node from the HNSW graph and heals adjacent edges.
    pub fn delete_node<V: VectorStore + ?Sized>(
        &mut self,
        doc_id: u32,
        all_vectors: &V,
        metric: Metric,
        is_normalized: bool,
    ) -> bool {
        if self.node_levels.remove(&doc_id).is_none() {
            return false;
        }

        let min_links = (self.config.m / 2).max(1);

        for l in 0..self.layers.len() {
            let neighbors = self.layers[l].remove(&doc_id).unwrap_or_default();
            let m_max = if l == 0 { self.config.m0 } else { self.config.m };

            let mut needs_healing: Vec<u32> = Vec::new();
            for (&other_id, other_list) in self.layers[l].iter_mut() {
                let before_len = other_list.len();
                other_list.retain(|&x| x != doc_id);
                if other_list.len() < before_len && other_list.len() < min_links {
                    needs_healing.push(other_id);
                }
            }

            for &nbr in &neighbors {
                if let Some(nbr_list) = self.layers[l].get(&nbr) {
                    if nbr_list.len() < min_links && !needs_healing.contains(&nbr) {
                        needs_healing.push(nbr);
                    }
                }
            }

            let mut to_cross_connect = Vec::new();
            let valid_candidates: Vec<u32> = neighbors
                .iter()
                .copied()
                .filter(|&c| c != doc_id && self.layers[l].contains_key(&c))
                .collect();

            for nbr in needs_healing {
                let mut candidates: Vec<u32> = {
                    let nbr_list = match self.layers[l].get(&nbr) {
                        Some(list) => list,
                        None => continue,
                    };
                    valid_candidates
                        .iter()
                        .copied()
                        .filter(|&c| c != nbr && !nbr_list.contains(&c))
                        .collect()
                };

                if candidates.is_empty() {
                    continue;
                }

                let nbr_vec = all_vectors.vec_slice(nbr as usize);
                candidates.sort_by(|&a, &b| {
                    let sa = compute_vector_score(
                        nbr_vec,
                        all_vectors.vec_slice(a as usize),
                        metric,
                        is_normalized,
                    )
                    .unwrap_or(f32::NEG_INFINITY);
                    let sb = compute_vector_score(
                        nbr_vec,
                        all_vectors.vec_slice(b as usize),
                        metric,
                        is_normalized,
                    )
                    .unwrap_or(f32::NEG_INFINITY);
                    sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
                });

                if let Some(nbr_list) = self.layers[l].get_mut(&nbr) {
                    for cand in candidates {
                        if nbr_list.len() >= m_max {
                            break;
                        }
                        if !nbr_list.contains(&cand) {
                            nbr_list.push(cand);
                            to_cross_connect.push((cand, nbr));
                        }
                    }
                }
            }

            for (cand, nbr) in to_cross_connect {
                if let Some(cand_list) = self.layers[l].get_mut(&cand) {
                    if !cand_list.contains(&nbr) && cand_list.len() < m_max {
                        cand_list.push(nbr);
                    }
                }
            }
        }

        if self.entry_point == Some(doc_id) {
            if let Some((&new_ep, &lvl)) = self.node_levels.iter().max_by_key(|(_, &lvl)| lvl) {
                self.entry_point = Some(new_ep);
                self.max_level = lvl;
            } else {
                self.entry_point = None;
                self.max_level = 0;
            }
        }

        true
    }

    #[allow(clippy::too_many_arguments)]
    fn search_layer<V: VectorStore + ?Sized>(
        &self,
        query: &[f32],
        enter_points: &[u32],
        ef: usize,
        level: usize,
        all_vectors: &V,
        metric: Metric,
        is_normalized: bool,
        mask: Option<&[bool]>,
    ) -> Result<Vec<Candidate>, SutraError> {
        let is_valid = |id: u32| -> bool {
            mask.map(|m| {
                let idx = id as usize;
                idx < m.len() && m[idx]
            })
            .unwrap_or(true)
        };

        let mut visited = VisitedTracker::new(all_vectors.vec_len());
        let mut results: BinaryHeap<std::cmp::Reverse<Candidate>> = BinaryHeap::with_capacity(ef + 1);
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef * 2);

        for &ep in enter_points {
            if !visited.insert(ep) {
                continue;
            }
            let score = compute_vector_score(
                query,
                all_vectors.vec_slice(ep as usize),
                metric,
                is_normalized,
            )?;
            let cand = Candidate { id: ep, score };
            candidates.push(cand);
            if is_valid(ep) {
                results.push(std::cmp::Reverse(cand));
                if results.len() > ef {
                    results.pop();
                }
            }
        }

        let layer_map = &self.layers[level];
        while let Some(current) = candidates.pop() {
            if results.len() >= ef {
                if let Some(worst_result) = results.peek() {
                    if current.score < worst_result.0.score {
                        break;
                    }
                }
            }

            if let Some(neighbors) = layer_map.get(&current.id) {
                for &nbr in neighbors {
                    if visited.insert(nbr) {
                        let score = compute_vector_score(
                            query,
                            all_vectors.vec_slice(nbr as usize),
                            metric,
                            is_normalized,
                        )?;
                        let cand = Candidate { id: nbr, score };
                        let worst = results.peek().map(|r| r.0.score).unwrap_or(f32::NEG_INFINITY);

                        if score > worst || results.len() < ef {
                            candidates.push(cand);
                            if is_valid(nbr) {
                                results.push(std::cmp::Reverse(cand));
                                if results.len() > ef {
                                    results.pop();
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = results.into_iter().map(|r| r.0).collect();
        out.sort_by(|a, b| b.cmp(a));
        Ok(out)
    }

    /// Queries the HNSW graph starting from top layer down to layer 0.
    #[allow(clippy::too_many_arguments)]
    pub fn search<V: VectorStore + ?Sized>(
        &self,
        query: &[f32],
        all_vectors: &V,
        k: usize,
        ef_search: usize,
        metric: Metric,
        is_normalized: bool,
        mask: Option<&[bool]>,
    ) -> Result<Vec<(u32, f32)>, SutraError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };

        let mut curr_obj = ep;
        let mut curr_score = compute_vector_score(
            query,
            all_vectors.vec_slice(curr_obj as usize),
            metric,
            is_normalized,
        )?;

        for l in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if let Some(neighbors) = self.layers[l].get(&curr_obj) {
                    for &nbr in neighbors {
                        let score = compute_vector_score(
                            query,
                            all_vectors.vec_slice(nbr as usize),
                            metric,
                            is_normalized,
                        )?;
                        if score > curr_score {
                            curr_score = score;
                            curr_obj = nbr;
                            changed = true;
                        }
                    }
                }
            }
        }

        let ef = ef_search.max(k);
        let mut candidates = self.search_layer(
            query,
            &[curr_obj],
            ef,
            0,
            all_vectors,
            metric,
            is_normalized,
            mask,
        )?;

        if candidates.len() < k {
            if let Some(m) = mask {
                if !self.layers.is_empty() {
                    let valid_ids: Vec<u32> = m
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, &ok)| {
                            let id = idx as u32;
                            if ok && self.layers[0].contains_key(&id) {
                                Some(id)
                            } else {
                                None
                            }
                        })
                        .collect();

                    if !valid_ids.is_empty() {
                        let sample_cap = ef.max(32);
                        let step = (valid_ids.len() / sample_cap).max(1);
                        let valid_eps: Vec<u32> = valid_ids
                            .iter()
                            .step_by(step)
                            .take(sample_cap)
                            .copied()
                            .collect();
                        let extra = self.search_layer(
                            query,
                            &valid_eps,
                            ef,
                            0,
                            all_vectors,
                            metric,
                            is_normalized,
                            mask,
                        )?;
                        let mut seen: HashSet<u32> = candidates.iter().map(|c| c.id).collect();
                        for c in extra {
                            if seen.insert(c.id) {
                                candidates.push(c);
                            }
                        }
                        candidates.sort_by(|a, b| b.cmp(a));
                    }
                }
            }
        }

        let mut results = Vec::with_capacity(k.min(candidates.len()));
        for c in candidates.into_iter().take(k) {
            results.push((c.id, c.score));
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_insert_delete_and_heal() {
        let mut index = HNSWIndex::new(HNSWConfig::default());
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
            vec![0.5, 0.5],
        ];

        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u32, v, &vectors, Metric::Cosine, true).unwrap();
        }

        assert_eq!(index.node_levels.len(), 5);
        assert!(index.entry_point.is_some());

        let q = vec![1.0, 0.0];
        let res = index.search(&q, &vectors, 2, 16, Metric::Cosine, true, None).unwrap();
        assert!(!res.is_empty());
        assert_eq!(res[0].0, 0);

        let ep = index.entry_point.unwrap();
        let deleted = index.delete_node(ep, &vectors, Metric::Cosine, true);
        assert!(deleted);
        assert!(!index.node_levels.contains_key(&ep));
        assert!(index.entry_point.is_some());
        assert_ne!(index.entry_point.unwrap(), ep);

        let res2 = index.search(&q, &vectors, 2, 16, Metric::Cosine, true, None).unwrap();
        assert!(!res2.is_empty());
        assert_ne!(res2[0].0, ep);
    }
}
