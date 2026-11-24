use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use rand::prelude::*;
use crate::distance::simd::compute_vector_score;
use crate::distance::Metric;
use crate::error::SutraError;

#[derive(Debug, Clone)]
pub struct HNSWConfig {
    pub m: usize,              // Max links per node (e.g. 16)
    pub m0: usize,             // Max links on layer 0 (e.g. 32)
    pub ef_construction: usize,// Beam width during construction (e.g. 64)
    pub ef_search: usize,      // Beam width during search (e.g. 32)
    pub ml: f64,               // Normalization factor 1 / ln(m)
    pub seed: u64,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: m * 2,
            ef_construction: 64,
            ef_search: 32,
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
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
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

    /// Dynamically inserts a document vector into the HNSW graph.
    pub fn insert(
        &mut self,
        doc_id: u32,
        vector: &[f32],
        all_vectors: &[Vec<f32>],
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
                // First node in index
                for l in 0..=node_level {
                    self.layers[l].insert(doc_id, Vec::new());
                }
                self.entry_point = Some(doc_id);
                self.max_level = node_level;
                return Ok(());
            }
        };

        let mut curr_obj = curr_ep;
        let mut curr_score = compute_vector_score(
            vector,
            &all_vectors[curr_obj as usize],
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
                            &all_vectors[nbr as usize],
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

            // Select M nearest neighbors
            let neighbors: Vec<u32> = candidates.iter().take(m_max).map(|c| c.id).collect();
            self.layers[l].insert(doc_id, neighbors.clone());

            // Add bidirectional connections and prune
            for &nbr in &neighbors {
                let nbr_neighbors = self.layers[l].entry(nbr).or_default();
                if !nbr_neighbors.contains(&doc_id) {
                    nbr_neighbors.push(doc_id);
                    if nbr_neighbors.len() > m_max {
                        // Prune worst neighbor
                        let mut scored_nbrs: Vec<(u32, f32)> = nbr_neighbors
                            .iter()
                            .map(|&n| {
                                let s = compute_vector_score(
                                    &all_vectors[nbr as usize],
                                    &all_vectors[n as usize],
                                    metric,
                                    is_normalized,
                                ).unwrap_or(f32::NEG_INFINITY);
                                (n, s)
                            })
                            .collect();
                        scored_nbrs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                        scored_nbrs.truncate(m_max);
                        *nbr_neighbors = scored_nbrs.into_iter().map(|(n, _)| n).collect();
                    }
                }
            }

            enter_pts = candidates.into_iter().map(|c| c.id).collect();
        }

        if node_level > self.max_level {
            self.max_level = node_level;
            self.entry_point = Some(doc_id);
        }

        Ok(())
    }

    /// Dynamically deletes a node from the HNSW graph and heals adjacent edges.
    /// Preserves graph connectivity and re-elects entry point if needed.
    pub fn delete_node(
        &mut self,
        doc_id: u32,
        all_vectors: &[Vec<f32>],
        metric: Metric,
        is_normalized: bool,
    ) -> bool {
        let node_level = match self.node_levels.remove(&doc_id) {
            Some(lvl) => lvl,
            None => return false,
        };

        let min_links = (self.config.m / 2).max(1);

        for l in 0..=node_level {
            if l >= self.layers.len() {
                continue;
            }

            // Get deleted node's neighbors at layer l
            let neighbors = self.layers[l].remove(&doc_id).unwrap_or_default();
            let m_max = if l == 0 { self.config.m0 } else { self.config.m };

            // Remove doc_id from each neighbor's edge list
            let mut to_cross_connect = Vec::new();

            for &nbr in &neighbors {
                if let Some(nbr_list) = self.layers[l].get_mut(&nbr) {
                    nbr_list.retain(|&x| x != doc_id);

                    // Edge Healing: If neighbor's degree dropped below min_links, reconnect
                    if nbr_list.len() < min_links {
                        // Consider potential candidates: other neighbors of deleted node
                        let mut candidates: Vec<u32> = neighbors
                            .iter()
                            .copied()
                            .filter(|&c| c != nbr && !nbr_list.contains(&c))
                            .collect();

                        // Score candidates against nbr vector
                        candidates.sort_by(|&a, &b| {
                            let sa = compute_vector_score(
                                &all_vectors[nbr as usize],
                                &all_vectors[a as usize],
                                metric,
                                is_normalized,
                            ).unwrap_or(f32::NEG_INFINITY);
                            let sb = compute_vector_score(
                                &all_vectors[nbr as usize],
                                &all_vectors[b as usize],
                                metric,
                                is_normalized,
                            ).unwrap_or(f32::NEG_INFINITY);
                            sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
                        });

                        for cand in candidates {
                            if nbr_list.len() >= m_max {
                                break;
                            }
                            nbr_list.push(cand);
                            to_cross_connect.push((cand, nbr));
                        }
                    }
                }
            }

            // Bidirectionally add connections without violating borrow checker
            for (cand, nbr) in to_cross_connect {
                if let Some(cand_list) = self.layers[l].get_mut(&cand) {
                    if !cand_list.contains(&nbr) && cand_list.len() < m_max {
                        cand_list.push(nbr);
                    }
                }
            }
        }

        // Re-elect entry point if deleted node was the entry point
        if self.entry_point == Some(doc_id) {
            self.entry_point = None;
            // Search from top layer down for any existing node
            for l in (0..self.layers.len()).rev() {
                if let Some(&new_ep) = self.layers[l].keys().next() {
                    self.entry_point = Some(new_ep);
                    self.max_level = l;
                    break;
                }
            }
            if self.entry_point.is_none() {
                self.max_level = 0;
            }
        }

        true
    }

    /// Searches a single layer for nearest neighbors using beam search.
    fn search_layer(
        &self,
        query: &[f32],
        enter_points: &[u32],
        ef: usize,
        level: usize,
        all_vectors: &[Vec<f32>],
        metric: Metric,
        is_normalized: bool,
        mask: Option<&[bool]>,
    ) -> Result<Vec<Candidate>, SutraError> {
        let mut visited = HashSet::new();
        // Min-heap for keeping track of the `ef` closest elements
        let mut results = BinaryHeap::new(); // Max-heap in std, we'll invert via Reverse or custom ordering
        // Candidate pool (max-heap by score)
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();

        for &ep in enter_points {
            visited.insert(ep);
            let score = compute_vector_score(query, &all_vectors[ep as usize], metric, is_normalized)?;
            let cand = Candidate { id: ep, score };
            candidates.push(cand);
            results.push(std::cmp::Reverse(cand));
        }

        while let Some(current) = candidates.pop() {
            let worst_result = results.peek().map(|r| r.0.score).unwrap_or(f32::NEG_INFINITY);
            if current.score < worst_result && results.len() >= ef {
                break;
            }

            if let Some(neighbors) = self.layers[level].get(&current.id) {
                for &nbr in neighbors {
                    if visited.insert(nbr) {
                        let score = compute_vector_score(query, &all_vectors[nbr as usize], metric, is_normalized)?;
                        let cand = Candidate { id: nbr, score };
                        let worst = results.peek().map(|r| r.0.score).unwrap_or(f32::NEG_INFINITY);

                        if score > worst || results.len() < ef {
                            candidates.push(cand);
                            results.push(std::cmp::Reverse(cand));
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = results
            .into_iter()
            .map(|r| r.0)
            .filter(|c| {
                if let Some(m) = mask {
                    let idx = c.id as usize;
                    idx < m.len() && m[idx]
                } else {
                    true
                }
            })
            .collect();

        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        Ok(out)
    }

    /// Queries the HNSW graph starting from top layer down to layer 0.
    pub fn search(
        &self,
        query: &[f32],
        all_vectors: &[Vec<f32>],
        k: usize,
        ef_search: usize,
        metric: Metric,
        is_normalized: bool,
        mask: Option<&[bool]>,
    ) -> Result<Vec<(u32, f32)>, SutraError> {
        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };

        let mut curr_obj = ep;
        let mut curr_score = compute_vector_score(query, &all_vectors[curr_obj as usize], metric, is_normalized)?;

        // Top layers: 1-element greedy routing
        for l in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if let Some(neighbors) = self.layers[l].get(&curr_obj) {
                    for &nbr in neighbors {
                        let score = compute_vector_score(query, &all_vectors[nbr as usize], metric, is_normalized)?;
                        if score > curr_score {
                            curr_score = score;
                            curr_obj = nbr;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Layer 0: beam search
        let candidates = self.search_layer(
            query,
            &[curr_obj],
            ef_search.max(k),
            0,
            all_vectors,
            metric,
            is_normalized,
            mask,
        )?;

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

        // Search before deletion
        let q = vec![1.0, 0.0];
        let res = index.search(&q, &vectors, 2, 16, Metric::Cosine, true, None).unwrap();
        assert!(!res.is_empty());
        assert_eq!(res[0].0, 0);

        // Delete entry point or node 0
        let ep = index.entry_point.unwrap();
        let deleted = index.delete_node(ep, &vectors, Metric::Cosine, true);
        assert!(deleted);
        assert!(!index.node_levels.contains_key(&ep));
        assert!(index.entry_point.is_some());
        assert_ne!(index.entry_point.unwrap(), ep);

        // Search again, should still return nearest valid remaining node
        let res2 = index.search(&q, &vectors, 2, 16, Metric::Cosine, true, None).unwrap();
        assert!(!res2.is_empty());
        assert_ne!(res2[0].0, ep);
    }
}
