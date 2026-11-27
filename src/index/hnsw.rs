use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::{Mutex, RwLock};
use rand::prelude::*;
use rayon::prelude::*;
use roaring::RoaringBitmap;
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

/// Abstract filter predicate evaluated inside HNSW Layer-0 traversal before inserting into `ef` results.
#[derive(Copy, Clone)]
enum FilterMask<'a> {
    None,
    Slice(&'a [bool]),
    Bitmap(&'a RoaringBitmap),
}

impl<'a> FilterMask<'a> {
    #[inline]
    fn contains(&self, id: u32) -> bool {
        match self {
            FilterMask::None => true,
            FilterMask::Slice(m) => {
                let idx = id as usize;
                idx < m.len() && m[idx]
            }
            FilterMask::Bitmap(bm) => bm.contains(id),
        }
    }

    fn is_filtered(&self) -> bool {
        !matches!(self, FilterMask::None)
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

    fn reset(&mut self, capacity: usize) {
        let needed = capacity.div_ceil(64);
        if self.bits.len() < needed {
            self.bits.resize(needed, 0u64);
        }
        self.bits[..needed].fill(0u64);
        self.overflow.clear();
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

struct SearchScratch {
    visited: VisitedTracker,
    results: BinaryHeap<std::cmp::Reverse<Candidate>>,
    candidates: BinaryHeap<Candidate>,
    nbr_buf: Vec<u32>,
}

impl SearchScratch {
    fn new(capacity: usize, ef: usize) -> Self {
        Self {
            visited: VisitedTracker::new(capacity),
            results: BinaryHeap::with_capacity(ef + 1),
            candidates: BinaryHeap::with_capacity(ef * 2),
            nbr_buf: Vec::with_capacity(64),
        }
    }
}

trait NeighborSource {
    fn node_len(&self) -> usize;
    fn read_neighbors<'a>(&'a self, id: u32, scratch: &'a mut Vec<u32>) -> &'a [u32];
}

impl NeighborSource for [Vec<u32>] {
    #[inline]
    fn node_len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn read_neighbors<'a>(&'a self, id: u32, _scratch: &'a mut Vec<u32>) -> &'a [u32] {
        self.get(id as usize).map(Vec::as_slice).unwrap_or(&[])
    }
}

impl NeighborSource for [RwLock<Vec<u32>>] {
    #[inline]
    fn node_len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn read_neighbors<'a>(&'a self, id: u32, scratch: &'a mut Vec<u32>) -> &'a [u32] {
        scratch.clear();
        if let Some(lock) = self.get(id as usize) {
            let guard = lock.read().unwrap();
            scratch.extend_from_slice(&guard);
        }
        scratch.as_slice()
    }
}

/// Hierarchical Navigable Small World (HNSW) graph index.
/// Uses flat `Vec<Vec<Vec<u32>>>` adjacency arrays indexed directly by dense `u32` node IDs
/// and `RoaringBitmap` tombstones for O(1) logical deletion.
#[derive(Debug, Clone)]
pub struct HNSWIndex {
    pub config: HNSWConfig,
    pub entry_point: Option<u32>,
    pub max_level: usize,
    /// `layers[level][node_id] -> neighbor node IDs`
    pub layers: Vec<Vec<Vec<u32>>>,
    /// Assigned max level per dense node ID (`None` if slot unused / purged)
    pub node_levels: Vec<Option<usize>>,
    /// O(1) deleted-node tombstones cleaned up during `compact()` / `compact_tombstones()`
    pub tombstones: RoaringBitmap,
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
            node_levels: Vec::new(),
            tombstones: RoaringBitmap::new(),
            rng,
        }
    }

    /// Reconstructs an `HNSWIndex` directly from persisted `.sutra.idx` state.
    pub fn from_persisted(
        config: HNSWConfig,
        entry_point: Option<u32>,
        max_level: usize,
        layers: Vec<Vec<Vec<u32>>>,
        node_levels: Vec<Option<usize>>,
        tombstones: RoaringBitmap,
    ) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            config,
            entry_point,
            max_level,
            layers,
            node_levels,
            tombstones,
            rng,
        }
    }

    /// Approximate heap bytes occupied by HNSW adjacency lists and level metadata.
    pub fn memory_bytes(&self) -> usize {
        let levels_bytes = self.node_levels.capacity() * std::mem::size_of::<Option<usize>>();
        let layers_outer = self.layers.capacity() * std::mem::size_of::<Vec<Vec<u32>>>();
        let adj_bytes: usize = self
            .layers
            .iter()
            .map(|layer| {
                layer.capacity() * std::mem::size_of::<Vec<u32>>()
                    + layer.iter().map(|v| v.capacity() * 4).sum::<usize>()
            })
            .sum();
        levels_bytes + layers_outer + adj_bytes
    }

    /// Number of currently active (non-tombstoned) nodes in the graph.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.node_levels
            .iter()
            .enumerate()
            .filter(|&(i, lvl)| lvl.is_some() && !self.tombstones.contains(i as u32))
            .count()
    }

    /// Returns assigned max level for `node_id` if present and active.
    #[inline]
    pub fn node_level(&self, node_id: u32) -> Option<usize> {
        if self.tombstones.contains(node_id) {
            return None;
        }
        self.node_levels.get(node_id as usize).copied().flatten()
    }

    /// Returns true if `node_id` is active on `level`.
    #[inline]
    pub fn contains_node(&self, level: usize, node_id: u32) -> bool {
        match self.node_level(node_id) {
            Some(max_l) => level <= max_l && level < self.layers.len() && (node_id as usize) < self.layers[level].len(),
            None => false,
        }
    }

    #[inline]
    fn neighbors_at(&self, level: usize, node_id: u32) -> &[u32] {
        self.layers
            .get(level)
            .and_then(|l| l.get(node_id as usize))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn ensure_node_capacity(&mut self, max_idx: usize) {
        if self.node_levels.len() <= max_idx {
            self.node_levels.resize(max_idx + 1, None);
        }
        for layer in &mut self.layers {
            if layer.len() <= max_idx {
                layer.resize_with(max_idx + 1, Vec::new);
            }
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

    /// Dynamically inserts a document vector into the flat HNSW graph.
    pub fn insert<V: VectorStore + ?Sized>(
        &mut self,
        doc_id: u32,
        vector: &[f32],
        all_vectors: &V,
        metric: Metric,
        is_normalized: bool,
    ) -> Result<(), SutraError> {
        let idx = doc_id as usize;
        if idx < self.node_levels.len() && self.node_levels[idx].is_some() && !self.tombstones.contains(doc_id) {
            self.delete_node(doc_id, all_vectors, metric, is_normalized);
        }
        self.tombstones.remove(doc_id);

        let node_level = self.random_level();
        let node_cap = self.node_levels.len().max(idx + 1).max(all_vectors.vec_len());
        while self.layers.len() <= node_level {
            self.layers.push(vec![Vec::new(); node_cap]);
        }
        self.ensure_node_capacity(node_cap - 1);
        for l in 0..=node_level {
            self.layers[l][idx].clear();
        }
        self.node_levels[idx] = Some(node_level);

        let curr_ep = match self.entry_point {
            Some(ep) if !self.tombstones.contains(ep) => ep,
            _ => {
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

        // Top-down greedy routing above `node_level`
        for l in (node_level + 1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                for &nbr in self.neighbors_at(l, curr_obj) {
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
                FilterMask::None,
            )?;

            let filtered_cands: Vec<Candidate> = candidates
                .iter()
                .copied()
                .filter(|c| c.id != doc_id && !self.tombstones.contains(c.id))
                .collect();
            let neighbors = Self::select_neighbors_heuristic(
                &filtered_cands,
                m_max,
                all_vectors,
                metric,
                is_normalized,
            );
            self.layers[l][idx] = neighbors.clone();

            for &nbr in &neighbors {
                let nbr_idx = nbr as usize;
                let nbr_neighbors = &mut self.layers[l][nbr_idx];
                if !nbr_neighbors.contains(&doc_id) {
                    nbr_neighbors.push(doc_id);
                    if nbr_neighbors.len() > m_max {
                        let nbr_vec = all_vectors.vec_slice(nbr_idx);
                        let mut scored_nbrs: Vec<Candidate> = nbr_neighbors
                            .iter()
                            .filter(|&&n| !self.tombstones.contains(n))
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
                        self.layers[l][nbr_idx] = Self::select_neighbors_heuristic(
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
            self.max_level = node_level;
            self.entry_point = Some(doc_id);
        }

        Ok(())
    }

    /// Inserts an already level-assigned node into `self` sequentially (used for bootstrap).
    fn insert_preassigned_seq<V: VectorStore + ?Sized>(
        &mut self,
        doc_id: u32,
        node_level: usize,
        all_vectors: &V,
        metric: Metric,
        is_normalized: bool,
        scratch: &mut SearchScratch,
    ) -> Result<(), SutraError> {
        let idx = doc_id as usize;
        let vector = all_vectors.vec_slice(idx);
        let curr_ep = match self.entry_point {
            Some(ep) if !self.tombstones.contains(ep) => ep,
            _ => {
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

        for l in (node_level + 1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                for &nbr in self.neighbors_at(l, curr_obj) {
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

        let mut enter_pts = vec![curr_obj];
        let insert_top = node_level.min(self.max_level);
        for l in (0..=insert_top).rev() {
            let m_max = if l == 0 { self.config.m0 } else { self.config.m };
            let candidates = Self::search_layer_source(
                self.layers[l].as_slice(),
                &self.tombstones,
                vector,
                &enter_pts,
                self.config.ef_construction,
                all_vectors,
                metric,
                is_normalized,
                FilterMask::None,
                scratch,
            )?;

            let filtered_cands: Vec<Candidate> = candidates
                .iter()
                .copied()
                .filter(|c| c.id != doc_id && !self.tombstones.contains(c.id))
                .collect();
            let neighbors = Self::select_neighbors_heuristic(
                &filtered_cands,
                m_max,
                all_vectors,
                metric,
                is_normalized,
            );
            self.layers[l][idx] = neighbors.clone();

            for &nbr in &neighbors {
                let nbr_idx = nbr as usize;
                let nbr_neighbors = &mut self.layers[l][nbr_idx];
                if !nbr_neighbors.contains(&doc_id) {
                    nbr_neighbors.push(doc_id);
                    if nbr_neighbors.len() > m_max {
                        let nbr_vec = all_vectors.vec_slice(nbr_idx);
                        let mut scored_nbrs: Vec<Candidate> = nbr_neighbors
                            .iter()
                            .filter(|&&n| !self.tombstones.contains(n))
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
                        self.layers[l][nbr_idx] = Self::select_neighbors_heuristic(
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
            self.max_level = node_level;
            self.entry_point = Some(doc_id);
        }
        Ok(())
    }

    /// Parallel batch HNSW graph construction across CPU cores via Rayon using per-node `RwLock`
    /// adjacency lists and lock-free read snapshots during SIMD scoring.
    pub fn build_batch<V: VectorStore + Sync + ?Sized>(
        &mut self,
        doc_ids: &[u32],
        all_vectors: &V,
        metric: Metric,
        is_normalized: bool,
    ) -> Result<(), SutraError> {
        if doc_ids.is_empty() {
            return Ok(());
        }

        for &doc_id in doc_ids {
            let idx = doc_id as usize;
            if idx < self.node_levels.len()
                && self.node_levels[idx].is_some()
                && !self.tombstones.contains(doc_id)
            {
                self.delete_node(doc_id, all_vectors, metric, is_normalized);
            }
            self.tombstones.remove(doc_id);
        }

        let max_id = doc_ids.iter().copied().max().unwrap_or(0) as usize;
        let cap = (max_id + 1)
            .max(all_vectors.vec_len())
            .max(self.node_levels.len());

        let mut assignments = Vec::with_capacity(doc_ids.len());
        let mut batch_max_level = self.max_level;
        for &doc_id in doc_ids {
            let lvl = self.random_level();
            if lvl > batch_max_level {
                batch_max_level = lvl;
            }
            assignments.push((doc_id, lvl));
        }

        while self.layers.len() <= batch_max_level {
            self.layers.push(vec![Vec::new(); cap]);
        }
        self.ensure_node_capacity(cap - 1);
        for &(doc_id, lvl) in &assignments {
            let idx = doc_id as usize;
            for l in 0..=lvl {
                self.layers[l][idx].clear();
            }
            self.node_levels[idx] = Some(lvl);
        }

        let bootstrap_len = assignments.len().min(64);
        let mut boot_scratch = SearchScratch::new(cap, self.config.ef_construction);
        for &(doc_id, lvl) in &assignments[..bootstrap_len] {
            self.insert_preassigned_seq(doc_id, lvl, all_vectors, metric, is_normalized, &mut boot_scratch)?;
        }

        if bootstrap_len == assignments.len() {
            return Ok(());
        }

        let raw_layers = std::mem::take(&mut self.layers);
        let locked_layers: Vec<Vec<RwLock<Vec<u32>>>> = raw_layers
            .into_iter()
            .map(|layer| layer.into_iter().map(RwLock::new).collect())
            .collect();

        let ep_state = RwLock::new((self.entry_point, self.max_level));
        let ep_bump_mutex = Mutex::new(());
        let tombs = &self.tombstones;
        let m = self.config.m;
        let m0 = self.config.m0;
        let ef_construction = self.config.ef_construction;

        let par_result: Result<(), SutraError> = assignments[bootstrap_len..]
            .par_iter()
            .try_for_each_init(
                || SearchScratch::new(cap, ef_construction),
                |scratch, &(doc_id, node_level)| -> Result<(), SutraError> {
                    let idx = doc_id as usize;
                    let vector = all_vectors.vec_slice(idx);

                    // Serialize only the O(log N) nodes that raise global `max_level`
                    // so upper-layer hubs never fragment concurrently.
                    let _rare_guard = if node_level > ep_state.read().unwrap().1 {
                        Some(ep_bump_mutex.lock().unwrap())
                    } else {
                        None
                    };

                    let (ep_opt, top_level) = *ep_state.read().unwrap();
                    let curr_ep = match ep_opt {
                        Some(ep) => ep,
                        None => return Ok(()),
                    };

                    let mut curr_obj = curr_ep;
                    let mut curr_score = compute_vector_score(
                        vector,
                        all_vectors.vec_slice(curr_obj as usize),
                        metric,
                        is_normalized,
                    )?;

                    for l in (node_level + 1..=top_level).rev() {
                        let layer_slice = locked_layers[l].as_slice();
                        let mut changed = true;
                        while changed {
                            changed = false;
                            let nbrs = layer_slice.read_neighbors(curr_obj, &mut scratch.nbr_buf);
                            for &nbr in nbrs {
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

                    let insert_top = node_level.min(top_level);
                    let mut enter_pts = vec![curr_obj];
                    let mut selected_by_level: Vec<(usize, usize, Vec<u32>)> =
                        Vec::with_capacity(insert_top + 1);

                    for l in (0..=insert_top).rev() {
                        let m_max = if l == 0 { m0 } else { m };
                        let candidates = Self::search_layer_source(
                            locked_layers[l].as_slice(),
                            tombs,
                            vector,
                            &enter_pts,
                            ef_construction,
                            all_vectors,
                            metric,
                            is_normalized,
                            FilterMask::None,
                            scratch,
                        )?;

                        let filtered_cands: Vec<Candidate> = candidates
                            .iter()
                            .copied()
                            .filter(|c| c.id != doc_id && !tombs.contains(c.id))
                            .collect();
                        let neighbors = Self::select_neighbors_heuristic(
                            &filtered_cands,
                            m_max,
                            all_vectors,
                            metric,
                            is_normalized,
                        );
                        *locked_layers[l][idx].write().unwrap() = neighbors.clone();
                        selected_by_level.push((l, m_max, neighbors));

                        if !candidates.is_empty() {
                            enter_pts = candidates.into_iter().map(|c| c.id).collect();
                        }
                    }

                    // Publish backward edges only after `doc_id`'s own adjacency lists across all
                    // inserted layers are populated so concurrent traversals never see an empty node.
                    for (l, m_max, neighbors) in selected_by_level {
                        let layer_slice = locked_layers[l].as_slice();
                        for nbr in neighbors {
                            let nbr_idx = nbr as usize;
                            let mut nbr_guard = layer_slice[nbr_idx].write().unwrap();
                            if !nbr_guard.contains(&doc_id) {
                                nbr_guard.push(doc_id);
                                if nbr_guard.len() > m_max {
                                    let nbr_vec = all_vectors.vec_slice(nbr_idx);
                                    let mut scored_nbrs: Vec<Candidate> = nbr_guard
                                        .iter()
                                        .filter(|&&n| !tombs.contains(n))
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
                                    *nbr_guard = Self::select_neighbors_heuristic(
                                        &scored_nbrs,
                                        m_max,
                                        all_vectors,
                                        metric,
                                        is_normalized,
                                    );
                                }
                            }
                        }
                    }

                    if node_level > ep_state.read().unwrap().1 {
                        *ep_state.write().unwrap() = (Some(doc_id), node_level);
                    }

                    Ok(())
                },
            );

        self.layers = locked_layers
            .into_iter()
            .map(|layer| {
                layer
                    .into_iter()
                    .map(|lock| lock.into_inner().unwrap())
                    .collect()
            })
            .collect();
        let (final_ep, final_max_lvl) = ep_state.into_inner().unwrap();
        self.entry_point = final_ep;
        self.max_level = final_max_lvl;

        par_result
    }

    /// Marks `doc_id` deleted in O(1) via `RoaringBitmap` tombstones instead of scanning
    /// all graph adjacency lists in O(N). Dangling edges are scrubbed during `compact()`.
    pub fn delete_node<V: VectorStore + ?Sized>(
        &mut self,
        doc_id: u32,
        _all_vectors: &V,
        _metric: Metric,
        _is_normalized: bool,
    ) -> bool {
        let idx = doc_id as usize;
        if idx >= self.node_levels.len()
            || self.node_levels[idx].is_none()
            || self.tombstones.contains(doc_id)
        {
            return false;
        }

        self.tombstones.insert(doc_id);

        if self.entry_point == Some(doc_id) {
            let best = self
                .node_levels
                .iter()
                .enumerate()
                .filter_map(|(i, lvl_opt)| {
                    let id = i as u32;
                    if !self.tombstones.contains(id) {
                        lvl_opt.map(|lvl| (id, lvl))
                    } else {
                        None
                    }
                })
                .max_by_key(|&(_, lvl)| lvl);

            if let Some((new_ep, lvl)) = best {
                self.entry_point = Some(new_ep);
                self.max_level = lvl;
            } else {
                self.entry_point = None;
                self.max_level = 0;
            }
        }

        true
    }

    /// Scrubs all tombstoned node IDs and incoming references across every flat graph layer.
    pub fn compact_tombstones(&mut self) {
        if self.tombstones.is_empty() {
            return;
        }
        let tombs = &self.tombstones;
        for layer in &mut self.layers {
            for (node_idx, nbrs) in layer.iter_mut().enumerate() {
                if tombs.contains(node_idx as u32) {
                    nbrs.clear();
                } else {
                    nbrs.retain(|&n| !tombs.contains(n));
                }
            }
        }
        for id in tombs.iter() {
            let idx = id as usize;
            if idx < self.node_levels.len() {
                self.node_levels[idx] = None;
            }
        }
        self.tombstones.clear();
    }

    #[allow(clippy::too_many_arguments)]
    fn search_layer_source<V: VectorStore + ?Sized, L: NeighborSource + ?Sized>(
        layer_vecs: &L,
        tombstones: &RoaringBitmap,
        query: &[f32],
        enter_points: &[u32],
        ef: usize,
        all_vectors: &V,
        metric: Metric,
        is_normalized: bool,
        filter: FilterMask<'_>,
        scratch: &mut SearchScratch,
    ) -> Result<Vec<Candidate>, SutraError> {
        let is_valid = |id: u32| -> bool { !tombstones.contains(id) && filter.contains(id) };

        scratch
            .visited
            .reset(all_vectors.vec_len().max(layer_vecs.node_len()));
        scratch.results.clear();
        scratch.candidates.clear();

        for &ep in enter_points {
            if !scratch.visited.insert(ep) {
                continue;
            }
            let score = compute_vector_score(
                query,
                all_vectors.vec_slice(ep as usize),
                metric,
                is_normalized,
            )?;
            let cand = Candidate { id: ep, score };
            scratch.candidates.push(cand);
            if is_valid(ep) {
                scratch.results.push(std::cmp::Reverse(cand));
                if scratch.results.len() > ef {
                    scratch.results.pop();
                }
            }
        }

        while let Some(current) = scratch.candidates.pop() {
            if scratch.results.len() >= ef {
                if let Some(worst_result) = scratch.results.peek() {
                    if current.score < worst_result.0.score {
                        break;
                    }
                }
            }

            let nbrs = layer_vecs.read_neighbors(current.id, &mut scratch.nbr_buf);
            for &nbr in nbrs {
                if scratch.visited.insert(nbr) {
                    let score = compute_vector_score(
                        query,
                        all_vectors.vec_slice(nbr as usize),
                        metric,
                        is_normalized,
                    )?;
                    let cand = Candidate { id: nbr, score };
                    let worst = scratch
                        .results
                        .peek()
                        .map(|r| r.0.score)
                        .unwrap_or(f32::NEG_INFINITY);

                    // Traverse masked-out / tombstoned nodes in `candidates` to preserve graph connectivity;
                    // filter & tombstone checks run BEFORE inserting into `results`.
                    if score > worst || scratch.results.len() < ef {
                        scratch.candidates.push(cand);
                        if is_valid(nbr) {
                            scratch.results.push(std::cmp::Reverse(cand));
                            if scratch.results.len() > ef {
                                scratch.results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = scratch.results.drain().map(|r| r.0).collect();
        out.sort_by(|a, b| b.cmp(a));
        Ok(out)
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
        filter: FilterMask<'_>,
    ) -> Result<Vec<Candidate>, SutraError> {
        let layer_vecs = self.layers[level].as_slice();
        let mut scratch = SearchScratch::new(all_vectors.vec_len().max(layer_vecs.len()), ef);
        Self::search_layer_source(
            layer_vecs,
            &self.tombstones,
            query,
            enter_points,
            ef,
            all_vectors,
            metric,
            is_normalized,
            filter,
            &mut scratch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn search_internal<V: VectorStore + ?Sized>(
        &self,
        query: &[f32],
        all_vectors: &V,
        k: usize,
        ef_search: usize,
        metric: Metric,
        is_normalized: bool,
        filter: FilterMask<'_>,
    ) -> Result<Vec<(u32, f32)>, SutraError> {
        if k == 0 || self.layers.is_empty() {
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
                for &nbr in self.neighbors_at(l, curr_obj) {
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

        let total_nodes = self.node_levels.len().max(1);
        let mut enter_pts = vec![curr_obj];
        let mut effective_ef = ef_search.max(k);

        // Predicate-aware ACORN multi-seeding when filter selectivity is restrictive (< 20%)
        if filter.is_filtered() {
            let valid_ids: Vec<u32> = match filter {
                FilterMask::Bitmap(bm) => bm
                    .iter()
                    .filter(|&id| {
                        (id as usize) < self.layers[0].len()
                            && self.node_levels.get(id as usize).copied().flatten().is_some()
                            && !self.tombstones.contains(id)
                    })
                    .collect(),
                FilterMask::Slice(m) => m
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &ok)| {
                        let id = idx as u32;
                        if ok
                            && idx < self.layers[0].len()
                            && self.node_levels.get(idx).copied().flatten().is_some()
                            && !self.tombstones.contains(id)
                        {
                            Some(id)
                        } else {
                            None
                        }
                    })
                    .collect(),
                FilterMask::None => Vec::new(),
            };

            if valid_ids.is_empty() {
                return Ok(Vec::new());
            }

            let selectivity = valid_ids.len() as f32 / total_nodes as f32;
            if selectivity < 0.25 {
                let sample_cap = effective_ef.max(32).min(valid_ids.len());
                let step = (valid_ids.len() / sample_cap).max(1);
                for &vid in valid_ids.iter().step_by(step).take(sample_cap) {
                    if vid != curr_obj {
                        enter_pts.push(vid);
                    }
                }
                effective_ef = ((effective_ef as f32) / selectivity.max(0.02))
                    .ceil()
                    .clamp(effective_ef as f32, 1024.0) as usize;
            }
        }

        let mut candidates = self.search_layer(
            query,
            &enter_pts,
            effective_ef,
            0,
            all_vectors,
            metric,
            is_normalized,
            filter,
        )?;

        if candidates.len() < k && filter.is_filtered() {
            let valid_ids: Vec<u32> = match filter {
                FilterMask::Bitmap(bm) => bm
                    .iter()
                    .filter(|&id| !self.tombstones.contains(id) && (id as usize) < self.layers[0].len())
                    .collect(),
                FilterMask::Slice(m) => m
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &ok)| {
                        let id = idx as u32;
                        if ok && !self.tombstones.contains(id) && idx < self.layers[0].len() {
                            Some(id)
                        } else {
                            None
                        }
                    })
                    .collect(),
                FilterMask::None => Vec::new(),
            };
            if !valid_ids.is_empty() {
                let sample_cap = effective_ef.max(64);
                let step = (valid_ids.len() / sample_cap).max(1);
                let valid_eps: Vec<u32> = valid_ids.into_iter().step_by(step).take(sample_cap).collect();
                let extra = self.search_layer(
                    query,
                    &valid_eps,
                    effective_ef,
                    0,
                    all_vectors,
                    metric,
                    is_normalized,
                    filter,
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

        let mut results = Vec::with_capacity(k.min(candidates.len()));
        for c in candidates.into_iter().take(k) {
            results.push((c.id, c.score));
        }

        Ok(results)
    }

    /// Queries the HNSW graph using a dense boolean slice filter.
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
        let filter = match mask {
            Some(m) => FilterMask::Slice(m),
            None => FilterMask::None,
        };
        self.search_internal(query, all_vectors, k, ef_search, metric, is_normalized, filter)
    }

    /// Queries the HNSW graph directly with a pre-evaluated `RoaringBitmap` filter mask.
    #[allow(clippy::too_many_arguments)]
    pub fn search_with_bitmap<V: VectorStore + ?Sized>(
        &self,
        query: &[f32],
        all_vectors: &V,
        k: usize,
        ef_search: usize,
        metric: Metric,
        is_normalized: bool,
        filter_bitmap: Option<&RoaringBitmap>,
    ) -> Result<Vec<(u32, f32)>, SutraError> {
        let filter = match filter_bitmap {
            Some(bm) => FilterMask::Bitmap(bm),
            None => FilterMask::None,
        };
        self.search_internal(query, all_vectors, k, ef_search, metric, is_normalized, filter)
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

        assert_eq!(index.active_count(), 5);
        assert!(index.entry_point.is_some());

        let q = vec![1.0, 0.0];
        let res = index.search(&q, &vectors, 2, 16, Metric::Cosine, true, None).unwrap();
        assert!(!res.is_empty());
        assert_eq!(res[0].0, 0);

        let ep = index.entry_point.unwrap();
        let deleted = index.delete_node(ep, &vectors, Metric::Cosine, true);
        assert!(deleted);
        assert!(index.tombstones.contains(ep));
        assert!(index.entry_point.is_some());
        assert_ne!(index.entry_point.unwrap(), ep);

        let res2 = index.search(&q, &vectors, 2, 16, Metric::Cosine, true, None).unwrap();
        assert!(!res2.is_empty());
        assert_ne!(res2[0].0, ep);

        index.compact_tombstones();
        assert!(index.tombstones.is_empty());
        assert!(index.node_level(ep).is_none());
    }
}
