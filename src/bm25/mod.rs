pub mod tokenizer;
pub mod stemmer;
pub mod stopwords;
pub mod analyzer;

pub use tokenizer::tokenize;
pub use analyzer::{Analyzer, AnalyzerConfig};

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};
use std::sync::RwLock;
use roaring::RoaringBitmap;

/// Lightweight non-cryptographic FxHasher eliminating SipHash overhead during sparse posting accumulation.
#[derive(Default, Clone)]
struct FxBuildHasher;

struct FxHasher(u64);

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;
    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        FxHasher(0)
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ (b as u64)).wrapping_mul(0x517cc1b727220a95);
        }
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ (i as u64)).wrapping_mul(0x517cc1b727220a95);
    }
}

/// In-memory BM25Okapi lexical index with Robertson-Spärck Jones IDF,
/// generation-stamped tombstone invalidation (O(doc_terms) updates without O(postings) scans),
/// and sparse query score accumulation scaling with matching postings rather than corpus size.
#[derive(Debug)]
pub struct BM25Index {
    pub k1: f32,
    pub b: f32,
    pub doc_count: u32,
    pub doc_lengths: Vec<u32>,
    pub doc_active: Vec<bool>,
    pub doc_generations: Vec<u32>,
    pub tombstones: RoaringBitmap,
    pub total_length: u64,
    pub analyzer: Analyzer,
    // term -> list of (doc_index, term_frequency, doc_generation)
    pub inverted_index: HashMap<String, Vec<(u32, u32, u32)>>,
    // term -> number of active docs containing term
    pub doc_frequencies: HashMap<String, u32>,
    // Retained for backward structural compatibility; queries compute IDF inline without lock thrashing.
    pub idf_cache: RwLock<HashMap<String, f32>>,
    stale_postings: usize,
}

impl Clone for BM25Index {
    fn clone(&self) -> Self {
        let cache = self
            .idf_cache
            .read()
            .map(|c| c.clone())
            .unwrap_or_else(|p| p.into_inner().clone());
        Self {
            k1: self.k1,
            b: self.b,
            doc_count: self.doc_count,
            doc_lengths: self.doc_lengths.clone(),
            doc_active: self.doc_active.clone(),
            doc_generations: self.doc_generations.clone(),
            tombstones: self.tombstones.clone(),
            total_length: self.total_length,
            analyzer: self.analyzer.clone(),
            inverted_index: self.inverted_index.clone(),
            doc_frequencies: self.doc_frequencies.clone(),
            idf_cache: RwLock::new(cache),
            stale_postings: self.stale_postings,
        }
    }
}

impl Default for BM25Index {
    fn default() -> Self {
        Self::new(1.5, 0.75)
    }
}

impl BM25Index {
    pub fn new(k1: f32, b: f32) -> Self {
        Self::with_analyzer(k1, b, Analyzer::default())
    }

    pub fn with_analyzer(k1: f32, b: f32, analyzer: Analyzer) -> Self {
        Self {
            k1,
            b,
            doc_count: 0,
            doc_lengths: Vec::new(),
            doc_active: Vec::new(),
            doc_generations: Vec::new(),
            tombstones: RoaringBitmap::new(),
            total_length: 0,
            analyzer,
            inverted_index: HashMap::new(),
            doc_frequencies: HashMap::new(),
            idf_cache: RwLock::new(HashMap::new()),
            stale_postings: 0,
        }
    }

    #[inline]
    pub fn avg_doc_length(&self) -> f32 {
        if self.doc_count == 0 {
            0.0
        } else {
            (self.total_length as f32) / (self.doc_count as f32)
        }
    }

    pub fn add_documents(&mut self, corpus: &[String]) {
        let start_idx = self.doc_lengths.len() as u32;
        for (i, text) in corpus.iter().enumerate() {
            let doc_idx = start_idx + i as u32;
            let tokens = self.analyzer.analyze(text);
            let len = tokens.len() as u32;
            self.doc_lengths.push(len);
            self.doc_active.push(true);
            self.doc_generations.push(0);
            self.total_length += len as u64;

            let mut counts: HashMap<String, u32> = HashMap::new();
            for t in tokens {
                *counts.entry(t).or_insert(0) += 1;
            }

            for (term, freq) in counts {
                self.inverted_index
                    .entry(term.clone())
                    .or_default()
                    .push((doc_idx, freq, 0));
                *self.doc_frequencies.entry(term).or_insert(0) += 1;
            }
        }

        self.doc_count += corpus.len() as u32;
    }

    fn decrement_doc_frequencies(&mut self, old_text: &str) {
        let old_tokens = self.analyzer.analyze(old_text);
        let mut seen_terms = HashSet::new();
        for t in old_tokens {
            if !seen_terms.insert(t.clone()) {
                continue;
            }
            self.stale_postings += 1;
            if let Some(df) = self.doc_frequencies.get_mut(&t) {
                if *df > 0 {
                    *df -= 1;
                }
                if *df == 0 {
                    self.doc_frequencies.remove(&t);
                }
            }
        }
    }

    /// Compacts tombstoned and superseded postings lists in a single pass.
    pub fn compact_postings(&mut self) {
        let gens = &self.doc_generations;
        let tombs = &self.tombstones;
        self.inverted_index.retain(|term, postings| {
            if !self.doc_frequencies.contains_key(term) {
                return false;
            }
            postings.retain(|&(doc_idx, _, gen)| {
                let idx = doc_idx as usize;
                idx < gens.len() && gens[idx] == gen && !tombs.contains(doc_idx)
            });
            !postings.is_empty()
        });
        self.stale_postings = 0;
    }

    /// Removes a tombstoned document in O(doc_terms) without scanning postings lists.
    pub fn remove_document(&mut self, doc_idx: u32, old_text: &str) {
        let idx = doc_idx as usize;
        if idx >= self.doc_active.len() || !self.doc_active[idx] {
            return;
        }

        self.decrement_doc_frequencies(old_text);

        if idx < self.doc_lengths.len() {
            self.total_length = self.total_length.saturating_sub(self.doc_lengths[idx] as u64);
            self.doc_lengths[idx] = 0;
        }
        self.doc_active[idx] = false;
        self.tombstones.insert(doc_idx);
        if idx < self.doc_generations.len() {
            self.doc_generations[idx] = self.doc_generations[idx].wrapping_add(1);
        }
        self.doc_count = self.doc_count.saturating_sub(1);
    }

    /// In-place update using generation stamping in O(doc_terms) — avoids O(postings) `retain()`.
    pub fn update_document(&mut self, doc_idx: u32, old_text: &str, new_text: &str) {
        let idx = doc_idx as usize;
        let was_active = idx < self.doc_active.len() && self.doc_active[idx];
        if was_active && old_text == new_text {
            return;
        }

        if was_active {
            self.decrement_doc_frequencies(old_text);
            if idx < self.doc_lengths.len() {
                self.total_length = self.total_length.saturating_sub(self.doc_lengths[idx] as u64);
            }
        }

        let new_tokens = self.analyzer.analyze(new_text);
        let new_len = new_tokens.len() as u32;
        while self.doc_lengths.len() <= idx {
            self.doc_lengths.push(0);
            self.doc_active.push(false);
            self.doc_generations.push(0);
        }
        self.doc_lengths[idx] = new_len;
        if !self.doc_active[idx] {
            self.doc_active[idx] = true;
            self.doc_count += 1;
        }
        self.tombstones.remove(doc_idx);
        let next_gen = self.doc_generations[idx].wrapping_add(1);
        self.doc_generations[idx] = next_gen;
        self.total_length += new_len as u64;

        let mut new_counts: HashMap<String, u32> = HashMap::new();
        for t in new_tokens {
            *new_counts.entry(t).or_insert(0) += 1;
        }

        for (term, freq) in new_counts {
            self.inverted_index
                .entry(term.clone())
                .or_default()
                .push((doc_idx, freq, next_gen));
            *self.doc_frequencies.entry(term).or_insert(0) += 1;
        }

        if self.stale_postings > 4096 && self.stale_postings > self.doc_lengths.len() {
            self.compact_postings();
        }
    }

    /// Computes Robertson-Spärck Jones IDF inline in O(1) without RwLock acquisition.
    #[inline]
    pub fn get_idf(&self, term: &str) -> f32 {
        let n_q = *self.doc_frequencies.get(term).unwrap_or(&0) as f32;
        if n_q == 0.0 || self.doc_count == 0 {
            0.0
        } else {
            let n = self.doc_count as f32;
            (1.0 + (n - n_q + 0.5) / (n_q + 0.5)).ln().max(0.0)
        }
    }

    /// Internal fast sparse accumulator using pre-sized FxHash + single-term no-hash fast path.
    fn accumulate_sparse_scores(&self, query: &str, mask: Option<&[bool]>) -> Vec<(u32, f32)> {
        let slot_count = self.doc_lengths.len();
        if self.doc_count == 0 || slot_count == 0 {
            return Vec::new();
        }

        let query_terms = self.analyzer.analyze(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let avg_dl = self.avg_doc_length().max(1.0);
        let mut seen_terms: HashSet<&str, FxBuildHasher> =
            HashSet::with_capacity_and_hasher(query_terms.len(), FxBuildHasher);
        type TermPostings<'a> = (f32, &'a [(u32, u32, u32)]);
        let mut active_terms: Vec<TermPostings<'_>> = Vec::with_capacity(query_terms.len());
        let mut total_postings = 0usize;

        for term in &query_terms {
            if !seen_terms.insert(term.as_str()) {
                continue;
            }
            let idf = self.get_idf(term);
            if idf <= 0.0 {
                continue;
            }
            if let Some(postings) = self.inverted_index.get(term) {
                if !postings.is_empty() {
                    total_postings += postings.len();
                    active_terms.push((idf, postings));
                }
            }
        }

        if active_terms.is_empty() {
            return Vec::new();
        }

        // Single matching term fast path: each active `doc_idx` occurs at most once (zero HashMap needed)
        if active_terms.len() == 1 {
            let (idf, postings) = active_terms[0];
            let mut out = Vec::with_capacity(postings.len());
            for &(doc_idx, tf, gen) in postings {
                let idx = doc_idx as usize;
                if idx >= slot_count
                    || !self.doc_active[idx]
                    || self.doc_generations[idx] != gen
                    || self.tombstones.contains(doc_idx)
                {
                    continue;
                }
                if let Some(m) = mask {
                    if idx >= m.len() || !m[idx] {
                        continue;
                    }
                }
                let doc_len = self.doc_lengths[idx] as f32;
                let tf_f = tf as f32;
                let denom = tf_f + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_dl));
                let numerator = tf_f * (self.k1 + 1.0);
                out.push((doc_idx, idf * (numerator / denom)));
            }
            return out;
        }

        let cap = total_postings.min(slot_count);
        let mut scores: HashMap<u32, f32, FxBuildHasher> =
            HashMap::with_capacity_and_hasher(cap, FxBuildHasher);

        for (idf, postings) in active_terms {
            for &(doc_idx, tf, gen) in postings {
                let idx = doc_idx as usize;
                if idx >= slot_count
                    || !self.doc_active[idx]
                    || self.doc_generations[idx] != gen
                    || self.tombstones.contains(doc_idx)
                {
                    continue;
                }
                if let Some(m) = mask {
                    if idx >= m.len() || !m[idx] {
                        continue;
                    }
                }
                let doc_len = self.doc_lengths[idx] as f32;
                let tf_f = tf as f32;
                let denom = tf_f + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_dl));
                let numerator = tf_f * (self.k1 + 1.0);
                *scores.entry(doc_idx).or_insert(0.0) += idf * (numerator / denom);
            }
        }

        scores.into_iter().collect()
    }

    /// Accumulates BM25 scores strictly over matching postings (`O(|postings|)` RAM and time),
    /// returning `(doc_idx, score)` pairs without allocating an O(N) vector across the corpus.
    pub fn score_query_sparse(&self, query: &str, mask: Option<&[bool]>) -> HashMap<u32, f32> {
        let pairs = self.accumulate_sparse_scores(query, mask);
        let mut out = HashMap::with_capacity(pairs.len());
        for (doc_idx, sc) in pairs {
            out.insert(doc_idx, sc);
        }
        out
    }

    /// Evaluates sparse BM25 scores and returns top-K `(doc_idx, score)` sorted descending.
    pub fn search_top_k(&self, query: &str, top_k: usize, mask: Option<&[bool]>) -> Vec<(usize, f32)> {
        if top_k == 0 {
            return Vec::new();
        }
        let pairs = self.accumulate_sparse_scores(query, mask);
        let mut hits: Vec<(usize, f32)> = pairs
            .into_iter()
            .filter(|&(_, sc)| sc > 0.0)
            .map(|(id, sc)| (id as usize, sc))
            .collect();
        if hits.len() > top_k {
            hits.select_nth_unstable_by(top_k - 1, |a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            hits.truncate(top_k);
        }
        hits.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        hits
    }

    /// Dense compatibility projection for inspection callers expecting a vector aligned to `doc_lengths.len()`.
    pub fn score_query(&self, query: &str, mask: Option<&[bool]>) -> Vec<f32> {
        let slot_count = self.doc_lengths.len();
        let mut scores = vec![0.0f32; slot_count];
        for (doc_idx, sc) in self.accumulate_sparse_scores(query, mask) {
            if (doc_idx as usize) < slot_count {
                scores[doc_idx as usize] = sc;
            }
        }
        scores
    }
}
