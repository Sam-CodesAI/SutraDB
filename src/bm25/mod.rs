pub mod tokenizer;
pub mod stemmer;
pub mod stopwords;
pub mod analyzer;

pub use tokenizer::tokenize;
pub use analyzer::{Analyzer, AnalyzerConfig};

use std::collections::HashMap;
use std::sync::RwLock;

/// In-memory BM25Okapi lexical index with Robertson-Spärck Jones IDF,
/// multi-lingual linguistic analyzer (Porter stemming, stopwords, CJK, identifier splitting),
/// and in-flight bitmasking support.
#[derive(Debug)]
pub struct BM25Index {
    pub k1: f32,
    pub b: f32,
    pub doc_count: u32,
    pub doc_lengths: Vec<u32>,
    pub total_length: u64,
    pub analyzer: Analyzer,
    // term -> list of (doc_index, term_frequency)
    pub inverted_index: HashMap<String, Vec<(u32, u32)>>,
    // term -> number of docs containing term
    pub doc_frequencies: HashMap<String, u32>,
    pub idf_cache: RwLock<HashMap<String, f32>>,
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
            total_length: self.total_length,
            analyzer: self.analyzer.clone(),
            inverted_index: self.inverted_index.clone(),
            doc_frequencies: self.doc_frequencies.clone(),
            idf_cache: RwLock::new(cache),
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
            total_length: 0,
            analyzer,
            inverted_index: HashMap::new(),
            doc_frequencies: HashMap::new(),
            idf_cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn avg_doc_length(&self) -> f32 {
        if self.doc_count == 0 {
            0.0
        } else {
            (self.total_length as f32) / (self.doc_count as f32)
        }
    }

    pub fn add_documents(&mut self, corpus: &[String]) {
        let start_idx = self.doc_count;
        for (i, text) in corpus.iter().enumerate() {
            let doc_idx = start_idx + i as u32;
            let tokens = self.analyzer.analyze(text);
            let len = tokens.len() as u32;
            self.doc_lengths.push(len);
            self.total_length += len as u64;

            let mut counts: HashMap<String, u32> = HashMap::new();
            for t in tokens {
                *counts.entry(t).or_insert(0) += 1;
            }

            for (term, freq) in counts {
                self.inverted_index
                    .entry(term.clone())
                    .or_default()
                    .push((doc_idx, freq));
                *self.doc_frequencies.entry(term).or_insert(0) += 1;
            }
        }

        self.doc_count += corpus.len() as u32;
        let mut cache = self
            .idf_cache
            .write()
            .unwrap_or_else(|p| p.into_inner());
        cache.clear();
    }

    /// In-place update of an existing document's lexical terms.
    /// Removes stale postings and doc frequencies, updates doc length and total length,
    /// inserts new postings, and invalidates the IDF cache.
    pub fn update_document(&mut self, doc_idx: u32, old_text: &str, new_text: &str) {
        if old_text == new_text {
            return;
        }

        let idx = doc_idx as usize;

        // 1. Remove old terms
        let old_tokens = self.analyzer.analyze(old_text);
        let mut old_counts: HashMap<String, u32> = HashMap::new();
        for t in old_tokens {
            *old_counts.entry(t).or_insert(0) += 1;
        }

        for term in old_counts.keys() {
            if let Some(postings) = self.inverted_index.get_mut(term) {
                postings.retain(|&(d, _)| d != doc_idx);
                if postings.is_empty() {
                    self.inverted_index.remove(term);
                }
            }
            if let Some(df) = self.doc_frequencies.get_mut(term) {
                if *df > 0 {
                    *df -= 1;
                }
                if *df == 0 {
                    self.doc_frequencies.remove(term);
                }
            }
        }

        if idx < self.doc_lengths.len() {
            self.total_length = self.total_length.saturating_sub(self.doc_lengths[idx] as u64);
        }

        // 2. Add new terms
        let new_tokens = self.analyzer.analyze(new_text);
        let new_len = new_tokens.len() as u32;
        if idx < self.doc_lengths.len() {
            self.doc_lengths[idx] = new_len;
        } else {
            while self.doc_lengths.len() <= idx {
                self.doc_lengths.push(0);
            }
            self.doc_lengths[idx] = new_len;
            if self.doc_count <= doc_idx {
                self.doc_count = doc_idx + 1;
            }
        }
        self.total_length += new_len as u64;

        let mut new_counts: HashMap<String, u32> = HashMap::new();
        for t in new_tokens {
            *new_counts.entry(t).or_insert(0) += 1;
        }

        for (term, freq) in new_counts {
            self.inverted_index
                .entry(term.clone())
                .or_default()
                .push((doc_idx, freq));
            *self.doc_frequencies.entry(term).or_insert(0) += 1;
        }

        let mut cache = self
            .idf_cache
            .write()
            .unwrap_or_else(|p| p.into_inner());
        cache.clear();
    }

    pub fn get_idf(&self, term: &str) -> f32 {
        {
            let cache = self
                .idf_cache
                .read()
                .unwrap_or_else(|p| p.into_inner());
            if let Some(&idf) = cache.get(term) {
                return idf;
            }
        }

        let n_q = *self.doc_frequencies.get(term).unwrap_or(&0) as f32;
        let idf = if n_q == 0.0 {
            0.0
        } else {
            let n = self.doc_count as f32;
            // Robertson-Spärck Jones non-negative formulation: ln(1 + (N - n_q + 0.5) / (n_q + 0.5))
            ((1.0 + (n - n_q + 0.5) / (n_q + 0.5))).ln().max(0.0)
        };

        {
            let mut cache = self
                .idf_cache
                .write()
                .unwrap_or_else(|p| p.into_inner());
            cache.insert(term.to_string(), idf);
        }
        idf
    }

    pub fn score_query(&self, query: &str, mask: Option<&[bool]>) -> Vec<f32> {
        if self.doc_count == 0 {
            return Vec::new();
        }

        let mut scores = vec![0.0f32; self.doc_count as usize];
        let query_terms = self.analyzer.analyze(query);
        if query_terms.is_empty() {
            return scores;
        }

        let avg_dl = self.avg_doc_length().max(1.0);
        let mut q_counts: HashMap<String, u32> = HashMap::new();
        for t in query_terms {
            *q_counts.entry(t).or_insert(0) += 1;
        }

        for (term, _) in q_counts {
            let idf = self.get_idf(&term);
            if idf <= 0.0 {
                continue;
            }

            if let Some(postings) = self.inverted_index.get(&term) {
                for &(doc_idx, tf) in postings {
                    let idx = doc_idx as usize;
                    if let Some(m) = mask {
                        if idx >= m.len() || !m[idx] {
                            continue;
                        }
                    }

                    let doc_len = self.doc_lengths[idx] as f32;
                    let tf_f = tf as f32;
                    let denom = tf_f + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_dl));
                    let numerator = tf_f * (self.k1 + 1.0);
                    scores[idx] += idf * (numerator / denom);
                }
            }
        }

        scores
    }
}
