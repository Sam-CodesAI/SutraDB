pub mod tokenizer;
pub mod stemmer;
pub mod stopwords;
pub mod analyzer;

pub use tokenizer::tokenize;
pub use analyzer::{Analyzer, AnalyzerConfig};

use std::collections::HashMap;

/// In-memory BM25Okapi lexical index with Robertson-Spärck Jones IDF,
/// multi-lingual linguistic analyzer (Porter stemming, stopwords, CJK, identifier splitting),
/// and in-flight bitmasking support.
#[derive(Debug, Clone)]
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
    pub idf_cache: HashMap<String, f32>,
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
            idf_cache: HashMap::new(),
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
        self.idf_cache.clear();
    }

    pub fn get_idf(&mut self, term: &str) -> f32 {
        if let Some(&idf) = self.idf_cache.get(term) {
            return idf;
        }

        let n_q = *self.doc_frequencies.get(term).unwrap_or(&0) as f32;
        let idf = if n_q == 0.0 {
            0.0
        } else {
            let n = self.doc_count as f32;
            // Robertson-Spärck Jones non-negative formulation: ln(1 + (N - n_q + 0.5) / (n_q + 0.5))
            ((1.0 + (n - n_q + 0.5) / (n_q + 0.5))).ln().max(0.0)
        };

        self.idf_cache.insert(term.to_string(), idf);
        idf
    }

    pub fn score_query(&mut self, query: &str, mask: Option<&[bool]>) -> Vec<f32> {
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
