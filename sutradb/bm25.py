"""
Lexical keyword search engine implementing BM25Okapi with an in-memory inverted index.
Used in SutraDB's hybrid search pipeline alongside dense vector similarity.
"""

from collections import Counter, defaultdict
import math
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .nlp import Analyzer, AnalyzerConfig, STOPWORDS, tokenize as nlp_tokenize

_WORD_PATTERN = re.compile(r"\b[a-zA-Z0-9_\-\.\$]+\b")


def tokenize(text: str, filter_stopwords: bool = False) -> List[str]:
    """Tokenizes string into lowercase alphanumeric and symbol terms."""
    if not text:
        return []
    tokens = [t.lower() for t in _WORD_PATTERN.findall(text)]
    if filter_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return tokens


class BM25Index:
    """In-memory BM25Okapi index with linguistic analyzer & dynamic document updates."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, analyzer: Optional[Analyzer] = None):
        self.k1 = k1
        self.b = b
        self.analyzer = analyzer or Analyzer()
        self.doc_count: int = 0
        self.doc_lengths: List[int] = []
        self.doc_active: List[bool] = []
        self.total_length: int = 0
        self.avg_doc_length: float = 0.0
        
        # Inverted index: term -> list of (doc_index, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        # Document frequencies: term -> number of active docs containing term
        self.doc_frequencies: Dict[str, int] = defaultdict(int)
        # Per-slot indexed term set for O(terms) deletion
        self._doc_terms: Dict[int, Set[str]] = {}
        # Cached IDF values
        self.idf_cache: Dict[str, float] = {}

    def add_documents(self, corpus: List[str]) -> None:
        """Indexes a batch of raw text strings using the linguistic analyzer."""
        start_idx = len(self.doc_lengths)

        for i, text in enumerate(corpus):
            doc_idx = start_idx + i
            tokens = self.analyzer.analyze(text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            self.doc_active.append(True)
            self.total_length += doc_len

            counts = Counter(tokens)
            self._doc_terms[doc_idx] = set(counts.keys())
            for term, freq in counts.items():
                self.inverted_index[term].append((doc_idx, freq))
                self.doc_frequencies[term] += 1

        self.doc_count += len(corpus)
        self.avg_doc_length = (self.total_length / self.doc_count) if self.doc_count > 0 else 0.0

        # Invalidate IDF cache
        self.idf_cache.clear()

    def remove_document(self, doc_idx: int, old_text: Optional[str] = None) -> None:
        """Removes a soft-deleted document from BM25 statistics and postings."""
        if doc_idx < 0 or doc_idx >= len(self.doc_active) or not self.doc_active[doc_idx]:
            return

        terms = self._doc_terms.pop(doc_idx, None)
        if terms is None:
            if old_text is not None:
                terms = set(self.analyzer.analyze(old_text))
            else:
                terms = {
                    t for t, postings in self.inverted_index.items()
                    if any(idx == doc_idx for idx, _ in postings)
                }

        for term in terms:
            postings = self.inverted_index.get(term)
            if postings:
                new_postings = [p for p in postings if p[0] != doc_idx]
                removed = len(postings) - len(new_postings)
                if new_postings:
                    self.inverted_index[term] = new_postings
                else:
                    self.inverted_index.pop(term, None)
                if removed > 0:
                    new_df = self.doc_frequencies.get(term, 0) - 1
                    if new_df > 0:
                        self.doc_frequencies[term] = new_df
                    else:
                        self.doc_frequencies.pop(term, None)

        self.total_length = max(0, self.total_length - self.doc_lengths[doc_idx])
        self.doc_lengths[doc_idx] = 0
        self.doc_active[doc_idx] = False
        self.doc_count = max(0, self.doc_count - 1)
        self.avg_doc_length = (self.total_length / self.doc_count) if self.doc_count > 0 else 0.0
        self.idf_cache.clear()

    def update_document(self, doc_idx: int, new_text: str) -> None:
        """Updates (or re-activates) a single slot in-place without disturbing other tombstones."""
        if doc_idx < 0:
            raise ValueError(f"Invalid doc_idx: {doc_idx}")
        if doc_idx < len(self.doc_active) and self.doc_active[doc_idx]:
            self.remove_document(doc_idx)

        while len(self.doc_lengths) <= doc_idx:
            self.doc_lengths.append(0)
            self.doc_active.append(False)

        tokens = self.analyzer.analyze(new_text)
        doc_len = len(tokens)
        self.doc_lengths[doc_idx] = doc_len
        self.doc_active[doc_idx] = True
        self.total_length += doc_len
        self.doc_count += 1

        counts = Counter(tokens)
        self._doc_terms[doc_idx] = set(counts.keys())
        for term, freq in counts.items():
            self.inverted_index[term].append((doc_idx, freq))
            self.doc_frequencies[term] += 1

        self.avg_doc_length = (self.total_length / self.doc_count) if self.doc_count > 0 else 0.0
        self.idf_cache.clear()

    def _get_idf(self, term: str) -> float:
        """Calculates Robertson-Spärck Jones IDF with non-negative smoothing."""
        if term in self.idf_cache:
            return self.idf_cache[term]

        if self.doc_count == 0:
            return 0.0

        n_q = self.doc_frequencies.get(term, 0)
        if n_q == 0:
            idf = 0.0
        else:
            # Robertson-Spärck Jones formulation guaranteeing positive scores
            idf = math.log(1.0 + (self.doc_count - n_q + 0.5) / (n_q + 0.5))

        self.idf_cache[term] = idf
        return idf

    def score_query(
        self,
        query: str,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculates BM25 scores for all document slots given a query string.
        Optionally zeroes out any document indices masked as False or tombstoned.
        """
        slot_count = len(self.doc_lengths)
        if slot_count == 0:
            return np.empty(0, dtype=np.float32)

        scores = np.zeros(slot_count, dtype=np.float32)
        if self.doc_count == 0:
            return scores

        query_terms = self.analyzer.analyze(query)
        if not query_terms:
            return scores

        # Calculate term frequency in query
        q_counts = Counter(query_terms)

        for term, _ in q_counts.items():
            idf = self._get_idf(term)
            if idf <= 0:
                continue

            postings = self.inverted_index.get(term)
            if not postings:
                continue

            for doc_idx, tf in postings:
                if doc_idx >= slot_count or not self.doc_active[doc_idx]:
                    continue
                if mask is not None and not mask[doc_idx]:
                    continue

                doc_len = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_length or 1.0)))
                numerator = tf * (self.k1 + 1.0)
                scores[doc_idx] += idf * (numerator / denom)

        return scores
