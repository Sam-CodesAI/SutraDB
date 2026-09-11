"""
Lexical keyword search engine implementing BM25Okapi with an in-memory inverted index.
Used in SutraDB's hybrid search pipeline alongside dense vector similarity.
"""

from collections import Counter, defaultdict
import math
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

# Standard lightweight English stopwords
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

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
    """In-memory BM25Okapi index with dynamic document updates."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count: int = 0
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        
        # Inverted index: term -> list of (doc_index, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        # Document frequencies: term -> number of docs containing term
        self.doc_frequencies: Dict[str, int] = defaultdict(int)
        # Cached IDF values
        self.idf_cache: Dict[str, float] = {}

    def add_documents(self, corpus: List[str]) -> None:
        """Indexes a batch of raw text strings."""
        start_idx = self.doc_count
        total_len = sum(self.doc_lengths)

        for i, text in enumerate(corpus):
            doc_idx = start_idx + i
            tokens = tokenize(text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_len += doc_len

            counts = Counter(tokens)
            for term, freq in counts.items():
                self.inverted_index[term].append((doc_idx, freq))
                self.doc_frequencies[term] += 1

        self.doc_count += len(corpus)
        if self.doc_count > 0:
            self.avg_doc_length = total_len / self.doc_count

        # Invalidate IDF cache
        self.idf_cache.clear()

    def _get_idf(self, term: str) -> float:
        """Calculates Robertson-Spärck Jones IDF with non-negative smoothing."""
        if term in self.idf_cache:
            return self.idf_cache[term]

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
        Calculates BM25 scores for all documents given a query string.
        Optionally zeroes out any document indices masked as False.
        """
        if self.doc_count == 0:
            return np.empty(0, dtype=np.float32)

        scores = np.zeros(self.doc_count, dtype=np.float32)
        query_terms = tokenize(query)
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
                if mask is not None and not mask[doc_idx]:
                    continue

                doc_len = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_length or 1.0)))
                numerator = tf * (self.k1 + 1.0)
                scores[doc_idx] += idf * (numerator / denom)

        return scores
