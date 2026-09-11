"""
Tests for the BM25Okapi lexical search engine.
"""

import numpy as np
from sutradb.bm25 import BM25Index, tokenize


def test_tokenize():
    text = "SutraDB: The ultra-fast Vector Database for 2026!"
    tokens = tokenize(text)
    assert "sutradb" in tokens
    assert "ultra-fast" in tokens or "ultra" in tokens
    assert "vector" in tokens
    assert "2026" in tokens


def test_bm25_scoring():
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Deep learning and transformer models require vector embeddings",
        "Python database for high speed nearest neighbor search",
        "Vector search engine with BM25 hybrid ranking"
    ]

    index = BM25Index()
    index.add_documents(corpus)

    # Query matching doc 3
    scores = index.score_query("nearest neighbor search")
    assert scores[2] > 0.0
    assert np.argmax(scores) == 2

    # Query matching doc 1 and doc 3
    scores_vector = index.score_query("vector")
    assert scores_vector[1] > 0.0
    assert scores_vector[3] > 0.0
    assert scores_vector[0] == 0.0


def test_bm25_masking():
    corpus = [
        "high performance vector search in python",
        "high performance vector index in rust",
        "high performance vector memory in c++"
    ]

    index = BM25Index()
    index.add_documents(corpus)

    # Mask out doc 0
    mask = np.array([False, True, True])
    scores = index.score_query("high performance", mask=mask)
    assert scores[0] == 0.0
    assert scores[1] > 0.0
    assert scores[2] > 0.0
