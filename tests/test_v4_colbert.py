import pytest
from sutradb import Document, SearchResult
from sutradb._sutradb import NativeCollection


def test_native_colbert_multivector_insertion_and_maxsim():
    """Verify ColBERT multi-vector token embedding insertion and late-interaction MaxSim query."""
    col = NativeCollection(name="colbert_test", dimension=4, metric="cosine")

    # Document 1: 2 tokens strongly aligned with [1, 0, 0, 0] and [0, 1, 0, 0]
    doc1 = {
        "id": "doc_1",
        "text": "Deep late interaction neural retrieval with ColBERT",
        "metadata": {"category": "ml"},
        "colbert_tokens": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
    }

    # Document 2: 2 tokens orthogonal [0, 0, 1, 0] and [0, 0, 0, 1]
    doc2 = {
        "id": "doc_2",
        "text": "Relational storage systems and B-tree indexes",
        "metadata": {"category": "database"},
        "colbert_tokens": [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }

    vecs = [
        [0.7071, 0.7071, 0.0, 0.0],
        [0.0, 0.0, 0.7071, 0.7071],
    ]

    inserted = col.insert([doc1, doc2], vecs)
    assert inserted == 2
    assert col.count() == 2

    # Query with 2 tokens matching doc 1 exactly
    query_tokens = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    results = col.query_colbert(query_tokens, top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "doc_1"
    # MaxSim score for doc_1 should be max(1, 0) + max(0, 1) = 2.0
    assert abs(results[0]["score"] - 2.0) < 1e-4
    assert abs(results[0]["colbert_score"] - 2.0) < 1e-4

    # Doc 2 should have score near 0.0
    assert results[1]["id"] == "doc_2"
    assert abs(results[1]["score"] - 0.0) < 1e-4


def test_colbert_centroid_training_and_clustering():
    """Verify spherical K-Means centroid training and candidate filtering for ColBERT."""
    col = NativeCollection(name="colbert_cluster_test", dimension=4, metric="cosine")

    docs = []
    vecs = []
    for i in range(10):
        docs.append({
            "id": f"doc_{i}",
            "text": f"Document {i}",
            "metadata": {"idx": i},
            "colbert_tokens": [
                [1.0 if (i % 2 == 0) else 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0 if (i % 2 != 0) else 0.0, 0.0, 0.0],
            ]
        })
        vecs.append([1.0, 0.0, 0.0, 0.0])

    col.insert(docs, vecs)
    assert col.count() == 10

    # Train centroids with k=2
    col.train_colbert_centroids(num_centroids=2, max_iters=10)

    # Query with token [1, 0, 0, 0]
    query_tokens = [[1.0, 0.0, 0.0, 0.0]]
    results = col.query_colbert(query_tokens, top_k=5)
    assert len(results) == 5
    # Even-indexed documents should have maxsim score 1.0
    assert abs(results[0]["score"] - 1.0) < 1e-4


def test_quad_modal_rrf_with_colbert():
    """Verify Quad-Modal Reciprocal Rank Fusion fusing Dense + BM25 + Sparse + ColBERT."""
    col = NativeCollection(name="quad_modal_test", dimension=4, metric="cosine")

    doc1 = {
        "id": "doc_perfect",
        "text": "Vector search engine optimization",
        "sparse_vector": {100: 2.0, 200: 3.0},
        "colbert_tokens": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        "metadata": {"tier": "gold"},
    }
    doc2 = {
        "id": "doc_partial",
        "text": "Database indexing and clustering",
        "sparse_vector": {300: 1.0},
        "colbert_tokens": [[0.0, 0.0, 1.0, 0.0]],
        "metadata": {"tier": "silver"},
    }

    col.insert([doc1, doc2], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    # Query across all 4 modalities
    query_tokens = [[1.0, 0.0, 0.0, 0.0]]
    results = col.query(
        vector=[1.0, 0.0, 0.0, 0.0],
        text="Vector search",
        sparse_vector={100: 1.0},
        colbert_tokens=query_tokens,
        hybrid=True,
        top_k=2,
        dense_weight=1.0,
        bm25_weight=1.0,
        sparse_weight=1.0,
        colbert_weight=1.0,
    )

    assert len(results) == 2
    assert results[0]["id"] == "doc_perfect"
    assert results[0]["dense_score"] is not None
    assert results[0]["bm25_score"] is not None
    assert results[0]["sparse_score"] is not None
    assert results[0]["colbert_score"] is not None
