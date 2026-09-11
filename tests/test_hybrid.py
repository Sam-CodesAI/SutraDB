"""
Tests for hybrid search and Reciprocal Rank Fusion in Collection.
"""

import numpy as np
from sutradb.core import Collection, Document


def test_collection_hybrid_query():
    col = Collection(name="test_hybrid", dimension=4, metric="cosine")

    # Document 0: High semantic match to query vector, but does NOT contain keyword "SKU-9941"
    # Document 1: Low semantic match, but contains exact keyword "SKU-9941"
    # Document 2: High semantic match AND contains keyword "SKU-9941"
    docs = [
        Document(
            id="doc-0",
            vector=[1.0, 0.0, 0.0, 0.0],
            text="High performance neural search engine with distributed embeddings",
            metadata={"category": "ai", "price": 100}
        ),
        Document(
            id="doc-1",
            vector=[0.0, 1.0, 0.0, 0.0],
            text="Replacement item part SKU-9941 available in warehouse",
            metadata={"category": "hardware", "price": 50}
        ),
        Document(
            id="doc-2",
            vector=[0.9, 0.1, 0.0, 0.0],
            text="Neural engine hardware accelerator SKU-9941 for AI acceleration",
            metadata={"category": "hardware", "price": 200}
        )
    ]

    col.insert(docs)
    assert col.count() == 3

    # Query with vector matching doc-0 & doc-2, and text keyword "SKU-9941" matching doc-1 & doc-2
    query_vector = [1.0, 0.0, 0.0, 0.0]
    query_text = "SKU-9941"

    results = col.query(
        vector=query_vector,
        text=query_text,
        hybrid=True,
        top_k=3
    )

    assert len(results) == 3
    # doc-2 matches BOTH vector and text, so it should rank #1 in RRF!
    assert results[0].id == "doc-2"
    assert results[0].dense_score is not None
    assert results[0].bm25_score is not None


def test_collection_metadata_filtered_query():
    col = Collection(name="test_filtered", dimension=3, metric="cosine")
    docs = [
        Document(id="a", vector=[1.0, 0.0, 0.0], text="Apple iPhone", metadata={"brand": "Apple", "in_stock": True}),
        Document(id="b", vector=[0.9, 0.1, 0.0], text="Apple iPad", metadata={"brand": "Apple", "in_stock": False}),
        Document(id="c", vector=[0.1, 0.9, 0.0], text="Samsung Galaxy", metadata={"brand": "Samsung", "in_stock": True}),
    ]
    col.insert(docs)

    # Query for Apple products that are in stock
    results = col.query(
        vector=[1.0, 0.0, 0.0],
        filter={"brand": "Apple", "in_stock": True},
        top_k=5
    )

    assert len(results) == 1
    assert results[0].id == "a"


def test_collection_delete():
    col = Collection(name="test_delete", dimension=2, metric="cosine")
    col.insert([
        Document(id="d1", vector=[1.0, 0.0], text="First"),
        Document(id="d2", vector=[0.0, 1.0], text="Second"),
    ])
    assert col.count() == 2

    deleted = col.delete("d1")
    assert deleted == 1
    assert col.count() == 1
    assert col.get("d1") is None
    assert col.get("d2") is not None
