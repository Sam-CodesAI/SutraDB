"""
Tests for compound metadata filtering engine.
"""

import numpy as np
from sutradb.filters import FilterEngine


def test_basic_comparators():
    meta = {"age": 28, "category": "engineering", "active": True}

    assert FilterEngine.evaluate(meta, {"age": 28})
    assert FilterEngine.evaluate(meta, {"age": {"$gte": 28}})
    assert FilterEngine.evaluate(meta, {"age": {"$gt": 20}})
    assert FilterEngine.evaluate(meta, {"age": {"$lte": 28}})
    assert FilterEngine.evaluate(meta, {"age": {"$lt": 30}})
    assert FilterEngine.evaluate(meta, {"age": {"$ne": 35}})
    assert not FilterEngine.evaluate(meta, {"age": {"$gt": 30}})


def test_membership_and_contains():
    meta = {
        "tags": ["python", "ai", "database"],
        "summary": "High speed vector indexing"
    }

    assert FilterEngine.evaluate(meta, {"tags": {"$contains": "ai"}})
    assert not FilterEngine.evaluate(meta, {"tags": {"$contains": "java"}})
    assert FilterEngine.evaluate(meta, {"summary": {"$contains": "vector"}})

    meta_status = {"status": "in_progress"}
    assert FilterEngine.evaluate(meta_status, {"status": {"$in": ["todo", "in_progress", "done"]}})
    assert FilterEngine.evaluate(meta_status, {"status": {"$nin": ["archived", "deleted"]}})
    assert not FilterEngine.evaluate(meta_status, {"status": {"$in": ["done", "failed"]}})


def test_compound_and_or_not():
    meta = {"role": "architect", "years": 8, "department": "infrastructure"}

    filter_and = {
        "$and": [
            {"role": "architect"},
            {"years": {"$gte": 5}}
        ]
    }
    assert FilterEngine.evaluate(meta, filter_and)

    filter_or = {
        "$or": [
            {"department": "sales"},
            {"department": "infrastructure"}
        ]
    }
    assert FilterEngine.evaluate(meta, filter_or)

    filter_not = {
        "$not": {"department": "marketing"}
    }
    assert FilterEngine.evaluate(meta, filter_not)


def test_build_mask():
    metadata_list = [
        {"category": "tech", "score": 90},
        {"category": "finance", "score": 85},
        {"category": "tech", "score": 70},
        {"category": "health", "score": 95},
    ]

    filter_spec = {
        "category": "tech",
        "score": {"$gte": 80}
    }

    mask = FilterEngine.build_mask(metadata_list, filter_spec)
    assert np.array_equal(mask, np.array([True, False, False, False]))


def test_nested_dict_filter():
    meta = {
        "author": {"name": "Alice", "role": "admin"},
        "status": "active"
    }

    # Exact dictionary match should succeed without Unknown filter operator error
    assert FilterEngine.evaluate(meta, {"author": {"name": "Alice", "role": "admin"}})
    assert not FilterEngine.evaluate(meta, {"author": {"name": "Bob", "role": "admin"}})
    # Standard operator should also work with nested condition
    assert FilterEngine.evaluate(meta, {"author": {"$eq": {"name": "Alice", "role": "admin"}}})
