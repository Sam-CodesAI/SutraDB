"""
Tests for nested metadata filtering with dot-notation path resolution and array operators.
"""

import pytest
from sutradb.filters import FilterEngine, resolve_nested_path


def test_resolve_nested_path():
    data = {
        "user": {
            "profile": {
                "city": "Seattle",
                "zip": 98101,
            },
            "roles": ["admin", "editor"],
        },
        "flat_key": 42,
    }

    assert resolve_nested_path(data, "flat_key") == 42
    assert resolve_nested_path(data, "user.profile.city") == "Seattle"
    assert resolve_nested_path(data, "user.profile.zip") == 98101
    assert resolve_nested_path(data, "user.roles.0") == "admin"
    assert resolve_nested_path(data, "user.roles.1") == "editor"
    assert resolve_nested_path(data, "user.nonexistent") is None
    assert resolve_nested_path(data, "non.existent.path") is None


def test_nested_filter_predicates():
    meta = {
        "user": {
            "profile": {
                "city": "San Francisco",
                "score": 95.5,
            },
            "tags": ["ai", "rust", "vector"],
        },
        "status": "published",
    }

    # Dot notation exact match
    assert FilterEngine.evaluate(meta, {"user.profile.city": "San Francisco"}) is True
    assert FilterEngine.evaluate(meta, {"user.profile.city": "New York"}) is False

    # Dot notation comparison
    assert FilterEngine.evaluate(meta, {"user.profile.score": {"$gte": 90}}) is True
    assert FilterEngine.evaluate(meta, {"user.profile.score": {"$lt": 90}}) is False

    # Array $all operator
    assert FilterEngine.evaluate(meta, {"user.tags": {"$all": ["ai", "rust"]}}) is True
    assert FilterEngine.evaluate(meta, {"user.tags": {"$all": ["ai", "python"]}}) is False

    # Array $size operator
    assert FilterEngine.evaluate(meta, {"user.tags": {"$size": 3}}) is True
    assert FilterEngine.evaluate(meta, {"user.tags": {"$size": 2}}) is False

    # String $regex operator
    assert FilterEngine.evaluate(meta, {"user.profile.city": {"$regex": "^San"}}) is True
    assert FilterEngine.evaluate(meta, {"user.profile.city": {"$regex": "York$"}}) is False

    # Compound $and with nested path
    compound_filter = {
        "$and": [
            {"user.profile.city": "San Francisco"},
            {"user.profile.score": {"$gt": 90}},
            {"status": "published"},
            {"user.tags": {"$all": ["vector"]}},
        ]
    }
    assert FilterEngine.evaluate(meta, compound_filter) is True
