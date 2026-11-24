"""
Tests for Geo-Spatial Metadata Filtering ($geo_within and $geo_box) in SutraDB.
"""

import pytest
from sutradb import Collection, Document, Metric, FilterEngine
from sutradb.filters import haversine_distance


def test_haversine_distance():
    # SF Downtown (37.7749, -122.4194) to Oakland (37.8044, -122.2712) ~ 13.5 km
    dist = haversine_distance(37.7749, -122.4194, 37.8044, -122.2712)
    assert 12000.0 < dist < 15000.0


def test_geo_within_filter_evaluation():
    doc_meta_array = {"name": "Cafe SF", "loc": [37.7749, -122.4194]}
    doc_meta_obj = {"name": "Cafe SF", "loc": {"lat": 37.7749, "lon": -122.4194}}

    # Within 2km of SF center
    f_pass = {
        "loc": {
            "$geo_within": {
                "center": [37.7750, -122.4190],
                "radius_meters": 2000.0
            }
        }
    }
    assert FilterEngine.evaluate(doc_meta_array, f_pass) is True
    assert FilterEngine.evaluate(doc_meta_obj, f_pass) is True

    # Within 50 meters (should fail)
    f_fail = {
        "loc": {
            "$geo_within": {
                "center": [37.7850, -122.4190],
                "radius_meters": 50.0
            }
        }
    }
    assert FilterEngine.evaluate(doc_meta_array, f_fail) is False
    assert FilterEngine.evaluate(doc_meta_obj, f_fail) is False


def test_geo_box_filter_evaluation():
    doc_meta = {"name": "SF Downtown", "loc": [37.7792, -122.4191]}

    # Box enclosing SF
    f_box = {
        "loc": {
            "$geo_box": {
                "bottom_left": [37.70, -122.50],
                "top_right": [37.85, -122.35]
            }
        }
    }
    assert FilterEngine.evaluate(doc_meta, f_box) is True

    # Box in NYC (should fail)
    f_box_nyc = {
        "loc": {
            "$geo_box": [40.70, -74.05, 40.85, -73.90]
        }
    }
    assert FilterEngine.evaluate(doc_meta, f_box_nyc) is False


def test_collection_query_with_geospatial():
    col = Collection(name="geo_col", dimension=4, metric=Metric.COSINE)
    docs = [
        Document(
            id="sf_store",
            vector=[1.0, 0.0, 0.0, 0.0],
            text="San Francisco Coffee",
            metadata={"city": "SF", "location": [37.7749, -122.4194]}
        ),
        Document(
            id="oak_store",
            vector=[1.0, 0.0, 0.0, 0.0],
            text="Oakland Bakery",
            metadata={"city": "Oakland", "location": [37.8044, -122.2712]}
        ),
        Document(
            id="sjc_store",
            vector=[1.0, 0.0, 0.0, 0.0],
            text="San Jose Tech",
            metadata={"city": "San Jose", "location": [37.3382, -121.8863]}
        ),
    ]
    col.insert(docs)

    # Filter within 10km of SF
    filter_sf = {
        "location": {
            "$geo_within": {
                "center": [37.7750, -122.4190],
                "radius_meters": 10000.0
            }
        }
    }
    results = col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=filter_sf, top_k=5)
    assert len(results) == 1
    assert results[0].id == "sf_store"
