"""
Comprehensive verification test suite for SutraDB v3.0 Industrial Features:
1. Multi-tenant namespace isolation
2. Idempotent upsert & in-place metadata patch
3. SQ8 scalar quantization & HNSW graph indexing
4. Native Rust C-extension acceleration (SIMD dot, cosine, euclidean, SQ8)
5. NativeCollection direct Python bindings
6. Prometheus /metrics observability
7. Official SutraClient SDK operations
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sutradb import SutraDB, Document, SearchResult, NativeCollection
import sutradb._sutradb as _native
from sutradb.server import create_app


def test_rust_simd_and_sq8_extension():
    """Verifies that native Rust C-extension functions are active and mathematically exact."""
    a = [1.0, 2.0, 3.0, 4.0]
    b = [2.0, 3.0, 4.0, 5.0]

    # SIMD dot product
    dot = _native.simd_dot(a, b)
    assert abs(dot - 40.0) < 1e-5

    # SIMD cosine similarity
    cos = _native.simd_cosine(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    expected_cos = 40.0 / (norm_a * norm_b)
    assert abs(cos - expected_cos) < 1e-5

    # SIMD euclidean similarity
    euc = _native.simd_euclidean(a, b)
    # dist = sqrt(1+1+1+1) = 2.0 -> sim = 1 / (1 + 2.0) = 1/3
    assert abs(euc - (1.0 / 3.0)) < 1e-5

    # SQ8 scalar quantization
    v = [0.1, 0.5, 0.9, -0.4, 0.2]
    q_bytes, min_val, scale = _native.sq8_quantize(v)
    assert len(q_bytes) == len(v)
    assert min_val <= -0.4
    assert scale > 0.0

    # Tokenizer
    tokens = _native.tokenize_text("Deploying containerized microservices to Kubernetes!")
    assert "kubernetes" in tokens
    assert "microservices" in tokens


def test_native_collection_python_binding(tmp_path):
    """Verifies direct Python usage of Rust NativeCollection."""
    store_file = str(tmp_path / "native.sutra")
    col = NativeCollection("native_test", 4, "cosine", store_file, True)

    docs = [
        {
            "id": "doc_1",
            "namespace": "tenant_alpha",
            "text": "Kubernetes cluster configuration",
            "metadata": {"env": "prod", "tier": "gold"}
        },
        {
            "id": "doc_2",
            "namespace": "tenant_beta",
            "text": "PostgreSQL database query optimization",
            "metadata": {"env": "prod", "tier": "silver"}
        }
    ]
    vecs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ]

    count = col.insert(docs, vecs)
    assert count == 2
    assert col.count() == 2

    # Namespace isolation: tenant_alpha must NEVER see tenant_beta docs
    res_alpha = col.query(
        vector=[0.0, 1.0, 0.0, 0.0],
        namespace="tenant_alpha",
        top_k=5
    )
    assert len(res_alpha) == 1
    assert res_alpha[0]["id"] == "doc_1"

    # In-place metadata update
    updated = col.update_metadata("doc_1", {"env": "staging", "tier": "platinum"})
    assert updated is True

    # Build HNSW index
    col.build_hnsw(m=16, ef_construction=64, ef_search=32)

    # Enable SQ8
    col.enable_sq8()

    st = col.stats()
    assert st["count"] == 2
    assert st["quantized_sq8"] is True
    assert st["has_hnsw"] is True


def test_core_collection_multi_tenancy_and_upsert(tmp_path):
    """Verifies Python Collection's namespace partitioning and idempotent upsert."""
    db = SutraDB(persist_directory=str(tmp_path))
    col = db.create_collection("mt_test", dimension=3)

    # 1. Idempotent upsert
    col.upsert([
        Document(id="u1", vector=[1.0, 0.0, 0.0], text="First", namespace="acme_corp"),
        Document(id="u2", vector=[0.0, 1.0, 0.0], text="Second", namespace="globex")
    ])
    assert col.count() == 2

    # Upsert existing doc with updated text
    col.upsert([
        Document(id="u1", vector=[1.0, 0.0, 0.0], text="First Updated", namespace="acme_corp")
    ])
    assert col.count() == 2
    assert col.get("u1").text == "First Updated"

    # 2. Namespace filtering in query
    results_acme = col.query(vector=[1.0, 0.0, 0.0], namespace="acme_corp")
    assert len(results_acme) == 1
    assert results_acme[0].id == "u1"
    assert results_acme[0].namespace == "acme_corp"

    results_globex = col.query(vector=[1.0, 0.0, 0.0], namespace="globex")
    assert len(results_globex) == 1
    assert results_globex[0].id == "u2"

    # 3. In-place metadata update
    col.update_metadata("u1", {"status": "archived", "views": 100})
    assert col.get("u1").metadata["status"] == "archived"

    # 4. HNSW and SQ8 flags in stats
    col.enable_sq8()
    col.build_hnsw(m=16)
    stats = col.stats()
    assert stats["quantized_sq8"] is True
    assert stats["has_hnsw"] is True


def test_server_v3_industrial_endpoints(tmp_path):
    """Verifies FastAPI REST server endpoints for metrics, upsert, metadata patch, and HNSW."""
    db = SutraDB(persist_directory=str(tmp_path))
    col = db.create_collection("rest_test", dimension=2)
    col.insert([
        Document(id="d1", vector=[1.0, 0.0], text="Item one", namespace="team_a", metadata={"tag": "ai"}),
        Document(id="d2", vector=[0.0, 1.0], text="Item two", namespace="team_b", metadata={"tag": "web"})
    ])

    app = create_app(data_dir=str(tmp_path), db=db)
    client = TestClient(app)

    # 1. Prometheus /metrics endpoint
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "text/plain" in metrics_resp.headers["content-type"]
    assert "sutradb_collections_total 1" in metrics_resp.text
    assert 'sutradb_documents_count{collection="rest_test"} 2' in metrics_resp.text

    # 2. Upsert endpoint
    upsert_resp = client.post(
        "/collections/rest_test/upsert",
        json={"documents": [{"id": "d3", "vector": [0.5, 0.5], "text": "Item three"}]}
    )
    assert upsert_resp.status_code == 200
    assert upsert_resp.json()["total_count"] == 3

    # 3. Patch metadata endpoint
    patch_resp = client.patch(
        "/collections/rest_test/documents/d1/metadata",
        json={"metadata": {"tag": "ai_updated", "version": 2}}
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["updated"] is True

    # 4. HNSW and SQ8 endpoints
    hnsw_resp = client.post("/collections/rest_test/hnsw", json={"m": 16, "ef_construction": 64, "ef_search": 32})
    assert hnsw_resp.status_code == 200

    sq8_resp = client.post("/collections/rest_test/sq8")
    assert sq8_resp.status_code == 200
    assert sq8_resp.json()["stats"]["quantized_sq8"] is True

    # 5. Query with namespace filter
    query_resp = client.post(
        "/collections/rest_test/query",
        json={"vector": [1.0, 0.0], "namespace": "team_a", "top_k": 5}
    )
    assert query_resp.status_code == 200
    results = query_resp.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == "d1"
