"""
Tier 4 E2E Real-World End-to-End Application Scenarios (6 scenarios).
Exercises multi-domain production workloads:
1. Multi-Tenant Enterprise RAG with WAL Crash Recovery & Compaction
2. Geo-Aware Hybrid E-Commerce Catalog with HNSW, Filters & SQ8 V4 Persistence
3. High-Concurrency Native Ingest & Search Service under Thread Barrier
4. Distributed Multi-Shard Multi-Modal Search Cluster
5. Full Web Stack: Ephemeral Uvicorn + SutraClient + /ui + /metrics + Playground
6. Unified SUTRA_MODE=all Server Lifecycle, Restart Recovery, Cleanup & Healthcheck
"""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Callable, Dict, List

from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest

from sutradb import (
    Collection,
    DistributedCluster,
    Document,
    NativeCollection,
    SutraClient,
    SutraDB,
)
from sutradb.server import create_app

from tests.e2e.conftest import REPO_ROOT, get_free_port


def test_scenario_1_multitenant_enterprise_rag_crash_recovery(tmp_path: Path) -> None:
    """
    Scenario 1: Multi-tenant enterprise RAG service.
    Ingests multi-modal documents across two tenants, promotes draft docs via
    update_metadata, deletes retracted docs, simulates an ungraceful shutdown
    (no save()), reopens SutraDB, verifies strict tenant isolation and metadata
    index parity, then compacts and persists a V4 SQ8 snapshot.
    """
    db_dir = tmp_path / "rag_store"
    db1 = SutraDB(persist_directory=str(db_dir))
    col1 = db1.create_collection("enterprise_kb", dimension=8, metric="cosine", enable_wal=True)

    col1.insert([
        Document(
            id="acme_1",
            vector=[1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
            text="Acme zero-trust kubernetes ingress deployment runbook",
            metadata={"status": "draft", "dept": "security", "rev": 1},
            namespace="tenant_acme",
            sparse_vector={10: 3.2, 25: 1.5},
            colbert_tokens=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ),
        Document(
            id="acme_retracted",
            vector=[0.99, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            text="Acme deprecated kubernetes runbook",
            metadata={"status": "published", "dept": "security", "rev": 0},
            namespace="tenant_acme",
            sparse_vector={10: 5.0},
        ),
        Document(
            id="globex_1",
            vector=[1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
            text="Globex zero-trust kubernetes ingress secret",
            metadata={"status": "published", "dept": "security", "rev": 2},
            namespace="tenant_globex",
            sparse_vector={10: 4.0},
        ),
    ])

    assert col1.update_metadata("acme_1", {"status": "published", "dept": "security", "rev": 2}) is True
    assert col1.delete(["acme_retracted"]) == 1
    if col1.wal:
        col1.wal.close()

    # Simulate crash recovery without prior save()
    db2 = SutraDB(persist_directory=str(db_dir))
    try:
        col2 = db2.get_collection("enterprise_kb")
        assert col2.count() == 2
        assert col2.get("acme_retracted") is None

        acme_hits = col2.query(
            vector=[1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
            text="kubernetes ingress runbook",
            sparse_vector={10: 2.0, 25: 1.0},
            colbert_tokens=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            filter={"status": "published", "rev": {"$gte": 2}},
            namespace="tenant_acme",
            top_k=5,
            hybrid=True,
        )
        assert [h.id for h in acme_hits] == ["acme_1"]
        assert acme_hits[0].namespace == "tenant_acme"

        col2.compact()
        col2.enable_sq8()
        bytes_written = col2.save()
        assert bytes_written > 0
        assert (db_dir / "enterprise_kb.sutra").read_bytes()[:8] == b"SUTRA\x04\x00\x00"
    finally:
        for c in db2.collections.values():
            if c.wal:
                c.wal.close()


def test_scenario_2_geo_aware_hybrid_ecommerce_catalog(
    tmp_path: Path,
    native_collection_factory: Callable[..., Any],
) -> None:
    """
    Scenario 2: Geo-aware hybrid e-commerce inventory search.
    Uses NativeCollection with Cosine + HNSW + $geo_within + $in + $gte/$lte + $not +
    $regex + SQ8 V4 persistence across save/load.
    """
    rng = np.random.default_rng(402)
    snap_path = str(tmp_path / "catalog_sq8.sutra")
    col = native_collection_factory(
        name="catalog",
        dimension=16,
        metric="cosine",
        storage_filename="catalog_sq8.sutra",
    )

    docs: List[Dict[str, Any]] = []
    vecs: List[List[float]] = []

    target_vec = [-0.6, 0.5, -0.4, 0.3] + [0.1] * 12
    for i in range(120):
        if i == 0:
            v = target_vec
            meta = {
                "sku": "PRO-CAM-2026",
                "category": "cameras",
                "price": 899.0,
                "discontinued": False,
                "warehouse": {"lat": 37.78, "lon": -122.41},
            }
            text = "mirrorless cinema camera 4k optical stabilization"
        elif i == 1:
            v = target_vec
            meta = {
                "sku": "PRO-CAM-OLD",
                "category": "cameras",
                "price": 850.0,
                "discontinued": True,
                "warehouse": {"lat": 37.78, "lon": -122.41},
            }
            text = "mirrorless cinema camera legacy"
        elif i == 2:
            v = target_vec
            meta = {
                "sku": "PRO-CAM-NYC",
                "category": "cameras",
                "price": 899.0,
                "discontinued": False,
                "warehouse": {"lat": 40.71, "lon": -74.00},
            }
            text = "mirrorless cinema camera nyc stock"
        else:
            raw = rng.normal(size=16).astype(np.float32)
            v = (raw / np.linalg.norm(raw)).tolist()
            meta = {
                "sku": f"ITEM-{i:04d}",
                "category": "accessories" if i % 2 == 0 else "audio",
                "price": float(50 + i * 5),
                "discontinued": False,
                "warehouse": {"lat": 37.77, "lon": -122.42},
            }
            text = f"studio accessory {i}"
        docs.append({"id": f"prod_{i}", "text": text, "metadata": meta})
        vecs.append(v)

    col.insert(docs, vecs)
    col.build_hnsw(m=16, ef_construction=64, ef_search=32)
    col.enable_sq8()
    col.save()

    reloaded = NativeCollection("catalog", 16, "cosine", snap_path, False)
    assert reloaded.load(snap_path) == 120

    flt = {
        "sku": {"$regex": r"^PRO-CAM-"},
        "category": {"$in": ["cameras", "lenses"]},
        "price": {"$gte": 500.0, "$lte": 1200.0},
        "$not": {"discontinued": True},
        "warehouse": {"$geo_within": {"center": [37.7749, -122.4194], "radius_km": 25.0}},
    }
    hits = reloaded.query(vector=target_vec, text="cinema camera", filter=flt, top_k=5, hybrid=True, use_ann=True)
    assert [h["id"] for h in hits] == ["prod_0"]


def test_scenario_3_high_concurrency_native_ingest_and_search(
    native_collection_factory: Callable[..., Any],
) -> None:
    """
    Scenario 3: High-concurrency NativeCollection ingest, metadata patching, deletion,
    and concurrent hybrid query service synchronized via Barrier, ending with compact().
    """
    col = native_collection_factory(name="live_service", dimension=8, metric="cosine")
    seed_vecs = np.eye(8, dtype=np.float32).tolist()
    col.insert(
        [{"id": f"seed_{i}", "text": f"baseline service record {i}", "metadata": {"active": True, "gen": 0}} for i in range(8)],
        seed_vecs,
    )

    barrier = threading.Barrier(8)
    errors: List[BaseException] = []

    def writer_fn(wid: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for step in range(15):
                doc_id = f"stream_{wid}_{step}"
                vec = [float(j == (wid % 8)) for j in range(8)]
                col.insert([{"id": doc_id, "text": f"stream event {wid} step {step}", "metadata": {"active": True, "wid": wid}}], [vec])
                col.update_metadata(doc_id, {"active": True, "wid": wid, "verified": True})
                if step % 5 == 0:
                    col.delete([doc_id])
        except BaseException as exc:
            errors.append(exc)

    def reader_fn() -> None:
        try:
            barrier.wait(timeout=5.0)
            for step in range(25):
                q_vec = [float(j == (step % 8)) for j in range(8)]
                res = col.query(vector=q_vec, text="stream event", filter={"active": True}, top_k=5, hybrid=True)
                assert isinstance(res, list)
                assert all(math.isfinite(r["score"]) for r in res)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer_fn, args=(i,)) for i in range(4)] + [
        threading.Thread(target=reader_fn) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)
        assert not t.is_alive()

    assert errors == []
    # 8 seed + 4 * (15 inserted - 3 deleted) = 56 active records
    assert col.count() == 56
    col.compact()
    st = col.stats()
    assert st["tombstones"] == 0
    assert st["total_records"] == 56


def test_scenario_4_distributed_multi_shard_multimodal_search() -> None:
    """
    Scenario 4: Distributed 4-shard cluster with coordinator exclusion,
    preference list verification, multi-modal ingestion, and weight sensitivity.
    """
    cluster = DistributedCluster(node_id="coordinator-main", vnodes_per_node=64)
    dcol = cluster.create_collection("global_search", dimension=8, metric="cosine", shard_count=4)

    assert "coordinator-main" not in dcol.hash_ring.get_nodes()
    prefs = dcol.hash_ring.get_preference_list("tenant-enterprise-01", 3)
    assert len(prefs) == 3 and len(set(prefs)) == 3 and "coordinator-main" not in prefs

    docs: List[Document] = []
    for i in range(32):
        vec = [1.0 if i == 3 else 0.0, 1.0 if i != 3 else 0.0] + [0.0] * 6
        docs.append(
            Document(
                id=f"global_{i}",
                vector=vec,
                text="distributed raft consensus" if i == 7 else f"shard document {i}",
                metadata={"region": "us-east" if i % 2 == 0 else "eu-west"},
                namespace="prod",
                sparse_vector={88: 6.0} if i == 13 else {1: 0.2},
                colbert_tokens=[[1.0, 0.0] + [0.0] * 6] if i == 21 else [[0.0, 1.0] + [0.0] * 6],
            )
        )
    assert dcol.insert(docs) == 32
    assert dcol.count() == 32

    sp_hits = dcol.query(
        sparse_vector={88: 3.0},
        colbert_tokens=[[1.0, 0.0] + [0.0] * 6],
        namespace="prod",
        sparse_weight=10.0,
        colbert_weight=0.01,
        top_k=5,
        hybrid=True,
    )
    assert sp_hits[0].id == "global_13"

    cb_hits = dcol.query(
        sparse_vector={88: 3.0},
        colbert_tokens=[[1.0, 0.0] + [0.0] * 6],
        namespace="prod",
        sparse_weight=0.01,
        colbert_weight=10.0,
        top_k=5,
        hybrid=True,
    )
    assert cb_hits[0].id == "global_21"


def test_scenario_5_full_rest_ui_playground_and_client_lifecycle(
    uvicorn_server_factory: Callable[..., Any],
    playground_server: str,
) -> None:
    """
    Scenario 5: Full web stack lifecycle combining live uvicorn REST server,
    SutraClient SDK, /compact, /metrics, /ui verification, and Playground search.
    """
    base_url, _ = uvicorn_server_factory()

    with SutraClient(base_url=base_url, timeout=10.0) as sdk:
        assert sdk.health()["status"] == "healthy"
        sdk.create_collection("web_kb", dimension=8, metric="cosine", enable_wal=True)
        sdk.insert(
            "web_kb",
            [
                {"id": "w1", "vector": [1.0] + [0.0] * 7, "text": "semantic vector search", "metadata": {"tier": "pro"}},
                {"id": "w2", "vector": [0.0, 1.0] + [0.0] * 6, "text": "to be deleted", "metadata": {"tier": "free"}},
            ],
        )
        sdk.delete("web_kb", ["w2"])
        assert httpx.post(f"{base_url}/collections/web_kb/compact", timeout=5.0).status_code == 200
        sdk.enable_sq8("web_kb")
        sdk.build_hnsw("web_kb", m=8, ef_construction=32, ef_search=16)

        hits = sdk.query("web_kb", vector=[1.0] + [0.0] * 7, text="semantic", top_k=5, use_ann=True)
        assert [h["id"] for h in hits] == ["w1"]

    ui_html = httpx.get(f"{base_url}/ui", timeout=5.0).text
    assert "5.1.0" in ui_html and "${r.id}" not in ui_html
    metrics_txt = httpx.get(f"{base_url}/metrics", timeout=5.0).text
    assert 'sutradb_documents_count{collection="web_kb"} 1' in metrics_txt

    pg_stats = httpx.get(f"{playground_server}/api/stats", timeout=5.0).json()
    assert pg_stats["total_documents"] > 0
    pg_search = httpx.post(
        f"{playground_server}/api/search",
        json={"text": "HNSW", "mode": "hybrid", "top_k": 3},
        timeout=5.0,
    )
    assert pg_search.status_code == 200
    assert len(pg_search.json()["results"]) > 0


def test_scenario_6_unified_sutra_all_mode_server_lifecycle_and_healthcheck(
    tmp_path: Path,
    sutradb_server_factory: Callable[..., Any],
    uvicorn_server_factory: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Scenario 6: Full SUTRA_MODE=all production daemon lifecycle.
    Starts sutradb-server + proxied REST uvicorn, verifies sutra-healthcheck.py passes,
    writes via REST & bridge, saves snapshot (wal_records -> 0), restarts sutradb-server
    on same data_dir to verify persistence, drops collection to verify disk cleanup,
    and confirms healthcheck fails after shutdown.
    """
    shared_dir = tmp_path / "all_mode_data"
    srv1 = sutradb_server_factory(data_dir=shared_dir)
    rest_url, _ = uvicorn_server_factory(backend_url=srv1.bridge_url)
    rest_port = int(rest_url.rsplit(":", 1)[1])

    hc_env = {
        "PATH": "/usr/bin:/bin",
        "SUTRA_MODE": "all",
        "SUTRA_GRPC_PORT": str(srv1.grpc_port),
        "SUTRA_REST_PORT": str(rest_port),
    }
    hc_up = subprocess.run(
        [sys.executable, str(REPO_ROOT / "sutra-healthcheck.py")],
        env=hc_env,
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    assert hc_up.returncode == 0, f"Healthcheck failed while servers up: {hc_up.stderr}"

    # Write via REST proxy, read via gRPC bridge
    assert httpx.post(
        f"{rest_url}/collections",
        json={"name": "prod_col", "dimension": 4, "metric": "cosine", "enable_wal": True},
        timeout=5.0,
    ).status_code == 201
    assert httpx.post(
        f"{rest_url}/collections/prod_col/documents",
        json={"documents": [{"id": "p1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "production doc"}]},
        timeout=5.0,
    ).status_code == 200

    st_before = httpx.get(f"{srv1.bridge_url}/collections/prod_col/stats", timeout=5.0).json()
    assert st_before["count"] == 1
    assert st_before["wal_records"] >= 1

    assert httpx.post(f"{rest_url}/collections/prod_col/save", timeout=5.0).status_code == 200
    st_after = httpx.get(f"{srv1.bridge_url}/collections/prod_col/stats", timeout=5.0).json()
    assert st_after["wal_records"] == 0
    assert (shared_dir / "prod_col.sutra").exists()

    srv1.stop()

    # Healthcheck must fail once sutradb-server is stopped
    hc_down = subprocess.run(
        [sys.executable, str(REPO_ROOT / "sutra-healthcheck.py")],
        env=hc_env,
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    assert hc_down.returncode != 0

    # Restart sutradb-server on same data_dir and verify recovery + DropCollection disk cleanup
    srv2 = sutradb_server_factory(data_dir=shared_dir)
    monkeypatch.setenv("SUTRA_BACKEND_URL", srv2.bridge_url)
    app2 = create_app(data_dir=str(shared_dir))
    with TestClient(app2) as rest2:
        q = rest2.post("/collections/prod_col/query", json={"vector": [1.0, 0.0, 0.0, 0.0], "top_k": 5})
        assert q.status_code == 200
        assert [r["id"] for r in q.json()["results"]] == ["p1"]
        assert rest2.delete("/collections/prod_col").status_code == 200

    assert not (shared_dir / "prod_col.sutra").exists()
    assert not (shared_dir / "prod_col.wal").exists()
