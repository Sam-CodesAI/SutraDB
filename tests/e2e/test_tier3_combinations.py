"""
Tier 3 E2E Pairwise Cross-Feature Combination Tests (10 tests).
Verifies interactions across F1-F8: Quantization + Fusion + Filters, HNSW + Delete + Compact + Save/Load,
WAL + Multi-Modal + Update, IVF + Multilingual BM25 + RRF, Concurrency + Snapshot, Cluster + Weights,
REST + /compact + /ui + /metrics, and SUTRA_MODE=all bridge + REST + Save + Restart + Drop.
"""

from __future__ import annotations

import math
from pathlib import Path
import threading
from typing import Any, Callable, List

import httpx
import numpy as np
import pytest

from sutradb import (
    Analyzer,
    AnalyzerConfig,
    Collection,
    DistributedCluster,
    Document,
    NativeCollection,
)


def test_c1_sq8_v4_negative_vectors_cosine_save_load_cross_engine(
    tmp_path: Path,
    native_collection_factory: Callable[..., Any],
) -> None:
    """SQ8 V4 + negative normalized vectors + Cosine metric + cross-engine Rust->Python reload."""
    snap = str(tmp_path / "c1_sq8_v4.sutra")
    nat = native_collection_factory(name="c1_sq8", dimension=4, metric="cosine", storage_filename="c1_sq8_v4.sutra")
    vecs = [
        [-0.70710677, 0.0, 0.70710677, 0.0],
        [0.0, -0.70710677, 0.0, 0.70710677],
        [0.5, 0.5, -0.5, -0.5],
    ]
    nat.insert(
        [
            {"id": "v0", "text": "neg zero", "metadata": {"grp": "a"}},
            {"id": "v1", "text": "neg one", "metadata": {"grp": "b"}},
            {"id": "v2", "text": "mixed", "metadata": {"grp": "a"}},
        ],
        vecs,
    )
    nat.enable_sq8()
    nat.save()
    assert Path(snap).read_bytes()[:8] == b"SUTRA\x04\x00\x00"

    py_col = Collection(name="c1_sq8", dimension=4, metric="cosine")
    assert py_col.load(snap) == 3
    hits = py_col.query(vector=vecs[0], filter={"grp": "a"}, top_k=2, hybrid=False)
    assert hits[0].id == "v0"
    assert hits[0].score > 0.95


def test_c2_bq_pq_linear_fusion_and_metadata_filter(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """BQ/PQ quantization + alpha=0.5 linear score fusion + compound metadata filter."""
    rng = np.random.default_rng(311)
    dim = 16
    vecs = rng.normal(size=(45, dim)).astype(np.float32).tolist()

    py_col = py_collection_factory(name="c2_py", dimension=dim, metric="cosine")
    py_col.insert([
        Document(
            id=f"d_{i}",
            vector=vecs[i],
            text="neural vector index" if i % 3 == 0 else "archive item",
            metadata={"score": i, "env": "prod" if i % 2 == 0 else "dev"},
        )
        for i in range(45)
    ])
    py_col.train_pq(m=4)
    py_col.enable_pq()

    res = py_col.query(
        vector=vecs[6],
        text="neural vector index",
        filter={"score": {"$gte": 6}, "env": "prod"},
        alpha=0.5,
        top_k=5,
        hybrid=True,
    )
    assert len(res) > 0
    assert all(r.metadata["env"] == "prod" and r.metadata["score"] >= 6 for r in res)
    assert all(math.isfinite(r.score) for r in res)

    nat_col = native_collection_factory(name="c2_nat", dimension=dim, metric="cosine")
    nat_col.insert(
        [
            {
                "id": f"d_{i}",
                "text": "neural vector index" if i % 3 == 0 else "archive item",
                "metadata": {"score": i, "env": "prod" if i % 2 == 0 else "dev"},
            }
            for i in range(45)
        ],
        vecs,
    )
    nat_col.enable_bq()
    nat_res = nat_col.query(
        vector=vecs[6],
        text="neural vector index",
        filter={"score": {"$gte": 6}, "env": "prod"},
        alpha=0.5,
        top_k=5,
        hybrid=True,
    )
    assert len(nat_res) > 0
    assert all(r["metadata"]["env"] == "prod" and r["metadata"]["score"] >= 6 for r in nat_res)
    assert all(math.isfinite(r["score"]) for r in nat_res)


def test_c3_hnsw_filter_delete_compact_save_load(
    tmp_path: Path,
    native_collection_factory: Callable[..., Any],
) -> None:
    """HNSW + metadata filter + delete + compact + save/load preserves graph & recall."""
    rng = np.random.default_rng(312)
    vecs = rng.normal(size=(60, 8)).astype(np.float32)
    vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).tolist()

    snap = str(tmp_path / "c3_hnsw.sutra")
    col = native_collection_factory(name="c3_hnsw", dimension=8, metric="cosine", storage_filename="c3_hnsw.sutra")
    col.insert(
        [{"id": f"h_{i}", "text": f"doc {i}", "metadata": {"keep": i % 3 == 0}} for i in range(60)],
        vecs,
    )
    col.build_hnsw(m=16, ef_construction=64, ef_search=32)
    col.delete(["h_0", "h_3", "h_6"])
    col.compact()
    assert col.stats()["has_hnsw"] is True
    col.save()

    reloaded = NativeCollection("c3_hnsw", 8, "cosine", snap, False)
    reloaded.load(snap)
    res = reloaded.query(vector=vecs[9], filter={"keep": True}, top_k=5, hybrid=False, use_ann=True)
    assert len(res) == 5
    assert res[0]["id"] == "h_9"
    assert all(r["id"] not in ("h_0", "h_3", "h_6") and r["metadata"]["keep"] is True for r in res)


def test_c4_wal_multimodal_namespace_update_metadata_recovery(tmp_path: Path) -> None:
    """WAL + namespace + sparse + colbert + update_metadata + delete recovery without save()."""
    path = tmp_path / "c4_wal.sutra"
    c1 = Collection(name="c4_wal", dimension=4, metric="cosine", storage_path=path, enable_wal=True)
    c1.insert([
        Document(
            id="keep_me",
            vector=[1.0, 0.0, 0.0, 0.0],
            text="keep doc",
            metadata={"stage": "draft"},
            namespace="ns_alpha",
            sparse_vector={10: 4.0},
            colbert_tokens=[[1.0, 0.0, 0.0, 0.0]],
        ),
        Document(
            id="drop_me",
            vector=[0.9, 0.1, 0.0, 0.0],
            text="drop doc",
            metadata={"stage": "draft"},
            namespace="ns_alpha",
        ),
    ])
    c1.update_metadata("keep_me", {"stage": "live"})
    c1.delete(["drop_me"])
    c1.wal.close()

    c2 = Collection(name="c4_wal", dimension=4, metric="cosine", storage_path=path, enable_wal=True)
    try:
        assert c2.count() == 1
        assert c2.get("drop_me") is None
        hits = c2.query(
            vector=[1.0, 0.0, 0.0, 0.0],
            sparse_vector={10: 2.0},
            colbert_tokens=[[1.0, 0.0, 0.0, 0.0]],
            filter={"stage": "live"},
            namespace="ns_alpha",
            top_k=5,
            hybrid=True,
        )
        assert [h.id for h in hits] == ["keep_me"]
    finally:
        if c2.wal:
            c2.wal.close()


def test_c5_ivf_multilingual_bm25_rrf_hybrid(py_collection_factory: Callable[..., Any]) -> None:
    """IVF index + Spanish BM25 analyzer + RRF hybrid query combination."""
    rng = np.random.default_rng(313)
    vecs = rng.normal(size=(40, 8)).astype(np.float32).tolist()
    vecs[11] = [1.0] + [0.0] * 7

    col = py_collection_factory(
        name="c5_ivf_es",
        dimension=8,
        metric="cosine",
        analyzer=Analyzer(AnalyzerConfig(language="spanish")),
    )
    col.insert([
        Document(
            id=f"es_{i}",
            vector=vecs[i],
            text="los motores vectoriales acelerados" if i == 11 else f"documento general numero {i}",
        )
        for i in range(40)
    ])
    col.build_index()
    res = col.query(
        vector=[1.0] + [0.0] * 7,
        text="motor vectorial acelerar",
        top_k=5,
        hybrid=True,
        use_ann=True,
    )
    assert res[0].id == "es_11"


def test_c6_regex_not_exists_geospatial_filter_parity(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """$regex + $not + $exists + $geo_within produce identical results in Python and Rust."""
    rows = [
        ("g1", {"service": "edge-gw-01", "deprecated": False, "loc": {"lat": 37.78, "lon": -122.41}, "owner": {"id": 1}}),
        ("g2", {"service": "edge-gw-02", "deprecated": True, "loc": {"lat": 37.79, "lon": -122.40}, "owner": {"id": 2}}),
        ("g3", {"service": "core-db-01", "deprecated": False, "loc": {"lat": 37.77, "lon": -122.42}, "owner": {"id": 3}}),
        ("g4", {"service": "edge-gw-03", "deprecated": False, "loc": {"lat": 40.71, "lon": -74.00}}),
    ]
    vecs = [[1.0, 0.0, 0.0, 0.0]] * 4
    flt = {
        "service": {"$regex": r"^edge-gw-\d+$"},
        "$not": {"deprecated": True},
        "owner": {"$exists": True},
        "loc": {"$geo_within": {"center": [37.7749, -122.4194], "radius_km": 20.0}},
    }

    py_col = py_collection_factory(name="c6_py", dimension=4)
    py_col.insert([Document(id=i, vector=vecs[idx], metadata=m) for idx, (i, m) in enumerate(rows)])
    nat_col = native_collection_factory(name="c6_nat", dimension=4)
    nat_col.insert([{"id": i, "text": "", "metadata": m} for i, m in rows], vecs)

    assert [r.id for r in py_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=flt, top_k=5)] == ["g1"]
    assert [r["id"] for r in nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter=flt, top_k=5)] == ["g1"]


def test_c7_concurrent_native_rw_and_snapshot_persistence(
    tmp_path: Path,
    native_collection_factory: Callable[..., Any],
) -> None:
    """Concurrent NativeCollection R/W + compact + save + reload into fresh instance."""
    snap = str(tmp_path / "c7_rw.sutra")
    col = native_collection_factory(name="c7_rw", dimension=4, metric="cosine", storage_filename="c7_rw.sutra")
    col.insert([{"id": "seed", "text": "seed", "metadata": {}}], [[1.0, 0.0, 0.0, 0.0]])

    barrier = threading.Barrier(4)
    errors: List[BaseException] = []

    def writer(wid: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for j in range(10):
                col.insert([{"id": f"w_{wid}_{j}", "text": "concurrent", "metadata": {"wid": wid}}], [[0.0, 1.0, 0.0, 0.0]])
        except BaseException as exc:
            errors.append(exc)

    def reader() -> None:
        try:
            barrier.wait(timeout=5.0)
            for _ in range(15):
                assert len(col.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=3)) >= 1
        except BaseException as exc:
            errors.append(exc)

    ts = [threading.Thread(target=writer, args=(0,)), threading.Thread(target=writer, args=(1,)), threading.Thread(target=reader), threading.Thread(target=reader)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=15.0)

    assert errors == []
    col.compact()
    col.save()

    reloaded = NativeCollection("c7_rw", 4, "cosine", snap, False)
    assert reloaded.load(snap) == 21


def test_c8_distributed_collection_multimodal_weights_and_filter() -> None:
    """DistributedCluster + coordinator exclusion + sparse/colbert weights flipping ranking + metadata filter."""
    cluster = DistributedCluster(node_id="cluster-coord", vnodes_per_node=32)
    dcol = cluster.create_collection("c8_dist", dimension=4, shard_count=3)
    assert "cluster-coord" not in dcol.hash_ring.get_nodes()

    dcol.insert([
        Document(
            id="sparse_winner",
            vector=[0.0, 1.0, 0.0, 0.0],
            text="doc",
            metadata={"env": "prod"},
            sparse_vector={50: 5.0},
            colbert_tokens=[[0.0, 0.0, 1.0, 0.0]],
        ),
        Document(
            id="colbert_winner",
            vector=[0.0, 1.0, 0.0, 0.0],
            text="doc",
            metadata={"env": "prod"},
            sparse_vector={11: 0.1},
            colbert_tokens=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        ),
        Document(
            id="staging_decoy",
            vector=[1.0, 0.0, 0.0, 0.0],
            text="doc",
            metadata={"env": "staging"},
            sparse_vector={50: 10.0},
            colbert_tokens=[[1.0, 0.0, 0.0, 0.0]],
        ),
    ])

    r_sparse = dcol.query(
        sparse_vector={50: 5.0, 11: 0.1},
        colbert_tokens=[[1.0, 0.0, 0.0, 0.0]],
        filter={"env": "prod"},
        sparse_weight=5.0,
        colbert_weight=0.01,
        top_k=5,
        hybrid=True,
    )
    r_colbert = dcol.query(
        sparse_vector={50: 5.0, 11: 0.1},
        colbert_tokens=[[1.0, 0.0, 0.0, 0.0]],
        filter={"env": "prod"},
        sparse_weight=0.1,
        colbert_weight=5.0,
        top_k=5,
        hybrid=True,
    )
    assert "staging_decoy" not in [r.id for r in r_sparse]
    assert "staging_decoy" not in [r.id for r in r_colbert]
    assert r_sparse[0].id == "sparse_winner"
    assert r_colbert[0].id == "colbert_winner"


def test_c9_rest_compact_ui_escaped_rendering_and_metrics(rest_client: Any) -> None:
    """REST CRUD + /compact + /ui XSS check + /metrics + /sample_vectors combination."""
    assert rest_client.post("/collections", json={"name": "c9_rest", "dimension": 4}).status_code == 201
    assert rest_client.post(
        "/collections/c9_rest/documents",
        json={
            "documents": [
                {"id": "d1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "hello <script>alert(1)</script>"},
                {"id": "d2", "vector": [0.0, 1.0, 0.0, 0.0], "text": "to delete"},
            ]
        },
    ).status_code == 200
    assert rest_client.delete("/collections/c9_rest/documents/d2").status_code == 200
    assert rest_client.post("/collections/c9_rest/compact").status_code == 200
    assert rest_client.get("/collections/c9_rest/stats").json()["count"] == 1

    samples = rest_client.get("/collections/c9_rest/sample_vectors").json()
    assert samples["count"] == 1
    assert 'sutradb_documents_count{collection="c9_rest"} 1' in rest_client.get("/metrics").text
    assert "${r.id}" not in rest_client.get("/ui").text


def test_c10_sutra_all_mode_bridge_rest_save_restart_drop(
    tmp_path: Path,
    sutradb_server_factory: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unified SUTRA_MODE=all: REST proxy -> Save -> server restart recovery -> DropCollection disk cleanup."""
    from fastapi.testclient import TestClient
    from sutradb.server import create_app

    shared_dir = tmp_path / "c10_data"
    srv1 = sutradb_server_factory(data_dir=shared_dir)
    monkeypatch.setenv("SUTRA_BACKEND_URL", srv1.bridge_url)
    app1 = create_app(data_dir=str(shared_dir))
    with TestClient(app1) as client1:
        assert client1.post("/collections", json={"name": "c10_col", "dimension": 4, "enable_wal": True}).status_code == 201
        assert client1.post(
            "/collections/c10_col/documents",
            json={"documents": [{"id": "x1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "persisted via rest"}]},
        ).status_code == 200
        assert client1.post("/collections/c10_col/save").status_code == 200
    assert (shared_dir / "c10_col.sutra").exists()
    srv1.stop()

    srv2 = sutradb_server_factory(data_dir=shared_dir)
    monkeypatch.setenv("SUTRA_BACKEND_URL", srv2.bridge_url)
    app2 = create_app(data_dir=str(shared_dir))
    with TestClient(app2) as client2:
        q = client2.post("/collections/c10_col/query", json={"vector": [1.0, 0.0, 0.0, 0.0], "top_k": 5})
        assert q.status_code == 200
        assert [r["id"] for r in q.json()["results"]] == ["x1"]
        assert client2.delete("/collections/c10_col").status_code == 200
    assert not (shared_dir / "c10_col.sutra").exists()
    assert not (shared_dir / "c10_col.wal").exists()
