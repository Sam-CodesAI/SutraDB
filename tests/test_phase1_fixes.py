"""Regression tests for Phase 1 fixes (zero-copy NumPy, filter parity, stats, CORS)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sutradb import Collection, Document, NativeCollection
from sutradb.filters import FilterEngine
from sutradb.server import create_app


def test_zero_copy_numpy_insert_and_native_json_metadata():
    col = NativeCollection("np_zero_copy", 4, "cosine")
    vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    docs = [
        {
            "id": "n1",
            "text": "rust zero copy numpy",
            "metadata": {
                "tags": ["rust", "simd"],
                "score": 99.5,
                "count": 42,
                "active": True,
                "extra": None,
                "nested": {"tier": "gold"},
            },
        },
        {
            "id": "n2",
            "text": "python vector search",
            "metadata": {"tags": ["python"], "count": 10, "active": False},
        },
        {
            "id": "n3",
            "text": "other document",
            "metadata": {"tags": ["go"], "count": 5},
        },
    ]

    assert col.insert(docs, vecs) == 3
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    res = col.query(vector=q, filter={"nested.tier": "gold", "active": True}, top_k=5)
    assert len(res) == 1
    assert res[0]["id"] == "n1"
    assert res[0]["metadata"]["count"] == 42
    assert res[0]["metadata"]["active"] is True
    assert res[0]["metadata"]["extra"] is None

    # Non-contiguous strided slice must also work safely
    big = np.arange(24, dtype=np.float32).reshape(6, 4)
    strided = big[::2]  # non-C-contiguous view (3, 4)
    assert not strided.flags["C_CONTIGUOUS"]
    assert col.insert(
        [{"id": f"s{i}", "text": "strided", "metadata": {}} for i in range(3)],
        strided,
    ) == 6


def test_python_and_rust_filter_array_parity():
    docs_meta = [
        {"id": "d0", "tags": ["rust", "db"]},
        {"id": "d1", "tags": ["python", "ml"]},
        {"id": "d2", "tags": "rust"},
        {"id": "d3", "tags": []},
    ]

    py_col = Collection("py_arr_parity", dimension=2, metric="cosine")
    py_col.insert([
        Document(id=m["id"], vector=[1.0, 0.0], metadata=m) for m in docs_meta
    ])

    nat_col = NativeCollection("nat_arr_parity", 2, "cosine")
    nat_col.insert(
        [{"id": m["id"], "text": "", "metadata": m} for m in docs_meta],
        [[1.0, 0.0] for _ in docs_meta],
    )

    for flt in (
        {"tags": "rust"},
        {"tags": {"$eq": "rust"}},
        {"tags": {"$ne": "rust"}},
        {"tags": {"$in": ["rust", "go"]}},
        {"tags": {"$nin": ["rust"]}},
    ):
        py_ids = sorted(r.id for r in py_col.query(vector=[1.0, 0.0], filter=flt, top_k=10))
        nat_ids = sorted(r["id"] for r in nat_col.query(vector=[1.0, 0.0], filter=flt, top_k=10))
        assert py_ids == nat_ids, f"Filter parity mismatch for {flt}: py={py_ids}, nat={nat_ids}"


def test_collection_stats_includes_enable_wal(tmp_path):
    c_no_wal = Collection("no_wal", dimension=2)
    assert c_no_wal.stats()["enable_wal"] is False

    c_wal = Collection("with_wal", dimension=2, storage_path=tmp_path / "with_wal", enable_wal=True)
    assert c_wal.stats()["enable_wal"] is True
    c_wal.wal.close()


def test_cors_never_pairs_wildcard_with_credentials(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    with TestClient(app) as client:
        resp = client.options(
            "/collections",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
        )
        acao = resp.headers.get("access-control-allow-origin", "")
        acac = resp.headers.get("access-control-allow-credentials", "").lower()
        assert not (acao == "*" and acac == "true")


def test_python_server_rbac_metrics_and_proxy_header_forwarding(tmp_path, monkeypatch):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    # 1. Standalone FastAPI RBAC + Prometheus histogram verification
    rbac = json.dumps({
        "ro-token": {"role": "read", "collections": ["allowed_*"]},
    })
    monkeypatch.setenv("SUTRA_API_KEY", "adm-token")
    monkeypatch.setenv("SUTRA_RBAC_CONFIG", rbac)
    monkeypatch.delenv("SUTRA_BACKEND_URL", raising=False)

    app = create_app(data_dir=str(tmp_path / "py_auth_data"))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 401
        assert client.get("/collections", headers={"Authorization": "Bearer wrong"}).status_code == 401

        r_create = client.post(
            "/collections",
            json={"name": "allowed_docs", "dimension": 2, "metric": "cosine"},
            headers={"Authorization": "Bearer adm-token"},
        )
        assert r_create.status_code == 201
        client.post(
            "/collections",
            json={"name": "secret_docs", "dimension": 2, "metric": "cosine"},
            headers={"X-API-Key": "adm-token"},
        )
        client.post(
            "/collections/allowed_docs/insert",
            json={"documents": [{"id": "p1", "vector": [1.0, 0.0], "text": "hello"}]},
            headers={"Authorization": "Bearer adm-token"},
        )

        assert client.get("/collections", headers={"X-API-Key": "ro-token"}).json()["collections"][0]["name"] == "allowed_docs"
        assert client.post(
            "/collections/allowed_docs/query",
            json={"vector": [1.0, 0.0], "top_k": 1, "hybrid": False},
            headers={"Authorization": "Bearer ro-token"},
        ).status_code == 200
        assert client.post(
            "/collections/allowed_docs/insert",
            json={"documents": [{"id": "p2", "vector": [0.0, 1.0]}]},
            headers={"Authorization": "Bearer ro-token"},
        ).status_code == 403
        assert client.post(
            "/collections/secret_docs/query",
            json={"vector": [1.0, 0.0], "top_k": 1},
            headers={"X-API-Key": "ro-token"},
        ).status_code == 403

        metrics_res = client.get("/metrics", headers={"Authorization": "Bearer ro-token"})
        assert metrics_res.status_code == 200
        assert 'sutradb_query_duration_seconds_bucket{collection="allowed_docs",mode="flat",le="0.5"}' in metrics_res.text
        assert "sutradb_ingest_documents_total 1" in metrics_res.text
        assert "sutradb_errors_total " in metrics_res.text

    # 2. SUTRA_BACKEND_URL proxy forwards Authorization and X-API-Key headers
    seen_headers = []

    class RecordingHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen_headers.append({
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "x_api_key": self.headers.get("X-API-Key"),
            })
            payload = b'{"collections":[]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), RecordingHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        monkeypatch.delenv("SUTRA_API_KEY", raising=False)
        monkeypatch.delenv("SUTRA_RBAC_CONFIG", raising=False)
        monkeypatch.setenv("SUTRA_BACKEND_URL", f"http://127.0.0.1:{srv.server_address[1]}")
        proxy_app = create_app(data_dir=str(tmp_path / "proxy_unused"))
        with TestClient(proxy_app) as pclient:
            res = pclient.get(
                "/collections",
                headers={"Authorization": "Bearer forwarded-tok", "X-API-Key": "alt-key"},
            )
            assert res.status_code == 200
        assert seen_headers == [{
            "path": "/collections",
            "authorization": "Bearer forwarded-tok",
            "x_api_key": "alt-key",
        }]
    finally:
        srv.shutdown()
        srv.server_close()
