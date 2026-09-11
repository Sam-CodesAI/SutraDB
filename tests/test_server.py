"""
Tests for SutraDB embedded HTTP micro-server.
"""

import json
import threading
import time
from urllib.request import Request, urlopen
import pytest
from sutradb.server import run_server, HTTPServer, SutraHTTPHandler
from sutradb.core import SutraDB


@pytest.fixture(scope="module")
def server_url(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("server_db")
    db = SutraDB(persist_directory=data_dir)
    SutraHTTPHandler.db = db

    # Pick an ephemeral port
    port = 8912
    httpd = HTTPServer(("127.0.0.1", port), SutraHTTPHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    time.sleep(0.1)
    url = f"http://127.0.0.1:{port}"
    yield url
    httpd.shutdown()
    httpd.server_close()


def test_server_health_and_lifecycle(server_url):
    # 1. Health check
    req = Request(f"{server_url}/health")
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "healthy"
        assert data["version"] == "2.0.0"

    # 2. Create collection
    create_body = json.dumps({"name": "articles", "dimension": 3, "metric": "cosine"}).encode("utf-8")
    req = Request(f"{server_url}/collections", data=create_body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req) as resp:
        assert resp.status == 201
        res = json.loads(resp.read().decode("utf-8"))
        assert res["stats"]["name"] == "articles"

    # 3. Insert documents
    docs = [
        {"id": "art_1", "vector": [1.0, 0.0, 0.0], "text": "Deep Learning with PyTorch", "metadata": {"lang": "python"}},
        {"id": "art_2", "vector": [0.0, 1.0, 0.0], "text": "Kernel development with Rust", "metadata": {"lang": "rust"}}
    ]
    insert_body = json.dumps({"documents": docs}).encode("utf-8")
    req = Request(f"{server_url}/collections/articles/insert", data=insert_body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["inserted"] == 2

    # 4. Query collection
    query_body = json.dumps({
        "vector": [1.0, 0.0, 0.0],
        "text": "PyTorch",
        "filter": {"lang": "python"},
        "top_k": 2
    }).encode("utf-8")
    req = Request(f"{server_url}/collections/articles/query", data=query_body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["count"] == 1
        assert res["results"][0]["id"] == "art_1"

    # 5. Get Stats
    req = Request(f"{server_url}/collections/articles/stats")
    with urlopen(req) as resp:
        assert resp.status == 200
        stats = json.loads(resp.read().decode("utf-8"))
        assert stats["count"] == 2
