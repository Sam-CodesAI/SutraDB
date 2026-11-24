"""
Unit tests for SutraDB Embedded Web UI dashboard endpoints.
"""

import pytest
import shutil
from fastapi.testclient import TestClient

from sutradb import __version__
from sutradb.server import create_app
from sutradb.core import SutraDB, Document


@pytest.fixture
def test_client(tmp_path):
    data_dir = tmp_path / "ui_test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db = SutraDB(persist_directory=str(data_dir))
    col = db.create_collection("ui_demo", dimension=3)
    col.insert([
        Document(id="doc_1", vector=[1.0, 0.0, 0.0], text="Sample vector doc", metadata={"cat": "test"}),
        Document(id="doc_2", vector=[0.0, 1.0, 0.0], text="Another vector doc", metadata={"cat": "demo"}),
    ])
    app = create_app(data_dir=str(data_dir), db=db)
    client = TestClient(app)
    yield client
    shutil.rmtree(str(data_dir), ignore_errors=True)


def test_ui_endpoints_serve_html(test_client):
    # GET /ui
    res_ui = test_client.get("/ui")
    assert res_ui.status_code == 200
    assert "text/html" in res_ui.headers["content-type"]
    assert "SutraDB Console" in res_ui.text
    assert f"v{__version__}" in res_ui.text

    # GET /dashboard
    res_dash = test_client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "text/html" in res_dash.headers["content-type"]
    assert "SutraDB Console" in res_dash.text


def test_ui_stats_endpoint(test_client):
    res = test_client.get("/ui/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["version"] == __version__
    assert data["total_collections"] >= 1
    assert data["total_documents"] >= 2
    assert len(data["collections"]) >= 1


def test_sample_vectors_projection_endpoint(test_client):
    res = test_client.get("/collections/ui_demo/sample_vectors")
    assert res.status_code == 200
    data = res.json()
    assert data["collection"] == "ui_demo"
    assert data["count"] == 2
    assert len(data["vectors"]) == 2
    assert data["vectors"][0]["id"] == "doc_1"
    assert "vector" in data["vectors"][0]
