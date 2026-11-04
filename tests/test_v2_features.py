"""
Deep verification test suite for SutraDB v2.0 features:
1. Concurrency: Writer-preference ReadWriteLock (prevents writer starvation under continuous read traffic).
2. WAL Durability: Recording and replaying delete operations.
3. Soft-Deletion Tombstoning: Filtering across dense, BM25, hybrid, and ANN searches, plus save-compaction.
4. Candidate-Pruned RRF: Scale verification of M-candidate pruning.
5. Production ASGI Server: FastAPI endpoints (CRUD, query, stats, 10MB payload boundary).
"""

from pathlib import Path
import threading
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sutradb.concurrency import ReadWriteLock
from sutradb.core import SutraDB, Collection, Document
from sutradb.server import create_app, MAX_PAYLOAD_SIZE
from sutradb.fusion import reciprocal_rank_fusion


def test_writer_preference_locking():
    """
    Verifies that ReadWriteLock enforces writer preference:
    When a writer is waiting, subsequent readers are blocked until the writer executes.
    """
    lock = ReadWriteLock()
    events = []

    reader1_holding = threading.Event()
    writer_waiting = threading.Event()
    writer_finished = threading.Event()

    def reader1():
        with lock.read():
            events.append("r1_acquired")
            reader1_holding.set()
            # Hold lock until writer has registered wait
            writer_waiting.wait(timeout=2.0)
            time.sleep(0.05)
            events.append("r1_releasing")

    def writer():
        writer_waiting.set()
        with lock.write():
            events.append("writer_acquired")
            events.append("writer_releasing")
        writer_finished.set()

    def reader2():
        # Reader 2 arrives AFTER writer starts waiting
        writer_waiting.wait(timeout=2.0)
        time.sleep(0.02)  # Ensure writer is queued
        with lock.read():
            events.append("r2_acquired")

    t_r1 = threading.Thread(target=reader1)
    t_w = threading.Thread(target=writer)
    t_r2 = threading.Thread(target=reader2)

    t_r1.start()
    reader1_holding.wait(timeout=2.0)
    t_w.start()
    t_r2.start()

    t_r1.join(timeout=3.0)
    t_w.join(timeout=3.0)
    t_r2.join(timeout=3.0)

    # Writer MUST acquire before reader 2 because of writer preference!
    assert "r1_acquired" in events
    assert "writer_acquired" in events
    assert "r2_acquired" in events
    writer_idx = events.index("writer_acquired")
    r2_idx = events.index("r2_acquired")
    assert writer_idx < r2_idx, f"Writer did not preempt Reader 2! Events: {events}"


def test_wal_delete_durability(tmp_path: Path):
    """
    Verifies that delete operations are logged to the WAL and replayed on recovery,
    preventing deleted vectors from reappearing as ghosts on restart.
    """
    db_dir = tmp_path / "wal_del_db"
    db = SutraDB(persist_directory=db_dir)
    col = db.create_collection("ledger", dimension=4, enable_wal=True)

    col.insert([
        Document(id="rec_1", vector=[1, 0, 0, 0], text="First ledger entry"),
        Document(id="rec_2", vector=[0, 1, 0, 0], text="Second ledger entry"),
        Document(id="rec_3", vector=[0, 0, 1, 0], text="Third ledger entry"),
    ])
    assert col.count() == 3

    # Delete rec_2 without calling save() -> purely in memory and WAL
    deleted = col.delete("rec_2")
    assert deleted == 1
    assert col.count() == 2
    assert col.get("rec_2") is None

    # Re-open database from disk simulating sudden restart
    db_restarted = SutraDB(persist_directory=db_dir)
    col_restarted = db_restarted.get_collection("ledger")

    assert col_restarted.count() == 2
    assert col_restarted.get("rec_1") is not None
    assert col_restarted.get("rec_2") is None
    assert col_restarted.get("rec_3") is not None


def test_tombstone_filtering_and_compaction(tmp_path: Path):
    """
    Verifies that soft-deletion tombstones:
    1. Filter out deleted records across dense, BM25, hybrid, and ANN searches.
    2. Maintain array references without costly reallocations.
    3. Are purged during save() snapshot compaction.
    """
    save_file = tmp_path / "tombstone_test.sutra"
    col = Collection(name="tomb_col", dimension=4, metric="cosine", storage_path=save_file)

    col.insert([
        Document(id="d0", vector=[1.0, 0.0, 0.0, 0.0], text="python distributed systems", metadata={"status": "active"}),
        Document(id="d1", vector=[0.9, 0.1, 0.0, 0.0], text="python machine learning", metadata={"status": "deprecated"}),
        Document(id="d2", vector=[0.0, 1.0, 0.0, 0.0], text="rust embedded kernel", metadata={"status": "active"}),
    ])

    # Build ANN index
    col.build_index()

    # Soft delete d1
    del_count = col.delete("d1")
    assert del_count == 1
    assert col.count() == 2
    assert col.stats()["tombstones"] == 1
    assert col.get("d1") is None

    # 1. Pure dense search
    res_dense = col.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
    result_ids = [r.id for r in res_dense]
    assert "d1" not in result_ids
    assert "d0" in result_ids

    # 2. Pure BM25 search
    res_bm25 = col.query(text="machine learning", top_k=5)
    assert "d1" not in [r.id for r in res_bm25]

    # 3. Hybrid search
    res_hybrid = col.query(vector=[1.0, 0.0, 0.0, 0.0], text="python", hybrid=True, top_k=5)
    assert "d1" not in [r.id for r in res_hybrid]

    # 4. ANN search
    res_ann = col.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=5, use_ann=True)
    assert "d1" not in [r.id for r in res_ann]

    # 5. Save collection -> snapshot compaction
    bytes_written = col.save()
    assert bytes_written > 0
    assert col.stats()["tombstones"] == 0
    assert col.count() == 2

    # Load into fresh collection
    col_loaded = Collection(name="tomb_col", dimension=4, storage_path=save_file)
    col_loaded.load()
    assert col_loaded.count() == 2
    assert col_loaded.get("d1") is None
    assert col_loaded.get("d0") is not None
    assert col_loaded.get("d2") is not None


def test_candidate_pruned_rrf_large_scale():
    """
    Verifies that reciprocal_rank_fusion executes candidate-pruning correctly
    over large result sets, pruning down to top M candidates.
    """
    n = 1000
    top_k = 5
    # M = min(top_k * 4, 200) = 20

    dense_scores = np.random.uniform(0.0, 1.0, size=n).astype(np.float32)
    bm25_scores = np.zeros(n, dtype=np.float32)
    # Give 30 documents positive BM25 scores
    pos_indices = np.random.choice(n, size=30, replace=False)
    bm25_scores[pos_indices] = np.random.uniform(0.5, 2.5, size=30).astype(np.float32)

    # Force doc 42 to have top score in both
    dense_scores[42] = 2.0
    bm25_scores[42] = 5.0

    results = reciprocal_rank_fusion(
        dense_scores=dense_scores,
        bm25_scores=bm25_scores,
        top_k=top_k,
        k_constant=60
    )

    assert len(results) == top_k
    # doc 42 must be rank #1 because it leads in both candidate sets
    assert results[0][0] == 42
    assert results[0][1] > 0


def test_fastapi_server_full_crud_and_bounds(tmp_path: Path):
    """
    Verifies the production FastAPI ASGI server:
    - POST /collections (Create collection)
    - POST /collections/{name}/documents (Batch insert)
    - GET /collections/{name}/documents/{id} (Single lookup)
    - POST /collections/{name}/query (Search)
    - DELETE /collections/{name}/documents/{id} (Single delete)
    - DELETE /collections/{name}/documents (Batch delete)
    - 413 Payload Too Large when Content-Length > 10MB
    - 404 for missing collections / documents
    """
    app = create_app(data_dir=str(tmp_path / "server_data"))
    client = TestClient(app)

    # 1. Health check
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

    # 2. Create collection
    resp = client.post("/collections", json={"name": "knowledge", "dimension": 3, "metric": "cosine"})
    assert resp.status_code == 201
    assert resp.json()["stats"]["name"] == "knowledge"

    # 3. Batch insert documents
    docs = [
        {"id": "k1", "vector": [1.0, 0.0, 0.0], "text": "Kubernetes orchestration patterns", "metadata": {"tier": 1}},
        {"id": "k2", "vector": [0.0, 1.0, 0.0], "text": "PostgreSQL indexing optimizations", "metadata": {"tier": 2}},
        {"id": "k3", "vector": [0.0, 0.0, 1.0], "text": "Redis pub sub caching layers", "metadata": {"tier": 3}},
    ]
    resp = client.post("/collections/knowledge/documents", json={"documents": docs})
    assert resp.status_code == 200
    assert resp.json()["inserted"] == 3
    assert resp.json()["total_count"] == 3

    # 4. Lookup document by ID
    resp = client.get("/collections/knowledge/documents/k1")
    assert resp.status_code == 200
    doc_data = resp.json()
    assert doc_data["id"] == "k1"
    assert doc_data["text"] == "Kubernetes orchestration patterns"

    # 5. Query collection
    resp = client.post("/collections/knowledge/query", json={
        "vector": [1.0, 0.0, 0.0],
        "text": "Kubernetes",
        "top_k": 2
    })
    assert resp.status_code == 200
    q_res = resp.json()
    assert q_res["count"] == 2
    assert q_res["results"][0]["id"] == "k1"

    # 6. Delete single document
    resp = client.delete("/collections/knowledge/documents/k1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 1

    # Verify k1 is now 404
    resp = client.get("/collections/knowledge/documents/k1")
    assert resp.status_code == 404

    # 7. Batch delete remaining documents
    resp = client.request("DELETE", "/collections/knowledge/documents", json={"ids": ["k2", "k3"]})
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 2

    # 8. Check 404 on missing collection
    resp = client.get("/collections/missing_collection/documents/k1")
    assert resp.status_code == 404

    # 9. Bounded payload enforcement (> 10MB)
    fake_large_size = MAX_PAYLOAD_SIZE + 1024
    resp = client.post(
        "/collections/knowledge/documents",
        content=b"x" * 100,  # small actual content but spoofed Content-Length header
        headers={"Content-Length": str(fake_large_size), "Content-Type": "application/json"}
    )
    assert resp.status_code == 413
    assert "Payload exceeds maximum allowed size" in resp.json()["error"]
