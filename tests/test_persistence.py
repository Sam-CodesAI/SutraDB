"""
Tests for persistence, binary serialization, and Write-Ahead Log (WAL).
"""

from pathlib import Path
import numpy as np
import pytest
from sutradb.core import SutraDB, Collection, Document
from sutradb.storage import BinaryStorage, WriteAheadLog


def test_binary_storage_roundtrip(tmp_path: Path):
    sutra_file = tmp_path / "test_snapshot.sutra"
    vectors = np.random.randn(50, 16).astype(np.float32)
    documents = [
        {"id": f"doc_{i}", "text": f"Sample document text {i}", "metadata": {"idx": i}}
        for i in range(50)
    ]

    bytes_written = BinaryStorage.write(sutra_file, vectors, documents, metric="cosine")
    assert bytes_written > 0
    assert sutra_file.exists()

    loaded_vectors, loaded_docs, metric = BinaryStorage.read(sutra_file)
    assert np.allclose(vectors, loaded_vectors)
    assert len(loaded_docs) == 50
    assert loaded_docs[0]["id"] == "doc_0"
    assert metric == "cosine"


def test_collection_save_and_load(tmp_path: Path):
    save_file = tmp_path / "my_col.sutra"
    col = Collection(name="my_col", dimension=8, metric="cosine", storage_path=save_file)

    for i in range(10):
        col.insert(Document(
            id=f"item_{i}",
            vector=np.random.randn(8).astype(np.float32),
            text=f"Item number {i}",
            metadata={"num": i}
        ))

    col.save()
    assert save_file.exists()

    # Load into a fresh collection
    col2 = Collection(name="my_col", dimension=8, storage_path=save_file)
    col2.load()

    assert col2.count() == 10
    doc = col2.get("item_5")
    assert doc is not None
    assert doc.metadata["num"] == 5


def test_wal_durability(tmp_path: Path):
    db_dir = tmp_path / "wal_db"
    db = SutraDB(persist_directory=db_dir)
    col = db.create_collection("audit_logs", dimension=4, enable_wal=True)

    col.insert([
        Document(id="wal_1", vector=[1, 0, 0, 0], text="First crash test"),
        Document(id="wal_2", vector=[0, 1, 0, 0], text="Second crash test"),
    ])

    # Simulate sudden crash by initializing new SutraDB without calling save()
    db_restarted = SutraDB(persist_directory=db_dir)
    col_recovered = db_restarted.get_collection("audit_logs")

    assert col_recovered.count() == 2
    assert col_recovered.get("wal_1") is not None
    assert col_recovered.get("wal_2") is not None


def test_snapshot_plus_wal_reload(tmp_path: Path):
    db_dir = tmp_path / "combined_db"
    db = SutraDB(persist_directory=db_dir)
    col = db.create_collection("products", dimension=16, enable_wal=True)

    # 1. Insert 5 items and take clean snapshot
    base_docs = [
        Document(id=f"base_{i}", vector=np.random.randn(16).astype(np.float32), text=f"Base item {i}")
        for i in range(5)
    ]
    col.insert(base_docs)
    col.save()

    # 2. Insert 2 more items logged only to WAL (no save)
    wal_docs = [
        Document(id=f"wal_{i}", vector=np.random.randn(16).astype(np.float32), text=f"WAL item {i}")
        for i in range(2)
    ]
    col.insert(wal_docs)

    # 3. Restart DB and reload collection
    db_reloaded = SutraDB(persist_directory=db_dir)
    col_reloaded = db_reloaded.get_collection("products")

    # Should have all 7 items and correct dimension 16
    assert col_reloaded.dimension == 16
    assert col_reloaded.count() == 7
    assert col_reloaded.get("base_0") is not None
    assert col_reloaded.get("wal_1") is not None
