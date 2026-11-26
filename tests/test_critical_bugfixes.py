"""
Unit and integration regression tests for the Top Critical Bugs:
1. PyO3 multi-threading concurrent queries (RuntimeError: Already borrowed elimination).
2. Python engine WAL recovery update_metadata (KeyError: 'vector' elimination).
3. Cross-engine binary storage format compatibility (HEADER_MAGIC_V2 and HEADER_MAGIC_V3).
"""

import concurrent.futures
from pathlib import Path
import numpy as np
import pytest

from sutradb import Collection, Document, NativeCollection
from sutradb.storage import BinaryStorage, HEADER_MAGIC_V2, HEADER_MAGIC_V3, HEADER_FORMAT, HEADER_SIZE
import struct
import zlib
import json


def test_concurrent_threads_query_pyo3_no_already_borrowed(tmp_path):
    """
    Verifies that multiple Python threads can concurrently execute .query() on a PyNativeCollection
    without throwing `RuntimeError: Already borrowed`.
    """
    db_path = str(tmp_path / "concurrent_native.sutra")
    col = NativeCollection("concurrent_test", 4, "cosine", db_path, False)

    docs = [
        {"id": f"doc_{i}", "text": f"document number {i} discussing vector search and concurrency", "metadata": {"idx": i}}
        for i in range(20)
    ]
    vectors = [
        [float(i % 4 == j) for j in range(4)]
        for i in range(20)
    ]
    col.insert(docs, vectors)

    query_vec = [1.0, 0.0, 0.0, 0.0]
    errors = []

    def run_worker(thread_id: int):
        try:
            for _ in range(30):
                # Hybrid and pure queries
                res = col.query(vector=query_vec, text="concurrency vector search", top_k=5)
                assert len(res) > 0
        except Exception as e:
            errors.append((thread_id, e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_worker, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert len(errors) == 0, f"Concurrent queries triggered borrow or execution errors: {errors}"


def test_wal_replay_update_metadata_no_vector_keyerror(tmp_path):
    """
    Verifies that replaying WAL with `op == 'update'` (from update_metadata) does NOT throw
    `KeyError: 'vector'` and preserves the document's vector and text while updating metadata.
    """
    col_path = tmp_path / "wal_col"
    wal_path = tmp_path / "wal_col.wal"

    # 1. Initialize collection with WAL enabled and insert document
    col1 = Collection(name="wal_test", dimension=4, metric="cosine", storage_path=col_path, enable_wal=True)
    original_vec = [0.2, 0.4, 0.6, 0.8]
    col1.insert(Document(id="doc_meta_test", vector=original_vec, text="persisted document text", metadata={"version": 1}))

    # 2. Call update_metadata
    updated = col1.update_metadata("doc_meta_test", {"version": 2, "approved": True})
    assert updated is True
    # Do NOT call col1.save() - simulate sudden shutdown where only WAL contains the update

    # 3. Reload collection from disk + WAL
    col2 = Collection(name="wal_test", dimension=4, metric="cosine", storage_path=col_path, enable_wal=True)

    assert len(col2.documents) == 1
    recovered_doc = col2.documents[0]
    assert recovered_doc.id == "doc_meta_test"
    assert recovered_doc.text == "persisted document text"
    assert recovered_doc.metadata == {"version": 2, "approved": True}
    assert np.allclose(col2.vectors[0], col1.vectors[0]), "Original vector must not be overwritten or corrupted"


def test_binary_storage_v2_and_v3_magic_bytes(tmp_path):
    """
    Verifies that BinaryStorage.read() accepts both HEADER_MAGIC_V2 and HEADER_MAGIC_V3,
    and rejects invalid magic bytes with a clear ValueError.
    """
    dim = 4
    num_docs = 2
    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    docs = [{"id": "d1", "text": "text1"}, {"id": "d2", "text": "text2"}]
    meta_bytes = json.dumps(docs).encode("utf-8")
    crc = zlib.crc32(vectors.tobytes())
    crc = zlib.crc32(meta_bytes, crc)
    metric_padded = b"cosine".ljust(16, b"\x00")

    # 1. Test V2 magic bytes
    v2_file = tmp_path / "v2.sutra"
    v2_header = struct.pack(HEADER_FORMAT, HEADER_MAGIC_V2, num_docs, dim, metric_padded, len(meta_bytes), crc, b"\x00" * 20)
    with open(v2_file, "wb") as f:
        f.write(v2_header)
        f.write(vectors.tobytes())
        f.write(meta_bytes)

    read_vecs, read_docs, read_metric = BinaryStorage.read(v2_file)
    assert np.allclose(read_vecs, vectors)
    assert len(read_docs) == 2
    assert read_metric == "cosine"

    # 2. Test V3 magic bytes
    v3_file = tmp_path / "v3.sutra"
    v3_header = struct.pack(HEADER_FORMAT, HEADER_MAGIC_V3, num_docs, dim, metric_padded, len(meta_bytes), crc, b"\x00" * 20)
    with open(v3_file, "wb") as f:
        f.write(v3_header)
        f.write(vectors.tobytes())
        f.write(meta_bytes)

    read_vecs3, read_docs3, read_metric3 = BinaryStorage.read(v3_file)
    assert np.allclose(read_vecs3, vectors)
    assert len(read_docs3) == 2
    assert read_metric3 == "cosine"

    # 3. Test Invalid magic bytes raises ValueError
    bad_file = tmp_path / "bad.sutra"
    bad_header = struct.pack(HEADER_FORMAT, b"INVALID\x00", num_docs, dim, metric_padded, len(meta_bytes), crc, b"\x00" * 20)
    with open(bad_file, "wb") as f:
        f.write(bad_header)
        f.write(vectors.tobytes())
        f.write(meta_bytes)

    with pytest.raises(ValueError, match="Invalid magic bytes"):
        BinaryStorage.read(bad_file)


def test_cross_engine_rust_snapshot_python_read(tmp_path):
    """
    Verifies that a binary snapshot created by the Rust NativeCollection (using HEADER_MAGIC_V3)
    can be read seamlessly by Python's BinaryStorage.read() and Python Collection.load().
    """
    snapshot_path = str(tmp_path / "rust_snapshot.sutra")
    rust_col = NativeCollection("cross_engine", 4, "euclidean", snapshot_path, False)

    docs = [
        {"id": "doc_rust_1", "text": "written by rust engine", "metadata": {"lang": "rust", "rating": 5}},
        {"id": "doc_rust_2", "text": "cross compatibility verification", "metadata": {"lang": "python", "rating": 10}},
    ]
    vectors = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ]
    rust_col.insert(docs, vectors)
    rust_col.save()

    # Read via Python BinaryStorage
    py_vectors, py_docs, py_metric = BinaryStorage.read(snapshot_path)
    assert py_metric == "euclidean"
    assert len(py_docs) == 2
    assert py_docs[0]["id"] == "doc_rust_1"
    assert py_docs[1]["id"] == "doc_rust_2"
    assert np.allclose(py_vectors, np.array(vectors, dtype=np.float32))

    # Read via Python Collection.load()
    py_col = Collection(name="cross_engine", dimension=4, metric="euclidean")
    py_col.load(snapshot_path)
    assert py_col.metric.value == "euclidean"
    assert len(py_col.documents) == 2
    assert py_col.documents[0].id == "doc_rust_1"
    assert py_col.documents[0].metadata["lang"] == "rust"


def test_cross_engine_python_snapshot_rust_read(tmp_path):
    """
    Verifies bidirectional compatibility:
    A binary snapshot saved by Python's Collection.save() (V2 magic bytes)
    can be loaded and queried seamlessly by Rust NativeCollection.load().
    """
    snapshot_path = str(tmp_path / "python_snapshot.sutra")
    py_col = Collection(name="py_origin", dimension=4, metric="cosine", storage_path=snapshot_path)

    docs = [
        Document(id="doc_py_1", vector=[0.1, 0.2, 0.3, 0.4], text="python saved document", metadata={"framework": "pytorch"}),
        Document(id="doc_py_2", vector=[0.5, 0.6, 0.7, 0.8], text="another document", metadata={"framework": "jax"}),
    ]
    py_col.insert(docs)
    py_col.save()

    # Load in Rust NativeCollection
    rust_col = NativeCollection("py_origin", 4, "cosine", snapshot_path, False)
    loaded_count = rust_col.load(snapshot_path)
    assert loaded_count == 2
    assert rust_col.count() == 2

    # Query in Rust engine
    res = rust_col.query(vector=[0.1, 0.2, 0.3, 0.4], text="python saved document", top_k=2)
    assert len(res) > 0
    assert res[0]["id"] == "doc_py_1"
    assert res[0]["metadata"]["framework"] == "pytorch"


def test_rust_sq8_snapshot_save_reload_and_python_read(tmp_path):
    """
    Verifies that enabling SQ8 on a Rust NativeCollection saves with quant_mode == 1,
    reloads into Rust with SQ8 preserved, and Python BinaryStorage.read() decodes SQ8 vectors accurately.
    """
    snapshot_path = str(tmp_path / "rust_sq8.sutra")
    rust_col = NativeCollection("sq8_col", 4, "cosine", snapshot_path, False)

    docs = [
        {"id": "sq8_1", "text": "quantized text 1", "metadata": {"tier": "free"}},
        {"id": "sq8_2", "text": "quantized text 2", "metadata": {"tier": "pro"}},
    ]
    vectors = [
        [0.0, 0.5, 0.8, 1.0],
        [1.0, 0.8, 0.5, 0.0],
    ]
    rust_col.insert(docs, vectors)
    rust_col.enable_sq8()
    rust_col.save()

    # Verify reload in Rust
    reloaded_rust = NativeCollection("sq8_col", 4, "euclidean", snapshot_path, False)
    reloaded_rust.load(snapshot_path)
    assert reloaded_rust.count() == 2
    stats = reloaded_rust.stats()
    assert stats["metric"] == "cosine", "Metric must be cosine, not corrupted by SQ8 quant_mode"
    assert stats["quantized_sq8"] is True, "SQ8 mode must remain enabled after reload"

    # Verify read in Python BinaryStorage (Rust normalizes Cosine vectors before SQ8 quantization)
    py_vecs, py_docs, py_metric = BinaryStorage.read(snapshot_path)
    assert py_metric == "cosine"
    assert len(py_docs) == 2
    assert py_docs[0]["id"] == "sq8_1"
    expected_v0 = np.array(vectors[0], dtype=np.float32)
    expected_v0 = expected_v0 / np.linalg.norm(expected_v0)
    assert np.allclose(py_vecs[0], expected_v0, atol=0.05)


def test_sq8_negative_coordinate_roundtrip_cross_engine(tmp_path):
    """
    Verifies that negative-coordinate vectors survive SQ8 V4 save/load roundtrips
    across both Rust NativeCollection and Python Collection / BinaryStorage.
    """
    neg_vec = [-0.70710677, 0.0, 0.70710677, 0.0]

    # 1. Rust -> Disk (V4 SQ8) -> Rust & Python
    rust_path = str(tmp_path / "rust_neg_sq8.sutra")
    rust_col = NativeCollection("neg_rust", 4, "cosine", rust_path, False)
    rust_col.insert([{"id": "neg_1", "text": "negative coords"}], [neg_vec])
    rust_col.enable_sq8()
    rust_col.save()

    reloaded_rust = NativeCollection("neg_rust", 4, "cosine", rust_path, False)
    reloaded_rust.load(rust_path)
    rust_res = reloaded_rust.query(vector=neg_vec, top_k=1)
    assert len(rust_res) == 1
    assert rust_res[0]["score"] > 0.99

    py_vecs, _, _ = BinaryStorage.read(rust_path)
    assert np.allclose(py_vecs[0], np.array(neg_vec, dtype=np.float32), atol=0.02)

    py_from_rust = Collection(name="neg_rust_py", dimension=4, metric="cosine")
    py_from_rust.load(rust_path)
    assert py_from_rust.stats()["quantized_sq8"] is True
    py_res = py_from_rust.query(vector=neg_vec, top_k=1)
    assert len(py_res) == 1
    assert py_res[0].score > 0.99

    # 2. Python -> Disk (V4 SQ8) -> Python & Rust
    py_path = str(tmp_path / "py_neg_sq8.sutra")
    py_col = Collection(name="neg_py", dimension=4, metric="cosine", storage_path=py_path)
    py_col.insert([Document(id="neg_1", vector=neg_vec, text="negative coords")])
    py_col.enable_sq8()
    py_col.save()

    py_reloaded = Collection(name="neg_py", dimension=4, metric="cosine", storage_path=py_path)
    py_reloaded.load()
    assert py_reloaded.stats()["quantized_sq8"] is True
    assert np.allclose(py_reloaded.vectors[0], np.array(neg_vec, dtype=np.float32), atol=0.02)
    assert py_reloaded.query(vector=neg_vec, top_k=1)[0].score > 0.99

    rust_from_py = NativeCollection("neg_py_rust", 4, "cosine", py_path, False)
    rust_from_py.load(py_path)
    assert rust_from_py.stats()["quantized_sq8"] is True
    assert rust_from_py.query(vector=neg_vec, top_k=1)[0]["score"] > 0.99


def test_python_filter_engine_type_safety():
    """
    Verifies that heterogeneous comparisons (e.g. string vs int in $gt, $gte, $lt, $lte)
    do not crash with TypeError, but safely evaluate to False.
    """
    from sutradb.filters import FilterEngine

    # String vs Int
    assert not FilterEngine.match_value("high", {"$gt": 80})
    assert not FilterEngine.match_value("high", {"$gte": 80})
    assert not FilterEngine.match_value("low", {"$lt": 50})
    assert not FilterEngine.match_value("low", {"$lte": 50})

    # Int vs String
    assert not FilterEngine.match_value(80, {"$gt": "high"})
    assert not FilterEngine.match_value(50, {"$lt": "low"})

    # None vs Int
    assert not FilterEngine.match_value(None, {"$gt": 10})
    assert not FilterEngine.match_value(None, {"$lt": 10})

    # In Collection.query with mixed metadata types
    col = Collection(name="type_safe_filter", dimension=2, metric="cosine")
    col.insert([
        Document(id="d1", vector=[1.0, 0.0], metadata={"score": "N/A"}),
        Document(id="d2", vector=[0.0, 1.0], metadata={"score": 95}),
    ])

    # Should not raise TypeError: '>' not supported between instances of 'str' and 'int'
    results = col.query(vector=[0.0, 1.0], filter={"score": {"$gt": 80}})
    assert len(results) == 1
    assert results[0].id == "d2"


def test_python_sq8_search_implementation():
    """
    Verifies that enabling SQ8 in Python Collection creates SQ8 vectors,
    incremental ingestion updates SQ8 vectors, and .query() executes asymmetric SQ8 scoring.
    """
    # 1. Cosine Collection with SQ8
    col = Collection(name="py_sq8_cosine", dimension=4, metric="cosine")
    col.insert([
        Document(id="doc_1", vector=[1.0, 0.0, 0.0, 0.0], text="neural search"),
        Document(id="doc_2", vector=[0.0, 1.0, 0.0, 0.0], text="database index"),
    ])

    col.enable_sq8()
    assert col._sq8_enabled is True
    assert len(col._sq8_vectors) == 2

    # Query with SQ8 active
    res = col.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(res) == 2
    assert res[0].id == "doc_1"
    assert res[0].score > 0.95

    # Incremental insertion with SQ8 active
    col.insert([
        Document(id="doc_3", vector=[0.7071, 0.7071, 0.0, 0.0], text="hybrid query"),
    ])
    assert len(col._sq8_vectors) == 3

    res_inc = col.query(vector=[0.7071, 0.7071, 0.0, 0.0], top_k=1)
    assert len(res_inc) == 1
    assert res_inc[0].id == "doc_3"

    # 2. Euclidean Collection with SQ8
    col_euc = Collection(name="py_sq8_euc", dimension=4, metric="euclidean")
    col_euc.insert([
        Document(id="e1", vector=[0.1, 0.2, 0.3, 0.4]),
        Document(id="e2", vector=[0.9, 0.8, 0.7, 0.6]),
    ])
    col_euc.enable_sq8()
    res_euc = col_euc.query(vector=[0.1, 0.2, 0.3, 0.4], top_k=2)
    assert len(res_euc) == 2
    assert res_euc[0].id == "e1"
    # Euclidean similarity is 1.0 / (1.0 + dist)
    assert 0.0 < res_euc[0].score <= 1.0
    assert res_euc[0].score > res_euc[1].score


def test_python_bq_dimension_mismatch_check():
    """
    Verifies that BQVector.hamming_distance raises ValueError on dimension mismatch.
    """
    from sutradb.quantization import BQVector

    bq1 = BQVector.quantize([1.0, -1.0, 0.5])
    bq2 = BQVector.quantize([1.0, -1.0, 0.5, -0.5])

    with pytest.raises(ValueError, match="Dimension mismatch in BQVector hamming_distance"):
        bq1.hamming_distance(bq2)


def test_python_wal_flush_and_sync(tmp_path):
    """
    Verifies that WriteAheadLog flush() and fdatasync operate without errors.
    """
    from sutradb.storage import WriteAheadLog

    wal_file = tmp_path / "test.wal"
    wal = WriteAheadLog(wal_file)
    wal.append({"op": "insert", "doc": {"id": "1", "data": "val"}})
    wal.flush()

    replayed = wal.replay()
    assert len(replayed) == 1
    assert replayed[0]["doc"]["id"] == "1"
    wal.close()


def test_python_sq8_cosine_unnormalized_query():
    """
    Verifies that unnormalized query vectors under Cosine metric with SQ8
    are normalized properly and do not produce similarity scores > 1.0.
    """
    col = Collection(name="py_sq8_cosine_unnorm", dimension=4, metric="cosine")
    col.insert([
        Document(id="doc_1", vector=[1.0, 0.0, 0.0, 0.0], text="unit vector"),
    ])
    col.enable_sq8()
    # Query with non-unit length vector [5.0, 0.0, 0.0, 0.0]
    res = col.query(vector=[5.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(res) == 1
    # Cosine similarity must remain <= 1.0
    assert 0.95 <= res[0].score <= 1.01


def test_python_bm25_analyzer_preservation_on_update_and_load(tmp_path):
    """
    Verifies that Python Collection preserves its configured Analyzer
    across in-place document updates, save(), and load().
    """
    from sutradb.nlp import Analyzer, AnalyzerConfig

    custom_cfg = AnalyzerConfig(
        language="spanish",
        stemmer=True,
        stopwords=False,
        split_identifiers=True,
        cjk_bigrams=True,
    )
    custom_analyzer = Analyzer(custom_cfg)

    col_path = str(tmp_path / "analyzer_py.sutra")
    col = Collection(
        name="py_analyzer_test",
        dimension=4,
        metric="cosine",
        storage_path=col_path,
        analyzer=custom_analyzer,
    )

    col.insert([
        Document(id="d1", vector=[0.1, 0.2, 0.3, 0.4], text="documento inicial"),
        Document(id="d2", vector=[0.5, 0.6, 0.7, 0.8], text="segundo documento"),
    ])

    # 1. Update an existing document (triggers BM25 re-indexing)
    col.insert([
        Document(id="d1", vector=[0.1, 0.2, 0.3, 0.4], text="documento actualizado"),
    ])

    # Verify analyzer configuration remains intact
    assert col.bm25_index.analyzer.config.language == "spanish"
    assert col.bm25_index.analyzer.config.cjk_bigrams is True
    assert col.bm25_index.analyzer.config.split_identifiers is True

    # 2. Save and reload
    col.save()
    col.load()

    assert col.bm25_index.analyzer.config.language == "spanish"
    assert col.bm25_index.analyzer.config.cjk_bigrams is True
    assert col.bm25_index.analyzer.config.split_identifiers is True



