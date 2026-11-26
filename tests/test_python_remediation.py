"""
Comprehensive regression test suite for M2 Python Engine remediation:
C1, C2, C4, C5, C7, H3, M1, M3, M4, M5.
"""

import json
import math
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import pytest

from sutradb import Collection, Document, Metric, SutraDB
from sutradb.ann import HNSWConfig, HNSWIndex, IVFConfig, IVFIndex
from sutradb.bm25 import BM25Index
from sutradb.filters import FilterEngine, extract_lat_lon, haversine_distance
from sutradb.fusion import linear_score_fusion, reciprocal_rank_fusion
from sutradb.nlp import Analyzer, AnalyzerConfig
from sutradb.storage import (
    HEADER_FORMAT,
    HEADER_MAGIC_V3,
    HEADER_MAGIC_V4,
    BinaryStorage,
    WriteAheadLog,
)


# ---------------------------------------------------------------------------
# C1 & M1: .sutra V4 SQ8, V3 legacy fallback, header bounds & WAL bounds
# ---------------------------------------------------------------------------

def test_c1_binary_storage_rejects_invalid_quant_mode_and_truncated_payload(tmp_path: Path):
    bad_quant = tmp_path / "bad_quant.sutra"
    vec = np.array([[1.0, 0.0]], dtype=np.float32)
    meta = json.dumps([{"id": "d1"}]).encode("utf-8")
    crc = zlib.crc32(meta, zlib.crc32(vec.tobytes()))
    extra = b"\x02" + b"\x00" * 19  # unsupported quant_mode=2
    header = struct.pack(
        HEADER_FORMAT, HEADER_MAGIC_V4, 1, 2, b"cosine".ljust(16, b"\x00"), len(meta), crc, extra
    )
    bad_quant.write_bytes(header + vec.tobytes() + meta)

    with pytest.raises(ValueError, match="Unsupported quant_mode"):
        BinaryStorage.read(bad_quant)

    truncated = tmp_path / "truncated.sutra"
    crafted_header = struct.pack(
        HEADER_FORMAT, HEADER_MAGIC_V4, 100000, 128, b"cosine".ljust(16, b"\x00"), 10, 0, b"\x01" + b"\x00" * 19
    )
    truncated.write_bytes(crafted_header + b"\x00" * 16)
    with pytest.raises(ValueError, match="Truncated payload|invalid header"):
        BinaryStorage.read(truncated)


def test_c1_v3_legacy_sq8_still_readable(tmp_path: Path):
    v3_sq8 = tmp_path / "legacy_v3_sq8.sutra"
    u8_data = np.array([[0, 255]], dtype=np.uint8)
    docs = [{"id": "d1", "text": "legacy", "metadata": {}}]
    meta = json.dumps(docs).encode("utf-8")
    crc = zlib.crc32(meta, zlib.crc32(u8_data.tobytes()))
    extra = b"\x01" + b"\x00" * 19
    header = struct.pack(
        HEADER_FORMAT, HEADER_MAGIC_V3, 1, 2, b"cosine".ljust(16, b"\x00"), len(meta), crc, extra
    )
    v3_sq8.write_bytes(header + u8_data.tobytes() + meta)

    vecs, loaded_docs, metric = BinaryStorage.read(v3_sq8)
    assert metric == "cosine"
    assert loaded_docs[0]["id"] == "d1"
    assert np.allclose(vecs[0], [0.0, 1.0], atol=1e-5)


def test_m1_wal_replay_ignores_corrupt_oversized_length(tmp_path: Path):
    wal_path = tmp_path / "corrupt.wal"
    wal = WriteAheadLog(wal_path)
    wal.append({"op": "insert", "docs": [{"id": "ok", "vector": [1.0, 0.0], "text": "hi", "metadata": {}}]})
    wal.close()

    with open(wal_path, "ab") as f:
        f.write(struct.pack("<II", 0x7FFFFFFF, 0))
        f.write(b"partial")

    wal2 = WriteAheadLog(wal_path)
    entries = wal2.replay()
    wal2.close()
    assert len(entries) == 1
    assert entries[0]["docs"][0]["id"] == "ok"


# ---------------------------------------------------------------------------
# C2: WAL parity (namespace, sparse_vector, colbert_tokens) & update replay
# ---------------------------------------------------------------------------

def test_c2_wal_preserves_multimodal_fields_and_update_replay(tmp_path: Path):
    sp = tmp_path / "wal_parity.sutra"
    c1 = Collection("wal_parity", dimension=2, metric="cosine", storage_path=sp, enable_wal=True)
    c1.insert([
        Document(
            id="d1",
            vector=[1.0, 0.0],
            text="initial alpha",
            metadata={"v": 1},
            namespace="tenant_secret",
            sparse_vector={10: 2.5, "tok": 1.25},
            colbert_tokens=[[1.0, 0.0], [0.0, 1.0]],
        )
    ])
    c1.enable_sq8()
    c1.enable_bq()

    # Append a raw WAL update record mutating vector, text, namespace, sparse_vector, colbert_tokens
    c1.wal.append({
        "op": "update",
        "doc": {
            "id": "d1",
            "namespace": "tenant_v2",
            "text": "updated omega",
            "metadata": {"v": 2},
            "vector": [0.0, 3.0],  # unnormalized -> must normalize to [0, 1] under Cosine
            "sparse_vector": {"42": 4.0},
            "colbert_tokens": [[0.0, 1.0]],
        },
    })
    c1.wal.close()

    c2 = Collection("wal_parity", dimension=2, metric="cosine", storage_path=sp, enable_wal=True)
    c2.enable_sq8()
    # Replay again with quantizers active via fresh load + WAL
    c3 = Collection("wal_parity_2", dimension=2, metric="cosine", storage_path=sp, enable_wal=False)
    c3.enable_sq8()
    c3.enable_bq()
    c3.wal = WriteAheadLog(sp.with_suffix(".wal"))
    c3._replay_wal_if_present()
    c3.wal.close()

    for col in (c2, c3):
        rec = col.get("d1")
        assert rec is not None
        assert rec.namespace == "tenant_v2"
        assert rec.metadata == {"v": 2}
        assert rec.sparse_vector == {42: 4.0}
        assert rec.colbert_tokens == [[0.0, 1.0]]
        assert np.allclose(col.vectors[0], [0.0, 1.0], atol=1e-5)

        bm25_hits = col.query(text="omega", top_k=5)
        assert len(bm25_hits) == 1 and bm25_hits[0].id == "d1"
        assert col.query(text="alpha", top_k=5) == []

    # Verify SQ8/BQ quantizers updated during replay on c3
    assert np.allclose(c3._sq8_vectors[0].dequantize(), [0.0, 1.0], atol=0.02)
    res_q = c3.query(vector=[0.0, 1.0], namespace="tenant_v2", top_k=1)
    assert len(res_q) == 1 and res_q[0].score > 0.98


# ---------------------------------------------------------------------------
# C4: Duplicate IDs within a single batch & atomic validation rollback
# ---------------------------------------------------------------------------

def test_c4_duplicate_ids_and_atomic_batch_validation(tmp_path: Path):
    sp = tmp_path / "atomic_c4.sutra"
    col = Collection("atomic_c4", dimension=2, metric="cosine", storage_path=sp, enable_wal=True)

    # Same new ID twice in one batch -> last wins, no IndexError
    count = col.insert([
        Document(id="same", vector=[1.0, 0.0], text="first", metadata={"seq": 1}),
        Document(id="same", vector=[0.0, 1.0], text="second", metadata={"seq": 2}),
    ])
    assert count == 1
    assert col.count() == 1
    assert col.get("same").text == "second"
    assert col.get("same").metadata == {"seq": 2}
    assert np.allclose(col.vectors[0], [0.0, 1.0])

    # Same existing ID twice in one batch
    count2 = col.upsert([
        Document(id="same", vector=[1.0, 0.0], text="third"),
        Document(id="same", vector=[1.0, 1.0], text="fourth"),
        Document(id="other", vector=[0.0, 1.0], text="other"),
    ])
    assert count2 == 2
    assert col.get("same").text == "fourth"
    assert len(col.bm25_index.score_query("fourth")) == 2

    # Mid-batch validation error must leave RAM and WAL completely untouched
    wal_before = col.wal.replay()
    with pytest.raises(ValueError, match="dimension mismatch"):
        col.insert([
            Document(id="same", vector=[1.0, 0.0], text="corrupted"),
            Document(id="brand_new", vector=[1.0, 0.0], text="should_not_exist"),
            Document(id="bad", vector=[1.0, 0.0, 0.0], text="wrong dim"),
        ])

    assert col.count() == 2
    assert col.get("same").text == "fourth"
    assert col.get("brand_new") is None
    assert "brand_new" not in col.id_to_index
    assert len(col.wal.replay()) == len(wal_before)

    # Invalid sparse_vector & colbert_tokens rejected atomically
    with pytest.raises(ValueError, match="sparse_vector"):
        col.insert([
            Document(id="brand_new", vector=[1.0, 0.0], sparse_vector={1: float("nan")}),
        ])
    with pytest.raises(ValueError, match="colbert_tokens"):
        col.insert([
            Document(id="brand_new", vector=[1.0, 0.0], colbert_tokens=[[1.0, 2.0, 3.0]]),
        ])
    assert col.count() == 2
    col.wal.close()


# ---------------------------------------------------------------------------
# C5: BQ/PQ linear_score_fusion NaN, negative dense scores & multi-modal RRF
# ---------------------------------------------------------------------------

def test_c5_bq_pq_linear_and_rrf_no_nan_no_neginf():
    for mode in ("bq", "pq"):
        col = Collection(f"c5_{mode}", dimension=4, metric="cosine")
        docs = [
            Document(
                id="target",
                vector=[-0.1, 0.995, 0.0, 0.0],
                text="apple orchard",
                sparse_vector={1: 2.0},
                colbert_tokens=[[1.0, 0.0, 0.0, 0.0]],
            )
        ]
        for i in range(1, 45):
            docs.append(
                Document(
                    id=f"neg_{i}",
                    vector=[-0.9, 0.435, 0.0, 0.0],
                    text="banana" if i < 40 else "cherry",
                    sparse_vector={2: 1.0},
                    colbert_tokens=[[0.0, 1.0, 0.0, 0.0]],
                )
            )
        col.insert(docs)
        if mode == "bq":
            col.enable_bq()
        else:
            col.train_pq(m=2)

        # 1. Pure dense query with negative similarities (>32 docs)
        pure = col.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
        assert len(pure) == 5
        assert all(math.isfinite(r.score) and math.isfinite(r.dense_score) for r in pure)
        assert pure[0].id == "target"

        # 2. Linear fusion alpha=0.5 (previously NaN due to -inf in dense_scores)
        lin = col.query(vector=[1.0, 0.0, 0.0, 0.0], text="apple", alpha=0.5, top_k=5)
        assert len(lin) == 5
        assert all(math.isfinite(r.score) and math.isfinite(r.dense_score) for r in lin)
        assert lin[0].id == "target"

        # 3. Bi-modal RRF
        rrf2 = col.query(vector=[1.0, 0.0, 0.0, 0.0], text="apple", hybrid=True, top_k=5)
        assert len(rrf2) == 5
        assert all(math.isfinite(r.score) and math.isfinite(r.dense_score) for r in rrf2)
        assert rrf2[0].id == "target"

        # 4. Quad-modal RRF with 0-hit BM25/sparse/colbert terms must NOT inject false zero-score ranks
        rrf4 = col.query(
            vector=[1.0, 0.0, 0.0, 0.0],
            text="nonexistentterm",
            sparse_vector={999: 1.0},
            colbert_tokens=[[0.0, 0.0, 1.0, 0.0]],
            hybrid=True,
            top_k=5,
        )
        assert len(rrf4) == 5
        for r in rrf4:
            assert math.isfinite(r.score)
            assert math.isfinite(r.dense_score)
            assert math.isfinite(r.bm25_score)
            assert math.isfinite(r.sparse_score)
            assert math.isfinite(r.colbert_score)

    assert linear_score_fusion(np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32), top_k=0) == []
    assert reciprocal_rank_fusion(np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32), top_k=-1) == []


# ---------------------------------------------------------------------------
# C7: Pure-Python HNSWIndex, filtered traversal, deletion, compaction
# ---------------------------------------------------------------------------

def test_c7_hnsw_filtered_recall_deletion_and_compaction():
    # Build >512 clustered docs where 400 are blocked near query and 600 match filter orthogonal
    col = Collection("c7_hnsw", dimension=4, metric="cosine")
    docs = []
    for i in range(700):
        if i < 250:
            tag = "blocked"
            vec = [1.0, 0.001 * i, 0.0, 0.0]
        else:
            tag = "allowed"
            vec = [0.0, 1.0, 0.001 * (i - 250), 0.0]
        docs.append(Document(id=f"d{i}", vector=vec, text=f"doc {i}", metadata={"tag": tag, "idx": i}))
    col.insert(docs)

    hnsw_info = col.build_hnsw(m=16, ef_construction=100, ef_search=50)
    assert isinstance(hnsw_info, dict)
    assert isinstance(col._hnsw_index, HNSWIndex)
    assert col.stats()["has_hnsw"] is True
    assert col.stats()["hnsw"]["count"] == 700

    # Filtered HNSW query with default hybrid=True (single modality dense)
    res_filtered = col.query(vector=[1.0, 0.0, 0.0, 0.0], filter={"tag": "allowed"}, top_k=10, use_ann=True)
    assert len(res_filtered) == 10
    assert all(r.metadata["tag"] == "allowed" for r in res_filtered)

    # Tiny selectivity (<2% and <=32) falls back to exact recall cleanly
    res_tiny = col.query(vector=[0.0, 1.0, 0.0, 0.0], filter={"idx": 300}, top_k=5, use_ann=True)
    assert len(res_tiny) == 1 and res_tiny[0].id == "d300"

    # Delete nodes and verify no dangling incoming edges remain in HNSW layers
    deleted_indices = [col.id_to_index[f"d{i}"] for i in range(250, 260)]
    assert col.delete([f"d{i}" for i in range(250, 260)]) == 10
    for layer in col._hnsw_index.layers:
        for deleted_idx in deleted_indices:
            assert deleted_idx not in layer
            for nbrs in layer.values():
                assert deleted_idx not in nbrs

    # Query after deletion must never return deleted IDs
    res_after_del = col.query(vector=[0.0, 1.0, 0.0, 0.0], filter={"tag": "allowed"}, top_k=10, use_ann=True)
    assert len(res_after_del) == 10
    assert all(r.id not in {f"d{i}" for i in range(250, 260)} for r in res_after_del)

    # Compact tombstones and verify active HNSW remains intact and functional
    purged = col.compact()
    assert purged == 10
    assert col.count() == 690
    assert col.stats()["tombstones"] == 0
    assert col.stats()["has_hnsw"] is True
    assert col.stats()["hnsw"]["count"] == 690
    assert col.stats()["hnsw"]["dirty"] is False
    res_post_compact = col.query(vector=[0.0, 1.0, 0.0, 0.0], top_k=5, use_ann=True, hybrid=False)
    assert len(res_post_compact) == 5
    assert res_post_compact[0].id == "d260"


# ---------------------------------------------------------------------------
# H3: Name validation, Haversine clamping, ReDoS protection
# ---------------------------------------------------------------------------

def test_h3_collection_name_validation_and_path_confinement(tmp_path: Path):
    db = SutraDB(persist_directory=tmp_path)
    for bad_name in ("../escaped", "a/b", "a\\b", "", "bad name", "bad\nname", "a" * 129):
        with pytest.raises(ValueError, match="Invalid collection name"):
            db.create_collection(bad_name, dimension=2)
        with pytest.raises(ValueError, match="Invalid collection name"):
            db.get_collection(bad_name)
        with pytest.raises(ValueError, match="Invalid collection name"):
            db.drop_collection(bad_name)
        with pytest.raises(ValueError, match="Invalid collection name"):
            Collection(bad_name, dimension=2)
    assert not (tmp_path.parent / "escaped.sutra").exists()


def test_h3_filters_haversine_and_redos_guardrails():
    # Antipodal / near-antipodal Haversine must not raise math domain error
    for eps in (0.0, 1e-16, -1e-16, 1e-15, -1e-15):
        d = haversine_distance(37.7749, -122.4194, -37.7749, 57.5806 + eps)
        assert math.isfinite(d) and d > 0.0

    assert extract_lat_lon([float("nan"), 0.0]) is None
    assert extract_lat_lon([True, 0.0]) is None
    assert not FilterEngine.match_value([1, 2], {"$size": True})
    assert not FilterEngine.match_value([0.0, 0.0], {"$geo_within": {"center": [0.0, 0.0], "radius_meters": float("nan")}})
    assert not FilterEngine.match_value([0.0, 0.0], {"$geo_within": {"center": [0.0, 0.0], "radius_meters": -10}})

    # Invalid regex syntax returns False cleanly
    assert FilterEngine.evaluate({"tag": "abc"}, {"tag": {"$regex": "[unclosed"}}) is False

    # Catastrophic ReDoS patterns fail fast (< 0.1s) without hanging
    for evil in ("(a+)+$", "(a|aa)+", "([a-zA-Z]+)*", ".*.*=", "a{1001}", r"(a+)\1+"):
        t0 = time.perf_counter()
        with pytest.raises(ValueError, match="Unsafe regex"):
            FilterEngine.evaluate({"tag": "a" * 30 + "!"}, {"tag": {"$regex": evil}})
        assert (time.perf_counter() - t0) < 0.1

    # Safe regexes work normally
    assert FilterEngine.evaluate({"tag": "cloud_prod_12"}, {"tag": {"$regex": r"^cloud_[a-z]+_\d+$"}})
    assert FilterEngine.evaluate({"msg": "123   abc"}, {"msg": {"$regex": r"\d+\s+\w+"}})
    assert FilterEngine.evaluate({"x": "ababa"}, {"x": {"$regex": r"^(?:a|b)+$"}})


# ---------------------------------------------------------------------------
# M3: IVF metric alignment (spherical k-means for COSINE)
# ---------------------------------------------------------------------------

def test_m3_ivf_cosine_normalized_centroids():
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(120, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    idx = IVFIndex(IVFConfig(n_lists=8, nprobe=3, seed=42), metric=Metric.COSINE)
    idx.train(vecs)
    norms = np.linalg.norm(idx.centroids, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    idx.add(vecs, start_idx=0)
    res_idx, res_scores = idx.search(vecs[0], vecs, k=3, metric=Metric.COSINE, is_normalized=True)
    assert res_idx[0] == 0
    assert res_scores[0] > 0.99


# ---------------------------------------------------------------------------
# M4: BM25 tombstone removal & ES/FR/DE stemmers
# ---------------------------------------------------------------------------

def test_m4_bm25_tombstones_and_multilingual_stemmers():
    idx = BM25Index()
    idx.add_documents([
        "alpha common",
        "beta common",
        "gamma unique",
    ])
    assert idx.doc_count == 3
    idx.remove_document(1, "beta common")
    assert idx.doc_count == 2
    assert idx.doc_frequencies.get("beta", 0) == 0
    assert idx.doc_frequencies.get("common", 0) == 1

    # Adding after delete must not collide with slot indices
    idx.add_documents(["delta new"])
    assert idx.doc_count == 3
    scores = idx.score_query("delta")
    assert len(scores) == 4
    assert scores[3] > 0.0 and scores[1] == 0.0

    # Collection delete updates BM25 immediately (no tombstoned doc IDF inflation)
    col = Collection("bm25_del", dimension=2)
    col.insert([
        Document(id="d1", vector=[1.0, 0.0], text="rareterm alpha"),
        Document(id="d2", vector=[0.0, 1.0], text="rareterm beta"),
        Document(id="d3", vector=[1.0, 1.0], text="other doc"),
    ])
    col.delete(["d2"])
    pre_compact_score = col.query(text="rareterm", top_k=5)[0].score
    col.compact()
    post_compact_score = col.query(text="rareterm", top_k=5)[0].score
    assert pytest.approx(pre_compact_score, rel=1e-5) == post_compact_score

    # ES / FR / DE stemmers
    assert Analyzer(AnalyzerConfig(language="spanish")).analyze("optimizacion desarrollando") == ["optim", "desarroll"]
    assert Analyzer(AnalyzerConfig(language="french")).analyze("deploiement optimisation") == ["deploi", "optim"]
    assert Analyzer(AnalyzerConfig(language="german")).analyze("optimierung verbindung") == ["optimier", "verbind"]


# ---------------------------------------------------------------------------
# M5: Collection.load .sutra fallback & WAL-only metric/config recovery
# ---------------------------------------------------------------------------

def test_m5_load_sutra_fallback_and_wal_metric_recovery(tmp_path: Path):
    # 1. save() without .sutra suffix + load() without .sutra suffix
    sp_no_ext = tmp_path / "mydb"
    cs = Collection("noext", dimension=2, metric="euclidean", storage_path=sp_no_ext)
    cs.insert([Document(id="1", vector=[3.0, 4.0], text="hello")])
    cs.save()
    assert (tmp_path / "mydb.sutra").exists()

    cl = Collection("noext", dimension=2, storage_path=sp_no_ext)
    assert cl.load() == 1
    assert cl.metric == Metric.EUCLIDEAN
    assert np.allclose(cl.vectors[0], [3.0, 4.0])

    # 2. WAL-only recovery preserves metric="euclidean" & enable_wal=True (both 0 docs and with docs)
    db_dir = tmp_path / "wal_metric_db"
    db1 = SutraDB(persist_directory=db_dir)
    c_empty = db1.create_collection("euc_empty", dimension=3, metric="euclidean", enable_wal=True)
    c_empty.wal.close()

    c_docs = db1.create_collection("dot_docs", dimension=2, metric="dot_product", enable_wal=True)
    c_docs.insert([Document(id="x1", vector=[3.0, 4.0], text="dot")])
    c_docs.wal.close()

    db2 = SutraDB(persist_directory=db_dir)
    rec_empty = db2.get_collection("euc_empty")
    assert rec_empty.dimension == 3
    assert rec_empty.metric == Metric.EUCLIDEAN
    assert rec_empty.wal is not None

    rec_docs = db2.get_collection("dot_docs")
    assert rec_docs.dimension == 2
    assert rec_docs.metric == Metric.DOT_PRODUCT
    assert rec_docs.wal is not None
    assert np.allclose(rec_docs.vectors[0], [3.0, 4.0])  # NOT cosine-normalized!
    rec_empty.wal.close()
    rec_docs.wal.close()
