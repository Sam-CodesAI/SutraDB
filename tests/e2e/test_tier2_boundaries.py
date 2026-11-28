"""
Tier 2 E2E Boundary & Adversarial Negative Tests (BVA + Audit Defects).
Covers all 8 feature domains (F1-F8) with >= 5 boundary/negative tests per domain
(42 total) including all 14 mandatory edge cases from the SutraDB v5.5.0 audit.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Callable, List
from urllib.parse import urlparse
import zlib

import httpx
import numpy as np
import pytest

from sutradb import (
    Collection,
    ConsistentHashRing,
    DistributedCluster,
    DistributedCollection,
    Document,
    FilterEngine,
    IVFConfig,
    NativeCollection,
    PQCodebook,
    SutraDB,
)
from sutradb._sutradb import batch_vector_distances, haversine as native_haversine
from sutradb.filters import haversine_distance
from sutradb.fusion import linear_score_fusion
from sutradb.storage import BinaryStorage, HEADER_FORMAT, HEADER_MAGIC_V2, HEADER_MAGIC_V3

from tests.e2e.conftest import REPO_ROOT, get_free_port


# =============================================================================
# F1: Dense & Quantization Boundaries (5 tests)
# =============================================================================


def test_f1_boundary_sq8_negative_coordinates_save_load(
    tmp_path: Path,
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory C1: Normalized negative-coordinate vector under SQ8 survives save/load with ~1.0 cosine."""
    neg_vec = [-0.70710677, 0.0, 0.70710677, 0.0]
    other_vec = [0.0, 1.0, 0.0, 0.0]

    # Python engine
    py_snap = tmp_path / "py_sq8_neg.sutra"
    py_col = py_collection_factory(name="py_sq8_neg", dimension=4, metric="cosine", storage_filename="py_sq8_neg.sutra")
    py_col.insert([
        Document(id="neg", vector=neg_vec, text="negative coords"),
        Document(id="orth", vector=other_vec, text="orthogonal"),
    ])
    py_col.enable_sq8()
    py_col.save()

    assert py_snap.read_bytes()[:8] == b"SUTRA\x04\x00\x00"
    py_reloaded = Collection(name="py_sq8_neg", dimension=4, metric="cosine")
    assert py_reloaded.load(py_snap) == 2
    assert np.allclose(py_reloaded.vectors[0], np.array(neg_vec, dtype=np.float32), atol=0.05)
    py_hits = py_reloaded.query(vector=neg_vec, top_k=2, hybrid=False)
    assert py_hits[0].id == "neg"
    assert py_hits[0].score > 0.95

    # Rust Native engine
    nat_snap = str(tmp_path / "nat_sq8_neg.sutra")
    nat_col = native_collection_factory(name="nat_sq8_neg", dimension=4, metric="cosine", storage_filename="nat_sq8_neg.sutra")
    nat_col.insert(
        [{"id": "neg", "text": "negative coords", "metadata": {}}, {"id": "orth", "text": "orthogonal", "metadata": {}}],
        [neg_vec, other_vec],
    )
    nat_col.enable_sq8()
    nat_col.save()

    nat_reloaded = NativeCollection("nat_sq8_neg", 4, "cosine", nat_snap, False)
    assert nat_reloaded.load(nat_snap) == 2
    nat_hits = nat_reloaded.query(vector=neg_vec, top_k=2, hybrid=False)
    assert nat_hits[0]["id"] == "neg"
    assert nat_hits[0]["score"] > 0.95


def test_f1_boundary_bq_pq_negative_similarities_gt32_docs(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory C6: On >32 docs where all cosines are negative, best (-0.1) ranks #1 under BQ and PQ."""
    dim = 16
    # Query points along +x0 (all 16 BQ bits = 1).
    # best has cosine = -0.1 and BQ Hamming distance d_H = 1 (only dim 0 is negative).
    # worst has cosine = -0.9 and negative trailing dims 2..7 so d_H = 7 >= 6 (no reliance on stable sort).
    q = [1.0] + [0.0] * (dim - 1)
    best = [-0.1, float(math.sqrt(1.0 - 0.01))] + [0.0] * (dim - 2)
    worst_dim1 = float(math.sqrt(1.0 - 0.81 - 6 * (0.05 ** 2)))
    worst = [-0.9, worst_dim1] + [-0.05] * 6 + [0.0] * (dim - 8)

    vecs = [list(worst) for _ in range(40)]
    vecs[17] = list(best)

    # Python BQ + PQ
    for mode in ("bq", "pq"):
        py_col = py_collection_factory(name=f"py_neg_{mode}", dimension=dim, metric="cosine")
        py_col.insert([Document(id=f"d_{i}", vector=vecs[i]) for i in range(40)])
        if mode == "bq":
            py_col.enable_bq()
        else:
            py_col.train_pq(m=4)
            py_col.enable_pq()
        res = py_col.query(vector=q, top_k=5, hybrid=False)
        assert res[0].id == "d_17", f"Python {mode} failed negative similarity ranking: {res[0]}"

    # Rust Native BQ + PQ
    for mode in ("bq", "pq"):
        nat_col = native_collection_factory(name=f"nat_neg_{mode}", dimension=dim, metric="cosine")
        nat_col.insert([{"id": f"d_{i}", "text": "", "metadata": {}} for i in range(40)], vecs)
        if mode == "bq":
            nat_col.enable_bq()
        else:
            nat_col.train_pq(m=4)
            nat_col.enable_pq()
        res = nat_col.query(vector=q, top_k=5, hybrid=False)
        assert res[0]["id"] == "d_17", f"Rust {mode} failed negative similarity ranking: {res[0]}"


def test_f1_boundary_rayon_64_docs_dimension_mismatch(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory C6: With >= 64 docs, querying with wrong dimension raises ValueError (no dummy 0.0)."""
    vecs = [[1.0, 0.0, 0.0, 0.0] for _ in range(64)]

    nat_col = native_collection_factory(name="rayon_dim", dimension=4, metric="cosine")
    nat_col.insert([{"id": f"r_{i}", "text": "", "metadata": {}} for i in range(64)], vecs)
    with pytest.raises(ValueError):
        nat_col.query(vector=[1.0, 0.0], top_k=5, hybrid=False)

    py_col = py_collection_factory(name="py_dim_64", dimension=4, metric="cosine")
    py_col.insert([Document(id=f"r_{i}", vector=vecs[i]) for i in range(64)])
    with pytest.raises(ValueError):
        py_col.query(vector=[1.0, 0.0], top_k=5, hybrid=False)


def test_f1_boundary_pq_invalid_m_and_empty_collection(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory H2: PQCodebook / train_pq rejects m=0, indivisible m, and empty collections via ValueError."""
    vecs = [[1.0, 0.0, 0.0, 0.0] for _ in range(10)]
    with pytest.raises(ValueError):
        PQCodebook.train(vecs, m=0)
    with pytest.raises(ValueError):
        PQCodebook.train(vecs, m=3)

    py_empty = py_collection_factory(name="py_pq_empty", dimension=4)
    with pytest.raises(ValueError):
        py_empty.train_pq(m=2)

    nat_empty = native_collection_factory(name="nat_pq_empty", dimension=4)
    with pytest.raises(ValueError):
        nat_empty.train_pq(m=2)

    nat_col = native_collection_factory(name="nat_pq_bad_m", dimension=4)
    nat_col.insert([{"id": f"p_{i}", "text": "", "metadata": {}} for i in range(10)], vecs)
    with pytest.raises(ValueError):
        nat_col.train_pq(m=0)
    with pytest.raises(ValueError):
        nat_col.train_pq(m=3)


def test_f1_boundary_insert_and_batch_distances_dimension_mismatch(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Dimension mismatches on insert and batch_vector_distances raise ValueError cleanly."""
    py_col = py_collection_factory(name="py_ins_dim", dimension=4)
    with pytest.raises(ValueError):
        py_col.insert(Document(id="bad", vector=[1.0, 2.0]))

    nat_col = native_collection_factory(name="nat_ins_dim", dimension=4)
    with pytest.raises(ValueError):
        nat_col.insert([{"id": "bad", "text": "", "metadata": {}}], [[1.0, 2.0]])

    with pytest.raises(ValueError):
        batch_vector_distances([1.0, 0.0, 0.0, 0.0], [[1.0, 0.0]], metric="cosine")


# =============================================================================
# F2: ANN Indexing Boundaries (5 tests)
# =============================================================================


def test_f2_boundary_filtered_hnsw_low_selectivity(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory C8: 1000 clustered vectors (400 blocked near query, 600 valid orthogonal) return top_k=10."""
    rng = np.random.default_rng(208)
    dim = 16
    vecs: List[List[float]] = []
    nat_docs: List[Dict[str, Any]] = []
    py_docs: List[Document] = []

    for i in range(1000):
        if i < 400:
            # Near query [1, 0, ...] but blocked by filter
            v = np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32) + rng.normal(scale=0.02, size=dim).astype(np.float32)
            allowed = False
        else:
            # Orthogonal valid candidates (>512 and >2%)
            v = np.array([0.0, 1.0] + [0.0] * (dim - 2), dtype=np.float32) + rng.normal(scale=0.05, size=dim).astype(np.float32)
            allowed = True
        v /= np.linalg.norm(v)
        v_list = v.tolist()
        vecs.append(v_list)
        nat_docs.append({"id": f"c_{i}", "text": "", "metadata": {"allowed": allowed}})
        py_docs.append(Document(id=f"c_{i}", vector=v_list, metadata={"allowed": allowed}))

    query_vec = [1.0] + [0.0] * (dim - 1)

    nat_col = native_collection_factory(name="nat_low_sel", dimension=dim, metric="cosine")
    nat_col.insert(nat_docs, vecs)
    nat_col.build_hnsw(m=16, ef_construction=64, ef_search=32)
    nat_res = nat_col.query(vector=query_vec, filter={"allowed": True}, top_k=10, hybrid=False, use_ann=True)
    assert len(nat_res) == 10
    assert all(r["metadata"]["allowed"] is True for r in nat_res)

    py_col = py_collection_factory(name="py_low_sel", dimension=dim, metric="cosine")
    py_col.insert(py_docs)
    py_col.build_hnsw(m=16, ef_construction=64, ef_search=32)
    py_res = py_col.query(vector=query_vec, filter={"allowed": True}, top_k=10, hybrid=False, use_ann=True)
    assert len(py_res) == 10
    assert all(r.metadata["allowed"] is True for r in py_res)


def test_f2_boundary_hnsw_delete_compact_save_load_preservation(
    tmp_path: Path,
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory M2 & M5: Delete + compact + save/load preserves has_hnsw and executes cleanly without dangling edges."""
    vecs = [[float(i == j) for j in range(4)] for i in range(4)] + [[0.7, 0.7, 0.0, 0.0]]
    nat_path = str(tmp_path / "nat_hnsw_persist.sutra")
    nat_col = native_collection_factory(
        name="nat_hnsw_persist", dimension=4, metric="cosine", storage_filename="nat_hnsw_persist.sutra"
    )
    nat_col.insert([{"id": f"h_{i}", "text": f"doc {i}", "metadata": {}} for i in range(5)], vecs)
    nat_col.build_hnsw(m=8, ef_construction=32, ef_search=16)
    assert nat_col.delete(["h_0", "h_1"]) == 2
    nat_col.compact()
    assert nat_col.stats()["has_hnsw"] is True
    nat_col.save()
    assert nat_col.stats()["has_hnsw"] is True

    reloaded = NativeCollection("nat_hnsw_persist", 4, "cosine", nat_path, False)
    reloaded.load(nat_path)
    res = reloaded.query(vector=[0.7, 0.7, 0.0, 0.0], top_k=3, hybrid=False, use_ann=True)
    assert res[0]["id"] == "h_4"
    assert all(r["id"] not in ("h_0", "h_1") for r in res)

    py_col = py_collection_factory(name="py_hnsw_persist", dimension=4, metric="cosine")
    py_col.insert([Document(id=f"h_{i}", vector=vecs[i]) for i in range(5)])
    py_col.build_hnsw(m=8, ef_construction=32, ef_search=16)
    py_col.delete(["h_0"])
    py_col.compact()
    assert py_col.stats()["has_hnsw"] is True
    py_res = py_col.query(vector=[0.7, 0.7, 0.0, 0.0], top_k=4, hybrid=False, use_ann=True)
    assert py_res[0].id == "h_4"


def test_f2_boundary_empty_collection_index_errors(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Building IVF on an empty collection raises an explicit error without panicking."""
    py_col = py_collection_factory(name="py_empty_ann", dimension=4)
    with pytest.raises(ValueError):
        py_col.build_index()

    nat_col = native_collection_factory(name="nat_empty_ann", dimension=4)
    with pytest.raises((ValueError, RuntimeError)):
        nat_col.build_ivf()


def test_f2_boundary_top_k_exceeds_active_count(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Querying top_k=100 on 7 active HNSW docs returns exact 7 unique non-deleted docs."""
    rng = np.random.default_rng(209)
    vecs = rng.normal(size=(10, 8)).tolist()

    py_col = py_collection_factory(name="py_topk_over", dimension=8)
    py_col.insert([Document(id=f"k_{i}", vector=vecs[i]) for i in range(10)])
    py_col.build_hnsw(m=8, ef_construction=32, ef_search=16)
    py_col.delete(["k_0", "k_1", "k_2"])
    py_res = py_col.query(vector=vecs[5], top_k=100, hybrid=False, use_ann=True)
    assert len(py_res) == 7
    assert len({r.id for r in py_res}) == 7

    nat_col = native_collection_factory(name="nat_topk_over", dimension=8)
    nat_col.insert([{"id": f"k_{i}", "text": "", "metadata": {}} for i in range(10)], vecs)
    nat_col.build_hnsw(m=8, ef_construction=32, ef_search=16)
    nat_col.delete(["k_0", "k_1", "k_2"])
    nat_res = nat_col.query(vector=vecs[5], top_k=100, hybrid=False, use_ann=True)
    assert len(nat_res) == 7
    assert len({r["id"] for r in nat_res}) == 7


def test_f2_boundary_ivf_euclidean_vs_cosine_alignment(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory M3: IVF with nprobe < n_lists aligns centroid selection with euclidean vs cosine metric."""
    # 24 distinct vectors across 4 distinct clusters (>= 20 distinct vectors so k-means++ never stalls).
    # Query = [1, 0, 0, 0].
    # Cluster 0 (e_0..e_5): near [1.0, 0.2, 0, 0] -> smallest L2 (~0.2), cosine ~ 0.9806.
    # Cluster 1 (e_6..e_11): far along +x [100+i, 0, 0, 0] -> collinear cosine = 1.0, huge L2 >= 99.0.
    # Cluster 2 (e_12..e_17): along +z [0, 0, 50+i, 0].
    # Cluster 3 (e_18..e_23): along +w [0, 0, 0, 50+i].
    vecs = (
        [[1.0 + 0.02 * i, 0.2 + 0.02 * i, 0.01 * i, 0.0] for i in range(6)]
        + [[100.0 + float(i), 0.0, 0.0, 0.0] for i in range(6)]
        + [[0.0, 0.0, 50.0 + float(i), 0.0] for i in range(6)]
        + [[0.0, 0.0, 0.0, 50.0 + float(i)] for i in range(6)]
    )
    q = [1.0, 0.0, 0.0, 0.0]

    # Euclidean IVF with nprobe=1 < n_lists=4 must probe Cluster 0 and return e_0
    py_euc = py_collection_factory(name="py_ivf_euc", dimension=4, metric="euclidean")
    py_euc.insert([Document(id=f"e_{i}", vector=vecs[i]) for i in range(len(vecs))])
    py_euc.build_index(IVFConfig(n_lists=4, nprobe=1))
    assert py_euc.query(vector=q, top_k=3, hybrid=False, use_ann=True)[0].id == "e_0"

    nat_euc = native_collection_factory(name="nat_ivf_euc", dimension=4, metric="euclidean")
    nat_euc.insert([{"id": f"e_{i}", "text": "", "metadata": {}} for i in range(len(vecs))], vecs)
    nat_euc.build_ivf(n_lists=4, nprobe=1)
    assert nat_euc.query(vector=q, top_k=3, hybrid=False, use_ann=True)[0]["id"] == "e_0"

    # Cosine IVF with nprobe=1 < n_lists=4 must probe Cluster 1 (collinear cosine=1.0) over Cluster 0
    collinear_ids = {f"e_{i}" for i in range(6, 12)}
    py_cos = py_collection_factory(name="py_ivf_cos", dimension=4, metric="cosine")
    py_cos.insert([Document(id=f"e_{i}", vector=vecs[i]) for i in range(len(vecs))])
    py_cos.build_index(IVFConfig(n_lists=4, nprobe=1))
    assert py_cos.query(vector=q, top_k=3, hybrid=False, use_ann=True)[0].id in collinear_ids

    nat_cos = native_collection_factory(name="nat_ivf_cos", dimension=4, metric="cosine")
    nat_cos.insert([{"id": f"e_{i}", "text": "", "metadata": {}} for i in range(len(vecs))], vecs)
    nat_cos.build_ivf(n_lists=4, nprobe=1)
    assert nat_cos.query(vector=q, top_k=3, hybrid=False, use_ann=True)[0]["id"] in collinear_ids


# =============================================================================
# F3: Hybrid & Multi-Modal Boundaries (5 tests)
# =============================================================================


def test_f3_boundary_linear_fusion_alpha_with_bq_pq_finite_scores(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory C5 & C6: alpha=0.5 with BQ/PQ on >32 docs & -inf dense scores yields strictly finite scores."""
    # Direct linear_score_fusion guard against -inf
    dense_arr = np.array([-np.inf, 0.8, -np.inf, 0.2], dtype=np.float32)
    bm25_arr = np.array([1.2, 0.5, 0.9, 0.0], dtype=np.float32)
    fused = linear_score_fusion(dense_arr, bm25_arr, alpha=0.5, top_k=4)
    assert len(fused) == 4
    assert all(math.isfinite(item[1]) for item in fused)

    # Full collection query with 60 docs + BQ/PQ + alpha=0.5, top_k=5
    # Stage-1 rerank window is max(5 * 4, 32) = 32 < 60, leaving >= 28 docs with dense_scores = -inf
    rng = np.random.default_rng(305)
    vecs = rng.normal(size=(60, 16)).astype(np.float32).tolist()

    for mode in ("bq", "pq"):
        py_col = py_collection_factory(name=f"py_lin_{mode}", dimension=16, metric="cosine")
        py_col.insert([Document(id=f"lb_{i}", vector=vecs[i], text=f"hybrid token {i}") for i in range(60)])
        if mode == "bq":
            py_col.enable_bq()
        else:
            py_col.train_pq(m=4)
            py_col.enable_pq()
        py_res = py_col.query(vector=vecs[0], text="hybrid token", alpha=0.5, top_k=5, hybrid=True)
        assert len(py_res) == 5
        assert all(math.isfinite(r.score) for r in py_res)

        nat_col = native_collection_factory(name=f"nat_lin_{mode}", dimension=16, metric="cosine")
        nat_col.insert([{"id": f"lb_{i}", "text": f"hybrid token {i}", "metadata": {}} for i in range(60)], vecs)
        if mode == "bq":
            nat_col.enable_bq()
        else:
            nat_col.train_pq(m=4)
            nat_col.enable_pq()
        nat_res = nat_col.query(vector=vecs[0], text="hybrid token", alpha=0.5, top_k=5, hybrid=True)
        assert len(nat_res) == 5
        assert all(math.isfinite(r["score"]) for r in nat_res)


def test_f3_boundary_bm25_tombstone_exclusion_and_idf_update(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory M4: Deleting a document excludes it from BM25 results and strictly increases surviving corpus IDF."""
    # N = 4 equal-length docs where 2 share the rare term; deleting b1 shifts N=4,df=2 -> N=3,df=1
    docs = [
        ("b1", [1.0, 0.0, 0.0, 0.0], "rare_cryptographic_term common_word"),
        ("b2", [0.0, 1.0, 0.0, 0.0], "rare_cryptographic_term other_word"),
        ("b3", [0.0, 0.0, 1.0, 0.0], "unrelated filler"),
        ("b4", [0.0, 0.0, 0.0, 1.0], "another document"),
    ]

    py_col = py_collection_factory(name="py_bm25_del", dimension=4)
    py_col.insert([Document(id=i, vector=v, text=t) for i, v, t in docs])
    py_before = py_col.query(text="rare_cryptographic_term", top_k=5, hybrid=False)
    assert len(py_before) == 2
    py_score_before = next(r.score for r in py_before if r.id == "b2")
    py_col.delete(["b1"])
    py_after = py_col.query(text="rare_cryptographic_term", top_k=5, hybrid=False)
    assert [r.id for r in py_after] == ["b2"]
    assert py_after[0].score > py_score_before, (
        f"Python BM25 IDF did not update on delete: before={py_score_before}, after={py_after[0].score}"
    )

    nat_col = native_collection_factory(name="nat_bm25_del", dimension=4)
    nat_col.insert([{"id": i, "text": t, "metadata": {}} for i, _, t in docs], [v for _, v, _ in docs])
    nat_before = nat_col.query(text="rare_cryptographic_term", top_k=5, hybrid=False)
    assert len(nat_before) == 2
    nat_score_before = next(r["score"] for r in nat_before if r["id"] == "b2")
    nat_col.delete(["b1"])
    nat_after = nat_col.query(text="rare_cryptographic_term", top_k=5, hybrid=False)
    assert [r["id"] for r in nat_after] == ["b2"]
    assert nat_after[0]["score"] > nat_score_before, (
        f"Rust BM25 IDF did not update on delete: before={nat_score_before}, after={nat_after[0]['score']}"
    )


def test_f3_boundary_sparse_vector_invalid_inputs_no_panic(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory H2: Malformed/empty/non-finite sparse vectors never panic."""
    nat_col = native_collection_factory(name="nat_bad_sp", dimension=4)
    nat_col.insert(
        [{"id": "s1", "text": "ok", "metadata": {}, "sparse_vector": {1: 1.0}}],
        [[1.0, 0.0, 0.0, 0.0]],
    )
    # Empty sparse vector query
    assert isinstance(nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], sparse_vector={}, top_k=5), list)
    # Non-numeric string key in sparse_vector
    try:
        res = nat_col.insert(
            [{"id": "s_bad", "text": "bad", "metadata": {}, "sparse_vector": {"not_an_int": 1.0}}],
            [[0.0, 1.0, 0.0, 0.0]],
        )
        assert isinstance(res, int)
    except (ValueError, TypeError):
        pass

    py_col = py_collection_factory(name="py_bad_sp", dimension=4)
    py_col.insert(Document(id="s1", vector=[1.0, 0.0, 0.0, 0.0], sparse_vector={1: 1.0}))
    assert isinstance(py_col.query(vector=[1.0, 0.0, 0.0, 0.0], sparse_vector={}, top_k=5), list)


def test_f3_boundary_colbert_dimension_mismatch_no_panic(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory H2: ColBERT token dimension mismatch raises ValueError cleanly without panicking."""
    nat_col = native_collection_factory(name="nat_cb_dim", dimension=4)
    nat_col.insert(
        [{"id": "c1", "text": "cb", "metadata": {}, "colbert_tokens": [[1.0, 0.0, 0.0, 0.0]]}],
        [[1.0, 0.0, 0.0, 0.0]],
    )
    with pytest.raises(ValueError):
        nat_col.query_colbert(query_tokens=[[1.0, 0.0]], top_k=2)

    py_col = py_collection_factory(name="py_cb_dim", dimension=4)
    py_col.insert(Document(id="c1", vector=[1.0, 0.0, 0.0, 0.0], colbert_tokens=[[1.0, 0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        py_col.query(colbert_tokens=[[1.0, 0.0]], top_k=2, hybrid=False)


def test_f3_boundary_empty_query_and_zero_weights(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Empty query fails safely with ValueError; zero weights never produce NaN."""
    py_col = py_collection_factory(name="py_zero_w", dimension=4)
    py_col.insert(Document(id="d1", vector=[1.0, 0.0, 0.0, 0.0], text="alpha"))
    with pytest.raises(ValueError):
        py_col.query(text="   ")

    res = py_col.query(vector=[1.0, 0.0, 0.0, 0.0], text="alpha", dense_weight=0.0, bm25_weight=0.0, hybrid=True)
    assert all(math.isfinite(r.score) for r in res)

    nat_col = native_collection_factory(name="nat_zero_w", dimension=4)
    nat_col.insert([{"id": "d1", "text": "alpha", "metadata": {}}], [[1.0, 0.0, 0.0, 0.0]])
    nat_res = nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], text="alpha", dense_weight=0.0, bm25_weight=0.0, hybrid=True)
    assert all(math.isfinite(r["score"]) for r in nat_res)


# =============================================================================
# F4: Metadata & Geospatial Filtering Boundaries (5 tests)
# =============================================================================


def test_f4_boundary_rust_regex_not_exists_and_type_parity(
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory H1: Rust $regex, $not+$contains, $exists on {} and [], and int/float index parity."""
    nat_col = native_collection_factory(name="nat_h1", dimension=4)
    nat_col.insert(
        [
            {"id": "d1", "text": "one", "metadata": {"tag": "cloud_prod", "tags": [], "user": {"name": "a"}, "num": 1}},
            {"id": "d2", "text": "two", "metadata": {"tag": "local_dev", "tags": ["spam"], "num": 2.0}},
        ],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )

    assert [r["id"] for r in nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter={"tag": {"$regex": "^cloud"}})] == ["d1"]
    assert [r["id"] for r in nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter={"$not": {"tags": {"$contains": "spam"}}})] == ["d1"]
    assert [r["id"] for r in nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter={"user": {"$exists": True}})] == ["d1"]
    assert len(nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter={"tags": {"$exists": True}})) == 2
    assert [r["id"] for r in nat_col.query(vector=[1.0, 0.0, 0.0, 0.0], filter={"num": 1.0})] == ["d1"]


def test_f4_boundary_redos_pathological_regex_bounded_time(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory H3: Evil ReDoS regex (a+)+$ fails or completes in bounded time (<3.0s) without GIL-lock hangs."""
    # Run each ReDoS probe in an isolated subprocess with timeout=3.0s because CPython _sre backtracking
    # holds the GIL and blocks ThreadPoolExecutor shutdown(wait=True) until backtracking completes.
    py_probe = (
        "from sutradb import FilterEngine\n"
        "try:\n"
        "    res = FilterEngine.evaluate({'s': 'a' * 30 + '!'}, {'s': {'$regex': '(a+)+$'}})\n"
        "    print('FALSE' if res is False else 'TRUE')\n"
        "except ValueError:\n"
        "    print('REJECTED')\n"
    )
    t0 = time.monotonic()
    try:
        py_proc = subprocess.run(
            [sys.executable, "-c", py_probe],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"Python FilterEngine ReDoS regex hung (>3.0s): {exc}")
    assert (time.monotonic() - t0) < 3.0
    assert py_proc.returncode == 0, f"Python ReDoS probe failed: {py_proc.stderr}"
    assert py_proc.stdout.strip() in ("REJECTED", "FALSE")

    nat_probe = (
        "from sutradb import NativeCollection\n"
        "col = NativeCollection('nat_redos', 4, 'cosine', None, False)\n"
        "col.insert([{'id': 'x', 'text': 'x', 'metadata': {'s': 'a' * 30 + '!'}}], [[1.0, 0.0, 0.0, 0.0]])\n"
        "try:\n"
        "    res = col.query(vector=[1.0, 0.0, 0.0, 0.0], filter={'s': {'$regex': '(a+)+$'}}, top_k=5)\n"
        "    print('EMPTY' if res == [] else f'HITS:{len(res)}')\n"
        "except ValueError:\n"
        "    print('REJECTED')\n"
    )
    t1 = time.monotonic()
    try:
        nat_proc = subprocess.run(
            [sys.executable, "-c", nat_probe],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"Rust NativeCollection ReDoS regex hung (>3.0s): {exc}")
    assert (time.monotonic() - t1) < 3.0
    assert nat_proc.returncode == 0, f"Rust ReDoS probe failed: {nat_proc.stderr}"
    assert nat_proc.stdout.strip() in ("REJECTED", "EMPTY")


def test_f4_boundary_haversine_antipodal_and_out_of_range_coords() -> None:
    """Mandatory H3: Antipodal points with float eps and out-of-range coords avoid math domain error."""
    for eps in (0.0, 1e-16, -1e-16, 1e-15):
        d_py = haversine_distance(37.7749, -122.4194, -37.7749, 57.5806 + eps)
        assert math.isfinite(d_py)
        assert abs(d_py - math.pi * 6_371_000.0) < 1000.0

        d_nat = native_haversine(37.7749, -122.4194, -37.7749, 57.5806 + eps)
        assert math.isfinite(d_nat)
        assert abs(d_nat - math.pi * 6_371_000.0) < 1000.0

    # Out-of-range latitude clamping
    d_clamp = haversine_distance(95.0, -185.0, -95.0, 185.0)
    assert math.isfinite(d_clamp)


def test_f4_boundary_unknown_filter_operator_rejection() -> None:
    """Unknown filter operators raise ValueError in Python FilterEngine without silent pass."""
    with pytest.raises(ValueError, match="Unknown filter operator"):
        FilterEngine.evaluate({"a": 1}, {"a": {"$bogus_op": 1}})


def test_f4_boundary_empty_and_non_matching_filters(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Empty filter returns all matches; impossible filter returns [] across exact and ANN paths."""
    vecs = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    py_col = py_collection_factory(name="py_empty_flt", dimension=4)
    py_col.insert([Document(id="d0", vector=vecs[0], metadata={"score": 10}), Document(id="d1", vector=vecs[1], metadata={"score": 20})])
    assert len(py_col.query(vector=vecs[0], filter={}, top_k=5)) == 2
    assert py_col.query(vector=vecs[0], filter={"score": {"$gt": 999999}}, top_k=5) == []

    nat_col = native_collection_factory(name="nat_empty_flt", dimension=4)
    nat_col.insert([{"id": "d0", "text": "", "metadata": {"score": 10}}, {"id": "d1", "text": "", "metadata": {"score": 20}}], vecs)
    assert len(nat_col.query(vector=vecs[0], filter={}, top_k=5)) == 2
    assert nat_col.query(vector=vecs[0], filter={"score": {"$gt": 999999}}, top_k=5) == []


# =============================================================================
# F5: Persistence, WAL & Crash Recovery Boundaries (6 tests)
# =============================================================================


def test_f5_boundary_same_batch_duplicate_ids(
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory C4: Duplicate IDs within the same insert batch succeed with count==1 and last winning."""
    py_col = py_collection_factory(name="py_dup_batch", dimension=4)
    count = py_col.insert([
        Document(id="dup", vector=[1.0, 0.0, 0.0, 0.0], text="first", metadata={"v": 1}),
        Document(id="dup", vector=[0.0, 1.0, 0.0, 0.0], text="second", metadata={"v": 2}),
    ])
    assert count == 1
    assert py_col.count() == 1
    assert py_col.get("dup").text == "second"
    assert py_col.get("dup").metadata["v"] == 2

    # Also test appending duplicate batch onto non-empty collection
    count2 = py_col.insert([
        Document(id="dup2", vector=[0.0, 0.0, 1.0, 0.0], text="a"),
        Document(id="dup2", vector=[0.0, 0.0, 0.0, 1.0], text="b"),
    ])
    assert count2 == 2
    assert py_col.get("dup2").text == "b"

    nat_col = native_collection_factory(name="nat_dup_batch", dimension=4)
    nat_count = nat_col.insert(
        [
            {"id": "dup", "text": "first", "metadata": {"v": 1}},
            {"id": "dup", "text": "second", "metadata": {"v": 2}},
        ],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    assert nat_count == 1
    assert nat_col.count() == 1
    assert nat_col.query(vector=[0.0, 1.0, 0.0, 0.0], top_k=1)[0]["text"] == "second"


def test_f5_boundary_atomic_batch_rollback_on_mid_batch_error(
    tmp_path: Path,
    py_collection_factory: Callable[..., Any],
    native_collection_factory: Callable[..., Any],
) -> None:
    """Mandatory H2: Mid-batch dimension error rolls back atomically (nothing added to RAM or WAL)."""
    py_path = tmp_path / "py_atomic.sutra"
    py_col = Collection(name="py_atomic", dimension=4, storage_path=py_path, enable_wal=True)
    py_col.insert(Document(id="base", vector=[1.0, 0.0, 0.0, 0.0], text="base"))
    with pytest.raises(ValueError):
        py_col.insert([
            Document(id="b1", vector=[0.0, 1.0, 0.0, 0.0], text="ok1"),
            Document(id="b2", vector=[0.0, 0.0, 1.0, 0.0], text="ok2"),
            Document(id="b3", vector=[1.0, 0.0], text="bad dim"),
        ])
    assert py_col.count() == 1
    assert py_col.get("b1") is None
    py_col.wal.close()

    # Reopen WAL to verify partial batch was NOT written to WAL
    py_reopened = Collection(name="py_atomic", dimension=4, storage_path=py_path, enable_wal=True)
    try:
        assert py_reopened.count() == 1
        assert py_reopened.get("b1") is None
    finally:
        if py_reopened.wal:
            py_reopened.wal.close()

    nat_path = str(tmp_path / "nat_atomic.sutra")
    nat_col = NativeCollection("nat_atomic", 4, "cosine", nat_path, True)
    nat_col.insert([{"id": "base", "text": "base", "metadata": {}}], [[1.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        nat_col.insert(
            [
                {"id": "b1", "text": "ok1", "metadata": {}},
                {"id": "b2", "text": "ok2", "metadata": {}},
                {"id": "b3", "text": "bad", "metadata": {}},
            ],
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0]],
        )
    assert nat_col.count() == 1


def test_f5_boundary_collection_name_path_traversal_rejection(tmp_path: Path) -> None:
    """Mandatory H3: Invalid/traversal collection names are rejected cleanly."""
    db = SutraDB(persist_directory=str(tmp_path / "sec_db"))
    bad_names = ["", "../escaped", "a/b", "a\\b", "a" * 129, "bad name", "col\x00name"]
    for bad in bad_names:
        with pytest.raises(ValueError):
            db.create_collection(bad, dimension=4)
        with pytest.raises((ValueError, KeyError)):
            db.get_collection(bad)


def test_f5_boundary_corrupt_sutra_crc_overflow_truncated(tmp_path: Path) -> None:
    """Mandatory H2: Corrupt CRC32, u32::MAX doc_count overflow, truncated JSON, and invalid quant_mode raise cleanly."""
    # 1. doc_count = 0xFFFFFFFF overflow
    overflow_file = tmp_path / "overflow.sutra"
    overflow_header = struct.pack(
        HEADER_FORMAT, HEADER_MAGIC_V3, 0xFFFFFFFF, 4, b"cosine".ljust(16, b"\x00"), 2, 0, b"\x00" * 20
    )
    overflow_file.write_bytes(overflow_header + b"[]")
    with pytest.raises(ValueError):
        BinaryStorage.read(overflow_file)
    nat = NativeCollection("ovf", 4, "cosine", str(overflow_file), False)
    with pytest.raises((ValueError, IOError)):
        nat.load(str(overflow_file))

    # 2. Truncated metadata JSON
    trunc_file = tmp_path / "trunc.sutra"
    vec_bytes = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32).tobytes()
    bad_meta = b'[{"id":"x"'
    crc = zlib.crc32(bad_meta, zlib.crc32(vec_bytes))
    trunc_header = struct.pack(
        HEADER_FORMAT, HEADER_MAGIC_V2, 1, 4, b"cosine".ljust(16, b"\x00"), len(bad_meta), crc, b"\x00" * 20
    )
    trunc_file.write_bytes(trunc_header + vec_bytes + bad_meta)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        BinaryStorage.read(trunc_file)
    with pytest.raises((ValueError, IOError)):
        nat.load(str(trunc_file))

    # 3. CRC32 mismatch (Collection::load is lazy; explicit verify_checksum detects payload CRC32 mismatch)
    bad_crc_file = tmp_path / "bad_crc.sutra"
    good_meta = b'[{"id":"x","text":""}]'
    bad_crc_header = struct.pack(
        HEADER_FORMAT, HEADER_MAGIC_V2, 1, 4, b"cosine".ljust(16, b"\x00"), len(good_meta), 0xDEADBEEF, b"\x00" * 20
    )
    bad_crc_file.write_bytes(bad_crc_header + vec_bytes + good_meta)
    with pytest.raises(ValueError):
        BinaryStorage.read(bad_crc_file)
    nat.load(str(bad_crc_file))
    with pytest.raises((ValueError, IOError)):
        nat.verify_checksum(str(bad_crc_file))

    # 4. Invalid quant_mode = 99 with valid header length and matching CRC32
    bad_qmode_file = tmp_path / "bad_qmode.sutra"
    qmode_crc = zlib.crc32(good_meta, zlib.crc32(vec_bytes))
    qmode_header = struct.pack(
        HEADER_FORMAT,
        HEADER_MAGIC_V3,
        1,
        4,
        b"cosine".ljust(16, b"\x00"),
        len(good_meta),
        qmode_crc,
        bytes([99]) + b"\x00" * 19,
    )
    bad_qmode_file.write_bytes(qmode_header + vec_bytes + good_meta)
    with pytest.raises(ValueError):
        BinaryStorage.read(bad_qmode_file)
    with pytest.raises(ValueError):
        nat.load(str(bad_qmode_file))


def test_f5_boundary_legacy_v2_v3_read_and_missing_suffix_load(tmp_path: Path) -> None:
    """Mandatory M5: Legacy V2/V3 .sutra files load cleanly and Collection.load() resolves missing .sutra suffix."""
    vecs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    meta = json.dumps([{"id": "leg1", "text": "legacy", "metadata": {}}]).encode("utf-8")
    crc_f32 = zlib.crc32(meta, zlib.crc32(vecs.tobytes()))
    v2_path = tmp_path / "legacy_v2.sutra"
    v2_path.write_bytes(
        struct.pack(HEADER_FORMAT, HEADER_MAGIC_V2, 1, 4, b"cosine".ljust(16, b"\x00"), len(meta), crc_f32, b"\x00" * 20)
        + vecs.tobytes()
        + meta
    )

    col_no_ext = Collection(name="leg", dimension=4)
    assert col_no_ext.load(tmp_path / "legacy_v2") == 1
    assert col_no_ext.get("leg1") is not None

    # V3 SQ8 legacy payload (quant_mode=1, no per-doc params)
    u8_data = np.array([[255, 0, 0, 0]], dtype=np.uint8).tobytes()
    crc_v3 = zlib.crc32(meta, zlib.crc32(u8_data))
    v3_path = tmp_path / "legacy_v3_sq8.sutra"
    extra = bytes([1]) + b"\x00" * 19
    v3_path.write_bytes(
        struct.pack(HEADER_FORMAT, HEADER_MAGIC_V3, 1, 4, b"cosine".ljust(16, b"\x00"), len(meta), crc_v3, extra)
        + u8_data
        + meta
    )
    read_vecs, read_docs, read_metric = BinaryStorage.read(v3_path)
    assert read_metric == "cosine"
    assert len(read_docs) == 1
    assert np.allclose(read_vecs[0], [1.0, 0.0, 0.0, 0.0], atol=0.01)


def test_f5_boundary_corrupt_wal_tail_recovery(tmp_path: Path) -> None:
    """Torn/truncated WAL frame tail heals without losing committed preceding entries."""
    py_path = tmp_path / "torn_wal.sutra"
    c1 = Collection(name="torn_wal", dimension=4, storage_path=py_path, enable_wal=True)
    c1.insert(Document(id="ok1", vector=[1.0, 0.0, 0.0, 0.0], text="valid entry"))
    c1.wal.close()

    wal_file = tmp_path / "torn_wal.wal"
    with open(wal_file, "ab") as f:
        f.write(b"\x10\x00\x00\x00\xaa\xbb\xcc\xddpartial")

    c2 = Collection(name="torn_wal", dimension=4, storage_path=py_path, enable_wal=True)
    try:
        assert c2.count() == 1
        assert c2.get("ok1") is not None
    finally:
        if c2.wal:
            c2.wal.close()


# =============================================================================
# F6: Concurrency & Distributed Cluster Boundaries (5 tests)
# =============================================================================


def test_f6_boundary_pynative_concurrent_readers_writers_barrier(native_collection_factory: Callable[..., Any]) -> None:
    """Mandatory C9: Synchronized barrier R/W threads on PyNativeCollection trigger zero borrow panics."""
    col = native_collection_factory(name="c9_stress", dimension=4, metric="cosine")
    col.insert(
        [{"id": f"init_{i}", "text": f"init {i}", "metadata": {"n": i}} for i in range(10)],
        [[1.0, 0.0, 0.0, 0.0] for _ in range(10)],
    )

    barrier = threading.Barrier(6)
    errors: List[BaseException] = []

    def reader_worker() -> None:
        try:
            barrier.wait(timeout=5.0)
            for _ in range(25):
                res = col.query(vector=[1.0, 0.0, 0.0, 0.0], text="init", top_k=5)
                assert isinstance(res, list)
        except BaseException as exc:
            errors.append(exc)

    def writer_worker(wid: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for step in range(10):
                doc_id = f"w{wid}_{step}"
                col.insert([{"id": doc_id, "text": "write", "metadata": {"wid": wid}}], [[0.0, 1.0, 0.0, 0.0]])
                col.update_metadata(doc_id, {"wid": wid, "updated": True})
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader_worker) for _ in range(3)] + [
        threading.Thread(target=writer_worker, args=(i,)) for i in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)
        assert not t.is_alive()

    assert errors == []
    assert col.count() == 40


def test_f6_boundary_coordinator_excluded_from_hash_ring() -> None:
    """Mandatory H4: Cluster coordinator node_id is excluded from shard routing (0 keys routed to coordinator)."""
    cluster = DistributedCluster(node_id="cluster-coordinator", vnodes_per_node=64)
    dcol = cluster.create_collection("sharded", dimension=4, shard_count=3)

    ring_nodes = dcol.hash_ring.get_nodes()
    assert "cluster-coordinator" not in ring_nodes
    for i in range(200):
        key = f"key_{i}"
        assert dcol.hash_ring.get_node(key) != "cluster-coordinator"
        assert "cluster-coordinator" not in dcol.hash_ring.get_preference_list(key, 5)


def test_f6_boundary_hash_ring_empty_single_and_removal() -> None:
    """ConsistentHashRing handles empty ring, single node, duplicate add, and node removal."""
    ring = ConsistentHashRing(vnodes_per_node=16)
    assert ring.get_node("any") is None
    assert ring.get_preference_list("any", 3) == []
    assert ring.remove_node("missing") is False

    ring.add_node("node-a")
    ring.add_node("node-a")
    assert ring.node_count() == 1
    assert ring.get_node("k1") == "node-a"
    assert ring.get_preference_list("k1", 5) == ["node-a"]

    ring.add_node("node-b")
    assert ring.remove_node("node-a") is True
    assert ring.node_count() == 1
    assert ring.get_node("k1") == "node-b"


def test_f6_boundary_distributed_collection_empty_and_zero_topk() -> None:
    """DistributedCollection handles empty queries, top_k=0, and extreme weights cleanly."""
    dcol = DistributedCollection(name="dist_empty", dimension=4, shard_count=3)
    assert dcol.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=5) == []

    dcol.insert(Document(id="d1", vector=[1.0, 0.0, 0.0, 0.0], text="hello", sparse_vector={1: 1.0}))
    assert dcol.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=0) == []
    res = dcol.query(vector=[1.0, 0.0, 0.0, 0.0], sparse_vector={1: 1.0}, sparse_weight=100.0, colbert_weight=0.0, top_k=5)
    assert len(res) == 1 and math.isfinite(res[0].score)


def test_f6_boundary_native_concurrent_delete_compact_stress(native_collection_factory: Callable[..., Any]) -> None:
    """Concurrent query + delete + metadata update threads followed by compact leave consistent state."""
    col = native_collection_factory(name="nat_del_stress", dimension=4)
    col.insert(
        [{"id": f"id_{i}", "text": f"stress {i}", "metadata": {"i": i}} for i in range(40)],
        [[1.0, 0.0, 0.0, 0.0] for _ in range(40)],
    )

    barrier = threading.Barrier(3)
    errors: List[BaseException] = []

    def deleter() -> None:
        try:
            barrier.wait(timeout=5.0)
            for i in range(0, 20, 2):
                col.delete([f"id_{i}"])
        except BaseException as exc:
            errors.append(exc)

    def updater() -> None:
        try:
            barrier.wait(timeout=5.0)
            for i in range(20, 40):
                col.update_metadata(f"id_{i}", {"i": i, "touched": True})
        except BaseException as exc:
            errors.append(exc)

    def querier() -> None:
        try:
            barrier.wait(timeout=5.0)
            for _ in range(20):
                col.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=deleter), threading.Thread(target=updater), threading.Thread(target=querier)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert errors == []
    col.compact()
    assert col.count() == 30
    assert col.stats()["tombstones"] == 0


# =============================================================================
# F7: REST API, Web UI & Playground Boundaries (6 tests)
# =============================================================================


def test_f7_boundary_rest_pagination_topk_and_schema_bounds(rest_client: Any) -> None:
    """Mandatory H3: Negative pagination limit/offset, oversized top_k, and bad schemas return 400/422."""
    assert rest_client.post("/collections", json={"name": "bounds_col", "dimension": 0}).status_code in (400, 422)
    assert rest_client.post("/collections", json={"name": "bounds_col", "dimension": 4, "metric": "invalid_metric"}).status_code in (400, 422)
    assert rest_client.post("/collections", json={"name": "bounds_col", "dimension": 4}).status_code == 201

    assert rest_client.get("/collections/bounds_col/documents?limit=-1").status_code in (400, 422)
    assert rest_client.get("/collections/bounds_col/documents?offset=-1").status_code in (400, 422)
    assert rest_client.get("/collections/bounds_col/documents?limit=100000").status_code in (400, 422)
    assert rest_client.post("/collections/bounds_col/query", json={"vector": [1, 0, 0, 0], "top_k": 0}).status_code in (400, 422)
    assert rest_client.post("/collections/bounds_col/query", json={"vector": [1, 0, 0, 0], "top_k": 100000}).status_code in (400, 422)


def test_f7_boundary_rest_invalid_collection_name_and_payload_limit(rest_client: Any) -> None:
    """Mandatory H3: Invalid collection names rejected, >10MB returns 413, CORS never pairs * with credentials=true."""
    for bad in ("bad name", "a" * 129, "../evil", "a/b"):
        r = rest_client.post("/collections", json={"name": bad, "dimension": 4})
        assert r.status_code in (400, 422), f"Accepted invalid collection name {bad!r}: {r.status_code}"

    # >10MB payload returns 413
    oversized = b'{"documents": [{"text": "' + (b"x" * (10 * 1024 * 1024 + 64)) + b'"}]}'
    r_big = rest_client.post(
        "/collections/nonexistent/documents",
        content=oversized,
        headers={"Content-Type": "application/json"},
    )
    assert r_big.status_code == 413

    # CORS check: must not return Access-Control-Allow-Origin: * alongside Access-Control-Allow-Credentials: true
    r_cors = rest_client.options(
        "/collections",
        headers={"Origin": "https://untrusted.example.com", "Access-Control-Request-Method": "GET"},
    )
    acao = r_cors.headers.get("access-control-allow-origin", "")
    acac = r_cors.headers.get("access-control-allow-credentials", "").lower()
    assert not (acao == "*" and acac == "true"), "Insecure CORS wildcard + credentials=true detected"


def test_f7_boundary_ui_xss_escaping_and_logic_fixes(rest_client: Any) -> None:
    """Mandatory H3 & M6: /ui escapes r.id, doc.id, col.name safely, fixes switchTab(event), slider, and 2D PCA."""
    html = rest_client.get("/ui").text

    # 1. r.id must be escaped in renderResults (no raw `${r.id}`)
    assert "${r.id}" not in html, "Unescaped ${r.id} XSS sink in /ui renderResults"
    # 2. col.name must not appear unescaped in inline HTML/onclick handlers
    assert "<b>${col.name}</b>" not in html, "Unescaped ${col.name} XSS sink in /ui table"
    assert "triggerHNSW('${col.name}')" not in html, "Unescaped single-quote JS sink in /ui onclick"
    assert "openPatchModal('${escapeHtml(doc.id)}'" not in html, "Single-quote JS breakout in /ui openPatchModal"
    # 3. switchTab must pass event explicitly rather than implicit global event
    assert not re.search(r"function\s+switchTab\s*\(\s*tabId\s*\)", html), "switchTab must accept event parameter"
    # 4. Hybrid slider must NOT be inverted (v === 0.0 must NOT claim 100% Dense)
    assert not re.search(r"v\s*===\s*0(\.0)?[^;\n]*100%\s*Dense", html), "Hybrid slider label is inverted"
    # 5. Real 2D PCA instead of Math.cos(i) / Math.sin(i) stub
    assert "Math.cos(i)" not in html and "Math.sin(i)" not in html, "Fake trig projection stub still present in /ui"


def test_f7_boundary_rest_linear_fusion_bq_returns_200_finite(rest_client: Any) -> None:
    """Mandatory C5: REST query with alpha=0.5 on 60-doc BQ collection (top_k=5) returns HTTP 200 with finite scores."""
    assert rest_client.post("/collections", json={"name": "bq_rest", "dimension": 16}).status_code == 201
    rng = np.random.default_rng(705)
    # 60 docs with top_k=5 ensures Stage-1 rerank window (max(5*4, 32) = 32 < 60) leaves >= 28 docs at -inf
    docs = [
        {"id": f"doc_{i}", "vector": rng.normal(size=16).tolist(), "text": f"vector search doc {i}"}
        for i in range(60)
    ]
    assert rest_client.post("/collections/bq_rest/documents", json={"documents": docs}).status_code == 200
    assert rest_client.post("/collections/bq_rest/bq").status_code == 200

    r_q = rest_client.post(
        "/collections/bq_rest/query",
        json={"vector": docs[0]["vector"], "text": "vector search", "alpha": 0.5, "top_k": 5, "hybrid": True},
    )
    assert r_q.status_code == 200
    results = r_q.json()["results"]
    assert len(results) == 5
    assert all(math.isfinite(item["score"]) for item in results)


def test_f7_boundary_playground_path_traversal_blocked(playground_server: str) -> None:
    """Mandatory H3: Raw un-normalized path traversal requests to Playground return 403 or 404, never 200."""
    parsed = urlparse(playground_server)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    assert port is not None

    for raw_path in ("/../Cargo.toml", "/../../pyproject.toml", "/%2e%2e/Cargo.toml", "/%2e%2e/%2e%2e/pyproject.toml"):
        with socket.create_connection((host, port), timeout=5.0) as sock:
            req = f"GET {raw_path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n".encode("ascii")
            sock.sendall(req)
            chunks: List[bytes] = []
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
        raw_resp = b"".join(chunks).decode("utf-8", errors="replace")
        status_line = raw_resp.splitlines()[0]
        status_code = int(status_line.split()[1])
        assert status_code in (403, 404), f"Raw traversal {raw_path} returned {status_line}"


def test_f7_boundary_playground_xss_and_bad_search_inputs(playground_server: str) -> None:
    """Mandatory H3: Playground HTML escapes rendered fields & regex meta-chars; bad /api/search returns 400."""
    html = httpx.get(f"{playground_server}/", timeout=5.0).text
    assert "${meta.category}" not in html, "Unescaped ${meta.category} in playground/index.html"
    assert "${err.message}" not in html, "Unescaped ${err.message} in playground/index.html"
    assert "new RegExp(`(${term})`" not in html, "Unescaped RegExp interpolation in playground highlight()"

    r_empty = httpx.post(f"{playground_server}/api/search", json={"text": "   "}, timeout=5.0)
    assert r_empty.status_code == 400

    r_bad_json = httpx.post(
        f"{playground_server}/api/search",
        content=b"{broken",
        headers={"Content-Type": "application/json"},
        timeout=5.0,
    )
    assert r_bad_json.status_code == 400


# =============================================================================
# F8: gRPC Server & Container Boundaries (5 tests)
# =============================================================================


def test_f8_boundary_server_invalid_collection_name_and_traversal(sutradb_server_factory: Callable[..., Any]) -> None:
    """Mandatory H3: sutradb-server rejects invalid/traversal collection names cleanly."""
    srv = sutradb_server_factory()
    for bad in ("", "../evil", "a/b", "a" * 129, "bad name"):
        resp = httpx.post(
            f"{srv.bridge_url}/collections",
            json={"name": bad, "dimension": 4, "metric": "cosine"},
            timeout=5.0,
        )
        assert resp.status_code in (400, 422), f"Accepted invalid name {bad!r}: {resp.status_code}"


def test_f8_boundary_server_malformed_sparse_and_dim_mismatch_no_crash(
    sutradb_server_factory: Callable[..., Any],
) -> None:
    """Mandatory H2: Malformed SparseVector and dimension mismatch return 4xx without crashing server."""
    srv = sutradb_server_factory()
    httpx.post(f"{srv.bridge_url}/collections", json={"name": "guard_col", "dimension": 4, "metric": "cosine"}, timeout=5.0)

    r_dim = httpx.post(
        f"{srv.bridge_url}/collections/guard_col/insert",
        json={"documents": [{"id": "bad_dim", "vector": [1.0, 0.0], "text": "bad"}]},
        timeout=5.0,
    )
    assert r_dim.status_code in (400, 422)

    r_sparse = httpx.post(
        f"{srv.bridge_url}/collections/guard_col/insert",
        json={
            "documents": [
                {
                    "id": "bad_sp",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "text": "bad sparse",
                    "sparse_vector": {"indices": [1, 2], "values": [1.0]},
                }
            ]
        },
        timeout=5.0,
    )
    assert r_sparse.status_code in (400, 422)
    assert httpx.get(f"{srv.bridge_url}/health", timeout=5.0).status_code == 200


def test_f8_boundary_server_version_5_5_0_and_wal_records(sutradb_server_factory: Callable[..., Any]) -> None:
    """Mandatory L1: sutradb-server --version and startup banner report 5.5.0 and accurate wal_records."""
    srv = sutradb_server_factory()
    ver_proc = subprocess.run([str(srv.binary), "--version"], capture_output=True, text=True, timeout=5.0, check=False)
    assert ver_proc.returncode == 0
    assert "5.5.0" in (ver_proc.stdout + ver_proc.stderr)

    httpx.post(
        f"{srv.bridge_url}/collections",
        json={"name": "wal_stat_col", "dimension": 4, "metric": "cosine", "enable_wal": True},
        timeout=5.0,
    )
    httpx.post(
        f"{srv.bridge_url}/collections/wal_stat_col/insert",
        json={
            "documents": [
                {"id": "w1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "one"},
                {"id": "w2", "vector": [0.0, 1.0, 0.0, 0.0], "text": "two"},
            ]
        },
        timeout=5.0,
    )
    st1 = httpx.get(f"{srv.bridge_url}/collections/wal_stat_col/stats", timeout=5.0).json()
    assert st1["wal_records"] >= 1
    httpx.post(f"{srv.bridge_url}/collections/wal_stat_col/save", timeout=5.0)
    st2 = httpx.get(f"{srv.bridge_url}/collections/wal_stat_col/stats", timeout=5.0).json()
    assert st2["wal_records"] == 0

    stdout, _ = srv.stop()
    assert "v5.5.0" in stdout, f"Expected v5.5.0 banner in sutradb-server stdout, got:\n{stdout}"


def test_f8_boundary_sutra_all_mode_single_owner_bridge_consistency(
    sutra_all_mode_stack: tuple[Any, Any],
) -> None:
    """Mandatory C10: In SUTRA_MODE=all, writes on bridge are immediately visible via REST and vice versa."""
    srv, rest_client = sutra_all_mode_stack

    # 1. Create + insert via sutradb-server bridge -> immediately visible on FastAPI REST
    assert httpx.post(
        f"{srv.bridge_url}/collections",
        json={"name": "unified_col", "dimension": 4, "metric": "cosine", "enable_wal": True},
        timeout=5.0,
    ).status_code in (200, 201)
    assert httpx.post(
        f"{srv.bridge_url}/collections/unified_col/insert",
        json={"documents": [{"id": "from_grpc", "vector": [1.0, 0.0, 0.0, 0.0], "text": "grpc write"}]},
        timeout=5.0,
    ).status_code == 200

    rest_q = rest_client.post("/collections/unified_col/query", json={"vector": [1.0, 0.0, 0.0, 0.0], "top_k": 5})
    assert rest_q.status_code == 200
    assert [r["id"] for r in rest_q.json()["results"]] == ["from_grpc"]

    # 2. Insert via FastAPI REST -> immediately visible on sutradb-server bridge
    assert rest_client.post(
        "/collections/unified_col/documents",
        json={"documents": [{"id": "from_rest", "vector": [0.0, 1.0, 0.0, 0.0], "text": "rest write"}]},
    ).status_code == 200
    bridge_q = httpx.post(
        f"{srv.bridge_url}/collections/unified_col/query",
        json={"vector": [0.0, 1.0, 0.0, 0.0], "top_k": 5},
        timeout=5.0,
    )
    assert bridge_q.status_code == 200
    assert bridge_q.json()["results"][0]["id"] == "from_rest"

    # 3. Delete via FastAPI REST -> gone on bridge
    assert rest_client.delete("/collections/unified_col/documents/from_grpc").status_code == 200
    bridge_st = httpx.get(f"{srv.bridge_url}/collections/unified_col/stats", timeout=5.0).json()
    assert bridge_st["count"] == 1


def test_f8_boundary_healthcheck_fails_when_port_down_and_optional_docker() -> None:
    """sutra-healthcheck.py exits non-zero when required gRPC/REST ports are unreachable."""
    dead_grpc = get_free_port()
    dead_rest = get_free_port()

    hc_fail = subprocess.run(
        [sys.executable, str(REPO_ROOT / "sutra-healthcheck.py")],
        env={
            "PATH": "/usr/bin:/bin",
            "SUTRA_MODE": "all",
            "SUTRA_GRPC_PORT": str(dead_grpc),
            "SUTRA_REST_PORT": str(dead_rest),
        },
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    assert hc_fail.returncode != 0

    if os.getenv("SUTRA_E2E_DOCKER") == "1":
        build_proc = subprocess.run(
            ["docker", "build", "-t", "sutradb-e2e-test:5.5.0", "."],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600.0,
            check=False,
        )
        assert build_proc.returncode == 0, f"docker build failed:\n{build_proc.stderr}"
