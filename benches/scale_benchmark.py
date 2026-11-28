#!/usr/bin/env python3
"""
SutraDB v6 Out-of-L3-Cache Scale Benchmark (100,000 x 768d)
==============================================================
Evaluates ingestion throughput (`insert_numpy`), Flat AVX2 SIMD, Parallel HNSW
(unfiltered + 1%-selectivity filtered), disk-backed mmap-evicted SQ8 (4x) &
1-Bit BQ (32x), and `.sutra` + `.sutra.idx` cold-restart speed on real
multi-hundred-megabyte workloads.
"""

import argparse
from dataclasses import asdict, dataclass
import gc
import importlib.util
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from sutradb import Collection, Metric
from sutradb._sutradb import detect_simd


@dataclass
class ScaleRow:
    engine: str
    variant: str
    docs_count: int
    dimension: int
    ingest_time_sec: float
    ingest_docs_per_sec: float
    build_only_docs_per_sec: Optional[float]
    ram_mb: float
    rss_delta_mb: float
    compression: str
    p50_ms: float
    p99_ms: float
    qps: float
    recall_at_10: float


def measure_rss_mb() -> float:
    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)


def generate_clustered_corpus(
    n_docs: int,
    dim: int,
    n_queries: int,
    seed: int = 42,
    n_clusters: int = 250,
    noise_std: float = 0.035,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[Dict[str, Any]], np.ndarray]:
    """
    Generates normalized 768d unit-sphere clustered embeddings (modeling semantic
    embedding geometry where `noise_std * sqrt(dim) ~ 0.97`, top-10 cosine ~0.50-0.65)
    plus 100 uniform `tenant_id` buckets (`tenant_00`..`tenant_99`).
    """
    rng = np.random.default_rng(seed)

    centroids = rng.standard_normal((n_clusters, dim), dtype=np.float32)
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)

    cluster_assign = rng.integers(0, n_clusters, size=n_docs)
    vectors = centroids[cluster_assign] + noise_std * rng.standard_normal((n_docs, dim), dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    q_clusters = rng.integers(0, n_clusters, size=n_queries)
    query_vectors = centroids[q_clusters] + noise_std * rng.standard_normal((n_queries, dim), dtype=np.float32)
    query_vectors /= np.maximum(np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-9)
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

    ids = [f"doc_{i:06d}" for i in range(n_docs)]
    texts = [f"document {i} tenant_{i % 100:02d} topic_{int(cluster_assign[i])}" for i in range(n_docs)]
    metadatas = [{"tenant_id": f"tenant_{i % 100:02d}"} for i in range(n_docs)]
    tenant_mask = (np.arange(n_docs) % 100) == 0

    return vectors, query_vectors, ids, texts, metadatas, tenant_mask


def compute_exact_topk(
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    top_k: int,
    subset_indices: Optional[np.ndarray] = None,
) -> List[set]:
    if subset_indices is not None:
        sub_vecs = doc_vectors[subset_indices]
        sims = query_vectors @ sub_vecs.T
        truth = []
        for row in sims:
            top_local = np.argpartition(row, -top_k)[-top_k:]
            global_idx = subset_indices[top_local]
            truth.append({f"doc_{int(idx):06d}" for idx in global_idx})
        return truth

    sims = query_vectors @ doc_vectors.T
    truth = []
    for row in sims:
        top_idx = np.argpartition(row, -top_k)[-top_k:]
        truth.append({f"doc_{int(idx):06d}" for idx in top_idx})
    return truth


def eval_queries(
    query_fn,
    query_vectors: np.ndarray,
    truth_sets: List[set],
    warmup: int = 5,
    top_k: int = 10,
) -> Tuple[float, float, float, float]:
    for i in range(min(warmup, len(query_vectors))):
        _ = query_fn(query_vectors[i])

    latencies_ms: List[float] = []
    hits = 0
    for q_vec, truth in zip(query_vectors, truth_sets):
        t0 = time.perf_counter()
        pred_ids = query_fn(q_vec)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        hits += len(set(pred_ids[:top_k]) & truth)

    p50 = float(np.percentile(latencies_ms, 50))
    p99 = float(np.percentile(latencies_ms, 99))
    qps = 1000.0 / float(np.mean(latencies_ms))
    recall = hits / max(len(query_vectors) * top_k, 1)
    return p50, p99, qps, recall


def check_optional_competitors(
    vectors: np.ndarray,
    query_vectors: np.ndarray,
    ids: List[str],
    metadatas: List[Dict[str, Any]],
    gt_unfiltered: List[set],
    top_k: int,
) -> Tuple[List[ScaleRow], List[str]]:
    extra_rows: List[ScaleRow] = []
    notes: List[str] = []
    n_docs, dim = vectors.shape
    raw_mb = (n_docs * dim * 4.0) / (1024.0 * 1024.0)

    # 1. qdrant-client (local in-memory)
    if importlib.util.find_spec("qdrant_client") is not None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        rss0 = measure_rss_mb()
        qc = QdrantClient(":memory:")
        qc.create_collection("bench", vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        t0 = time.perf_counter()
        batch_sz = 5000
        for s in range(0, n_docs, batch_sz):
            e = min(s + batch_sz, n_docs)
            pts = [
                PointStruct(id=i, vector=vectors[i].tolist(), payload=metadatas[i])
                for i in range(s, e)
            ]
            qc.upsert("bench", points=pts)
        ingest_t = time.perf_counter() - t0
        rss1 = measure_rss_mb()

        def q_qdrant(qv):
            hits = qc.query_points("bench", query=qv.tolist(), limit=top_k).points
            return [ids[int(h.id)] for h in hits]

        p50, p99, qps, rec = eval_queries(q_qdrant, query_vectors, gt_unfiltered, top_k=top_k)
        extra_rows.append(
            ScaleRow(
                engine="Qdrant",
                variant="In-Memory (:memory:)",
                docs_count=n_docs,
                dimension=dim,
                ingest_time_sec=ingest_t,
                ingest_docs_per_sec=n_docs / max(ingest_t, 1e-6),
                build_only_docs_per_sec=None,
                ram_mb=raw_mb,
                rss_delta_mb=max(rss1 - rss0, 0.0),
                compression="1x",
                p50_ms=p50,
                p99_ms=p99,
                qps=qps,
                recall_at_10=rec,
            )
        )
    else:
        notes.append("`qdrant-client`: skipped (not installed in `.venv`).")

    # 2. chromadb
    if importlib.util.find_spec("chromadb") is not None:
        notes.append("`chromadb`: detected.")
    else:
        notes.append("`chromadb`: skipped (not installed in `.venv`).")

    # 3. lancedb
    if importlib.util.find_spec("lancedb") is not None:
        notes.append("`lancedb`: detected.")
    else:
        notes.append("`lancedb`: skipped (not installed in `.venv`).")

    return extra_rows, notes


def run_scale_benchmark(
    n_docs: int,
    dim: int,
    n_queries: int,
    seed: int,
    hnsw_m: int,
    hnsw_efc: int,
    hnsw_efs: int,
    noise_std: float = 0.035,
    top_k: int = 10,
) -> Dict[str, Any]:
    wall_t0 = time.perf_counter()
    print(f"Generating {n_docs:,} x {dim}d normalized clustered embeddings (seed={seed}, noise_std={noise_std})...")
    vectors, query_vectors, ids, texts, metadatas, tenant_mask = generate_clustered_corpus(
        n_docs=n_docs,
        dim=dim,
        n_queries=n_queries,
        seed=seed,
        noise_std=noise_std,
    )
    tenant00_indices = np.flatnonzero(tenant_mask)

    print(f"Computing NumPy exact ground truth (unfiltered & 1% filter = {len(tenant00_indices):,} docs)...")
    t_gt = time.perf_counter()
    gt_unfiltered = compute_exact_topk(vectors, query_vectors, top_k=top_k)
    gt_filtered_1pct = compute_exact_topk(vectors, query_vectors, top_k=top_k, subset_indices=tenant00_indices)
    print(f"Ground truth computed in {time.perf_counter() - t_gt:.2f}s")

    rows: List[ScaleRow] = []
    raw_f32_mb = (n_docs * dim * 4.0) / (1024.0 * 1024.0)

    # 0. NumPy Brute-Force Reference
    print("\n[1/5] Running NumPy Brute-Force Reference...")
    p50, p99, qps, rec = eval_queries(
        lambda qv: [
            f"doc_{int(idx):06d}"
            for idx in np.argpartition(vectors @ qv, -top_k)[-top_k:][
                np.argsort(-(vectors @ qv)[np.argpartition(vectors @ qv, -top_k)[-top_k:]])
            ]
        ],
        query_vectors,
        gt_unfiltered,
        top_k=top_k,
    )
    rows.append(
        ScaleRow(
            engine="NumPy",
            variant="Exact Brute-Force (GEMV)",
            docs_count=n_docs,
            dimension=dim,
            ingest_time_sec=0.0,
            ingest_docs_per_sec=0.0,
            build_only_docs_per_sec=None,
            ram_mb=raw_f32_mb,
            rss_delta_mb=raw_f32_mb,
            compression="1x",
            p50_ms=p50,
            p99_ms=p99,
            qps=qps,
            recall_at_10=rec,
        )
    )

    # 1. SutraDB Flat + HNSW + Cold Restart on same 100K x 768d Collection
    print("\n[2/5] Running SutraDB Flat (AVX2 SIMD) via insert_numpy...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        snap_path = Path(tmp_dir) / "scale_hnsw.sutra"
        rss0 = measure_rss_mb()
        col = Collection("scale_bench", dimension=dim, metric=Metric.COSINE, storage_path=snap_path)

        t0 = time.perf_counter()
        col._native.insert_numpy(
            vectors=vectors,
            ids=ids,
            texts=texts,
            metadatas=metadatas,
        )
        ingest_t = time.perf_counter() - t0
        rss1 = measure_rss_mb()
        st_flat = col.stats()
        flat_ram_mb = float(st_flat.get("vector_memory_mb", raw_f32_mb))

        p50, p99, qps, rec = eval_queries(
            lambda qv: [r.id for r in col.query(vector=qv, top_k=top_k, hybrid=False, use_ann=False)],
            query_vectors,
            gt_unfiltered,
            top_k=top_k,
        )
        rows.append(
            ScaleRow(
                engine="SutraDB",
                variant="Flat (Rust AVX2 SIMD)",
                docs_count=n_docs,
                dimension=dim,
                ingest_time_sec=ingest_t,
                ingest_docs_per_sec=n_docs / max(ingest_t, 1e-6),
                build_only_docs_per_sec=None,
                ram_mb=flat_ram_mb,
                rss_delta_mb=max(rss1 - rss0, flat_ram_mb),
                compression="1x",
                p50_ms=p50,
                p99_ms=p99,
                qps=qps,
                recall_at_10=rec,
            )
        )

        # 2. Parallel HNSW
        print(f"\n[3/5] Building Parallel HNSW (m={hnsw_m}, ef_c={hnsw_efc}, ef_s={hnsw_efs})...")
        rss_h0 = measure_rss_mb()
        t_h0 = time.perf_counter()
        col.build_hnsw(m=hnsw_m, ef_construction=hnsw_efc, ef_search=hnsw_efs)
        hnsw_build_t = time.perf_counter() - t_h0
        rss_h1 = measure_rss_mb()
        st_hnsw = col.stats()
        hnsw_graph_mb = float(st_hnsw.get("hnsw", {}).get("memory_mb", 0.0))
        hnsw_total_ram = flat_ram_mb + hnsw_graph_mb

        pre_restart_top10: List[List[str]] = []

        def q_hnsw_unfiltered(qv):
            res = [r.id for r in col.query(vector=qv, top_k=top_k, hybrid=False, use_ann=True)]
            if len(pre_restart_top10) < 5:
                pre_restart_top10.append(res)
            return res

        p50, p99, qps, rec = eval_queries(q_hnsw_unfiltered, query_vectors, gt_unfiltered, top_k=top_k)
        rows.append(
            ScaleRow(
                engine="SutraDB",
                variant=f"HNSW Unfiltered (m={hnsw_m})",
                docs_count=n_docs,
                dimension=dim,
                ingest_time_sec=ingest_t + hnsw_build_t,
                ingest_docs_per_sec=n_docs / max(ingest_t + hnsw_build_t, 1e-6),
                build_only_docs_per_sec=n_docs / max(hnsw_build_t, 1e-6),
                ram_mb=hnsw_total_ram,
                rss_delta_mb=max(rss_h1 - rss0, hnsw_total_ram),
                compression="1x",
                p50_ms=p50,
                p99_ms=p99,
                qps=qps,
                recall_at_10=rec,
            )
        )

        # Filtered HNSW (1% selectivity: tenant_id == "tenant_00")
        flt = {"tenant_id": "tenant_00"}
        p50_f, p99_f, qps_f, rec_f = eval_queries(
            lambda qv: [r.id for r in col.query(vector=qv, filter=flt, top_k=top_k, hybrid=False, use_ann=True)],
            query_vectors,
            gt_filtered_1pct,
            top_k=top_k,
        )
        rows.append(
            ScaleRow(
                engine="SutraDB",
                variant="HNSW 1%-Filtered (ACORN)",
                docs_count=n_docs,
                dimension=dim,
                ingest_time_sec=ingest_t + hnsw_build_t,
                ingest_docs_per_sec=n_docs / max(ingest_t + hnsw_build_t, 1e-6),
                build_only_docs_per_sec=n_docs / max(hnsw_build_t, 1e-6),
                ram_mb=hnsw_total_ram,
                rss_delta_mb=max(rss_h1 - rss0, hnsw_total_ram),
                compression="1x",
                p50_ms=p50_f,
                p99_ms=p99_f,
                qps=qps_f,
                recall_at_10=rec_f,
            )
        )

        # Cold Restart (.sutra + .sutra.idx)
        print("\n[4/5] Benchmarking Cold Restart (.sutra + .sutra.idx save/load)...")
        t_save0 = time.perf_counter()
        saved_bytes = col.save()
        save_ms = (time.perf_counter() - t_save0) * 1000.0
        idx_bytes = (snap_path.parent / f"{snap_path.name}.idx").stat().st_size

        del col
        gc.collect()

        t_load0 = time.perf_counter()
        reloaded = Collection("scale_bench", dimension=dim, metric=Metric.COSINE, storage_path=snap_path)
        reloaded.load()
        cold_load_ms = (time.perf_counter() - t_load0) * 1000.0
        st_reload = reloaded.stats()
        assert st_reload.get("has_hnsw") is True, "Cold reload did not restore HNSW index!"

        parity_ok = True
        for i in range(min(5, len(query_vectors))):
            post_ids = [r.id for r in reloaded.query(vector=query_vectors[i], top_k=top_k, hybrid=False, use_ann=True)]
            if post_ids != pre_restart_top10[i]:
                parity_ok = False
                break
        assert parity_ok, "Post-reload HNSW top-10 diverged from pre-save results!"
        del reloaded
        gc.collect()

    # 3. Disk-backed FP32-evicted SQ8 (4x) & 1-Bit BQ (32x)
    print("\n[5/5] Running Disk-Backed Evicted SQ8 (4x) & 1-Bit BQ (32x)...")
    with tempfile.TemporaryDirectory() as q_dir:
        sq8_path = Path(q_dir) / "scale_sq8.sutra"
        col_sq8 = Collection("scale_sq8", dimension=dim, metric=Metric.COSINE, storage_path=sq8_path)
        t0 = time.perf_counter()
        col_sq8._native.insert_numpy(vectors=vectors, ids=ids, texts=texts, metadatas=metadatas)
        col_sq8.save()
        col_sq8.enable_sq8()  # triggers maybe_evict_fp32_arena() because mmap_storage is active
        sq8_t = time.perf_counter() - t0
        st_sq8 = col_sq8.stats()
        sq8_ram_mb = float(st_sq8["vector_memory_mb"])
        assert sq8_ram_mb < raw_f32_mb * 0.35, f"Expected FP32 eviction in SQ8, got {sq8_ram_mb:.1f} MB"

        p50, p99, qps, rec = eval_queries(
            lambda qv: [r.id for r in col_sq8.query(vector=qv, top_k=top_k, hybrid=False, use_ann=False)],
            query_vectors,
            gt_unfiltered,
            top_k=top_k,
        )
        rows.append(
            ScaleRow(
                engine="SutraDB",
                variant="SQ8 (Mmap-Evicted 4x)",
                docs_count=n_docs,
                dimension=dim,
                ingest_time_sec=sq8_t,
                ingest_docs_per_sec=n_docs / max(sq8_t, 1e-6),
                build_only_docs_per_sec=None,
                ram_mb=sq8_ram_mb,
                rss_delta_mb=sq8_ram_mb,
                compression="4.0x",
                p50_ms=p50,
                p99_ms=p99,
                qps=qps,
                recall_at_10=rec,
            )
        )
        del col_sq8
        gc.collect()

        bq_path = Path(q_dir) / "scale_bq.sutra"
        col_bq = Collection("scale_bq", dimension=dim, metric=Metric.COSINE, storage_path=bq_path)
        t0 = time.perf_counter()
        col_bq._native.insert_numpy(vectors=vectors, ids=ids, texts=texts, metadatas=metadatas)
        col_bq.save()
        col_bq.enable_bq()  # triggers maybe_evict_fp32_arena() + disk-backed Stage-2 reranking
        bq_t = time.perf_counter() - t0
        st_bq = col_bq.stats()
        bq_ram_mb = float(st_bq["vector_memory_mb"])
        assert bq_ram_mb < raw_f32_mb * 0.08, f"Expected FP32 eviction in BQ, got {bq_ram_mb:.1f} MB"

        p50, p99, qps, rec = eval_queries(
            lambda qv: [r.id for r in col_bq.query(vector=qv, top_k=top_k, hybrid=False, use_ann=False)],
            query_vectors,
            gt_unfiltered,
            top_k=top_k,
        )
        rows.append(
            ScaleRow(
                engine="SutraDB",
                variant="1-Bit BQ (Mmap-Evicted 32x)",
                docs_count=n_docs,
                dimension=dim,
                ingest_time_sec=bq_t,
                ingest_docs_per_sec=n_docs / max(bq_t, 1e-6),
                build_only_docs_per_sec=None,
                ram_mb=bq_ram_mb,
                rss_delta_mb=bq_ram_mb,
                compression="32.0x",
                p50_ms=p50,
                p99_ms=p99,
                qps=qps,
                recall_at_10=rec,
            )
        )
        del col_bq
        gc.collect()

    comp_rows, comp_notes = check_optional_competitors(vectors, query_vectors, ids, metadatas, gt_unfiltered, top_k)
    rows.extend(comp_rows)

    total_wall_sec = time.perf_counter() - wall_t0
    return {
        "docs": n_docs,
        "dim": dim,
        "queries": n_queries,
        "seed": seed,
        "raw_f32_mb": raw_f32_mb,
        "hnsw_m": hnsw_m,
        "hnsw_efc": hnsw_efc,
        "hnsw_efs": hnsw_efs,
        "hnsw_build_time_sec": hnsw_build_t,
        "hnsw_graph_mb": hnsw_graph_mb,
        "save_ms": save_ms,
        "cold_load_ms": cold_load_ms,
        "saved_sutra_mb": saved_bytes / (1024.0 * 1024.0),
        "saved_idx_mb": idx_bytes / (1024.0 * 1024.0),
        "post_reload_parity": parity_ok,
        "total_wall_sec": total_wall_sec,
        "rows": rows,
        "competitor_notes": comp_notes,
    }


def write_scale_report(summary: Dict[str, Any], report_path: Path, json_path: Optional[Path] = None) -> None:
    simd_info = detect_simd()
    cpu_arch = platform.machine()
    cpu_count = os.cpu_count() or 1
    py_ver = platform.python_version()
    rows: List[ScaleRow] = summary["rows"]

    md = f"""# SutraDB v6 Out-of-L3-Cache Scale Benchmark Report ({summary['docs']:,} x {summary['dim']}d)

Generated: `{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}`

## Environment & Dataset
- **Architecture / Cores:** `{cpu_arch}` (`{cpu_count} vCPUs`), `{simd_info}`
- **Python Runtime:** `Python {py_ver}`
- **Dataset:** `{summary['docs']:,} vectors x {summary['dim']} dimensions` (`Cosine`, normalized unit-sphere clustered embeddings, seed=`{summary['seed']}`)
- **Raw FP32 Matrix Size:** `{summary['raw_f32_mb']:.1f} MiB` (`{(summary['docs'] * summary['dim'] * 4) / 1e6:.1f} MB` decimal, well beyond CPU L3 cache)
- **Queries Evaluated:** `{summary['queries']}` unfiltered + `{summary['queries']}` at 1% metadata filter selectivity (`tenant_id == "tenant_00"`, 1,000 matching docs), `top_k = 10`
- **HNSW Parameters:** `m={summary['hnsw_m']}`, `ef_construction={summary['hnsw_efc']}`, `ef_search={summary['hnsw_efs']}`
- **Total Benchmark Wall Time:** `{summary['total_wall_sec']:.1f} s`

---

## Results Table

| Engine & Mode | Ingest (docs/s) | P50 (ms) | P99 (ms) | QPS | Vector/Index RAM | Compression | Recall@10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for r in rows:
        ingest_str = f"{r.ingest_docs_per_sec:,.0f}" if r.ingest_docs_per_sec > 0 else "-"
        if r.build_only_docs_per_sec is not None:
            ingest_str = f"{r.build_only_docs_per_sec:,.0f} (build)"
        md += (
            f"| **{r.engine}** ({r.variant}) | {ingest_str} | {r.p50_ms:.3f} | {r.p99_ms:.3f} | "
            f"{r.qps:,.1f} | {r.ram_mb:.1f} MiB | {r.compression} | {r.recall_at_10 * 100:.2f}% |\n"
        )

    md += f"""
---

## Cold Restart & Snapshot Sizes
- **Save (`.sutra` + `.sutra.idx`):** `{summary['save_ms']:.1f} ms` (`{summary['saved_sutra_mb']:.1f} MiB` `.sutra` + `{summary['saved_idx_mb']:.1f} MiB` `.sutra.idx`)
- **Cold `load()` Restart Time:** `{summary['cold_load_ms']:.2f} ms` (`has_hnsw=True`, top-10 exact parity preserved: `{summary['post_reload_parity']}`)
- **HNSW Graph Build Time ({cpu_count} vCPUs):** `{summary['hnsw_build_time_sec']:.2f} s` (`{summary['docs'] / max(summary['hnsw_build_time_sec'], 1e-6):,.0f} docs/s`, graph overhead `{summary['hnsw_graph_mb']:.1f} MiB`)

## Optional Local Competitors
"""
    for note in summary["competitor_notes"]:
        md += f"- {note}\n"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
    print(f"\nSaved Markdown report to {report_path}")

    if json_path is not None:
        serializable = dict(summary)
        serializable["rows"] = [asdict(r) for r in rows]
        json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SutraDB v6 100K x 768d Scale Benchmark")
    parser.add_argument("--docs", type=int, default=100000)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-efc", type=int, default=100)
    parser.add_argument("--hnsw-efs", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.035)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report", type=str, default="benches/SCALE_BENCHMARK_REPORT.md")
    parser.add_argument("--json-out", type=str, default="benches/scale_benchmark_results.json")
    args = parser.parse_args()

    summary = run_scale_benchmark(
        n_docs=args.docs,
        dim=args.dim,
        n_queries=args.queries,
        seed=args.seed,
        hnsw_m=args.hnsw_m,
        hnsw_efc=args.hnsw_efc,
        hnsw_efs=args.hnsw_efs,
        noise_std=args.noise_std,
        top_k=args.top_k,
    )
    write_scale_report(summary, Path(args.report), Path(args.json_out))


if __name__ == "__main__":
    main()
