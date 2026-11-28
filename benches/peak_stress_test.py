#!/usr/bin/env python3
"""
SutraDB v5.5.0 Peak-Level Stress & Chaos Test Suite.

Exercises compiled `target/release/sutradb-server` + `sutradb-core==5.5.0` (PyO3):
1. Live Server 32-Client Concurrent Storm (gRPC HTTP bridge + RBAC + Prometheus)
2. Real OS SIGKILL (`kill -9`) Crash-Recovery Torture (3 cycles)
3. 50% Delete/Upsert Churn + Compaction + Cold-Restart Integrity (20K x 256d)
4. 10,000-Query Memory-Leak & Tail-Latency Soak
5. Adversarial Fuzz & Security Gauntlet (Live sutradb-server + FastAPI /ui/stats proxy)

All artifacts live exclusively inside `tempfile.TemporaryDirectory()` (zero stray disk files).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import numpy as np
import psutil
import uvicorn

import sutradb
from sutradb import NativeCollection, SutraDB
from sutradb.server import create_app


REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_BIN = REPO_ROOT / "target" / "release" / "sutradb-server"


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def get_two_free_ports() -> Tuple[int, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
        s1.bind(("127.0.0.1", 0))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
            s2.bind(("127.0.0.1", 0))
            return int(s1.getsockname()[1]), int(s2.getsockname()[1])


def percentile_ms(samples_sec: List[float], pct: float) -> float:
    if not samples_sec:
        return 0.0
    arr = np.array(samples_sec, dtype=np.float64) * 1000.0
    return float(np.percentile(arr, pct))


def parse_prom_metric_lines(text: str, name: str, label_substr: str = "") -> List[float]:
    vals: List[float] = []
    prefix = f"{name}"
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if not (line.startswith(f"{prefix} ") or line.startswith(f"{prefix}{{")):
            continue
        if label_substr and label_substr not in line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            vals.append(float(parts[1]))
    return vals


@dataclass
class LiveServer:
    proc: subprocess.Popen[str]
    grpc_port: int
    bridge_port: int
    bridge_url: str
    data_dir: Path

    def stop(self) -> Tuple[str, str]:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                out, err = self.proc.communicate(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                out, err = self.proc.communicate(timeout=5.0)
        else:
            out, err = self.proc.communicate(timeout=5.0)
        return out or "", err or ""

    def sigkill(self) -> None:
        if self.proc.poll() is None:
            os.kill(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=10.0)


def start_server(
    data_dir: Path,
    extra_env: Optional[Dict[str, str]] = None,
    grpc_port: Optional[int] = None,
    bridge_port: Optional[int] = None,
) -> LiveServer:
    if not SERVER_BIN.is_file():
        raise FileNotFoundError(f"Missing release server binary at {SERVER_BIN}")
    data_dir.mkdir(parents=True, exist_ok=True)

    if grpc_port is None and bridge_port is None:
        g_port, b_port = get_two_free_ports()
    else:
        g_port = grpc_port or get_free_port()
        b_port = bridge_port or get_free_port()
        while b_port == g_port:
            b_port = get_free_port()
    bridge_url = f"http://127.0.0.1:{b_port}"

    env = os.environ.copy()
    env.pop("SUTRA_API_KEY", None)
    env.pop("SUTRA_RBAC_CONFIG", None)
    if extra_env:
        env.update(extra_env)

    proc = subprocess.Popen(
        [
            str(SERVER_BIN),
            "--host",
            "127.0.0.1",
            "--port",
            str(g_port),
            "--data-dir",
            str(data_dir),
            "--http-bridge-addr",
            f"127.0.0.1:{b_port}",
        ],
        cwd=str(data_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out, err = proc.communicate(timeout=2.0)
            raise RuntimeError(f"sutradb-server exited early ({proc.returncode}):\n{out}\n{err}")
        try:
            r = httpx.get(f"{bridge_url}/health", timeout=0.5)
            if r.status_code == 200:
                return LiveServer(
                    proc=proc,
                    grpc_port=g_port,
                    bridge_port=b_port,
                    bridge_url=bridge_url,
                    data_dir=data_dir,
                )
        except httpx.RequestError:
            pass
        time.sleep(0.05)

    proc.kill()
    out, err = proc.communicate(timeout=2.0)
    raise TimeoutError(f"sutradb-server health check timed out:\n{out}\n{err}")


# =============================================================================
# Scenario 1: Live Server 32-Client Concurrent Storm (gRPC/HTTP + RBAC + Prom)
# =============================================================================
async def scenario_1_concurrent_storm() -> Dict[str, Any]:
    print("\n[Scenario 1] Launching 32-Client Concurrent Storm (20s) with RBAC + Prometheus...")
    dim = 64
    init_docs = 2_000
    rng = np.random.default_rng(101)
    vecs = rng.normal(size=(init_docs, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12

    admin_key = "storm-admin-key-001"
    rbac_cfg = {
        "storm-writer": {"role": "write", "collections": ["storm_*"]},
        "storm-reader": {"role": "read", "collections": ["storm_*"]},
        "other-tenant": {"role": "read", "collections": ["isolated_*"]},
    }

    with tempfile.TemporaryDirectory(prefix="sutra_stress_s1_") as tmp:
        srv = start_server(
            Path(tmp),
            extra_env={
                "SUTRA_API_KEY": admin_key,
                "SUTRA_RBAC_CONFIG": json.dumps(rbac_cfg),
            },
        )
        try:
            admin_hdr = {"Authorization": f"Bearer {admin_key}"}
            writer_hdr = {"X-API-Key": "storm-writer"}
            reader_hdr = {"Authorization": "Bearer storm-reader"}

            with httpx.Client(timeout=30.0) as client:
                r_c = client.post(
                    f"{srv.bridge_url}/collections",
                    headers=admin_hdr,
                    json={"name": "storm_main", "dimension": dim, "metric": "cosine", "enable_wal": True},
                )
                assert r_c.status_code in (200, 201), r_c.text

                # Seed 2,000 docs (1% selectivity on tenant_id == "t_00")
                for start in range(0, init_docs, 250):
                    end = min(init_docs, start + 250)
                    batch = [
                        {
                            "id": f"seed_{i}",
                            "vector": vecs[i].tolist(),
                            "text": f"distributed vector hnsw shard document number {i} topic_{i % 20}",
                            "metadata": {"tenant_id": f"t_{i % 100:02d}", "seq": i},
                        }
                        for i in range(start, end)
                    ]
                    r_ins = client.post(
                        f"{srv.bridge_url}/collections/storm_main/insert",
                        headers=writer_hdr,
                        json={"documents": batch},
                    )
                    assert r_ins.status_code == 200, r_ins.text

                r_hnsw = client.post(
                    f"{srv.bridge_url}/collections/storm_main/hnsw",
                    headers=admin_hdr,
                    json={"m": 16, "ef_construction": 80, "ef_search": 48},
                )
                assert r_hnsw.status_code == 200, r_hnsw.text

            latencies: List[float] = []
            errors_500 = 0
            other_bad_status: List[Tuple[str, int, str]] = []
            op_counts = {"hnsw": 0, "filtered": 0, "bm25": 0, "insert": 0, "patch": 0, "delete": 0}
            admin_ops = {"compact": 0, "save": 0}
            write_seq = 0
            lock = asyncio.Lock()

            stop_at = time.monotonic() + 20.0
            limits = httpx.Limits(max_connections=64, max_keepalive_connections=48)
            async with httpx.AsyncClient(timeout=30.0, limits=limits) as aclient:

                async def worker(wid: int) -> None:
                    nonlocal errors_500, write_seq
                    wrng = np.random.default_rng(1000 + wid)
                    while time.monotonic() < stop_at:
                        roll = float(wrng.random())
                        qvec = wrng.normal(size=dim).astype(np.float32)
                        qvec /= np.linalg.norm(qvec) + 1e-12
                        t0 = time.perf_counter()
                        try:
                            if roll < 0.35:
                                # HNSW query
                                op = "hnsw"
                                resp = await aclient.post(
                                    f"{srv.bridge_url}/collections/storm_main/query",
                                    headers=reader_hdr,
                                    json={"vector": qvec.tolist(), "top_k": 10, "hybrid": False, "use_ann": True},
                                )
                            elif roll < 0.60:
                                # 1% filtered query
                                op = "filtered"
                                resp = await aclient.post(
                                    f"{srv.bridge_url}/collections/storm_main/query",
                                    headers=reader_hdr,
                                    json={
                                        "vector": qvec.tolist(),
                                        "filter": {"tenant_id": "t_00"},
                                        "top_k": 10,
                                        "hybrid": False,
                                        "use_ann": True,
                                    },
                                )
                            elif roll < 0.78:
                                # BM25 query
                                op = "bm25"
                                topic = f"topic_{int(wrng.integers(0, 20))}"
                                resp = await aclient.post(
                                    f"{srv.bridge_url}/collections/storm_main/query",
                                    headers=reader_hdr,
                                    json={"text": f"distributed shard {topic}", "top_k": 10, "hybrid": False},
                                )
                            elif roll < 0.88:
                                # Batch insert
                                op = "insert"
                                async with lock:
                                    base = write_seq
                                    write_seq += 4
                                docs = [
                                    {
                                        "id": f"dyn_{base + j}",
                                        "vector": (wrng.normal(size=dim).astype(np.float32)).tolist(),
                                        "text": f"live batch storm write {base + j} topic_{j}",
                                        "metadata": {"tenant_id": f"t_{(base + j) % 100:02d}", "seq": base + j},
                                    }
                                    for j in range(4)
                                ]
                                resp = await aclient.post(
                                    f"{srv.bridge_url}/collections/storm_main/insert",
                                    headers=writer_hdr,
                                    json={"documents": docs},
                                )
                            elif roll < 0.94:
                                # Metadata patch
                                op = "patch"
                                target_id = f"seed_{int(wrng.integers(0, init_docs))}"
                                resp = await aclient.patch(
                                    f"{srv.bridge_url}/collections/storm_main/documents/{target_id}/metadata",
                                    headers=writer_hdr,
                                    json={"metadata": {"tenant_id": f"t_{wid % 100:02d}", "patched": True}},
                                )
                                if resp.status_code == 404:
                                    # Deleted earlier by another worker — acceptable race
                                    latencies.append(time.perf_counter() - t0)
                                    op_counts[op] += 1
                                    continue
                            else:
                                # Delete old seed doc (high numbers only so some remain)
                                op = "delete"
                                target_id = f"seed_{int(wrng.integers(init_docs // 2, init_docs))}"
                                resp = await aclient.delete(
                                    f"{srv.bridge_url}/collections/storm_main/documents/{target_id}",
                                    headers=writer_hdr,
                                )
                                if resp.status_code == 404:
                                    latencies.append(time.perf_counter() - t0)
                                    op_counts[op] += 1
                                    continue

                            dt = time.perf_counter() - t0
                            if resp.status_code >= 500:
                                errors_500 += 1
                                other_bad_status.append((op, resp.status_code, resp.text[:200]))
                            elif resp.status_code != 200:
                                other_bad_status.append((op, resp.status_code, resp.text[:200]))
                            else:
                                _ = resp.json()
                                latencies.append(dt)
                                op_counts[op] += 1
                        except Exception as exc:
                            other_bad_status.append(("exc", 0, repr(exc)))

                async def maintainer() -> None:
                    await asyncio.sleep(6.0)
                    if time.monotonic() < stop_at:
                        rc = await aclient.post(f"{srv.bridge_url}/collections/storm_main/compact", headers=admin_hdr)
                        if rc.status_code != 200:
                            other_bad_status.append(("compact", rc.status_code, rc.text[:200]))
                        else:
                            admin_ops["compact"] += 1
                    await asyncio.sleep(6.0)
                    if time.monotonic() < stop_at:
                        rs = await aclient.post(f"{srv.bridge_url}/collections/storm_main/save", headers=admin_hdr)
                        if rs.status_code != 200:
                            other_bad_status.append(("save", rs.status_code, rs.text[:200]))
                        else:
                            admin_ops["save"] += 1

                await asyncio.gather(*(worker(i) for i in range(32)), maintainer())

                m_resp = await aclient.get(f"{srv.bridge_url}/metrics", headers=reader_hdr)
                assert m_resp.status_code == 200, m_resp.text
                metrics_text = m_resp.text

                # Verify unauthorized /metrics blocked
                unauth_m = await aclient.get(f"{srv.bridge_url}/metrics")
                assert unauth_m.status_code == 401

            out, err = srv.stop()
            assert srv.proc.returncode is not None
            assert "panicked at" not in out and "panicked at" not in err, f"Panic detected:\n{err}"

            total_ok = sum(op_counts.values())
            q_prom_counts = sum(parse_prom_metric_lines(metrics_text, "sutradb_query_duration_seconds_count", 'collection="storm_main"'))
            ingest_prom = sum(parse_prom_metric_lines(metrics_text, "sutradb_ingest_documents_total"))
            expected_queries = op_counts["hnsw"] + op_counts["filtered"] + op_counts["bm25"]
            expected_ingested = init_docs + op_counts["insert"] * 4

            assert errors_500 == 0, f"HTTP 500s observed: {other_bad_status[:5]}"
            assert not other_bad_status, f"Unexpected HTTP errors: {other_bad_status[:5]}"
            assert admin_ops["compact"] >= 1 and admin_ops["save"] >= 1
            assert abs(q_prom_counts - expected_queries) == 0, (
                f"Prom query count {q_prom_counts} != executed {expected_queries}"
            )
            assert abs(ingest_prom - expected_ingested) == 0, (
                f"Prom ingest total {ingest_prom} != expected {expected_ingested}"
            )

            p50 = percentile_ms(latencies, 50)
            p99 = percentile_ms(latencies, 99)
            qps = total_ok / 20.0
            print(
                f"  -> OK={total_ok} ops ({qps:.1f} ops/s), P50={p50:.2f}ms, P99={p99:.2f}ms, "
                f"queries={expected_queries} (prom={int(q_prom_counts)}), ingested={expected_ingested}"
            )
            return {
                "passed": True,
                "total_ops": total_ok,
                "ops_per_sec": qps,
                "p50_ms": p50,
                "p99_ms": p99,
                "errors_500": errors_500,
                "prom_queries": int(q_prom_counts),
                "prom_ingested": int(ingest_prom),
                "compact_save": admin_ops,
            }
        finally:
            if srv.proc.poll() is None:
                srv.stop()


# =============================================================================
# Scenario 2: Real OS SIGKILL (`kill -9`) Crash-Recovery Torture (3 Cycles)
# =============================================================================
def scenario_2_sigkill_torture() -> Dict[str, Any]:
    print("\n[Scenario 2] Launching 3-Cycle Real OS SIGKILL (kill -9) WAL Crash-Recovery Torture...")
    dim = 32
    rng = np.random.default_rng(202)
    with tempfile.TemporaryDirectory(prefix="sutra_stress_s2_") as tmp:
        data_dir = Path(tmp)
        acked_alive: Set[str] = set()
        deleted_ids: Set[str] = set()
        total_acked_batches = 0

        for cycle in range(1, 4):
            srv = start_server(data_dir)
            try:
                with httpx.Client(timeout=10.0) as client:
                    if cycle == 1:
                        rc = client.post(
                            f"{srv.bridge_url}/collections",
                            json={"name": "wal_torture", "dimension": dim, "metric": "cosine", "enable_wal": True},
                        )
                        assert rc.status_code in (200, 201), rc.text
                    else:
                        # Verify everything acknowledged before prior SIGKILL survived!
                        st = client.get(f"{srv.bridge_url}/collections/wal_torture/stats").json()
                        assert st["count"] == len(acked_alive), (
                            f"Cycle {cycle} recovery count mismatch: got {st['count']}, expected {len(acked_alive)}"
                        )
                        # Verify every single surviving doc ID + absence of deleted IDs
                        r_docs = client.get(
                            f"{srv.bridge_url}/collections/wal_torture/documents?limit=10000&offset=0"
                        )
                        assert r_docs.status_code == 200
                        recovered_ids = {d["id"] for d in r_docs.json()["documents"]}
                        assert recovered_ids == acked_alive, (
                            f"Cycle {cycle} missing={len(acked_alive - recovered_ids)} extra={len(recovered_ids - acked_alive)}"
                        )

                    # Stream synchronous acked inserts + deletes while background writer runs
                    for b in range(25):
                        batch_ids = [f"c{cycle}_b{b}_d{j}" for j in range(10)]
                        batch_vecs = rng.normal(size=(10, dim)).astype(np.float32)
                        batch_vecs /= np.linalg.norm(batch_vecs, axis=1, keepdims=True) + 1e-12
                        docs = [
                            {
                                "id": did,
                                "vector": batch_vecs[j].tolist(),
                                "text": f"crash cycle {cycle} batch {b} doc {j}",
                                "metadata": {"cycle": cycle, "batch": b},
                            }
                            for j, did in enumerate(batch_ids)
                        ]
                        r_ins = client.post(
                            f"{srv.bridge_url}/collections/wal_torture/insert",
                            json={"documents": docs},
                        )
                        if r_ins.status_code == 200:
                            acked_alive.update(batch_ids)
                            total_acked_batches += 1

                        # Delete 2 earlier docs per every 5 batches
                        if b % 5 == 4 and len(acked_alive) > 20:
                            victim = sorted(acked_alive)[0]
                            r_del = client.delete(f"{srv.bridge_url}/collections/wal_torture/documents/{victim}")
                            if r_del.status_code == 200:
                                acked_alive.remove(victim)
                                deleted_ids.add(victim)

                    # Fire an in-flight async insert concurrently with SIGKILL
                    stop_bg = False

                    def bg_stream() -> None:
                        idx = 0
                        with httpx.Client(timeout=1.0) as bgc:
                            while not stop_bg:
                                did = f"bg_c{cycle}_{idx}"
                                vec = (rng.normal(size=dim).astype(np.float32)).tolist()
                                try:
                                    resp = bgc.post(
                                        f"{srv.bridge_url}/collections/wal_torture/insert",
                                        json={"documents": [{"id": did, "vector": vec, "text": "inflight"}]},
                                    )
                                    if resp.status_code == 200:
                                        acked_alive.add(did)
                                except Exception:
                                    break
                                idx += 1

                    th = threading.Thread(target=bg_stream, daemon=True)
                    th.start()
                    time.sleep(0.08)
                    srv.sigkill()
                    stop_bg = True
                    th.join(timeout=2.0)
            finally:
                if srv.proc.poll() is None:
                    srv.sigkill()

        # Final cycle 4 cold verification
        final_srv = start_server(data_dir)
        try:
            with httpx.Client(timeout=10.0) as client:
                r_st = client.get(f"{final_srv.bridge_url}/collections/wal_torture/stats")
                assert r_st.status_code == 200
                final_count = r_st.json()["count"]
                assert final_count == len(acked_alive), (
                    f"Final count {final_count} != acked {len(acked_alive)}"
                )
                r_all = client.get(
                    f"{final_srv.bridge_url}/collections/wal_torture/documents?limit=10000&offset=0"
                )
                assert r_all.status_code == 200
                recovered_ids = {d["id"] for d in r_all.json()["documents"]}
                assert recovered_ids == acked_alive
                assert not (recovered_ids & deleted_ids)
                # Ensure queries work post-recovery
                qv = rng.normal(size=dim).astype(np.float32)
                qv /= np.linalg.norm(qv) + 1e-12
                rq = client.post(
                    f"{final_srv.bridge_url}/collections/wal_torture/query",
                    json={"vector": qv.tolist(), "top_k": 5, "hybrid": False},
                )
                assert rq.status_code == 200 and len(rq.json()["results"]) == 5
        finally:
            final_srv.stop()

        print(f"  -> Survived 3 SIGKILLs: {len(acked_alive)} acked docs 100% preserved, 0 ghost deletes")
        return {
            "passed": True,
            "cycles": 3,
            "acked_surviving_docs": len(acked_alive),
            "deleted_docs": len(deleted_ids),
            "acked_batches": total_acked_batches,
        }


# =============================================================================
# Scenario 3: 50% Delete/Upsert Churn + Compaction + Cold-Restart Integrity
# =============================================================================
def scenario_3_churn_compaction_recall() -> Dict[str, Any]:
    print("\n[Scenario 3] Launching 20,000 x 256d Churn (50% delete + 5K upsert) + Compact + Cold Load...")
    n_init = 20_000
    n_del = 10_000
    n_upsert = 5_000
    dim = 256
    rng = np.random.default_rng(303)

    centroids = rng.normal(size=(50, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12

    def make_vecs(count: int) -> np.ndarray:
        cids = rng.integers(0, 50, size=count)
        # Scale noise by 1/sqrt(dim) (~0.055 * sqrt(256) = 0.88) to match realistic clustered embeddings
        noise = rng.normal(scale=0.055, size=(count, dim)).astype(np.float32)
        mat = centroids[cids] + noise
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat

    v_init = make_vecs(n_init)
    v_upsert = make_vecs(n_upsert)
    deleted_ids = {f"d_{i}" for i in range(0, n_init, 2)}  # 10,000 odd/even deletions
    expected_alive_count = n_init - n_del + n_upsert  # 15,000

    with tempfile.TemporaryDirectory(prefix="sutra_stress_s3_") as tmp:
        snap_sq8 = Path(tmp) / "churn_sq8.sutra"
        snap_bq = Path(tmp) / "churn_bq.sutra"

        # Collection A: HNSW + SQ8
        col_a = NativeCollection("churn_sq8", dim, "cosine", str(snap_sq8), False)
        ids_init = [f"d_{i}" for i in range(n_init)]
        texts_init = [f"telemetry shard cluster node {(i % 50)} tag_{i % 10}" for i in range(n_init)]
        metas_init = [{"tenant": f"ten_{i % 100:02d}", "phase": "init"} for i in range(n_init)]
        col_a.insert_numpy(v_init, ids_init, texts_init, metas_init)
        col_a.build_hnsw(16, 128, 128)
        col_a.enable_sq8()

        # Collection B: BQ
        col_b = NativeCollection("churn_bq", dim, "cosine", str(snap_bq), False)
        col_b.insert_numpy(v_init, ids_init, texts_init, metas_init)
        col_b.enable_bq()

        # 50% Delete + 5,000 new upserts
        del_list = sorted(deleted_ids)
        ids_upsert = [f"new_{i}" for i in range(n_upsert)]
        texts_upsert = [f"upserted fresh vector node {(i % 50)} tag_{i % 10}" for i in range(n_upsert)]
        metas_upsert = [{"tenant": f"ten_{i % 100:02d}", "phase": "upsert"} for i in range(n_upsert)]
        for col in (col_a, col_b):
            d_cnt = col.delete(del_list)
            assert d_cnt == n_del
            col.insert_numpy(v_upsert, ids_upsert, texts_upsert, metas_upsert)
            assert col.count() == expected_alive_count

        # Compact + Save + Cold Load
        for col in (col_a, col_b):
            reclaimed = col.compact()
            assert reclaimed == n_del, f"Compact reclaimed count mismatch: {reclaimed} != {n_del}"
            assert col.count() == expected_alive_count, f"Post-compact count mismatch: {col.count()}"
            col.save()

        del col_a, col_b

        cold_a = NativeCollection("churn_sq8", dim, "cosine", str(snap_sq8), False)
        cold_b = NativeCollection("churn_bq", dim, "cosine", str(snap_bq), False)
        assert cold_a.load() == expected_alive_count
        assert cold_b.load() == expected_alive_count
        assert cold_a.count() == expected_alive_count
        assert cold_b.count() == expected_alive_count
        assert cold_a.stats()["has_hnsw"] is True
        assert cold_a.stats()["quantized_sq8"] is True
        assert cold_b.stats()["quantized_bq"] is True

        # Exact matrix of surviving vectors in order: odd init indices (1,3,...,19999) + v_upsert
        surviving_ids = [f"d_{i}" for i in range(1, n_init, 2)] + [f"new_{i}" for i in range(n_upsert)]
        surviving_vecs = np.vstack([v_init[1::2], v_upsert])
        assert len(surviving_ids) == expected_alive_count

        queries = make_vecs(50)
        ghosts = 0
        hnsw_hits = 0
        for q in queries:
            # Ground truth top-10 via exact NumPy cosine
            sims = surviving_vecs @ q
            gt_idx = np.argpartition(sims, -10)[-10:]
            gt_ids = {surviving_ids[idx] for idx in gt_idx}

            res_hnsw = cold_a.query(vector=q, top_k=10, hybrid=False, use_ann=True)
            res_filt = cold_a.query(vector=q, filter={"tenant": "ten_00"}, top_k=10, hybrid=False, use_ann=True)
            res_bm25 = cold_a.query(text="upserted telemetry node 7", top_k=10, hybrid=False)
            res_sq8 = cold_a.query(vector=q, top_k=10, hybrid=False, use_ann=False)
            res_bq = cold_b.query(vector=q, top_k=10, hybrid=False, use_ann=False)

            for rlist in (res_hnsw, res_filt, res_bm25, res_sq8, res_bq):
                for r in rlist:
                    if r["id"] in deleted_ids:
                        ghosts += 1

            hnsw_hits += sum(1 for r in res_hnsw if r["id"] in gt_ids)

        recall_hnsw = (hnsw_hits / (len(queries) * 10)) * 100.0
        assert ghosts == 0, f"Detected {ghosts} ghost/deleted docs!"
        assert recall_hnsw >= 95.0, f"Post-compaction HNSW Recall@10 too low: {recall_hnsw:.2f}%"

        print(f"  -> Churn 20K->10K->15K compacted & cold-loaded: ghosts=0, HNSW Recall@10={recall_hnsw:.2f}%")
        return {
            "passed": True,
            "alive_docs": expected_alive_count,
            "ghosts": ghosts,
            "hnsw_recall_10_pct": recall_hnsw,
        }


# =============================================================================
# Scenario 4: 10,000-Query Memory-Leak & Tail-Latency Soak
# =============================================================================
def scenario_4_memory_soak() -> Dict[str, Any]:
    print("\n[Scenario 4] Launching 10,000-Query PyO3 -> Rust Memory-Leak & Latency Soak...")
    n_docs = 5_000
    dim = 128
    rng = np.random.default_rng(404)
    vecs = rng.normal(size=(n_docs, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12

    with tempfile.TemporaryDirectory(prefix="sutra_stress_s4_") as tmp:
        col_hnsw = NativeCollection("soak_hnsw_sq8", dim, "cosine", str(Path(tmp) / "s4_a.sutra"), False)
        col_bq = NativeCollection("soak_bq", dim, "cosine", str(Path(tmp) / "s4_b.sutra"), False)
        ids_s4 = [f"s_{i}" for i in range(n_docs)]
        texts_s4 = [f"vector search quantization neural ranking doc {i} cat_{i % 25}" for i in range(n_docs)]
        metas_s4 = [{"tenant": f"t_{i % 100:02d}", "score": i} for i in range(n_docs)]
        col_hnsw.insert_numpy(vecs, ids_s4, texts_s4, metas_s4)
        col_hnsw.build_hnsw(16, 64, 48)
        col_hnsw.enable_sq8()

        col_bq.insert_numpy(vecs, ids_s4, texts_s4, metas_s4)
        col_bq.enable_bq()

        qvecs = rng.normal(size=(256, dim)).astype(np.float32)
        qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-12

        proc = psutil.Process(os.getpid())
        rss_at_1k_mb = 0.0
        early_lat: List[float] = []
        late_lat: List[float] = []

        for qi in range(10_000):
            qv = qvecs[qi % 256]
            mode = qi % 6
            t0 = time.perf_counter()
            if mode == 0:
                _ = col_hnsw.query(vector=qv, top_k=10, hybrid=False, use_ann=True)
            elif mode == 1:
                _ = col_hnsw.query(vector=qv, filter={"tenant": "t_00"}, top_k=10, hybrid=False, use_ann=True)
            elif mode == 2:
                _ = col_hnsw.query(vector=qv, top_k=10, hybrid=False, use_ann=False)
            elif mode == 3:
                _ = col_bq.query(vector=qv, top_k=10, hybrid=False, use_ann=False)
            elif mode == 4:
                _ = col_hnsw.query(text=f"neural ranking cat_{qi % 25}", top_k=10, hybrid=False)
            else:
                _ = col_hnsw.query(vector=qv, text=f"quantization cat_{qi % 25}", top_k=10, hybrid=True, use_ann=True)
            dt = time.perf_counter() - t0

            if qi < 2_000:
                early_lat.append(dt)
            elif qi >= 8_000:
                late_lat.append(dt)

            if qi == 999:
                rss_at_1k_mb = proc.memory_info().rss / (1024 * 1024)

        rss_at_10k_mb = proc.memory_info().rss / (1024 * 1024)
        rss_delta_mb = rss_at_10k_mb - rss_at_1k_mb
        early_p50 = percentile_ms(early_lat, 50)
        early_p99 = percentile_ms(early_lat, 99)
        late_p50 = percentile_ms(late_lat, 50)
        late_p99 = percentile_ms(late_lat, 99)

        assert rss_delta_mb < 10.0, f"Memory leak detected: RSS grew by {rss_delta_mb:.2f} MiB!"
        print(
            f"  -> RSS@1K={rss_at_1k_mb:.2f} MiB, RSS@10K={rss_at_10k_mb:.2f} MiB (delta={rss_delta_mb:+.2f} MiB) | "
            f"P50 {early_p50:.3f}->{late_p50:.3f}ms, P99 {early_p99:.3f}->{late_p99:.3f}ms"
        )
        return {
            "passed": True,
            "rss_1k_mb": rss_at_1k_mb,
            "rss_10k_mb": rss_at_10k_mb,
            "rss_delta_mb": rss_delta_mb,
            "early_p50_ms": early_p50,
            "early_p99_ms": early_p99,
            "late_p50_ms": late_p50,
            "late_p99_ms": late_p99,
        }


# =============================================================================
# Scenario 5: Adversarial Fuzz & Security Gauntlet (Live Server)
# =============================================================================
def scenario_5_adversarial_gauntlet() -> Dict[str, Any]:
    print("\n[Scenario 5] Launching Live Adversarial Fuzz & Security Gauntlet...")
    admin_key = "gauntlet-admin"
    rbac_cfg = {
        "tenant-a": {"role": "read", "collections": ["tenant_a_*"]},
        "tenant-b-writer": {"role": "write", "collections": ["tenant_b_*"]},
    }

    with tempfile.TemporaryDirectory(prefix="sutra_stress_s5_") as tmp:
        tmp_path = Path(tmp)
        srv = start_server(
            tmp_path / "data",
            extra_env={
                "SUTRA_API_KEY": admin_key,
                "SUTRA_RBAC_CONFIG": json.dumps(rbac_cfg),
            },
        )
        # Also spin up FastAPI frontend proxying /ui/stats to live sutradb-server
        prev_backend = os.environ.get("SUTRA_BACKEND_URL")
        os.environ["SUTRA_BACKEND_URL"] = srv.bridge_url
        try:
            app = create_app(data_dir=str(tmp_path / "ui_dummy"), db=None)
        finally:
            if prev_backend is None:
                os.environ.pop("SUTRA_BACKEND_URL", None)
            else:
                os.environ["SUTRA_BACKEND_URL"] = prev_backend

        ui_port = get_free_port()
        u_cfg = uvicorn.Config(app, host="127.0.0.1", port=ui_port, log_level="error")
        u_srv = uvicorn.Server(u_cfg)
        u_srv.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        u_th = threading.Thread(target=u_srv.run, daemon=True)
        u_th.start()
        ui_url = f"http://127.0.0.1:{ui_port}"
        for _ in range(100):
            try:
                if httpx.get(f"{ui_url}/health", timeout=0.5).status_code == 200:
                    break
            except httpx.RequestError:
                pass
            time.sleep(0.05)

        checks = 0
        failures: List[str] = []

        def check_status(label: str, actual: int, allowed: Tuple[int, ...]) -> None:
            nonlocal checks
            checks += 1
            if actual not in allowed:
                failures.append(f"{label}: got {actual}, expected {allowed}")

        try:
            admin_hdr = {"Authorization": f"Bearer {admin_key}"}
            ta_hdr = {"Authorization": "Bearer tenant-a"}
            tb_hdr = {"X-API-Key": "tenant-b-writer"}

            with httpx.Client(timeout=10.0) as client:
                assert client.post(
                    f"{srv.bridge_url}/collections",
                    headers=admin_hdr,
                    json={"name": "tenant_a_vault", "dimension": 4, "metric": "cosine", "enable_wal": True},
                ).status_code in (200, 201)
                assert client.post(
                    f"{srv.bridge_url}/collections",
                    headers=admin_hdr,
                    json={"name": "tenant_b_vault", "dimension": 4, "metric": "cosine"},
                ).status_code in (200, 201)
                assert client.post(
                    f"{srv.bridge_url}/collections/tenant_a_vault/insert",
                    headers=admin_hdr,
                    json={"documents": [{"id": "a0", "vector": [1.0, 0.0, 0.0, 0.0], "text": "alpha"}]},
                ).status_code == 200
                assert client.post(
                    f"{srv.bridge_url}/collections/tenant_b_vault/insert",
                    headers=admin_hdr,
                    json={"documents": [{"id": "b0", "vector": [0.0, 1.0, 0.0, 0.0], "text": "beta"}]},
                ).status_code == 200
                assert client.post(f"{srv.bridge_url}/collections/tenant_a_vault/hnsw", headers=admin_hdr).status_code == 200
                assert client.post(f"{srv.bridge_url}/collections/tenant_a_vault/save", headers=admin_hdr).status_code == 200

                # 1. NaN / Inf / overflow float attacks
                r = client.post(
                    f"{srv.bridge_url}/collections/tenant_a_vault/insert",
                    headers=admin_hdr,
                    content=b'{"documents":[{"id":"bad_nan","vector":[1.0,NaN,0.0,0.0]}]}',
                )
                check_status("raw_nan_insert", r.status_code, (400,))

                r = client.post(
                    f"{srv.bridge_url}/collections/tenant_a_vault/insert",
                    headers=admin_hdr,
                    content=b'{"documents":[{"id":"bad_inf","vector":[1.0,1e50,0.0,0.0]}]}',
                )
                check_status("f32_overflow_inf_insert", r.status_code, (400,))

                r = client.post(
                    f"{srv.bridge_url}/collections/tenant_a_vault/query",
                    headers=ta_hdr,
                    content=b'{"vector":[1.0,1e50,0.0,0.0],"top_k":5}',
                )
                check_status("f32_overflow_inf_query", r.status_code, (400,))

                # 2. Wrong dimensions
                r = client.post(
                    f"{srv.bridge_url}/collections/tenant_a_vault/insert",
                    headers=admin_hdr,
                    json={"documents": [{"id": "bad_dim", "vector": [1.0, 0.0]}]},
                )
                check_status("wrong_dim_insert", r.status_code, (400,))
                r = client.post(
                    f"{srv.bridge_url}/collections/tenant_a_vault/query",
                    headers=ta_hdr,
                    json={"vector": [1.0, 0.0, 0.0], "top_k": 5},
                )
                check_status("wrong_dim_query", r.status_code, (400,))

                # 3. top_k boundaries (0 and 1,000,000)
                r = client.post(
                    f"{srv.bridge_url}/collections/tenant_a_vault/query",
                    headers=ta_hdr,
                    json={"vector": [1.0, 0.0, 0.0, 0.0], "top_k": 0},
                )
                check_status("top_k_zero", r.status_code, (400,))
                r = client.post(
                    f"{srv.bridge_url}/collections/tenant_a_vault/query",
                    headers=ta_hdr,
                    json={"vector": [1.0, 0.0, 0.0, 0.0], "top_k": 1_000_000},
                )
                check_status("top_k_1M", r.status_code, (400,))

                # 4. Path traversal & null bytes in collection names
                for bad_name in ("../../etc/passwd", "..%2F..%2Fetc%2Fpasswd", "bad\x00col", "bad%00col", "a..b"):
                    r_c = client.post(
                        f"{srv.bridge_url}/collections",
                        headers=admin_hdr,
                        json={"name": bad_name, "dimension": 4},
                    )
                    check_status(f"create_{bad_name!r}", r_c.status_code, (400,))
                    if "\x00" not in bad_name:
                        r_g = client.get(f"{srv.bridge_url}/collections/{bad_name}/stats", headers=admin_hdr)
                        check_status(f"stats_{bad_name!r}", r_g.status_code, (400, 404))
                    else:
                        with socket.create_connection(("127.0.0.1", srv.bridge_port), timeout=2.0) as sock:
                            raw_req = (
                                f"GET /collections/{bad_name}/stats HTTP/1.1\r\n"
                                f"Host: 127.0.0.1\r\nAuthorization: Bearer {admin_key}\r\nConnection: close\r\n\r\n"
                            ).encode("latin-1")
                            sock.sendall(raw_req)
                            status_line = sock.recv(256).decode("latin-1", errors="replace").splitlines()[0]
                            raw_code = int(status_line.split()[1])
                            check_status("stats_raw_nul", raw_code, (400, 404))

                # 5. Invalid & catastrophic ReDoS regex filters (bounded safely, fast return)
                t_re0 = time.perf_counter()
                r_re1 = client.post(
                    f"{srv.bridge_url}/collections/tenant_a_vault/query",
                    headers=ta_hdr,
                    json={"vector": [1.0, 0.0, 0.0, 0.0], "filter": {"id": {"$regex": "(unclosed["}}, "top_k": 5},
                )
                r_re2 = client.post(
                    f"{srv.bridge_url}/collections/tenant_a_vault/query",
                    headers=ta_hdr,
                    json={
                        "vector": [1.0, 0.0, 0.0, 0.0],
                        "filter": {"id": {"$regex": "(a+)+$"}},
                        "top_k": 5,
                    },
                )
                re_elapsed = time.perf_counter() - t_re0
                check_status("invalid_regex", r_re1.status_code, (200, 400))
                check_status("redos_regex", r_re2.status_code, (200, 400))
                assert re_elapsed < 1.0, f"Regex filter hung for {re_elapsed:.2f}s"

                # 6. Wrong / missing API keys & cross-tenant probes (including /ui/stats)
                check_status("missing_key", client.get(f"{srv.bridge_url}/collections").status_code, (401,))
                check_status(
                    "wrong_key",
                    client.get(f"{srv.bridge_url}/collections", headers={"Authorization": "Bearer nope"}).status_code,
                    (401,),
                )
                check_status(
                    "cross_tenant_query",
                    client.post(
                        f"{srv.bridge_url}/collections/tenant_b_vault/query",
                        headers=ta_hdr,
                        json={"vector": [0.0, 1.0, 0.0, 0.0], "top_k": 5},
                    ).status_code,
                    (403,),
                )
                check_status(
                    "reader_cannot_write",
                    client.post(
                        f"{srv.bridge_url}/collections/tenant_a_vault/insert",
                        headers=ta_hdr,
                        json={"documents": [{"id": "x", "vector": [1.0, 0.0, 0.0, 0.0]}]},
                    ).status_code,
                    (403,),
                )
                check_status(
                    "writer_cannot_compact",
                    client.post(f"{srv.bridge_url}/collections/tenant_b_vault/compact", headers=tb_hdr).status_code,
                    (403,),
                )

                # Proxy /ui/stats cross-tenant isolation
                check_status("ui_stats_noauth", client.get(f"{ui_url}/ui/stats").status_code, (401,))
                r_ui_ta = client.get(f"{ui_url}/ui/stats", headers=ta_hdr)
                check_status("ui_stats_ta", r_ui_ta.status_code, (200,))
                ui_cols = [c["name"] for c in r_ui_ta.json()["collections"]]
                assert ui_cols == ["tenant_a_vault"], f"Cross-tenant leak in /ui/stats: {ui_cols}"

                # Health remains clean 200 throughout gauntlet
                check_status("health_alive_1", client.get(f"{srv.bridge_url}/health").status_code, (200,))

            srv.stop()

            # 7. Bit-flip corrupt .sutra and .sutra.idx files, verify server boots safely without crash!
            # Wait for clean disk corruption test
            corrupt_idx = tmp_path / "data" / "tenant_a_vault.sutra.idx"
            assert corrupt_idx.is_file()
            idx_bytes = bytearray(corrupt_idx.read_bytes())
            idx_bytes[-10] ^= 0xFF
            corrupt_idx.write_bytes(bytes(idx_bytes))

            srv2 = start_server(
                tmp_path / "data",
                extra_env={"SUTRA_API_KEY": admin_key, "SUTRA_RBAC_CONFIG": json.dumps(rbac_cfg)},
            )
            try:
                with httpx.Client(timeout=5.0) as client:
                    check_status("health_after_corrupt_idx", client.get(f"{srv2.bridge_url}/health").status_code, (200,))
                    # Healthy tenant_b_vault still works!
                    r_b = client.post(
                        f"{srv2.bridge_url}/collections/tenant_b_vault/query",
                        headers=admin_hdr,
                        json={"vector": [0.0, 1.0, 0.0, 0.0], "top_k": 5},
                    )
                    check_status("healthy_col_after_corrupt_idx", r_b.status_code, (200,))
            finally:
                srv2.stop()

            corrupt_sutra = tmp_path / "data" / "tenant_b_vault.sutra"
            if corrupt_sutra.is_file():
                sb = bytearray(corrupt_sutra.read_bytes())
                sb[min(20, len(sb) - 1)] ^= 0xFF
                corrupt_sutra.write_bytes(bytes(sb))

            srv3 = start_server(
                tmp_path / "data",
                extra_env={"SUTRA_API_KEY": admin_key, "SUTRA_RBAC_CONFIG": json.dumps(rbac_cfg)},
            )
            try:
                with httpx.Client(timeout=5.0) as client:
                    check_status("health_after_corrupt_sutra", client.get(f"{srv3.bridge_url}/health").status_code, (200,))
            finally:
                out3, err3 = srv3.stop()
                assert "panicked at" not in err3, f"Server panicked on corrupt file: {err3}"

            assert not failures, f"Gauntlet assertion failures: {failures}"
            print(f"  -> All {checks} adversarial probes passed cleanly (0 crashes, 0 HTTP 500s)")
            return {"passed": True, "probes_passed": checks, "failures": 0}
        finally:
            u_srv.should_exit = True
            u_th.join(timeout=3.0)
            if srv.proc.poll() is None:
                srv.stop()


def main() -> int:
    print(f"SutraDB v{sutradb.__version__} Peak Stress & Chaos Torture Suite")
    print(f"Server Binary: {SERVER_BIN}")
    t_start = time.perf_counter()

    r1 = asyncio.run(scenario_1_concurrent_storm())
    r2 = scenario_2_sigkill_torture()
    r3 = scenario_3_churn_compaction_recall()
    r4 = scenario_4_memory_soak()
    r5 = scenario_5_adversarial_gauntlet()

    total_wall = time.perf_counter() - t_start
    print("\n" + "=" * 78)
    print(f"ALL 5 PEAK CHAOS SCENARIOS PASSED in {total_wall:.1f}s")
    print("=" * 78)
    print(json.dumps({"s1": r1, "s2": r2, "s3": r3, "s4": r4, "s5": r5, "wall_sec": round(total_wall, 2)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
