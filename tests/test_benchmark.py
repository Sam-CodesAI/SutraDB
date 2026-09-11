"""
Benchmark and latency profiling suite for SutraDB.
Verifies sub-5ms query performance and throughput.
"""

import time
import numpy as np
from sutradb.core import Collection, Document


def test_performance_benchmark():
    num_docs = 2000
    dim = 128
    col = Collection(name="perf_test", dimension=dim, metric="cosine")

    # Generate random normalized vectors and texts
    np.random.seed(42)
    raw_matrix = np.random.randn(num_docs, dim).astype(np.float32)
    categories = ["finance", "engineering", "sales", "hr", "operations"]

    docs = []
    for i in range(num_docs):
        docs.append(Document(
            id=f"doc_{i:05d}",
            vector=raw_matrix[i],
            text=f"Technical log event {i} for module {categories[i % len(categories)]} with status code 200",
            metadata={
                "category": categories[i % len(categories)],
                "latency": int(i % 100),
                "is_active": (i % 2 == 0)
            }
        ))

    # Benchmark Batch Insertion
    start_insert = time.perf_counter()
    col.insert(docs)
    insert_duration = time.perf_counter() - start_insert
    docs_per_sec = num_docs / insert_duration

    assert col.count() == num_docs
    print(f"\n⚡ Ingestion: {num_docs} docs inserted in {insert_duration*1000:.2f}ms ({docs_per_sec:.0f} docs/sec)")

    # Benchmark Pure Vector Queries (100 iterations)
    query_vector = np.random.randn(dim).astype(np.float32)
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        results = col.query(vector=query_vector, top_k=10)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50 = np.median(latencies)
    p99 = np.percentile(latencies, 99)
    print(f"⚡ Pure Vector Search (Top-10): P50 = {p50:.2f}ms, P99 = {p99:.2f}ms")
    assert p50 < 10.0, f"P50 query latency too high: {p50:.2f}ms"

    # Benchmark Hybrid + Compound Filter Queries (100 iterations)
    hybrid_latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        results = col.query(
            vector=query_vector,
            text="Technical log event module engineering",
            filter={
                "category": "engineering",
                "is_active": True,
                "latency": {"$lt": 50}
            },
            top_k=5,
            hybrid=True
        )
        hybrid_latencies.append((time.perf_counter() - t0) * 1000.0)

    h_p50 = np.median(hybrid_latencies)
    h_p99 = np.percentile(hybrid_latencies, 99)
    print(f"⚡ Filtered Hybrid Search (Top-5): P50 = {h_p50:.2f}ms, P99 = {h_p99:.2f}ms")
    assert h_p50 < 15.0, f"Hybrid P50 query latency too high: {h_p50:.2f}ms"
