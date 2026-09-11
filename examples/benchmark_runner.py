"""
Standalone high-throughput benchmark runner for SutraDB.
Profiles vector ingestion, SIMD search latency, and memory footprint.
"""

import os
import time
import numpy as np
import psutil
from sutradb import SutraDB, Document


def run_comprehensive_benchmark(n_vectors: int = 10000, dim: int = 384):
    print("=" * 65)
    print(f"🚀 SUTRADB BENCHMARK: {n_vectors:,} Vectors | {dim} Dimensions")
    print("=" * 65)

    process = psutil.Process(os.getpid())
    base_ram = process.memory_info().rss / (1024 * 1024)

    db = SutraDB()
    col = db.create_collection("benchmark_collection", dimension=dim, metric="cosine")

    print("\n1. Generating synthetic embedding matrix...")
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    categories = ["cloud", "database", "ai", "security", "devops"]

    docs = [
        Document(
            id=f"doc_{i:06d}",
            vector=vectors[i],
            text=f"High scale systems log record {i} for microservice {categories[i % len(categories)]}",
            metadata={
                "category": categories[i % len(categories)],
                "tier": "enterprise" if (i % 3 == 0) else "standard",
                "timestamp": 1700000000 + i
            }
        )
        for i in range(n_vectors)
    ]

    print("\n2. Ingestion Benchmark...")
    t0 = time.perf_counter()
    col.insert(docs)
    ingest_time = time.perf_counter() - t0
    docs_per_sec = n_vectors / ingest_time
    print(f"   -> Inserted {n_vectors:,} documents in {ingest_time:.3f}s")
    print(f"   -> Throughput: {docs_per_sec:,.0f} docs/second")

    post_ram = process.memory_info().rss / (1024 * 1024)
    net_ram = post_ram - base_ram
    stats = col.stats()
    print(f"   -> RAM Footprint: {net_ram:.1f} MB (Vector data: {stats['vector_memory_mb']:.1f} MB)")

    print("\n3. Vector Similarity Query Benchmark (1,000 iterations)...")
    query_vector = np.random.randn(dim).astype(np.float32)
    latencies = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = col.query(vector=query_vector, top_k=10, hybrid=False)
        latencies.append((time.perf_counter() - start) * 1000.0)

    p50 = np.median(latencies)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    print(f"   -> Pure Vector Search (Top-10):")
    print(f"      P50: {p50:.2f} ms | P90: {p90:.2f} ms | P99: {p99:.2f} ms")

    print("\n4. Filtered Hybrid Search Benchmark (500 iterations)...")
    hybrid_latencies = []
    for _ in range(500):
        start = time.perf_counter()
        _ = col.query(
            vector=query_vector,
            text="microservice database log",
            filter={"category": "database", "tier": "enterprise"},
            top_k=5,
            hybrid=True
        )
        hybrid_latencies.append((time.perf_counter() - start) * 1000.0)

    h_p50 = np.median(hybrid_latencies)
    h_p90 = np.percentile(hybrid_latencies, 90)
    h_p99 = np.percentile(hybrid_latencies, 99)
    print(f"   -> Filtered Hybrid Search (Top-5):")
    print(f"      P50: {h_p50:.2f} ms | P90: {h_p90:.2f} ms | P99: {h_p99:.2f} ms")

    print("\n" + "=" * 65)
    print("✅ Benchmark completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    run_comprehensive_benchmark(n_vectors=5000, dim=128)
