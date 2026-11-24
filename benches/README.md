# 🚀 SutraDB Automated Comparative Benchmarking Suite

Automated, empirical performance profiling and comparative benchmark suite for SutraDB. Measures vector ingestion throughput, memory footprint (Resident Set Size), latency percentiles (P50, P90, P95, P99), QPS, and exact Recall@10 across vector indexing algorithms, quantization schemes, and multi-modal hybrid retrieval.

---

## 🏃 Quickstart

### 1. Run Python Multi-Engine Benchmark Suite

```bash
# Quick mode (1,000 documents, 64-dim, 25 queries)
python benches/benchmark.py --quick

# Standard mode (10,000 documents, 128-dim, 100 queries)
python benches/benchmark.py --docs 10000 --dim 128 --queries 100 --top-k 10

# Stress mode (50,000 documents, 128-dim, 200 queries)
python benches/benchmark.py --stress
```

### CLI Options

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--docs` | int | `10000` | Number of documents to index |
| `--dim` | int | `128` | Vector dimensionality (e.g. 64, 128, 384, 768) |
| `--queries` | int | `100` | Number of query vectors for latency percentiles |
| `--top-k` | int | `10` | Top-K nearest neighbors to retrieve |
| `--quick` | flag | `false` | Preset for 1K docs (useful for CI smoke tests) |
| `--stress` | flag | `false` | Preset for 50K docs |
| `--report` | string | `benches/BENCHMARK_REPORT.md` | Path to generated Markdown report |
| `--json-out` | string | `benches/benchmark_results.json` | Path to generated JSON metrics file |

---

### 2. Run Native Rust SIMD Micro-Benchmark

Benchmarks raw distance computation throughput (Cosine, Squared Euclidean, Dot Product) executing over 1,000,000 vector pairs with runtime AVX-512 / AVX2 / ARM NEON auto-dispatch:

```bash
cargo run --release --example simd_distances
```

---

## 📊 Measured Modalities & Quantization Variants

1. **NumPy Exact Baseline:** Pure Python / BLAS matrix dot product on float32 arrays (ground truth source).
2. **SutraDB Flat (Rust SIMD):** Exact unindexed vector search with zero GIL overhead and multi-threaded Rayon parallel execution.
3. **SutraDB HNSW:** Hierarchical Navigable Small World graph for logarithmic $O(\log N)$ ANN search.
4. **SutraDB SQ8:** 8-Bit Scalar Quantization cutting RAM consumption by **75% (4x)** with **>98% Recall@10**.
5. **SutraDB 1-Bit BQ:** Binary Quantization packing 64 float32 dimensions into a single `u64` word, delivering a **32x RAM reduction** and hardware `POPCNT` evaluation.
6. **SutraDB PQ:** Product Quantization decomposing vector dimensions into $M$ orthogonal subspaces for up to **64x RAM savings**.
7. **SutraDB BM25 Lexical:** Pure lexical keyword retrieval using the linguistic Porter stemmer and stopword pruning pipeline.
8. **SutraDB Hybrid RRF:** Multi-modal Reciprocal Rank Fusion blending dense semantic similarity with exact keyword matches.

---

## 📄 Automated Reports

Every benchmark run generates:
- **`benches/BENCHMARK_REPORT.md`**: Complete Markdown report with hardware specs, Markdown comparison tables, and architectural takeaways.
- **`benches/benchmark_results.json`**: Raw machine-readable JSON telemetry for CI/CD tracking.
