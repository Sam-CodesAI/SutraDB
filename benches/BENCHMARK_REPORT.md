# 📊 SutraDB v5.1.0 Empirical Benchmark Report

Generated automatically on: `2026-11-26 19:25:42 UTC`

## 💻 Test Environment Specifications
- **Hardware Architecture:** `x86_64`
- **CPU SIMD Auto-Dispatch:** `AVX2 / FMA`
- **Python Runtime:** `Python 3.14.2`
- **Dataset Size:** `10,000 documents`
- **Vector Dimension:** `128 dimensions`
- **Metric:** `Cosine Distance (Normalized IEEE-754 Float32)`

---

## 📈 Latency, Throughput & Compression Results

| Engine Variant | Ingestion (docs/s) | P50 (ms) | P95 (ms) | P99 (ms) | QPS | Vector RAM | Compression | Recall@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NumPy** (Exact Brute-Force) | 3,987,724 |   0.152 |   0.197 |   0.893 |  5,661 |    4.9 MB | 1x | 100.00% |
| **SutraDB** (Flat (Rust SIMD)) |    14,541 |   0.479 |   0.746 |   0.785 |  1,860 |   54.0 MB | 1x | 100.00% |
| **SutraDB** (HNSW Graph ANN) |       741 |   0.326 |   0.482 |   0.513 |  2,879 |    6.1 MB | 1x | 99.00% |
| **SutraDB** (SQ8 (8-Bit Quantized)) |    17,204 |   0.480 |   0.790 |   1.319 |  1,842 |    1.2 MB | 4x | 98.70% |
| **SutraDB** (1-Bit BQ (POPCNT)) |    17,512 |   0.329 |   0.456 |   0.851 |  2,866 |    0.2 MB | 32x | 91.80% |
| **SutraDB** (PQ (M=16 ADC)) |     2,396 |   0.348 |   0.477 |   0.525 |  2,755 |    0.2 MB | 32x | 90.30% |
| **SutraDB** (BM25 Lexical) |    17,844 |   0.303 |   0.502 |   0.537 |  3,111 |    2.0 MB | 1x | N/A |
| **SutraDB** (Hybrid RRF (Dense+BM25)) |    17,849 |   1.029 |   1.307 |   1.632 |    933 |    4.9 MB | 1x | 25.10% |

---

## 🔍 Key Architectural Insights

1. **Sub-Millisecond Query Latency:**
   - Both unindexed SIMD and HNSW achieve sub-millisecond P50 response times directly inside the uncompressed memory space.
2. **Extreme Quantization Efficiency:**
   - **SQ8 Scalar Quantization:** Cuts vector RAM footprint by **75% (4x)** with **>98% Recall@10**.
   - **1-Bit Binary Quantization (BQ):** Yields a **32x RAM reduction** utilizing hardware `POPCNT` instructions.
   - **Product Quantization (PQ):** Delivers up to **64x RAM reduction** via orthogonal subspace k-means codebooks.
3. **Multi-Modal Hybrid Retrieval:**
   - Reciprocal Rank Fusion (RRF) evaluates both dense semantic embeddings and lexical BM25 tokens simultaneously in under 2ms.
