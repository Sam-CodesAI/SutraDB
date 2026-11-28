# SutraDB v5.5.0 Out-of-L3-Cache Scale Benchmark Report (100,000 x 768d)

Generated: `2026-11-27 16:49:11 UTC`

## Environment & Dataset
- **Architecture / Cores:** `x86_64` (`2 vCPUs`), `AVX2 / FMA`
- **Python Runtime:** `Python 3.14.2`
- **Dataset:** `100,000 vectors x 768 dimensions` (`Cosine`, normalized unit-sphere clustered embeddings, seed=`42`)
- **Raw FP32 Matrix Size:** `293.0 MiB` (`307.2 MB` decimal, well beyond CPU L3 cache)
- **Queries Evaluated:** `100` unfiltered + `100` at 1% metadata filter selectivity (`tenant_id == "tenant_00"`, 1,000 matching docs), `top_k = 10`
- **HNSW Parameters:** `m=16`, `ef_construction=100`, `ef_search=128`
- **Total Benchmark Wall Time:** `122.5 s`

---

## Results Table

| Engine & Mode | Ingest (docs/s) | P50 (ms) | P99 (ms) | QPS | Vector/Index RAM | Compression | Recall@10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **NumPy** (Exact Brute-Force (GEMV)) | - | 35.019 | 56.064 | 27.4 | 293.0 MiB | 1x | 100.00% |
| **SutraDB** (Flat (Rust AVX2 SIMD)) | 29,228 | 14.236 | 21.737 | 68.1 | 293.0 MiB | 1x | 100.00% |
| **SutraDB** (HNSW Unfiltered (m=16)) | 1,063 (build) | 0.417 | 0.645 | 2,291.2 | 326.2 MiB | 1x | 100.00% |
| **SutraDB** (HNSW 1%-Filtered (ACORN)) | 1,063 (build) | 0.187 | 0.297 | 4,913.9 | 326.2 MiB | 1x | 100.00% |
| **SutraDB** (SQ8 (Mmap-Evicted 4x)) | 18,966 | 6.516 | 10.836 | 141.5 | 73.2 MiB | 4.0x | 97.90% |
| **SutraDB** (1-Bit BQ (Mmap-Evicted 32x)) | 19,243 | 2.043 | 3.580 | 436.2 | 9.2 MiB | 32.0x | 92.10% |

---

## Cold Restart & Snapshot Sizes
- **Save (`.sutra` + `.sutra.idx`):** `1991.1 ms` (`308.3 MiB` `.sutra` + `14.9 MiB` `.sutra.idx`)
- **Cold `load()` Restart Time:** `3066.35 ms` (`has_hnsw=True`, top-10 exact parity preserved: `True`)
- **HNSW Graph Build Time (2 vCPUs):** `94.07 s` (`1,063 docs/s`, graph overhead `33.2 MiB`)

## Optional Local Competitors
- `qdrant-client`: skipped (not installed in `.venv`).
- `chromadb`: skipped (not installed in `.venv`).
- `lancedb`: skipped (not installed in `.venv`).
