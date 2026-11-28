# सूत्र DB (SutraDB) v5.5.0: Zero-Build Embedded Developer Workbench & Distributed Edition

[![PyPI](https://img.shields.io/pypi/v/sutradb-core.svg?style=flat&color=blue&logo=pypi)](https://pypi.org/project/sutradb-core/)
[![Docker Pulls](https://img.shields.io/docker/pulls/samcodesai/sutradb.svg?style=flat&logo=docker&logoColor=white&color=2496ED)](https://hub.docker.com/r/samcodesai/sutradb)
[![Docker Image Version](https://img.shields.io/docker/v/samcodesai/sutradb/5.5.0?style=flat&logo=docker&logoColor=white&label=docker%20image&color=2496ED)](https://hub.docker.com/r/samcodesai/sutradb)
[![GHCR](https://img.shields.io/badge/GHCR-ghcr.io%2Fsam--codesai%2Fsutradb-blue?logo=github)](https://github.com/Sam-CodesAI/SutraDB/pkgs/container/sutradb)
[![GitHub Release](https://img.shields.io/github/v/release/Sam-CodesAI/SutraDB?style=flat&logo=github&color=purple)](https://github.com/Sam-CodesAI/SutraDB/releases/tag/v5.5.0)
[![Tests](https://img.shields.io/badge/Tests-332%2F332%20Passing-brightgreen)](https://github.com/Sam-CodesAI/SutraDB)
[![Rust](https://img.shields.io/badge/Engine-Rust%20SIMD%20%2B%20gRPC-orange?logo=rust)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?logo=python)](https://python.org)
[![Cluster](https://img.shields.io/badge/Distributed-Multi--Node%20Raft-purple)](https://github.com/Sam-CodesAI/SutraDB)
[![SIMD](https://img.shields.io/badge/SIMD-AVX512%20%7C%20AVX2%20%7C%20NEON-blue)](https://github.com/Sam-CodesAI/SutraDB)
[![ColBERT](https://img.shields.io/badge/Multi--Vector-ColBERT%20MaxSim-red)](https://github.com/Sam-CodesAI/SutraDB)
[![gRPC](https://img.shields.io/badge/Protocol-Binary%20gRPC%20HTTP%2F2-purple?logo=grpc)](https://grpc.io)
[![Quantization](https://img.shields.io/badge/RAM%20Savings-Up%20to%2064x%20(PQ)-green)](https://github.com/Sam-CodesAI/SutraDB)
[![Workbench](https://img.shields.io/badge/Workbench-Zero--Build%20%2Fui-38bdf8)](https://github.com/Sam-CodesAI/SutraDB)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **सूत्र (Sūtra)**: *An aphorism or thread of knowledge designed to hold vast wisdom in the most concise, unbreakable form.*

**SutraDB** is an **embedded Rust + Python hybrid vector database** engineered for **local RAG**, AI agents, and voice pipelines with **zero Docker** required. Inside a single PyO3 ABI3 wheel (or standalone Rust binary), it combines dense **HNSW**, **BM25** lexical search, **WAND sparse** retrieval, and **ColBERT** MaxSim reranking over contiguous memory slabs with runtime AVX-512, AVX2+FMA, and ARM NEON SIMD. A cost-based `RoaringBitmap` planner delivers **sub-millisecond metadata filtering** (`0.187 ms` P50 at 100K × 768d, 100% Recall@10), backed by 32× 1-bit binary quantization, `.sutra.idx` sidecars, and a `SWB1` binary WAL.

---

## Quickstart

Install prebuilt Linux, macOS, or Windows wheels (`pip install sutradb-core==5.5.0`) and run embedded hybrid search with **zero Docker** setup:

```bash
pip install sutradb-core==5.5.0
```

```python
from sutradb import SutraDB
db = SutraDB(persist_directory="./sutra_data")
col = db.get_or_create_collection("kb")
col.add_texts(["Rust HNSW with Roaring filtering", "BM25 + ColBERT local RAG"], metadatas=[{"tier": "prod"}, {"tier": "prod"}])
print(col.search("sub-millisecond metadata filtering", filter={"tier": "prod"}, top_k=2))
```

### Optional: 1-Click Workbench & Standalone Server (`:8765` / `:50051`)

```bash
docker run -d --name sutradb -p 8765:8765 -p 50051:50051 -v sutra_data:/data samcodesai/sutradb:5.5.0
# Developer Workbench: http://localhost:8765/ui
```

---

## Verified 100K × 768d Benchmarks

Measured on a 2-vCPU Linux `x86_64` host (`AMD EPYC 7763`, `AVX2 / FMA`, `Python 3.14.2`, Cosine metric, `top_k=10`). Reproducible scripts live in [`benches/scale_benchmark.py`](benches/scale_benchmark.py) and [`benches/benchmark.py`](benches/benchmark.py).

### Out-of-L3-Cache Scale (`100,000` vectors × `768` dims — `293.0 MiB` raw FP32)

| Variant | Ingest / Build | P50 (ms) | P99 (ms) | QPS | Vector/Index RAM | Recall@10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **NumPy** (Exact GEMV baseline) | - | 35.019 | 56.064 | 27.4 | 293.0 MiB | 100.00% |
| **SutraDB Flat** (Rust AVX2 SIMD) | 29,228 docs/s | 14.236 | 21.737 | 68.1 | 293.0 MiB | 100.00% |
| **SutraDB HNSW Unfiltered** (`m=16`) | 1,063 docs/s (build) | **0.417** | **0.645** | **2,291.2** | 326.2 MiB | **100.00%** |
| **SutraDB HNSW 1%-Filtered** (Cost-Based Roaring SIMD) | 1,063 docs/s (build) | **0.187** | **0.297** | **4,913.9** | 326.2 MiB | **100.00%** |
| **SutraDB SQ8** (Mmap-Evicted 4x) | 18,966 docs/s | 6.516 | 10.836 | 141.5 | 73.2 MiB | 97.90% |
| **SutraDB 1-Bit BQ** (Mmap-Evicted 32x) | 19,243 docs/s | 2.043 | 3.580 | 436.2 | **9.2 MiB** | 92.10% |

* **Instant-Restart Snapshot (`.sutra` + `.sutra.idx` sidecar):** Saves `308.3 MiB` + `14.9 MiB` in `1,991 ms` and restores full HNSW search with exact parity in **`3,066 ms`** (vs. `94.1 s` rebuilding HNSW on 2 vCPUs).

### Multi-Modal Suite (`10,000` vectors × `128` dims)

| Variant | Ingestion (docs/s) | P50 (ms) | P99 (ms) | QPS | Vector/Index RAM | Recall@10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **NumPy** (Exact Brute-Force) | 3,749,940 | 0.153 | 0.263 | 6,285 | 4.9 MB | 100.00% |
| **SutraDB Flat** (Rust SIMD) | 15,237 | 0.480 | 0.618 | 2,068 | 4.9 MB | 100.00% |
| **SutraDB HNSW Graph ANN** (`m=32`) | 710 (`741` build) | 0.321 | 0.402 | 3,075 | 9.0 MB | 99.20% |
| **SutraDB SQ8** (4x Quantized) | 16,103 | 0.467 | 0.571 | 2,117 | 1.2 MB | 98.70% |
| **SutraDB 1-Bit BQ** (32x POPCNT) | 16,564 | 0.332 | 0.465 | 3,008 | 0.2 MB | 91.80% |
| **SutraDB PQ** (`M=16`, 32x ADC) | 2,471 | 0.344 | 0.399 | 2,907 | 0.2 MB | 90.30% |
| **SutraDB BM25 Lexical** | 16,287 | 0.488 | 0.634 | 2,032 | 1.9 MB | N/A |
| **SutraDB Hybrid RRF** (Dense+BM25) | 16,209 | 1.275 | 1.468 | 778 | 4.9 MB | N/A (Hybrid) |

---

## Chaos & Durability Verification

Verified with [`benches/peak_stress_test.py`](benches/peak_stress_test.py) against `target/release/sutradb-server` and `sutradb-core==5.5.0`:

| Torture Scenario | Result | Verified Outcome |
| :--- | :---: | :--- |
| **32-Client Concurrent Storm** (`20s`, RBAC + Prometheus + live `compact()`/`save()`) | **PASS** | `3,055 ops` (`152.75 ops/s`), `0` panics, `0` HTTP 500s, exact `/metrics` parity (`2,389` queries) |
| **3× OS `SIGKILL` (`kill -9`) WAL Recovery** | **PASS** | **`744 / 744`** HTTP-200 acknowledged surviving docs recovered (**100%** durability, `0` ghosts) |
| **50% Churn + Compact + Cold Reload** (`20K × 256d → 10K → 15K`) | **PASS** | `0` ghost docs across HNSW / 1%-filter / BM25 / SQ8 / BQ, HNSW Recall@10 = **`98.20%`** |
| **10,000-Query PyO3 Memory-Leak Soak** | **PASS** | `RSS@1K = 291.32 MiB → RSS@10K = 291.32 MiB` (**`+0.00 MiB` delta**) |
| **Adversarial Fuzz & Security Gauntlet** | **PASS** | `30 / 30` probes cleanly rejected/contained (`NaN`/`1e50`, `../`/NUL, ReDoS regex, cross-tenant RBAC, corrupt `.sutra`/`.sutra.idx`) |

---

## When to Use SutraDB

* **Use SutraDB when:** You want an embedded, single-process Rust + Python vector DB for **local RAG**, voice AI agents, desktop/edge apps, or low-latency microservices needing dense + BM25 + WAND sparse + ColBERT hybrid retrieval and sub-millisecond filtered queries without Docker or cloud roundtrips (up to ~100K–1M vectors per node).
* **Use something else when:** You need petabyte/100M+ horizontal auto-sharding with multi-segment LSM background compaction (planned for SutraDB v6.0 — consider Qdrant/Milvus today) or deep relational ACID SQL joins across tables (consider `pgvector` in PostgreSQL).

---

## Architecture

```
                               CLIENT REQUEST
       [ Text Query: "P99 latency bug" | Vector: [0.12, ...] | Namespace: "tenant_42" ]
       [ Metadata Filter: {"status": "resolved", "priority": {"$lte": 2}} ]
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │   Rust Engine Core (GIL-Free)│
                       └──────────────┬───────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌──────────────────┐        ┌──────────────────┐         ┌──────────────────┐
│ Roaring Secondary│        │ Dense / Cost-    │         │ Sparse BM25 &    │
│ & AST Filter     │        │ Based HNSW/SIMD  │         │ WAND SPLADE Core │
└────────┬─────────┘        └────────┬─────────┘         └────────┬─────────┘
         │                           │                            │
         │ Bitmap Mask               │ Exact / ANN Scores         │ Sparse Scores
         │ (valid_count <= 2048      │ (AVX-512/AVX2/NEON         │ (Block-Max WAND
         │  bypasses HNSW hops)      │  or 2-Stage BQ/SQ8/PQ)     │  & ColBERT MaxSim)
         └─────────────┬─────────────┴──────────────┬─────────────┘
                       │                            │
                       ▼                            ▼
                    ┌───────────────────────────────┐
                    │ Reciprocal Rank Fusion (RRF)  │
                    │  & Weighted Linear Scorer     │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Ranked Final Results + WAL/   │
                    │ .sutra.idx Atomic Sidecar     │
                    └───────────────────────────────┘
```

* **Runtime SIMD Auto-Dispatch & Single-Slab Arenas:** Contiguous `VectorArena`, `SQ8Arena`, `BQArena`, and `PQArena` slabs paired with AVX-512, AVX2+FMA, and ARM NEON kernels (`std::arch`) + Rayon parallelism with zero Python GIL contention.
* **Linguistic NLP Analyzer Pipeline:** Zero-dependency English Porter stemmer, Spanish/French/German suffix stemmers, CJK bigrams, and code identifier splitting (`getUserProfile` → `["get", "user", "profile"]`).
* **Quantization Suite:** **SQ8** (4× RAM reduction, `97.90%` Recall@10 at 100K×768d), **1-Bit BQ** (32× RAM reduction with hardware `POPCNT` and two-stage reranking), and **PQ** (up to 64× RAM reduction via subspace k-means ADC).
* **Persistence, Auth & Observability:** Atomic `.sutra.idx` sidecars, buffered `SWB1` binary WAL with CRC32 partial-tail auto-healing, constant-time API key RBAC (`admin` | `write` | `read`), Prometheus `0.0.4` histograms (`GET /metrics`), and zero-build Developer Workbench (`/ui`).

---

## Detailed Usage Examples (Python, LangChain & LlamaIndex)

### 1. Local Source Build (`maturin`)

```bash
git clone https://github.com/Sam-CodesAI/SutraDB.git
cd SutraDB
maturin develop --release
```

### 2. Automated Text Ingestion & Instant Search (`add_texts` & `search`)

SutraDB features an end-to-end automated embedding and ingestion pipeline. Ingest raw text strings directly without manual vectorization:

```python
from sutradb import SutraDB, DefaultEmbeddingFunction, OpenAIEmbeddingFunction

# Initialize SutraDB with disk persistence
db = SutraDB(persist_directory="./sutra_data")

# Create collection (defaults to zero-dependency 384-dim DefaultEmbeddingFunction)
col = db.get_or_create_collection("support_kb")

# 1. Ingest raw text strings with automatic embedding & BM25 indexing
col.add_texts(
    texts=[
        "Rust delivers memory safety without garbage collection and zero-cost abstractions.",
        "Python is the lingua franca of machine learning, PyTorch, and AI agents.",
        "SutraDB combines SIMD hardware acceleration with multi-modal hybrid retrieval."
    ],
    metadatas=[
        {"tier": "gold", "lang": "rust"},
        {"tier": "silver", "lang": "python"},
        {"tier": "gold", "lang": "hybrid"}
    ],
    namespace="tenant_enterprise"
)

# 2. Instant multi-modal search directly with query string!
results = col.search(
    query="vector search in rust with SIMD",
    namespace="tenant_enterprise",
    filter={"tier": "gold"},
    top_k=2
)

for r in results:
    print(f"[{r.score:.4f}] {r.text} | Metadata: {r.metadata}")

# Pluggable embeddings: OpenAI, FastEmbed (ONNX local + SPLADE), HuggingFace:
# col_openai = db.create_collection("openai_docs", embedding_function=OpenAIEmbeddingFunction())
```

### 3. Multi-Tenant Hybrid Search & SQ8 Quantization

```python
from sutradb import SutraDB, Document

# 1. Initialize SutraDB with disk persistence
db = SutraDB(persist_directory="./sutra_data")
col = db.get_or_create_collection("enterprise_kb", dimension=4, metric="cosine")

# 2. Insert multi-tenant documents
col.insert([
    Document(
        id="doc_1",
        namespace="tenant_acme",
        vector=[0.95, 0.05, 0.10, 0.00],
        text="Deploying Kubernetes microservices and container ingress",
        metadata={"category": "devops", "env": "prod"}
    ),
    Document(
        id="doc_2",
        namespace="tenant_globex",
        vector=[0.02, 0.98, 0.05, 0.01],
        text="PostgreSQL connection pooling and query optimization",
        metadata={"category": "database", "env": "prod"}
    )
])

# 3. Build HNSW index for O(log N) ANN vector search
col.build_hnsw(m=16, ef_construction=64, ef_search=32)

# 4. Enable SQ8 8-bit quantization (75% RAM savings)
col.enable_sq8()

# 5. Hybrid search isolated to tenant_acme
results = col.query(
    vector=[1.0, 0.0, 0.0, 0.0],
    text="Kubernetes microservices",
    namespace="tenant_acme",
    filter={"category": "devops"},
    top_k=5,
    hybrid=True
)

for r in results:
    print(f"[{r.score:.4f}] ({r.namespace}) {r.id}: {r.text}")
```

### 4. Direct Native Rust Acceleration (Python)

```python
from sutradb import NativeCollection

# Direct zero-overhead C/Rust engine
col = NativeCollection("fast_kb", dimension=128, metric="cosine")
col.insert(docs, vectors)
col.build_hnsw()
results = col.query(vector=q_vec, top_k=10, use_ann=True)
```

### 5. Official LangChain VectorStore Integration

```python
from sutradb import SutraDBVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = SutraDBVectorStore(
    collection_name="production_rag",
    dimension=1536,
    embedding_function=embeddings,
    persist_directory="./sutra_data",
    hybrid=True,  # Combines dense embeddings + BM25Okapi via RRF
)

# Standard similarity search or Maximal Marginal Relevance (MMR)
results = store.similarity_search("How to optimize vector search?", k=5)
diverse_results = store.max_marginal_relevance_search("P99 latency", k=4, fetch_k=20)
retriever = store.as_retriever(k=3)
```

### 6. Official LlamaIndex VectorStore Integration

```python
from sutradb import SutraDBLlamaIndexVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

vector_store = SutraDBLlamaIndexVectorStore(
    collection_name="knowledge_vault",
    dimension=1536,
    hybrid=True,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Query with hybrid mode
query_engine = index.as_query_engine(vector_store_query_mode="hybrid")
response = query_engine.query("Explain memory mapping persistence")
```

### 7. Nested JSON Metadata & Array Filtering

Filter nested object paths with dot-notation and advanced operators:

```python
results = col.query(
    text="distributed microservices",
    filter={
        "user.profile.city": "San Francisco",
        "user.profile.score": {"$gte": 90.0},
        "tags": {"$all": ["cloud", "docker"]},
        "tags": {"$size": 3},
        "env": {"$in": ["prod", "staging"]}
    },
    top_k=5
)
```

### 8. 1-Bit Binary Quantization (32x RAM Reduction & Two-Stage Search)

Compress high-dimensional vectors by 32x and evaluate Hamming candidate distances via hardware `POPCNT`:

```python
# Enable 1-bit quantization
col.enable_bq()

# Queries automatically run Two-Stage Search:
# Stage 1: Fast Hamming distance scan over 1-bit bitmasks to select top candidates
# Stage 2: Exact full-precision distance reranking of top candidate pool
results = col.query(vector=query_vector, top_k=5)
```

### 9. Geo-Spatial Metadata Filtering ($geo_within & $geo_box)

Filter vectors using great-circle Haversine distances in meters or rectangular bounding boxes:

```python
# Radius query within 5,000 meters of San Francisco center
sf_results = col.query(
    vector=query_vector,
    filter={
        "location": {
            "$geo_within": {
                "center": [37.7749, -122.4194],
                "radius_meters": 5000.0
            }
        }
    },
    top_k=10
)

# Bounding box query covering SF Bay Area
bay_results = col.query(
    text="local restaurants",
    filter={
        "location": {
            "$geo_box": {
                "bottom_left": [37.70, -122.50],
                "top_right": [37.85, -122.20]
            }
        }
    },
    top_k=10
)
```

### 10. Product Quantization (PQ) & Asymmetric Distance Computation (ADC)

Compress high-dimensional vectors by up to **64x** using sub-space decomposition and evaluate similarities in sub-milliseconds via L1-cache Look-Up Tables (LUT):

```python
# 1. Train codebook using k-means across sub-spaces
codebook = col.train_pq(num_subspaces=16, k_centroids=256, max_iter=20)

# 2. Enable PQ mode for the collection
col.enable_pq(codebook)

# 3. Two-Stage ADC Query:
# Stage 1: Asymmetric Distance Computation via precomputed query-to-centroid Look-Up Table (LUT)
# Stage 2: Top candidate pool reranked with full-precision vectors
results = col.query(vector=query_vector, top_k=5)
```

### 11. Linguistic NLP Pipeline & Hardware SIMD Dispatch

Configure advanced stemming, stopword pruning, identifier splitting, and CJK tokenization directly on collections:

```python
from sutradb import SutraDB, Analyzer, AnalyzerConfig

# 1. Initialize collection with custom multi-lingual analyzer
db = SutraDB()
col = db.create_collection(
    name="multilingual_kb",
    dimension=128,
    analyzer={
        "language": "english",          # "english", "spanish", "french", "german"
        "stemmer": True,                # Algorithmic Porter / Romance suffix stemmer
        "stopwords": True,              # Pre-compiled stopword removal
        "split_identifiers": True,      # getUserProfile -> ["get", "user", "profile"]
        "cjk_bigrams": True             # 自然语言处理 -> ["自然", "语言", "处理"]
    }
)

# 2. Inspect host CPU SIMD acceleration features at runtime
from sutradb._sutradb import detect_simd, batch_vector_distances
simd_level = detect_simd()
print(f"Host SIMD Acceleration: {simd_level}")  # e.g., "AVX-512", "AVX2", "ARM NEON"

# 3. Multicore batch distance calculation (GIL-free via Rayon)
distances = batch_vector_distances(
    query=[0.1] * 128,
    vectors=[[0.2] * 128 for _ in range(10000)],
    metric="cosine",
    is_normalized=True
)
```

### 12. Distributed Multi-Node Cluster & Raft Consensus

SutraDB v5.0 introduces native multi-shard horizontal partitioning and Raft consensus replication:

```python
from sutradb import DistributedCluster

# Initialize multi-node cluster coordinator
cluster = DistributedCluster(node_id="coordinator-0", vnodes_per_node=64)

# Create collection partitioned across 3 logical shards
col = cluster.create_collection("enterprise_wiki", dimension=384, shard_count=3)

# 1. Ingest texts with automatic consistent-hash partition routing
col.add_texts(
    texts=[
        "Kubernetes container orchestration and ingress controllers.",
        "Raft consensus protocol for distributed state machines.",
        "ColBERT multi-vector token late interaction retrieval."
    ],
    metadatas=[{"tier": "gold"}, {"tier": "gold"}, {"tier": "silver"}],
    namespace="tenant_alpha"
)

# 2. Parallel scatter-gather search across all shards with global top-K RRF fusion
results = col.search(
    query="Raft consensus in distributed systems",
    namespace="tenant_alpha",
    top_k=2
)

for r in results:
    print(f"[{r.score:.4f}] {r.text} | Metadata: {r.metadata}")
```

---

## 🛠️ Zero-Build Embedded Developer Workbench (`/ui`)

SutraDB v5.5.0 ships with an industrial, **zero-build, zero-external-CDN Developer Workbench** served natively directly from the engine at `http://localhost:8765/ui`.

```
┌────────────────────────────────────────────────────────────────────────┐
│ ⚡ SutraDB Console | Developer Workbench                       v5.5.0  │
├────────────┬─────────────┬──────────────┬──────────┬──────────┬────────┤
│ Hybrid Lab │ Batch Ingest│ Doc Explorer │  2D PCA  │ Admin    │ Code   │
├────────────┴─────────────┴──────────────┴──────────┴──────────┴────────┤
│ Query: [ "p99 latency in SIMD vector engine"                          ]│
│ Filter: [ {"status": "resolved", "tier": "gold"}                      ]│
│ RRF Balance: [===●=======] Dense 0.40 / BM25 0.60    Latency: ⚡ 0.12ms │
├────────────────────────────────────────────────────────────────────────┤
│ 1. [RRF: 0.9412] "Hardware SIMD auto-dispatch (AVX-512 / NEON)..."     │
│ 2. [RRF: 0.8120] "ColBERT token late-interaction MaxSim scoring..."    │
└────────────────────────────────────────────────────────────────────────┘
```

### 1-Click Launch Options

**Option A: Instant Docker Container (Recommended)**
```bash
docker run -d \
  --name sutradb \
  -p 8765:8765 \
  -p 50051:50051 \
  -v sutra_data:/data \
  samcodesai/sutradb:latest

# Open your browser at:
# 👉 http://localhost:8765/ui
```

**Option B: Python CLI**
```bash
pip install sutradb-core
python3 -m sutradb.server 8765
```

---

### Interactive Workbench Panels

| Panel | Core Features & Capabilities |
| :--- | :--- |
| **⚡ Hybrid Query Lab** | • Real-time Dense Vector + BM25 Lexical search execution.<br>• Interactive Reciprocal Rank Fusion (RRF) balance slider (`0.0` dense-only to `1.0` BM25-only).<br>• Full AST JSON metadata filter expression support (`{"tier": "gold", "price": {"$lte": 100}}`).<br>• Real-time microsecond query latency stopwatch (`⚡ 0.12ms`). |
| **📥 Ingestion & Batch Upsert** | • Single-document manual insertion with raw vector and metadata editors.<br>• Bulk JSON array and newline-delimited JSONL drag-and-drop or copy-paste ingestion zone.<br>• Automatic schema validation and vector dimension verification. |
| **🗂️ Document Explorer** | • Paginated collection document browser (`GET /collections/{name}/documents`).<br>• Real-time Document ID filtering and search.<br>• In-place modal metadata patcher (`PATCH /collections/{name}/documents/{id}/metadata`).<br>• 1-click document deletion with automatic secondary index and BM25 term pruning. |
| **📊 2D Vector Projection (PCA)** | • Zero-dependency in-browser HTML5 Canvas covariance eigenvector decomposition.<br>• Color-coded cluster scatter plot with responsive zoom and pan.<br>• Interactive hover tooltips displaying Document ID, text preview, and metadata. |
| **🛠️ Collection Administration** | • Live collection inventory with document counters, dimensionality, and storage breakdown.<br>• 1-click index builders: **HNSW Graph**, **SQ8 Quantization** (75% RAM savings), **1-Bit BQ** (32x savings), **PQ** (up to 64x savings), and **WAL Compaction**.<br>• Collection drop confirmation modal. |
| **💻 Dynamic Code Generator** | • Automatically synchronizes with your active query, filters, and RRF weights.<br>• Live, syntax-highlighted snippets ready to copy for **Python SDK**, **LangChain**, **cURL REST**, and **Rust gRPC**. |

### 🔒 Zero-Build & Air-Gapped Supply-Chain Guarantee
- **No Node.js / npm required:** Embedded entirely inside `sutradb/ui.py` using standard HTML5, CSS Grid, and vanilla ES6.
- **Zero External CDNs:** 100% offline and air-gapped compliant. No external fonts, icons, or JavaScript bundles that could leak telemetry or cause supply-chain poisoning.
- **Unprivileged Runtime:** Packaged inside Docker running strictly under non-root UID `10001:10001`.

---

## ⚡ High-Throughput Binary gRPC Server (`sutradb-server`)

SutraDB v4.0 introduces a native, asynchronous **binary gRPC engine** built directly in Rust using `tonic` (HTTP/2) and `prost` (Protocol Buffers). By streaming raw IEEE-754 binary floats directly across network interfaces, SutraDB eliminates the ~3–5ms JSON serialization overhead of traditional REST endpoints, driving query round-trip latency below **0.4ms**.

### Running the Standalone Native Server

```bash
# Build & run high-performance release binary
cargo run --release --bin sutradb-server -- --host 0.0.0.0 --port 50051 --data-dir ./sutra_data

# Or inspect options
sutradb-server --help
```

### Protocol Buffers Contract (`proto/sutra.proto`)

```protobuf
syntax = "proto3";
package sutra;

service SutraService {
  rpc Query(QueryRequest) returns (QueryResponse);
  rpc BatchQuery(BatchQueryRequest) returns (BatchQueryResponse);
  rpc Insert(InsertRequest) returns (InsertResponse);
  rpc Delete(DeleteRequest) returns (DeleteResponse);
  rpc Health(HealthRequest) returns (HealthResponse);
  rpc CreateCollection(CreateCollectionRequest) returns (CreateCollectionResponse);
  rpc DropCollection(DropCollectionRequest) returns (DropCollectionResponse);
  rpc ListCollections(ListCollectionsRequest) returns (ListCollectionsResponse);
  rpc GetStats(CollectionStatsRequest) returns (CollectionStatsResponse);
  rpc BatchDistances(BatchDistanceRequest) returns (BatchDistanceResponse);
}
```

---

## 🌐 REST Microservice Endpoints

- `GET /ui` & `GET /dashboard` - Zero-Build Embedded Developer Workbench
- `GET /ui/stats` - Telemetry metrics for developer workbench
- `GET /health` - Health check & collection registry
- `GET /metrics` - Prometheus metrics exporter
- `GET /collections/{name}/documents` - Paginated document explorer (`limit`, `offset`)
- `GET /collections/{name}/documents/{id}` - Single document retrieval
- `DELETE /collections/{name}/documents/{id}` - Document deletion with index cleanup
- `GET /collections/{name}/sample_vectors` - Sample vectors for 2D visual projection
- `POST /collections/{name}/upsert` - Idempotent upsert
- `PATCH /collections/{name}/documents/{id}/metadata` - In-place metadata patch
- `POST /collections/{name}/hnsw` - Trigger HNSW graph build
- `POST /collections/{name}/pq` - Train & enable Product Quantization (up to 64x RAM reduction)
- `POST /collections/{name}/bq` - Enable 1-Bit Binary Quantization (32x RAM reduction)
- `POST /collections/{name}/sq8` - Enable SQ8 vector quantization
- `POST /collections/{name}/query` - Hybrid, vector, or lexical search

---

## 🚀 Production Deployment (Docker & Kubernetes)

SutraDB is packaged as a supply-chain hardened, multi-stage container image conforming strictly to production security invariants:
- **Unprivileged Execution:** Runs strictly under non-root UID/GID `10001:10001`.
- **Zero Attack Surface:** All compilers, Rust toolchains, and package managers are excluded from the runtime image (`python:3.12-slim`).
- **Dual-Daemon Orchestration:** Native binary gRPC daemon on port `50051` and FastAPI REST / Developer Workbench on port `8765`.
- **Graceful Signal Handling:** Intercepts `SIGTERM` and `SIGINT` to safely flush Write-Ahead Logs and drain active connections without data loss.

### Official Container Images

| Registry | Image Name | Pull Command |
| :--- | :--- | :--- |
| **Docker Hub** | `samcodesai/sutradb` | `docker pull samcodesai/sutradb:5.5.0` |
| **GitHub Packages (GHCR)** | `ghcr.io/sam-codesai/sutradb` | `docker pull ghcr.io/sam-codesai/sutradb:5.5.0` |

### 1. Docker Compose (Quickstart)

```bash
# Clone and launch dual daemons (gRPC :50051 & REST :8765) with persistent volume
docker compose up -d

# Verify container status and automated healthcheck
docker compose ps
docker compose logs -f
```

### 2. Standalone Docker Run

```bash
# Run in dual-daemon mode (default with Developer Workbench on :8765)
docker run -d \
  --name sutradb \
  -p 50051:50051 \
  -p 8765:8765 \
  -v sutra_data:/data \
  samcodesai/sutradb:latest

# Run strictly as high-throughput binary gRPC engine
docker run -d \
  --name sutradb-grpc \
  -p 50051:50051 \
  -v sutra_data:/data \
  -e SUTRA_MODE=grpc \
  samcodesai/sutradb:latest

# Run strictly as REST microservice + Developer Workbench
docker run -d \
  --name sutradb-rest \
  -p 8765:8765 \
  -v sutra_data:/data \
  -e SUTRA_MODE=rest \
  samcodesai/sutradb:latest
```

### 3. Kubernetes Deployment (`k8s/`)

Deploy SutraDB as a production StatefulSet with persistent volume claims, automated liveness/readiness probes, and unprivileged security context:

```bash
# Apply with Kustomize
kubectl apply -k k8s/

# Or apply individual manifests
kubectl apply -f k8s/sutradb-namespace.yaml
kubectl apply -f k8s/sutradb-configmap.yaml
kubectl apply -f k8s/sutradb-service.yaml
kubectl apply -f k8s/sutradb-statefulset.yaml

# Check rollout status
kubectl rollout status statefulset/sutradb -n sutradb
```

---

## 🧪 Test Suite

Run both native Rust and Python test suites (`332/332` passing):

```bash
# 1. Rust engine, cluster & gRPC integration tests (98 tests)
cargo test --release

# 2. Python test suite (234 tests)
pytest -v
```

---

## 📜 License

MIT License. Engineered by [Samarth](https://github.com/Sam-CodesAI).
