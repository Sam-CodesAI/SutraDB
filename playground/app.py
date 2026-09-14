"""
SutraSearch Playground — Semantic Search Engine powered by SutraDB.

Seeds a rich knowledge base of 50+ programming/tech concepts, generates
deterministic synthetic embeddings via text hashing, and serves a beautiful
web UI with hybrid vector + BM25 search.

Usage:
    python playground/app.py [port]
"""

import hashlib
import json
import math
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np

# Ensure the parent repo is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sutradb import SutraDB, Document


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Embedding Generator
# ──────────────────────────────────────────────────────────────────────────────

EMBED_DIM = 64


def text_to_vector(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """
    Generates a deterministic, semantically-meaningful embedding from text.
    Uses token hashing with frequency weighting so documents sharing
    keywords have higher cosine similarity.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    if not tokens:
        vec[0] = 1.0
        return vec

    for token in tokens:
        # Clean token
        token = "".join(c for c in token if c.isalnum())
        if not token:
            continue
        # Hash to bucket
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        bucket = h % dim
        # Sign from second hash
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        # IDF-like weight: rarer chars → higher weight
        weight = 1.0 + math.log(1 + len(token)) * 0.5
        vec[bucket] += sign * weight

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec /= norm
    else:
        vec[0] = 1.0
    return vec


# ──────────────────────────────────────────────────────────────────────────────
# Knowledge Base: 50+ Tech Concepts
# ──────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    # ── AI / Machine Learning ────────────────────────────────────────────────
    {
        "id": "ai-001",
        "text": "Transformer architecture uses self-attention mechanisms to process sequences in parallel, enabling models like GPT and BERT to capture long-range dependencies efficiently",
        "metadata": {"category": "ai", "difficulty": "advanced", "tags": ["transformers", "attention", "nlp"]}
    },
    {
        "id": "ai-002",
        "text": "Retrieval Augmented Generation (RAG) combines a vector database retriever with a language model generator to produce factually grounded responses using external knowledge",
        "metadata": {"category": "ai", "difficulty": "advanced", "tags": ["rag", "llm", "retrieval"]}
    },
    {
        "id": "ai-003",
        "text": "Fine-tuning large language models with LoRA (Low-Rank Adaptation) reduces trainable parameters by 99% while preserving model quality on domain-specific tasks",
        "metadata": {"category": "ai", "difficulty": "advanced", "tags": ["lora", "fine-tuning", "llm"]}
    },
    {
        "id": "ai-004",
        "text": "Convolutional neural networks (CNNs) apply learnable filters across spatial dimensions to detect features like edges, textures, and objects in images",
        "metadata": {"category": "ai", "difficulty": "intermediate", "tags": ["cnn", "computer-vision", "deep-learning"]}
    },
    {
        "id": "ai-005",
        "text": "Gradient descent optimization iteratively adjusts model weights by computing partial derivatives of the loss function with respect to each parameter",
        "metadata": {"category": "ai", "difficulty": "beginner", "tags": ["optimization", "backpropagation", "training"]}
    },
    {
        "id": "ai-006",
        "text": "Vector embeddings represent words, sentences, or documents as dense floating-point arrays in high-dimensional space where semantic similarity maps to geometric proximity",
        "metadata": {"category": "ai", "difficulty": "intermediate", "tags": ["embeddings", "vector-search", "nlp"]}
    },
    {
        "id": "ai-007",
        "text": "Reinforcement learning from human feedback (RLHF) trains reward models on preference data to align language model outputs with human intent and safety guidelines",
        "metadata": {"category": "ai", "difficulty": "advanced", "tags": ["rlhf", "alignment", "llm"]}
    },
    {
        "id": "ai-008",
        "text": "Diffusion models generate high-quality images by learning to reverse a gradual noise corruption process, used in Stable Diffusion and DALL-E",
        "metadata": {"category": "ai", "difficulty": "advanced", "tags": ["diffusion", "image-generation", "generative-ai"]}
    },
    {
        "id": "ai-009",
        "text": "K-nearest neighbors (KNN) classification assigns labels based on majority vote of the K closest training examples measured by distance metrics like Euclidean or cosine",
        "metadata": {"category": "ai", "difficulty": "beginner", "tags": ["knn", "classification", "algorithms"]}
    },
    {
        "id": "ai-010",
        "text": "Mixture of Experts (MoE) architectures route input tokens to specialized sub-networks using a gating mechanism, enabling trillion-parameter models with efficient inference",
        "metadata": {"category": "ai", "difficulty": "advanced", "tags": ["moe", "scaling", "architecture"]}
    },

    # ── Databases ────────────────────────────────────────────────────────────
    {
        "id": "db-001",
        "text": "PostgreSQL MVCC (Multi-Version Concurrency Control) stores multiple row versions to allow concurrent reads and writes without locking, using transaction snapshots for isolation",
        "metadata": {"category": "databases", "difficulty": "advanced", "tags": ["postgresql", "mvcc", "concurrency"]}
    },
    {
        "id": "db-002",
        "text": "Redis provides sub-millisecond in-memory key-value storage with data structures like sorted sets, streams, and HyperLogLog for caching and real-time analytics",
        "metadata": {"category": "databases", "difficulty": "intermediate", "tags": ["redis", "caching", "in-memory"]}
    },
    {
        "id": "db-003",
        "text": "B-tree indexes organize data in balanced sorted tree structures enabling logarithmic-time lookups, range scans, and ordered retrieval in relational databases",
        "metadata": {"category": "databases", "difficulty": "intermediate", "tags": ["indexing", "b-tree", "query-optimization"]}
    },
    {
        "id": "db-004",
        "text": "Write-ahead logging (WAL) ensures database durability by writing transaction changes to a sequential log before applying them to data pages, enabling crash recovery",
        "metadata": {"category": "databases", "difficulty": "advanced", "tags": ["wal", "durability", "crash-recovery"]}
    },
    {
        "id": "db-005",
        "text": "Vector databases like SutraDB store high-dimensional embeddings and support approximate nearest neighbor search using algorithms like HNSW, IVF, or brute-force scanning",
        "metadata": {"category": "databases", "difficulty": "intermediate", "tags": ["vector-db", "ann", "embeddings"]}
    },
    {
        "id": "db-006",
        "text": "SQL query optimization involves analyzing execution plans, creating composite indexes, avoiding N+1 queries, and leveraging materialized views for expensive aggregations",
        "metadata": {"category": "databases", "difficulty": "intermediate", "tags": ["sql", "optimization", "query-planning"]}
    },
    {
        "id": "db-007",
        "text": "MongoDB document model stores JSON-like BSON documents with nested objects and arrays, supporting flexible schemas and horizontal scaling through automatic sharding",
        "metadata": {"category": "databases", "difficulty": "beginner", "tags": ["mongodb", "nosql", "document-store"]}
    },
    {
        "id": "db-008",
        "text": "Database connection pooling with PgBouncer or HikariCP reuses established TCP connections to reduce handshake overhead and prevent connection exhaustion under high load",
        "metadata": {"category": "databases", "difficulty": "intermediate", "tags": ["connection-pooling", "pgbouncer", "performance"]}
    },
    {
        "id": "db-009",
        "text": "Time-series databases like TimescaleDB and InfluxDB optimize storage and queries for timestamped data using hypertables, automatic partitioning, and downsampling policies",
        "metadata": {"category": "databases", "difficulty": "intermediate", "tags": ["timeseries", "iot", "monitoring"]}
    },
    {
        "id": "db-010",
        "text": "Graph databases like Neo4j model relationships as first-class citizens using nodes and edges, enabling efficient traversal queries for social networks, recommendations, and fraud detection",
        "metadata": {"category": "databases", "difficulty": "intermediate", "tags": ["graph-db", "neo4j", "relationships"]}
    },

    # ── DevOps / Infrastructure ──────────────────────────────────────────────
    {
        "id": "devops-001",
        "text": "Kubernetes orchestrates containerized applications across node clusters using Pods, Deployments, Services, and Ingress resources for automatic scaling, healing, and load balancing",
        "metadata": {"category": "devops", "difficulty": "advanced", "tags": ["kubernetes", "containers", "orchestration"]}
    },
    {
        "id": "devops-002",
        "text": "Docker containers package applications with their dependencies into portable images using layered union filesystems, enabling consistent deployment across development and production environments",
        "metadata": {"category": "devops", "difficulty": "beginner", "tags": ["docker", "containers", "deployment"]}
    },
    {
        "id": "devops-003",
        "text": "CI/CD pipelines automate build, test, and deployment workflows using tools like GitHub Actions, GitLab CI, or Jenkins to deliver code changes safely and rapidly",
        "metadata": {"category": "devops", "difficulty": "intermediate", "tags": ["ci-cd", "automation", "deployment"]}
    },
    {
        "id": "devops-004",
        "text": "Terraform Infrastructure as Code (IaC) defines cloud resources in declarative HCL files, enabling version-controlled, reproducible infrastructure provisioning across AWS, GCP, and Azure",
        "metadata": {"category": "devops", "difficulty": "intermediate", "tags": ["terraform", "iac", "cloud"]}
    },
    {
        "id": "devops-005",
        "text": "Prometheus collects time-series metrics from application endpoints and Kubernetes pods, while Grafana provides customizable dashboards for real-time observability and alerting",
        "metadata": {"category": "devops", "difficulty": "intermediate", "tags": ["prometheus", "grafana", "monitoring"]}
    },
    {
        "id": "devops-006",
        "text": "Service mesh architectures using Istio or Linkerd inject sidecar proxies to handle mutual TLS encryption, traffic routing, circuit breaking, and distributed tracing between microservices",
        "metadata": {"category": "devops", "difficulty": "advanced", "tags": ["service-mesh", "istio", "microservices"]}
    },
    {
        "id": "devops-007",
        "text": "GitOps methodology uses Git repositories as the single source of truth for infrastructure and application state, with operators like ArgoCD automatically reconciling cluster state",
        "metadata": {"category": "devops", "difficulty": "intermediate", "tags": ["gitops", "argocd", "kubernetes"]}
    },
    {
        "id": "devops-008",
        "text": "Load balancing distributes incoming network traffic across multiple servers using algorithms like round-robin, least-connections, or consistent hashing to ensure high availability",
        "metadata": {"category": "devops", "difficulty": "beginner", "tags": ["load-balancing", "ha", "networking"]}
    },

    # ── Programming Languages & Patterns ─────────────────────────────────────
    {
        "id": "prog-001",
        "text": "Rust ownership system uses the borrow checker to enforce memory safety at compile time without garbage collection, preventing data races, dangling pointers, and use-after-free bugs",
        "metadata": {"category": "programming", "difficulty": "advanced", "tags": ["rust", "memory-safety", "systems"]}
    },
    {
        "id": "prog-002",
        "text": "Python async/await concurrency uses an event loop to schedule coroutines for I/O-bound parallelism without threads, ideal for web servers and API clients handling thousands of connections",
        "metadata": {"category": "programming", "difficulty": "intermediate", "tags": ["python", "async", "concurrency"]}
    },
    {
        "id": "prog-003",
        "text": "TypeScript generics enable type-safe reusable components by parameterizing functions, classes, and interfaces over types with constraints like extends and conditional inference",
        "metadata": {"category": "programming", "difficulty": "intermediate", "tags": ["typescript", "generics", "type-safety"]}
    },
    {
        "id": "prog-004",
        "text": "Go goroutines are lightweight user-space threads scheduled by the Go runtime on OS threads, communicating through typed channels for safe concurrent message passing",
        "metadata": {"category": "programming", "difficulty": "intermediate", "tags": ["go", "goroutines", "concurrency"]}
    },
    {
        "id": "prog-005",
        "text": "SOLID principles guide object-oriented design: Single Responsibility, Open-Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion for maintainable codebases",
        "metadata": {"category": "programming", "difficulty": "beginner", "tags": ["solid", "design-patterns", "oop"]}
    },
    {
        "id": "prog-006",
        "text": "Functional programming with immutable data structures, pure functions, and higher-order combinators like map, filter, and reduce enables predictable, parallelizable computations",
        "metadata": {"category": "programming", "difficulty": "intermediate", "tags": ["functional", "immutability", "hof"]}
    },
    {
        "id": "prog-007",
        "text": "WebAssembly (Wasm) compiles C, C++, and Rust to a portable bytecode format that runs in browser sandboxes at near-native speed for compute-intensive web applications",
        "metadata": {"category": "programming", "difficulty": "advanced", "tags": ["wasm", "browser", "performance"]}
    },
    {
        "id": "prog-008",
        "text": "Event-driven architecture decouples producers and consumers using message brokers like Kafka or RabbitMQ, enabling asynchronous processing, replay, and independent service scaling",
        "metadata": {"category": "programming", "difficulty": "intermediate", "tags": ["event-driven", "kafka", "architecture"]}
    },

    # ── Web Development ──────────────────────────────────────────────────────
    {
        "id": "web-001",
        "text": "React Server Components render on the server and stream HTML to the client, reducing JavaScript bundle size while maintaining interactive islands with client components",
        "metadata": {"category": "web", "difficulty": "advanced", "tags": ["react", "rsc", "server-rendering"]}
    },
    {
        "id": "web-002",
        "text": "CSS Container Queries allow components to adapt their layout based on the size of their parent container rather than the viewport, enabling truly reusable responsive components",
        "metadata": {"category": "web", "difficulty": "intermediate", "tags": ["css", "responsive", "components"]}
    },
    {
        "id": "web-003",
        "text": "Next.js App Router uses file-system routing with layouts, loading states, and error boundaries built into the directory structure for composable full-stack React applications",
        "metadata": {"category": "web", "difficulty": "intermediate", "tags": ["nextjs", "routing", "full-stack"]}
    },
    {
        "id": "web-004",
        "text": "OAuth 2.0 authorization framework delegates authentication to identity providers using access tokens, refresh tokens, and scopes for secure third-party API access without sharing credentials",
        "metadata": {"category": "web", "difficulty": "intermediate", "tags": ["oauth", "authentication", "security"]}
    },
    {
        "id": "web-005",
        "text": "GraphQL APIs define typed schemas with queries, mutations, and subscriptions allowing clients to request exactly the data they need in a single round-trip, eliminating over-fetching",
        "metadata": {"category": "web", "difficulty": "intermediate", "tags": ["graphql", "api", "schema"]}
    },
    {
        "id": "web-006",
        "text": "Progressive Web Apps (PWAs) use service workers for offline caching, web app manifests for installability, and push notifications to deliver native-like experiences on the web",
        "metadata": {"category": "web", "difficulty": "intermediate", "tags": ["pwa", "service-workers", "offline"]}
    },
    {
        "id": "web-007",
        "text": "HTTP/3 replaces TCP with QUIC protocol using UDP for zero round-trip connection establishment, multiplexed streams without head-of-line blocking, and built-in TLS 1.3 encryption",
        "metadata": {"category": "web", "difficulty": "advanced", "tags": ["http3", "quic", "networking"]}
    },

    # ── Security ──────────────────────────────────────────────────────────────
    {
        "id": "sec-001",
        "text": "Zero-trust security architecture verifies every request regardless of network location using continuous authentication, micro-segmentation, and least-privilege access policies",
        "metadata": {"category": "security", "difficulty": "advanced", "tags": ["zero-trust", "architecture", "access-control"]}
    },
    {
        "id": "sec-002",
        "text": "SQL injection attacks exploit unsanitized user input in database queries; prevention requires parameterized prepared statements, input validation, and ORM usage",
        "metadata": {"category": "security", "difficulty": "beginner", "tags": ["sql-injection", "web-security", "prevention"]}
    },
    {
        "id": "sec-003",
        "text": "JWT (JSON Web Tokens) encode claims as signed Base64 payloads for stateless authentication, requiring secure key rotation, short expiry, and proper validation of algorithm headers",
        "metadata": {"category": "security", "difficulty": "intermediate", "tags": ["jwt", "authentication", "tokens"]}
    },
    {
        "id": "sec-004",
        "text": "End-to-end encryption (E2EE) ensures only communicating parties can read messages by performing key exchange with protocols like Signal or Double Ratchet, preventing server-side decryption",
        "metadata": {"category": "security", "difficulty": "advanced", "tags": ["encryption", "e2ee", "cryptography"]}
    },
    {
        "id": "sec-005",
        "text": "Container security scanning tools like Trivy and Snyk analyze Docker images for known CVE vulnerabilities in OS packages and application dependencies before deployment",
        "metadata": {"category": "security", "difficulty": "intermediate", "tags": ["containers", "scanning", "vulnerabilities"]}
    },

    # ── Data Engineering ──────────────────────────────────────────────────────
    {
        "id": "data-001",
        "text": "Apache Spark processes large-scale distributed datasets using resilient distributed datasets (RDDs) and DataFrames with lazy evaluation and in-memory computation for batch and streaming",
        "metadata": {"category": "data-engineering", "difficulty": "advanced", "tags": ["spark", "distributed", "big-data"]}
    },
    {
        "id": "data-002",
        "text": "Apache Kafka is a distributed event streaming platform that persists ordered, partitioned log topics for real-time data pipelines, event sourcing, and stream processing at millions of events per second",
        "metadata": {"category": "data-engineering", "difficulty": "advanced", "tags": ["kafka", "streaming", "event-sourcing"]}
    },
    {
        "id": "data-003",
        "text": "ETL pipelines extract data from source systems, transform it through cleaning and enrichment stages, and load it into data warehouses like Snowflake or BigQuery for analytics",
        "metadata": {"category": "data-engineering", "difficulty": "intermediate", "tags": ["etl", "data-warehouse", "analytics"]}
    },
    {
        "id": "data-004",
        "text": "Apache Parquet columnar storage format compresses data 75% smaller than CSV with predicate pushdown and column pruning for fast analytical queries on object storage like S3",
        "metadata": {"category": "data-engineering", "difficulty": "intermediate", "tags": ["parquet", "columnar", "storage"]}
    },
    {
        "id": "data-005",
        "text": "Data lakehouse architecture combines the flexibility of data lakes with ACID transaction guarantees using table formats like Delta Lake, Apache Iceberg, or Apache Hudi",
        "metadata": {"category": "data-engineering", "difficulty": "advanced", "tags": ["lakehouse", "delta-lake", "iceberg"]}
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Database Seeding
# ──────────────────────────────────────────────────────────────────────────────

def seed_database(db: SutraDB) -> "Collection":
    """Seeds SutraDB with the knowledge base and returns the collection."""
    collection = db.get_or_create_collection(
        name="knowledge_base",
        dimension=EMBED_DIM,
        metric="cosine"
    )

    if collection.count() > 0:
        print(f"  \u2713 Collection already seeded with {collection.count()} documents")
        return collection

    docs = []
    for entry in KNOWLEDGE_BASE:
        vec = text_to_vector(entry["text"])
        docs.append(Document(
            id=entry["id"],
            vector=vec,
            text=entry["text"],
            metadata=entry["metadata"]
        ))

    collection.insert(docs)
    print(f"  \u2713 Seeded {collection.count()} documents into 'knowledge_base' collection")
    return collection


# ──────────────────────────────────────────────────────────────────────────────
# HTTP Server
# ──────────────────────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).resolve().parent


class PlaygroundHandler(BaseHTTPRequestHandler):
    """HTTP handler for the SutraSearch Playground."""

    db: SutraDB = None  # type: ignore
    collection: "Collection" = None  # type: ignore

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return None
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def _serve_static(self, filepath: Path) -> None:
        """Serves a static file."""
        if not filepath.exists() or not filepath.is_file():
            self.send_error(404, "File not found")
            return

        ext = filepath.suffix.lower()
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json",
            ".png": "image/png",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
        }
        ct = content_types.get(ext, "application/octet-stream")

        data = filepath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        # API: /api/stats
        if path == "/api/stats":
            col = self.collection
            stats = col.stats()
            categories = sorted(set(
                d.metadata.get("category", "unknown")
                for d in col.documents
            ))
            self._send_json(200, {
                "collections": [stats],
                "categories": categories,
                "total_documents": col.count()
            })
            return

        # Static files: serve index.html at root
        if path in ("", "/", "/index.html"):
            self._serve_static(STATIC_DIR / "index.html")
            return

        # Other static files
        safe_path = path.lstrip("/")
        self._serve_static(STATIC_DIR / safe_path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/search":
            try:
                body = self._read_json() or {}
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON: {e}"})
                return

            text = body.get("text", "").strip()
            mode = body.get("mode", "hybrid")
            category = body.get("category", "")
            top_k = int(body.get("top_k", 10))

            if not text:
                self._send_json(400, {"error": "Search text is required"})
                return

            # Build metadata filter
            filter_spec: Optional[Dict[str, Any]] = None
            if category:
                filter_spec = {"category": category}

            # Generate query vector from text
            query_vector = text_to_vector(text).tolist()

            # Determine search mode
            t_start = time.perf_counter()
            try:
                if mode == "text":
                    results = self.collection.query(
                        text=text,
                        filter=filter_spec,
                        top_k=top_k,
                        hybrid=False
                    )
                elif mode == "vector":
                    results = self.collection.query(
                        vector=query_vector,
                        filter=filter_spec,
                        top_k=top_k,
                        hybrid=False
                    )
                else:
                    # Hybrid (default)
                    results = self.collection.query(
                        vector=query_vector,
                        text=text,
                        filter=filter_spec,
                        top_k=top_k,
                        hybrid=True,
                        rrf_constant=60
                    )
            except ValueError as ve:
                self._send_json(400, {"error": str(ve)})
                return

            elapsed_ms = (time.perf_counter() - t_start) * 1000

            self._send_json(200, {
                "results": [r.to_dict() for r in results],
                "query_time_ms": round(elapsed_ms, 3),
                "mode": mode,
                "query": text
            })
            return

        self._send_json(404, {"error": "Endpoint not found"})

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress noisy access logs, print only errors."""
        if args and str(args[0]).startswith("4") or str(args[0]).startswith("5"):
            sys.stderr.write(f"[ERR] {self.path} -> {args[0]}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║      ⚡ SutraSearch Playground                   ║")
    print("  ║      Hybrid Vector + BM25 Search Engine          ║")
    print("  ║      Powered by SutraDB v2.0                     ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    # Initialize SutraDB
    data_dir = STATIC_DIR / "sutra_data"
    db = SutraDB(persist_directory=str(data_dir))
    print("  → Initializing SutraDB...")
    collection = seed_database(db)

    stats = collection.stats()
    print(f"  → Collection: {stats['name']} | {stats['count']} docs | {stats['dimension']}D | {stats['metric']}")
    print(f"  → Vector memory: {stats['vector_memory_mb']} MB")
    print()

    # Inject into handler
    PlaygroundHandler.db = db
    PlaygroundHandler.collection = collection

    # Start server
    server = HTTPServer(("0.0.0.0", port), PlaygroundHandler)
    print(f"  🌐 Server running at: http://localhost:{port}")
    print(f"  📡 API endpoint:      http://localhost:{port}/api/search")
    print(f"  📊 Stats endpoint:    http://localhost:{port}/api/stats")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down gracefully...")
        server.server_close()


if __name__ == "__main__":
    main()
