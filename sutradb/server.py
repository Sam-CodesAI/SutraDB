"""
Production ASGI HTTP REST server for SutraDB powered by FastAPI and Uvicorn.
Provides bounded payload enforcement (max 10MB), high-throughput asynchronous execution,
and complete document CRUD with vector & lexical query endpoints.
"""

import hmac
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import httpx
from pydantic import BaseModel, Field
import uvicorn

from sutradb import __version__
from sutradb.core import SutraDB, Collection
from sutradb.ui import get_dashboard_html

# Bounded payload maximum size (10MB)
MAX_PAYLOAD_SIZE = 10 * 1024 * 1024
QUERY_DURATION_BUCKETS = (0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5)
_ROLE_RANK = {"read": 1, "write": 2, "admin": 3}


def _escape_prom_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class _RbacPolicy:
    def __init__(
        self,
        key: str,
        role: str,
        collections: Optional[List[str]] = None,
        prefixes: Optional[List[str]] = None,
    ) -> None:
        self.key = key
        self.role = role
        self.rank = _ROLE_RANK[role]
        self.collections = collections
        self.prefixes = prefixes

    def allows_collection(self, name: str) -> bool:
        if self.collections is None and self.prefixes is None:
            return True
        if self.collections is not None:
            for pat in self.collections:
                if pat == "*" or pat == name:
                    return True
                if pat.endswith("*") and name.startswith(pat[:-1]):
                    return True
        if self.prefixes is not None:
            for pref in self.prefixes:
                if name.startswith(pref):
                    return True
        return False


def _load_rbac_policies() -> List[_RbacPolicy]:
    policy_map: Dict[str, _RbacPolicy] = {}
    admin_key = os.getenv("SUTRA_API_KEY", "").strip()
    if admin_key:
        policy_map[admin_key] = _RbacPolicy(key=admin_key, role="admin")

    raw_rbac = os.getenv("SUTRA_RBAC_CONFIG", "").strip()
    if raw_rbac:
        parsed = json.loads(raw_rbac)
        entries = parsed.get("keys", parsed) if isinstance(parsed, dict) and "keys" in parsed and isinstance(parsed["keys"], dict) else parsed
        if not isinstance(entries, dict):
            raise ValueError("SUTRA_RBAC_CONFIG must be a JSON object")
        for k, spec in entries.items():
            key_str = str(k).strip()
            if not key_str:
                raise ValueError("SUTRA_RBAC_CONFIG contains empty key")
            if isinstance(spec, str):
                role = spec.lower()
                if role not in _ROLE_RANK:
                    raise ValueError(f"Unsupported RBAC role: {role}")
                policy_map[key_str] = _RbacPolicy(key=key_str, role=role)
            elif isinstance(spec, dict):
                role = str(spec.get("role", "read")).lower()
                if role not in _ROLE_RANK:
                    raise ValueError(f"Unsupported RBAC role: {role}")
                cols = spec.get("collections")
                prefs = spec.get("prefixes")
                policy_map[key_str] = _RbacPolicy(
                    key=key_str,
                    role=role,
                    collections=list(cols) if isinstance(cols, list) else None,
                    prefixes=list(prefs) if isinstance(prefs, list) else None,
                )
            else:
                raise ValueError("Invalid RBAC entry format")
    return list(policy_map.values())


def _extract_api_token(headers: Any) -> Optional[str]:
    auth = headers.get("authorization")
    if auth:
        s = str(auth).strip()
        if s[:7].lower() == "bearer ":
            tok = s[7:].strip()
            if tok:
                return tok
    api_key = headers.get("x-api-key")
    if api_key:
        tok = str(api_key).strip()
        if tok:
            return tok
    return None


def _authenticate_policy(policies: List[_RbacPolicy], headers: Any) -> Optional[_RbacPolicy]:
    token = _extract_api_token(headers)
    if not token:
        return None
    matched: Optional[_RbacPolicy] = None
    token_b = token.encode("utf-8")
    for p in policies:
        if hmac.compare_digest(p.key.encode("utf-8"), token_b):
            matched = p
    return matched


class _PrometheusMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ingest_documents_total = 0
        self.errors_total = 0
        self.histograms: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def record_query(self, collection: str, mode: str, duration_secs: float) -> None:
        dur = max(float(duration_secs), 0.0)
        with self._lock:
            key = (collection, mode)
            entry = self.histograms.get(key)
            if entry is None:
                entry = {"buckets": [0] * len(QUERY_DURATION_BUCKETS), "count": 0, "sum": 0.0}
                self.histograms[key] = entry
            entry["count"] += 1
            entry["sum"] += dur
            for idx, bound in enumerate(QUERY_DURATION_BUCKETS):
                if dur <= bound:
                    entry["buckets"][idx] += 1

    def record_ingest(self, docs_count: int) -> None:
        with self._lock:
            self.ingest_documents_total += int(docs_count)

    def record_error(self) -> None:
        with self._lock:
            self.errors_total += 1

    def render(self, active_db: SutraDB) -> str:
        names = sorted(active_db.list_collections())
        snapshots = []
        for name in names:
            try:
                col = active_db.get_collection(name)
                st = col.stats()
                count = int(st.get("count", 0))
                vec_mem = int(st.get("vector_memory_bytes", 0))
                hnsw_mem = int(st.get("hnsw", {}).get("memory_bytes", 0)) if isinstance(st.get("hnsw"), dict) else 0
                wal_pending = 0
                if st.get("enable_wal"):
                    wal_file = getattr(col, "_wal_file", None)
                    if wal_file and Path(wal_file).exists():
                        wal_pending = int(Path(wal_file).stat().st_size > 0)
                snapshots.append((name, count, vec_mem + hnsw_mem, vec_mem, wal_pending))
            except Exception:
                continue

        with self._lock:
            ingest_total = self.ingest_documents_total
            errors_total = self.errors_total
            hist_items = sorted(
                ((k, list(v["buckets"]), int(v["count"]), float(v["sum"])) for k, v in self.histograms.items()),
                key=lambda item: item[0],
            )

        lines = [
            "# HELP sutradb_collections_total Total number of collections",
            "# TYPE sutradb_collections_total gauge",
            f"sutradb_collections_total {len(snapshots)}",
            "# HELP sutradb_collection_documents Active documents stored per collection",
            "# TYPE sutradb_collection_documents gauge",
        ]
        for name, count, _, _, _ in snapshots:
            safe = _escape_prom_label(name)
            lines.append(f'sutradb_collection_documents{{collection="{safe}"}} {count}')

        lines.extend([
            "# HELP sutradb_documents_count Active documents stored per collection",
            "# TYPE sutradb_documents_count gauge",
        ])
        for name, count, _, _, _ in snapshots:
            safe = _escape_prom_label(name)
            lines.append(f'sutradb_documents_count{{collection="{safe}"}} {count}')

        lines.extend([
            "# HELP sutradb_collection_memory_bytes Estimated vector and index memory usage in bytes per collection",
            "# TYPE sutradb_collection_memory_bytes gauge",
        ])
        for name, _, mem_bytes, _, _ in snapshots:
            safe = _escape_prom_label(name)
            lines.append(f'sutradb_collection_memory_bytes{{collection="{safe}"}} {mem_bytes}')

        lines.extend([
            "# HELP sutradb_vector_memory_bytes Estimated vector memory usage in bytes per collection",
            "# TYPE sutradb_vector_memory_bytes gauge",
        ])
        for name, _, _, vec_mem, _ in snapshots:
            safe = _escape_prom_label(name)
            lines.append(f'sutradb_vector_memory_bytes{{collection="{safe}"}} {vec_mem}')

        lines.extend([
            "# HELP sutradb_wal_pending_records Uncompacted Write-Ahead Log records per collection",
            "# TYPE sutradb_wal_pending_records gauge",
        ])
        for name, _, _, _, wal_pending in snapshots:
            safe = _escape_prom_label(name)
            lines.append(f'sutradb_wal_pending_records{{collection="{safe}"}} {wal_pending}')

        lines.extend([
            "# HELP sutradb_ingest_documents_total Total documents ingested into SutraDB",
            "# TYPE sutradb_ingest_documents_total counter",
            f"sutradb_ingest_documents_total {ingest_total}",
            "# HELP sutradb_errors_total Total failed gRPC and HTTP operations",
            "# TYPE sutradb_errors_total counter",
            f"sutradb_errors_total {errors_total}",
            "# HELP sutradb_query_duration_seconds Query execution latency in seconds",
            "# TYPE sutradb_query_duration_seconds histogram",
        ])
        for (col_name, mode), buckets, count, total_sum in hist_items:
            safe_col = _escape_prom_label(col_name)
            for bound, b_count in zip(QUERY_DURATION_BUCKETS, buckets):
                lines.append(
                    f'sutradb_query_duration_seconds_bucket{{collection="{safe_col}",mode="{mode}",le="{bound}"}} {b_count}'
                )
            lines.append(
                f'sutradb_query_duration_seconds_bucket{{collection="{safe_col}",mode="{mode}",le="+Inf"}} {count}'
            )
            lines.append(
                f'sutradb_query_duration_seconds_sum{{collection="{safe_col}",mode="{mode}"}} {total_sum:.9f}'
            )
            lines.append(
                f'sutradb_query_duration_seconds_count{{collection="{safe_col}",mode="{mode}"}} {count}'
            )

        return "\n".join(lines) + "\n"


def _classify_query_mode(col: Collection, payload: QueryRequest) -> str:
    has_vec = payload.vector is not None and len(payload.vector) > 0
    has_txt = bool(payload.text and payload.text.strip())
    if not has_vec and has_txt:
        return "bm25"
    if has_vec and has_txt and payload.hybrid:
        return "hybrid"
    st = col.stats()
    if has_vec:
        if payload.use_ann and st.get("has_hnsw") and not st.get("hnsw", {}).get("dirty", False):
            return "hnsw"
        if st.get("quantized_pq"):
            return "pq"
        if st.get("quantized_bq") and st.get("metric") != "euclidean":
            return "bq"
        if st.get("quantized_sq8"):
            return "sq8"
        return "flat"
    return "hybrid"


class PayloadTooLargeError(Exception):
    """Raised when streaming request body exceeds bounded limit."""
    pass


def _is_payload_too_large(exc: BaseException) -> bool:
    if isinstance(exc, PayloadTooLargeError):
        return True
    if hasattr(exc, "exceptions"):
        return any(_is_payload_too_large(sub) for sub in exc.exceptions)
    return False


# Pydantic Schemas
class CreateCollectionRequest(BaseModel):
    name: str = Field(..., description="Unique collection name")
    dimension: int = Field(..., gt=0, description="Dense vector dimension")
    metric: str = Field("cosine", description="Distance metric: cosine, euclidean, dot")
    enable_wal: bool = Field(False, description="Enable Write-Ahead Log durability")


class QueryRequest(BaseModel):
    vector: Optional[List[float]] = Field(None, description="Query dense embedding vector")
    text: Optional[str] = Field(None, description="Lexical search query string")
    filter: Optional[Dict[str, Any]] = Field(None, description="Compound metadata filter dictionary")
    namespace: Optional[str] = Field(None, description="Multi-tenant namespace filter")
    top_k: int = Field(10, gt=0, le=10000, description="Max results to return (capped at 10000)")
    hybrid: bool = Field(True, description="Enable hybrid dense + BM25 search")
    alpha: Optional[float] = Field(None, description="Convex linear weight if not using RRF")
    rrf_constant: int = Field(60, ge=1, description="Reciprocal Rank Fusion smoothing constant")
    dense_weight: float = Field(1.0, ge=0.0, description="Dense score weight multiplier")
    bm25_weight: float = Field(1.0, ge=0.0, description="BM25 score weight multiplier")
    use_ann: bool = Field(False, description="Use IVF-Flat ANN index if trained")


class HNSWRequest(BaseModel):
    m: int = Field(16, ge=4, le=128, description="Links per node")
    ef_construction: int = Field(64, ge=8, description="Beam width during construction")
    ef_search: int = Field(32, ge=4, description="Beam width during search")


class MetadataUpdateRequest(BaseModel):
    metadata: Dict[str, Any] = Field(..., description="New metadata dictionary")


class BatchDeleteRequest(BaseModel):
    ids: List[str] = Field(..., min_length=1, description="List of document IDs to delete")


def create_app(data_dir: Optional[str] = None, db: Optional[SutraDB] = None) -> FastAPI:
    """Creates and configures the production FastAPI ASGI application."""
    import os
    effective_data_dir = data_dir if data_dir is not None else os.getenv("SUTRA_DATA_DIR", "./sutra_data")
    backend_url = os.getenv("SUTRA_BACKEND_URL", "").strip().rstrip("/")
    app = FastAPI(
        title="SutraDB API",
        version=__version__,
        description="High-Performance Hybrid Vector & BM25 Search Engine in pure Python."
    )

    # Single-owner invariant: when SUTRA_BACKEND_URL is set, proxy all API operations to sutradb-server
    app.state.db = db if db is not None else (None if backend_url else SutraDB(persist_directory=effective_data_dir))
    app.state.backend_url = backend_url or None
    app.state.auth_policies = _load_rbac_policies()
    app.state.metrics = _PrometheusMetrics()

    # CORS Middleware — restrict origins in production via SUTRA_CORS_ORIGINS env
    import os as _os
    cors_origins_raw = _os.getenv("SUTRA_CORS_ORIGINS", "*")
    cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials="*" not in cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if backend_url:
        @app.middleware("http")
        async def proxy_to_backend(request: Request, call_next):
            path = request.url.path.rstrip("/") or "/"
            if path in ("/ui", "/dashboard"):
                return await call_next(request)

            forward_headers = {}
            for hdr in ("content-type", "accept", "authorization", "x-api-key"):
                val = request.headers.get(hdr)
                if val:
                    forward_headers[hdr] = val

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    if path == "/ui/stats":
                        upstream = await client.get(f"{backend_url}/collections", headers=forward_headers)
                        if upstream.status_code != 200:
                            return Response(
                                content=upstream.content,
                                status_code=upstream.status_code,
                                media_type=upstream.headers.get("content-type", "application/json"),
                            )
                        data = upstream.json()
                        cols = data.get("collections", [])
                        total_docs = sum(int(c.get("count", 0)) for c in cols if isinstance(c, dict))
                        return JSONResponse(
                            status_code=200,
                            content={
                                "version": __version__,
                                "total_collections": len(cols),
                                "total_documents": total_docs,
                                "collections": cols,
                            },
                        )

                    body_bytes = await request.body()
                    target_url = f"{backend_url}{request.url.path}"
                    if request.url.query:
                        target_url = f"{target_url}?{request.url.query}"

                    upstream = await client.request(
                        method=request.method,
                        url=target_url,
                        content=body_bytes,
                        headers=forward_headers,
                    )
                    return Response(
                        content=upstream.content,
                        status_code=upstream.status_code,
                        media_type=upstream.headers.get("content-type", "application/json"),
                    )
            except httpx.RequestError as exc:
                return JSONResponse(
                    status_code=503,
                    content={"detail": f"Backend service unavailable: {exc}"},
                )
    else:
        @app.middleware("http")
        async def enforce_auth_and_metrics(request: Request, call_next):
            path = request.url.path.rstrip("/") or "/"
            method = request.method.upper()
            request.state.auth_policy = None

            policies: List[_RbacPolicy] = app.state.auth_policies
            if policies and method != "OPTIONS" and path not in ("/", "/health", "/ui", "/dashboard", "/docs", "/openapi.json"):
                policy = _authenticate_policy(policies, request.headers)
                if policy is None:
                    app.state.metrics.record_error()
                    return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key"})

                parts = [seg for seg in path.strip("/").split("/") if seg]
                required_role = "read"
                target_collection: Optional[str] = None
                if parts == ["collections"] and method == "POST":
                    required_role = "admin"
                    try:
                        body_bytes = await request.body()
                        if body_bytes:
                            parsed_body = json.loads(body_bytes)
                            if isinstance(parsed_body, dict) and isinstance(parsed_body.get("name"), str):
                                target_collection = parsed_body["name"]
                    except Exception:
                        pass
                elif len(parts) >= 2 and parts[0] == "collections":
                    target_collection = parts[1]
                    tail = parts[2:]
                    if len(parts) == 2 and method == "DELETE":
                        required_role = "admin"
                    elif tail in (["save"], ["compact"], ["hnsw"], ["sq8"], ["bq"], ["pq"], ["pq", "train"]):
                        required_role = "admin"
                    elif (
                        tail in (["insert"], ["documents"], ["upsert"]) and method in ("POST", "DELETE")
                    ) or (
                        len(tail) == 2 and tail[0] == "documents" and method == "DELETE"
                    ) or (
                        len(tail) == 3 and tail[0] == "documents" and tail[2] == "metadata" and method == "PATCH"
                    ):
                        required_role = "write"
                    else:
                        required_role = "read"

                if policy.rank < _ROLE_RANK[required_role]:
                    app.state.metrics.record_error()
                    return JSONResponse(
                        status_code=403,
                        content={"detail": f"Insufficient role: requires {required_role}, got {policy.role}"},
                    )
                if target_collection is not None and not policy.allows_collection(target_collection):
                    app.state.metrics.record_error()
                    return JSONResponse(
                        status_code=403,
                        content={"detail": f"API key is not authorized for collection '{target_collection}'"},
                    )
                request.state.auth_policy = policy

            response = await call_next(request)
            if response.status_code >= 400:
                app.state.metrics.record_error()
            return response

    # Bounded payload middleware (max 10MB)
    @app.middleware("http")
    async def enforce_payload_limit(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_PAYLOAD_SIZE:
                    return JSONResponse(
                        status_code=413,
                        content={"error": "Payload exceeds maximum allowed size of 10MB"}
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid Content-Length header"}
                )

        received_bytes = 0
        original_receive = request._receive

        async def bounded_receive():
            nonlocal received_bytes
            message = await original_receive()
            if message.get("type") == "http.request":
                body_chunk = message.get("body", b"")
                received_bytes += len(body_chunk)
                if received_bytes > MAX_PAYLOAD_SIZE:
                    raise PayloadTooLargeError("Payload exceeds maximum allowed size of 10MB")
            return message

        request._receive = bounded_receive
        try:
            return await call_next(request)
        except Exception as e:
            if _is_payload_too_large(e):
                return JSONResponse(
                    status_code=413,
                    content={"error": "Payload exceeds maximum allowed size of 10MB"}
                )
            raise

    @app.get("/", tags=["Health"])
    @app.get("/health", tags=["Health"])
    def health_check():
        active_db: SutraDB = app.state.db
        return {
            "status": "healthy",
            "version": __version__,
            "collections": active_db.list_collections()
        }

    @app.get("/ui", response_class=HTMLResponse, tags=["Dashboard"])
    @app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
    def web_dashboard():
        """Serves the interactive embedded Web UI management console."""
        return HTMLResponse(content=get_dashboard_html(version=__version__))

    def _resolve_collection(active_db: SutraDB, name: str):
        try:
            return active_db.get_collection(name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    @app.get("/ui/stats", tags=["Dashboard"])
    def dashboard_stats(request: Request):
        """Aggregated stats for the web console."""
        active_db: SutraDB = app.state.db
        policy: Optional[_RbacPolicy] = getattr(request.state, "auth_policy", None)
        total_docs = 0
        cols_summary = []
        for name in active_db.list_collections():
            if policy is not None and not policy.allows_collection(name):
                continue
            try:
                col = active_db.get_collection(name)
                st = col.stats()
                total_docs += st.get("count", 0)
                cols_summary.append(st)
            except Exception:
                continue
        return {
            "version": __version__,
            "total_collections": len(cols_summary),
            "total_documents": total_docs,
            "collections": cols_summary,
        }

    @app.get("/collections/{name}/sample_vectors", tags=["Collections"])
    def get_sample_vectors(name: str, limit: int = Query(100, ge=1, le=10000)):
        """Returns sample vectors for in-browser PCA 2D projection."""
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)
        docs = col.get_all()[:limit]
        return {
            "collection": name,
            "count": len(docs),
            "vectors": [
                {
                    "id": d.id,
                    "vector": [float(x) for x in (d.vector[:min(len(d.vector), 128)])],
                    "text": d.text[:100] if d.text else "",
                    "metadata": d.metadata,
                }
                for d in docs
            ],
        }

    @app.get("/metrics", tags=["Observability"])
    def prometheus_metrics():
        active_db: SutraDB = app.state.db
        return Response(
            content=app.state.metrics.render(active_db),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/collections", tags=["Collections"])
    def list_collections(request: Request):
        active_db: SutraDB = app.state.db
        policy: Optional[_RbacPolicy] = getattr(request.state, "auth_policy", None)
        cols = []
        for name in active_db.list_collections():
            if policy is not None and not policy.allows_collection(name):
                continue
            try:
                col = active_db.get_collection(name)
                cols.append(col.stats())
            except KeyError:
                continue
        return {"collections": cols}

    @app.post("/collections", status_code=status.HTTP_201_CREATED, tags=["Collections"])
    def create_collection(payload: CreateCollectionRequest):
        active_db: SutraDB = app.state.db
        try:
            col = active_db.create_collection(
                name=payload.name,
                dimension=payload.dimension,
                metric=payload.metric,
                enable_wal=payload.enable_wal
            )
            return {"message": "Collection created", "stats": col.stats()}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/collections/{name}/stats", tags=["Collections"])
    def get_collection_stats(name: str):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)
        return col.stats()

    @app.delete("/collections/{name}", tags=["Collections"])
    def drop_collection(name: str):
        active_db: SutraDB = app.state.db
        try:
            dropped = active_db.drop_collection(name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not dropped:
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
        return {"message": f"Collection '{name}' dropped successfully"}

    @app.post("/collections/{name}/documents", tags=["Documents"])
    @app.post("/collections/{name}/insert", tags=["Documents"])
    async def insert_documents(name: str, request: Request):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)

        try:
            body = await request.json()
        except Exception as e:
            if _is_payload_too_large(e):
                raise
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {str(e)}")

        docs = []
        if isinstance(body, dict):
            docs = body.get("documents", [body] if ("vector" in body or "text" in body) else [])
        elif isinstance(body, list):
            docs = body

        if not docs:
            raise HTTPException(status_code=400, detail="Missing 'documents' list")

        try:
            total_count = col.insert(docs)
            app.state.metrics.record_ingest(len(docs))
            return {"inserted": len(docs), "total_count": total_count}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/collections/{name}/compact", tags=["Persistence"])
    def compact_collection(name: str):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)
        try:
            col.compact()
            return {"message": "Collection compacted", "count": col.count()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/collections/{name}/documents", tags=["Documents"])
    def list_documents(
        name: str,
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)

        docs = col.get_all()
        total = len(docs)
        sliced = docs[offset:offset + limit]
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "documents": [doc.to_dict(include_vector=False) for doc in sliced]
        }

    @app.get("/collections/{name}/documents/{id}", tags=["Documents"])
    def get_document(name: str, id: str):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)

        doc = col.get(id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document '{id}' not found in collection '{name}'")
        return doc.to_dict(include_vector=True)

    @app.delete("/collections/{name}/documents/{id}", tags=["Documents"])
    def delete_single_document(name: str, id: str):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)

        deleted = col.delete(id)
        if deleted == 0:
            raise HTTPException(status_code=404, detail=f"Document '{id}' not found in collection '{name}'")
        return {"deleted": deleted, "message": f"Document '{id}' deleted successfully"}

    @app.delete("/collections/{name}/documents", tags=["Documents"])
    async def delete_documents_batch(name: str, request: Request):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)

        try:
            body = await request.json()
        except Exception as e:
            if _is_payload_too_large(e):
                raise
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {str(e)}")

        ids = body.get("ids", []) if isinstance(body, dict) else (body if isinstance(body, list) else [])
        if not ids:
            raise HTTPException(status_code=400, detail="Missing 'ids' list for batch deletion")

        deleted = col.delete(ids)
        return {"deleted": deleted, "message": f"Deleted {deleted} documents"}

    @app.post("/collections/{name}/query", tags=["Query"])
    def query_collection(name: str, payload: QueryRequest):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)

        try:
            t0 = time.perf_counter()
            mode = _classify_query_mode(col, payload)
            results = col.query(
                vector=payload.vector,
                text=payload.text,
                filter=payload.filter,
                namespace=payload.namespace,
                top_k=payload.top_k,
                hybrid=payload.hybrid,
                alpha=payload.alpha,
                rrf_constant=payload.rrf_constant,
                dense_weight=payload.dense_weight,
                bm25_weight=payload.bm25_weight,
                use_ann=payload.use_ann
            )
            app.state.metrics.record_query(name, mode, time.perf_counter() - t0)
            return {
                "count": len(results),
                "results": [r.to_dict() for r in results]
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/collections/{name}/upsert", tags=["Documents"])
    async def upsert_documents(name: str, request: Request):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)

        try:
            body = await request.json()
        except Exception as e:
            if _is_payload_too_large(e):
                raise
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {str(e)}")

        docs = []
        if isinstance(body, dict):
            docs = body.get("documents", [body] if ("vector" in body or "text" in body) else [])
        elif isinstance(body, list):
            docs = body

        if not docs:
            raise HTTPException(status_code=400, detail="Missing 'documents' list")

        try:
            total_count = col.upsert(docs)
            app.state.metrics.record_ingest(len(docs))
            return {"upserted": len(docs), "total_count": total_count}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.patch("/collections/{name}/documents/{id}/metadata", tags=["Documents"])
    def update_document_metadata(name: str, id: str, payload: MetadataUpdateRequest):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)

        updated = col.update_metadata(id, payload.metadata)
        if not updated:
            raise HTTPException(status_code=404, detail=f"Document '{id}' not found in collection '{name}'")
        return {"updated": True, "message": f"Document '{id}' metadata updated"}

    @app.post("/collections/{name}/hnsw", tags=["Indexing"])
    def build_hnsw(name: str, payload: Optional[HNSWRequest] = None):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)

        m = payload.m if payload else 16
        ef_c = payload.ef_construction if payload else 64
        ef_s = payload.ef_search if payload else 32
        res = col.build_hnsw(m=m, ef_construction=ef_c, ef_search=ef_s)
        return {"message": "HNSW index built", "config": res}

    @app.post("/collections/{name}/sq8", tags=["Indexing"])
    def enable_sq8(name: str):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)

        col.enable_sq8()
        return {"message": "SQ8 quantization enabled", "stats": col.stats()}

    @app.post("/collections/{name}/bq", tags=["Indexing"])
    def enable_bq(name: str):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)

        col.enable_bq()
        return {"message": "1-Bit Binary Quantization (BQ) enabled (32x RAM reduction)", "stats": col.stats()}

    @app.post("/collections/{name}/pq", tags=["Indexing"])
    def enable_pq(name: str):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)

        try:
            col.enable_pq()
            return {"message": "Product Quantization (PQ) enabled (up to 64x RAM reduction)", "stats": col.stats()}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/collections/{name}/save", tags=["Persistence"])
    def save_collection(name: str):
        active_db: SutraDB = app.state.db
        col = _resolve_collection(active_db, name)

        try:
            bytes_written = col.save()
            return {"message": "Snapshot saved", "bytes": bytes_written}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Module-level default application instance
app = create_app()


class SutraHTTPHandler(BaseHTTPRequestHandler):
    """
    Backwards-compatible HTTP request handler using standard library http.server.
    Supports complete document CRUD and bounded payload size checks.
    """

    db: Optional[SutraDB] = None  # Injected on server startup

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        """Sends a JSON response with CORS headers."""
        data = json.dumps(payload, separators=(',', ':')).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        """Reads and parses JSON payload from request body with size bounds."""
        content_length_header = self.headers.get("Content-Length", "0")
        try:
            content_length = int(content_length_header)
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length header"})
            return None
        if content_length == 0:
            return None
        if content_length > MAX_PAYLOAD_SIZE:
            self._send_json(413, {"error": "Payload exceeds maximum allowed size of 10MB"})
            return None
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def do_OPTIONS(self) -> None:
        """Handles CORS pre-flight requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:
        """Routes GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("/ui", "/dashboard"):
            html = get_dashboard_html(__version__)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return

        if path == "" or path == "/health":
            self._send_json(200, {
                "status": "healthy",
                "version": __version__,
                "collections": self.db.list_collections() if self.db else []
            })
            return

        if path == "/collections":
            cols = []
            if self.db:
                for name in self.db.list_collections():
                    try:
                        col = self.db.get_collection(name)
                        cols.append(col.stats())
                    except KeyError:
                        continue
            self._send_json(200, {"collections": cols})
            return

        parts = path.split("/")
        # /collections/<name>/stats
        if len(parts) == 4 and parts[1] == "collections" and parts[3] == "stats":
            col_name = parts[2]
            try:
                col = self.db.get_collection(col_name)
                self._send_json(200, col.stats())
            except KeyError:
                self._send_json(404, {"error": f"Collection '{col_name}' not found"})
            return

        # /collections/<name>/documents (list)
        if len(parts) == 4 and parts[1] == "collections" and parts[3] == "documents":
            col_name = parts[2]
            try:
                col = self.db.get_collection(col_name)
                docs = col.get_all()
                self._send_json(200, {
                    "total": len(docs),
                    "limit": 50,
                    "offset": 0,
                    "documents": [doc.to_dict(include_vector=False) for doc in docs[:50]]
                })
            except KeyError:
                self._send_json(404, {"error": f"Collection '{col_name}' not found"})
            return

        # /collections/<name>/documents/<id>
        if len(parts) == 5 and parts[1] == "collections" and parts[3] == "documents":
            col_name = parts[2]
            doc_id = parts[4]
            try:
                col = self.db.get_collection(col_name)
                doc = col.get(doc_id)
                if doc is None:
                    self._send_json(404, {"error": f"Document '{doc_id}' not found in collection '{col_name}'"})
                else:
                    self._send_json(200, doc.to_dict(include_vector=True))
            except KeyError:
                self._send_json(404, {"error": f"Collection '{col_name}' not found"})
            return

        self._send_json(404, {"error": "Endpoint not found"})

    def do_POST(self) -> None:
        """Routes POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = path.split("/")

        try:
            body = self._read_json()
            if body is None and int(self.headers.get("Content-Length", 0)) > MAX_PAYLOAD_SIZE:
                return
            body = body or {}
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON body: {str(e)}"})
            return

        # POST /collections -> Create collection
        if path == "/collections":
            name = body.get("name")
            dimension = body.get("dimension")
            metric = body.get("metric", "cosine")
            enable_wal = bool(body.get("enable_wal", False))
            if not name or not dimension:
                self._send_json(400, {"error": "Fields 'name' and 'dimension' are required"})
                return
            try:
                col = self.db.create_collection(name=name, dimension=int(dimension), metric=metric, enable_wal=enable_wal)
                self._send_json(201, {"message": "Collection created", "stats": col.stats()})
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return

        # POST /collections/<name>/documents or POST /collections/<name>/insert
        if len(parts) == 4 and parts[1] == "collections" and parts[3] in ("insert", "documents"):
            col_name = parts[2]
            try:
                col = self.db.get_collection(col_name)
                docs = body.get("documents", []) if isinstance(body, dict) else (body if isinstance(body, list) else [])
                if not docs:
                    self._send_json(400, {"error": "Missing 'documents' list"})
                    return
                count = col.insert(docs)
                self._send_json(200, {"inserted": len(docs), "total_count": count})
            except KeyError:
                self._send_json(404, {"error": f"Collection '{col_name}' not found"})
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return

        # POST /collections/<name>/query
        if len(parts) == 4 and parts[1] == "collections" and parts[3] == "query":
            col_name = parts[2]
            try:
                col = self.db.get_collection(col_name)
                vector = body.get("vector")
                text = body.get("text")
                filter_spec = body.get("filter")
                top_k = int(body.get("top_k", 10))
                hybrid = bool(body.get("hybrid", True))
                alpha = body.get("alpha")
                rrf_constant = int(body.get("rrf_constant", 60))
                dense_weight = float(body.get("dense_weight", 1.0))
                bm25_weight = float(body.get("bm25_weight", 1.0))
                use_ann = bool(body.get("use_ann", False))

                results = col.query(
                    vector=vector,
                    text=text,
                    filter=filter_spec,
                    top_k=top_k,
                    hybrid=hybrid,
                    alpha=float(alpha) if alpha is not None else None,
                    rrf_constant=rrf_constant,
                    dense_weight=dense_weight,
                    bm25_weight=bm25_weight,
                    use_ann=use_ann
                )
                self._send_json(200, {
                    "count": len(results),
                    "results": [r.to_dict() for r in results]
                })
            except KeyError:
                self._send_json(404, {"error": f"Collection '{col_name}' not found"})
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return

        # POST /collections/<name>/save
        if len(parts) == 4 and parts[1] == "collections" and parts[3] == "save":
            col_name = parts[2]
            try:
                col = self.db.get_collection(col_name)
                bytes_written = col.save()
                self._send_json(200, {"message": "Snapshot saved", "bytes": bytes_written})
            except KeyError:
                self._send_json(404, {"error": f"Collection '{col_name}' not found"})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        self._send_json(404, {"error": "Endpoint not found"})

    def do_DELETE(self) -> None:
        """Routes DELETE requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = path.split("/")

        # DELETE /collections/<name>
        if len(parts) == 3 and parts[1] == "collections":
            col_name = parts[2]
            dropped = self.db.drop_collection(col_name)
            if dropped:
                self._send_json(200, {"message": f"Collection '{col_name}' dropped successfully"})
            else:
                self._send_json(404, {"error": f"Collection '{col_name}' not found"})
            return

        # DELETE /collections/<name>/documents/<id>
        if len(parts) == 5 and parts[1] == "collections" and parts[3] == "documents":
            col_name = parts[2]
            doc_id = parts[4]
            try:
                col = self.db.get_collection(col_name)
                deleted = col.delete(doc_id)
                if deleted > 0:
                    self._send_json(200, {"deleted": deleted, "message": f"Document '{doc_id}' deleted successfully"})
                else:
                    self._send_json(404, {"error": f"Document '{doc_id}' not found in collection '{col_name}'"})
            except KeyError:
                self._send_json(404, {"error": f"Collection '{col_name}' not found"})
            return

        # DELETE /collections/<name>/documents (batch)
        if len(parts) == 4 and parts[1] == "collections" and parts[3] == "documents":
            col_name = parts[2]
            try:
                col = self.db.get_collection(col_name)
                body = self._read_json() or {}
                ids = body.get("ids", []) if isinstance(body, dict) else (body if isinstance(body, list) else [])
                if not ids:
                    self._send_json(400, {"error": "Missing 'ids' list for batch deletion"})
                    return
                deleted = col.delete(ids)
                self._send_json(200, {"deleted": deleted, "message": f"Deleted {deleted} documents"})
            except KeyError:
                self._send_json(404, {"error": f"Collection '{col_name}' not found"})
            return

        self._send_json(404, {"error": "Endpoint not found"})

    def log_message(self, format: str, *args: Any) -> None:
        """Mutes noisy default HTTP access logs."""
        pass


def run_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    data_dir: str = "./sutra_data",
    db: Optional[SutraDB] = None
) -> None:
    """Starts the SutraDB ASGI server with Uvicorn."""
    app_instance = create_app(data_dir=data_dir, db=db)
    print(f"⚡ SutraDB v{__version__} Production ASGI server running at http://{host}:{port}")
    print(f"📁 Persistence directory: {Path(data_dir).resolve()}")
    uvicorn.run(app_instance, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import os
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv("PORT", "8765"))
    data_dir_arg = sys.argv[2] if len(sys.argv) > 2 else os.getenv("SUTRA_DATA_DIR", "./sutra_data")
    run_server(port=port_arg, data_dir=data_dir_arg)
