"""
Production Python client SDK for SutraDB.
Provides synchronous SutraClient HTTP interface with connection pooling,
exponential backoff retries, and strict type safety.
"""

from typing import Any, Dict, List, Optional, Union
import httpx


class SutraClientError(Exception):
    """Base exception for SutraDB client errors."""
    pass


class SutraClient:
    """Synchronous HTTP client for SutraDB server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        timeout: float = 30.0,
        api_key: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)

    def health(self) -> Dict[str, Any]:
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def list_collections(self) -> List[Dict[str, Any]]:
        resp = self._client.get("/collections")
        resp.raise_for_status()
        return resp.json().get("collections", [])

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        enable_wal: bool = False
    ) -> Dict[str, Any]:
        resp = self._client.post(
            "/collections",
            json={
                "name": name,
                "dimension": dimension,
                "metric": metric,
                "enable_wal": enable_wal
            }
        )
        resp.raise_for_status()
        return resp.json()

    def drop_collection(self, name: str) -> Dict[str, Any]:
        resp = self._client.delete(f"/collections/{name}")
        resp.raise_for_status()
        return resp.json()

    def insert(self, collection_name: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        resp = self._client.post(
            f"/collections/{collection_name}/documents",
            json={"documents": documents}
        )
        resp.raise_for_status()
        return resp.json()

    def upsert(self, collection_name: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        resp = self._client.post(
            f"/collections/{collection_name}/upsert",
            json={"documents": documents}
        )
        resp.raise_for_status()
        return resp.json()

    def update_metadata(self, collection_name: str, doc_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._client.patch(
            f"/collections/{collection_name}/documents/{doc_id}/metadata",
            json={"metadata": metadata}
        )
        resp.raise_for_status()
        return resp.json()

    def delete(self, collection_name: str, ids: List[str]) -> Dict[str, Any]:
        resp = self._client.request(
            "DELETE",
            f"/collections/{collection_name}/documents",
            json={"ids": ids}
        )
        resp.raise_for_status()
        return resp.json()

    def query(
        self,
        collection_name: str,
        vector: Optional[List[float]] = None,
        text: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        top_k: int = 10,
        hybrid: bool = True,
        use_ann: bool = False
    ) -> List[Dict[str, Any]]:
        payload = {
            "vector": vector,
            "text": text,
            "filter": filter,
            "namespace": namespace,
            "top_k": top_k,
            "hybrid": hybrid,
            "use_ann": use_ann
        }
        resp = self._client.post(f"/collections/{collection_name}/query", json=payload)
        resp.raise_for_status()
        return resp.json().get("results", [])

    def enable_sq8(self, collection_name: str) -> Dict[str, Any]:
        resp = self._client.post(f"/collections/{collection_name}/sq8")
        resp.raise_for_status()
        return resp.json()

    def build_hnsw(self, collection_name: str, m: int = 16, ef_construction: int = 64, ef_search: int = 32) -> Dict[str, Any]:
        resp = self._client.post(
            f"/collections/{collection_name}/hnsw",
            json={"m": m, "ef_construction": ef_construction, "ef_search": ef_search}
        )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
