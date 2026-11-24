"""
Automated Embedding Pipelines & Providers for SutraDB.
Provides unified zero-friction embedding generation across Dense, Sparse (SPLADE),
and Multi-Vector (ColBERT) representations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import hashlib
import os
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


class EmbeddingFunction(ABC):
    """
    Abstract Base Class for all SutraDB embedding providers.
    Implements callable interface and standard LangChain / LlamaIndex compatible methods.
    """
    dimension: int = 384

    @abstractmethod
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Encodes a list of raw text strings into dense vector embeddings."""
        pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain standard document batch embedding interface."""
        return self(texts)

    def embed_query(self, text: str) -> List[float]:
        """LangChain standard single query embedding interface."""
        return self([text])[0]


class DefaultEmbeddingFunction(EmbeddingFunction):
    """
    High-speed, deterministic, zero-dependency embedding generator.
    Produces unit-sphere normalized 384-dimensional dense vectors using sub-token
    feature hashing and multi-probe projection.
    
    Requires zero external API keys or heavy packages, enabling instant local quickstarts
    and offline / air-gapped CI/CD.
    """

    def __init__(self, dimension: int = 384, seed: int = 42):
        self.dimension = int(dimension)
        self.seed = seed

    def __call__(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            tokens = text.lower().strip().split()
            if not tokens:
                vec[0] = 1.0
                results.append(vec.tolist())
                continue

            for tok in tokens:
                # Deterministic SHA-256 token digest
                h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
                # Multi-probe projection across 4 coordinates
                for p in range(4):
                    idx = (h + p * 31 + self.seed) % self.dimension
                    sign = 1.0 if ((h >> (p * 4)) & 1) == 1 else -1.0
                    weight = 1.0 / (1.0 + p * 0.5)
                    vec[idx] += sign * weight

            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                vec /= norm
            else:
                vec[0] = 1.0
            results.append(vec.tolist())
        return results

    def encode_sparse(self, texts: List[str]) -> List[Dict[int, float]]:
        """Generates pseudo-lexical sparse weights matching token vocabulary bins."""
        results: List[Dict[int, float]] = []
        for text in texts:
            sp: Dict[int, float] = {}
            tokens = text.lower().strip().split()
            for tok in tokens:
                h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
                idx = h % 30000
                sp[idx] = sp.get(idx, 0.0) + 1.0
            results.append(sp)
        return results

    def encode_colbert(self, texts: List[str]) -> List[List[List[float]]]:
        """Generates multi-vector token embeddings for ColBERT late interaction."""
        results: List[List[List[float]]] = []
        for text in texts:
            tokens = text.lower().strip().split()
            tok_vecs: List[List[float]] = []
            for tok in tokens[:32]:
                v = np.zeros(self.dimension, dtype=np.float32)
                h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
                for p in range(4):
                    idx = (h + p * 31 + self.seed) % self.dimension
                    sign = 1.0 if ((h >> (p * 4)) & 1) == 1 else -1.0
                    v[idx] += sign
                norm = float(np.linalg.norm(v))
                if norm > 1e-12:
                    v /= norm
                else:
                    v[0] = 1.0
                tok_vecs.append(v.tolist())

            if not tok_vecs:
                v = np.zeros(self.dimension, dtype=np.float32)
                v[0] = 1.0
                tok_vecs.append(v.tolist())
            results.append(tok_vecs)
        return results


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """
    OpenAI embedding provider using lightweight HTTPX client.
    Works seamlessly without needing the official 'openai' package installed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-small",
        dimension: Optional[int] = None,
        base_url: str = "https://api.openai.com/v1"
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

        if dimension:
            self.dimension = int(dimension)
        elif "text-embedding-3-large" in model_name:
            self.dimension = 3072
        else:
            self.dimension = 1536

    def __call__(self, texts: List[str]) -> List[List[float]]:
        import httpx

        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be passed via api_key or OPENAI_API_KEY environment variable"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload: Dict[str, Any] = {
            "input": texts,
            "model": self.model_name,
        }
        if "text-embedding-3" in self.model_name and self.dimension:
            payload["dimensions"] = self.dimension

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{self.base_url}/embeddings", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        sorted_items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_items]


class FastEmbedFunction(EmbeddingFunction):
    """
    Local ONNX embedding provider using FastEmbed.
    Supports dense vectors and optional neural sparse SPLADE representations.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        enable_sparse: bool = False,
        sparse_model_name: str = "prithivida/Splade_PP_en_v1"
    ):
        try:
            from fastembed import TextEmbedding
        except ImportError as err:
            raise ImportError(
                "fastembed is required for FastEmbedFunction. Install it with: pip install fastembed"
            ) from err

        self.model = TextEmbedding(model_name=model_name)
        self.model_name = model_name
        self.enable_sparse = enable_sparse
        self.sparse_model = None

        if "bge-small" in model_name or "MiniLM" in model_name:
            self.dimension = 384
        elif "bge-base" in model_name:
            self.dimension = 768
        elif "bge-large" in model_name:
            self.dimension = 1024
        else:
            self.dimension = 384

        if enable_sparse:
            from fastembed import SparseTextEmbedding
            self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        generator = self.model.embed(texts)
        return [vec.tolist() for vec in generator]

    def encode_sparse(self, texts: List[str]) -> List[Dict[int, float]]:
        if not self.sparse_model:
            return [{} for _ in texts]
        generator = self.sparse_model.embed(texts)
        results: List[Dict[int, float]] = []
        for sparse_vec in generator:
            sp = {int(idx): float(val) for idx, val in zip(sparse_vec.indices, sparse_vec.values)}
            results.append(sp)
        return results


class HuggingFaceEmbeddingFunction(EmbeddingFunction):
    """
    SentenceTransformers local embedding provider.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError(
                "sentence-transformers is required for HuggingFaceEmbeddingFunction. Install with: pip install sentence-transformers"
            ) from err

        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()
