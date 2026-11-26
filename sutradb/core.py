"""
Core database abstractions for SutraDB: Document, SearchResult, Collection, and SutraDB manager.
Unifies Python `Collection` directly onto Rust `NativeCollection` (`sutradb._sutradb`) with zero-copy NumPy ingestion
while preserving 100% compatibility with existing Python APIs, embeddings callbacks, and inspection properties.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid
import numpy as np

from sutradb.distance import Metric
from sutradb.nlp import Analyzer, AnalyzerConfig
from sutradb.storage import WriteAheadLog
from sutradb.ann import IVFIndex, IVFConfig, HNSWIndex, HNSWConfig, auto_n_lists
from sutradb.quantization import BQVector, SQ8Vector, PQVector

try:
    from sutradb._sutradb import NativeCollection
except ImportError:
    NativeCollection = None

_COLLECTION_NAME_RE = re.compile(r"\A[A-Za-z0-9_-]{1,128}\Z")


def validate_collection_name(name: str) -> str:
    """Validates collection names against ^[a-zA-Z0-9_-]{1,128}$ to block path traversal."""
    if not isinstance(name, str) or not _COLLECTION_NAME_RE.match(name):
        raise ValueError(
            f"Invalid collection name '{name}': must match ^[a-zA-Z0-9_-]{{1,128}}$"
        )
    return name


def _restore_sparse_vector(raw: Any) -> Optional[Dict[Union[int, str], float]]:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        return None
    restored: Dict[Union[int, str], float] = {}
    for k, v in raw.items():
        key: Union[int, str] = int(k) if (isinstance(k, str) and k.isdigit()) else k
        restored[key] = float(v)
    return restored


@dataclass
class Document:
    """Represents a single record stored in SutraDB."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vector: Optional[Union[List[float], np.ndarray]] = None
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    namespace: Optional[str] = None
    sparse_vector: Optional[Dict[Union[int, str], float]] = None
    colbert_tokens: Optional[List[List[float]]] = None

    def to_dict(self, include_vector: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "namespace": self.namespace,
            "sparse_vector": self.sparse_vector,
            "colbert_tokens": self.colbert_tokens,
        }
        if include_vector and self.vector is not None:
            data["vector"] = self.vector.tolist() if isinstance(self.vector, np.ndarray) else list(self.vector)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], vector: Optional[np.ndarray] = None) -> "Document":
        vec = vector if vector is not None else data.get("vector")
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            vector=vec,
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            namespace=data.get("namespace"),
            sparse_vector=_restore_sparse_vector(data.get("sparse_vector")),
            colbert_tokens=data.get("colbert_tokens"),
        )


@dataclass
class SearchResult:
    """Returned item from a vector, lexical, or hybrid search."""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    sparse_score: Optional[float] = None
    colbert_score: Optional[float] = None
    namespace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": round(self.score, 6),
            "text": self.text,
            "metadata": self.metadata,
            "dense_score": round(self.dense_score, 6) if self.dense_score is not None else None,
            "bm25_score": round(self.bm25_score, 6) if self.bm25_score is not None else None,
            "sparse_score": round(self.sparse_score, 6) if self.sparse_score is not None else None,
            "colbert_score": round(self.colbert_score, 6) if self.colbert_score is not None else None,
            "namespace": self.namespace,
        }


class _NativeBM25View:
    """Compatibility view exposing analyzer config and Rust BM25 scoring for inspection."""

    def __init__(self, col: "Collection"):
        self._col = col

    @property
    def analyzer(self) -> Analyzer:
        return self._col.analyzer

    def score_query(self, query_text: str) -> np.ndarray:
        return np.asarray(self._col._native.bm25_score_query(query_text), dtype=np.float32)


class _NativeWalProxy:
    """Compatibility WAL proxy delegating replay/append inspection while Rust owns WAL durability."""

    def __init__(self, wal_path: Path, native_col: Optional[Any] = None):
        self.path = wal_path
        self._native_col = native_col
        self._handle: Optional[Any] = None

    @property
    def _file(self) -> Any:
        if self._handle is None:
            if not self.path.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.touch(exist_ok=True)
            self._handle = open(self.path, "rb")
        return self._handle

    def append(self, record: Dict[str, Any]) -> None:
        wal = WriteAheadLog(self.path)
        try:
            wal.append(record)
        finally:
            wal.close()

    def replay(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        wal = WriteAheadLog(self.path)
        try:
            return wal.replay()
        finally:
            wal.close()

    def clear(self) -> None:
        wal = WriteAheadLog(self.path)
        try:
            wal.clear()
        finally:
            wal.close()

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        if self._native_col is not None:
            self._native_col.close_wal()


class Collection:
    """A unified vector, lexical BM25, neural sparse, and ColBERT collection backed by Rust NativeCollection."""

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: Union[Metric, str] = Metric.COSINE,
        storage_path: Optional[Union[str, Path]] = None,
        enable_wal: bool = False,
        ann_config: Optional[IVFConfig] = None,
        analyzer: Optional[Union[Analyzer, AnalyzerConfig, Dict[str, Any]]] = None,
        embedding_function: Optional[Any] = None,
    ):
        if NativeCollection is None:
            raise RuntimeError(
                "SutraDB Rust extension (_sutradb) is not available. "
                "Build it with `maturin develop --release`."
            )

        self.name = validate_collection_name(name)
        self.dimension = int(dimension)
        self.metric = Metric(metric) if isinstance(metric, str) else metric
        self.storage_path = Path(storage_path) if storage_path else None
        self.embedding_function = embedding_function
        self._ann_config = ann_config
        self._enable_wal = bool(enable_wal)
        self._custom_wal: Optional[Any] = None

        if isinstance(analyzer, Analyzer):
            self.analyzer = analyzer
        elif isinstance(analyzer, AnalyzerConfig):
            self.analyzer = Analyzer(analyzer)
        elif isinstance(analyzer, dict):
            self.analyzer = Analyzer(AnalyzerConfig(**analyzer))
        else:
            self.analyzer = Analyzer()

        analyzer_dict = {
            "lowercase": self.analyzer.config.lowercase,
            "stemmer": self.analyzer.config.stemmer,
            "stopwords": self.analyzer.config.stopwords,
            "language": self.analyzer.config.language,
            "split_identifiers": self.analyzer.config.split_identifiers,
            "cjk_bigrams": self.analyzer.config.cjk_bigrams,
        }

        self._sutra_file: Optional[Path] = None
        self._wal_file: Optional[Path] = None
        native_storage_str: Optional[str] = None
        should_write_wal_config = False

        if self.storage_path is not None:
            self._sutra_file = (
                self.storage_path
                if self.storage_path.suffix == ".sutra"
                else self.storage_path.with_suffix(".sutra")
            )
            self._wal_file = self.storage_path.with_suffix(".wal")
            # Pass normalized .sutra path to Rust so with_extension("wal") aligns with self._wal_file.
            native_storage_str = str(self._sutra_file)
            if self._enable_wal:
                wal_empty = (not self._wal_file.exists()) or (self._wal_file.stat().st_size == 0)
                if wal_empty and not self._sutra_file.exists():
                    should_write_wal_config = True

        self._native = NativeCollection(
            name=self.name,
            dimension=self.dimension,
            metric=self.metric.value,
            storage_path=native_storage_str,
            enable_wal=self._enable_wal,
            analyzer=analyzer_dict,
        )
        if should_write_wal_config:
            self._native.write_wal_config()

        self._wal_proxy: Optional[_NativeWalProxy] = (
            _NativeWalProxy(self._wal_file, self._native)
            if (self._enable_wal and self._wal_file is not None)
            else None
        )

    @property
    def vectors(self) -> np.ndarray:
        return self._native.vectors_numpy()

    @property
    def documents(self) -> List[Document]:
        raw_docs = self._native.all_documents_raw()
        return [Document.from_dict(d, d["vector"]) for d in raw_docs]

    @property
    def id_to_index(self) -> Dict[str, int]:
        return self._native.id_to_index()

    @property
    def bm25_index(self) -> _NativeBM25View:
        return _NativeBM25View(self)

    @property
    def wal(self) -> Optional[Any]:
        if self._custom_wal is not None:
            return self._custom_wal
        return self._wal_proxy

    @wal.setter
    def wal(self, value: Optional[Any]) -> None:
        self._custom_wal = value
        if value is not None and hasattr(value, "path"):
            self._native.attach_wal(str(value.path))

    def _replay_wal_if_present(self) -> None:
        wal_obj = self.wal
        if wal_obj is not None and hasattr(wal_obj, "path"):
            self._native.replay_wal(str(wal_obj.path))
        elif self._wal_file is not None and self._wal_file.exists():
            self._native.replay_wal(str(self._wal_file))

    @property
    def _sq8_enabled(self) -> bool:
        return bool(self._native.stats()["quantized_sq8"])

    @property
    def _bq_enabled(self) -> bool:
        return bool(self._native.stats()["quantized_bq"])

    @property
    def _pq_enabled(self) -> bool:
        return bool(self._native.stats()["quantized_pq"])

    @property
    def _sq8_vectors(self) -> List[SQ8Vector]:
        return [
            SQ8Vector(
                np.frombuffer(data, dtype=np.uint8).copy()
                if isinstance(data, (bytes, bytearray))
                else np.asarray(data, dtype=np.uint8),
                min_val,
                scale,
            )
            for data, min_val, scale in self._native.sq8_vectors_info()
        ]

    @property
    def _bq_vectors(self) -> List[BQVector]:
        return [BQVector(bits, dim) for bits, dim in self._native.bq_vectors_info()]

    @property
    def _pq_vectors(self) -> List[PQVector]:
        return [
            PQVector(
                np.frombuffer(codes, dtype=np.uint8).copy()
                if isinstance(codes, (bytes, bytearray))
                else np.asarray(codes, dtype=np.uint8)
            )
            for codes in self._native.pq_vectors_info()
        ]

    @property
    def _hnsw_index(self) -> Optional[HNSWIndex]:
        layers = self._native.hnsw_layers()
        if layers is None:
            return None
        st = self._native.stats().get("hnsw", {})
        cfg = HNSWConfig(
            m=int(st.get("m", 16)),
            ef_construction=int(st.get("ef_construction", 200)),
            ef_search=int(st.get("ef_search", 64)),
        )
        idx = HNSWIndex(cfg)
        idx.layers = layers
        return idx

    @property
    def _ann_index(self) -> Optional[IVFIndex]:
        st = self._native.stats()
        ann = st.get("ann")
        if not ann:
            return None
        ivf = IVFIndex(
            IVFConfig(n_lists=int(ann.get("n_lists", 1)), nprobe=int(ann.get("nprobe", 16))),
            metric=self.metric,
        )
        ivf.is_trained = bool(ann.get("trained", False))
        ivf._n_vectors = int(ann.get("n_vectors", 0))
        return ivf

    def insert(
        self,
        documents: Union[Document, List[Document], Dict[str, Any], List[Dict[str, Any]]],
        vectors: Optional[Union[List[List[float]], np.ndarray]] = None,
    ) -> int:
        """
        Inserts one or more documents into the collection via zero-copy NumPy -> Rust PyO3 transfer.
        Automatically computes embeddings for documents missing vectors when EmbeddingFunction is configured.
        """
        if not isinstance(documents, list):
            doc_list = [documents]
        else:
            doc_list = documents

        if not doc_list:
            return self.count()

        n = len(doc_list)
        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Any] = []
        namespaces: List[Optional[str]] = []
        sparse_vectors: List[Optional[Mapping[Any, Any]]] = []
        colbert_tokens_list: List[Optional[List[List[float]]]] = []
        vec_rows: Optional[List[np.ndarray]] = [] if vectors is None else None
        missing_indices: List[int] = []

        for i, item in enumerate(doc_list):
            if isinstance(item, Document):
                doc_id = item.id
                text = item.text
                meta = item.metadata
                ns = item.namespace
                sp = item.sparse_vector
                cb = item.colbert_tokens
                raw_vec = item.vector
            elif isinstance(item, dict):
                doc_id = item.get("id")
                if doc_id is None:
                    doc_id = str(uuid.uuid4())
                text = item.get("text", "")
                meta = item.get("metadata", {})
                ns = item.get("namespace")
                sp = item.get("sparse_vector")
                cb = item.get("colbert_tokens")
                raw_vec = item.get("vector")
            else:
                raise TypeError(f"Unsupported document type: {type(item)}")

            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f"Document id must be a non-empty string, got {doc_id!r}")

            if sp is not None:
                if not isinstance(sp, Mapping):
                    raise ValueError(f"Document {doc_id} sparse_vector must be a mapping")
                for sk, sv in sp.items():
                    if isinstance(sk, bool) or not isinstance(sk, (int, str)):
                        raise ValueError(
                            f"Document {doc_id} sparse_vector key must be int or str, got {type(sk).__name__}"
                        )
                    if (
                        isinstance(sv, bool)
                        or not isinstance(sv, (int, float, np.number))
                        or not math.isfinite(float(sv))
                    ):
                        raise ValueError(f"Document {doc_id} sparse_vector value must be a finite number")

            if cb is not None:
                for tok in cb:
                    tok_arr = np.asarray(tok, dtype=np.float32)
                    if tok_arr.ndim != 1 or tok_arr.shape[0] != self.dimension:
                        raise ValueError(
                            f"Document {doc_id} colbert_tokens token dimension mismatch: "
                            f"expected 1D vector of length {self.dimension}"
                        )

            ids.append(doc_id)
            texts.append(text)
            metadatas.append(meta if meta is not None else {})
            namespaces.append(ns)
            sparse_vectors.append(sp)
            colbert_tokens_list.append(cb)

            if vec_rows is not None:
                if raw_vec is None:
                    missing_indices.append(i)
                    vec_rows.append(np.empty(0, dtype=np.float32))
                else:
                    vec_arr = np.asarray(raw_vec, dtype=np.float32).ravel()
                    if vec_arr.shape[0] != self.dimension:
                        raise ValueError(
                            f"Vector dimension mismatch for doc {doc_id}: "
                            f"expected {self.dimension}, got {vec_arr.shape[0]}"
                        )
                    vec_rows.append(vec_arr)

        if vec_rows is not None and missing_indices:
            if self.embedding_function is None:
                from sutradb.embeddings import DefaultEmbeddingFunction
                self.embedding_function = DefaultEmbeddingFunction(dimension=self.dimension)
            texts_to_embed = [texts[idx] for idx in missing_indices]
            computed_vecs = self.embedding_function.embed_documents(texts_to_embed)
            sparse_fn = getattr(self.embedding_function, "encode_sparse", None)
            colbert_fn = getattr(self.embedding_function, "encode_colbert", None)
            computed_sparse = sparse_fn(texts_to_embed) if callable(sparse_fn) else None
            computed_colbert = colbert_fn(texts_to_embed) if callable(colbert_fn) else None

            for j, doc_idx in enumerate(missing_indices):
                vec_arr = np.asarray(computed_vecs[j], dtype=np.float32).ravel()
                if vec_arr.shape[0] != self.dimension:
                    raise ValueError(
                        f"Vector dimension mismatch for doc {ids[doc_idx]}: "
                        f"expected {self.dimension}, got {vec_arr.shape[0]}"
                    )
                vec_rows[doc_idx] = vec_arr
                src_doc = doc_list[doc_idx]
                if isinstance(src_doc, Document):
                    src_doc.vector = vec_arr
                elif isinstance(src_doc, dict):
                    src_doc["vector"] = vec_arr
                if sparse_vectors[doc_idx] is None and computed_sparse:
                    sparse_vectors[doc_idx] = computed_sparse[j]
                    if isinstance(src_doc, Document):
                        src_doc.sparse_vector = computed_sparse[j]
                if colbert_tokens_list[doc_idx] is None and computed_colbert:
                    colbert_tokens_list[doc_idx] = computed_colbert[j]
                    if isinstance(src_doc, Document):
                        src_doc.colbert_tokens = computed_colbert[j]

        if vectors is not None:
            vec_matrix = np.asarray(vectors, dtype=np.float32)
            if vec_matrix.ndim != 2 or vec_matrix.shape[0] != n or vec_matrix.shape[1] != self.dimension:
                got_shape = vec_matrix.shape if vec_matrix.ndim >= 1 else ()
                raise ValueError(
                    f"Vector dimension mismatch: expected ({n}, {self.dimension}), got {got_shape}"
                )
        else:
            assert vec_rows is not None
            vec_matrix = np.vstack(vec_rows).astype(np.float32, copy=False)

        if not vec_matrix.flags["C_CONTIGUOUS"]:
            vec_matrix = np.ascontiguousarray(vec_matrix, dtype=np.float32)

        return self._native.insert_numpy(
            vec_matrix,
            ids,
            texts,
            metadatas,
            namespaces,
            sparse_vectors,
            colbert_tokens_list,
        )

    def add_texts(
        self,
        texts: Union[str, Sequence[str]],
        metadatas: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, Sequence[str]]] = None,
        namespace: Optional[str] = None,
    ) -> List[str]:
        """Ingests raw text strings directly into the collection with automated embedding."""
        text_list = [texts] if isinstance(texts, str) else list(texts)
        n = len(text_list)
        if n == 0:
            return []

        if metadatas is None:
            meta_list = [{} for _ in range(n)]
        elif isinstance(metadatas, dict):
            meta_list = [metadatas]
        else:
            meta_list = list(metadatas)

        if ids is None:
            id_list = [str(uuid.uuid4()) for _ in range(n)]
        elif isinstance(ids, str):
            id_list = [ids]
        else:
            id_list = list(ids)

        if len(meta_list) != n:
            raise ValueError(f"Length of metadatas ({len(meta_list)}) does not match texts ({n})")
        if len(id_list) != n:
            raise ValueError(f"Length of ids ({len(id_list)}) does not match texts ({n})")

        if self.embedding_function is None:
            from sutradb.embeddings import DefaultEmbeddingFunction
            self.embedding_function = DefaultEmbeddingFunction(dimension=self.dimension)

        dense_vecs = self.embedding_function.embed_documents(text_list)
        sparse_fn = getattr(self.embedding_function, "encode_sparse", None)
        colbert_fn = getattr(self.embedding_function, "encode_colbert", None)
        sparse_vecs = sparse_fn(text_list) if callable(sparse_fn) else [None] * n
        colbert_toks = colbert_fn(text_list) if callable(colbert_fn) else [None] * n

        docs = [
            Document(
                id=doc_id,
                vector=vec,
                text=txt,
                metadata=meta,
                namespace=namespace,
                sparse_vector=sp,
                colbert_tokens=cb,
            )
            for doc_id, vec, txt, meta, sp, cb in zip(
                id_list, dense_vecs, text_list, meta_list, sparse_vecs, colbert_toks
            )
        ]
        self.insert(docs)
        return id_list

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        hybrid: bool = True,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        sparse_weight: float = 1.0,
        colbert_weight: float = 1.0,
        use_ann: bool = False,
    ) -> List[SearchResult]:
        """Executes end-to-end multi-modal search directly from a query text string."""
        if self.embedding_function is None:
            from sutradb.embeddings import DefaultEmbeddingFunction
            self.embedding_function = DefaultEmbeddingFunction(dimension=self.dimension)

        q_vec = self.embedding_function.embed_query(query)
        sparse_fn = getattr(self.embedding_function, "encode_sparse", None)
        colbert_fn = getattr(self.embedding_function, "encode_colbert", None)
        sp_query = sparse_fn([query])[0] if callable(sparse_fn) else None
        cb_query = colbert_fn([query])[0] if callable(colbert_fn) else None

        return self.query(
            vector=q_vec,
            text=query,
            sparse_vector=sp_query,
            colbert_tokens=cb_query,
            filter=filter,
            namespace=namespace,
            top_k=top_k,
            hybrid=hybrid,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            sparse_weight=sparse_weight,
            colbert_weight=colbert_weight,
            use_ann=use_ann,
        )

    def upsert(
        self,
        documents: Union[Document, List[Document], Dict[str, Any], List[Dict[str, Any]]],
    ) -> int:
        """Idempotent batch insertion or update."""
        return self.insert(documents)

    def update_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """In-place update of document metadata without copying vectors."""
        return self._native.update_metadata(doc_id, metadata)

    def enable_sq8(self) -> None:
        """Enables 8-bit scalar quantization to reduce RAM by 75%."""
        self._native.enable_sq8()

    def enable_bq(self) -> None:
        """Enables 1-bit binary quantization (BQ)."""
        self._native.enable_bq()

    def train_pq(self, m: Optional[int] = None, max_samples: Optional[int] = None) -> None:
        """Trains a Product Quantization codebook across M sub-spaces."""
        if m is not None and m <= 0:
            raise ValueError(f"PQ sub-spaces M must be > 0, got {m}")
        self._native.train_pq(m=m, max_samples=max_samples)

    def enable_pq(self) -> None:
        """Enables Product Quantization. Automatically trains codebook if not yet initialized."""
        self._native.enable_pq()

    def build_hnsw(self, m: int = 16, ef_construction: int = 200, ef_search: int = 64) -> Dict[str, Any]:
        """Builds HNSW graph index for O(log N) ANN vector search."""
        self._native.build_hnsw(m=m, ef_construction=ef_construction, ef_search=ef_search)
        return self.stats().get("hnsw", {})

    def build_index(self, config: Optional[IVFConfig] = None) -> IVFIndex:
        """Builds or rebuilds the IVF-Flat approximate nearest neighbor index."""
        n = self.count()
        if n == 0:
            raise ValueError("Cannot build index on an empty collection.")
        effective_config = config or self._ann_config or IVFConfig(n_lists=auto_n_lists(n))
        self._ann_config = effective_config
        self._native.build_ivf(n_lists=effective_config.n_lists, nprobe=effective_config.nprobe)
        idx = IVFIndex(effective_config, metric=self.metric)
        idx.is_trained = True
        idx._n_vectors = n
        return idx

    def query(
        self,
        vector: Optional[Union[List[float], np.ndarray]] = None,
        text: Optional[str] = None,
        sparse_vector: Optional[Dict[Union[int, str], float]] = None,
        colbert_tokens: Optional[List[List[float]]] = None,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        top_k: int = 10,
        hybrid: bool = True,
        rrf_constant: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        sparse_weight: float = 1.0,
        colbert_weight: float = 1.0,
        alpha: Optional[float] = None,
        use_ann: bool = False,
    ) -> List[SearchResult]:
        """Executes vector, lexical, sparse, ColBERT, or hybrid search via Rust NativeCollection."""
        if top_k <= 0:
            return []

        q_vec: Optional[np.ndarray] = None
        if vector is not None:
            q_vec = np.asarray(vector, dtype=np.float32).ravel()
            if not q_vec.flags["C_CONTIGUOUS"]:
                q_vec = np.ascontiguousarray(q_vec, dtype=np.float32)

        q_text = text if (text and text.strip()) else None
        q_sparse = (
            {str(k): float(v) for k, v in sparse_vector.items()}
            if sparse_vector
            else None
        )

        if q_vec is None and q_text is None and not q_sparse and not colbert_tokens:
            if self.count() == 0:
                return []
            raise ValueError(
                "Query must specify at least one modality: vector, text, sparse_vector, or colbert_tokens."
            )

        raw_results = self._native.query(
            vector=q_vec,
            text=q_text,
            sparse_vector=q_sparse,
            colbert_tokens=colbert_tokens,
            filter=filter,
            namespace=namespace,
            top_k=top_k,
            hybrid=hybrid,
            rrf_constant=rrf_constant,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            sparse_weight=sparse_weight,
            colbert_weight=colbert_weight,
            alpha=alpha,
            use_ann=use_ann,
        )

        return [
            SearchResult(
                id=r["id"],
                score=float(r["score"]),
                text=r["text"],
                metadata=r["metadata"],
                dense_score=r.get("dense_score"),
                bm25_score=r.get("bm25_score"),
                sparse_score=r.get("sparse_score"),
                colbert_score=r.get("colbert_score"),
                namespace=r.get("namespace"),
            )
            for r in raw_results
        ]

    def get(self, doc_id: str) -> Optional[Document]:
        """Retrieves a single active document by its unique ID."""
        d = self._native.get(doc_id)
        if d is None:
            return None
        return Document.from_dict(d, d.get("vector"))

    def get_all(self) -> List[Document]:
        """Retrieves all active (non-tombstoned) documents currently stored."""
        raw_docs = self._native.get_all(include_vectors=True)
        return [Document.from_dict(d, d.get("vector")) for d in raw_docs]

    def delete(self, doc_ids: Union[str, List[str]]) -> int:
        """Soft-deletes one or more documents by ID using tombstones."""
        return self._native.delete(doc_ids)

    def count(self) -> int:
        """Returns the number of active (non-tombstoned) documents currently stored."""
        return self._native.count()

    def compact(self) -> int:
        """Compacts tombstoned documents and rebuilds active ANN indexes. Returns purged count."""
        return self._native.compact()

    def save(self, filepath: Optional[Union[str, Path]] = None) -> int:
        """Persists the collection to a memory-aligned .sutra binary file."""
        target = filepath if filepath is not None else self._sutra_file
        if target is None:
            raise ValueError("No storage path provided for collection save")
        target_path = Path(target)
        if target_path.suffix != ".sutra":
            target_path = target_path.with_suffix(".sutra")
        return self._native.save(str(target_path))

    def load(self, filepath: Optional[Union[str, Path]] = None) -> int:
        """Loads vectors and documents from a .sutra binary file."""
        target = filepath if filepath is not None else self.storage_path
        if target is None:
            raise ValueError("No storage path provided for collection load")
        target_path = Path(target)
        if not target_path.exists() and target_path.suffix != ".sutra":
            candidate = target_path.with_suffix(".sutra")
            if candidate.exists():
                target_path = candidate
        if not target_path.exists():
            raise FileNotFoundError(f"Cannot find database file at {target_path}")

        count = self._native.load(str(target_path))
        st = self._native.stats()
        self.dimension = int(st["dimension"])
        self.metric = Metric(st["metric"])
        return count

    def stats(self) -> Dict[str, Any]:
        """Returns collection operational and memory statistics."""
        res = self._native.stats()
        res["vector_memory_mb"] = round(float(res.get("vector_memory_mb", 0.0)), 3)
        return res


class SutraDB:
    """Top-level database management interface."""

    def __init__(self, persist_directory: Optional[Union[str, Path]] = None):
        self.persist_dir = Path(persist_directory) if persist_directory else None
        if self.persist_dir:
            try:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        self.collections: Dict[str, Collection] = {}

    def _collection_paths(self, name: str) -> Tuple[Path, Path]:
        validate_collection_name(name)
        assert self.persist_dir is not None
        base_dir = self.persist_dir.resolve()
        disk_file = (self.persist_dir / f"{name}.sutra").resolve()
        wal_file = (self.persist_dir / f"{name}.wal").resolve()
        if not disk_file.is_relative_to(base_dir) or not wal_file.is_relative_to(base_dir):
            raise ValueError(f"Collection path escapes persist directory: {name}")
        return disk_file, wal_file

    def create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        metric: Union[Metric, str] = Metric.COSINE,
        enable_wal: bool = False,
        analyzer: Optional[Union[Analyzer, AnalyzerConfig, Dict[str, Any]]] = None,
        embedding_function: Optional[Any] = None,
    ) -> Collection:
        """Creates and registers a new collection."""
        validate_collection_name(name)
        if name in self.collections:
            raise ValueError(f"Collection '{name}' already exists.")

        if dimension is None:
            if embedding_function is not None and hasattr(embedding_function, "dimension"):
                dimension = int(embedding_function.dimension)
            else:
                dimension = 384

        storage_path = None
        if self.persist_dir:
            storage_path, _ = self._collection_paths(name)

        col = Collection(
            name=name,
            dimension=dimension,
            metric=metric,
            storage_path=storage_path,
            enable_wal=enable_wal,
            analyzer=analyzer,
            embedding_function=embedding_function,
        )
        self.collections[name] = col
        return col

    def get_collection(self, name: str) -> Collection:
        """Retrieves an existing registered collection."""
        validate_collection_name(name)
        if name not in self.collections:
            if self.persist_dir:
                disk_file, wal_file = self._collection_paths(name)
                if disk_file.exists():
                    has_wal = wal_file.exists()
                    col = Collection(
                        name=name, dimension=1, storage_path=disk_file, enable_wal=False
                    )
                    col.load()
                    if has_wal:
                        col.wal = WriteAheadLog(wal_file)
                        col._replay_wal_if_present()
                    self.collections[name] = col
                    return col
                elif wal_file.exists():
                    temp_wal = WriteAheadLog(wal_file)
                    entries = temp_wal.replay()
                    temp_wal.close()
                    dim = 1
                    metric: str = Metric.COSINE.value
                    enable_wal = True
                    found_config = False
                    for entry in entries:
                        if entry.get("op") == "config":
                            dim = int(entry.get("dimension", dim))
                            metric = str(entry.get("metric", metric))
                            enable_wal = bool(entry.get("enable_wal", True))
                            found_config = True
                            break
                    if not found_config:
                        for entry in entries:
                            if entry.get("op") == "insert" and entry.get("docs"):
                                vec = entry["docs"][0].get("vector", [])
                                if vec:
                                    dim = len(vec)
                                    break
                    col = Collection(
                        name=name,
                        dimension=dim,
                        metric=metric,
                        storage_path=disk_file,
                        enable_wal=enable_wal,
                    )
                    self.collections[name] = col
                    return col

            raise KeyError(f"Collection '{name}' not found.")
        return self.collections[name]

    def get_or_create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        metric: Union[Metric, str] = Metric.COSINE,
        enable_wal: bool = False,
        analyzer: Optional[Union[Analyzer, AnalyzerConfig, Dict[str, Any]]] = None,
        embedding_function: Optional[Any] = None,
    ) -> Collection:
        """Retrieves or initializes a collection."""
        try:
            return self.get_collection(name)
        except KeyError:
            return self.create_collection(
                name,
                dimension,
                metric,
                enable_wal=enable_wal,
                analyzer=analyzer,
                embedding_function=embedding_function,
            )

    def list_collections(self) -> List[str]:
        """Returns a list of all active collection names."""
        names = set(self.collections.keys())
        if self.persist_dir and self.persist_dir.exists():
            for f in self.persist_dir.glob("*.sutra"):
                if _COLLECTION_NAME_RE.match(f.stem):
                    names.add(f.stem)
            for f in self.persist_dir.glob("*.wal"):
                if _COLLECTION_NAME_RE.match(f.stem):
                    names.add(f.stem)
        return sorted(list(names))

    def drop_collection(self, name: str) -> bool:
        """Removes a collection from memory and deletes its files on disk."""
        validate_collection_name(name)
        col = self.collections.pop(name, None)
        dropped = col is not None

        if col is not None and col.wal is not None:
            col.wal.close()

        if self.persist_dir:
            sutra_file, wal_file = self._collection_paths(name)
            if sutra_file.exists():
                sutra_file.unlink()
                dropped = True
            if wal_file.exists():
                wal_file.unlink()
                dropped = True

        return dropped
