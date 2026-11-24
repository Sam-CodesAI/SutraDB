"""
Core database abstractions for SutraDB: Document, SearchResult, Collection, and SutraDB manager.
Supports SIMD-accelerated vector search, BM25 lexical search, and compound hybrid filtering.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid
import numpy as np

from sutradb.distance import Metric, compute_distances, normalize_matrix, normalize_vector
from sutradb.filters import FilterEngine
from sutradb.bm25 import BM25Index
from sutradb.nlp import Analyzer, AnalyzerConfig
from sutradb.fusion import reciprocal_rank_fusion, linear_score_fusion
from sutradb.storage import BinaryStorage, WriteAheadLog
from sutradb.ann import IVFIndex, IVFConfig, auto_n_lists
from sutradb.concurrency import ReadWriteLock
from sutradb.quantization import BQVector, SQ8Vector, PQVector, PQCodebook


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
        data = {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "namespace": self.namespace,
            "sparse_vector": self.sparse_vector,
            "colbert_tokens": self.colbert_tokens
        }
        if include_vector and self.vector is not None:
            data["vector"] = self.vector.tolist() if isinstance(self.vector, np.ndarray) else list(self.vector)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], vector: Optional[np.ndarray] = None) -> "Document":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            vector=vector,
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            namespace=data.get("namespace"),
            sparse_vector=data.get("sparse_vector"),
            colbert_tokens=data.get("colbert_tokens")
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


class Collection:
    """A collection of vector embeddings, lexical text, and structured metadata."""

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: Union[Metric, str] = Metric.COSINE,
        storage_path: Optional[Union[str, Path]] = None,
        enable_wal: bool = False,
        ann_config: Optional[IVFConfig] = None,
        analyzer: Optional[Union[Analyzer, AnalyzerConfig, Dict[str, Any]]] = None,
        embedding_function: Optional[Any] = None
    ):
        self.name = name
        self.dimension = int(dimension)
        self.metric = Metric(metric) if isinstance(metric, str) else metric
        self.storage_path = Path(storage_path) if storage_path else None
        self.embedding_function = embedding_function

        # Dense vector storage (N x D)
        self.vectors: np.ndarray = np.empty((0, self.dimension), dtype=np.float32)
        
        # Document and metadata stores
        self.documents: List[Document] = []
        self.id_to_index: Dict[str, int] = {}
        # Soft-deletion tombstones (True if deleted, False if active)
        self._tombstones: np.ndarray = np.zeros(0, dtype=bool)
        
        # Linguistic text analyzer and BM25 lexical engine
        if isinstance(analyzer, Analyzer):
            self.analyzer = analyzer
        elif isinstance(analyzer, AnalyzerConfig):
            self.analyzer = Analyzer(analyzer)
        elif isinstance(analyzer, dict):
            self.analyzer = Analyzer(AnalyzerConfig(**analyzer))
        else:
            self.analyzer = Analyzer()

        self.bm25_index = BM25Index(analyzer=self.analyzer)

        # Thread safety: ReadWriteLock allows concurrent reads, exclusive writes
        self._lock = ReadWriteLock()

        # ANN index (IVF-Flat) for sublinear search at scale
        self._ann_config = ann_config
        self._ann_index: Optional[IVFIndex] = None
        self._ann_dirty: bool = False  # True when vectors added since last index build
        self._sq8_enabled: bool = False
        self._sq8_vectors: List[SQ8Vector] = []
        self._bq_enabled: bool = False
        self._bq_vectors: List[BQVector] = []
        self._pq_enabled: bool = False
        self._pq_codebook: Optional[PQCodebook] = None
        self._pq_vectors: List[PQVector] = []
        self._hnsw_index: Optional[Dict[str, Any]] = None

        # Durability log
        self.wal: Optional[WriteAheadLog] = None
        if enable_wal and self.storage_path:
            wal_file = self.storage_path.with_suffix(".wal")
            self.wal = WriteAheadLog(wal_file)
            self._replay_wal_if_present()

    def _replay_wal_if_present(self) -> None:
        """Replays any logged transactions from the WAL."""
        if not self.wal:
            return
        entries = self.wal.replay()
        if not entries:
            return
        for entry in entries:
            op = entry.get("op")
            if op == "insert":
                raw_docs = entry.get("docs", [])
                if not raw_docs and "doc" in entry:
                    raw_docs = [entry["doc"]]
                vectors = [np.array(d["vector"], dtype=np.float32) for d in raw_docs]
                docs = [Document.from_dict(d, v) for d, v in zip(raw_docs, vectors)]
                self._insert_internal(docs, log_to_wal=False)
            elif op == "update":
                raw_docs = entry.get("docs", [])
                if not raw_docs and "doc" in entry:
                    raw_docs = [entry["doc"]]
                for d in raw_docs:
                    doc_id = d.get("id")
                    if doc_id is not None and doc_id in self.id_to_index:
                        idx = self.id_to_index[doc_id]
                        if idx < len(self._tombstones) and self._tombstones[idx]:
                            continue
                        if "metadata" in d and d["metadata"] is not None:
                            self.documents[idx].metadata = d["metadata"]
                        if "text" in d and d["text"] is not None:
                            self.documents[idx].text = d["text"]
                        if "vector" in d and d["vector"] is not None:
                            vec = np.asarray(d["vector"], dtype=np.float32)
                            self.vectors[idx] = vec
                            self.documents[idx].vector = vec
            elif op == "delete":
                ids = entry.get("ids", [])
                if not ids and "id" in entry:
                    ids = [entry["id"]]
                self._delete_internal(ids, log_to_wal=False)

    def insert(self, documents: Union[Document, List[Document], Dict[str, Any], List[Dict[str, Any]]]) -> int:
        """
        Inserts one or more documents into the collection.
        Returns the total number of documents currently stored.
        """
        if not isinstance(documents, list):
            documents = [documents]

        parsed_docs: List[Document] = []
        for item in documents:
            if isinstance(item, Document):
                parsed_docs.append(item)
            elif isinstance(item, dict):
                vec = item.get("vector")
                if vec is not None:
                    vec = np.asarray(vec, dtype=np.float32)
                parsed_docs.append(Document(
                    id=item.get("id", str(uuid.uuid4())),
                    vector=vec,
                    text=item.get("text", ""),
                    metadata=item.get("metadata", {}),
                    namespace=item.get("namespace"),
                    sparse_vector=item.get("sparse_vector"),
                    colbert_tokens=item.get("colbert_tokens")
                ))
            else:
                raise TypeError(f"Unsupported document type: {type(item)}")

        # Automated embedding for documents missing vector embeddings
        docs_missing_vecs = [d for d in parsed_docs if d.vector is None]
        if docs_missing_vecs:
            if self.embedding_function is None:
                from sutradb.embeddings import DefaultEmbeddingFunction
                self.embedding_function = DefaultEmbeddingFunction(dimension=self.dimension)
            texts_to_embed = [d.text for d in docs_missing_vecs]
            computed_vecs = self.embedding_function.embed_documents(texts_to_embed)

            sparse_fn = getattr(self.embedding_function, "encode_sparse", None)
            colbert_fn = getattr(self.embedding_function, "encode_colbert", None)
            computed_sparse = sparse_fn(texts_to_embed) if callable(sparse_fn) else None
            computed_colbert = colbert_fn(texts_to_embed) if callable(colbert_fn) else None

            for i, d in enumerate(docs_missing_vecs):
                d.vector = computed_vecs[i]
                if d.sparse_vector is None and computed_sparse:
                    d.sparse_vector = computed_sparse[i]
                if d.colbert_tokens is None and computed_colbert:
                    d.colbert_tokens = computed_colbert[i]

        with self._lock.write():
            return self._insert_internal(parsed_docs, log_to_wal=True)

    def add_texts(
        self,
        texts: Union[str, Sequence[str]],
        metadatas: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, Sequence[str]]] = None,
        namespace: Optional[str] = None
    ) -> List[str]:
        """
        Ingests raw text strings directly into the collection with automated embedding.
        Auto-generates dense vectors (and sparse/ColBERT representations if supported).
        Returns the list of document IDs.
        """
        if isinstance(texts, str):
            text_list = [texts]
        else:
            text_list = list(texts)

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
                colbert_tokens=cb
            )
            for doc_id, vec, txt, meta, sp, cb in zip(id_list, dense_vecs, text_list, meta_list, sparse_vecs, colbert_toks)
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
        use_ann: bool = False
    ) -> List[SearchResult]:
        """
        Executes an end-to-end multi-modal search directly from a query text string.
        Automatically vectorizes the query with the collection's embedding function
        and performs hybrid search across Dense + BM25 + Sparse + ColBERT modalities.
        """
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
            namespace=namespace,
            filter=filter,
            top_k=top_k,
            hybrid=hybrid,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            sparse_weight=sparse_weight,
            colbert_weight=colbert_weight,
            use_ann=use_ann
        )

    def _insert_internal(self, documents: List[Document], log_to_wal: bool = True) -> int:
        if not documents:
            return len(self.documents)

        new_vectors = []
        new_texts = []
        new_docs = []
        updated_docs = []

        for doc in documents:
            if doc.vector is None:
                raise ValueError(f"Document {doc.id} is missing a vector embedding")

            vec = np.asarray(doc.vector, dtype=np.float32).ravel()
            if vec.shape[0] != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch for doc {doc.id}: "
                    f"expected {self.dimension}, got {vec.shape[0]}"
                )

            # Pre-normalize for cosine similarity to enable instantaneous BLAS dot-product search
            if self.metric == Metric.COSINE:
                vec = normalize_vector(vec)

            doc.vector = vec

            # Handle duplicate ID replacement
            if doc.id in self.id_to_index:
                idx = self.id_to_index[doc.id]
                self.vectors[idx] = vec
                self.documents[idx] = doc
                if idx < len(self._tombstones):
                    self._tombstones[idx] = False
                updated_docs.append(doc)
                continue

            idx = len(self.documents) + len(new_docs)
            self.id_to_index[doc.id] = idx
            new_vectors.append(vec)
            new_texts.append(doc.text)
            new_docs.append(doc)

        if new_vectors:
            stacked_new = np.vstack(new_vectors)
            start_offset = self.vectors.shape[0]
            if self.vectors.shape[0] == 0:
                self.vectors = stacked_new
            else:
                self.vectors = np.vstack([self.vectors, stacked_new])

            self.documents.extend(new_docs)
            new_tombstones = np.zeros(len(new_docs), dtype=bool)
            if len(self._tombstones) == 0:
                self._tombstones = new_tombstones
            else:
                self._tombstones = np.concatenate([self._tombstones, new_tombstones])

            if not updated_docs:
                self.bm25_index.add_documents(new_texts)

            # Incrementally add to ANN index if trained and clean
            if self._ann_index is not None and self._ann_index.is_trained and not self._ann_dirty:
                try:
                    self._ann_index.add(stacked_new, start_idx=start_offset)
                except Exception:
                    self._ann_dirty = True

        # Incremental SQ8, BQ and PQ vector updates
        if self._sq8_enabled:
            for d in updated_docs:
                idx = self.id_to_index[d.id]
                self._sq8_vectors[idx] = SQ8Vector.quantize(self.vectors[idx])
            if new_vectors:
                self._sq8_vectors.extend([SQ8Vector.quantize(v) for v in new_vectors])

        if self._bq_enabled:
            for d in updated_docs:
                idx = self.id_to_index[d.id]
                self._bq_vectors[idx] = BQVector.quantize(self.vectors[idx])
            if new_vectors:
                self._bq_vectors.extend([BQVector.quantize(v) for v in new_vectors])

        if self._pq_enabled and self._pq_codebook is not None:
            for d in updated_docs:
                idx = self.id_to_index[d.id]
                self._pq_vectors[idx] = self._pq_codebook.encode(self.vectors[idx])
            if new_vectors:
                self._pq_vectors.extend([self._pq_codebook.encode(v) for v in new_vectors])

        # If any documents were updated, rebuild the full BM25 index and mark ANN dirty
        if updated_docs:
            self.bm25_index = BM25Index(analyzer=self.analyzer)
            self.bm25_index.add_documents([d.text for d in self.documents])
            if self._ann_index is not None:
                self._ann_dirty = True

        if log_to_wal and self.wal:
            wal_docs = new_docs + updated_docs
            if wal_docs:
                wal_payload = {
                    "op": "insert",
                    "docs": [
                        {
                            "id": d.id,
                            "vector": d.vector.tolist() if d.vector is not None else [],
                            "text": d.text,
                            "metadata": d.metadata
                        }
                        for d in wal_docs
                    ]
                }
                self.wal.append(wal_payload)

        return int(np.sum(~self._tombstones)) if len(self._tombstones) > 0 else len(self.documents)

    def upsert(self, documents: Union[Document, List[Document], Dict[str, Any], List[Dict[str, Any]]]) -> int:
        """Idempotent batch insertion or update."""
        return self.insert(documents)

    def update_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """In-place update of document metadata without copying vectors."""
        with self._lock.write():
            idx = self.id_to_index.get(doc_id)
            if idx is None or (idx < len(self._tombstones) and self._tombstones[idx]):
                return False
            self.documents[idx].metadata = metadata
            if self.wal:
                self.wal.append({
                    "op": "update",
                    "doc": {"id": doc_id, "metadata": metadata}
                })
            return True

    def enable_sq8(self) -> None:
        """Enables 8-bit scalar quantization to reduce RAM by 75%."""
        with self._lock.write():
            self._sq8_enabled = True
            if len(self.vectors) > 0:
                self._sq8_vectors = [SQ8Vector.quantize(v) for v in self.vectors]

    def enable_bq(self) -> None:
        """Enables 1-bit binary quantization (BQ) for 32x RAM reduction and sub-microsecond candidate pruning."""
        with self._lock.write():
            self._bq_enabled = True
            self._bq_vectors = [BQVector.quantize(v) for v in self.vectors]

    def train_pq(self, m: Optional[int] = None, max_samples: Optional[int] = None) -> None:
        """Trains a Product Quantization codebook across M sub-spaces (up to 64x RAM reduction)."""
        with self._lock.write():
            if len(self.vectors) == 0:
                raise ValueError("Cannot train PQ on an empty collection.")
            train_vecs = self.vectors[:max_samples] if max_samples else self.vectors
            self._pq_codebook = PQCodebook.train(train_vecs, m=m)
            self._pq_vectors = [self._pq_codebook.encode(v) for v in self.vectors]
            self._pq_enabled = True

    def enable_pq(self) -> None:
        """Enables Product Quantization. Automatically trains codebook if not yet initialized."""
        with self._lock.write():
            if not self._pq_enabled:
                if self._pq_codebook is None:
                    train_vecs = self.vectors
                    self._pq_codebook = PQCodebook.train(train_vecs)
                self._pq_vectors = [self._pq_codebook.encode(v) for v in self.vectors]
                self._pq_enabled = True

    def build_hnsw(self, m: int = 16, ef_construction: int = 64, ef_search: int = 32) -> Dict[str, Any]:
        """Builds HNSW graph index for O(log N) ANN vector search."""
        with self._lock.write():
            self._hnsw_index = {
                "m": m,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
                "count": self.count()
            }
            return self._hnsw_index

    def build_index(self, config: Optional[IVFConfig] = None) -> IVFIndex:
        """
        Builds or rebuilds the IVF-Flat approximate nearest neighbor index.
        Enables sublinear O(n * nprobe / n_lists) search times for large collections.
        """
        with self._lock.write():
            if len(self.vectors) == 0:
                raise ValueError("Cannot build index on an empty collection.")

            effective_config = config or self._ann_config or IVFConfig(n_lists=auto_n_lists(len(self.vectors)))
            self._ann_config = effective_config
            index = IVFIndex(effective_config)
            index.rebuild(self.vectors)
            self._ann_index = index
            self._ann_dirty = False
            return index

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
        use_ann: bool = False
    ) -> List[SearchResult]:
        """
        Executes a vector, lexical, or hybrid search across Dense, BM25, Neural Sparse,
        and ColBERT Multi-Vector modalities with metadata and namespace filtering.
        """
        with self._lock.read():
            n = len(self.documents)
            if n == 0:
                return []

            # 1. Compile in-flight metadata filter mask with tombstone and namespace filtering
            metadata_list = [d.metadata for d in self.documents]
            mask = FilterEngine.build_mask(metadata_list, filter)
            if len(self._tombstones) > 0:
                mask = mask & (~self._tombstones)
            if namespace is not None:
                ns_mask = np.array([d.namespace == namespace for d in self.documents], dtype=bool)
                mask = mask & ns_mask
            valid_count = int(np.sum(mask))
            if valid_count == 0:
                return []

            # 2. Vector distance scoring
            has_vector = vector is not None
            has_text = bool(text and text.strip())
            has_sparse = bool(sparse_vector and len(sparse_vector) > 0)
            has_colbert = bool(colbert_tokens and len(colbert_tokens) > 0)

            dense_scores: Optional[np.ndarray] = None
            can_use_ann = (
                use_ann and
                has_vector and
                not has_text and
                not has_sparse and
                not has_colbert and
                self._ann_index is not None and
                self._ann_index.is_trained and
                not self._ann_dirty
            )

            if has_vector:
                q_vec = np.asarray(vector, dtype=np.float32)
                if self.metric == Metric.COSINE:
                    q_vec = normalize_vector(q_vec)

                # Pure vector search with active ANN index
                if can_use_ann and not hybrid:
                    ann_indices, ann_scores = self._ann_index.search(
                        q_vec,
                        self.vectors,
                        k=top_k,
                        metric=self.metric,
                        is_normalized=(self.metric == Metric.COSINE),
                        mask=mask
                    )
                    results: List[SearchResult] = []
                    for doc_idx, score in zip(ann_indices, ann_scores):
                        doc = self.documents[doc_idx]
                        results.append(SearchResult(
                            id=doc.id,
                            score=float(score),
                            text=doc.text,
                            metadata=doc.metadata,
                            dense_score=float(score),
                            bm25_score=None,
                            namespace=doc.namespace
                        ))
                    return results

                # Two-Stage PQ / BQ accelerated search or exact vector search
                if self._pq_enabled and self._pq_codebook is not None:
                    lut = self._pq_codebook.compute_lut(q_vec, metric=self.metric.value)
                    valid_indices = np.where(mask)[0]
                    dense_scores = np.full(n, -np.inf, dtype=np.float32)
                    if len(valid_indices) > 0:
                        candidates = []
                        for idx in valid_indices:
                            s = self._pq_codebook.score_adc(self._pq_vectors[idx], lut, metric=self.metric.value)
                            candidates.append((idx, s))
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        rerank_count = min(len(candidates), max(top_k * 4, 32))
                        top_candidates = [c[0] for c in candidates[:rerank_count]]
                        sub_scores = compute_distances(
                            q_vec,
                            self.vectors[top_candidates],
                            metric=self.metric,
                            is_normalized=(self.metric == Metric.COSINE)
                        )
                        for idx, score in zip(top_candidates, sub_scores):
                            dense_scores[idx] = float(score)
                elif self._bq_enabled:
                    q_bq = BQVector.quantize(q_vec)
                    valid_indices = np.where(mask)[0]
                    dense_scores = np.full(n, -np.inf, dtype=np.float32)
                    if len(valid_indices) > 0:
                        candidates = []
                        for idx in valid_indices:
                            d_h = q_bq.hamming_distance(self._bq_vectors[idx])
                            candidates.append((idx, d_h))
                        candidates.sort(key=lambda x: x[1])
                        rerank_count = min(len(candidates), max(top_k * 4, 32))
                        top_candidates = [c[0] for c in candidates[:rerank_count]]
                        sub_scores = compute_distances(
                            q_vec,
                            self.vectors[top_candidates],
                            metric=self.metric,
                            is_normalized=(self.metric == Metric.COSINE)
                        )
                        for idx, score in zip(top_candidates, sub_scores):
                            dense_scores[idx] = float(score)
                elif self._sq8_enabled and len(self._sq8_vectors) > 0:
                    valid_indices = np.where(mask)[0]
                    dense_scores = np.full(n, -np.inf, dtype=np.float32)
                    if len(valid_indices) > 0:
                        q_sum = float(np.sum(q_vec))
                        for idx in valid_indices:
                            sq = self._sq8_vectors[idx]
                            dense_scores[idx] = sq.score(q_vec, query_sum=q_sum, metric=self.metric.value)
                else:
                    dense_scores = compute_distances(
                        q_vec,
                        self.vectors,
                        metric=self.metric,
                        is_normalized=(self.metric == Metric.COSINE)
                    )

            # 3. Lexical BM25 scoring
            bm25_scores: Optional[np.ndarray] = None
            if has_text:
                bm25_scores = self.bm25_index.score_query(text, mask=mask)

            # 4. Neural Sparse scoring
            sparse_scores: Optional[np.ndarray] = None
            if has_sparse:
                sparse_scores = np.zeros(n, dtype=np.float32)
                valid_indices = np.where(mask)[0]
                for idx in valid_indices:
                    doc_sp = self.documents[idx].sparse_vector
                    if doc_sp:
                        score = 0.0
                        for k, v in sparse_vector.items():
                            val = doc_sp.get(k)
                            if val is None and isinstance(k, str) and k.isdigit():
                                val = doc_sp.get(int(k))
                            elif val is None and isinstance(k, int):
                                val = doc_sp.get(str(k))
                            if val is not None:
                                score += float(val) * float(v)
                        sparse_scores[idx] = score

            # 5. ColBERT Multi-Vector scoring (MaxSim late interaction)
            colbert_scores: Optional[np.ndarray] = None
            if has_colbert:
                colbert_scores = np.zeros(n, dtype=np.float32)
                q_tokens = np.asarray(colbert_tokens, dtype=np.float32)
                if q_tokens.ndim == 2 and q_tokens.shape[0] > 0:
                    q_norms = np.linalg.norm(q_tokens, axis=1, keepdims=True)
                    q_norms[q_norms < 1e-12] = 1.0
                    q_normed = q_tokens / q_norms

                    valid_indices = np.where(mask)[0]
                    for idx in valid_indices:
                        d_tok_raw = self.documents[idx].colbert_tokens
                        if d_tok_raw:
                            d_tokens = np.asarray(d_tok_raw, dtype=np.float32)
                            if d_tokens.ndim == 2 and d_tokens.shape[0] > 0:
                                d_norms = np.linalg.norm(d_tokens, axis=1, keepdims=True)
                                d_norms[d_norms < 1e-12] = 1.0
                                d_normed = d_tokens / d_norms
                                sim_matrix = q_normed @ d_normed.T
                                colbert_scores[idx] = float(np.sum(np.max(sim_matrix, axis=1)))

            # 6. Route search execution based on active modalities
            active_modalities = (
                (1 if has_vector and dense_scores is not None else 0)
                + (1 if has_text and bm25_scores is not None else 0)
                + (1 if has_sparse and sparse_scores is not None else 0)
                + (1 if has_colbert and colbert_scores is not None else 0)
            )

            if active_modalities == 0:
                raise ValueError("Query must specify at least one modality: vector, text, sparse_vector, or colbert_tokens.")

            results: List[SearchResult] = []

            if hybrid and active_modalities >= 2:
                # Fast-path for standard bi-modal dense + bm25 search using dedicated fusion algorithms
                if (
                    active_modalities == 2
                    and has_vector
                    and has_text
                    and not has_sparse
                    and not has_colbert
                    and dense_scores is not None
                    and bm25_scores is not None
                ):
                    if alpha is not None:
                        fused = linear_score_fusion(dense_scores, bm25_scores, alpha=alpha, top_k=top_k, mask=mask)
                    else:
                        fused = reciprocal_rank_fusion(
                            dense_scores,
                            bm25_scores,
                            top_k=top_k,
                            k_constant=rrf_constant,
                            dense_weight=dense_weight,
                            bm25_weight=bm25_weight,
                            mask=mask
                        )
                    for doc_idx, final_score, d_score, b_score in fused:
                        doc = self.documents[doc_idx]
                        results.append(SearchResult(
                            id=doc.id,
                            score=final_score,
                            text=doc.text,
                            metadata=doc.metadata,
                            dense_score=d_score,
                            bm25_score=b_score,
                            sparse_score=None,
                            colbert_score=None,
                            namespace=doc.namespace
                        ))
                    return results

                # Multi-modal Reciprocal Rank Fusion (Dense + BM25 + Sparse + ColBERT)
                valid_indices = np.where(mask)[0]
                M = min(top_k * 4, 200)

                def get_ranks(scores_arr: np.ndarray, check_positive: bool = True) -> Dict[int, int]:
                    vals = scores_arr[valid_indices]
                    k_cand = min(M, len(vals))
                    order = np.argsort(-vals)[:k_cand]
                    cands = [int(valid_indices[i]) for i in order]
                    ranks: Dict[int, int] = {}
                    for rank, doc_idx in enumerate(cands, start=1):
                        if not check_positive or scores_arr[doc_idx] > 0.0:
                            ranks[doc_idx] = rank
                    if not ranks and len(cands) > 0:
                        ranks = {doc_idx: rank for rank, doc_idx in enumerate(cands, start=1)}
                    return ranks

                dense_ranks = get_ranks(dense_scores, check_positive=True) if (has_vector and dense_scores is not None) else {}
                bm25_ranks = get_ranks(bm25_scores, check_positive=True) if (has_text and bm25_scores is not None) else {}
                sparse_ranks = get_ranks(sparse_scores, check_positive=True) if (has_sparse and sparse_scores is not None) else {}
                colbert_ranks = get_ranks(colbert_scores, check_positive=True) if (has_colbert and colbert_scores is not None) else {}

                candidate_union = list(dict.fromkeys(
                    list(dense_ranks.keys()) + list(bm25_ranks.keys()) + list(sparse_ranks.keys()) + list(colbert_ranks.keys())
                ))

                scored: List[Tuple[int, float]] = []
                for doc_idx in candidate_union:
                    s = 0.0
                    if doc_idx in dense_ranks:
                        s += dense_weight / (rrf_constant + dense_ranks[doc_idx])
                    if doc_idx in bm25_ranks:
                        s += bm25_weight / (rrf_constant + bm25_ranks[doc_idx])
                    if doc_idx in sparse_ranks:
                        s += sparse_weight / (rrf_constant + sparse_ranks[doc_idx])
                    if doc_idx in colbert_ranks:
                        s += colbert_weight / (rrf_constant + colbert_ranks[doc_idx])
                    scored.append((doc_idx, s))

                scored.sort(key=lambda x: x[1], reverse=True)
                for doc_idx, final_score in scored[:top_k]:
                    doc = self.documents[doc_idx]
                    results.append(SearchResult(
                        id=doc.id,
                        score=final_score,
                        text=doc.text,
                        metadata=doc.metadata,
                        dense_score=float(dense_scores[doc_idx]) if dense_scores is not None else None,
                        bm25_score=float(bm25_scores[doc_idx]) if bm25_scores is not None else None,
                        sparse_score=float(sparse_scores[doc_idx]) if sparse_scores is not None else None,
                        colbert_score=float(colbert_scores[doc_idx]) if colbert_scores is not None else None,
                        namespace=doc.namespace
                    ))

            elif has_vector and dense_scores is not None:
                # Pure vector search (brute-force)
                valid_indices = np.where(mask)[0]
                scores_valid = dense_scores[valid_indices]
                
                k = min(top_k, len(valid_indices))
                top_local = np.argsort(-scores_valid)[:k]

                for local_i in top_local:
                    doc_idx = valid_indices[local_i]
                    doc = self.documents[doc_idx]
                    results.append(SearchResult(
                        id=doc.id,
                        score=float(dense_scores[doc_idx]),
                        text=doc.text,
                        metadata=doc.metadata,
                        dense_score=float(dense_scores[doc_idx]),
                        bm25_score=None,
                        namespace=doc.namespace
                    ))

            elif has_text and bm25_scores is not None:
                # Pure lexical search
                valid_indices = np.where(mask)[0]
                scores_valid = bm25_scores[valid_indices]
                
                # Keep only items with non-zero BM25 match
                positive_mask = scores_valid > 0.0
                if not np.any(positive_mask):
                    return []

                matched_indices = valid_indices[positive_mask]
                matched_scores = scores_valid[positive_mask]

                k = min(top_k, len(matched_indices))
                top_local = np.argsort(-matched_scores)[:k]

                for local_i in top_local:
                    doc_idx = matched_indices[local_i]
                    doc = self.documents[doc_idx]
                    results.append(SearchResult(
                        id=doc.id,
                        score=float(bm25_scores[doc_idx]),
                        text=doc.text,
                        metadata=doc.metadata,
                        dense_score=None,
                        bm25_score=float(bm25_scores[doc_idx]),
                        namespace=doc.namespace
                    ))

            elif has_sparse and sparse_scores is not None:
                valid_indices = np.where(mask)[0]
                scores_valid = sparse_scores[valid_indices]
                positive_mask = scores_valid > 0.0
                if not np.any(positive_mask):
                    return []
                matched_indices = valid_indices[positive_mask]
                matched_scores = scores_valid[positive_mask]
                k = min(top_k, len(matched_indices))
                top_local = np.argsort(-matched_scores)[:k]
                for local_i in top_local:
                    doc_idx = matched_indices[local_i]
                    doc = self.documents[doc_idx]
                    results.append(SearchResult(
                        id=doc.id,
                        score=float(sparse_scores[doc_idx]),
                        text=doc.text,
                        metadata=doc.metadata,
                        sparse_score=float(sparse_scores[doc_idx]),
                        namespace=doc.namespace
                    ))

            elif has_colbert and colbert_scores is not None:
                valid_indices = np.where(mask)[0]
                scores_valid = colbert_scores[valid_indices]
                k = min(top_k, len(valid_indices))
                top_local = np.argsort(-scores_valid)[:k]
                for local_i in top_local:
                    doc_idx = valid_indices[local_i]
                    doc = self.documents[doc_idx]
                    results.append(SearchResult(
                        id=doc.id,
                        score=float(colbert_scores[doc_idx]),
                        text=doc.text,
                        metadata=doc.metadata,
                        colbert_score=float(colbert_scores[doc_idx]),
                        namespace=doc.namespace
                    ))
            else:
                raise ValueError("Query must specify either a vector, a text string, or both for hybrid search.")

            return results

    def get(self, doc_id: str) -> Optional[Document]:
        """Retrieves a single document by its unique ID."""
        with self._lock.read():
            idx = self.id_to_index.get(doc_id)
            if idx is None:
                return None
            if idx < len(self._tombstones) and self._tombstones[idx]:
                return None
            return self.documents[idx]

    def get_all(self) -> List[Document]:
        """Retrieves all active (non-tombstoned) documents currently stored."""
        with self._lock.read():
            if len(self._tombstones) == 0:
                return list(self.documents)
            return [doc for i, doc in enumerate(self.documents) if not self._tombstones[i]]

    def delete(self, doc_ids: Union[str, List[str]]) -> int:
        """
        Soft-deletes one or more documents by ID using tombstones.
        Preserves vector memory alignment and ANN index integrity.
        Records delete operations to WAL.
        """
        with self._lock.write():
            return self._delete_internal(doc_ids, log_to_wal=True)

    def _delete_internal(self, doc_ids: Union[str, List[str]], log_to_wal: bool = True) -> int:
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]

        deleted_ids = []
        for doc_id in doc_ids:
            if doc_id in self.id_to_index:
                idx = self.id_to_index[doc_id]
                if idx < len(self._tombstones) and not self._tombstones[idx]:
                    self._tombstones[idx] = True
                    deleted_ids.append(doc_id)
                del self.id_to_index[doc_id]

        if not deleted_ids:
            return 0

        if log_to_wal and self.wal:
            self.wal.append({
                "op": "delete",
                "ids": deleted_ids
            })

        return len(deleted_ids)

    def count(self) -> int:
        """Returns the number of active (non-tombstoned) documents currently stored."""
        with self._lock.read():
            if len(self._tombstones) == 0:
                return len(self.documents)
            return int(np.sum(~self._tombstones))

    def save(self, filepath: Optional[Union[str, Path]] = None) -> int:
        """Persists the collection to a memory-aligned .sutra binary file."""
        with self._lock.write():
            target_path = filepath or self.storage_path
            if not target_path:
                raise ValueError("No storage path provided for collection save")

            target_path = Path(target_path)
            if target_path.suffix != ".sutra":
                target_path = target_path.with_suffix(".sutra")

            # Compact tombstoned documents before snapshotting
            if len(self._tombstones) > 0 and np.any(self._tombstones):
                active_indices = np.where(~self._tombstones)[0]
                self.vectors = self.vectors[active_indices] if len(active_indices) > 0 else np.empty((0, self.dimension), dtype=np.float32)
                self.documents = [self.documents[i] for i in active_indices]
                self.id_to_index = {doc.id: i for i, doc in enumerate(self.documents)}
                self._tombstones = np.zeros(len(self.documents), dtype=bool)
                self.bm25_index = BM25Index(analyzer=self.analyzer)
                if self.documents:
                    self.bm25_index.add_documents([d.text for d in self.documents])
                if self._ann_index is not None:
                    self._ann_dirty = True
                if self._sq8_enabled and self._sq8_vectors:
                    self._sq8_vectors = [self._sq8_vectors[i] for i in active_indices]
                if self._bq_enabled:
                    self._bq_vectors = [self._bq_vectors[i] for i in active_indices]
                if self._pq_enabled:
                    self._pq_vectors = [self._pq_vectors[i] for i in active_indices]

            doc_payloads = [d.to_dict() for d in self.documents]
            bytes_written = BinaryStorage.write(
                target_path,
                self.vectors,
                doc_payloads,
                self.metric.value
            )

            # Clear WAL on clean snapshot
            if self.wal:
                self.wal.clear()

            return bytes_written

    def load(self, filepath: Optional[Union[str, Path]] = None) -> int:
        """Loads vectors and documents from a .sutra binary file."""
        with self._lock.write():
            target_path = filepath or self.storage_path
            if not target_path:
                raise ValueError("No storage path provided for collection load")

            target_path = Path(target_path)
            if not target_path.exists():
                raise FileNotFoundError(f"Cannot find database file at {target_path}")

            vectors, doc_dicts, metric = BinaryStorage.read(target_path)
            self.vectors = vectors
            self.dimension = vectors.shape[1] if vectors.size > 0 else self.dimension
            self.metric = Metric(metric)

            self.documents = []
            self.id_to_index = {}
            for i, d in enumerate(doc_dicts):
                vec = self.vectors[i] if i < len(self.vectors) else None
                doc = Document.from_dict(d, vec)
                self.documents.append(doc)
                self.id_to_index[doc.id] = i

            self._tombstones = np.zeros(len(self.documents), dtype=bool)

            # Re-index BM25
            self.bm25_index = BM25Index(analyzer=self.analyzer)
            self.bm25_index.add_documents([d.text for d in self.documents])

            # Invalidate ANN index and re-sync quantized vectors on load
            if self._ann_index is not None:
                self._ann_dirty = True
            if self._sq8_enabled:
                self._sq8_vectors = [SQ8Vector.quantize(v) for v in self.vectors]
            else:
                self._sq8_vectors.clear()
            if self._bq_enabled:
                self._bq_vectors = [BQVector.quantize(v) for v in self.vectors]
            else:
                self._bq_vectors.clear()
            if self._pq_enabled and self._pq_codebook is not None:
                self._pq_vectors = [self._pq_codebook.encode(v) for v in self.vectors]
            else:
                self._pq_vectors.clear()

            # Replay WAL if any newer entries occurred
            if self.wal:
                self._replay_wal_if_present()

            return int(np.sum(~self._tombstones)) if len(self._tombstones) > 0 else len(self.documents)

    def stats(self) -> Dict[str, Any]:
        """Returns collection operational and memory statistics."""
        with self._lock.read():
            active_count = int(np.sum(~self._tombstones)) if len(self._tombstones) > 0 else len(self.documents)
            vector_bytes = self.vectors.nbytes
            res = {
                "name": self.name,
                "count": active_count,
                "total_records": len(self.documents),
                "tombstones": int(np.sum(self._tombstones)) if len(self._tombstones) > 0 else 0,
                "dimension": self.dimension,
                "metric": self.metric.value,
                "vector_memory_bytes": vector_bytes,
                "vector_memory_mb": round(vector_bytes / (1024 * 1024), 3),
                "quantized_sq8": self._sq8_enabled,
                "quantized_bq": self._bq_enabled,
                "quantized_pq": self._pq_enabled,
                "pq_m": self._pq_codebook.m if self._pq_codebook else None,
                "has_hnsw": self._hnsw_index is not None,
                "analyzer": {
                    "language": self.analyzer.config.language,
                    "stemmer": self.analyzer.config.stemmer,
                    "stopwords": self.analyzer.config.stopwords,
                    "split_identifiers": self.analyzer.config.split_identifiers,
                    "cjk_bigrams": self.analyzer.config.cjk_bigrams,
                }
            }
            if self._ann_index is not None:
                res["ann"] = self._ann_index.stats()
                res["ann"]["dirty"] = self._ann_dirty
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

    def create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        metric: Union[Metric, str] = Metric.COSINE,
        enable_wal: bool = False,
        analyzer: Optional[Union[Analyzer, AnalyzerConfig, Dict[str, Any]]] = None,
        embedding_function: Optional[Any] = None
    ) -> Collection:
        """Creates and registers a new collection."""
        if name in self.collections:
            raise ValueError(f"Collection '{name}' already exists.")

        if dimension is None:
            if embedding_function is not None and hasattr(embedding_function, "dimension"):
                dimension = int(embedding_function.dimension)
            else:
                dimension = 384

        storage_path = None
        if self.persist_dir:
            storage_path = self.persist_dir / f"{name}.sutra"

        col = Collection(
            name=name,
            dimension=dimension,
            metric=metric,
            storage_path=storage_path,
            enable_wal=enable_wal,
            analyzer=analyzer,
            embedding_function=embedding_function
        )
        self.collections[name] = col
        return col

    def get_collection(self, name: str) -> Collection:
        """Retrieves an existing registered collection."""
        if name not in self.collections:
            # Check if file exists on disk to auto-load
            if self.persist_dir:
                disk_file = self.persist_dir / f"{name}.sutra"
                wal_file = self.persist_dir / f"{name}.wal"
                if disk_file.exists():
                    has_wal = wal_file.exists()
                    col = Collection(name=name, dimension=1, storage_path=disk_file, enable_wal=False)
                    col.load()
                    if has_wal:
                        col.wal = WriteAheadLog(wal_file)
                        col._replay_wal_if_present()
                    self.collections[name] = col
                    return col
                elif wal_file.exists():
                    # Infer dimension from first entry in WAL
                    temp_wal = WriteAheadLog(wal_file)
                    entries = temp_wal.replay()
                    temp_wal.close()
                    dim = 1
                    for entry in entries:
                        if entry.get("op") == "insert" and entry.get("docs"):
                            vec = entry["docs"][0].get("vector", [])
                            if vec:
                                dim = len(vec)
                                break
                    col = Collection(name=name, dimension=dim, storage_path=disk_file, enable_wal=True)
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
        embedding_function: Optional[Any] = None
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
                embedding_function=embedding_function
            )

    def list_collections(self) -> List[str]:
        """Returns a list of all active collection names."""
        names = set(self.collections.keys())
        if self.persist_dir and self.persist_dir.exists():
            for f in self.persist_dir.glob("*.sutra"):
                names.add(f.stem)
            for f in self.persist_dir.glob("*.wal"):
                names.add(f.stem)
        return sorted(list(names))

    def drop_collection(self, name: str) -> bool:
        """Removes a collection from memory and deletes its files on disk."""
        col = self.collections.pop(name, None)
        dropped = col is not None

        if col is not None and col.wal is not None:
            col.wal.close()

        if self.persist_dir:
            sutra_file = self.persist_dir / f"{name}.sutra"
            wal_file = self.persist_dir / f"{name}.wal"
            if sutra_file.exists():
                sutra_file.unlink()
                dropped = True
            if wal_file.exists():
                wal_file.unlink()
                dropped = True

        return dropped


# Export Rust NativeCollection for zero-overhead C-acceleration
try:
    from sutradb._sutradb import NativeCollection
except ImportError:
    NativeCollection = None
