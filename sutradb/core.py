"""
Core database abstractions for SutraDB: Document, SearchResult, Collection, and SutraDB manager.
Supports SIMD-accelerated vector search, BM25 lexical search, and compound hybrid filtering.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid
import numpy as np

from sutradb.distance import Metric, compute_distances, normalize_matrix, normalize_vector
from sutradb.filters import FilterEngine
from sutradb.bm25 import BM25Index
from sutradb.fusion import reciprocal_rank_fusion, linear_score_fusion
from sutradb.storage import BinaryStorage, WriteAheadLog


@dataclass
class Document:
    """Represents a single record stored in SutraDB."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vector: Optional[Union[List[float], np.ndarray]] = None
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], vector: Optional[np.ndarray] = None) -> "Document":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            vector=vector,
            text=data.get("text", ""),
            metadata=data.get("metadata", {})
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": round(self.score, 6),
            "text": self.text,
            "metadata": self.metadata,
            "dense_score": round(self.dense_score, 6) if self.dense_score is not None else None,
            "bm25_score": round(self.bm25_score, 6) if self.bm25_score is not None else None,
        }


class Collection:
    """A collection of vector embeddings, lexical text, and structured metadata."""

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: Union[Metric, str] = Metric.COSINE,
        storage_path: Optional[Union[str, Path]] = None,
        enable_wal: bool = False
    ):
        self.name = name
        self.dimension = int(dimension)
        self.metric = Metric(metric) if isinstance(metric, str) else metric
        self.storage_path = Path(storage_path) if storage_path else None

        # Dense vector storage (N x D)
        self.vectors: np.ndarray = np.empty((0, self.dimension), dtype=np.float32)
        
        # Document and metadata stores
        self.documents: List[Document] = []
        self.id_to_index: Dict[str, int] = {}
        
        # BM25 lexical engine
        self.bm25_index = BM25Index()

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
                vectors = [np.array(d["vector"], dtype=np.float32) for d in raw_docs]
                docs = [Document.from_dict(d, v) for d, v in zip(raw_docs, vectors)]
                self._insert_internal(docs, log_to_wal=False)

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
                    metadata=item.get("metadata", {})
                ))
            else:
                raise TypeError(f"Unsupported document type: {type(item)}")

        return self._insert_internal(parsed_docs, log_to_wal=True)

    def _insert_internal(self, documents: List[Document], log_to_wal: bool = True) -> int:
        if not documents:
            return len(self.documents)

        new_vectors = []
        new_texts = []
        new_docs = []

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
                # Rebuilding BM25 when updating documents
                continue

            idx = len(self.documents) + len(new_docs)
            self.id_to_index[doc.id] = idx
            new_vectors.append(vec)
            new_texts.append(doc.text)
            new_docs.append(doc)

        if new_vectors:
            stacked_new = np.vstack(new_vectors)
            if self.vectors.shape[0] == 0:
                self.vectors = stacked_new
            else:
                self.vectors = np.vstack([self.vectors, stacked_new])

            self.documents.extend(new_docs)
            self.bm25_index.add_documents(new_texts)

        if log_to_wal and self.wal and new_docs:
            wal_payload = {
                "op": "insert",
                "docs": [
                    {
                        "id": d.id,
                        "vector": d.vector.tolist() if d.vector is not None else [],
                        "text": d.text,
                        "metadata": d.metadata
                    }
                    for d in new_docs
                ]
            }
            self.wal.append(wal_payload)

        return len(self.documents)

    def query(
        self,
        vector: Optional[Union[List[float], np.ndarray]] = None,
        text: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        hybrid: bool = True,
        rrf_constant: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        alpha: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Executes a vector, lexical, or hybrid search with optional metadata filtering.
        
        Args:
            vector: Query dense vector (dimension D).
            text: Query text string for BM25 lexical matching.
            filter: Compound metadata filter dictionary.
            top_k: Maximum number of results to return.
            hybrid: If True and both vector and text are supplied, combines with RRF.
            rrf_constant: Smoothing factor for Reciprocal Rank Fusion (default 60).
            dense_weight: Importance multiplier for dense score.
            bm25_weight: Importance multiplier for BM25 score.
            alpha: If specified, uses linear convex fusion instead of RRF (alpha * dense + (1-alpha) * bm25).
        """
        n = len(self.documents)
        if n == 0:
            return []

        # 1. Compile in-flight metadata filter mask
        metadata_list = [d.metadata for d in self.documents]
        mask = FilterEngine.build_mask(metadata_list, filter)
        valid_count = int(np.sum(mask))
        if valid_count == 0:
            return []

        # 2. Vector distance scoring
        has_vector = vector is not None
        has_text = bool(text and text.strip())

        dense_scores: Optional[np.ndarray] = None
        if has_vector:
            q_vec = np.asarray(vector, dtype=np.float32)
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

        # 4. Route search execution based on mode
        results: List[SearchResult] = []

        if hybrid and has_vector and has_text and dense_scores is not None and bm25_scores is not None:
            # Hybrid search combining dense + lexical
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
                    bm25_score=b_score
                ))

        elif has_vector and dense_scores is not None:
            # Pure vector search
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
                    bm25_score=None
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
                    bm25_score=float(bm25_scores[doc_idx])
                ))
        else:
            raise ValueError("Query must specify either a vector, a text string, or both for hybrid search.")

        return results

    def get(self, doc_id: str) -> Optional[Document]:
        """Retrieves a single document by its unique ID."""
        idx = self.id_to_index.get(doc_id)
        if idx is None:
            return None
        return self.documents[idx]

    def delete(self, doc_ids: Union[str, List[str]]) -> int:
        """
        Deletes one or more documents by ID.
        Re-indexes collection memory and BM25 index.
        """
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]

        ids_to_remove = set(doc_ids)
        keep_indices = [i for i, d in enumerate(self.documents) if d.id not in ids_to_remove]
        deleted_count = len(self.documents) - len(keep_indices)

        if deleted_count == 0:
            return 0

        # Rebuild contiguous memory structures
        self.vectors = self.vectors[keep_indices]
        self.documents = [self.documents[i] for i in keep_indices]
        self.id_to_index = {doc.id: i for i, doc in enumerate(self.documents)}

        # Rebuild BM25 index
        self.bm25_index = BM25Index()
        self.bm25_index.add_documents([d.text for d in self.documents])

        return deleted_count

    def count(self) -> int:
        """Returns the number of documents currently stored."""
        return len(self.documents)

    def save(self, filepath: Optional[Union[str, Path]] = None) -> int:
        """Persists the collection to a memory-aligned .sutra binary file."""
        target_path = filepath or self.storage_path
        if not target_path:
            raise ValueError("No storage path provided for collection save")

        target_path = Path(target_path)
        if target_path.suffix != ".sutra":
            target_path = target_path.with_suffix(".sutra")

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

        # Re-index BM25
        self.bm25_index = BM25Index()
        self.bm25_index.add_documents([d.text for d in self.documents])

        # Replay WAL if any newer entries occurred
        if self.wal:
            self._replay_wal_if_present()

        return len(self.documents)

    def stats(self) -> Dict[str, Any]:
        """Returns collection operational and memory statistics."""
        vector_bytes = self.vectors.nbytes
        return {
            "name": self.name,
            "count": len(self.documents),
            "dimension": self.dimension,
            "metric": self.metric.value,
            "vector_memory_bytes": vector_bytes,
            "vector_memory_mb": round(vector_bytes / (1024 * 1024), 3)
        }


class SutraDB:
    """Top-level database management interface."""

    def __init__(self, persist_directory: Optional[Union[str, Path]] = None):
        self.persist_dir = Path(persist_directory) if persist_directory else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collections: Dict[str, Collection] = {}

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: Union[Metric, str] = Metric.COSINE,
        enable_wal: bool = False
    ) -> Collection:
        """Creates and registers a new collection."""
        if name in self.collections:
            raise ValueError(f"Collection '{name}' already exists.")

        storage_path = None
        if self.persist_dir:
            storage_path = self.persist_dir / f"{name}.sutra"

        col = Collection(
            name=name,
            dimension=dimension,
            metric=metric,
            storage_path=storage_path,
            enable_wal=enable_wal
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
                    col = Collection(name=name, dimension=1, storage_path=disk_file, enable_wal=wal_file.exists())
                    col.load()
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
        dimension: int,
        metric: Union[Metric, str] = Metric.COSINE,
        enable_wal: bool = False
    ) -> Collection:
        """Retrieves or initializes a collection."""
        try:
            return self.get_collection(name)
        except KeyError:
            return self.create_collection(name, dimension, metric, enable_wal=enable_wal)

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
