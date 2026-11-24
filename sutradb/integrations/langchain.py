"""
LangChain official integration for SutraDB.
Provides SutraDBVectorStore conforming to LangChain's VectorStore standard,
supporting hybrid dense + BM25 search, MMR diversity reranking, and multi-tenant namespaces.
"""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

from sutradb.core import SutraDB, Collection, Document as SutraDoc


# Graceful LangChain import or lightweight fallback
try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError:
    class Document:  # type: ignore[no-redef]
        """Minimal fallback Document matching langchain_core.documents.Document."""
        def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None, id: Optional[str] = None):
            self.page_content = page_content
            self.metadata = metadata or {}
            self.id = id

        def __repr__(self) -> str:
            return f"Document(id='{self.id}', page_content='{self.page_content[:30]}...', metadata={self.metadata})"

    class Embeddings:  # type: ignore[no-redef]
        """Minimal fallback Embeddings interface."""
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            raise NotImplementedError

        def embed_query(self, text: str) -> List[float]:
            raise NotImplementedError

    class VectorStore:  # type: ignore[no-redef]
        """Minimal fallback VectorStore interface."""
        pass


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Calculates cosine similarity between two 1D sequences."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _maximal_marginal_relevance(
    query_embedding: List[float],
    embedding_list: List[List[float]],
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculates Maximal Marginal Relevance (MMR) indices."""
    if min(k, len(embedding_list)) <= 0:
        return []

    # Pre-calculate similarity of each candidate to the query
    query_sims = [_cosine_similarity(query_embedding, emb) for emb in embedding_list]

    selected_indices: List[int] = []
    remaining_indices = list(range(len(embedding_list)))

    # Select the candidate with highest query similarity first
    first_idx = max(remaining_indices, key=lambda i: query_sims[i])
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    while len(selected_indices) < min(k, len(embedding_list)) and remaining_indices:
        best_score = -float("inf")
        best_idx = remaining_indices[0]

        for idx in remaining_indices:
            # Query relevance
            rel = query_sims[idx]
            # Redundancy penalty: max similarity to already selected candidates
            max_redundancy = max(_cosine_similarity(embedding_list[idx], embedding_list[s]) for s in selected_indices)
            mmr_score = lambda_mult * rel - (1.0 - lambda_mult) * max_redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return selected_indices


class SutraDBVectorStore(VectorStore):
    """
    SutraDB VectorStore provider for LangChain.
    Combines sub-0.2ms native Rust vector search, Robertson-Spärck Jones BM25 lexical ranking,
    and compound metadata filtering into a high-performance vector store.
    """

    def __init__(
        self,
        collection_name: str = "langchain",
        dimension: Optional[int] = None,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        metric: str = "cosine",
        hybrid: bool = True,
        namespace: Optional[str] = None,
        collection: Optional[Collection] = None,
        db: Optional[SutraDB] = None,
    ) -> None:
        self._embedding_function = embedding_function
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.metric = metric
        self.hybrid = hybrid
        self.default_namespace = namespace

        if collection is not None:
            self.collection = collection
        else:
            self.db = db or SutraDB(persist_directory=persist_directory or "./sutra_data")
            effective_dim = dimension or (len(embedding_function.embed_query("test")) if embedding_function else 1536)
            try:
                self.collection = self.db.get_collection(collection_name)
            except KeyError:
                self.collection = self.db.create_collection(
                    name=collection_name,
                    dimension=effective_dim,
                    metric=metric,
                )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding_function

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Runs embeddings over texts and inserts them into SutraDB."""
        text_list = list(texts)
        if not text_list:
            return []

        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(text_list))]

        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(text_list)
        else:
            # Fallback to zero vectors if no embedding function is provided
            dim = self.collection.dimension
            embeddings = [[0.0] * dim for _ in range(len(text_list))]

        ns = namespace or self.default_namespace
        docs: List[SutraDoc] = []
        for i, text in enumerate(text_list):
            meta = (metadatas[i] if metadatas and i < len(metadatas) else {}) or {}
            docs.append(
                SutraDoc(
                    id=ids[i],
                    vector=embeddings[i],
                    text=text,
                    metadata=meta,
                    namespace=ns,
                )
            )

        self.collection.upsert(docs)
        return ids

    def add_documents(
        self,
        documents: List[Document],
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Adds a list of LangChain Document objects into SutraDB."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.id for doc in documents if doc.id is not None]
        if len(ids) != len(documents):
            ids = None  # let add_texts generate UUIDs
        return self.add_texts(texts=texts, metadatas=metadatas, ids=ids, namespace=namespace, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Searches documents using query text with dense embeddings and/or BM25."""
        query_vector: Optional[List[float]] = None
        if self._embedding_function is not None:
            query_vector = self._embedding_function.embed_query(query)

        ns = namespace or self.default_namespace
        results = self.collection.query(
            vector=query_vector,
            text=query,
            filter=filter,
            top_k=k,
            hybrid=self.hybrid,
            namespace=ns,
            **kwargs,
        )

        output: List[Tuple[Document, float]] = []
        for r in results:
            doc = Document(
                page_content=r.text or "",
                metadata=r.metadata or {},
                id=r.id,
            )
            output.append((doc, float(r.score)))
        return output

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Searches documents returning Document objects."""
        results = self.similarity_search_with_score(query=query, k=k, filter=filter, namespace=namespace, **kwargs)
        return [doc for doc, _ in results]

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Searches documents directly using a query embedding vector."""
        ns = namespace or self.default_namespace
        results = self.collection.query(
            vector=embedding,
            filter=filter,
            top_k=k,
            hybrid=False,
            namespace=ns,
            **kwargs,
        )

        output: List[Tuple[Document, float]] = []
        for r in results:
            doc = Document(
                page_content=r.text or "",
                metadata=r.metadata or {},
                id=r.id,
            )
            output.append((doc, float(r.score)))
        return output

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Searches documents by vector returning Document objects."""
        results = self.similarity_search_by_vector_with_score(embedding=embedding, k=k, filter=filter, namespace=namespace, **kwargs)
        return [doc for doc, _ in results]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Executes Maximal Marginal Relevance (MMR) retrieval to optimize diversity
        and eliminate duplicate/redundant chunks from the final result set.
        """
        if self._embedding_function is None:
            raise ValueError("max_marginal_relevance_search requires an embedding_function")

        query_embedding = self._embedding_function.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding=query_embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            namespace=namespace,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Executes MMR retrieval using precomputed query vector."""
        ns = namespace or self.default_namespace
        results = self.collection.query(
            vector=embedding,
            filter=filter,
            top_k=fetch_k,
            hybrid=False,
            namespace=ns,
            **kwargs,
        )
        if not results:
            return []

        # Extract embeddings for candidate documents
        candidates: List[Document] = []
        candidate_embeddings: List[List[float]] = []

        for r in results:
            doc_obj = self.collection.get(r.id)
            if doc_obj is not None:
                candidates.append(
                    Document(
                        page_content=doc_obj.text or "",
                        metadata=doc_obj.metadata or {},
                        id=doc_obj.id,
                    )
                )
                candidate_embeddings.append(doc_obj.vector)

        if not candidates:
            return []

        selected_indices = _maximal_marginal_relevance(
            query_embedding=embedding,
            embedding_list=candidate_embeddings,
            lambda_mult=lambda_mult,
            k=k,
        )
        return [candidates[i] for i in selected_indices]

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Deletes documents by ID list."""
        if not ids:
            return False
        deleted = self.collection.delete(ids)
        return deleted > 0

    @classmethod
    def from_texts(
        cls: Type[SutraDBVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = "langchain",
        persist_directory: Optional[str] = None,
        metric: str = "cosine",
        hybrid: bool = True,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> SutraDBVectorStore:
        """Classmethod factory creating a SutraDBVectorStore directly from text strings."""
        sample_vec = embedding.embed_query("init")
        dimension = len(sample_vec)

        store = cls(
            collection_name=collection_name,
            dimension=dimension,
            embedding_function=embedding,
            persist_directory=persist_directory,
            metric=metric,
            hybrid=hybrid,
            namespace=namespace,
            **kwargs,
        )
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids, namespace=namespace)
        return store

    @classmethod
    def from_documents(
        cls: Type[SutraDBVectorStore],
        documents: List[Document],
        embedding: Embeddings,
        collection_name: str = "langchain",
        persist_directory: Optional[str] = None,
        metric: str = "cosine",
        hybrid: bool = True,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> SutraDBVectorStore:
        """Classmethod factory creating a SutraDBVectorStore directly from Document objects."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.id for doc in documents if doc.id is not None]
        if len(ids) != len(documents):
            ids = None
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
            metric=metric,
            hybrid=hybrid,
            namespace=namespace,
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> Any:
        """Returns a LangChain-compatible retriever."""
        try:
            return super().as_retriever(**kwargs)
        except Exception:
            class SimpleRetriever:
                def __init__(self, store: SutraDBVectorStore, kwargs: Dict[str, Any]):
                    self.store = store
                    self.kwargs = kwargs

                def invoke(self, query: str) -> List[Document]:
                    k = self.kwargs.get("k", 4)
                    return self.store.similarity_search(query, k=k)

                def get_relevant_documents(self, query: str) -> List[Document]:
                    return self.invoke(query)

            return SimpleRetriever(self, kwargs)
