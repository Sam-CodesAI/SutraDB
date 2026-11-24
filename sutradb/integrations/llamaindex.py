"""
LlamaIndex official integration for SutraDB.
Provides SutraDBVectorStore conforming to LlamaIndex's vector store contract,
supporting dense embedding search, BM25 sparse keyword queries, and hybrid RRF retrieval.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

from sutradb.core import SutraDB, Collection, Document as SutraDoc


# Graceful LlamaIndex import or lightweight standalone fallbacks
try:
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
        VectorStoreQueryMode,
        MetadataFilters,
    )
    from llama_index.core.schema import BaseNode, TextNode
    HAS_LLAMA_INDEX = True
except ImportError:
    HAS_LLAMA_INDEX = False

    class VectorStoreQueryMode(str, Enum):  # type: ignore[no-redef]
        DEFAULT = "default"
        SPARSE = "sparse"
        HYBRID = "hybrid"

    class BaseNode:  # type: ignore[no-redef]
        def __init__(
            self,
            text: str = "",
            id_: Optional[str] = None,
            embedding: Optional[List[float]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            ref_doc_id: Optional[str] = None,
        ):
            import uuid
            self.id_ = id_ or str(uuid.uuid4())
            self.text = text
            self.embedding = embedding
            self.metadata = metadata or {}
            self.ref_doc_id = ref_doc_id

        def get_content(self, **kwargs: Any) -> str:
            return self.text

        @property
        def node_id(self) -> str:
            return self.id_

    class TextNode(BaseNode):  # type: ignore[no-redef]
        pass

    class VectorStoreQuery:  # type: ignore[no-redef]
        def __init__(
            self,
            query_embedding: Optional[List[float]] = None,
            similarity_top_k: int = 10,
            query_str: Optional[str] = None,
            mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
            filters: Optional[Any] = None,
            alpha: Optional[float] = None,
            namespace: Optional[str] = None,
        ):
            self.query_embedding = query_embedding
            self.similarity_top_k = similarity_top_k
            self.query_str = query_str
            self.mode = mode
            self.filters = filters
            self.alpha = alpha
            self.namespace = namespace

    class VectorStoreQueryResult:  # type: ignore[no-redef]
        def __init__(
            self,
            nodes: Optional[List[BaseNode]] = None,
            similarities: Optional[List[float]] = None,
            ids: Optional[List[str]] = None,
        ):
            self.nodes = nodes or []
            self.similarities = similarities or []
            self.ids = ids or []

    class BasePydanticVectorStore:  # type: ignore[no-redef]
        pass


class SutraDBLlamaIndexVectorStore:
    """
    SutraDB VectorStore for LlamaIndex pipelines.
    Supports standard dense semantic vector search, BM25 sparse keyword queries,
    and reciprocal rank hybrid retrieval for agentic RAG.
    """

    stores_text: bool = True
    is_embedding_query: bool = True

    def __init__(
        self,
        collection_name: str = "llamaindex",
        dimension: int = 1536,
        persist_directory: Optional[str] = None,
        metric: str = "cosine",
        hybrid: bool = True,
        namespace: Optional[str] = None,
        collection: Optional[Collection] = None,
        db: Optional[SutraDB] = None,
    ) -> None:
        self.collection_name = collection_name
        self.dimension = dimension
        self.persist_directory = persist_directory
        self.metric = metric
        self.hybrid = hybrid
        self.default_namespace = namespace

        if collection is not None:
            self.collection = collection
        else:
            self.db = db or SutraDB(persist_directory=persist_directory or "./sutra_data")
            try:
                self.collection = self.db.get_collection(collection_name)
            except KeyError:
                self.collection = self.db.create_collection(
                    name=collection_name,
                    dimension=dimension,
                    metric=metric,
                )

    @property
    def client(self) -> Collection:
        return self.collection

    def add(
        self,
        nodes: Sequence[BaseNode],
        namespace: Optional[str] = None,
        **add_kwargs: Any,
    ) -> List[str]:
        """Inserts LlamaIndex nodes into SutraDB collection."""
        if not nodes:
            return []

        ns = namespace or self.default_namespace
        docs: List[SutraDoc] = []
        node_ids: List[str] = []

        for node in nodes:
            nid = getattr(node, "node_id", getattr(node, "id_", str(id(node))))
            text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
            vector = getattr(node, "embedding", None)
            if vector is None:
                vector = [0.0] * self.collection.dimension

            metadata = dict(getattr(node, "metadata", {}))
            ref_id = getattr(node, "ref_doc_id", None)
            if ref_id is not None:
                metadata["_ref_doc_id"] = ref_id

            docs.append(
                SutraDoc(
                    id=nid,
                    vector=vector,
                    text=text,
                    metadata=metadata,
                    namespace=ns,
                )
            )
            node_ids.append(nid)

        self.collection.upsert(docs)
        return node_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Deletes all nodes associated with a parent reference document ID."""
        matching_ids = [
            doc.id for doc in self.collection.get_all()
            if doc.metadata.get("_ref_doc_id") == ref_doc_id
        ]
        if matching_ids:
            self.collection.delete(matching_ids)

    def delete_nodes(self, node_ids: List[str], **delete_kwargs: Any) -> None:
        """Deletes a list of specific node IDs."""
        if node_ids:
            self.collection.delete(node_ids)

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Executes query against SutraDB collection matching LlamaIndex query specification."""
        mode = getattr(query, "mode", VectorStoreQueryMode.DEFAULT)
        top_k = getattr(query, "similarity_top_k", 10)
        query_embedding = getattr(query, "query_embedding", None)
        query_str = getattr(query, "query_str", None)
        alpha = getattr(query, "alpha", None)
        ns = getattr(query, "namespace", None) or self.default_namespace

        # Convert LlamaIndex filters if present
        filter_dict: Optional[Dict[str, Any]] = None
        raw_filters = getattr(query, "filters", None)
        if raw_filters is not None:
            if isinstance(raw_filters, dict):
                filter_dict = raw_filters
            elif hasattr(raw_filters, "filters"):
                # MetadataFilters object
                filter_dict = {}
                for f in getattr(raw_filters, "filters", []):
                    key = getattr(f, "key", None)
                    val = getattr(f, "value", None)
                    op = getattr(f, "operator", None)
                    if key is not None and val is not None:
                        op_str = str(op).lower() if op else "exact"
                        if "exact" in op_str or "==" in op_str:
                            filter_dict[key] = val
                        elif ">=" in op_str:
                            filter_dict[key] = {"$gte": val}
                        elif ">" in op_str:
                            filter_dict[key] = {"$gt": val}
                        elif "<=" in op_str:
                            filter_dict[key] = {"$lte": val}
                        elif "<" in op_str:
                            filter_dict[key] = {"$lt": val}
                        elif "in" in op_str:
                            filter_dict[key] = {"$in": val}
                        else:
                            filter_dict[key] = val

        # Configure query parameters by mode
        if mode == VectorStoreQueryMode.SPARSE:
            is_hybrid = False
            q_vec = None
            q_text = query_str
        elif mode == VectorStoreQueryMode.HYBRID:
            is_hybrid = True
            q_vec = query_embedding
            q_text = query_str
        else:
            # DEFAULT: dense vector query
            is_hybrid = False
            q_vec = query_embedding
            q_text = None

        results = self.collection.query(
            vector=q_vec,
            text=q_text,
            filter=filter_dict,
            top_k=top_k,
            hybrid=is_hybrid,
            alpha=alpha,
            namespace=ns,
            **kwargs,
        )

        nodes: List[BaseNode] = []
        similarities: List[float] = []
        ids: List[str] = []

        for r in results:
            doc_obj = self.collection.get(r.id)
            meta = r.metadata or {}
            ref_id = meta.get("_ref_doc_id")

            node = TextNode(
                text=r.text or "",
                id_=r.id,
                embedding=doc_obj.vector if doc_obj else None,
                metadata=meta,
                ref_doc_id=ref_id,
            )
            nodes.append(node)
            similarities.append(float(r.score))
            ids.append(r.id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
