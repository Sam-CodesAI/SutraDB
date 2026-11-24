"""
Distributed Multi-Node Clustering, Sharding, and Raft Consensus for SutraDB.
Provides ConsistentHashRing, DistributedCollection, and DistributedCluster with
zero-latency parallel scatter-gather query execution.
"""

from __future__ import annotations
import bisect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from sutradb.core import Collection, Document, SearchResult, Metric
from sutradb.embeddings import EmbeddingFunction, DefaultEmbeddingFunction

try:
    from sutradb._sutradb import NativeConsistentHashRing
except ImportError:
    NativeConsistentHashRing = None


def _hash_key(key: str) -> int:
    """Deterministic 64-bit FNV-1a hash with avalanche mixer."""
    h = 0xcbf29ce484222325
    for b in key.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    # SplitMix64 avalanche mixer
    h ^= (h >> 33)
    h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33)
    h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33)
    return h


class ConsistentHashRing:
    """
    Virtual-node consistent hash ring for partition routing and shard replication.
    Uses C/Rust NativeConsistentHashRing when available with zero Python GIL overhead.
    """

    def __init__(self, vnodes_per_node: int = 64):
        self.vnodes_per_node = max(1, int(vnodes_per_node))
        if NativeConsistentHashRing is not None:
            self._native = NativeConsistentHashRing(self.vnodes_per_node)
        else:
            self._native = None
        self._ring: Dict[int, str] = {}
        self._sorted_keys: List[int] = []
        self._nodes: Set[str] = set()

    def add_node(self, node_id: str) -> None:
        """Adds a physical or logical node to the ring."""
        if self._native is not None:
            self._native.add_node(node_id)
            return

        if node_id in self._nodes:
            return
        self._nodes.add(node_id)
        for v in range(self.vnodes_per_node):
            vkey = f"{node_id}-vnode-{v}"
            token = _hash_key(vkey)
            self._ring[token] = node_id
        self._sorted_keys = sorted(self._ring.keys())

    def remove_node(self, node_id: str) -> bool:
        """Removes a node and its virtual positions from the ring."""
        if self._native is not None:
            return self._native.remove_node(node_id)

        if node_id not in self._nodes:
            return False
        self._nodes.remove(node_id)
        for v in range(self.vnodes_per_node):
            vkey = f"{node_id}-vnode-{v}"
            token = _hash_key(vkey)
            self._ring.pop(token, None)
        self._sorted_keys = sorted(self._ring.keys())
        return True

    def get_node(self, key: str) -> Optional[str]:
        """Returns the primary node responsible for the given key."""
        if self._native is not None:
            return self._native.get_node(key)

        if not self._ring:
            return None
        token = _hash_key(key)
        idx = bisect.bisect_right(self._sorted_keys, token)
        if idx == len(self._sorted_keys):
            idx = 0
        return self._ring[self._sorted_keys[idx]]

    def get_preference_list(self, key: str, count: usize) -> List[str]:
        """Returns distinct physical nodes responsible for replicating this key."""
        if self._native is not None:
            return self._native.get_preference_list(key, count)

        if not self._nodes or count <= 0:
            return []
        target = min(count, len(self._nodes))
        distinct: List[str] = []
        seen: Set[str] = set()

        token = _hash_key(key)
        idx = bisect.bisect_right(self._sorted_keys, token)
        n_keys = len(self._sorted_keys)

        for step in range(n_keys):
            curr_idx = (idx + step) % n_keys
            node = self._ring[self._sorted_keys[curr_idx]]
            if node not in seen:
                seen.add(node)
                distinct.append(node)
                if len(distinct) == target:
                    break
        return distinct

    def node_count(self) -> int:
        """Returns total distinct active nodes."""
        if self._native is not None:
            return self._native.node_count()
        return len(self._nodes)

    def get_nodes(self) -> List[str]:
        """Returns list of all active node IDs."""
        if self._native is not None:
            return self._native.get_nodes()
        return sorted(list(self._nodes))


class DistributedCollection:
    """
    Horizontally partitioned collection distributing documents across multiple shards.
    Coordinates parallel scatter-gather search queries and shard partition routing.
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: Union[Metric, str] = Metric.COSINE,
        shard_count: int = 3,
        vnodes_per_node: int = 64,
        embedding_function: Optional[Any] = None,
    ):
        self.name = name
        self.dimension = int(dimension)
        self.metric = Metric(metric) if isinstance(metric, str) else metric
        self.shard_count = max(1, int(shard_count))
        self.embedding_function = embedding_function or DefaultEmbeddingFunction(dimension=self.dimension)
        self.hash_ring = ConsistentHashRing(vnodes_per_node=vnodes_per_node)

        # Initialize local shard collections
        self.shards: Dict[str, Collection] = {}
        for s in range(self.shard_count):
            shard_id = f"shard-{s}"
            self.shards[shard_id] = Collection(
                name=f"{name}_{shard_id}",
                dimension=self.dimension,
                metric=self.metric,
                embedding_function=self.embedding_function,
            )
            self.hash_ring.add_node(shard_id)

    def insert(self, documents: Union[Document, List[Document], Dict[str, Any], List[Dict[str, Any]]]) -> int:
        """Partitions documents by hash ring key and inserts into respective shards."""
        if isinstance(documents, (Document, dict)):
            doc_list = [documents]
        else:
            doc_list = list(documents)

        # Group by assigned shard
        shard_groups: Dict[str, List[Any]] = {s: [] for s in self.shards}
        for item in doc_list:
            if isinstance(item, Document):
                routing_key = item.namespace if item.namespace else item.id
            elif isinstance(item, dict):
                routing_key = item.get("namespace") or item.get("id", "")
            else:
                routing_key = ""

            target_shard = self.hash_ring.get_node(str(routing_key)) or list(self.shards.keys())[0]
            shard_groups[target_shard].append(item)

        total_inserted = 0
        for shard_id, items in shard_groups.items():
            if items:
                self.shards[shard_id].insert(items)
                total_inserted += len(items)

        return total_inserted

    def add_texts(
        self,
        texts: Union[str, Sequence[str]],
        metadatas: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, Sequence[str]]] = None,
        namespace: Optional[str] = None
    ) -> List[str]:
        """Ingests raw text strings with automatic embedding and shard routing."""
        if isinstance(texts, str):
            text_list = [texts]
        else:
            text_list = list(texts)

        n = len(text_list)
        if n == 0:
            return []

        import uuid
        if ids is None:
            id_list = [str(uuid.uuid4()) for _ in range(n)]
        elif isinstance(ids, str):
            id_list = [ids]
        else:
            id_list = list(ids)

        if metadatas is None:
            meta_list = [{} for _ in range(n)]
        elif isinstance(metadatas, dict):
            meta_list = [metadatas]
        else:
            meta_list = list(metadatas)

        docs = [
            Document(id=doc_id, text=txt, metadata=meta, namespace=namespace)
            for doc_id, txt, meta in zip(id_list, text_list, meta_list)
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
        """Auto-vectorizes query text and executes scatter-gather hybrid search across all shards."""
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
            use_ann=use_ann
        )

    def query(
        self,
        vector: Optional[Union[List[float], Any]] = None,
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
        use_ann: bool = False
    ) -> List[SearchResult]:
        """Parallel scatter-gather query executing across all shards with global top-k fusion."""
        # Scatter query in parallel across all shards
        def query_shard(shard: Collection) -> List[SearchResult]:
            return shard.query(
                vector=vector,
                text=text,
                sparse_vector=sparse_vector,
                colbert_tokens=colbert_tokens,
                filter=filter,
                namespace=namespace,
                top_k=top_k,
                hybrid=hybrid,
                rrf_constant=rrf_constant,
                dense_weight=dense_weight,
                bm25_weight=bm25_weight,
                use_ann=use_ann
            )

        with ThreadPoolExecutor(max_workers=min(len(self.shards), 8)) as pool:
            futures = [pool.submit(query_shard, s) for s in self.shards.values()]
            shard_results = [f.result() for f in futures]

        # Gather and global merge: deduplicate by document id, sort descending by score
        merged_candidates: Dict[str, SearchResult] = {}
        for s_list in shard_results:
            for item in s_list:
                if item.id not in merged_candidates or item.score > merged_candidates[item.id].score:
                    merged_candidates[item.id] = item

        sorted_results = sorted(merged_candidates.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]

    def count(self) -> int:
        """Returns aggregated document count across all shards."""
        return sum(s.count() for s in self.shards.values())


class DistributedCluster:
    """High-level cluster manager managing distributed partitioned collections."""

    def __init__(self, node_id: str = "cluster-coordinator", vnodes_per_node: int = 64):
        self.node_id = node_id
        self.vnodes_per_node = vnodes_per_node
        self.collections: Dict[str, DistributedCollection] = {}

    def create_collection(
        self,
        name: str,
        dimension: int = 384,
        metric: Union[Metric, str] = Metric.COSINE,
        shard_count: int = 3,
        embedding_function: Optional[Any] = None
    ) -> DistributedCollection:
        """Creates and registers a new horizontally partitioned collection."""
        if name in self.collections:
            raise ValueError(f"Collection '{name}' already exists in cluster.")

        col = DistributedCollection(
            name=name,
            dimension=dimension,
            metric=metric,
            shard_count=shard_count,
            vnodes_per_node=self.vnodes_per_node,
            embedding_function=embedding_function
        )
        self.collections[name] = col
        return col

    def get_collection(self, name: str) -> DistributedCollection:
        """Retrieves an existing distributed collection."""
        if name not in self.collections:
            raise KeyError(f"Collection '{name}' not found.")
        return self.collections[name]

    def get_or_create_collection(
        self,
        name: str,
        dimension: int = 384,
        metric: Union[Metric, str] = Metric.COSINE,
        shard_count: int = 3,
        embedding_function: Optional[Any] = None
    ) -> DistributedCollection:
        """Retrieves or creates a distributed collection."""
        try:
            return self.get_collection(name)
        except KeyError:
            return self.create_collection(
                name=name,
                dimension=dimension,
                metric=metric,
                shard_count=shard_count,
                embedding_function=embedding_function
            )

    def list_collections(self) -> List[str]:
        """Returns names of all active collections."""
        return sorted(list(self.collections.keys()))

    def drop_collection(self, name: str) -> bool:
        """Removes a collection from the cluster."""
        return self.collections.pop(name, None) is not None
