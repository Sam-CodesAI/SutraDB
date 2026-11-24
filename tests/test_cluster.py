"""
Tests for SutraDB v5.0 Distributed Multi-Node Cluster, Consistent Hash Ring,
and Raft Consensus horizontal partitioning.
"""

import pytest
import numpy as np

from sutradb import (
    ConsistentHashRing,
    DistributedCluster,
    DistributedCollection,
    Document,
    SearchResult,
    Metric,
    DefaultEmbeddingFunction,
)
try:
    from sutradb._sutradb import NativeConsistentHashRing
except ImportError:
    NativeConsistentHashRing = None


class TestConsistentHashRing:
    """Verifies virtual-node consistent hash ring behavior."""

    def test_basic_node_lifecycle(self):
        ring = ConsistentHashRing(vnodes_per_node=32)
        assert ring.node_count() == 0
        assert ring.get_node("doc_1") is None

        ring.add_node("node-alpha")
        assert ring.node_count() == 1
        assert ring.get_node("doc_1") == "node-alpha"
        assert ring.get_node("doc_2") == "node-alpha"

        ring.add_node("node-beta")
        ring.add_node("node-gamma")
        assert ring.node_count() == 3

        # Deterministic routing
        target = ring.get_node("user_100")
        assert target in ["node-alpha", "node-beta", "node-gamma"]
        assert ring.get_node("user_100") == target

    def test_node_removal_and_redistribution(self):
        ring = ConsistentHashRing(vnodes_per_node=32)
        ring.add_node("node-1")
        ring.add_node("node-2")
        assert ring.node_count() == 2

        removed = ring.remove_node("node-1")
        assert removed is True
        assert ring.node_count() == 1
        assert ring.get_node("any_doc") == "node-2"

        # Removing non-existent node returns False
        assert ring.remove_node("non-existent") is False

    def test_preference_list_replication(self):
        ring = ConsistentHashRing(vnodes_per_node=32)
        ring.add_node("n1")
        ring.add_node("n2")
        ring.add_node("n3")

        pref = ring.get_preference_list("order_99", 2)
        assert len(pref) == 2
        assert pref[0] != pref[1]

        pref_all = ring.get_preference_list("order_99", 5)
        assert len(pref_all) == 3

    def test_hash_ring_uniformity(self):
        ring = ConsistentHashRing(vnodes_per_node=64)
        ring.add_node("node-a")
        ring.add_node("node-b")
        ring.add_node("node-c")

        total = 3000
        counts = {}
        for i in range(total):
            node = ring.get_node(f"key-{i}")
            counts[node] = counts.get(node, 0) + 1

        assert len(counts) == 3
        expected = total / 3
        for node, count in counts.items():
            deviation = abs(count - expected) / expected
            assert deviation < 0.25, f"Node {node} deviation {deviation} exceeds threshold"


class TestDistributedCollectionAndCluster:
    """Verifies multi-shard horizontal partitioning and scatter-gather queries."""

    def test_cluster_collection_creation_and_sharding(self):
        cluster = DistributedCluster(node_id="coordinator-main", vnodes_per_node=32)
        col = cluster.create_collection("articles", dimension=4, shard_count=3)

        assert col.name == "articles"
        assert col.dimension == 4
        assert col.shard_count == 3
        assert len(col.shards) == 3
        assert col.count() == 0

    def test_sharded_insertion_and_count(self):
        cluster = DistributedCluster()
        col = cluster.create_collection("products", dimension=4, shard_count=3)

        docs = [
            Document(id=f"doc-{i}", vector=[1.0, 0.0, 0.0, 0.0], text=f"Text item {i}")
            for i in range(12)
        ]
        inserted = col.insert(docs)
        assert inserted == 12
        assert col.count() == 12

        # Verify documents were distributed across multiple shards
        shard_counts = [s.count() for s in col.shards.values()]
        assert sum(shard_counts) == 12
        # At least 2 shards received documents
        non_empty_shards = [c for c in shard_counts if c > 0]
        assert len(non_empty_shards) >= 2

    def test_add_texts_and_scatter_gather_search(self):
        cluster = DistributedCluster()
        col = cluster.create_collection("wiki", dimension=384, shard_count=3)

        texts = [
            "Rust provides fearless concurrency, memory safety, and zero-cost abstractions.",
            "Python offers extensive machine learning, deep learning, and data science ecosystems.",
            "Distributed systems leverage Raft consensus for fault-tolerant state machine replication.",
            "Photosynthesis in green plants converts atmospheric carbon dioxide into oxygen.",
            "Vector databases use HNSW graphs for approximate nearest neighbor retrieval.",
        ]
        metas = [
            {"topic": "systems", "lang": "rust"},
            {"topic": "ai", "lang": "python"},
            {"topic": "distributed", "lang": "systems"},
            {"topic": "biology", "lang": "nature"},
            {"topic": "database", "lang": "algorithms"},
        ]

        ids = col.add_texts(texts=texts, metadatas=metas)
        assert len(ids) == 5
        assert col.count() == 5

        # Scatter-gather query across all shards
        results = col.search("Raft consensus in distributed systems", top_k=2)
        assert len(results) == 2
        # The distributed systems document should rank high
        matched_topics = [r.metadata.get("topic") for r in results]
        assert "distributed" in matched_topics

    def test_scatter_gather_with_metadata_filter(self):
        cluster = DistributedCluster()
        col = cluster.create_collection("cloud_nodes", dimension=64, shard_count=3)

        docs = [
            Document(id="n1", vector=[0.9, 0.1] + [0.0]*62, text="High CPU node", metadata={"tier": "premium", "cores": 64}),
            Document(id="n2", vector=[0.8, 0.2] + [0.0]*62, text="High RAM node", metadata={"tier": "standard", "cores": 32}),
            Document(id="n3", vector=[0.95, 0.05] + [0.0]*62, text="GPU cluster node", metadata={"tier": "premium", "cores": 128}),
            Document(id="n4", vector=[0.1, 0.9] + [0.0]*62, text="Storage backup gateway", metadata={"tier": "standard", "cores": 8}),
        ]
        col.insert(docs)

        # Filter tier == premium
        results = col.query(
            vector=[1.0, 0.0] + [0.0]*62,
            filter={"tier": "premium"},
            top_k=5
        )
        assert len(results) == 2
        for r in results:
            assert r.metadata["tier"] == "premium"

    def test_cluster_lifecycle_operations(self):
        cluster = DistributedCluster()
        col1 = cluster.create_collection("col_a", dimension=4)
        col2 = cluster.get_or_create_collection("col_b", dimension=4)
        col3 = cluster.get_or_create_collection("col_a")

        assert col3 is col1
        assert cluster.list_collections() == ["col_a", "col_b"]

        dropped = cluster.drop_collection("col_a")
        assert dropped is True
        assert cluster.list_collections() == ["col_b"]
        assert cluster.drop_collection("nonexistent") is False
