use _sutradb::cluster::hash_ring::ConsistentHashRing;
use _sutradb::cluster::raft::{ClusterCommand, MemoryRaftCluster, RaftRole};
use _sutradb::cluster::coordinator::ClusterCoordinator;
use _sutradb::collection::{NativeCollection, NativeDocument};
use _sutradb::distance::Metric;

#[test]
fn test_distributed_hash_ring_preference_and_failover() {
    let mut ring = ConsistentHashRing::new(64);
    ring.add_node("shard-0");
    ring.add_node("shard-1");
    ring.add_node("shard-2");

    assert_eq!(ring.node_count(), 3);

    // Replicate across 2 distinct nodes
    let pref = ring.get_preference_list("user_doc_12345", 2);
    assert_eq!(pref.len(), 2);
    assert_ne!(pref[0], pref[1]);

    // Primary shard failover
    let primary = ring.get_node("user_doc_12345").unwrap().to_string();
    assert!(ring.remove_node(&primary));
    assert_eq!(ring.node_count(), 2);

    let new_primary = ring.get_node("user_doc_12345").unwrap();
    assert_ne!(new_primary, &primary);
}

#[test]
fn test_raft_3_node_election_replication_failover() {
    let cluster = MemoryRaftCluster::new(&["raft-node-1", "raft-node-2", "raft-node-3"]);

    // 1. Initial Leader Election
    assert!(cluster.elect_leader("raft-node-1"));
    {
        let leader = cluster.nodes.get("raft-node-1").unwrap().lock().unwrap();
        assert_eq!(leader.role, RaftRole::Leader);
        assert_eq!(leader.current_term, 1);
    }

    // 2. Propose CreateCollection via consensus
    let create_cmd = ClusterCommand::CreateCollection {
        name: "events".to_string(),
        dimension: 4,
        metric: "cosine".to_string(),
    };
    let commit_idx1 = cluster.propose_and_commit("raft-node-1", create_cmd).unwrap();
    assert_eq!(commit_idx1, 1);

    // 3. Propose Insert via consensus
    let docs = vec![
        NativeDocument {
            id: "evt-1".to_string(),
            namespace: Some("tenant-prod".to_string()),
            text: "High load database query timeout alert".to_string(),
            metadata: serde_json::json!({"level": "critical", "code": 500}),
            sparse_vector: None,
            colbert_tokens: None,
        }
    ];
    let vecs = vec![vec![0.9, 0.1, 0.0, 0.0]];
    let insert_cmd = ClusterCommand::Insert {
        collection: "events".to_string(),
        documents: docs,
        vectors: vecs,
    };
    let commit_idx2 = cluster.propose_and_commit("raft-node-1", insert_cmd).unwrap();
    assert_eq!(commit_idx2, 2);

    // Verify all 3 nodes have committed and applied the documents
    for id in &["raft-node-1", "raft-node-2", "raft-node-3"] {
        let node = cluster.nodes.get(*id).unwrap().lock().unwrap();
        assert_eq!(node.commit_index, 2);
        assert_eq!(node.last_applied, 2);
        assert!(node.collections.contains_key("events"));
        assert_eq!(node.collections.get("events").unwrap().count(), 1);
    }

    // 4. Leader Failover: kill raft-node-1
    {
        let mut old_leader = cluster.nodes.get("raft-node-1").unwrap().lock().unwrap();
        old_leader.role = RaftRole::Follower; // Node-1 disconnected
    }

    // Node-2 detects timeout, starts election and wins with Node-3's vote
    assert!(cluster.elect_leader("raft-node-2"));
    {
        let new_leader = cluster.nodes.get("raft-node-2").unwrap().lock().unwrap();
        assert_eq!(new_leader.role, RaftRole::Leader);
        assert_eq!(new_leader.current_term, 2);
    }

    // 5. New leader processes writes and replicates to Node-3
    let new_doc = vec![
        NativeDocument {
            id: "evt-2".to_string(),
            namespace: Some("tenant-prod".to_string()),
            text: "Auto-remediation script executed successfully".to_string(),
            metadata: serde_json::json!({"level": "info", "code": 200}),
            sparse_vector: None,
            colbert_tokens: None,
        }
    ];
    let new_vec = vec![vec![0.1, 0.9, 0.0, 0.0]];
    let insert_cmd2 = ClusterCommand::Insert {
        collection: "events".to_string(),
        documents: new_doc,
        vectors: new_vec,
    };
    let commit_idx3 = cluster.propose_and_commit("raft-node-2", insert_cmd2).unwrap();
    assert_eq!(commit_idx3, 3);

    // Verify Node-2 and Node-3 have 2 documents
    for id in &["raft-node-2", "raft-node-3"] {
        let node = cluster.nodes.get(*id).unwrap().lock().unwrap();
        assert_eq!(node.commit_index, 3);
        assert_eq!(node.collections.get("events").unwrap().count(), 2);
    }
}

#[test]
fn test_distributed_scatter_gather_query_execution() {
    let coordinator = ClusterCoordinator::new("coord-us-east", 64);

    // Setup 4 shards
    for i in 0..4 {
        let shard_id = format!("shard-{}", i);
        let col = NativeCollection::new("telemetry".to_string(), 4, Metric::Cosine, None, false).unwrap();
        coordinator.register_shard(&shard_id, col);
    }

    let mut docs = Vec::new();
    let mut vecs = Vec::new();

    for i in 0..20 {
        docs.push(NativeDocument {
            id: format!("metric-{}", i),
            namespace: Some(format!("tenant-{}", i % 3)),
            text: format!("Server metric observation #{} CPU load average", i),
            metadata: serde_json::json!({"server_id": i, "val": i * 10}),
            sparse_vector: None,
            colbert_tokens: None,
        });
        vecs.push(vec![
            (i as f32) * 0.05,
            1.0 - (i as f32) * 0.05,
            0.0,
            0.0,
        ]);
    }

    let inserted = coordinator.insert_partitioned("telemetry", docs, vecs).unwrap();
    assert_eq!(inserted, 20);
    assert_eq!(coordinator.cluster_count("telemetry"), 20);

    // Parallel scatter-gather search across all 4 shards
    let query_vec = [0.5, 0.5, 0.0, 0.0];
    let top_results = coordinator.scatter_gather_query(
        "telemetry",
        Some(&query_vec),
        Some("CPU load average"),
        None,
        None,
        None,
        None,
        5,
        true,
    ).unwrap();

    assert_eq!(top_results.len(), 5);
    for r in &top_results {
        assert!(r.text.contains("CPU load average"));
        assert!(r.score > 0.0);
    }

    // Tenant-isolated scatter-gather query (tenant-1 only)
    let tenant_results = coordinator.scatter_gather_query(
        "telemetry",
        Some(&query_vec),
        None,
        None,
        None,
        Some("tenant-1"),
        None,
        10,
        false,
    ).unwrap();

    assert!(!tenant_results.is_empty());
    for r in &tenant_results {
        assert_eq!(r.namespace.as_deref(), Some("tenant-1"));
    }
}
