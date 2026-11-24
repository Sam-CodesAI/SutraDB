use std::collections::{BTreeMap, HashSet};

/// Deterministic 64-bit FNV-1a hash function.
/// Guarantees platform-independent consistent token allocation.
#[inline]
pub fn hash_key(key: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &byte in key.as_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    // High-entropy 64-bit avalanche mixer
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

/// Consistent Hash Ring with virtual nodes (vnodes) for uniform partition distribution.
#[derive(Debug, Clone)]
pub struct ConsistentHashRing {
    vnodes_per_node: usize,
    ring: BTreeMap<u64, String>,
    nodes: HashSet<String>,
}

impl ConsistentHashRing {
    /// Initializes a consistent hash ring with the specified number of virtual nodes per physical node.
    pub fn new(vnodes_per_node: usize) -> Self {
        Self {
            vnodes_per_node: vnodes_per_node.max(1),
            ring: BTreeMap::new(),
            nodes: HashSet::new(),
        }
    }

    /// Adds a physical node to the ring, allocating virtual node positions.
    pub fn add_node(&mut self, node_id: &str) {
        if self.nodes.contains(node_id) {
            return;
        }
        self.nodes.insert(node_id.to_string());
        for v in 0..self.vnodes_per_node {
            let vnode_name = format!("{}-vnode-{}", node_id, v);
            let token = hash_key(&vnode_name);
            self.ring.insert(token, node_id.to_string());
        }
    }

    /// Removes a node and all of its virtual nodes from the ring.
    pub fn remove_node(&mut self, node_id: &str) -> bool {
        if !self.nodes.remove(node_id) {
            return false;
        }
        for v in 0..self.vnodes_per_node {
            let vnode_name = format!("{}-vnode-{}", node_id, v);
            let token = hash_key(&vnode_name);
            self.ring.remove(&token);
        }
        true
    }

    /// Returns the node responsible for the given key.
    pub fn get_node(&self, key: &str) -> Option<&str> {
        if self.ring.is_empty() {
            return None;
        }
        let token = hash_key(key);
        // Find the first virtual node whose token >= token
        if let Some((_, node)) = self.ring.range(token..).next() {
            Some(node.as_str())
        } else {
            // Wrap-around to the first node in the ring
            self.ring.iter().next().map(|(_, node)| node.as_str())
        }
    }

    /// Returns a list of distinct physical nodes responsible for replicating this key.
    pub fn get_preference_list(&self, key: &str, count: usize) -> Vec<&str> {
        if self.nodes.is_empty() || count == 0 {
            return Vec::new();
        }
        let target_count = count.min(self.nodes.len());
        let mut distinct = Vec::with_capacity(target_count);
        let mut seen = HashSet::with_capacity(target_count);

        let token = hash_key(key);

        // Iterate starting from token to end of ring, then wrap around
        let ring_tail = self.ring.range(token..);
        let ring_head = self.ring.range(..token);

        for (_, node) in ring_tail.chain(ring_head) {
            if seen.insert(node.as_str()) {
                distinct.push(node.as_str());
                if distinct.len() == target_count {
                    break;
                }
            }
        }

        distinct
    }

    /// Returns the total number of distinct active nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns a copy of the registered node IDs.
    pub fn get_nodes(&self) -> Vec<String> {
        let mut n: Vec<String> = self.nodes.iter().cloned().collect();
        n.sort();
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_consistent_hash_ring_basic() {
        let mut ring = ConsistentHashRing::new(64);
        assert_eq!(ring.get_node("doc_1"), None);

        ring.add_node("node-1");
        assert_eq!(ring.get_node("doc_1"), Some("node-1"));
        assert_eq!(ring.get_node("doc_2"), Some("node-1"));

        ring.add_node("node-2");
        ring.add_node("node-3");
        assert_eq!(ring.node_count(), 3);

        let n1 = ring.get_node("doc_abc");
        assert!(n1.is_some());
        let assigned = n1.unwrap();
        assert!(assigned == "node-1" || assigned == "node-2" || assigned == "node-3");

        // Deterministic repeat lookup
        assert_eq!(ring.get_node("doc_abc"), Some(assigned));
    }

    #[test]
    fn test_hash_ring_distribution_uniformity() {
        let mut ring = ConsistentHashRing::new(128);
        ring.add_node("node-a");
        ring.add_node("node-b");
        ring.add_node("node-c");

        let total_keys = 6000;
        let mut counts: HashMap<&str, usize> = HashMap::new();

        for i in 0..total_keys {
            let key = format!("document-uuid-{}", i);
            let node = ring.get_node(&key).unwrap();
            *counts.entry(node).or_insert(0) += 1;
        }

        assert_eq!(counts.len(), 3);
        let expected = total_keys / 3;
        for (node, count) in &counts {
            let delta = (*count as f64 - expected as f64).abs();
            let deviation = delta / expected as f64;
            // Standard deviation within 15%
            assert!(
                deviation < 0.15,
                "Node {} has count {}, deviation {} exceeds 15%",
                node,
                count,
                deviation
            );
        }
    }

    #[test]
    fn test_preference_list_replication() {
        let mut ring = ConsistentHashRing::new(64);
        ring.add_node("node-1");
        ring.add_node("node-2");
        ring.add_node("node-3");

        let pref = ring.get_preference_list("test_key", 2);
        assert_eq!(pref.len(), 2);
        assert_ne!(pref[0], pref[1]);

        let pref_all = ring.get_preference_list("test_key", 5);
        assert_eq!(pref_all.len(), 3); // Capped by total node count
    }

    #[test]
    fn test_node_removal() {
        let mut ring = ConsistentHashRing::new(64);
        ring.add_node("node-1");
        ring.add_node("node-2");
        assert_eq!(ring.node_count(), 2);

        assert!(ring.remove_node("node-1"));
        assert_eq!(ring.node_count(), 1);
        assert_eq!(ring.get_node("any_key"), Some("node-2"));

        assert!(!ring.remove_node("node-nonexistent"));
    }
}
