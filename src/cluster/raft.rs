use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};

use crate::collection::{NativeCollection, NativeDocument};
use crate::distance::Metric;
use crate::error::SutraError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RaftRole {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ClusterCommand {
    Insert {
        collection: String,
        documents: Vec<NativeDocument>,
        vectors: Vec<Vec<f32>>,
    },
    Delete {
        collection: String,
        ids: Vec<String>,
    },
    CreateCollection {
        name: String,
        dimension: usize,
        metric: String,
    },
    DropCollection {
        name: String,
    },
    Noop,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogEntry {
    pub index: u64,
    pub term: u64,
    pub command: ClusterCommand,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteArgs {
    pub term: u64,
    pub candidate_id: String,
    pub last_log_index: u64,
    pub last_log_term: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteReply {
    pub term: u64,
    pub vote_granted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesArgs {
    pub term: u64,
    pub leader_id: String,
    pub prev_log_index: u64,
    pub prev_log_term: u64,
    pub entries: Vec<LogEntry>,
    pub leader_commit: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesReply {
    pub term: u64,
    pub success: bool,
    pub match_index: u64,
}

/// The core Raft state machine handling terms, elections, logs, and commits.
pub struct RaftNode {
    pub node_id: String,
    pub peers: Vec<String>,
    
    // Persistent state
    pub current_term: u64,
    pub voted_for: Option<String>,
    pub log: Vec<LogEntry>,

    // Volatile state
    pub commit_index: u64,
    pub last_applied: u64,
    pub role: RaftRole,
    pub leader_id: Option<String>,

    // Leader-specific volatile state
    pub next_index: HashMap<String, u64>,
    pub match_index: HashMap<String, u64>,

    // State machine collections
    pub collections: HashMap<String, NativeCollection>,
}

impl RaftNode {
    pub fn new(node_id: &str, peers: Vec<String>) -> Self {
        Self {
            node_id: node_id.to_string(),
            peers,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            role: RaftRole::Follower,
            leader_id: None,
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            collections: HashMap::new(),
        }
    }

    /// Quorum size: floor(N / 2) + 1
    pub fn quorum_size(&self) -> usize {
        self.peers.len().div_ceil(2) + 1
    }

    pub fn last_log_index(&self) -> u64 {
        self.log.last().map(|e| e.index).unwrap_or(0)
    }

    pub fn last_log_term(&self) -> u64 {
        self.log.last().map(|e| e.term).unwrap_or(0)
    }

    /// Transitions to Candidate and initiates an election term.
    pub fn start_election(&mut self) -> RequestVoteArgs {
        self.current_term += 1;
        self.role = RaftRole::Candidate;
        self.voted_for = Some(self.node_id.clone());
        self.leader_id = None;

        RequestVoteArgs {
            term: self.current_term,
            candidate_id: self.node_id.clone(),
            last_log_index: self.last_log_index(),
            last_log_term: self.last_log_term(),
        }
    }

    /// Transitions to Leader upon winning election quorum.
    pub fn become_leader(&mut self) {
        self.role = RaftRole::Leader;
        self.leader_id = Some(self.node_id.clone());
        let last_idx = self.last_log_index();
        for peer in &self.peers {
            self.next_index.insert(peer.clone(), last_idx + 1);
            self.match_index.insert(peer.clone(), 0);
        }
    }

    /// Transitions to Follower with updated term.
    pub fn become_follower(&mut self, term: u64, leader_id: Option<String>) {
        self.current_term = term;
        self.role = RaftRole::Follower;
        self.voted_for = None;
        self.leader_id = leader_id;
    }

    /// Handles incoming RequestVote RPC.
    pub fn handle_request_vote(&mut self, args: &RequestVoteArgs) -> RequestVoteReply {
        // 1. Reply false if term < current_term
        if args.term < self.current_term {
            return RequestVoteReply {
                term: self.current_term,
                vote_granted: false,
            };
        }

        // 2. If term > current_term, step down to follower
        if args.term > self.current_term {
            self.become_follower(args.term, None);
        }

        // 3. Check vote eligibility and log up-to-dateness
        let can_vote = match &self.voted_for {
            None => true,
            Some(cand) => cand == &args.candidate_id,
        };

        let my_last_index = self.last_log_index();
        let my_last_term = self.last_log_term();
        let log_ok = if args.last_log_term != my_last_term {
            args.last_log_term > my_last_term
        } else {
            args.last_log_index >= my_last_index
        };

        if can_vote && log_ok {
            self.voted_for = Some(args.candidate_id.clone());
            RequestVoteReply {
                term: self.current_term,
                vote_granted: true,
            }
        } else {
            RequestVoteReply {
                term: self.current_term,
                vote_granted: false,
            }
        }
    }

    /// Handles incoming AppendEntries RPC (replication & heartbeats).
    pub fn handle_append_entries(&mut self, args: &AppendEntriesArgs) -> AppendEntriesReply {
        // 1. Reply false if term < current_term
        if args.term < self.current_term {
            return AppendEntriesReply {
                term: self.current_term,
                success: false,
                match_index: self.last_log_index(),
            };
        }

        // 2. Update term and recognize leader
        if args.term > self.current_term {
            // Higher term: step down and reset vote
            self.become_follower(args.term, Some(args.leader_id.clone()));
        } else if self.role == RaftRole::Candidate {
            // Same-term AppendEntries from a leader means someone else won;
            // step down but preserve voted_for (Raft spec §5.2)
            self.role = RaftRole::Follower;
            self.leader_id = Some(args.leader_id.clone());
        } else {
            // Same term, already follower: just update leader_id
            self.leader_id = Some(args.leader_id.clone());
        }

        // 3. Log matching property: reply false if log doesn't contain an entry at prev_log_index matching prev_log_term
        if args.prev_log_index > 0 {
            if self.log.len() < args.prev_log_index as usize {
                return AppendEntriesReply {
                    term: self.current_term,
                    success: false,
                    match_index: self.last_log_index(),
                };
            }
            let entry = &self.log[(args.prev_log_index - 1) as usize];
            if entry.term != args.prev_log_term {
                return AppendEntriesReply {
                    term: self.current_term,
                    success: false,
                    match_index: self.last_log_index(),
                };
            }
        }

        // 4. Overwrite conflicting entries and append new entries
        for entry in &args.entries {
            let idx = entry.index as usize;
            if idx <= self.log.len() {
                if self.log[idx - 1].term != entry.term {
                    self.log.truncate(idx - 1);
                    self.log.push(entry.clone());
                }
            } else {
                self.log.push(entry.clone());
            }
        }

        // 5. Update commit_index and apply to state machine
        if args.leader_commit > self.commit_index {
            self.commit_index = args.leader_commit.min(self.last_log_index());
            self.apply_committed();
        }

        AppendEntriesReply {
            term: self.current_term,
            success: true,
            match_index: self.last_log_index(),
        }
    }

    /// Client proposal: appends a command to leader's log and builds AppendEntries args for peers.
    pub fn propose(&mut self, command: ClusterCommand) -> Result<(u64, Vec<(String, AppendEntriesArgs)>), SutraError> {
        if self.role != RaftRole::Leader {
            return Err(SutraError::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("Not leader. Current leader: {:?}", self.leader_id),
            )));
        }

        let new_index = self.last_log_index() + 1;
        let entry = LogEntry {
            index: new_index,
            term: self.current_term,
            command,
        };
        self.log.push(entry);

        let mut peer_msgs = Vec::new();
        for peer in &self.peers {
            let next_idx = *self.next_index.get(peer).unwrap_or(&1);
            let prev_idx = next_idx - 1;
            let prev_term = if prev_idx > 0 && prev_idx <= self.log.len() as u64 {
                self.log[(prev_idx - 1) as usize].term
            } else {
                0
            };

            let entries = if next_idx <= self.log.len() as u64 {
                self.log[(next_idx - 1) as usize..].to_vec()
            } else {
                Vec::new()
            };

            peer_msgs.push((peer.clone(), AppendEntriesArgs {
                term: self.current_term,
                leader_id: self.node_id.clone(),
                prev_log_index: prev_idx,
                prev_log_term: prev_term,
                entries,
                leader_commit: self.commit_index,
            }));
        }

        // Single-node cluster: no peers to replicate to, so auto-commit immediately
        if self.peers.is_empty() {
            self.commit_index = new_index;
            self.apply_committed();
        }

        Ok((new_index, peer_msgs))
    }

    /// Processes peer AppendEntries reply on Leader, advancing match_index and commit_index.
    pub fn handle_append_reply(&mut self, peer: &str, reply: &AppendEntriesReply) -> bool {
        if self.role != RaftRole::Leader {
            return false;
        }

        if reply.term > self.current_term {
            self.become_follower(reply.term, None);
            return false;
        }

        if reply.success {
            self.next_index.insert(peer.to_string(), reply.match_index + 1);
            self.match_index.insert(peer.to_string(), reply.match_index);

            // Check if there is an N > commit_index such that a majority of match_index[i] >= N
            let mut matches: Vec<u64> = self.match_index.values().cloned().collect();
            matches.push(self.last_log_index()); // Leader's own match index
            matches.sort();

            let quorum_match = matches[matches.len() - self.quorum_size()];
            if quorum_match > self.commit_index {
                let entry_term = self.log[(quorum_match - 1) as usize].term;
                if entry_term == self.current_term {
                    self.commit_index = quorum_match;
                    self.apply_committed();
                    return true;
                }
            }
        } else {
            // Decrement next_index and retry
            let next = self.next_index.entry(peer.to_string()).or_insert(1);
            if *next > 1 {
                *next -= 1;
            }
        }

        false
    }

    /// Applies all committed, unapplied entries to the local collection engine.
    pub fn apply_committed(&mut self) -> usize {
        let mut applied_count = 0;
        while self.last_applied < self.commit_index {
            self.last_applied += 1;
            let cmd = self.log[(self.last_applied - 1) as usize].command.clone();
            self.apply_command(&cmd);
            applied_count += 1;
        }
        applied_count
    }

    /// Executes command against local collections.
    fn apply_command(&mut self, command: &ClusterCommand) {
        match command {
            ClusterCommand::CreateCollection { name, dimension, metric } => {
                let m = metric.parse::<Metric>().unwrap_or(Metric::Cosine);
                if let Ok(col) = NativeCollection::new(name.clone(), *dimension, m, None, false) {
                    self.collections.insert(name.clone(), col);
                }
            }
            ClusterCommand::DropCollection { name } => {
                self.collections.remove(name);
            }
            ClusterCommand::Insert { collection, documents, vectors } => {
                if let Some(col) = self.collections.get_mut(collection) {
                    if let Err(e) = col.insert(documents.clone(), vectors.clone()) {
                        eprintln!("[Raft] Error applying insert to '{}': {}", collection, e);
                    }
                }
            }
            ClusterCommand::Delete { collection, ids } => {
                if let Some(col) = self.collections.get_mut(collection) {
                    if let Err(e) = col.delete(ids) {
                        eprintln!("[Raft] Error applying delete to '{}': {}", collection, e);
                    }
                }
            }
            ClusterCommand::Noop => {}
        }
    }
}

/// Simulated in-memory Raft cluster network router for rapid, deterministic testing.
pub struct MemoryRaftCluster {
    pub nodes: HashMap<String, Arc<Mutex<RaftNode>>>,
}

impl MemoryRaftCluster {
    pub fn new(node_ids: &[&str]) -> Self {
        let mut nodes = HashMap::new();
        for &id in node_ids {
            let peers: Vec<String> = node_ids.iter().filter(|&&p| p != id).map(|&p| p.to_string()).collect();
            let node = RaftNode::new(id, peers);
            nodes.insert(id.to_string(), Arc::new(Mutex::new(node)));
        }
        Self { nodes }
    }

    /// Triggers election on candidate and delivers votes across all peers.
    pub fn elect_leader(&self, candidate_id: &str) -> bool {
        let (args, peers) = {
            let mut cand = self.nodes.get(candidate_id).unwrap().lock().unwrap();
            let args = cand.start_election();
            (args, cand.peers.clone())
        };

        let mut votes = 1; // Votes for self
        let quorum = peers.len().div_ceil(2) + 1;

        for peer_id in peers {
            if let Some(peer_arc) = self.nodes.get(&peer_id) {
                let mut peer = peer_arc.lock().unwrap();
                let reply = peer.handle_request_vote(&args);
                if reply.vote_granted {
                    votes += 1;
                }
            }
        }

        if votes >= quorum {
            let mut cand = self.nodes.get(candidate_id).unwrap().lock().unwrap();
            cand.become_leader();
            true
        } else {
            false
        }
    }

    /// Proposes command on leader and synchronizes replication across all reachable peers.
    pub fn propose_and_commit(&self, leader_id: &str, command: ClusterCommand) -> Result<u64, SutraError> {
        let (index, peer_msgs) = {
            let mut leader = self.nodes.get(leader_id).unwrap().lock().unwrap();
            leader.propose(command)?
        };

        for (peer_id, args) in peer_msgs {
            if let Some(peer_arc) = self.nodes.get(&peer_id) {
                let reply = {
                    let mut peer = peer_arc.lock().unwrap();
                    peer.handle_append_entries(&args)
                };
                let mut leader = self.nodes.get(leader_id).unwrap().lock().unwrap();
                leader.handle_append_reply(&peer_id, &reply);
            }
        }

        // Send a follow-up commit sync heartbeat to followers
        let sync_msgs = {
            let leader = self.nodes.get(leader_id).unwrap().lock().unwrap();
            let mut msgs = Vec::new();
            for peer in &leader.peers {
                msgs.push((peer.clone(), AppendEntriesArgs {
                    term: leader.current_term,
                    leader_id: leader.node_id.clone(),
                    prev_log_index: leader.last_log_index(),
                    prev_log_term: leader.last_log_term(),
                    entries: Vec::new(),
                    leader_commit: leader.commit_index,
                }));
            }
            msgs
        };

        for (peer_id, args) in sync_msgs {
            if let Some(peer_arc) = self.nodes.get(&peer_id) {
                let mut peer = peer_arc.lock().unwrap();
                peer.handle_append_entries(&args);
            }
        }

        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raft_election_and_quorum() {
        let cluster = MemoryRaftCluster::new(&["node-1", "node-2", "node-3"]);
        assert!(cluster.elect_leader("node-1"));

        let node1 = cluster.nodes.get("node-1").unwrap().lock().unwrap();
        assert_eq!(node1.role, RaftRole::Leader);
        assert_eq!(node1.current_term, 1);

        let node2 = cluster.nodes.get("node-2").unwrap().lock().unwrap();
        assert_eq!(node2.role, RaftRole::Follower);
        assert_eq!(node2.voted_for, Some("node-1".to_string()));
    }

    #[test]
    fn test_raft_log_replication_and_state_machine_apply() {
        let cluster = MemoryRaftCluster::new(&["n1", "n2", "n3"]);
        assert!(cluster.elect_leader("n1"));

        // 1. Propose CreateCollection
        let cmd = ClusterCommand::CreateCollection {
            name: "kb".to_string(),
            dimension: 4,
            metric: "cosine".to_string(),
        };
        let idx = cluster.propose_and_commit("n1", cmd).unwrap();
        assert_eq!(idx, 1);

        // Verify applied on all nodes
        for id in &["n1", "n2", "n3"] {
            let n = cluster.nodes.get(*id).unwrap().lock().unwrap();
            assert_eq!(n.commit_index, 1);
            assert_eq!(n.last_applied, 1);
            assert!(n.collections.contains_key("kb"));
        }

        // 2. Propose Insert documents
        let docs = vec![
            NativeDocument {
                id: "d1".to_string(),
                namespace: Some("tenant_a".to_string()),
                text: "Distributed vector systems".to_string(),
                metadata: serde_json::json!({"tier": "gold"}),
                sparse_vector: None,
                colbert_tokens: None,
            }
        ];
        let vecs = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let insert_cmd = ClusterCommand::Insert {
            collection: "kb".to_string(),
            documents: docs,
            vectors: vecs,
        };
        cluster.propose_and_commit("n1", insert_cmd).unwrap();

        for id in &["n1", "n2", "n3"] {
            let n = cluster.nodes.get(*id).unwrap().lock().unwrap();
            assert_eq!(n.commit_index, 2);
            assert_eq!(n.last_applied, 2);
            let col = n.collections.get("kb").unwrap();
            assert_eq!(col.count(), 1);
        }
    }

    #[test]
    fn test_raft_leader_failover_and_re_election() {
        let cluster = MemoryRaftCluster::new(&["node-a", "node-b", "node-c"]);
        assert!(cluster.elect_leader("node-a"));

        // Leader node-a crashes / stops
        {
            let mut n_a = cluster.nodes.get("node-a").unwrap().lock().unwrap();
            n_a.role = RaftRole::Follower; // Simulate stepping down or disconnecting
        }

        // Node-b starts election and wins with node-c's vote
        assert!(cluster.elect_leader("node-b"));

        let n_b = cluster.nodes.get("node-b").unwrap().lock().unwrap();
        assert_eq!(n_b.role, RaftRole::Leader);
        assert_eq!(n_b.current_term, 2);

        let n_c = cluster.nodes.get("node-c").unwrap().lock().unwrap();
        assert_eq!(n_c.voted_for, Some("node-b".to_string()));
    }
}
