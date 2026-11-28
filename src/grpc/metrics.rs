use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::collection::NativeCollection;
use crate::distance::Metric;

pub const QUERY_DURATION_BUCKETS: [f64; 10] = [
    0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5,
];

#[derive(Debug, Clone)]
struct QueryHistogram {
    buckets: [u64; 10],
    count: u64,
    sum: f64,
}

impl Default for QueryHistogram {
    fn default() -> Self {
        Self {
            buckets: [0; 10],
            count: 0,
            sum: 0.0,
        }
    }
}

impl QueryHistogram {
    fn observe(&mut self, secs: f64) {
        self.count += 1;
        self.sum += secs;
        for (idx, &bound) in QUERY_DURATION_BUCKETS.iter().enumerate() {
            if secs <= bound {
                self.buckets[idx] += 1;
            }
        }
    }
}

pub struct CollectionMetricSnapshot {
    pub name: String,
    pub documents: u64,
    pub memory_bytes: u64,
    pub vector_memory_bytes: u64,
    pub wal_pending_records: u64,
}

#[derive(Default)]
pub struct MetricsRegistry {
    ingest_documents_total: AtomicU64,
    errors_total: AtomicU64,
    query_histograms: Mutex<HashMap<(String, &'static str), QueryHistogram>>,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_query(&self, collection: &str, mode: &'static str, duration_secs: f64) {
        if let Ok(mut guard) = self.query_histograms.lock() {
            let entry = guard
                .entry((collection.to_string(), mode))
                .or_default();
            entry.observe(duration_secs.max(0.0));
        }
    }

    pub fn record_ingest(&self, docs_count: u64) {
        self.ingest_documents_total
            .fetch_add(docs_count, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn render_prometheus(&self, snapshots: &[CollectionMetricSnapshot]) -> String {
        let mut out = String::with_capacity(2048);

        let _ = writeln!(out, "# HELP sutradb_collections_total Total number of collections");
        let _ = writeln!(out, "# TYPE sutradb_collections_total gauge");
        let _ = writeln!(out, "sutradb_collections_total {}", snapshots.len());

        let _ = writeln!(
            out,
            "# HELP sutradb_collection_documents Active documents stored per collection"
        );
        let _ = writeln!(out, "# TYPE sutradb_collection_documents gauge");
        for snap in snapshots {
            let safe = escape_label(&snap.name);
            let _ = writeln!(
                out,
                "sutradb_collection_documents{{collection=\"{}\"}} {}",
                safe, snap.documents
            );
        }

        let _ = writeln!(
            out,
            "# HELP sutradb_documents_count Active documents stored per collection"
        );
        let _ = writeln!(out, "# TYPE sutradb_documents_count gauge");
        for snap in snapshots {
            let safe = escape_label(&snap.name);
            let _ = writeln!(
                out,
                "sutradb_documents_count{{collection=\"{}\"}} {}",
                safe, snap.documents
            );
        }

        let _ = writeln!(
            out,
            "# HELP sutradb_collection_memory_bytes Estimated vector and index memory usage in bytes per collection"
        );
        let _ = writeln!(out, "# TYPE sutradb_collection_memory_bytes gauge");
        for snap in snapshots {
            let safe = escape_label(&snap.name);
            let _ = writeln!(
                out,
                "sutradb_collection_memory_bytes{{collection=\"{}\"}} {}",
                safe, snap.memory_bytes
            );
        }

        let _ = writeln!(
            out,
            "# HELP sutradb_vector_memory_bytes Estimated vector memory usage in bytes per collection"
        );
        let _ = writeln!(out, "# TYPE sutradb_vector_memory_bytes gauge");
        for snap in snapshots {
            let safe = escape_label(&snap.name);
            let _ = writeln!(
                out,
                "sutradb_vector_memory_bytes{{collection=\"{}\"}} {}",
                safe, snap.vector_memory_bytes
            );
        }

        let _ = writeln!(
            out,
            "# HELP sutradb_wal_pending_records Uncompacted Write-Ahead Log records per collection"
        );
        let _ = writeln!(out, "# TYPE sutradb_wal_pending_records gauge");
        for snap in snapshots {
            let safe = escape_label(&snap.name);
            let _ = writeln!(
                out,
                "sutradb_wal_pending_records{{collection=\"{}\"}} {}",
                safe, snap.wal_pending_records
            );
        }

        let _ = writeln!(
            out,
            "# HELP sutradb_ingest_documents_total Total documents ingested into SutraDB"
        );
        let _ = writeln!(out, "# TYPE sutradb_ingest_documents_total counter");
        let _ = writeln!(
            out,
            "sutradb_ingest_documents_total {}",
            self.ingest_documents_total.load(Ordering::Relaxed)
        );

        let _ = writeln!(
            out,
            "# HELP sutradb_errors_total Total failed gRPC and HTTP operations"
        );
        let _ = writeln!(out, "# TYPE sutradb_errors_total counter");
        let _ = writeln!(
            out,
            "sutradb_errors_total {}",
            self.errors_total.load(Ordering::Relaxed)
        );

        let _ = writeln!(
            out,
            "# HELP sutradb_query_duration_seconds Query execution latency in seconds"
        );
        let _ = writeln!(out, "# TYPE sutradb_query_duration_seconds histogram");
        if let Ok(guard) = self.query_histograms.lock() {
            let mut entries: Vec<(&(String, &'static str), &QueryHistogram)> =
                guard.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            for ((col, mode), hist) in entries {
                let safe_col = escape_label(col);
                for (idx, &bound) in QUERY_DURATION_BUCKETS.iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "sutradb_query_duration_seconds_bucket{{collection=\"{}\",mode=\"{}\",le=\"{}\"}} {}",
                        safe_col, mode, bound, hist.buckets[idx]
                    );
                }
                let _ = writeln!(
                    out,
                    "sutradb_query_duration_seconds_bucket{{collection=\"{}\",mode=\"{}\",le=\"+Inf\"}} {}",
                    safe_col, mode, hist.count
                );
                let _ = writeln!(
                    out,
                    "sutradb_query_duration_seconds_sum{{collection=\"{}\",mode=\"{}\"}} {:.9}",
                    safe_col, mode, hist.sum
                );
                let _ = writeln!(
                    out,
                    "sutradb_query_duration_seconds_count{{collection=\"{}\",mode=\"{}\"}} {}",
                    safe_col, mode, hist.count
                );
            }
        }

        out
    }
}

#[inline]
fn escape_label(val: &str) -> String {
    val.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Classifies the effective execution mode for Prometheus query latency histograms.
pub(crate) fn classify_query_mode(
    col: &NativeCollection,
    has_vector: bool,
    has_text: bool,
    has_sparse: bool,
    has_colbert: bool,
    hybrid: bool,
    use_ann: bool,
) -> &'static str {
    let signal_count =
        (has_vector as u8) + (has_text as u8) + (has_sparse as u8) + (has_colbert as u8);
    if !has_vector && has_text && !has_sparse && !has_colbert {
        return "bm25";
    }
    if signal_count >= 2 && hybrid {
        return "hybrid";
    }
    if has_vector {
        if use_ann && !col.dirty_index && !col.vectors_evicted && col.hnsw.is_some() {
            return "hnsw";
        }
        if col.use_pq {
            return "pq";
        }
        if col.use_bq && col.metric != Metric::Euclidean {
            return "bq";
        }
        if col.use_sq8 {
            return "sq8";
        }
        return "flat";
    }
    "hybrid"
}
