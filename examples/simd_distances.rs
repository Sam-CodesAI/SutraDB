//! Native SIMD Distance Micro-Benchmark
//! Evaluates raw distance throughput (computations/sec) for Cosine, Euclidean, and Dot Product.

use std::time::Instant;
use _sutradb::distance::simd::{compute_vector_score, dot_product, euclidean_distance_squared, detect_simd_capabilities};
use _sutradb::distance::Metric;

fn main() {
    println!("============================================================");
    println!(" ⚡ SutraDB Native Rust SIMD Micro-Benchmark");
    println!("============================================================");
    println!("Detected SIMD CPU Extension: {}", detect_simd_capabilities());

    let dim = 128;
    let n_pairs = 1_000_000;
    println!("Vectors: {} pairs | Dimension: {}", n_pairs, dim);

    let v1: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
    let v2: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();

    // 1. Dot Product (8-way unrolled SIMD)
    let start = Instant::now();
    let mut sum_dot = 0.0f32;
    for _ in 0..n_pairs {
        sum_dot += dot_product(&v1, &v2);
    }
    let duration_dot = start.elapsed();
    let qps_dot = (n_pairs as f64) / duration_dot.as_secs_f64();
    println!("Dot Product: {:.2?} ({:.2}M ops/sec) [chk: {:.4}]", duration_dot, qps_dot / 1_000_000.0, sum_dot / (n_pairs as f32));

    // 2. Squared Euclidean Distance
    let start = Instant::now();
    let mut sum_l2 = 0.0f32;
    for _ in 0..n_pairs {
        sum_l2 += euclidean_distance_squared(&v1, &v2);
    }
    let duration_l2 = start.elapsed();
    let qps_l2 = (n_pairs as f64) / duration_l2.as_secs_f64();
    println!("Squared L2:  {:.2?} ({:.2}M ops/sec) [chk: {:.4}]", duration_l2, qps_l2 / 1_000_000.0, sum_l2 / (n_pairs as f32));

    // 3. Cosine Score Evaluation
    let start = Instant::now();
    let mut sum_cos = 0.0f32;
    for _ in 0..n_pairs {
        sum_cos += compute_vector_score(&v1, &v2, Metric::Cosine, true).unwrap();
    }
    let duration_cos = start.elapsed();
    let qps_cos = (n_pairs as f64) / duration_cos.as_secs_f64();
    println!("Cosine Sim:  {:.2?} ({:.2}M ops/sec) [chk: {:.4}]", duration_cos, qps_cos / 1_000_000.0, sum_cos / (n_pairs as f32));
    println!("============================================================");
}
