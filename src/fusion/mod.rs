use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct FusedCandidate {
    pub doc_idx: usize,
    pub score: f32,
    pub dense_score: f32,
    pub bm25_score: f32,
}

/// Candidate-pruned Reciprocal Rank Fusion (RRF).
/// RRF_Score(d) = w_dense / (k + rank_dense(d)) + w_bm25 / (k + rank_bm25(d))
pub fn reciprocal_rank_fusion(
    dense_scores: &[f32],
    bm25_scores: &[f32],
    top_k: usize,
    k_constant: usize,
    dense_weight: f32,
    bm25_weight: f32,
    mask: Option<&[bool]>,
) -> Vec<FusedCandidate> {
    let n = dense_scores.len();
    if top_k == 0 || n == 0 {
        return Vec::new();
    }

    // Collect valid indices
    let valid_indices: Vec<usize> = match mask {
        Some(m) => (0..n).filter(|&i| i < m.len() && m[i]).collect(),
        None => (0..n).collect(),
    };

    if valid_indices.is_empty() {
        return Vec::new();
    }

    // Candidate bound M = min(top_k * 4, 200)
    let m_bound = (top_k * 4).min(200);

    // 1. Top dense candidates
    let mut dense_pairs: Vec<(usize, f32)> = valid_indices
        .iter()
        .map(|&i| (i, dense_scores[i]))
        .collect();

    dense_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let k_dense = m_bound.min(dense_pairs.len());
    let top_dense = &dense_pairs[..k_dense];

    let mut dense_ranks: HashMap<usize, usize> = HashMap::new();
    for (rank, &(idx, score)) in top_dense.iter().enumerate() {
        if score > 0.0 {
            dense_ranks.insert(idx, rank + 1);
        }
    }
    // If all candidate dense scores are non-positive (e.g. opposing cosine), award rank to all candidates
    if dense_ranks.is_empty() && !top_dense.is_empty() {
        for (rank, &(idx, _)) in top_dense.iter().enumerate() {
            dense_ranks.insert(idx, rank + 1);
        }
    }

    // 2. Top BM25 candidates (strictly positive scores)
    let mut bm25_pairs: Vec<(usize, f32)> = valid_indices
        .iter()
        .filter(|&&i| bm25_scores[i] > 0.0)
        .map(|&i| (i, bm25_scores[i]))
        .collect();

    bm25_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let k_bm25 = m_bound.min(bm25_pairs.len());
    let top_bm25 = &bm25_pairs[..k_bm25];

    let mut bm25_ranks: HashMap<usize, usize> = HashMap::new();
    for (rank, &(idx, _)) in top_bm25.iter().enumerate() {
        bm25_ranks.insert(idx, rank + 1);
    }

    // 3. Candidate union
    let mut union_set = Vec::new();
    for &(idx, _) in top_dense {
        if !union_set.contains(&idx) {
            union_set.push(idx);
        }
    }
    for &(idx, _) in top_bm25 {
        if !union_set.contains(&idx) {
            union_set.push(idx);
        }
    }

    // 4. Compute RRF scores over union
    let mut candidates = Vec::with_capacity(union_set.len());
    let k_const = k_constant as f32;

    for &doc_idx in &union_set {
        let mut score = 0.0f32;
        if let Some(&r) = dense_ranks.get(&doc_idx) {
            score += dense_weight / (k_const + r as f32);
        }
        if let Some(&r) = bm25_ranks.get(&doc_idx) {
            score += bm25_weight / (k_const + r as f32);
        }

        candidates.push(FusedCandidate {
            doc_idx,
            score,
            dense_score: dense_scores[doc_idx],
            bm25_score: bm25_scores[doc_idx],
        });
    }

    // 5. Deterministic tie-breaking sort (score desc, dense_score desc, bm25_score desc)
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.dense_score.partial_cmp(&a.dense_score).unwrap_or(Ordering::Equal))
            .then_with(|| b.bm25_score.partial_cmp(&a.bm25_score).unwrap_or(Ordering::Equal))
    });

    candidates.truncate(top_k);
    candidates
}

/// Convex linear combination: score = alpha * norm(dense) + (1 - alpha) * norm(bm25)
pub fn linear_score_fusion(
    dense_scores: &[f32],
    bm25_scores: &[f32],
    alpha: f32,
    top_k: usize,
    mask: Option<&[bool]>,
) -> Vec<FusedCandidate> {
    let n = dense_scores.len();
    if top_k == 0 || n == 0 {
        return Vec::new();
    }

    let valid_indices: Vec<usize> = match mask {
        Some(m) => (0..n).filter(|&i| i < m.len() && m[i]).collect(),
        None => (0..n).collect(),
    };

    if valid_indices.is_empty() {
        return Vec::new();
    }

    let mut d_min = f32::INFINITY;
    let mut d_max = f32::NEG_INFINITY;
    let mut b_min = f32::INFINITY;
    let mut b_max = f32::NEG_INFINITY;

    for &i in &valid_indices {
        let d = dense_scores[i];
        let b = bm25_scores[i];
        if d < d_min { d_min = d; }
        if d > d_max { d_max = d; }
        if b < b_min { b_min = b; }
        if b > b_max { b_max = b; }
    }

    let d_range = (d_max - d_min).max(1e-12);
    let b_range = (b_max - b_min).max(1e-12);

    let mut scored: Vec<FusedCandidate> = valid_indices
        .iter()
        .map(|&i| {
            let d_norm = if d_max > d_min { (dense_scores[i] - d_min) / d_range } else { 1.0 };
            let b_norm = if b_max > b_min { (bm25_scores[i] - b_min) / b_range } else { 0.0 };
            let score = alpha * d_norm + (1.0 - alpha) * b_norm;
            FusedCandidate {
                doc_idx: i,
                score,
                dense_score: dense_scores[i],
                bm25_score: bm25_scores[i],
            }
        })
        .collect();

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    scored.truncate(top_k);
    scored
}
