"""
Score fusion algorithms for hybrid vector + lexical search in SutraDB.
Implements Reciprocal Rank Fusion (RRF) and Convex Linear Combination.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np


def _finite_or_zero(val: float) -> float:
    f = float(val)
    return f if math.isfinite(f) else 0.0


def reciprocal_rank_fusion(
    dense_scores: np.ndarray,
    bm25_scores: np.ndarray,
    top_k: int = 10,
    k_constant: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    mask: Optional[np.ndarray] = None
) -> List[Tuple[int, float, float, float]]:
    """
    Combines dense similarity scores and BM25 lexical scores using Reciprocal Rank Fusion (RRF).
    Candidate-pruned: evaluates top M candidates from dense and top M from BM25, computing
    RRF scores strictly over the candidate union.
    
    Formula:
        RRF_Score(d) = (w_dense / (k + rank_dense(d))) + (w_bm25 / (k + rank_bm25(d)))
        
    Returns:
        List of tuples: (doc_index, final_rrf_score, dense_raw_score, bm25_raw_score)
        Sorted descending by final_rrf_score.
    """
    n = len(dense_scores)
    if top_k <= 0 or n == 0:
        return []

    # Filter by mask if provided
    valid_indices = np.where(mask)[0] if mask is not None else np.arange(n)
    if len(valid_indices) == 0:
        return []

    # Bound candidate count M: min(top_k * 4, 200)
    M = min(top_k * 4, 200)

    # 1. Top M candidates from dense search (strictly finite scores only)
    dense_valid = dense_scores[valid_indices]
    dense_finite_mask = np.isfinite(dense_valid)
    dense_finite_indices = valid_indices[dense_finite_mask]
    dense_finite_vals = dense_valid[dense_finite_mask]

    dense_candidates: List[int] = []
    dense_ranks: Dict[int, int] = {}
    if len(dense_finite_vals) > 0:
        k_dense = min(M, len(dense_finite_vals))
        if k_dense < len(dense_finite_vals):
            top_dense_local = np.argpartition(-dense_finite_vals, k_dense - 1)[:k_dense]
            top_dense_local = top_dense_local[np.argsort(-dense_finite_vals[top_dense_local])]
        else:
            top_dense_local = np.argsort(-dense_finite_vals)

        dense_candidates = [int(dense_finite_indices[i]) for i in top_dense_local]
        for rank, doc_idx in enumerate(dense_candidates, start=1):
            if dense_scores[doc_idx] > 0.0:
                dense_ranks[doc_idx] = rank

        # If all finite candidate dense scores are non-positive, award rank to all finite candidates
        if not dense_ranks and len(dense_candidates) > 0:
            dense_ranks = {
                doc_idx: rank for rank, doc_idx in enumerate(dense_candidates, start=1)
            }

    # 2. Top M candidates from BM25 lexical search (strictly finite & positive BM25 scores)
    bm25_valid = bm25_scores[valid_indices]
    pos_mask = np.isfinite(bm25_valid) & (bm25_valid > 0.0)
    pos_indices = valid_indices[pos_mask]
    pos_scores = bm25_valid[pos_mask]

    bm25_candidates: List[int] = []
    bm25_ranks: Dict[int, int] = {}
    if len(pos_scores) > 0:
        k_bm25 = min(M, len(pos_scores))
        if k_bm25 < len(pos_scores):
            top_bm25_local = np.argpartition(-pos_scores, k_bm25 - 1)[:k_bm25]
            top_bm25_local = top_bm25_local[np.argsort(-pos_scores[top_bm25_local])]
        else:
            top_bm25_local = np.argsort(-pos_scores)
        bm25_candidates = [int(pos_indices[i]) for i in top_bm25_local]
        bm25_ranks = {
            doc_idx: rank for rank, doc_idx in enumerate(bm25_candidates, start=1)
        }

    # 3. Candidate union
    candidate_union = list(dict.fromkeys(dense_candidates + bm25_candidates))

    # 4. Compute RRF scores strictly over the candidate union
    scored_candidates: List[Tuple[int, float, float, float, float, float]] = []
    for doc_idx in candidate_union:
        score = 0.0
        if doc_idx in dense_ranks:
            score += dense_weight / (k_constant + dense_ranks[doc_idx])
        if doc_idx in bm25_ranks:
            score += bm25_weight / (k_constant + bm25_ranks[doc_idx])

        raw_d = float(dense_scores[doc_idx])
        raw_b = float(bm25_scores[doc_idx])
        tie_d = raw_d if math.isfinite(raw_d) else -math.inf
        tie_b = raw_b if math.isfinite(raw_b) else -math.inf
        scored_candidates.append((
            doc_idx,
            score,
            _finite_or_zero(raw_d),
            _finite_or_zero(raw_b),
            tie_d,
            tie_b,
        ))

    # 5. Sort candidate union by combined RRF score descending with deterministic tie-breaking
    scored_candidates.sort(key=lambda item: (item[1], item[4], item[5]), reverse=True)
    return [(item[0], item[1], item[2], item[3]) for item in scored_candidates[:top_k]]


def linear_score_fusion(
    dense_scores: np.ndarray,
    bm25_scores: np.ndarray,
    alpha: float = 0.6,
    top_k: int = 10,
    mask: Optional[np.ndarray] = None
) -> List[Tuple[int, float, float, float]]:
    """
    Combines dense and lexical scores via min-max normalized convex linear combination:
        Final_Score = alpha * norm(dense) + (1 - alpha) * norm(bm25)
    """
    n = len(dense_scores)
    if top_k <= 0 or n == 0:
        return []

    valid_indices = np.where(mask)[0] if mask is not None else np.arange(n)
    if len(valid_indices) == 0:
        return []

    dense_v = np.asarray(dense_scores[valid_indices], dtype=np.float32)
    bm25_v = np.asarray(bm25_scores[valid_indices], dtype=np.float32)

    # Min-max normalize finite dense scores to [0, 1]; non-finite entries remain 0.0
    d_finite = np.isfinite(dense_v)
    d_norm = np.zeros_like(dense_v, dtype=np.float32)
    if np.any(d_finite):
        d_vals = dense_v[d_finite]
        d_min = float(np.min(d_vals))
        d_max = float(np.max(d_vals))
        if d_max > d_min:
            d_norm[d_finite] = (d_vals - d_min) / max(d_max - d_min, 1e-12)
        else:
            d_norm[d_finite] = 1.0

    # Min-max normalize finite BM25 scores to [0, 1]; non-finite entries remain 0.0
    b_finite = np.isfinite(bm25_v)
    b_norm = np.zeros_like(bm25_v, dtype=np.float32)
    if np.any(b_finite):
        b_vals = bm25_v[b_finite]
        b_min = float(np.min(b_vals))
        b_max = float(np.max(b_vals))
        if b_max > b_min:
            b_norm[b_finite] = (b_vals - b_min) / max(b_max - b_min, 1e-12)

    combined = alpha * d_norm + (1.0 - alpha) * b_norm

    order = sorted(
        range(len(valid_indices)),
        key=lambda idx: (
            float(combined[idx]) if math.isfinite(float(combined[idx])) else -math.inf,
            float(dense_v[idx]) if math.isfinite(float(dense_v[idx])) else -math.inf,
            float(bm25_v[idx]) if math.isfinite(float(bm25_v[idx])) else -math.inf,
        ),
        reverse=True,
    )[:top_k]

    results = []
    for local_idx in order:
        global_idx = int(valid_indices[local_idx])
        results.append((
            global_idx,
            _finite_or_zero(combined[local_idx]),
            _finite_or_zero(dense_scores[global_idx]),
            _finite_or_zero(bm25_scores[global_idx])
        ))

    return results
