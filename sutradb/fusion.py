"""
Score fusion algorithms for hybrid vector + lexical search in SutraDB.
Implements Reciprocal Rank Fusion (RRF) and Convex Linear Combination.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


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

    # 1. Top M candidates from dense search
    dense_valid = dense_scores[valid_indices]
    k_dense = min(M, len(dense_valid))
    if k_dense < len(dense_valid):
        top_dense_local = np.argpartition(-dense_valid, k_dense - 1)[:k_dense]
        top_dense_local = top_dense_local[np.argsort(-dense_valid[top_dense_local])]
    else:
        top_dense_local = np.argsort(-dense_valid)

    dense_candidates = [int(valid_indices[i]) for i in top_dense_local]
    dense_ranks: Dict[int, int] = {}
    for rank, doc_idx in enumerate(dense_candidates, start=1):
        if dense_scores[doc_idx] > 0.0:
            dense_ranks[doc_idx] = rank

    # If all candidate dense scores are non-positive (e.g. opposing/orthogonal cosine), award rank to all candidates
    if not dense_ranks and len(dense_candidates) > 0:
        dense_ranks = {
            doc_idx: rank for rank, doc_idx in enumerate(dense_candidates, start=1)
        }

    # 2. Top M candidates from BM25 lexical search (only considering positive BM25 scores)
    bm25_valid = bm25_scores[valid_indices]
    pos_mask = bm25_valid > 0.0
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
    scored_candidates: List[Tuple[int, float, float, float]] = []
    for doc_idx in candidate_union:
        score = 0.0
        if doc_idx in dense_ranks:
            score += dense_weight / (k_constant + dense_ranks[doc_idx])
        if doc_idx in bm25_ranks:
            score += bm25_weight / (k_constant + bm25_ranks[doc_idx])

        scored_candidates.append((
            doc_idx,
            score,
            float(dense_scores[doc_idx]),
            float(bm25_scores[doc_idx])
        ))

    # 5. Sort candidate union by combined RRF score descending with deterministic tie-breaking
    scored_candidates.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
    return scored_candidates[:top_k]


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
    if n == 0:
        return []

    valid_indices = np.where(mask)[0] if mask is not None else np.arange(n)
    if len(valid_indices) == 0:
        return []

    dense_v = dense_scores[valid_indices]
    bm25_v = bm25_scores[valid_indices]

    # Min-max normalize dense scores to [0, 1]
    d_min, d_max = dense_v.min(), dense_v.max()
    d_norm = (dense_v - d_min) / (d_max - d_min + 1e-12) if d_max > d_min else np.ones_like(dense_v)

    # Min-max normalize BM25 scores to [0, 1]
    b_min, b_max = bm25_v.min(), bm25_v.max()
    b_norm = (bm25_v - b_min) / (b_max - b_min + 1e-12) if b_max > b_min else np.zeros_like(bm25_v)

    combined = alpha * d_norm + (1.0 - alpha) * b_norm
    top_local_indices = np.argsort(-combined)[:top_k]

    results = []
    for local_idx in top_local_indices:
        global_idx = valid_indices[local_idx]
        results.append((
            global_idx,
            float(combined[local_idx]),
            float(dense_scores[global_idx]),
            float(bm25_scores[global_idx])
        ))

    return results
