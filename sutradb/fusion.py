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
    
    Formula:
        RRF_Score(d) = (w_dense / (k + rank_dense(d))) + (w_bm25 / (k + rank_bm25(d)))
        
    Returns:
        List of tuples: (doc_index, final_rrf_score, dense_raw_score, bm25_raw_score)
        Sorted descending by final_rrf_score.
    """
    n = len(dense_scores)
    if n == 0:
        return []

    # Filter by mask if provided
    valid_indices = np.where(mask)[0] if mask is not None else np.arange(n)
    if len(valid_indices) == 0:
        return []

    # Map doc_index -> RRF accumulator
    rrf_map: Dict[int, float] = {idx: 0.0 for idx in valid_indices}

    # Dense ranking (sort valid indices by dense score descending)
    dense_valid = dense_scores[valid_indices]
    dense_sorted_local = np.argsort(-dense_valid)
    for rank, local_idx in enumerate(dense_sorted_local, start=1):
        global_idx = valid_indices[local_idx]
        if dense_scores[global_idx] > 1e-6:
            rrf_map[global_idx] += dense_weight / (k_constant + rank)

    # BM25 ranking (sort valid indices by BM25 score descending)
    # Only award BM25 RRF rank if document had non-zero BM25 score
    bm25_valid = bm25_scores[valid_indices]
    bm25_sorted_local = np.argsort(-bm25_valid)
    for rank, local_idx in enumerate(bm25_sorted_local, start=1):
        global_idx = valid_indices[local_idx]
        if bm25_scores[global_idx] > 0.0:
            rrf_map[global_idx] += bm25_weight / (k_constant + rank)

    # Sort candidates by combined RRF score
    sorted_candidates = sorted(rrf_map.items(), key=lambda item: item[1], reverse=True)
    top_candidates = sorted_candidates[:top_k]

    results = [
        (idx, score, float(dense_scores[idx]), float(bm25_scores[idx]))
        for idx, score in top_candidates
    ]
    return results


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
