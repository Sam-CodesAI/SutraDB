"""
IVF-Flat Approximate Nearest Neighbor index for SutraDB.

Implements Inverted File indexing with flat (exact) inner-cluster search,
identical to FAISS IVF-Flat. Provides sublinear query time O(n * nprobe / n_lists)
instead of brute-force O(n), enabling 100K–10M scale vector search.

Architecture:
    1. Training phase: k-means++ computes `n_lists` centroids from a sample.
    2. Insertion phase: each vector is assigned to its nearest centroid's inverted list.
    3. Query phase: find `nprobe` nearest centroids, scan only their inverted lists.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from sutradb.distance import Metric, compute_distances, normalize_vector


# ──────────────────────────────────────────────────────────────────────────────
# k-means++ Implementation
# ──────────────────────────────────────────────────────────────────────────────

def _kmeans_plus_plus_init(
    vectors: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    k-means++ centroid initialization.
    Selects initial centroids with probability proportional to squared distance
    from the nearest existing centroid, ensuring well-spread initial clusters.
    """
    n, dim = vectors.shape
    centroids = np.empty((n_clusters, dim), dtype=np.float32)

    # First centroid: random
    idx = rng.integers(0, n)
    centroids[0] = vectors[idx]

    # Subsequent centroids: D² weighted sampling
    for k in range(1, n_clusters):
        # Compute distances from each vector to nearest existing centroid
        # Use batch dot-product for speed
        dists = np.full(n, np.inf, dtype=np.float32)
        for j in range(k):
            diff = vectors - centroids[j]
            d2 = np.sum(diff * diff, axis=1)
            dists = np.minimum(dists, d2)

        # Probability proportional to squared distance
        probs = dists / (dists.sum() + 1e-30)
        idx = rng.choice(n, p=probs)
        centroids[k] = vectors[idx]

    return centroids


def _kmeans(
    vectors: np.ndarray,
    n_clusters: int,
    max_iters: int = 25,
    tol: float = 1e-4,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lloyd's k-means algorithm with k-means++ initialization.

    Args:
        vectors: (N, D) array of float32 vectors.
        n_clusters: Number of clusters to form.
        max_iters: Maximum Lloyd iterations.
        tol: Convergence tolerance (relative centroid shift).
        seed: Random seed for reproducibility.

    Returns:
        centroids: (n_clusters, D) float32 centroid matrix.
        assignments: (N,) int32 cluster assignments.
    """
    n, dim = vectors.shape
    n_clusters = min(n_clusters, n)
    rng = np.random.default_rng(seed)

    # Initialize centroids with k-means++
    centroids = _kmeans_plus_plus_init(vectors, n_clusters, rng)
    assignments = np.zeros(n, dtype=np.int32)

    for iteration in range(max_iters):
        # Assignment step: each vector → nearest centroid
        # Vectorized via batch matrix multiplication
        # dist²(v, c) = ||v||² + ||c||² - 2 * v·c
        v_sq = np.sum(vectors * vectors, axis=1, keepdims=True)  # (N, 1)
        c_sq = np.sum(centroids * centroids, axis=1, keepdims=True)  # (K, 1)
        cross = vectors @ centroids.T  # (N, K)
        dists = v_sq + c_sq.T - 2.0 * cross  # (N, K)

        new_assignments = np.argmin(dists, axis=1).astype(np.int32)

        # Update step: recompute centroids as cluster means
        old_centroids = centroids.copy()
        for k in range(n_clusters):
            members = vectors[new_assignments == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
            else:
                # Empty cluster: re-seed from the largest cluster
                largest_cluster = np.bincount(new_assignments, minlength=n_clusters).argmax()
                members_of_largest = np.where(new_assignments == largest_cluster)[0]
                random_idx = rng.choice(members_of_largest)
                centroids[k] = vectors[random_idx]

        assignments = new_assignments

        # Check convergence
        shift = np.linalg.norm(centroids - old_centroids)
        if shift < tol:
            break

    return centroids.astype(np.float32), assignments


# ──────────────────────────────────────────────────────────────────────────────
# IVF-Flat Index
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IVFConfig:
    """Configuration for the IVF-Flat index."""
    n_lists: int = 256
    """Number of Voronoi cells (inverted lists). Rule of thumb: sqrt(N) to 4*sqrt(N)."""
    nprobe: int = 16
    """Number of lists to search at query time. Higher = better recall, slower speed."""
    train_sample_size: int = 50_000
    """Maximum number of vectors to sample for k-means training."""
    kmeans_iters: int = 25
    """Maximum k-means iterations."""
    auto_threshold: int = 5_000
    """Auto-build the index when collection size exceeds this count."""
    seed: int = 42
    """Random seed for reproducibility."""


class IVFIndex:
    """
    Inverted File Flat (IVF-Flat) approximate nearest neighbor index.

    Partitions the vector space into `n_lists` Voronoi cells using k-means,
    then at query time only searches the `nprobe` nearest cells.

    Complexity:
        - Brute-force: O(N * D)
        - IVF-Flat:    O(n_lists * D + N * D * nprobe / n_lists)

    For 100K vectors with n_lists=316, nprobe=16: scans ~5% of vectors.
    For 1M vectors with n_lists=1000, nprobe=32: scans ~3.2% of vectors.
    """

    def __init__(self, config: Optional[IVFConfig] = None):
        self.config = config or IVFConfig()
        self.centroids: Optional[np.ndarray] = None  # (n_lists, D)
        self.inverted_lists: Dict[int, List[int]] = {}  # cluster_id → [doc_indices]
        self.is_trained: bool = False
        self._n_vectors: int = 0

    @property
    def n_lists(self) -> int:
        return self.config.n_lists

    @property
    def nprobe(self) -> int:
        return self.config.nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        self.config.nprobe = max(1, min(value, self.config.n_lists))

    def train(self, vectors: np.ndarray) -> None:
        """
        Trains the index by running k-means on the given vectors.
        Should be called once with a representative sample of the corpus.
        """
        n, dim = vectors.shape
        actual_n_lists = min(self.config.n_lists, n)

        # Subsample for training if dataset is very large
        if n > self.config.train_sample_size:
            rng = np.random.default_rng(self.config.seed)
            indices = rng.choice(n, size=self.config.train_sample_size, replace=False)
            train_vectors = vectors[indices]
        else:
            train_vectors = vectors

        self.centroids, _ = _kmeans(
            train_vectors,
            n_clusters=actual_n_lists,
            max_iters=self.config.kmeans_iters,
            seed=self.config.seed
        )

        # Initialize empty inverted lists
        self.inverted_lists = {i: [] for i in range(len(self.centroids))}
        self.is_trained = True
        self._n_vectors = 0

    def add(self, vectors: np.ndarray, start_idx: int = 0) -> None:
        """
        Assigns vectors to their nearest centroid's inverted list.

        Args:
            vectors: (N, D) float32 matrix.
            start_idx: The global document index offset for these vectors.
        """
        if not self.is_trained:
            raise RuntimeError("IVFIndex must be trained before adding vectors. Call .train() first.")

        n = vectors.shape[0]
        # Compute nearest centroid for each vector
        # Using squared Euclidean for assignment (equivalent ranking to L2)
        v_sq = np.sum(vectors * vectors, axis=1, keepdims=True)
        c_sq = np.sum(self.centroids * self.centroids, axis=1, keepdims=True)
        cross = vectors @ self.centroids.T
        dists = v_sq + c_sq.T - 2.0 * cross
        assignments = np.argmin(dists, axis=1)

        for i in range(n):
            cluster_id = int(assignments[i])
            self.inverted_lists[cluster_id].append(start_idx + i)

        self._n_vectors += n

    def search(
        self,
        query: np.ndarray,
        all_vectors: np.ndarray,
        k: int,
        metric: Metric,
        is_normalized: bool = False,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the nprobe nearest clusters for the top-k results.

        Args:
            query: (D,) query vector.
            all_vectors: (N, D) full vector matrix (for distance computation).
            k: Number of results to return.
            metric: Distance metric to use.
            is_normalized: Whether vectors are pre-normalized (for cosine).
            mask: Optional (N,) boolean mask from metadata filtering.

        Returns:
            indices: (k,) int array of document indices sorted by score (descending).
            scores: (k,) float array of corresponding scores.
        """
        if not self.is_trained or self.centroids is None:
            raise RuntimeError("IVFIndex must be trained before searching.")

        q = np.asarray(query, dtype=np.float32).ravel()

        # Step 1: Find nprobe nearest centroids
        centroid_scores = compute_distances(q, self.centroids, metric=metric, is_normalized=False)
        nprobe = min(self.config.nprobe, len(self.centroids))
        top_clusters = np.argsort(-centroid_scores)[:nprobe]

        # Step 2: Collect candidate doc indices from those clusters
        candidate_indices: List[int] = []
        for cluster_id in top_clusters:
            candidate_indices.extend(self.inverted_lists.get(int(cluster_id), []))

        if not candidate_indices:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        candidate_indices_arr = np.array(candidate_indices, dtype=np.int64)

        # Step 3: Apply metadata mask if provided
        if mask is not None:
            valid = mask[candidate_indices_arr]
            candidate_indices_arr = candidate_indices_arr[valid]

        if len(candidate_indices_arr) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        # Step 4: Compute exact distances for candidates only
        candidate_vectors = all_vectors[candidate_indices_arr]
        scores = compute_distances(q, candidate_vectors, metric=metric, is_normalized=is_normalized)

        # Step 5: Top-k selection
        actual_k = min(k, len(scores))
        top_local = np.argsort(-scores)[:actual_k]

        result_indices = candidate_indices_arr[top_local]
        result_scores = scores[top_local]

        return result_indices, result_scores

    def rebuild(self, vectors: np.ndarray) -> None:
        """Convenience: retrain + re-add all vectors from scratch."""
        self.train(vectors)
        self.add(vectors, start_idx=0)

    def stats(self) -> Dict:
        """Returns index statistics."""
        if not self.is_trained:
            return {"trained": False, "n_vectors": 0}

        list_sizes = [len(v) for v in self.inverted_lists.values()]
        return {
            "trained": True,
            "n_lists": len(self.centroids),
            "nprobe": self.config.nprobe,
            "n_vectors": self._n_vectors,
            "avg_list_size": round(np.mean(list_sizes), 1) if list_sizes else 0,
            "max_list_size": max(list_sizes) if list_sizes else 0,
            "min_list_size": min(list_sizes) if list_sizes else 0,
            "empty_lists": sum(1 for s in list_sizes if s == 0),
        }


def auto_n_lists(n_vectors: int) -> int:
    """Heuristic: sqrt(N) clamped to [16, 4096]."""
    return max(16, min(4096, int(math.sqrt(n_vectors))))
