"""
Approximate Nearest Neighbor (ANN) indexes for SutraDB:
1. IVF-Flat (Inverted File with flat inner-cluster exact search)
2. HNSW (Hierarchical Navigable Small World multi-layer graph with masked filtered traversal and online node deletion)
"""

import heapq
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from sutradb.distance import Metric, compute_distances, normalize_matrix, normalize_vector


# ──────────────────────────────────────────────────────────────────────────────
# k-means++ Implementation
# ──────────────────────────────────────────────────────────────────────────────

def _kmeans_plus_plus_init(
    vectors: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    use_cosine: bool = False,
) -> np.ndarray:
    """
    k-means++ centroid initialization.
    Selects initial centroids with probability proportional to distance
    from the nearest existing centroid, ensuring well-spread initial clusters.
    """
    n, dim = vectors.shape
    centroids = np.empty((n_clusters, dim), dtype=np.float32)

    # First centroid: random
    idx = int(rng.integers(0, n))
    centroids[0] = vectors[idx]

    # Subsequent centroids: D² weighted sampling
    for k in range(1, n_clusters):
        dists = np.full(n, np.inf, dtype=np.float32)
        for j in range(k):
            if use_cosine:
                d2 = np.maximum(0.0, 1.0 - (vectors @ centroids[j]))
            else:
                diff = vectors - centroids[j]
                d2 = np.sum(diff * diff, axis=1)
            dists = np.minimum(dists, d2)

        total = float(dists.sum())
        if not math.isfinite(total) or total <= 1e-30:
            probs = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            probs = dists.astype(np.float64) / total
            probs /= probs.sum()
        idx = int(rng.choice(n, p=probs))
        centroids[k] = vectors[idx]

    return centroids


def _kmeans(
    vectors: np.ndarray,
    n_clusters: int,
    max_iters: int = 25,
    tol: float = 1e-4,
    seed: int = 42,
    use_cosine: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lloyd's k-means (or spherical k-means when use_cosine=True) with k-means++ initialization.
    """
    n, dim = vectors.shape
    n_clusters = min(n_clusters, n)
    rng = np.random.default_rng(seed)

    work_vectors = normalize_matrix(vectors) if use_cosine else np.asarray(vectors, dtype=np.float32)

    # Initialize centroids with k-means++
    centroids = _kmeans_plus_plus_init(work_vectors, n_clusters, rng, use_cosine=use_cosine)
    assignments = np.zeros(n, dtype=np.int32)

    for _ in range(max_iters):
        if use_cosine:
            sims = work_vectors @ centroids.T
            new_assignments = np.argmax(sims, axis=1).astype(np.int32)
        else:
            v_sq = np.sum(work_vectors * work_vectors, axis=1, keepdims=True)
            c_sq = np.sum(centroids * centroids, axis=1, keepdims=True)
            cross = work_vectors @ centroids.T
            dists = v_sq + c_sq.T - 2.0 * cross
            new_assignments = np.argmin(dists, axis=1).astype(np.int32)

        # Update step: recompute centroids as cluster means
        old_centroids = centroids.copy()
        for k in range(n_clusters):
            members = work_vectors[new_assignments == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
            else:
                largest_cluster = np.bincount(new_assignments, minlength=n_clusters).argmax()
                members_of_largest = np.where(new_assignments == largest_cluster)[0]
                random_idx = int(rng.choice(members_of_largest))
                centroids[k] = work_vectors[random_idx]

        if use_cosine:
            centroids = normalize_matrix(centroids)

        assignments = new_assignments

        shift = float(np.linalg.norm(centroids - old_centroids))
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
    Supports metric-aligned Voronoi partitioning (spherical k-means for COSINE).
    """

    def __init__(
        self,
        config: Optional[IVFConfig] = None,
        metric: Union[Metric, str] = Metric.COSINE,
    ):
        self.config = config or IVFConfig()
        self.metric: Metric = Metric(metric) if isinstance(metric, str) else metric
        self.centroids: Optional[np.ndarray] = None  # (n_lists, D)
        self.inverted_lists: Dict[int, List[int]] = {}  # cluster_id -> [doc_indices]
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

    def train(self, vectors: np.ndarray, metric: Optional[Union[Metric, str]] = None) -> None:
        """
        Trains the index by running k-means on the given vectors.
        """
        if metric is not None:
            self.metric = Metric(metric) if isinstance(metric, str) else metric

        n, _ = vectors.shape
        if n == 0:
            raise ValueError("Cannot train IVFIndex on empty vectors")

        actual_n_lists = min(self.config.n_lists, n)

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
            seed=self.config.seed,
            use_cosine=(self.metric == Metric.COSINE),
        )

        self.inverted_lists = {i: [] for i in range(len(self.centroids))}
        self.is_trained = True
        self._n_vectors = 0

    def add(self, vectors: np.ndarray, start_idx: int = 0) -> None:
        """
        Assigns vectors to their nearest centroid's inverted list using sequential indices.
        """
        n = vectors.shape[0]
        self.add_with_indices(vectors, range(start_idx, start_idx + n))

    def add_with_indices(self, vectors: np.ndarray, indices: Sequence[int]) -> None:
        """
        Assigns vectors to their nearest centroid's inverted list under explicit document indices.
        """
        if not self.is_trained or self.centroids is None:
            raise RuntimeError("IVFIndex must be trained before adding vectors. Call .train() first.")

        n = vectors.shape[0]
        if n == 0:
            return
        if n != len(indices):
            raise ValueError(f"Vector count ({n}) does not match indices length ({len(indices)})")

        if self.metric == Metric.COSINE:
            v_norm = normalize_matrix(vectors)
            assignments = np.argmax(v_norm @ self.centroids.T, axis=1)
        else:
            v_sq = np.sum(vectors * vectors, axis=1, keepdims=True)
            c_sq = np.sum(self.centroids * self.centroids, axis=1, keepdims=True)
            cross = vectors @ self.centroids.T
            dists = v_sq + c_sq.T - 2.0 * cross
            assignments = np.argmin(dists, axis=1)

        for i, doc_idx in enumerate(indices):
            cluster_id = int(assignments[i])
            self.inverted_lists[cluster_id].append(int(doc_idx))

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
        """
        if not self.is_trained or self.centroids is None:
            raise RuntimeError("IVFIndex must be trained before searching.")
        if k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        q = np.asarray(query, dtype=np.float32).ravel()

        # Step 1: Find nprobe nearest centroids
        centroids_normalized = (self.metric == Metric.COSINE and metric == Metric.COSINE)
        centroid_scores = compute_distances(q, self.centroids, metric=metric, is_normalized=centroids_normalized)
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

    def rebuild(self, vectors: np.ndarray, metric: Optional[Union[Metric, str]] = None) -> None:
        """Convenience: retrain + re-add all vectors from scratch."""
        self.train(vectors, metric=metric)
        self.add(vectors, start_idx=0)

    def stats(self) -> Dict[str, Any]:
        """Returns index statistics."""
        if not self.is_trained or self.centroids is None:
            return {"trained": False, "n_vectors": 0, "metric": self.metric.value}

        list_sizes = [len(v) for v in self.inverted_lists.values()]
        return {
            "trained": True,
            "metric": self.metric.value,
            "n_lists": len(self.centroids),
            "nprobe": self.config.nprobe,
            "n_vectors": self._n_vectors,
            "avg_list_size": round(float(np.mean(list_sizes)), 1) if list_sizes else 0.0,
            "max_list_size": max(list_sizes) if list_sizes else 0,
            "min_list_size": min(list_sizes) if list_sizes else 0,
            "empty_lists": sum(1 for s in list_sizes if s == 0),
        }


# ──────────────────────────────────────────────────────────────────────────────
# HNSW Index
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HNSWConfig:
    """Configuration for the Hierarchical Navigable Small World (HNSW) index."""
    m: int = 16
    m0: Optional[int] = None
    ef_construction: int = 200
    ef_search: int = 50
    ml: Optional[float] = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.m < 2:
            raise ValueError(f"HNSW m must be >= 2, got {self.m}")
        if self.ef_construction < 1:
            raise ValueError(f"HNSW ef_construction must be >= 1, got {self.ef_construction}")
        if self.ef_search < 1:
            raise ValueError(f"HNSW ef_search must be >= 1, got {self.ef_search}")
        if self.m0 is None:
            self.m0 = self.m * 2
        if self.ml is None:
            self.ml = 1.0 / math.log(self.m)


class HNSWIndex:
    """
    Pure-Python + NumPy multi-layer HNSW graph index.
    Supports:
      - Metric-aware vectorized neighbor scoring (COSINE, EUCLIDEAN, DOT_PRODUCT)
      - Predicate-aware masked traversal (masked nodes traverse candidates without consuming ef results)
      - Disconnected-island fallback on filtered search
      - Online node deletion with asymmetric incoming edge cleanup and neighbor healing
    """

    def __init__(self, config: Optional[HNSWConfig] = None):
        self.config = config or HNSWConfig()
        self.entry_point: Optional[int] = None
        self.max_level: int = 0
        self.layers: List[Dict[int, List[int]]] = []
        self.node_levels: Dict[int, int] = {}
        self._rng = random.Random(self.config.seed)

    def _random_level(self) -> int:
        r = max(self._rng.random(), 1e-12)
        return min(int(math.floor(-math.log(r) * (self.config.ml or 1.0))), 16)

    @staticmethod
    def _prepare_query(q: np.ndarray, metric: Metric) -> np.ndarray:
        q_arr = np.asarray(q, dtype=np.float32).ravel()
        if metric == Metric.COSINE:
            return normalize_vector(q_arr)
        return q_arr

    @staticmethod
    def _score_batch(
        q_prepared: np.ndarray,
        node_ids: Sequence[int],
        all_vectors: np.ndarray,
        metric: Metric,
        is_normalized: bool,
    ) -> np.ndarray:
        if not node_ids:
            return np.empty(0, dtype=np.float32)
        vecs = all_vectors[list(node_ids)]
        if metric == Metric.COSINE:
            if is_normalized:
                return (vecs @ q_prepared).astype(np.float32, copy=False)
            return (normalize_matrix(vecs) @ q_prepared).astype(np.float32, copy=False)
        if metric == Metric.DOT_PRODUCT:
            return (vecs @ q_prepared).astype(np.float32, copy=False)
        # Metric.EUCLIDEAN
        dists = np.linalg.norm(vecs - q_prepared, axis=1)
        return (1.0 / (1.0 + dists)).astype(np.float32, copy=False)

    @classmethod
    def _select_neighbors_heuristic(
        cls,
        q_prepared: np.ndarray,
        candidates: List[Tuple[int, float]],
        m_max: int,
        all_vectors: np.ndarray,
        metric: Metric,
        is_normalized: bool,
    ) -> List[int]:
        """HNSW Algorithm 4 heuristic selection preserving diverse bridge edges across clusters."""
        if len(candidates) <= m_max:
            return [nid for nid, _ in candidates]

        c_ids = [nid for nid, _ in candidates]
        cand_vecs = all_vectors[c_ids]
        if metric == Metric.COSINE:
            if not is_normalized:
                cand_vecs = normalize_matrix(cand_vecs)
            sim_mat = cand_vecs @ cand_vecs.T
        elif metric == Metric.DOT_PRODUCT:
            sim_mat = cand_vecs @ cand_vecs.T
        else:
            diff = cand_vecs[:, None, :] - cand_vecs[None, :, :]
            sim_mat = 1.0 / (1.0 + np.linalg.norm(diff, axis=2))

        selected_pos: List[int] = []
        discarded_pos: List[int] = []
        for idx, (_, sc_q) in enumerate(candidates):
            if len(selected_pos) >= m_max:
                break
            if not selected_pos:
                selected_pos.append(idx)
                continue
            max_sel = max(float(sim_mat[idx, j]) for j in selected_pos)
            if sc_q >= max_sel:
                selected_pos.append(idx)
            else:
                discarded_pos.append(idx)

        if len(selected_pos) < m_max and discarded_pos:
            for idx in discarded_pos:
                if len(selected_pos) >= m_max:
                    break
                selected_pos.append(idx)

        return [c_ids[i] for i in selected_pos]

    def insert(
        self,
        doc_id: int,
        vector: np.ndarray,
        all_vectors: np.ndarray,
        metric: Metric = Metric.COSINE,
        is_normalized: bool = False,
    ) -> None:
        """Inserts or updates a node in the multi-layer HNSW graph."""
        if doc_id in self.node_levels:
            self.delete_node(doc_id, all_vectors, metric=metric, is_normalized=is_normalized)

        node_level = self._random_level()
        while len(self.layers) <= node_level:
            self.layers.append({})

        if self.entry_point is None:
            for level in range(node_level + 1):
                self.layers[level][doc_id] = []
            self.node_levels[doc_id] = node_level
            self.entry_point = doc_id
            self.max_level = node_level
            return

        q_prep = self._prepare_query(vector, metric)
        curr = self.entry_point
        curr_score = float(self._score_batch(q_prep, [curr], all_vectors, metric, is_normalized)[0])

        # Greedy top-down routing above node_level
        for level in range(self.max_level, node_level, -1):
            layer_map = self.layers[level]
            while True:
                neighbors = layer_map.get(curr, ())
                if not neighbors:
                    break
                scores = self._score_batch(q_prep, neighbors, all_vectors, metric, is_normalized)
                best_idx = int(np.argmax(scores))
                best_score = float(scores[best_idx])
                if best_score > curr_score:
                    curr = neighbors[best_idx]
                    curr_score = best_score
                else:
                    break

        self.node_levels[doc_id] = node_level
        enter_pts: List[int] = [curr]

        # Link bidirectionally at layers min(node_level, max_level) down to 0
        for level in range(min(node_level, self.max_level), -1, -1):
            candidates = self._search_layer(
                q_prep,
                enter_pts,
                self.config.ef_construction,
                level,
                all_vectors,
                metric,
                is_normalized,
                mask=None,
            )
            m_max = (self.config.m0 or self.config.m * 2) if level == 0 else self.config.m
            filtered_cands = [(nid, sc) for nid, sc in candidates if nid != doc_id]
            selected = self._select_neighbors_heuristic(
                q_prep, filtered_cands, m_max, all_vectors, metric, is_normalized
            )
            self.layers[level][doc_id] = list(selected)

            for nbr in selected:
                nbr_list = self.layers[level].setdefault(nbr, [])
                if doc_id not in nbr_list:
                    nbr_list.append(doc_id)
                if len(nbr_list) > m_max:
                    nbr_q = self._prepare_query(all_vectors[nbr], metric)
                    nbr_scores = self._score_batch(nbr_q, nbr_list, all_vectors, metric, is_normalized)
                    order = np.argsort(-nbr_scores)
                    ranked_cands = [(nbr_list[int(i)], float(nbr_scores[int(i)])) for i in order]
                    self.layers[level][nbr] = self._select_neighbors_heuristic(
                        nbr_q, ranked_cands, m_max, all_vectors, metric, is_normalized
                    )

            if candidates:
                enter_pts = [nid for nid, _ in candidates]

        for level in range(node_level + 1):
            self.layers[level].setdefault(doc_id, [])

        if node_level > self.max_level:
            self.max_level = node_level
            self.entry_point = doc_id

    def _search_layer(
        self,
        q_prepared: np.ndarray,
        enter_points: Sequence[int],
        ef: int,
        level: int,
        all_vectors: np.ndarray,
        metric: Metric,
        is_normalized: bool,
        mask: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, float]]:
        """
        Predicate-aware beam search within a single HNSW layer.
        Masked-out nodes remain traversable in `candidates` without consuming `ef` capacity in `results`.
        """
        if level < 0 or level >= len(self.layers):
            return []
        layer_map = self.layers[level]

        def is_valid(nid: int) -> bool:
            if mask is None:
                return True
            return 0 <= nid < len(mask) and bool(mask[nid])

        candidates: List[Tuple[float, int]] = []  # max-heap via negated score
        results: List[Tuple[float, int]] = []     # min-heap bounded by ef
        visited: Set[int] = set()

        valid_eps = []
        for ep in enter_points:
            if ep in layer_map and ep not in visited:
                visited.add(ep)
                valid_eps.append(ep)

        if valid_eps:
            ep_scores = self._score_batch(q_prepared, valid_eps, all_vectors, metric, is_normalized)
            for ep, sc in zip(valid_eps, ep_scores):
                sc_f = float(sc)
                heapq.heappush(candidates, (-sc_f, ep))
                if is_valid(ep):
                    heapq.heappush(results, (sc_f, ep))
                    if len(results) > ef:
                        heapq.heappop(results)

        while candidates:
            neg_sc, curr_id = heapq.heappop(candidates)
            curr_sc = -neg_sc
            if len(results) >= ef and curr_sc < results[0][0]:
                break

            neighbors = [n for n in layer_map.get(curr_id, ()) if n not in visited]
            if not neighbors:
                continue
            visited.update(neighbors)

            n_scores = self._score_batch(q_prepared, neighbors, all_vectors, metric, is_normalized)
            for nbr, sc in zip(neighbors, n_scores):
                sc_f = float(sc)
                worst = results[0][0] if len(results) >= ef else -math.inf
                if sc_f > worst or len(results) < ef:
                    heapq.heappush(candidates, (-sc_f, nbr))
                    if is_valid(nbr):
                        heapq.heappush(results, (sc_f, nbr))
                        if len(results) > ef:
                            heapq.heappop(results)

        return sorted(((nid, sc) for sc, nid in results), key=lambda x: x[1], reverse=True)

    def search(
        self,
        query: np.ndarray,
        all_vectors: np.ndarray,
        k: int,
        ef_search: Optional[int] = None,
        metric: Metric = Metric.COSINE,
        is_normalized: bool = False,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Searches the HNSW graph for up to k nearest valid neighbors."""
        if k <= 0 or self.entry_point is None or not self.layers:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        q_prep = self._prepare_query(query, metric)
        curr = self.entry_point
        curr_score = float(self._score_batch(q_prep, [curr], all_vectors, metric, is_normalized)[0])

        for level in range(self.max_level, 0, -1):
            layer_map = self.layers[level]
            while True:
                neighbors = layer_map.get(curr, ())
                if not neighbors:
                    break
                scores = self._score_batch(q_prep, neighbors, all_vectors, metric, is_normalized)
                best_idx = int(np.argmax(scores))
                best_score = float(scores[best_idx])
                if best_score > curr_score:
                    curr = neighbors[best_idx]
                    curr_score = best_score
                else:
                    break

        ef = max(ef_search if ef_search is not None else self.config.ef_search, k)
        results = self._search_layer(
            q_prep,
            [curr],
            ef,
            0,
            all_vectors,
            metric,
            is_normalized,
            mask=mask,
        )

        # Disconnected-island fallback for restrictive filters
        if mask is not None and len(results) < k and self.layers:
            layer0 = self.layers[0]
            seen = {nid for nid, _ in results}
            valid_nodes = [
                nid for nid in layer0
                if 0 <= nid < len(mask) and bool(mask[nid]) and nid not in seen
            ]
            if valid_nodes:
                sample_count = min(len(valid_nodes), max(ef, 32))
                if sample_count < len(valid_nodes):
                    step = max(1, len(valid_nodes) // sample_count)
                    seed_eps = valid_nodes[::step][:sample_count]
                else:
                    seed_eps = valid_nodes

                fallback_res = self._search_layer(
                    q_prep,
                    seed_eps,
                    ef,
                    0,
                    all_vectors,
                    metric,
                    is_normalized,
                    mask=mask,
                )
                merged: Dict[int, float] = {nid: sc for nid, sc in results}
                for nid, sc in fallback_res:
                    if nid not in merged or sc > merged[nid]:
                        merged[nid] = sc
                results = sorted(merged.items(), key=lambda x: x[1], reverse=True)

        top = results[:k]
        if not top:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        return (
            np.array([nid for nid, _ in top], dtype=np.int64),
            np.array([sc for _, sc in top], dtype=np.float32),
        )

    def delete_node(
        self,
        doc_id: int,
        all_vectors: np.ndarray,
        metric: Metric = Metric.COSINE,
        is_normalized: bool = False,
    ) -> bool:
        """
        Removes a node from all graph layers, purges all incoming edges to it,
        and reconnects under-linked neighbors.
        """
        if doc_id not in self.node_levels:
            return False

        self.node_levels.pop(doc_id, None)
        min_links = max(1, self.config.m // 2)

        for level_idx, layer_map in enumerate(self.layers):
            old_neighbors = layer_map.pop(doc_id, [])
            affected: Set[int] = set()

            # Purge ALL incoming references to doc_id in this layer
            for nid, nbrs in layer_map.items():
                if doc_id in nbrs:
                    layer_map[nid] = [x for x in nbrs if x != doc_id]
                    affected.add(nid)

            for nid in old_neighbors:
                if nid in layer_map:
                    affected.add(nid)

            m_max = (self.config.m0 or self.config.m * 2) if level_idx == 0 else self.config.m
            valid_candidates = [nid for nid in old_neighbors if nid in layer_map]

            for nbr in affected:
                nbr_list = layer_map.get(nbr)
                if nbr_list is None or len(nbr_list) >= min_links:
                    continue
                pool = [c for c in valid_candidates if c != nbr and c not in nbr_list]
                if not pool:
                    continue
                q_nbr = self._prepare_query(all_vectors[nbr], metric)
                scores = self._score_batch(q_nbr, pool, all_vectors, metric, is_normalized)
                order = np.argsort(-scores)
                for i in order:
                    if len(nbr_list) >= min_links:
                        break
                    cand = pool[int(i)]
                    nbr_list.append(cand)
                    cand_list = layer_map.get(cand)
                    if cand_list is not None and nbr not in cand_list and len(cand_list) < m_max:
                        cand_list.append(nbr)

        if self.entry_point == doc_id or self.entry_point not in self.node_levels:
            if not self.node_levels:
                self.entry_point = None
                self.max_level = 0
                self.layers.clear()
            else:
                best_id, best_lvl = max(self.node_levels.items(), key=lambda item: item[1])
                self.entry_point = best_id
                self.max_level = best_lvl
                while len(self.layers) > self.max_level + 1:
                    self.layers.pop()

        return True

    def stats(self) -> Dict[str, Any]:
        """Returns HNSW graph statistics."""
        return {
            "m": self.config.m,
            "m0": self.config.m0,
            "ef_construction": self.config.ef_construction,
            "ef_search": self.config.ef_search,
            "count": len(self.node_levels),
            "max_level": self.max_level,
            "entry_point": self.entry_point,
        }


def auto_n_lists(n_vectors: int) -> int:
    """Heuristic: sqrt(N) clamped to [16, 4096]."""
    return max(16, min(4096, int(math.sqrt(n_vectors))))
