"""
tda_core.py — TDA math for topology analysis of CHMC v6 activations/weights
=============================================================================

CPU-only. Dependencies: numpy, scipy, gudhi (3.x).

What is here:
  - sparse Vietoris-Rips complex on the kNN graph (k≈32) instead of full O(n^2);
    triangles = cliques of size 3 in the kNN graph (standard VR sparsification,
    preserving H_0/H_1 at scales up to ~the kNN radius).
  - H_0: EXACT via union-find over the edges of the kNN graph + cross-check
    with GUDHI (SimplexTree.compute_persistence, assign_filtration = max of vertex/edge weights).
  - H_1: GUDHI SimplexTree on the clique complex up to dimension 2.
    Filtration: vertex=0, edge=d(i,j), triangle=max(of the three edges) — canonical VR filtration.
  - Betti at scale eps*: number of open intervals [b, d) at s=eps*.
    eps* = median nearest-neighbour distance in the cloud (typical data scale).
  - topo-rank(eps*) = b_0(eps*) + b_1(eps*) — "topological rank" from the plan:
    cone/contractible cloud -> only H_0 -> topo-rank=1; two clusters -> 2;
    circle -> b_0+b_1 = 2.
  - W_1 between persistence diagrams: exact computation on trimmed diagrams
    (top-K by persistence, K=64) via scipy.linear_sum_assignment with the L_inf metric;
    matching a point (b,d) to the diagonal costs p/2, where p = d - b.
    Essential classes (death=inf) are excluded before computing W_1 (standard practice).
  - effective_rank / d90 — same conventions as in v5/generate_cov_stats.py and v5/rank_gap.py:
    p_i = s_i^2 / sum(s_j^2); eff_rank = exp(H(p)); d90 = min k with cumsum(p) >= 0.90.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

# Interval: (birth, death), death may be inf.
Interval = Tuple[float, float]


# ──────────────────────────────────────────────────────────────
# kNN graph and sparse VR complex
# ──────────────────────────────────────────────────────────────

def knn_edges(X: np.ndarray, k: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Undirected edges of the kNN graph.

    Returns (edge_idx [E x 2], edge_w [E]) — i < j, weight = Euclidean distance.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = X.shape[0]
    kq = min(k + 1, n)
    tree = cKDTree(X)
    dist, idx = tree.query(X, k=kq)
    seen = set()
    edges_i, edges_j, edges_w = [], [], []
    for r in range(n):
        for j, d in zip(idx[r], dist[r]):
            if j == r:
                continue
            a, b = (int(r), int(j)) if int(r) < int(j) else (int(j), int(r))
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            edges_i.append(a)
            edges_j.append(b)
            edges_w.append(float(d))
    return np.array([edges_i, edges_j]).T.astype(np.int64), np.asarray(edges_w, dtype=np.float64)


def vr_triangles(edge_idx: np.ndarray, edge_w: np.ndarray,
                 max_triangles: int = 300_000) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Triangles of the VR complex = cliques of size 3 in the kNN graph.

    Triangle weight = max(of the three edges). Returns (tri_idx [T x 3], tri_w [T], truncated).
    """
    adj: Dict[int, List[Tuple[int, float]]] = {}
    for (i, j), w in zip(edge_idx.tolist(), edge_w.tolist()):
        adj.setdefault(i, []).append((j, w))
        adj.setdefault(j, []).append((i, w))

    edge_set = set(map(tuple, edge_idx.tolist()))
    tri_i: List[Tuple[int, int, int]] = []
    seen = set()
    truncated = False
    for u in sorted(adj):
        nbrs = sorted(v for v, _ in adj[u])
        L = len(nbrs)
        for a in range(L):
            va = nbrs[a]
            if va <= u:
                continue
            for b in range(a + 1, L):
                vb = nbrs[b]
                if vb <= va:
                    continue
                if (va, vb) not in edge_set:
                    continue
                key = (u, va, vb)
                if key in seen:
                    continue
                seen.add(key)
                tri_i.append(key)
                if len(tri_i) >= max_triangles:
                    truncated = True
                    break
            if truncated:
                break
        if truncated:
            break

    if not tri_i:
        return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.float64), truncated
    A = np.array(tri_i, dtype=np.int64)
    ew = {}
    for (i, j), w in zip(edge_idx.tolist(), edge_w.tolist()):
        ew[(min(i, j), max(i, j))] = float(w)

    def ewget(a, b):
        return ew[(a, b)] if a < b else ew[(b, a)]

    tri_w = np.array([max(ewget(a, b), ewget(b, c), ewget(c, a)) for (a, b, c) in A],
                     dtype=np.float64)
    return A, tri_w, truncated


# ──────────────────────────────────────────────────────────────
# H_0: union-find (exact) + GUDHI (cross-check)
# ──────────────────────────────────────────────────────────────

def h0_intervals_unionfind(n_points: int, edge_idx: np.ndarray,
                           edge_w: np.ndarray) -> List[Interval]:
    """Exact H_0 intervals of the VR complex (all births = 0).

    n - c finite intervals (0, w_merge), where c = number of components of the edge graph,
    + c essential ones (0, inf) — one per component. In a sparse kNN VR
    distant clusters are not connected by edges and stay separate forever;
    matches GUDHI (which also has one essential class per component).
    """
    parent = list(range(n_points))
    size = [1] * n_points

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    intervals: List[Interval] = []
    order = np.argsort(edge_w, kind="stable")
    for pos in order:
        i, j = int(edge_idx[pos, 0]), int(edge_idx[pos, 1])
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        w = float(edge_w[pos])
        if size[ri] > size[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        size[ri] += size[rj]
        intervals.append((0.0, w))
    n_essential = len({find(x) for x in range(n_points)})
    intervals.extend([(0.0, math.inf)] * n_essential)
    return intervals


def _gudhi_intervals(tree, dim: int) -> List[Interval]:
    raw = tree.persistence_intervals_in_dimension(dim)
    out: List[Interval] = []
    for iv in raw:
        b, d = float(iv[0]), float(iv[1])
        if not math.isfinite(d):
            d = math.inf
        out.append((b, d))
    return out


def persistence_gudhi(X: np.ndarray, k: int = 32,
                      max_triangles: int = 300_000) -> Dict[str, object]:
    """Full H_0/H_1 persistence of the sparse VR complex via GUDHI.

    Filtration: vertex=0, edge=d(i,j), triangle=max(of the edges).
    """
    import gudhi  # local import: tests may skip gudhi-dependent cases

    edge_idx, edge_w = knn_edges(X, k)
    tri_idx, tri_w, truncated = vr_triangles(edge_idx, edge_w, max_triangles)

    tree = gudhi.SimplexTree()
    n = X.shape[0]
    # insert_batch(vertex_array (k+1 x N), filtrations (N,)); faces are inserted automatically.
    tree.insert_batch(np.arange(n).reshape(1, -1).astype(np.int64), np.zeros(n))
    if len(edge_idx):
        tree.insert_batch(edge_idx.T.astype(np.int64), edge_w)
    if len(tri_idx):
        tree.insert_batch(tri_idx.T.astype(np.int64), tri_w)

    # persistence_dim_max=True: H_1 is computed even for purely graph complexes.
    tree.compute_persistence(persistence_dim_max=True)
    h0 = _gudhi_intervals(tree, 0)
    h1 = _gudhi_intervals(tree, 1)
    return {
        "h0": h0,
        "h1": h1,
        "n_edges": int(len(edge_idx)),
        "n_triangles": int(len(tri_idx)),
        "triangles_truncated": bool(truncated),
    }


# ──────────────────────────────────────────────────────────────
# Betti at scale, topo-rank, eps*
# ──────────────────────────────────────────────────────────────

def betti_at_scale(intervals: Sequence[Interval], s: float) -> int:
    """Number of classes alive at scale s: b <= s < d."""
    return sum(1 for (b, d) in intervals if b <= s < d)


def eps_star(X: np.ndarray, k: int = 32) -> float:
    """Typical data scale: median nearest-neighbour distance."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = X.shape[0]
    tree = cKDTree(X)
    d, _ = tree.query(X, k=min(2, n))
    nn = d[:, 1] if n > 1 else np.array([0.0])
    med = float(np.median(nn))
    return med if med > 0 else float(np.mean(nn)) if len(nn) and np.mean(nn) > 0 else 1e-9


def topo_rank(intervals_by_dim: Dict[int, Sequence[Interval]], s: float) -> int:
    """topo-rank(s) = sum_k b_k(s) over the available dimensions (H_0 + H_1)."""
    return sum(betti_at_scale(iv, s) for iv in intervals_by_dim.values())


# ──────────────────────────────────────────────────────────────
# W_1 between diagrams (trimmed, top-K by persistence)
# ──────────────────────────────────────────────────────────────

def _trim(diagram: Sequence[Interval], k: int = 64) -> np.ndarray:
    """Top-K intervals by persistence; essential ones (inf) are dropped.

    Returns an array [M x 2] (birth, death); may be empty.
    """
    finite = [(b, d) for (b, d) in diagram if math.isfinite(d)]
    if not finite:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.array(finite, dtype=np.float64)
    pers = arr[:, 1] - arr[:, 0]
    order = np.argsort(pers)[::-1][:k]
    return arr[order]


def wasserstein1(diag_a: Sequence[Interval], diag_b: Sequence[Interval],
                 k: int = 64) -> float:
    """W_1 between two diagrams on trimmed top-K (L_inf metric).

    Exact minimum matching with the option to "fall to the diagonal"
    at cost p/2. If after trimming n != m, the extra points are matched to the diagonal.
    """
    A = _trim(diag_a, k)
    B = _trim(diag_b, k)
    n, m = len(A), len(B)
    if n == 0 and m == 0:
        return 0.0

    def pa(P):
        return (P[:, 1] - P[:, 0]) / 2.0

    # extended cost matrix of size (n+m) x (n+m)
    N = n + m
    C = np.zeros((N, N), dtype=np.float64)
    if n and m:
        # L_inf between points of the two diagrams
        diff = np.abs(A[:, None, :] - B[None, :, :])
        C[:n, :m] = diff.max(axis=2)
    if n:
        C[:n, m:] = pa(A)[:, None]      # A -> diagonal
    if m:
        C[n:, :m] = pa(B)[None, :]      # B -> diagonal
    row, col = linear_sum_assignment(C)
    return float(C[row, col].sum())


# ──────────────────────────────────────────────────────────────
# Ranks (v5 conventions: exp-entropy of normalized sigma^2)
# ──────────────────────────────────────────────────────────────

def effective_rank(s: np.ndarray) -> float:
    """effective_rank = exp(entropy(p)), p_i = s_i^2 / sum(s_j^2)."""
    s = np.asarray(s, dtype=np.float64).ravel()
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    p = (s ** 2) / float((s ** 2).sum())
    nz = p[p > 0]
    return float(math.exp(-float(np.sum(nz * np.log(nz)))))


def d90(s: np.ndarray) -> int:
    """Minimum number of principal components covering 90% of the sigma^2 energy."""
    s = np.asarray(s, dtype=np.float64).ravel()
    s = s[s > 0]
    if s.size == 0:
        return 0
    p = (s ** 2) / float((s ** 2).sum())
    k = int(np.searchsorted(np.cumsum(p), 0.90)) + 1
    return min(k, len(p))


def activation_pca_d90(X: np.ndarray) -> Tuple[int, float]:
    """d90 and effective_rank from the singular values of centered activations."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    return d90(s), effective_rank(s)
