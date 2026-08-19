"""
test_tda_core.py — CPU tests for the TDA core (synthetic data, no GPU/models)
==============================================================================

Run:  venv\\Scripts\\python.exe tda_analysis\\tests\\test_tda_core.py

Checks:
  - kNN graph and H_0 union-find on two clusters (b_0=2; in a sparse kNN VR
    distant clusters do not merge — sparsification semantics);
  - GUDHI vs union-find cross-check of b_0 on a grid of scales;
  - circle: the H_1 cycle lives at intermediate scale, persistence ~ diameter;
  - cone (contractible cloud): H_1 persistences are small compared to the circle;
  - W_1: identical diagrams -> 0; one point vs empty -> p/2; triangle sanity check;
  - effective_rank on a matrix of known rank ~= k; d90 on a constructed spectrum;
  - topo-rank of two clusters = 2 (stays 2 at all scales due to sparsification);
  - performance of persistence_gudhi at working size (n=256, d=32).
"""

import sys
import time
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent))

import tda_core as T                                  # noqa: E402
import tda_log                                        # noqa: E402


def _two_clusters(n_per=8, sep=10.0, seed=0):
    rng = np.random.default_rng(seed)
    c1 = rng.normal(0, 0.05, size=(n_per, 3))
    c2 = rng.normal(0, 0.05, size=(n_per, 3)) + sep
    return np.vstack([c1, c2]), sep


def _n_components(n, edge_idx):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edge_idx.tolist():
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[rj] = ri
    return len({find(x) for x in range(n)})


def _circle(n=64, r=1.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def _cone(n=200, seed=1):
    """Points on a cone (contractible): (r cos t, r sin t, r)."""
    rng = np.random.default_rng(seed)
    rr = np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([rr * np.cos(th), rr * np.sin(th), rr])


# ── kNN / H_0 union-find ───────────────────────────────────────

def test_knn_edges_two_clusters():
    X, sep = _two_clusters()
    ei, ew = T.knn_edges(X, k=4)
    assert ei.shape[1] == 2 and len(ei) > 0
    # all edges are inside the clusters (weights << inter-cluster distance)
    assert ew.max() < sep / 2, f"inter-cluster edge: {ew.max()} vs sep={sep}"
    # i < j in each edge
    assert np.all(ei[:, 0] < ei[:, 1])


def test_h0_unionfind_two_clusters():
    X, sep = _two_clusters()
    n = X.shape[0]
    ei, ew = T.knn_edges(X, k=4)
    c = _n_components(n, ei)
    assert c == 2, f"kNN graph must have 2 components, not {c}"
    ivs = T.h0_intervals_unionfind(n, ei, ew)
    # n - c finite intervals (one per successful merge) + c essential ones
    # (one per component of the kNN graph — matches the GUDHI counter)
    assert len(ivs) == n, f"len={len(ivs)}, expected {n}"
    finite = [d for _, d in ivs if np.isfinite(d)]
    assert len(finite) == n - c
    # Semantics of kNN sparsification: sep >> kNN radius, there are NO edges between clusters,
    # so the components never merge — b_0 stays 2 at all scales.
    assert max(finite) < sep / 2, f"inter-cluster merge: {max(finite)}, sep={sep}"
    assert T.betti_at_scale(ivs, 0.5) == c
    assert T.betti_at_scale(ivs, sep + 1.0) == c


def test_gudhi_h0_matches_unionfind():
    X, _ = _two_clusters(n_per=10)
    ei, ew = T.knn_edges(X, k=6)
    uf = T.h0_intervals_unionfind(X.shape[0], ei, ew)
    g = T.persistence_gudhi(X, k=6)
    for s in (0.2, 0.5, 1.0, 3.0, 8.0, 12.0):
        b_uf = T.betti_at_scale(uf, s)
        b_gu = T.betti_at_scale(g["h0"], s)
        assert b_uf == b_gu, f"s={s}: union-find={b_uf}, gudhi={b_gu}"


def test_topo_rank_two_clusters():
    X, _ = _two_clusters()
    ei, ew = T.knn_edges(X, k=4)
    h0 = T.h0_intervals_unionfind(X.shape[0], ei, ew)
    g = T.persistence_gudhi(X, k=4)
    # In a sparse kNN VR there are no edges between clusters -> components never
    # merge (sparsification semantics): topo-rank stays 2 at all scales.
    assert T.topo_rank({"0": h0, "1": g["h1"]}, 0.5) == 2   # two components, no cycles
    assert T.topo_rank({"0": h0, "1": g["h1"]}, 999.0) == 2


# ── H_1: circle vs cone ───────────────────────────────────────

def test_circle_h1():
    # Full VR (k = n-1 -> complete graph): on a small synthetic cloud
    # sparsification is not needed; then the H_1 cycle dies at ~diameter (the disk
    # fills up). In a kNN-sparse VR the same class becomes essential
    # (inf) — an artifact of sparsification, see test_h0_unionfind_two_clusters.
    X = _circle(n=64)
    g = T.persistence_gudhi(X, k=X.shape[0] - 1)
    assert len(g["h1"]) > 0, "the circle must have an H_1 class"
    # the cycle lives at intermediate scale (NN spacing ~ 2*pi/64 ~ 0.1)
    assert T.betti_at_scale(g["h1"], 0.5) == 1, f"h1={g['h1']}"
    max_pers = max(d - b for b, d in g["h1"] if np.isfinite(d))
    assert max_pers > 0.5, f"cycle persistence is small: {max_pers}"


def test_cone_contractible_vs_circle():
    # Full VR (k = n-1): the circle has one large finite H_1 class
    # (persistence ~ diameter); the cone is contractible -> all H_1 are short.
    Xc = _circle(n=64)
    Xk = _cone(n=80)   # C(80,3)=82160 < the limit of 300k triangles
    gc = T.persistence_gudhi(Xc, k=Xc.shape[0] - 1)
    gk = T.persistence_gudhi(Xk, k=Xk.shape[0] - 1)
    pers_c = max((d - b for b, d in gc["h1"] if np.isfinite(d)), default=0.0)
    pers_k = max((d - b for b, d in gk["h1"] if np.isfinite(d)), default=0.0)
    # the cone is contractible: H_1 persistences are noticeably smaller than for the circle
    assert pers_c > 0.5 and pers_k < 0.3 * pers_c, f"cone={pers_k}, circle={pers_c}"


# ── W_1 ────────────────────────────────────────────────────────

def test_w1_identical_zero():
    D = [(0.0, 0.3), (0.1, 2.5), (0.4, 0.9)]
    assert abs(T.wasserstein1(D, list(reversed(D)))) < 1e-9


def test_w1_single_vs_empty():
    # one point (b,d) against an empty diagram: falls to the diagonal at cost p/2
    v = T.wasserstein1([(0.0, 2.0)], [])
    assert abs(v - 1.0) < 1e-9, f"W1={v}, expected 1.0 (=p/2)"


def test_w1_empty_vs_empty():
    assert T.wasserstein1([], []) == 0.0


def test_w1_triangle_sanity():
    # A=(0,3), B=(1,4): L_inf=1; both to the diagonal = 1.5+1.5=3 -> min = 1
    v = T.wasserstein1([(0.0, 3.0)], [(1.0, 4.0)])
    assert abs(v - 1.0) < 1e-9, f"W1={v}"


def test_w1_trim_semantics():
    # A: one large (p=2) + 100 small ones (p=0.05); B: only the large one.
    # After trim to top-64: A has 64 points, B has 1. The large ones match (0),
    # the 63 extra small ones fall to the diagonal at p/2 = 0.025 -> W1 = 63*0.025.
    # (This is standard W_1 semantics: unmatched features cost p/2.)
    A = [(0.0, 2.0)] + [(float(i) * 0.001, float(i) * 0.001 + 0.05) for i in range(100)]
    B = [(0.0, 2.0)]
    v = T.wasserstein1(A, B, k=64)
    assert abs(v - 63 * 0.025) < 1e-9, f"W1={v}, expected {63*0.025}"


def test_w1_trim_drops_low_persistence_features():
    # A: 100 features with p_i = 0.01*(i+1); B: empty. Each unmatched feature
    # costs p/2 -> without trim W1 = sum(p)/2; with top-64 only i>=37 are counted.
    A = [(float(i) * 0.5, float(i) * 0.5 + 0.01 * (i + 1)) for i in range(100)]
    B = []
    v_full = T.wasserstein1(A, B, k=10**6)
    v_trim = T.wasserstein1(A, B, k=64)
    p_all = [0.01 * (i + 1) for i in range(100)]
    exp_full = sum(p_all) / 2                      # 25.25
    exp_trim = sum(sorted(p_all, reverse=True)[:64]) / 2   # top-64 by persistence
    assert abs(v_full - exp_full) < 1e-9, f"full: {v_full} vs {exp_full}"
    assert abs(v_trim - exp_trim) < 1e-9, f"trim: {v_trim} vs {exp_trim}"
    assert v_trim < v_full


# ── ranks (v5 conventions) ─────────────────────────────────────

def test_effective_rank_known():
    # Flat spectrum: U @ V^T (random orthonormal bases) -> exactly k unit sigmas.
    # p_i = 1/k for each -> eff_rank = exp(H(uniform_k)) = exactly k.
    rng = np.random.default_rng(7)
    k, m, p_ = 12, 60, 40
    U, _ = np.linalg.qr(rng.normal(size=(m, k)))
    V, _ = np.linalg.qr(rng.normal(size=(p_, k)))
    A = U @ V.T + rng.normal(size=(m, p_)) * 1e-9
    s = np.linalg.svd(A, compute_uv=False)
    er = T.effective_rank(s)
    assert abs(er - k) < 0.5, f"eff_rank={er}, expected ~{k}"


def test_effective_rank_flat_vs_spiky():
    # Uniform spectrum (10 unit sigmas) -> eff_rank ~ 10;
    # spiky spectrum (one large sigma) -> eff_rank ~ 1.
    flat = np.ones(10)
    spiky = np.array([10.0] + [0.01] * 9)
    assert T.effective_rank(flat) > 8.0, f"flat: {T.effective_rank(flat)}"
    assert T.effective_rank(spiky) < 2.0, f"spiky: {T.effective_rank(spiky)}"


def test_d90_constructed():
    # p = [0.5, 0.2, 0.15, 0.1, 0.05] -> cumsum: .5,.7,.85,.95 -> d90=4
    s = np.sqrt(np.array([0.5, 0.2, 0.15, 0.1, 0.05]))
    assert T.d90(s) == 4, f"d90={T.d90(s)}"


def test_eps_star_two_clusters():
    X, sep = _two_clusters()
    eps = T.eps_star(X, k=4)
    assert 0 < eps < sep / 2, f"eps*={eps}, sep={sep}"


# ── performance at working size ───────────────────────────────

def test_persistence_runtime_workload_size():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(256, 32))
    t0 = time.time()
    g = T.persistence_gudhi(X, k=32)
    dt = time.time() - t0
    assert len(g["h0"]) >= 1 and g["n_edges"] > 0
    print(f"    [runtime] persistence_gudhi n=256,d=32,k=32: {dt:.1f}s "
          f"(edges={g['n_edges']}, tris={g['n_triangles']})")
    assert dt < 120, f"GUDHI too slow for the workload: {dt:.1f}s"


# ── logging (Tee) ───────────────────────────────────────────────

def test_tee_writes_to_all_streams():
    import io
    a, b = io.StringIO(), io.StringIO()
    tda_log.Tee(a, b).write("hello\n")
    assert a.getvalue() == "hello\n" and b.getvalue() == "hello\n"


def test_tee_survives_broken_stream():
    import io

    class Broken:
        def write(self, s):
            raise OSError("console gone")

    ok = io.StringIO()
    tda_log.Tee(Broken(), ok).write("x\n")   # must not crash
    assert ok.getvalue() == "x\n"


def test_log_file_path_shape():
    p = tda_log.log_file_for("collect", "smollm-135m", base_dir=Path("/tmp/tda"))
    assert p.parent.name == "logs"
    assert p.name.startswith("collect_smollm-135m_") and p.suffix == ".log"


# ── runner ─────────────────────────────────────────────────────

def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
