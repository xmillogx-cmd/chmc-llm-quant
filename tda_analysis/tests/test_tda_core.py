"""
test_tda_core.py — CPU-тесты TDA-ядра (синтетические данные, без GPU/моделей)
==============================================================================

Запуск:  venv\\Scripts\\python.exe tda_analysis\\tests\\test_tda_core.py

Проверяет:
  - kNN-граф и H_0 union-find на двух кластерах (b_0=2; в разреженном kNN VR
    удалённые кластеры не сливаются — семантика спарсификации);
  - кроссчек GUDHI vs union-find по b_0 на сетке масштабов;
  - окружность: H_1 цикл живёт на среднем масштабе, персистентность ~ диаметр;
  - конус (контрактируемое облако): H_1-персистентности малы против окружности;
  - W_1: идентичные диаграммы -> 0; одна точка vs пусто -> p/2; треугольник-санитайз;
  - effective_rank на матрице известного ранга ~= k; d90 на сконструированном спектре;
  - topo-rank двух кластеров = 2 (остаётся 2 на всех масштабах из-за спарсификации);
  - производительность persistence_gudhi на рабочем размере (n=256, d=32).
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
    """Точки на конусе (контрактируемо): (r cos t, r sin t, r)."""
    rng = np.random.default_rng(seed)
    rr = np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([rr * np.cos(th), rr * np.sin(th), rr])


# ── kNN / H_0 union-find ───────────────────────────────────────

def test_knn_edges_two_clusters():
    X, sep = _two_clusters()
    ei, ew = T.knn_edges(X, k=4)
    assert ei.shape[1] == 2 and len(ei) > 0
    # все рёбра внутри кластеров (веса << межкластерного расстояния)
    assert ew.max() < sep / 2, f"межкластерное ребро: {ew.max()} vs sep={sep}"
    # i < j в каждом ребре
    assert np.all(ei[:, 0] < ei[:, 1])


def test_h0_unionfind_two_clusters():
    X, sep = _two_clusters()
    n = X.shape[0]
    ei, ew = T.knn_edges(X, k=4)
    c = _n_components(n, ei)
    assert c == 2, f"kNN-граф должен иметь 2 компоненты, а не {c}"
    ivs = T.h0_intervals_unionfind(n, ei, ew)
    # n - c конечных интервалов (по одному на успешное слияние) + c существенных
    # (по одному на компоненту kNN-графа — совпадает со счётчиком GUDHI)
    assert len(ivs) == n, f"len={len(ivs)}, ожидалось {n}"
    finite = [d for _, d in ivs if np.isfinite(d)]
    assert len(finite) == n - c
    # Семантика kNN-спарсификации: sep >> kNN-радиус, между кластерами рёбер НЕТ,
    # поэтому компоненты никогда не сливаются — b_0 остаётся 2 на всех масштабах.
    assert max(finite) < sep / 2, f"межкластерное слияние: {max(finite)}, sep={sep}"
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
    # В разреженном kNN VR между кластерами нет рёбер -> компоненты никогда не
    # сливаются (семантика спарсификации): topo-rank остаётся 2 на всех масштабах.
    assert T.topo_rank({"0": h0, "1": g["h1"]}, 0.5) == 2   # два компонента, нет циклов
    assert T.topo_rank({"0": h0, "1": g["h1"]}, 999.0) == 2


# ── H_1: окружность vs конус ───────────────────────────────────

def test_circle_h1():
    # Полный VR (k = n-1 -> полный граф): на маленьком синтетическом облаке
    # спарсификация не нужна; тогда цикл H_1 умирает при ~диаметре (диск
    # заполняется). В kNN-спарсированном VR тот же класс становится essential
    # (inf) — артефакт спарсификации, см. test_h0_unionfind_two_clusters.
    X = _circle(n=64)
    g = T.persistence_gudhi(X, k=X.shape[0] - 1)
    assert len(g["h1"]) > 0, "у окружности должен быть H_1-класс"
    # цикл жив на среднем масштабе (NN spacing ~ 2*pi/64 ~ 0.1)
    assert T.betti_at_scale(g["h1"], 0.5) == 1, f"h1={g['h1']}"
    max_pers = max(d - b for b, d in g["h1"] if np.isfinite(d))
    assert max_pers > 0.5, f"персистентность цикла мала: {max_pers}"


def test_cone_contractible_vs_circle():
    # Полный VR (k = n-1): у окружности один крупный конечный H_1-класс
    # (персистентность ~ диаметр); конус контрактируем -> все H_1 короткие.
    Xc = _circle(n=64)
    Xk = _cone(n=80)   # C(80,3)=82160 < лимита 300k треугольников
    gc = T.persistence_gudhi(Xc, k=Xc.shape[0] - 1)
    gk = T.persistence_gudhi(Xk, k=Xk.shape[0] - 1)
    pers_c = max((d - b for b, d in gc["h1"] if np.isfinite(d)), default=0.0)
    pers_k = max((d - b for b, d in gk["h1"] if np.isfinite(d)), default=0.0)
    # конус контрактируем: H_1-персистентности заметно меньше, чем у окружности
    assert pers_c > 0.5 and pers_k < 0.3 * pers_c, f"cone={pers_k}, circle={pers_c}"


# ── W_1 ────────────────────────────────────────────────────────

def test_w1_identical_zero():
    D = [(0.0, 0.3), (0.1, 2.5), (0.4, 0.9)]
    assert abs(T.wasserstein1(D, list(reversed(D)))) < 1e-9


def test_w1_single_vs_empty():
    # одна точка (b,d) против пустой диаграммы: падает в диагональ за p/2
    v = T.wasserstein1([(0.0, 2.0)], [])
    assert abs(v - 1.0) < 1e-9, f"W1={v}, ожидалось 1.0 (=p/2)"


def test_w1_empty_vs_empty():
    assert T.wasserstein1([], []) == 0.0


def test_w1_triangle_sanity():
    # A=(0,3), B=(1,4): L_inf=1; в диагональ обоим = 1.5+1.5=3 -> min = 1
    v = T.wasserstein1([(0.0, 3.0)], [(1.0, 4.0)])
    assert abs(v - 1.0) < 1e-9, f"W1={v}"


def test_w1_trim_semantics():
    # A: одна крупная (p=2) + 100 мелких (p=0.05); B: только крупная.
    # После trim top-64: у A 64 точки, у B 1. Крупные совпадают (0),
    # 63 лишних мелких падают в диагональ по p/2 = 0.025 -> W1 = 63*0.025.
    # (Это стандартная семантика W_1: несопоставленные фичи стоят p/2.)
    A = [(0.0, 2.0)] + [(float(i) * 0.001, float(i) * 0.001 + 0.05) for i in range(100)]
    B = [(0.0, 2.0)]
    v = T.wasserstein1(A, B, k=64)
    assert abs(v - 63 * 0.025) < 1e-9, f"W1={v}, ожидалось {63*0.025}"


def test_w1_trim_drops_low_persistence_features():
    # A: 100 фич с p_i = 0.01*(i+1); B: пусто. Каждая несопоставленная фича
    # стоит p/2 -> без trim W1 = sum(p)/2; с top-64 учитываются только i>=37.
    A = [(float(i) * 0.5, float(i) * 0.5 + 0.01 * (i + 1)) for i in range(100)]
    B = []
    v_full = T.wasserstein1(A, B, k=10**6)
    v_trim = T.wasserstein1(A, B, k=64)
    p_all = [0.01 * (i + 1) for i in range(100)]
    exp_full = sum(p_all) / 2                      # 25.25
    exp_trim = sum(sorted(p_all, reverse=True)[:64]) / 2   # top-64 по персистентности
    assert abs(v_full - exp_full) < 1e-9, f"full: {v_full} vs {exp_full}"
    assert abs(v_trim - exp_trim) < 1e-9, f"trim: {v_trim} vs {exp_trim}"
    assert v_trim < v_full


# ── ранги (соглашения v5) ──────────────────────────────────────

def test_effective_rank_known():
    # Плоский спектр: U @ V^T (случайные ортонормированные базы) -> ровно k единичных sigma.
    # p_i = 1/k на каждом -> eff_rank = exp(H(uniform_k)) = точно k.
    rng = np.random.default_rng(7)
    k, m, p_ = 12, 60, 40
    U, _ = np.linalg.qr(rng.normal(size=(m, k)))
    V, _ = np.linalg.qr(rng.normal(size=(p_, k)))
    A = U @ V.T + rng.normal(size=(m, p_)) * 1e-9
    s = np.linalg.svd(A, compute_uv=False)
    er = T.effective_rank(s)
    assert abs(er - k) < 0.5, f"eff_rank={er}, ожидалось ~{k}"


def test_effective_rank_flat_vs_spiky():
    # Равномерный спектр (10 единичных sigma) -> eff_rank ~ 10;
    # острый спектр (один большой sigma) -> eff_rank ~ 1.
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


# ── производительность на рабочем размере ─────────────────────

def test_persistence_runtime_workload_size():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(256, 32))
    t0 = time.time()
    g = T.persistence_gudhi(X, k=32)
    dt = time.time() - t0
    assert len(g["h0"]) >= 1 and g["n_edges"] > 0
    print(f"    [runtime] persistence_gudhi n=256,d=32,k=32: {dt:.1f}s "
          f"(edges={g['n_edges']}, tris={g['n_triangles']})")
    assert dt < 120, f"GUDHI слишком медленный для рабочей нагрузки: {dt:.1f}s"


# ── логирование (Tee) ───────────────────────────────────────────

def test_tee_writes_to_all_streams():
    import io
    a, b = io.StringIO(), io.StringIO()
    tda_log.Tee(a, b).write("привет\n")
    assert a.getvalue() == "привет\n" and b.getvalue() == "привет\n"


def test_tee_survives_broken_stream():
    import io

    class Broken:
        def write(self, s):
            raise OSError("console gone")

    ok = io.StringIO()
    tda_log.Tee(Broken(), ok).write("x\n")   # не должно падать
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
