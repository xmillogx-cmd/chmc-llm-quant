#!/usr/bin/env python3
"""
analyze.py — TDA analysis of CHMC v6 compression (CPU, no GPU)
==============================================================

Reads results_v6/tda_analysis/tda_activations_<model>.pt (collected by
collect_activations.py) and writes exactly 4 files into the same directory
with a <model> suffix (the .pt from collect is the fifth artifact; so exactly
5 files per model, and different models coexist in one folder):

  1. tda_layer_table_<model>.csv   — per block + weight aggregates: Betti at eps*,
     topo-rank, W_1(H0/H1), pre/post cosine, activation d90;
     for weights: effective_rank and d90 before/after (all layers / attn / mlp).
  2. tda_summary_<model>.json      — aggregates + verdict "did compression break topology".
  3. tda_diagrams_<model>.png      — 4 panels: H_0 pre/post diagrams, b_1(eps*) per block,
     W_1(H1)+cosine per block, weight effective rank pre vs post.
  4. tda_report_<model>.md         — report (in Russian): method, caveats, numbers, verdict.

Method (see tda_core.py): reduced VR (kNN k=32), H_0 union-find + GUDHI cross-check,
H_1 via GUDHI SimplexTree; eps* = median NN distance; W_1 over the top-64 intervals.

Usage:
    python tda_analysis/analyze.py [--data PATH] [--out-dir DIR]
Time estimate (SmolLM, 30 blocks): ~2-5 min on CPU.

Logging: all output is duplicated to tda_analysis/logs/analyze_<model>_<time>.log
(console + file, including the traceback on failure).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent          # tda_analysis/
ROOT_DIR = BASE_DIR.parent                          # cmq_experiment/
sys.path.insert(0, str(BASE_DIR))

import tda_core as T                                 # noqa: E402
import tda_log                                       # noqa: E402

K_NN = 32            # разреженность VR (план: kNN вместо полного O(n^2))
SAMPLE = 512         # точек на облако для персистентности (план: 512, не 2048)
W1_TRIM = 64         # top-K интервалов для W_1
SEED = 42


# ──────────────────────────────────────────────────────────────

def _sample(X: np.ndarray, n: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if X.shape[0] <= n:
        return X
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx]


def _cosine_rows(A: np.ndarray, B: np.ndarray) -> float:
    """Средний построчный косинус pre vs post (аналог 'per-layer cosine 0.85')."""
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    denom = np.maximum(na * nb, 1e-12)
    return float(np.mean((A * B).sum(axis=1) / denom))


def _act2d(t: torch.Tensor) -> np.ndarray:
    """Активации блока -> (N, d) float64. Старые .pt хранят (B=1, N, d) —
    batch-меру убираем; новые collect сразу пишут (N, d)."""
    X = t.numpy()
    if X.ndim == 3:
        assert X.shape[0] == 1, f"неожиданный batch={X.shape[0]}"
        X = X[0]
    return np.ascontiguousarray(X, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="TDA analysis of CHMC v6 compression")
    default_data = ROOT_DIR / "results_v6" / "tda_analysis" / "tda_activations_smollm-135m.pt"
    parser.add_argument("--data", default=str(default_data))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    data_file = Path(args.data)
    out_dir = Path(args.out_dir) if args.out_dir else data_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    model_tag = data_file.stem.removeprefix("tda_activations_") or data_file.stem
    _log_file = tda_log.start_logging(
        tda_log.log_file_for("analyze", model_tag))  # noqa: F841 (дожитие до конца процесса)

    print(f"Loading {data_file} ...")
    payload = torch.load(data_file, map_location="cpu", weights_only=False)
    model = payload["model"]
    blocks: list[str] = payload["blocks"]
    pre_t: list[torch.Tensor] = payload["pre"]
    post_t: list[torch.Tensor] = payload["post"]
    sv_pre: dict[str, torch.Tensor] = payload["weight_sv_pre"]
    sv_post: dict[str, torch.Tensor] = payload["weight_sv_post"]
    ranks: dict[str, int] = payload.get("ranks", {})
    ppl = payload.get("ppl", {})

    n_blocks = len(blocks)
    n_tok = _act2d(pre_t[0]).shape[0] if pre_t else int(payload.get("n_tokens", 0))
    print(f"Model={model} | blocks={n_blocks} | tokens={n_tok} "
          f"| hidden={payload['hidden_size']}")
    if ppl:
        print(f"PPL baseline={ppl.get('baseline')} compressed={ppl.get('compressed')} "
              f"ratio={ppl.get('ratio')}")

    # ── по блокам: персистентность pre/post + метрики ───────────────
    rows = []
    h0_crosscheck_bad = 0
    for i, name in enumerate(blocks):
        A_pre, A_post = _act2d(pre_t[i]), _act2d(post_t[i])
        X_pre = _sample(A_pre, SAMPLE)
        X_post = _sample(A_post, SAMPLE)

        eps_p = T.eps_star(X_pre)
        eps_q = T.eps_star(X_post)

        # H_0: union-find (точная) + GUDHI (кроссчек на eps*)
        ei, ew = T.knn_edges(X_pre, K_NN)
        h0_u_pre = T.h0_intervals_unionfind(X_pre.shape[0], ei, ew)
        ej, ewj = T.knn_edges(X_post, K_NN)
        h0_u_post = T.h0_intervals_unionfind(X_post.shape[0], ej, ewj)

        g_pre = T.persistence_gudhi(X_pre, K_NN)
        g_post = T.persistence_gudhi(X_post, K_NN)
        if abs(T.betti_at_scale(g_pre["h0"], eps_p) - T.betti_at_scale(h0_u_pre, eps_p)) > 0:
            h0_crosscheck_bad += 1

        b0p, b0q = T.betti_at_scale(h0_u_pre, eps_p), T.betti_at_scale(h0_u_post, eps_q)
        b1p = T.betti_at_scale(g_pre["h1"], eps_p)
        b1q = T.betti_at_scale(g_post["h1"], eps_q)
        tr_pre = T.topo_rank({"0": h0_u_pre, "1": g_pre["h1"]}, eps_p)
        tr_post = T.topo_rank({"0": h0_u_post, "1": g_post["h1"]}, eps_q)

        w1_h0 = T.wasserstein1(h0_u_pre, h0_u_post, W1_TRIM)
        w1_h1 = T.wasserstein1(g_pre["h1"], g_post["h1"], W1_TRIM)

        cos_mean = _cosine_rows(A_pre, A_post)
        d90_act_p, eff_act_p = T.activation_pca_d90(A_pre)
        d90_act_q, eff_act_q = T.activation_pca_d90(A_post)

        rows.append({
            "item": name, "kind": "block",
            "eps_star_pre": round(eps_p, 6), "eps_star_post": round(eps_q, 6),
            "b0_pre": b0p, "b0_post": b0q, "b1_pre": b1p, "b1_post": b1q,
            "topo_rank_pre": tr_pre, "topo_rank_post": tr_post,
            "w1_h0": round(w1_h0, 6), "w1_h1": round(w1_h1, 6),
            "cos_mean": round(cos_mean, 5),
            "d90_act_pre": d90_act_p, "d90_act_post": d90_act_q,
            "eff_rank_act_pre": round(eff_act_p, 3), "eff_rank_act_post": round(eff_act_q, 3),
        })
        print(f"  [{i+1}/{n_blocks}] {name}: b0 {b0p}->{b0q} | b1 {b1p}->{b1q} | "
              f"topo-rank {tr_pre}->{tr_post} | W1(H1)={w1_h1:.4f} | cos={cos_mean:.4f}")

    # ── по весам: effective rank / d90 до и после (соглашения v5) ───
    def _agg(names):
        eff_p, eff_q, d90_p, d90_q = [], [], [], []
        for n in names:
            s1 = sv_pre[n].numpy().astype(np.float64)
            s2 = sv_post[n].numpy().astype(np.float64)
            eff_p.append(T.effective_rank(s1))
            eff_q.append(T.effective_rank(s2))
            d90_p.append(T.d90(s1))
            d90_q.append(T.d90(s2))
        return {
            "n_layers": len(names),
            "eff_rank_min_pre": round(min(eff_p), 3), "eff_rank_med_pre": round(float(np.median(eff_p)), 3),
            "eff_rank_min_post": round(min(eff_q), 3), "eff_rank_med_post": round(float(np.median(eff_q)), 3),
            "d90_min_pre": int(min(d90_p)), "d90_med_pre": int(np.median(d90_p)),
            "d90_min_post": int(min(d90_q)), "d90_med_post": int(np.median(d90_q)),
        }

    all_names = list(sv_pre.keys())
    attn_names = [n for n in all_names if "attn" in n]
    mlp_names = [n for n in all_names if "mlp" in n or "down_proj" in n]
    w_all, w_attn, w_mlp = _agg(all_names), _agg(attn_names), _agg(mlp_names)

    # ── CSV (ровно один из 5 артефактов) ────────────────────────────
    csv_file = out_dir / f"tda_layer_table_{model}.csv"
    fields = ["item", "kind", "n_layers",
              "eps_star_pre", "eps_star_post",
              "b0_pre", "b0_post", "b1_pre", "b1_post",
              "topo_rank_pre", "topo_rank_post",
              "w1_h0", "w1_h1", "cos_mean",
              "d90_act_pre", "d90_act_post", "eff_rank_act_pre", "eff_rank_act_post",
              "eff_rank_min_pre", "eff_rank_med_pre", "eff_rank_min_post", "eff_rank_med_post",
              "d90_min_pre", "d90_med_pre", "d90_min_post", "d90_med_post"]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        for label, agg in (("weight_all", w_all), ("weight_attn", w_attn), ("weight_mlp", w_mlp)):
            row = {"item": label, "kind": "weight"}
            row.update(agg)
            w.writerow(row)
    print(f"CSV: {csv_file}")

    # ── вердикт ─────────────────────────────────────────────────────
    cos_arr = np.array([r["cos_mean"] for r in rows])
    w1h1_arr = np.array([r["w1_h1"] for r in rows])
    b1_drop = int(sum(1 for r in rows if r["b1_post"] < r["b1_pre"]))
    b1_gain = int(sum(1 for r in rows if r["b1_post"] > r["b1_pre"]))
    tr_change = float(np.mean([r["topo_rank_post"] - r["topo_rank_pre"] for r in rows]))

    def _corr(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.std() < 1e-12 or b.std() < 1e-12:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    corr_w1_cos = _corr(w1h1_arr, 1.0 - cos_arr)
    verdict_bits = []
    if cos_arr.mean() > 0.95:
        verdict_bits.append("средний косинус pre/post высокий (>0.95): направление потоков почти не изменилось")
    else:
        verdict_bits.append(f"средний косинус pre/post {cos_arr.mean():.3f}: заметное изменение направлений")
    if b1_gain > 2 * max(b1_drop, 1) and w1h1_arr.mean() > 0:
        verdict_bits.append("после сжатия появляются новые H_1-циклы (b1_gain > b1_drop): топология НЕ просто 'стала проще'")
    elif b1_drop > 2 * max(b1_gain, 1):
        verdict_bits.append("H_1-циклов после сжатия меньше: топология упрощается/вырождается")
    else:
        verdict_bits.append("число H_1-циклов при eps* примерно сохранено")
    if w_all["eff_rank_med_post"] < 0.8 * w_all["eff_rank_med_pre"]:
        verdict_bits.append(f"effective rank весов упал (med {w_all['eff_rank_med_pre']} -> {w_all['eff_rank_med_post']})")
    else:
        verdict_bits.append(f"effective rank весов почти не изменился (med {w_all['eff_rank_med_pre']} -> {w_all['eff_rank_med_post']}) — остаточная квантизация держит 'хвост' спектра")

    # ── JSON summary ────────────────────────────────────────────────
    summary = {
        "model": model,
        "data_file": str(data_file),
        "method": {
            "vr_sparsification": f"kNN k={K_NN}, triangles=cliques-3 (cap 300k)",
            "h0_backend": "union-find (exact) + GUDHI cross-check",
            "h1_backend": "GUDHI SimplexTree, filtration=max vertex/edge weight",
            "eps_star": "median nearest-neighbour distance per cloud",
            "w1": f"trimmed top-{W1_TRIM} by persistence, L_inf, diagonal match cost p/2; essential classes excluded",
            "sample_points": SAMPLE, "seed": SEED,
            "effective_rank": "exp(entropy of normalized sigma^2) — v5 convention (generate_cov_stats.py / rank_gap.py)",
        },
        "ppl": ppl,
        "config_overrides": {k: payload["config"].get(k) for k in
                             ("strict_sequential", "dampening", "residual_bits", "group_size")},
        "aggregates": {
            "cos_mean_blocks": round(float(cos_arr.mean()), 5),
            "w1_h0_mean": round(float(np.mean([r['w1_h0'] for r in rows])), 6),
            "w1_h1_mean": round(float(w1h1_arr.mean()), 6),
            "b1_drop_blocks": b1_drop, "b1_gain_blocks": b1_gain,
            "topo_rank_delta_mean": round(tr_change, 4),
            "h0_crosscheck_mismatches": h0_crosscheck_bad,
        },
        "weights": {"all": w_all, "attn": w_attn, "mlp": w_mlp},
        "correlations_honest_proxy": {
            "note": "W_1(H1) vs (1-cosine) по блокам — прокси 'топологическое изменение <-> потеря выравнивания'; delta-PPL на блок не измерялся",
            "pearson_w1h1_vs_1minus_cos": corr_w1_cos,
        },
        "verdict_bits": verdict_bits,
    }
    json_file = out_dir / f"tda_summary_{model}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"JSON: {json_file}")

    # ── figure (matplotlib Agg) ─────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"TDA of CHMC v6 compression — {model} (BPW={payload['config'].get('bit_budget_bpw')})",
                 fontsize=13)

    # (a) H_0 diagram pre/post, средний блок
    ax = axes[0][0]
    mid = n_blocks // 2
    Xp_mid = _sample(_act2d(pre_t[mid]), SAMPLE)
    Xq_mid = _sample(_act2d(post_t[mid]), SAMPLE)
    ei_m, ew_m = T.knn_edges(Xp_mid, K_NN)
    h0p_m = [iv for iv in T.h0_intervals_unionfind(Xp_mid.shape[0], ei_m, ew_m) if np.isfinite(iv[1])]
    ej_m, ewj_m = T.knn_edges(Xq_mid, K_NN)
    h0q_m = [iv for iv in T.h0_intervals_unionfind(Xq_mid.shape[0], ej_m, ewj_m) if np.isfinite(iv[1])]
    ax.scatter([b for b, _ in h0p_m], [d for _, d in h0p_m], s=25, alpha=0.7, color="tab:blue", label=f"H0 pre ({blocks[mid]})")
    ax.scatter([b for b, _ in h0q_m], [d for _, d in h0q_m], s=25, alpha=0.7, color="tab:orange", label=f"H0 post ({blocks[mid]})")
    all_d = [d for _, d in h0p_m + h0q_m]
    lim = (max(all_d) if all_d else 1e-3) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("birth"); ax.set_ylabel("death")
    ax.set_title(f"(a) H0 persistence diagram, block {blocks[mid]} (essential class excluded)")
    ax.legend(fontsize=8)

    # (b) b1(eps*) pre vs post по блокам
    ax = axes[0][1]
    xs = np.arange(n_blocks)
    ax.bar(xs - 0.2, [r["b1_pre"] for r in rows], width=0.4, color="tab:blue", label="b1 pre")
    ax.bar(xs + 0.2, [r["b1_post"] for r in rows], width=0.4, color="tab:orange", label="b1 post")
    ax.set_xlabel("block index"); ax.set_ylabel("b_1(eps*)")
    ax.set_title("(b) H1 Betti at eps* per block (pre vs post)")
    ax.legend(fontsize=8)

    # (c) W1(H1) + cosine по блокам
    ax = axes[1][0]
    ax.bar(xs, w1h1_arr, color="tab:red", alpha=0.6, label="W1(H1)")
    ax.set_xlabel("block index"); ax.set_ylabel("W1(H1), trimmed top-64", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(xs, cos_arr, "o-", color="tab:green", ms=3, label="mean cosine pre/post")
    ax2.set_ylabel("mean cosine", color="tab:green")
    ax.set_title("(c) topology change (W1 H1) vs alignment loss (cosine)")

    # (d) effective rank весов pre vs post
    ax = axes[1][1]
    er_pre = [T.effective_rank(sv_pre[n].numpy().astype(np.float64)) for n in all_names]
    er_post = [T.effective_rank(sv_post[n].numpy().astype(np.float64)) for n in all_names]
    ax.scatter(er_pre, er_post, s=18, alpha=0.7)
    mmax = max(max(er_pre), max(er_post)) * 1.05
    ax.plot([0, mmax], [0, mmax], "k--", lw=0.8)
    at = [n for n in all_names if "attn" in n]
    ml = [n for n in all_names if "mlp" in n or "down_proj" in n]
    ax.scatter([er_pre[all_names.index(n)] for n in at], [er_post[all_names.index(n)] for n in at],
               s=26, facecolor="none", edgecolor="tab:blue", label="attn")
    ax.scatter([er_pre[all_names.index(n)] for n in ml], [er_post[all_names.index(n)] for n in ml],
               s=26, facecolor="none", edgecolor="tab:orange", label="mlp")
    ax.set_xlabel("effective rank pre (FP32)"); ax.set_ylabel("effective rank post (compressed)")
    ax.set_title("(d) weight effective rank per layer (exp-entropy sigma^2, v5 convention)")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_file = out_dir / f"tda_diagrams_{model}.png"
    fig.savefig(png_file, dpi=140)
    plt.close(fig)
    print(f"PNG: {png_file}")

    # ── отчёт (MD) ──────────────────────────────────────────────────
    md_lines = _build_report(payload, rows, w_all, w_attn, w_mlp, summary, corr_w1_cos)
    md_file = out_dir / f"tda_report_{model}.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"MD:  {md_file}")

    print("\n=== VERDICT ===")
    for v in verdict_bits:
        print(" -", v)
    if corr_w1_cos is not None:
        print(f" - corr(W1(H1), 1-cos) = {corr_w1_cos:.3f} (честный прокси, не delta-PPL)")


def _build_report(payload, rows, w_all, w_attn, w_mlp, summary, corr):
    model = payload["model"]
    ppl = payload.get("ppl", {})
    cfg = payload["config"]
    cos_arr = np.array([r["cos_mean"] for r in rows])
    w1h0 = float(np.mean([r["w1_h0"] for r in rows]))
    w1h1 = float(np.mean([r["w1_h1"] for r in rows]))
    b1p_tot = sum(r["b1_pre"] for r in rows)
    b1q_tot = sum(r["b1_post"] for r in rows)

    L = []
    L.append(f"# TDA-анализ сжатия CHMC v6 — {model}")
    L.append("")
    L.append("## Что и зачем")
    L.append("")
    L.append(f"Проверка гипотезы из плана 'Топологический путь': квантизация Q: R^d -> дискретная сетка "
             f"может ломать топологию манифольда активаций, даже когда MSE низкий. Сравниваем персистентную "
             f"топологию блок-активаций (residual stream) и спектр весов ДО/ПОСЛЕ сжатия CHMC v6 при равном "
             f"BPW={cfg.get('bit_budget_bpw')} (конфиг: strict_sequential={cfg.get('strict_sequential')}, "
             f"dampening={cfg.get('dampening')}).")
    L.append("")
    if ppl:
        L.append(f"PPL baseline={ppl.get('baseline'):.4f} -> compressed={ppl.get('compressed'):.4f} "
                 f"(ratio {ppl.get('ratio'):.6f}) — из того же прогона, что и данные.")
        L.append("")
    L.append("## Методика и оговорки")
    L.append("")
    L.append(f"- VR-комплекс разрежен: kNN-граф (k={K_NN}), треугольники = клики размера 3; полный O(n^2) VR не строился.")
    L.append("- Семантика спарсификации: фичи, которые в полном VR умерли бы на масштабах за пределами kNN-радиуса, в разреженном комплексе НЕ имеют симплексов, их убивающих, -> становятся существенными (death=inf) и учитываются в b_k(eps*). Pre/post считаются в одних условиях, поэтому сравнение корректно; W_1 существенные классы исключает.")
    L.append(f"- Персистентность считалась на {SAMPLE} точках (детерминированная выборка, seed=42), а не на всех токенах — смещение к крупным структурам, приемлемо для сравнения pre/post в одних условиях.")
    L.append("- eps* = медианное расстояние до ближайшего соседа в облаке; Betti и topo-rank читаются при s=eps*.")
    L.append(f"- W_1 — точный расчёт на обрезанных диаграммах (top-{W1_TRIM} по персистентности, L_inf); существенные H_0-классы (death=inf) исключены до расчёта.")
    L.append("- H_0: union-find (точно) + кроссчек GUDHI; H_1: GUDHI SimplexTree (filtration = max веса вершин/рёбер).")
    L.append(f"- effective_rank/d90 — соглашения v5 (generate_cov_stats.py, rank_gap.py): exp-энтропия нормализованных sigma^2. Числа сопоставимы с ранними замерами (min eff-rank ~17, median ~60).")
    L.append("- Ограничение: W_1 и Betti — метрики на БЛОК-активациях; связь с delta-PPL не прямая, корреляции ниже — честный прокси.")
    L.append("")
    L.append("## Ключевые числа")
    L.append("")
    L.append("| Метрика | pre | post |")
    L.append("|---|---|---|")
    L.append(f"| Средний косинус pre/post (по блокам) | — | {cos_arr.mean():.4f} |")
    L.append(f"| W_1(H0), среднее по блокам | — | {w1h0:.5f} |")
    L.append(f"| W_1(H1), среднее по блокам | — | {w1h1:.5f} |")
    L.append(f"| Сумма b_1(eps*) по блокам | {b1p_tot} | {b1q_tot} |")
    L.append(f"| Effective rank весов, median (все слои) | {w_all['eff_rank_med_pre']} | {w_all['eff_rank_med_post']} |")
    L.append(f"| Effective rank весов, min (все слои) | {w_all['eff_rank_min_pre']} | {w_all['eff_rank_min_post']} |")
    L.append(f"| d90 весов, median (все слои) | {w_all['d90_med_pre']} | {w_all['d90_med_post']} |")
    L.append("")
    L.append("По attn: eff-rank med "
             f"{w_attn['eff_rank_med_pre']} -> {w_attn['eff_rank_med_post']}; по mlp: "
             f"{w_mlp['eff_rank_med_pre']} -> {w_mlp['eff_rank_med_post']}.")
    L.append("")
    if corr is not None:
        L.append(f"Корреляция Пирсона W_1(H1) vs (1 - cosine) по блокам: **{corr:.3f}**.")
        L.append("")
    L.append("## Вердикт")
    L.append("")
    for v in summary["verdict_bits"]:
        L.append(f"- {v}")
    L.append("")
    L.append("**Чтение:** если косинусы высокие, а W_1(H1) и новые циклы есть — компрессия сохраняет "
             "направление потока, но перестраивает тонкую топологическую структуру (циклы/компоненты при eps*). "
             "Если effective rank весов почти не упал — остаточная 4-bit квантизация заполняет весь спектр шумом: "
             "'топологическая компрессия' на весах НЕ произошла, сжатие реально несёт только low-rank часть. "
             "Это и есть формализация разрыва 'MSE низкий, но PPL растёт': метрики, нечувствительные к топологии/спектру, пропускают эту деградацию.")
    L.append("")
    L.append("## Файлы (ровно 5 артефактов на модель в results_v6/tda_analysis/)")
    L.append("")
    L.append(f"1. tda_activations_{model}.pt — сырые данные (блок-активации pre/post, SVD весов до/после, ranks, PPL)")
    L.append(f"2. tda_layer_table_{model}.csv — таблица по блокам + агрегаты весов")
    L.append(f"3. tda_summary_{model}.json — агрегаты, методика, вердикт (машиночитаемый)")
    L.append(f"4. tda_diagrams_{model}.png — 4 панели: H0-диаграммы, b_1(eps*) по блокам, W_1(H1)+косинус, eff-rank весов")
    L.append(f"5. tda_report_{model}.md — этот отчёт")
    L.append("")
    return L


if __name__ == "__main__":
    main()
