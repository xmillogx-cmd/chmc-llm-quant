#!/usr/bin/env python3
"""
drift_diagnostics.py — CHMC v6.2 step 1: drift diagnostics (no new compression)
================================================================================

Hypothesis: the post-quantization drift d(x) = (W_comp - W)x is a deterministic
low-rank field stretched along the activation cone. The check runs on SAVED TDA
data (.pt from collect_activations.py): it already contains y_orig/y_comp pairs —
pre[i] = output of block i in the FP32 model, post[i] = the same block after CHMC v6
(strict_sequential=true, dampening=0.05), the same 2048 calibration tokens.

For 4 representative blocks [0, n/4, n/2, n-1] we compute:
  * D = post - pre (N x d);
  * PCA(D): top-32 spectrum (energy shares), d90_drift (component at 90% energy),
    effective rank of the drift (v5 convention, exp-entropy of the shares);
  * cone basis: PCA of centered pre-activations of the same block;
  * cone_axis_overlap = |cos(top-1 drift, top-1 cone)|;
  * top10_cone_overlap = energy-weighted best-match: for each of the top-10 drift directions
    the best |cos| against the top-10 cone directions, averaged by energy shares (axis signs are arbitrary);
  * drift_norm_mean/std, rel.disp = mean ||d||/||pre||.

Acceptance criteria (from the v6.2 plan):
  PASS: d90_drift < 30 for most blocks AND top10_cone_overlap > 0.5;
  STOP: d90_drift > 50 or overlap < 0.3 -> negative result, record it.

Output (results_v6/drift_correction/step1_diagnostics/):
  drift_pca_<model>.json — per block + verdict;
  drift_spectrum_plots.png — shared file: rows=models, columns=blocks,
    spectrum + cumulative energy with the d90 marker.

Usage (CPU, seconds per model):
    python drift_correction/drift_diagnostics.py --data results_v6/tda_analysis/tda_activations_smollm-135m.pt
    python drift_correction/drift_diagnostics.py --all

Logging: drift_correction/logs/step1_<model>_<time>.log (console + file).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent          # drift_correction/
ROOT_DIR = BASE_DIR.parent                          # cmq_experiment/
sys.path.insert(0, str(ROOT_DIR / "tda_analysis"))  # tda_log

import tda_log                                       # noqa: E402

TOP_K = 32           # сколько компонент спектра хранить/рисовать
CONE_TOP = 10        # конусный базис для overlap-метрик
D90_PASS = 30        # критерий приёмки
D90_STOP = 50        # жёсткое правило: > 50 -> гипотеза неверна, STOP
OVERLAP_PASS = 0.5
OVERLAP_STOP = 0.3


def _act2d(t: torch.Tensor) -> np.ndarray:
    """Активации блока -> (N, d) float64. Старые .pt хранят (B=1, N, d)."""
    X = t.numpy()
    if X.ndim == 3:
        assert X.shape[0] == 1, f"неожиданный batch={X.shape[0]}"
        X = X[0]
    return np.ascontiguousarray(X, dtype=np.float64)


def _pca_svd(Xc: np.ndarray):
    """SVD центрированной (N x d). Возвращает (s, V), V — оси в R^d (d x r)."""
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return s, Vt.T


def _energy_fractions(s: np.ndarray) -> np.ndarray:
    e = s ** 2
    return e / e.sum()


def _d90(frac: np.ndarray) -> int:
    """Минимальное k с cumsum(frac)[:k] >= 0.90 (1-based)."""
    c = np.cumsum(frac)
    return int(np.searchsorted(c, 0.90) + 1)


def _eff_rank(frac: np.ndarray) -> float:
    """v5-соглашение: exp(энтропии нормализованных долей)."""
    p = frac[frac > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def analyze_block(pre: np.ndarray, post: np.ndarray) -> dict:
    D = post - pre
    N, d = D.shape

    Dc = D - D.mean(axis=0)
    s_d, Vd = _pca_svd(Dc)
    fr_d = _energy_fractions(s_d)
    d90_drift = _d90(fr_d)

    # конусный базис из pre-активаций того же блока
    s_c, Vc = _pca_svd(pre - pre.mean(axis=0))
    fr_c = _energy_fractions(s_c)
    d90_cone = _d90(fr_c)

    # overlap: знак PCA-осей произволен -> модуль косинуса
    M = np.abs(Vd[:, :CONE_TOP].T @ Vc[:, :CONE_TOP])   # (10 x 10)
    cone_axis_overlap = float(M[0, 0])
    # energy-weighted best-match: направления с нулевой энергией (при малом d90 их
    # ориентация произвольна) не должны вносить шум -> весим долями энергии drift-PCA
    w = fr_d[:CONE_TOP]
    top10_cone_overlap = float((w * M.max(axis=1)).sum() / max(w.sum(), 1e-12))

    na = np.linalg.norm(pre, axis=1)
    nd = np.linalg.norm(D, axis=1)
    rel_disp = float(nd.mean() / np.maximum(na.mean(), 1e-12))

    # cone capture: доля энергии дрейфа в span(top-30 конусных направлений) —
    # верхняя оценка того, что мог бы захватить идеализованный ДW = E·V·Vᵀ (k=30)
    proj30 = Dc @ Vc[:, :30]
    cone_capture_30 = float((proj30 ** 2).sum() / max((Dc ** 2).sum(), 1e-12))

    d90_ok = d90_drift < D90_PASS
    ov_ok = top10_cone_overlap > OVERLAP_PASS
    hard_fail = (d90_drift > D90_STOP) or (top10_cone_overlap < OVERLAP_STOP)
    verdict = "FAIL" if hard_fail else ("PASS" if (d90_ok and ov_ok) else "MARGINAL")

    return {
        "n_tokens": int(N),
        "dim": int(d),
        "drift_norm_mean": float(nd.mean()),
        "drift_norm_std": float(nd.std()),
        "rel_disp_mean": rel_disp,
        "pca_spectrum_top32": [float(x) for x in fr_d[:TOP_K]],
        "d90_drift": d90_drift,
        "eff_rank_drift": _eff_rank(fr_d),
        "cone_d90": d90_cone,
        "cone_eff_rank": _eff_rank(fr_c),
        "cone_axis_overlap": cone_axis_overlap,
        "top10_cone_overlap": top10_cone_overlap,
        "cone_capture_30": cone_capture_30,
        "verdict": verdict,
    }


def run_model(data_file: Path) -> dict:
    payload = torch.load(data_file, map_location="cpu")
    model = str(payload.get("model", data_file.stem.replace("tda_activations_", "")))
    pre_t, post_t = payload["pre"], payload["post"]
    n_blocks = len(pre_t)

    cand = [0, n_blocks // 4, n_blocks // 2, n_blocks - 1]
    blocks = sorted(set(cand))

    out = {
        "model": model,
        "data_file": str(data_file),
        "n_tokens": int(payload.get("n_tokens", _act2d(pre_t[0]).shape[0])),
        "hidden_size": int(payload.get("hidden_size", 0)),
        "ppl": payload.get("ppl"),
        "method": {
            "drift": "D = post - pre на выходных decoder-блоков (те же 2048 токенов)",
            "pre": "выход блока в FP32-модели",
            "post": "выход того же блока после CHMC v6 (strict_sequential, dampening=0.05)",
            "pca": "numpy SVD центрированных матриц (детерминированно)",
            "cone_basis": "top-10 правых сингулярных векторов cent(pre) того же блока",
            "overlap": "|cos| (знак осей произволен); top10 = energy-weighted best-match по drift-направлениям",
            "cone_capture_30": "||Dc @ Vc[:, :30]||^2 / ||Dc||^2 — доля энергии дрейфа в span(top-30 конуса)",
        },
        "blocks": {},
    }

    for bi in blocks:
        pre, post = _act2d(pre_t[bi]), _act2d(post_t[bi])
        res = analyze_block(pre, post)
        out["blocks"][f"layers.{bi}"] = res
        print(f"  [block {bi}] d90_drift={res['d90_drift']} | "
              f"top10_cone_overlap={res['top10_cone_overlap']:.3f} | "
              f"cone_axis={res['cone_axis_overlap']:.3f} | "
              f"cap30={res['cone_capture_30']:.3f} | "
              f"rel.disp={res['rel_disp_mean']:.4f} | {res['verdict']}")

    verdicts = [v["verdict"] for v in out["blocks"].values()]
    n_pass, n_fail = verdicts.count("PASS"), verdicts.count("FAIL")
    if n_fail > 0:
        overall = "NEGATIVE (hard fail: d90>50 или overlap<0.3)"
    elif n_pass >= max(1, len(verdicts) - 1):
        overall = "CONFIRMED"
    else:
        overall = "MARGINAL"
    out["overall"] = {
        "n_blocks": len(verdicts),
        "pass": n_pass,
        "marginal": verdicts.count("MARGINAL"),
        "fail": n_fail,
        "verdict": overall,
    }
    print(f"  OVERALL: {overall} (pass={n_pass}, marginal={out['overall']['marginal']}, fail={n_fail})")
    return out


def main():
    ap = argparse.ArgumentParser(description="Шаг 1 v6.2: диагностика дрейфа на сохранённых .pt")
    ap.add_argument("--data", help="путь к tda_activations_<model>.pt")
    ap.add_argument("--all", action="store_true",
                    help="все три модели из results_v6/tda_analysis/")
    ap.add_argument("--out-dir", default=str(ROOT_DIR / "results_v6" / "drift_correction" / "step1_diagnostics"))
    args = ap.parse_args()

    if not args.all and not args.data:
        ap.error("нужно --data PATH или --all")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tda_analysis_dir = ROOT_DIR / "results_v6" / "tda_analysis"
    if args.all:
        data_files = sorted(tda_analysis_dir.glob("tda_activations_*.pt"))
        if not data_files:
            raise SystemExit(f".pt не найдены в {tda_analysis_dir}")
    else:
        data_files = [Path(args.data)]

    results = []
    for df in data_files:
        model_tag = df.stem.replace("tda_activations_", "")
        log_path = tda_log.log_file_for("step1", model_tag, base_dir=BASE_DIR)
        start_f = tda_log.start_logging(log_path)

        print(f"=== {model_tag} ===")
        res = run_model(df)
        json_file = out_dir / f"drift_pca_{res['model']}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"JSON: {json_file}")

        results.append((df.name, res))
        start_f.close()

    # общий plot: строки = модели, колонки = 4 блока
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 4, figsize=(17, 3.6 * n_rows), squeeze=False)
    for r, (fname, res) in enumerate(results):
        model = res["model"]
        for c, (bname, bres) in enumerate(res["blocks"].items()):
            ax = axes[r][c]
            spec = np.array(bres["pca_spectrum_top32"])
            x = np.arange(1, len(spec) + 1)
            ax.plot(x, spec, "o-", ms=3, lw=1, color="#1f6fd6", label="energy fraction")
            ax.set_yscale("log")
            ax2 = ax.twinx()
            cum = np.cumsum(spec)
            ax2.plot(x, cum, "-", lw=1.5, color="#e8871a", alpha=0.9, label="cumulative")
            d90 = bres["d90_drift"]
            if d90 <= len(spec):
                ax.axvline(d90, color="red", ls="--", lw=1)
            ax2.axhline(0.9, color="gray", ls=":", lw=0.8)
            ax.set_title(f"{model} | {bname}\nd90={d90} ov={bres['top10_cone_overlap']:.2f} "
                         f"[{bres['verdict']}]", fontsize=9)
            if c == 0:
                ax.set_ylabel(model, fontsize=8)
            ax.set_xlabel("comp #", fontsize=8)
    fig.suptitle("Drift PCA (post - pre CHMC v6): спектр и кумулятивная энергия; "
                 "красная линия = d90_drift", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    png_file = out_dir / "drift_spectrum_plots.png"
    fig.savefig(png_file, dpi=140)
    plt.close(fig)
    print(f"\nPNG: {png_file}")

    # сводка по критериям
    print("\n=== СВОДКА (критерии: d90<30 И overlap>0.5 -> PASS; d90>50 или overlap<0.3 -> STOP) ===")
    for fname, res in results:
        o = res["overall"]
        print(f"{res['model']}: {o['verdict']} (pass={o['pass']}/{o['n_blocks']}, "
              f"marginal={o['marginal']}, fail={o['fail']})")


if __name__ == "__main__":
    main()
