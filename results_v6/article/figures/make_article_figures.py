#!/usr/bin/env python3
"""
make_article_figures.py - figures for the article (CPU)
==================================================

Generates three PNGs in results_v6/article/figures/:
  1. fig_versions.png   - evolution of the PPL ratio v1 to v6 (log scale, GPTQ level = 1.0);
  2. fig_ppl.png        - PPL ratio relative to the original model: CHMC v6 vs GPTQ across three models;
  3. fig_bottleneck.png - "bottleneck": energy share of the top-3 PCs (centered combined
                          cloud [pre;post]) and mean pairwise cosine of pre-tokens per block,
                          three models.

The methodology for fig_bottleneck.png matches Figure 1 / Table 2 of the article:
deterministic subsample of 512 tokens (uniform step, as in project_3d.py),
PCA = numpy SVD of the centered combined set [pre;post] per block.

Usage:
    venv\\Scripts\\python.exe results_v6\\article\\figures\\make_article_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent                      # results_v6/article/figures/
ROOT = HERE.parents[2]                                      # cmq_experiment/
TDA_DIR = ROOT / "results_v6" / "tda_analysis"

MODELS = [
    ("smollm-135m", "#1f6fd6"),
    ("qwen2.5-0.5b", "#e8871a"),
    ("tinyllama-1.1b", "#2e9e5b"),
]

N_PTS = 512


def _act2d(t: torch.Tensor) -> np.ndarray:
    X = t.numpy()
    if X.ndim == 3:
        assert X.shape[0] == 1
        X = X[0]
    return np.ascontiguousarray(X, dtype=np.float64)


def _token_idx(n: int, m: int) -> np.ndarray:
    if n <= m:
        return np.arange(n)
    step = n // m
    return np.arange(0, n, step)[:m]


# ── 1. fig_versions.png ─────────────────────────────────────────

def make_fig_versions():
    versions = ["v1", "v2", "v3", "v4", "v5", "v6"]
    ratios = [4894, 2.35, 1.84, 1.49, 1.29, 1.18]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    colors = ["#d64545", "#e3a04b", "#cfcfcf", "#cfcfcf", "#9fb6d8", "#1f6fd6"]
    bars = ax.bar(versions, ratios, color=colors, width=0.62)
    ax.axhline(1.0, color="#5a6472", ls="--", lw=1.2)
    ax.text(5.45, 1.03, "GPTQ level (ratio = 1.0)", ha="right", va="bottom",
            fontsize=9, color="#5a6472")
    for b, r in zip(bars, ratios):
        label = f"~{r:g}×" if r < 10 else f"{r:g}×"
        ax.annotate(label, xy=(b.get_x() + b.get_width() / 2, r),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9.5)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 12000)
    ax.set_ylabel("PPL ratio vs GPTQ (log scale)")
    ax.set_title("CHMC v1 to v6 on SmolLM-135M: the path from collapse to parity", fontsize=11.5)
    ax.grid(axis="y", alpha=0.25, which="both")
    fig.tight_layout()
    out = HERE / "fig_versions.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] {out}")


# ── 2. fig_ppl.png ──────────────────────────────────────────────

def make_fig_ppl():
    # PPL ratio relative to the original model (FP16). See Table 4 of the article.
    labels = ["SmolLM-135M", "Qwen2.5-0.5B*", "TinyLlama-1.1B"]
    chmc = [1.1825, 1.174, 1.059]      # SmolLM - midpoint of the range 1.179-1.186
    gptq = [1.176, 1.141, 1.081]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    b1 = ax.bar(x - w / 2, chmc, w, color="#e8871a", label="CHMC v6 (mean over seeds)")
    b2 = ax.bar(x + w / 2, gptq, w, color="#1f6fd6", label="GPTQ (4-bit / group-128)")
    for bars, vals in ((b1, chmc), (b2, gptq)):
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9.5)
    ax.axhline(1.0, color="#5a6472", ls="--", lw=1.2)
    ax.text(2.45, 1.003, "original model (FP16)", ha="right", va="bottom",
            fontsize=9, color="#5a6472")
    ax.set_xticks(x, labels)
    ax.set_ylim(1.0, 1.235)
    ax.set_ylabel("PPL ratio vs original model")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("CHMC v6 vs GPTQ at the same bit budget of 4.2875 BPW\n"
                 "(* Qwen - single run; others - mean over 3-4 seeds)", fontsize=11.5)
    fig.tight_layout()
    out = HERE / "fig_ppl.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] {out}")


# ── 3. fig_bottleneck.png ───────────────────────────────────────

def make_fig_bottleneck():
    series = {}
    for model, _ in MODELS:
        payload = torch.load(TDA_DIR / f"tda_activations_{model}.pt", map_location="cpu")
        pre_t, post_t = payload["pre"], payload["post"]
        n_blocks = len(pre_t)
        top3, cosmean = [], []
        for bi in range(n_blocks):
            A_pre, A_post = _act2d(pre_t[bi]), _act2d(post_t[bi])
            idx = _token_idx(A_pre.shape[0], N_PTS)
            Pp, Qq = A_pre[idx], A_post[idx]

            # top-3 energy share: centered combined set [pre;post]
            X = np.vstack([Pp, Qq])
            Xc = X - X.mean(axis=0)
            _, S, _ = np.linalg.svd(Xc, full_matrices=False)
            top3.append(float((S[:3] ** 2).sum() / (S ** 2).sum()))

            # mean pairwise cosine of pre-tokens (raw vectors)
            Pn = Pp / np.maximum(np.linalg.norm(Pp, axis=1, keepdims=True), 1e-12)
            G = Pn @ Pn.T
            n = len(G)
            cosmean.append(float((G.sum() - n) / (n * (n - 1))))
        series[model] = (np.arange(n_blocks), np.array(top3), np.array(cosmean))
        print(f"[data] {model}: blocks={n_blocks} top3 range "
              f"{min(top3):.3f}–{max(top3):.3f}, cos range {min(cosmean):.3f}–{max(cosmean):.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)
    for model, color in MODELS:
        xs, top3, cosm = series[model]
        ax1.plot(xs, np.array(top3) * 100, color=color, lw=1.8, marker="o", ms=3.2, label=model)
        ax2.plot(xs, cosm, color=color, lw=1.8, marker="o", ms=3.2, label=model)

    ax1.set_xlabel("decoder block")
    ax1.set_ylabel("top-3 PC energy share, %")
    ax1.set_title("Concentration: top-3 PCs of the combined cloud [pre;post]", fontsize=11)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9)

    ax2.set_xlabel("decoder block")
    ax2.set_ylabel("mean pairwise cosine of pre-tokens")
    ax2.set_title("Anisotropy: mean cosine within a block", fontsize=11)
    ax2.grid(alpha=0.25)

    fig.suptitle("Bottleneck: after the first 1-2 blocks, almost the entire network lies "
                 "in a low-dimensional cone (peak position varies across models)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = HERE / "fig_bottleneck.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] {out}")


if __name__ == "__main__":
    make_fig_versions()
    make_fig_ppl()
    make_fig_bottleneck()
