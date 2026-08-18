#!/usr/bin/env python3
"""
make_figures.py — генерация новых фигур для статьи (CPU, секунды-минуты)

fig_versions.png     — прогресс версий v1→v6 на SmolLM (PPL ratio, log-scale), линия GPTQ.
fig_ppl.png          — CHMC vs GPTQ по 3 моделям (равный бюджет 4.29 bit/weight).
fig_bottleneck.png   — «бутылочное горлышко» (2 панели): L — доля энергии top-3 ПК
                       объединённого центрированного облака [pre;post] по блокам
                       (та же величина, что «PCA-3D var» в project_3d.py);
                       R — средний попарный косинус pre-токенов. Данные:
                       results_v6/tda_analysis/tda_activations_*.pt.

Вывод: results_v6/article/figures/*.png (dpi=150).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]          # cmq_experiment/
OUT = Path(__file__).resolve().parent / "figures"
TDA_DIR = ROOT / "results_v6" / "tda_analysis"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, GRAY, RED = "#1f6fd6", "#e8871a", "#2e9e5b", "#9aa4b2", "#d64545"
MODELS = ["smollm-135m", "qwen2.5-0.5b", "tinyllama-1.1b"]
MCOLORS = {"smollm-135m": BLUE, "qwen2.5-0.5b": ORANGE, "tinyllama-1.1b": GREEN}


def _act2d(t: torch.Tensor) -> np.ndarray:
    X = t.numpy()
    if X.ndim == 3:
        assert X.shape[0] == 1
        X = X[0]
    return np.ascontiguousarray(X, dtype=np.float64)


def _style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e3e8ef", lw=0.8)
    ax.set_axisbelow(True)


# ── fig 1: версии v1→v6 (SmolLM PPL ratio, log) ───────────────────────
def make_versions():
    versions = ["v1\nplain SVD", "v2\nadaptive rank", "v3\nчестный учёт бит",
                "v4\ncovariance + residual", "v5\n7 стабилизаторов", "v6\nGPTQ-оптимизаторы"]
    ratios = [4894.0, 2.35, 1.84, 1.49, 1.29, 1.18]
    gptq = 1.176

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=150)
    x = np.arange(len(versions))
    colors = [BLUE] * 5 + [ORANGE]
    bars = ax.bar(x, ratios, width=0.62, color=colors, alpha=0.92, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 14000)
    ax.axhspan(1.15, 1.20, color="#d9f0e2", alpha=0.7, zorder=1)
    ax.axhline(gptq, color=RED, ls="--", lw=1.6, zorder=4)
    ax.text(len(versions) - 0.45, gptq * 1.03, f"GPTQ = {gptq}", color=RED,
            fontsize=10, ha="right")
    ax.text(len(versions) - 0.45, 1.176 / 1.09, "зона паритета", color="#2e7d4f",
            fontsize=9, ha="right")
    for xi, r in zip(x, ratios):
        lab = f"{r:.0f}×" if r >= 10 else f"{r:.2f}×"
        ax.text(xi, r * 1.18, lab, ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x, versions, fontsize=9)
    ax.set_ylabel("PPL ratio (SmolLM-135M), log scale")
    _style_ax(ax)
    fig.tight_layout()
    p = OUT / "fig_versions.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"[ok] {p}")


# ── fig 2: CHMC vs GPTQ по моделям (равный бюджет) ────────────────────
def make_ppl():
    names = ["SmolLM-135M", "Qwen2.5-0.5B", "TinyLlama-1.1B"]
    chmc = [1.1825, 1.174, 1.059]     # smollm: середина диапазона 1.179–1.186 (3–4 сида)
    gptq = [1.176, 1.141, 1.081]

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
    x = np.arange(len(names))
    w = 0.36
    b1 = ax.bar(x - w / 2, chmc, w, color=BLUE, label="CHMC v6", zorder=3)
    b2 = ax.bar(x + w / 2, gptq, w, color=GRAY, label="GPTQ (BPW 4.2875)", zorder=3)

    chmc_lab = ["1.179–1.186", "1.174*", "1.059"]
    gptq_lab = [f"{v:.3f}" for v in gptq]
    for xi, v, lab in zip(x - w / 2, chmc, chmc_lab):
        ax.text(xi, v + 0.0018, lab, ha="center", va="bottom", fontsize=9.5)
    for xi, lab in zip(x + w / 2, gptq_lab):
        ax.text(xi, float(lab) + 0.0018, lab, ha="center", va="bottom", fontsize=9.5)

    # подпись выигрыша tinyllama
    ax.annotate("победа CHMC\n(более 30σ)", xy=(2 - w / 2, 1.059),
                xytext=(1.45, 1.045), fontsize=9, color="#2e7d4f", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#2e7d4f", lw=1.2))

    ax.set_ylim(1.03, 1.21)
    ax.set_xticks(x, names, fontsize=10)
    ax.axhline(1.0, color="none")
    ax.set_ylabel("PPL ratio (compressed / baseline)")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _style_ax(ax)
    fig.text(0.99, 0.01, "* одиночный прогон (репы впереди)", ha="right",
             fontsize=8, color="#5a6472", style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    p = OUT / "fig_ppl.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"[ok] {p}")


# ── fig 3: «бутылочное горлышко» по блокам (из .pt TDA) ───────────────
def make_bottleneck():
    """Два честных профиля по всем блокам:
    L — доля энергии top-3 ПК ОБЪЕДИНЁНОГО центрированного облака [pre;post]
        (та же величина, что «PCA-3D var» в заголовках панелей project_3d.py);
    R — средний попарный косинус pre-токенов того же блока (анизотропия конуса).
    """
    N_SUB = 512
    data = {}
    for m in MODELS:
        payload = torch.load(TDA_DIR / f"tda_activations_{m}.pt", map_location="cpu")
        pre_t, post_t = payload["pre"], payload["post"]
        n = len(pre_t)
        idx_all = np.arange(0, N_SUB * 4, 4)[:N_SUB] if N_SUB * 4 <= 2048 else None
        joint, cosines = [], []
        for i in range(n):
            A = _act2d(pre_t[i])
            B = _act2d(post_t[i])
            step = max(1, len(A) // N_SUB)
            idx = np.arange(0, len(A), step)[:N_SUB]
            Xc = np.vstack([A[idx], B[idx]])
            Xc = Xc - Xc.mean(axis=0)
            _, S, _ = np.linalg.svd(Xc, full_matrices=False)
            e = S ** 2
            joint.append(float(e[:3].sum() / e.sum()))
            Xn = A[idx] / np.maximum(np.linalg.norm(A[idx], axis=1), 1e-12)[:, None]
            C = Xn @ Xn.T
            cosines.append(float((C.sum() - len(idx)) / (len(idx) * (len(idx) - 1))))
        data[m] = (np.array(joint), np.array(cosines))
        j, c = data[m]
        print(f"[{m}] n_blocks={n}")
        print(f"  joint top3: min={j.min():.3f} max={j.max():.3f}@blk{int(j.argmax())} "
              f"| edges blk0={j[0]:.3f}, blk{n-1}={j[-1]:.3f}")
        print(f"  mean cos : min={c.min():.3f} max={c.max():.3f}@blk{int(c.argmax())} "
              f"| edges blk0={c[0]:.3f}, blk{n-1}={c[-1]:.3f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150)
    for m in MODELS:
        j, c = data[m]
        axL.plot(np.arange(len(j)), j * 100, "-o", ms=3, lw=1.6, color=MCOLORS[m], label=m)
        axR.plot(np.arange(len(c)), c, "-o", ms=3, lw=1.6, color=MCOLORS[m])
    axL.set_ylim(0, 105)
    axL.set_xlabel("decoder block index (выход блока)")
    axL.set_ylabel("доля энергии top-3 ПК\nобъединённого облака [pre;post], %")
    axL.legend(fontsize=9, frameon=False)
    _style_ax(axL)
    axR.axhline(0.0, color="#c3cad4", lw=0.8)
    axR.set_xlabel("decoder block index (выход блока)")
    axR.set_ylabel("средний попарный косинус\npre-токенов")
    _style_ax(axR)
    fig.suptitle("«Бутылочное горлышко»: после первых 1–2 блоков почти вся сеть лежит в "
                 "низкоразмерном конусе; края сети — изотропнее", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUT / "fig_bottleneck.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"[ok] {p}")


if __name__ == "__main__":
    make_versions()
    make_ppl()
    make_bottleneck()
