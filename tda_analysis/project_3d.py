#!/usr/bin/env python3
"""
project_3d.py — 3D projection of activations BEFORE/AFTER CHMC v6 compression (CPU)
===================================================================================

Answers the question: "what does quantization do to each block's activation cloud?".
From the .pt produced by collect_activations.py we take pre/post block activations
and build one panel for each of 4 representative blocks [0, n//4, n/2, n-1]:

  * points = calibration tokens (deterministic subsample, ~512);
  * the PCA-3D basis is computed ONCE on the centered union [pre; post] —
    both clouds are projected into one coordinate system (numpy SVD, deterministic);
  * blue points = pre (unquantized), orange points = post (after CHMC v6);
  * gray lines = token correspondence: row i in pre is the same calibration
    position as row i in post (~150 lines via a deterministic stride).

Output: results_v6/tda_3d/<model>_activations_3d.png (SEPARATE folder — NOT part of
the "exactly 5 artifacts" rule for results_v6/tda_analysis/). The log prints the
mean relative shift ||pre-post||/||pre|| per block and the fraction of variance
explained by the first 3 components.

Usage:
    python tda_analysis/project_3d.py --data results_v6/tda_analysis/tda_activations_smollm-135m.pt

Logging: all output is duplicated to tda_analysis/logs/project3d_<model>_<time>.log.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent          # tda_analysis/
ROOT_DIR = BASE_DIR.parent                          # cmq_experiment/

import tda_log                                       # noqa: E402

N_PTS = 512        # tokens per cloud (deterministic subsample)
N_LINES = 150      # correspondence lines pre->post


def _act2d(t: torch.Tensor) -> np.ndarray:
    """Block activations -> (N, d) float64. Old .pt files store (B=1, N, d)."""
    X = t.numpy()
    if X.ndim == 3:
        assert X.shape[0] == 1, f"unexpected batch={X.shape[0]}"
        X = X[0]
    return np.ascontiguousarray(X, dtype=np.float64)


def _token_idx(n: int, m: int) -> np.ndarray:
    """Deterministic subsample of ~m token indices (uniform step)."""
    if n <= m:
        return np.arange(n)
    step = n // m
    idx = np.arange(0, n, step)[:m]
    return idx


def _rel_disp(A: np.ndarray, B: np.ndarray) -> float:
    """Mean relative displacement ||pre-post||/||pre|| over tokens."""
    na = np.linalg.norm(A, axis=1)
    d = np.linalg.norm(A - B, axis=1)
    return float(np.mean(d / np.maximum(na, 1e-12)))


def main():
    ap = argparse.ArgumentParser(description="3D projection of activations before/after compression")
    ap.add_argument("--data", required=True, help="path to tda_activations_<model>.pt")
    ap.add_argument("--out-dir", default=str(ROOT_DIR / "results_v6" / "tda_3d"))
    args = ap.parse_args()

    data_file = Path(args.data)
    model = data_file.stem.replace("tda_activations_", "")
    log_path = tda_log.log_file_for("project3d", model, base_dir=BASE_DIR)
    start_f = tda_log.start_logging(log_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_file = out_dir / f"{model}_activations_3d.png"

    payload = torch.load(data_file, map_location="cpu")
    pre_t, post_t = payload["pre"], payload["post"]
    n_blocks = len(pre_t)
    hidden = int(payload.get("hidden_size", _act2d(pre_t[0]).shape[1]))
    ppl = payload.get("ppl", {})

    print(f"Model={model} | blocks={n_blocks} | hidden={hidden}")
    if isinstance(ppl, dict):
        print(f"PPL baseline={ppl.get('baseline')} compressed={ppl.get('compressed')}"
              f" ratio={ppl.get('ratio')}")

    # 4 representative blocks: first, quarter, half, last (no duplicates)
    cand = [0, n_blocks // 4, n_blocks // 2, n_blocks - 1]
    blocks = sorted(set(cand))

    fig, axes = plt.subplots(2, 2, figsize=(15, 13), subplot_kw={"projection": "3d"})
    axes = axes.ravel()

    for p, bi in enumerate(blocks):
        A_pre, A_post = _act2d(pre_t[bi]), _act2d(post_t[bi])
        idx = _token_idx(A_pre.shape[0], N_PTS)
        Pp, Qq = A_pre[idx], A_post[idx]

        # Common PCA-3D basis on the centered union [pre; post]
        X = np.vstack([Pp, Qq])
        Xc = X - X.mean(axis=0)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        V3 = Vt[:3].T
        var_ratio = float((S[:3] ** 2).sum() / (S ** 2).sum())
        Yp, Yq = Xc[: Pp.shape[0]] @ V3, Xc[Pp.shape[0]:] @ V3

        ax = axes[p]
        # per-token correspondence lines (~150 via a uniform step)
        li = _token_idx(len(idx), N_LINES)
        for j in li:
            ax.plot([Yp[j, 0], Yq[j, 0]], [Yp[j, 1], Yq[j, 1]], [Yp[j, 2], Yq[j, 2]],
                    color="0.7", lw=0.4, alpha=0.35, zorder=1)
        ax.scatter(Yp[:, 0], Yp[:, 1], Yp[:, 2], s=9, c="#1f6fd6", alpha=0.55, label="pre (FP)", zorder=2)
        ax.scatter(Yq[:, 0], Yq[:, 1], Yq[:, 2], s=9, c="#e8871a", alpha=0.55, label="post (CHMC v6)", zorder=3)

        rd = _rel_disp(A_pre, A_post)
        ax.set_title(f"block {bi} | rel.disp={rd:.4f} | PCA-3D var={var_ratio:.1%}", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        if p % 2 == 0:
            ax.set_zlabel("PC3")
        ax.legend(loc="best", fontsize=8)

    supt = f"{model}: pre activations (blue) vs post CHMC v6 (orange), lines = same token"
    if isinstance(ppl, dict) and ppl.get("ratio"):
        supt += f" | PPL ratio={ppl['ratio']:.4f}"
    fig.suptitle(supt, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(png_file, dpi=150)
    plt.close(fig)

    print(f"PNG: {png_file}")
    start_f.close()


if __name__ == "__main__":
    main()
