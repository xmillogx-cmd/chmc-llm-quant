#!/usr/bin/env python3
"""d_k per block: the minimum k such that the top-k PCs explain >= 90% / >= 50% of the energy.
Verifying the article's claim "9 out of 576 dimensions" (pre-activations, centered)."""
import numpy as np
import torch

from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent / "tda_analysis")
N_SUB = 512


def act(t):
    X = t.numpy()
    return X[0] if X.ndim == 3 else X


def d_k(e, thr):
    cum = np.cumsum(e) / e.sum()
    k = int(np.searchsorted(cum, thr) + 1)
    return min(k, len(e))


for m in ["smollm-135m", "qwen2.5-0.5b", "tinyllama-1.1b"]:
    p = torch.load(f"{ROOT}/tda_activations_{m}.pt", map_location="cpu")
    pre, post = p["pre"], p["post"]
    n = len(pre)
    print(f"=== {m} ===")
    for i in range(n):
        A = act(pre[i]).astype(np.float64)
        step = max(1, len(A) // N_SUB)
        idx = np.arange(0, len(A), step)[:N_SUB]
        Xc = A[idx] - A[idx].mean(0)
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        e = s ** 2
        d10, d50, d90 = d_k(e, 0.10), d_k(e, 0.50), d_k(e, 0.90)
        print(f" blk{i:2d}: d10={d10:3d} d50={d50:4d} d90={d90:4d}")
