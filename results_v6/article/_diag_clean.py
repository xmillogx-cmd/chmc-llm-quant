#!/usr/bin/env python3
"""CLEAN per-block profile (correct SVD: _, s, _ = svd):
pre-alone / post-alone / combined [pre;post] centered top-3 + mean cos(pre_i, post_i)."""
import numpy as np
import torch

from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent / "tda_analysis")
N_SUB = 512


def act(t):
    X = t.numpy()
    return X[0] if X.ndim == 3 else X


def top3_frac(Xc):
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    e = s ** 2
    return float(e[:3].sum() / e.sum())


for m in ["smollm-135m", "qwen2.5-0.5b", "tinyllama-1.1b"]:
    p = torch.load(f"{ROOT}/tda_activations_{m}.pt", map_location="cpu")
    pre, post = p["pre"], p["post"]
    n = len(pre)
    print(f"=== {m} ===")
    for i in range(n):
        A = act(pre[i]).astype(np.float64)
        B = act(post[i]).astype(np.float64)
        step = max(1, len(A) // N_SUB)
        idx = np.arange(0, len(A), step)[:N_SUB]
        Pp, Qq = A[idx], B[idx]
        fp = top3_frac(Pp - Pp.mean(0))
        fq = top3_frac(Qq - Qq.mean(0))
        Xc = np.vstack([Pp, Qq])
        fj = top3_frac(Xc - Xc.mean(0))
        An = Pp / (np.linalg.norm(Pp, axis=1)[:, None] + 1e-12)
        Bn = Qq / (np.linalg.norm(Qq, axis=1)[:, None] + 1e-12)
        cp = float((An * Bn).sum(1).mean())
        print(f" blk{i:2d}: pre={fp*100:6.2f}% post={fq*100:6.2f}% joint={fj*100:6.2f}% cosAB={cp:+.3f}")
