#!/usr/bin/env python3
"""Полный per-block профиль (joint top-3 + mean cos) для проверки подписей."""
import numpy as np
import torch

from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent / "tda_analysis")
N_SUB = 512
for m in ["smollm-135m", "qwen2.5-0.5b", "tinyllama-1.1b"]:
    p = torch.load(f"{ROOT}/tda_activations_{m}.pt", map_location="cpu")
    pre, post = p["pre"], p["post"]
    n = len(pre)

    def act(t):
        X = t.numpy()
        return X[0] if X.ndim == 3 else X

    rows = []
    for i in range(n):
        A = act(pre[i]).astype(np.float64)
        B = act(post[i]).astype(np.float64)
        step = max(1, len(A) // N_SUB)
        idx = np.arange(0, len(A), step)[:N_SUB]
        Xc = np.vstack([A[idx], B[idx]]) - 0.5 * (np.mean(A[idx], 0) + np.mean(B[idx], 0))
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
        e = S ** 2
        j3 = float(e[:3].sum() / e.sum())
        Xn = A[idx] / (np.linalg.norm(A[idx], axis=1)[:, None])
        C = Xn @ Xn.T
        mc = float((C.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
        rows.append((i, j3, mc))

    print(f"=== {m} ===")
    for i, j3, mc in rows:
        bar = "#" * int(j3 * 40)
        print(f" blk{i:2d}: joint_top3={j3*100:5.1f}% mean_cos={mc:+.3f} {bar}")
