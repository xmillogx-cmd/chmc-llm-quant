#!/usr/bin/env python3
"""Temporal diagnostics: block activation spectrum (pre) + replication of the var_ratio from project_3d."""
import numpy as np
import torch

from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent / "tda_analysis")
for m in ["smollm-135m", "qwen2.5-0.5b", "tinyllama-1.1b"]:
    p = torch.load(f"{ROOT}/tda_activations_{m}.pt", map_location="cpu")
    pre, post = p["pre"], p["post"]
    n = len(pre)
    print(f"=== {m} (blocks={n}, hidden={p['hidden_size']}) ===")
    def act(t):
        X = t.numpy()
        return X[0] if X.ndim == 3 else X

    for bi in [0, n // 4, n // 2, n - 1]:
        A = act(pre[bi]).astype(np.float64)
        B = act(post[bi]).astype(np.float64)
        Ac = A - A.mean(0)
        s, _, _ = np.linalg.svd(Ac, full_matrices=False)
        e = s ** 2
        idx = np.arange(0, len(A), 4)[:512]
        Xs = A[idx] / (np.linalg.norm(A[idx], axis=1)[:, None])
        C = Xs @ Xs.T
        mean_cos = float((C.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
        Pp, Qq = A[idx], B[idx]
        Xc = np.vstack([Pp, Qq])
        Xc = Xc - Xc.mean(0)
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
        e2 = S ** 2
        print(f" blk {bi:2d}: pre top3={e[:3].sum()/e.sum():.4f} top10={e[:10].sum()/e.sum():.4f} "
              f"| mean_cos_tokens={mean_cos:+.3f} | combined(pre;post) top3={e2[:3].sum()/e2.sum():.4f}")
