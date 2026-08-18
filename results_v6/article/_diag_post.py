#!/usr/bin/env python3
"""Кто несёт концентрацию: pre-alone vs post-alone (centered top-3), 4 блока на модель."""
import numpy as np
import torch

from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent / "tda_analysis")
for m in ["smollm-135m", "qwen2.5-0.5b", "tinyllama-1.1b"]:
    p = torch.load(f"{ROOT}/tda_activations_{m}.pt", map_location="cpu")
    pre, post = p["pre"], p["post"]
    n = len(pre)

    def act(t):
        X = t.numpy()
        return X[0] if X.ndim == 3 else X

    print(f"=== {m} ===")
    for bi in [0, n // 4, n // 2, n - 1]:
        A = act(pre[bi]).astype(np.float64)
        B = act(post[bi]).astype(np.float64)
        out = []
        for name, X in (("pre", A), ("post", B)):
            Xc = X - X.mean(0)
            s, _, _ = np.linalg.svd(Xc, full_matrices=False)
            e = s ** 2
            out.append(f"{name} top3={e[:3].sum()/e.sum():.4f}")
        print(f" blk{bi:2d}: " + " | ".join(out))
