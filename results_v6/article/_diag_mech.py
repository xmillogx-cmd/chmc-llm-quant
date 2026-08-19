#!/usr/bin/env python3
"""Mechanism of concentration in the combined cloud [pre;post] on qwen blk3 (and smollm blk15)."""
import numpy as np
import torch

from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent / "tda_analysis")


def act(t):
    X = t.numpy()
    return X[0] if X.ndim == 3 else X


for m, bi in [("qwen2.5-0.5b", 3), ("smollm-135m", 15)]:
    p = torch.load(f"{ROOT}/tda_activations_{m}.pt", map_location="cpu")
    A = act(p["pre"][bi]).astype(np.float64)
    B = act(p["post"][bi]).astype(np.float64)
    idx = np.arange(0, len(A), 4)[:512]
    Pp, Qq = A[idx], B[idx]

    muA, muB = Pp.mean(0), Qq.mean(0)
    d_mu = float(np.linalg.norm(muA - muB))
    spreadA = float(np.sqrt(((Pp - muA) ** 2).sum() / len(Pp)))
    spreadB = float(np.sqrt(((Qq - muB) ** 2).sum() / len(Qq)))

    An = Pp / np.linalg.norm(Pp, axis=1)[:, None]
    Bn = Qq / np.linalg.norm(Qq, axis=1)[:, None]
    cosAB = float((An * Bn).sum(1).mean())

    def top5(X):
        s, _, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
        return s[:5], float((s ** 2).sum())

    sA, EA = top5(Pp)
    sB, EB = top5(Qq)
    Xc = np.vstack([Pp, Qq]) - np.vstack([Pp, Qq]).mean(0)
    sX, EX = top5(Xc)

    print(f"=== {m} blk{bi} ===")
    print(f"  E(A)={EA:.3e}  E(B)={EB:.3e}  ratio B/A={EB/EA:.2f}")
    print(f"  |muA-muB|={d_mu:.4f}   spreadA={spreadA:.4f}  spreadB={spreadB:.4f}")
    print(f"  d_mu/spreadA={d_mu/spreadA:.3f}  d_mu/spreadB={d_mu/spreadB:.3f}")
    print(f"  mean cos(A_i, B_i) = {cosAB:+.3f}")
    print(f"  top5 s(A_c): {np.round(sA, 4)}")
    print(f"  top5 s(B_c): {np.round(sB, 4)}")
    print(f"  top5 s(Xc) : {np.round(sX, 4)}   (top3 frac = {(sX[:3]**2).sum()/(sX**2).sum():.4f})")
