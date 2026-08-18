#!/usr/bin/env python3
"""
cpu_sanity_whitening.py — CPU-only sanity check for the v7 whitening branch.

No model load, no GPU. Tiny tensors. Checks:
  1. REDUCTION: with diagonal C (X = I), whitening == diagonal baseline
     (same seed) — strict generalization holds.
  2. PATH DIFFERS: with correlated C, whitening != baseline (branch is wired
     and actually changes the transform).
  3. OBJECTIVE: whitening better minimizes the true data-weighted error
     ||(W - W_lr) @ C^{1/2}||_F than the diagonal baseline (correlated C).
  4. WIRING: compress_layer_v6(whitening=True) differs from (whitening=False)
     end-to-end on correlated X (flag is passed through and branch executes).
"""

import sys
from pathlib import Path

import torch

V6_DIR = Path(__file__).resolve().parent.parent / "v6"
sys.path.insert(0, str(V6_DIR))
import chmc_v6 as C  # noqa: E402

torch.manual_seed(0)
OUT_F, IN_F, N_TOK = 8, 16, 64
W = torch.randn(OUT_F, IN_F)

# correlated calibration: X = Z @ A.T  ->  C = A (Z^T Z) A^T / n (non-diagonal)
Z = torch.randn(N_TOK, IN_F)
A = torch.randn(IN_F, IN_F)
X_corr = Z @ A.T

# diagonal calibration: X = I  ->  C = I/n (diagonal)
X_diag = torch.eye(IN_F)

DAMP = 0.05
RANK = 4
NITER = 5


def baseline_lr(W, X, rank, damp, niter):
    """Replicates the v6 diagonal-damping branch (compute_damped_covariance)."""
    diag_c = (X ** 2).mean(dim=0)
    diag_c = diag_c + damp * diag_c.mean().clamp(min=1e-8)
    Ww = W * torch.sqrt(diag_c).unsqueeze(0)
    U, S, Vt = torch.svd_lowrank(Ww, q=rank, niter=niter)
    return ((U * S.unsqueeze(0)) @ Vt.T) / torch.sqrt(diag_c).unsqueeze(0)


def whitened_lr(W, X, rank, damp, niter):
    """Replicates the v7 whitening branch in chmc_v6.py."""
    n = max(X.shape[0], 1)
    Cc = (X.T @ X) / n
    Cc = Cc + damp * Cc.diagonal().mean().clamp_min(1e-12) * torch.eye(
        Cc.shape[0], device=Cc.device, dtype=Cc.dtype)
    evals, evecs = torch.linalg.eigh(Cc)
    evals = evals.clamp_min(1e-12)
    se = torch.sqrt(evals)
    C_sqrt = (evecs * se.unsqueeze(0)) @ evecs.T
    C_inv_sqrt = (evecs / se.unsqueeze(0)) @ evecs.T
    Ww = W @ C_sqrt
    U, S, Vt = torch.svd_lowrank(Ww, q=rank, niter=niter)
    return ((U * S.unsqueeze(0)) @ Vt.T) @ C_inv_sqrt


def rel_diff(a, b):
    return (a - b).norm().item() / max(a.norm().item(), 1e-12)


ok = True

# 1. REDUCTION (diagonal C)
torch.manual_seed(42)
w_base = baseline_lr(W, X_diag, RANK, DAMP, NITER)
torch.manual_seed(42)
w_white = whitened_lr(W, X_diag, RANK, DAMP, NITER)
d1 = rel_diff(w_base, w_white)
p1 = d1 < 1e-5
ok &= p1
print(f"1. REDUCTION (X=I):   rel diff = {d1:.2e}  -> {'PASS' if p1 else 'FAIL'}")

# 2. PATH DIFFERS (correlated C)
torch.manual_seed(42)
w_base2 = baseline_lr(W, X_corr, RANK, DAMP, NITER)
torch.manual_seed(42)
w_white2 = whitened_lr(W, X_corr, RANK, DAMP, NITER)
d2 = rel_diff(w_base2, w_white2)
p2 = d2 > 1e-3
ok &= p2
print(f"2. PATH DIFFERS:      rel diff = {d2:.2e}  -> "
      f"{'PASS (branch active)' if p2 else 'FAIL (paths identical?)'}")

# 3. OBJECTIVE (true data-weighted error, undamped C)
Cc = (X_corr.T @ X_corr) / N_TOK
evals, evecs = torch.linalg.eigh(Cc.clamp_min(1e-12))
C_sqrt = (evecs * torch.sqrt(evals).unsqueeze(0)) @ evecs.T
obj_base = ((W - w_base2) @ C_sqrt).norm().item()
obj_white = ((W - w_white2) @ C_sqrt).norm().item()
p3 = obj_white < obj_base
ok &= p3
print(f"3. OBJECTIVE:         baseline={obj_base:.4f}  whitened={obj_white:.4f}  -> "
      f"{'PASS (whitened better)' if p3 else 'CHECK (randomized SVD noise?)'}")

# 4. WIRING through compress_layer_v6 (end-to-end, CPU)
torch.manual_seed(42)
Wc_base, _ = C.compress_layer_v6(
    W, X_corr, rank=RANK, residual_bits=4, group_size=8,
    use_compensation=True, dampening=DAMP, niter=NITER, whitening=False)
torch.manual_seed(42)
Wc_white, _ = C.compress_layer_v6(
    W, X_corr, rank=RANK, residual_bits=4, group_size=8,
    use_compensation=True, dampening=DAMP, niter=NITER, whitening=True)
d4 = rel_diff(Wc_base, Wc_white)
p4 = d4 > 1e-4
ok &= p4
print(f"4. WIRING e2e:        rel diff = {d4:.2e}  -> "
      f"{'PASS (flag wired)' if p4 else 'FAIL (flag not taking effect)'}")

print()
print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
