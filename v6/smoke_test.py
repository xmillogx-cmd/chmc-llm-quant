#!/usr/bin/env python3
"""Smoke test: exercise every v6 path on one synthetic layer (no model load)."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from chmc_v6 import (
    compress_layer_v6, quantize_with_compensation_strict,
    quantize_groupwise_v2, accumulate_hessian, find_best_dampening,
    random_hadamard_rotation, compute_block_covariance,
    angular_quantize_residual, tangent_space_svd,
    allocate_ranks_bit_budget,
)
from bpw import chmc_bpw, rank_for_bpw

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
W = torch.randn(576, 576, device=DEV) * 0.02
X = torch.randn(256, 576, device=DEV)

print(f"device={DEV}")
print("\n[1] BPW allocation (target 4.2875)")
shapes = [("q", 576, 576), ("k", 576, 192), ("gate", 1536, 576), ("down", 576, 1536)]
ranks = allocate_ranks_bit_budget(shapes, 4.2875, 4, 128)
print("   ranks:", ranks)
for n, o, i in shapes:
    d = chmc_bpw(o, i, ranks[n], 4, 128)
    print(f"   {n}: rank={ranks[n]} bpw={d['bpw']:.4f}")

print("\n[2] compress_layer_v6 paths (recon_err should be small & finite)")
configs = {
    "baseline_v5":   {},
    "A1_strict":     {"strict_sequential": True},
    "A2_groupdim1":  {"group_dim": 1},
    "A3_hess8":      {"hessian_batches": 8},
    "A4_damp0.05":   {"dampening": 0.05},
    "B1_hadamard":   {"hadamard": True},
    "B2_blockcov":   {"block_cov": True},
    "B4_angular":    {"use_compensation": False, "angular": True},
    "B5_tangent":    {"tangent": True},
    "A1+A2":         {"strict_sequential": True, "group_dim": 1},
}
for name, cfg in configs.items():
    Wc, stats = compress_layer_v6(W, X, rank=4, **cfg)
    ok = torch.isfinite(Wc).all().item() and Wc.shape == W.shape
    print(f"   {name:<14} recon_err={stats['recon_err_ratio']:.5f}  finite={ok}")
    assert ok, f"{name} produced non-finite/wrong-shape output"

print("\n[3] individual functions")
R = W - (W @ W.T @ W) * 0  # just use a residual-like matrix
R = torch.randn(576, 576, device=DEV) * 0.01
q_strict = quantize_with_compensation_strict(R, X, bits=4, group_size=128)
print(f"   strict: {q_strict.shape} finite={torch.isfinite(q_strict).all().item()}")
q_g1 = quantize_groupwise_v2(R, bits=4, group_size=128, group_dim=1)[0]
q_g0 = quantize_groupwise_v2(R, bits=4, group_size=128, group_dim=0)[0]
print(f"   groupdim1: {q_g1.shape}  groupdim0: {q_g0.shape}")
H8 = accumulate_hessian(X, n_batches=8)
print(f"   hess8: {H8.shape} finite={torch.isfinite(H8).all().item()}")
lam, err = find_best_dampening(R, X, bits=4, group_size=128)
print(f"   best_dampening: {lam} (score {err:.4f})")
Rh = random_hadamard_rotation(576, seed=42, device=DEV)
orth = (Rh @ Rh.T - torch.eye(576, device=DEV)).abs().max().item()
print(f"   hadamard: {Rh.shape} orth_err={orth:.2e}")
Cb = compute_block_covariance(X, block_size=16)
print(f"   block_cov: {Cb.shape} finite={torch.isfinite(Cb).all().item()}")
qa = angular_quantize_residual(R, bits=4, group_size=128)
print(f"   angular: {qa.shape} finite={torch.isfinite(qa).all().item()}")
Wt = tangent_space_svd(W, X, rank=4)
print(f"   tangent: {Wt.shape} finite={torch.isfinite(Wt).all().item()}")

print("\nALL SMOKE TESTS PASSED")
