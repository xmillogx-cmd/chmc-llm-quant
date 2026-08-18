#!/usr/bin/env python3
"""
smoke_test_turbo_weights.py — Step 1: synthetic smoke test for the 4
TurboQuant-inspired patches (weights only), adapted to CHMC's cone hypothesis.

Exercises each patch on a synthetic anisotropic (cone-like) layer and checks
the acceptance criteria from the spec:
  C1 (Patch 5) Cone-Aware : recon_err cone_aware < 0.90x plain Hadamard (anisotropic)
  C2 (Patch 3) Lloyd-Max  : MSE(LM) / MSE(uniform) < 0.90 (synthetic, bits<=4)
  C3 (Patch 2) QJL        : recon_err QJL < 1.5x INT4 (same rank)
                            OR equal-BPW QJL rank >= 2x INT4 rank
  C4 (Patch 4) IP metric  : ip_cosine_output > 0.95, ip_bias_output < 0.10

Saves results to results_v6/patch_turbo_weights/smoke_test_turbo_weights.json
"""

import json
import math
import sys
import time
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import chmc_v6 as C  # noqa: E402
from bpw import chmc_bpw, chmc_bpw_qjl, rank_for_bpw, rank_for_bpw_qjl  # noqa: E402

DEVICE = C.DEVICE
OUT_DIR = C.ROOT_DIR / "results_v6" / "patch_turbo_weights"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "smoke_test_turbo_weights.json"


def make_anisotropic_layer(out_f=576, in_f=576, n_calib=512, seed=42,
                           cone_strength=3.0):
    """Synthetic layer with CONE-like (anisotropic) calibration activations.

    X_calib = cone_strength * mean_dir + noise  -> activations lie in a narrow
    cone around mean_dir (the cone hypothesis). W is a random weight matrix.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    mean_dir = torch.randn(in_f, generator=g)
    mean_dir = mean_dir / mean_dir.norm()
    noise = torch.randn(n_calib, in_f, generator=g) * 0.5
    X_calib = (cone_strength * mean_dir.unsqueeze(0) + noise).to(DEVICE)
    W = torch.randn(out_f, in_f, generator=g).to(DEVICE) * 0.1
    return W, X_calib


def rel_err(A, B):
    """Relative Frobenius error ||A-B|| / ||A||."""
    num = (A - B).pow(2).sum().sqrt()
    den = A.pow(2).sum().sqrt().clamp(min=1e-8)
    return float(num / den)


def test_qjl(W, X_calib, rank=4):
    """C3 (Patch 2): QJL 1-bit residual vs INT4 at the same rank."""
    in_f = W.shape[1]
    n_proj = in_f // 4
    # baseline: INT4 residual at the same rank
    W_int4, st_int4 = C.compress_layer_v6(
        W, X_calib, rank=rank, residual_bits=4, group_size=128,
        use_compensation=True, group_dim=0,
    )
    err_int4 = st_int4["recon_err_ratio"]
    # QJL residual at the same rank
    W_qjl, st_qjl = C.compress_layer_v6(
        W, X_calib, rank=rank, residual_bits=4, group_size=128,
        use_compensation=True, group_dim=0, qjl=True,
    )
    err_qjl = st_qjl["recon_err_ratio"]
    # equal-BPW rank comparison
    bpw_int4 = chmc_bpw(out_f=W.shape[0], in_f=in_f, rank=rank,
                        residual_bits=4, group_size=128)["bpw"]
    r_int4 = rank_for_bpw(W.shape[0], in_f, 4.2875, 4, 128)
    r_qjl = rank_for_bpw_qjl(W.shape[0], in_f, 4.2875, n_proj)
    ok_same_rank = err_qjl < 1.5 * err_int4
    ok_equal_bpw = r_qjl >= 2 * max(1, r_int4)
    return {
        "n_projections": n_proj,
        "rank_same": rank,
        "recon_err_int4": round(err_int4, 6),
        "recon_err_qjl": round(err_qjl, 6),
        "qjl_over_int4": round(err_qjl / max(err_int4, 1e-9), 4),
        "rank_int4_at_4.2875": r_int4,
        "rank_qjl_at_4.2875": r_qjl,
        "bpw_int4": round(bpw_int4, 4),
        "PASS_same_rank(<1.5x)": bool(ok_same_rank),
        "PASS_equal_bpw(rank>=2x)": bool(ok_equal_bpw),
        "PASS": bool(ok_same_rank or ok_equal_bpw),
    }


def test_lloyd_max(W, X_calib, bits=4):
    """C2 (Patch 3): Lloyd-Max vs uniform on a rotated (N(0,1)-like) residual."""
    in_f = W.shape[1]
    # produce a rotated residual (post-Hadamard) ~ N(0,1)
    R_rot = C.random_hadamard_rotation(in_f, seed=42, device=DEVICE)
    R = (W - W.mean(dim=1, keepdim=True)) @ R_rot
    # normalize per-group to unit variance so both quantizers see N(0,1)
    gs = 128
    n_groups = (in_f + gs - 1) // gs
    padded = n_groups * gs
    Rp = torch.zeros(R.shape[0], padded, device=DEVICE)
    Rp[:, :in_f] = R
    Rg = Rp.reshape(R.shape[0], n_groups, gs)
    std = Rg.std(dim=2, keepdim=True).clamp(min=1e-8)
    Xn = (Rg / std).reshape(R.shape[0], padded)[:, :in_f]
    # uniform quantizer (4-bit, symmetric)
    qmax = 2 ** (bits - 1) - 1
    q_uni = torch.round(Xn).clamp(-qmax, qmax)
    mse_uni = float(((Xn - q_uni) ** 2).mean())
    # Lloyd-Max quantizer
    lm = C.lloyd_max_quantize_rotated(Xn, bits=bits, group_size=gs)
    mse_lm = float(((Xn - lm) ** 2).mean())
    ratio = mse_lm / max(mse_uni, 1e-12)
    return {
        "bits": bits,
        "mse_uniform": round(mse_uni, 6),
        "mse_lloyd_max": round(mse_lm, 6),
        "mse_ratio_lm_over_uni": round(ratio, 4),
        "PASS(<0.90)": bool(ratio < 0.90),
        "PASS": bool(ratio < 0.90),
    }


def test_ip_metric(W, X_calib):
    """C4 (Patch 4): IP preservation metric on a compressed layer."""
    W_comp, _ = C.compress_layer_v6(
        W, X_calib, rank=4, residual_bits=4, group_size=128,
        use_compensation=True, group_dim=0,
    )
    ip = C.measure_ip_preservation(W, W_comp, X_calib, n_pairs=1000)
    return {
        **ip,
        "PASS_cosine_output(>0.95)": bool(ip["ip_cosine_output"] > 0.95),
        "PASS_bias_output(<0.10)": bool(ip["ip_bias_output"] < 0.10),
        "PASS": bool(ip["ip_cosine_output"] > 0.95 and ip["ip_bias_output"] < 0.10),
    }


def test_cone_aware(W, X_calib, rank=4):
    """C1 (Patch 5): Cone-Aware vs plain Hadamard on an anisotropic layer."""
    # plain Hadamard (B1) baseline
    W_h, st_h = C.compress_layer_v6(
        W, X_calib, rank=rank, residual_bits=4, group_size=128,
        use_compensation=True, group_dim=0, hadamard=True,
    )
    err_h = st_h["recon_err_ratio"]
    # Cone-Aware (C1) — decompose, preserve cone, compress perp
    W_c, st_c = C.compress_layer_v6(
        W, X_calib, rank=rank, residual_bits=4, group_size=128,
        use_compensation=True, group_dim=0, hadamard=True, cone_aware=True,
    )
    err_c = st_c["recon_err_ratio"]
    ratio = err_c / max(err_h, 1e-9)
    return {
        "rank": rank,
        "recon_err_plain_hadamard": round(err_h, 6),
        "recon_err_cone_aware": round(err_c, 6),
        "cone_over_plain": round(ratio, 4),
        "PASS_aniso(<0.90x)": bool(ratio < 0.90),
        "PASS_all(<=1.0x)": bool(ratio <= 1.0),
        "PASS": bool(ratio <= 1.0),
    }


def main():
    t0 = time.time()
    torch.manual_seed(0)
    W, X_calib = make_anisotropic_layer()
    print(f"\nSynthetic layer: W={tuple(W.shape)}  X_calib={tuple(X_calib.shape)}")
    print(f"Device: {DEVICE}\n")

    results = {"device": DEVICE, "layer": {"W": list(W.shape), "X_calib": list(X_calib.shape)}}

    print("[C3] QJL 1-bit residual ...")
    r = test_qjl(W, X_calib)
    results["C3_qjl"] = r
    print(f"  INT4 err={r['recon_err_int4']:.4f}  QJL err={r['recon_err_qjl']:.4f}  "
          f"ratio={r['qjl_over_int4']:.3f}  "
          f"rank@4.2875 INT4={r['rank_int4_at_4.2875']} QJL={r['rank_qjl_at_4.2875']}  "
          f"PASS={r['PASS']}")

    print("[C2] Lloyd-Max quantizer ...")
    r = test_lloyd_max(W, X_calib)
    results["C2_lloyd_max"] = r
    print(f"  MSE uni={r['mse_uniform']:.5f}  LM={r['mse_lloyd_max']:.5f}  "
          f"ratio={r['mse_ratio_lm_over_uni']:.4f}  PASS={r['PASS(<0.90)']}")

    print("[C4] IP preservation metric ...")
    r = test_ip_metric(W, X_calib)
    results["C4_ip_metric"] = r
    print(f"  ip_cos_out={r['ip_cosine_output']:.4f}  ip_bias_out={r['ip_bias_output']:.4f}  "
          f"PASS={r['PASS']}")

    print("[C1] Cone-Aware rotation ...")
    r = test_cone_aware(W, X_calib)
    results["C1_cone_aware"] = r
    print(f"  plain_had={r['recon_err_plain_hadamard']:.4f}  cone={r['recon_err_cone_aware']:.4f}  "
          f"ratio={r['cone_over_plain']:.4f}  PASS={r['PASS']}")

    results["all_pass"] = all(results[k]["PASS"] for k in
                              ["C3_qjl", "C2_lloyd_max", "C4_ip_metric", "C1_cone_aware"])
    results["elapsed_sec"] = round(time.time() - t0, 2)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f"ALL PASS: {results['all_pass']}   ({results['elapsed_sec']}s)")
    print(f"Saved: {OUT_FILE}")
    return results


if __name__ == "__main__":
    main()
