#!/usr/bin/env python3
"""
run_chmc_v5_mixed.py — CHMC v5 with mixed precision for sensitive layers.

Strategy: identify top-K sensitive layers, give them higher rank (16 instead of 8)
and/or more residual bits (INT8 instead of INT4). Keep the rest at rank=8 + INT4.
"""

import sys
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).parent.resolve()
ROOT_DIR = BASE_DIR.parent
RESULTS  = ROOT_DIR / "results_v5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from eval_utils_v5 import (
    compute_perplexity, load_wikitext_eval, collect_calibration_inputs,
    get_compressible_layers, get_weight, set_weight,
)
from stabilizers import (
    quantize_with_compensation,
    compute_damped_covariance,
)


def compress_layer_mixed(
    W_orig: torch.Tensor,
    X_calib: torch.Tensor,
    rank: int = 8,
    residual_bits: int = 4,
    group_size: int = 128,
    dampening: float = 0.01,
) -> tuple:
    """Compress one layer with weighted SVD + H⁻¹ compensation."""
    W = W_orig.detach().float()
    out_f, in_f = W.shape
    rank = min(rank, out_f, in_f)

    # Weighted SVD with dampening
    diag_c = compute_damped_covariance(X_calib, dampening=dampening)
    W_weighted = W * torch.sqrt(diag_c).unsqueeze(0)
    U, S, V = torch.svd_lowrank(W_weighted, q=rank, niter=5)
    W_lr = ((U * S.unsqueeze(0)) @ V.T) / torch.sqrt(diag_c).unsqueeze(0)

    R_full = W - W_lr

    # H⁻¹ error compensation with act-order
    R_q = quantize_with_compensation(
        R_full, X_calib, bits=residual_bits, group_size=group_size, dampening=dampening
    )

    W_comp = W_lr + R_q
    total_norm = W.pow(2).sum().sqrt().item()
    recon_err = float((W - W_comp).pow(2).sum().sqrt() / max(total_norm, 1e-8))

    return W_comp, {
        "rank": rank,
        "residual_bits": residual_bits,
        "recon_err": round(recon_err, 6),
    }


def find_sensitive_layers(
    mdl, tok, layers, calib_inputs, encoded, baseline_ppl, device, top_k=10
):
    """Find the most sensitive layers by measuring reconstruction error.

    BUG-8 fix: use reconstruction error instead of PPL delta — saves ~2.5 hours
    because we avoid running a full forward pass per layer.
    """
    print(f"\n  Finding {top_k} most sensitive layers (reconstruction error)...")

    sensitivities = {}

    for name in layers:
        W_orig = get_weight(mdl, name).detach().float()
        X_calib = calib_inputs.get(name, torch.randn(64, W_orig.shape[1], device=device))

        # Compress this layer aggressively (rank=4, INT4) and measure reconstruction error
        W_comp, stats = compress_layer_mixed(W_orig, X_calib, rank=4, residual_bits=4)
        sensitivities[name] = stats["recon_err"]  # relative reconstruction error

    sorted_layers = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)

    print("  Top-10 sensitive layers (by recon_err):")
    for name, err in sorted_layers[:10]:
        print(f"    {name}: {err:.6f}")

    return sorted_layers


def run_chmc_v5_mixed(model_path: str):
    """Run CHMC v5 with mixed precision strategy."""
    tag = Path(model_path).name
    print(f"\n{'=' * 60}")
    print(f"CHMC v5 Mixed Precision: {tag}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # Load model
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
        # BUG-10: no device_map="auto" on CPU — adds accelerate overhead
    )
    mdl.eval()

    encoded = load_wikitext_eval(tok)
    baseline_ppl = compute_perplexity(mdl, tok, encoded)
    print(f"\n  Baseline PPL: {baseline_ppl:.4f}")

    layers = get_compressible_layers(mdl)
    calib_inputs = collect_calibration_inputs(mdl, layers, tok, n_tokens=2048)
    device = next(mdl.parameters()).device

    # Find sensitive layers
    sorted_sens = find_sensitive_layers(
        mdl, tok, layers, calib_inputs, encoded, baseline_ppl, device, top_k=15
    )

    # Assign mixed precision config
    sensitive_names = {name for name, _ in sorted_sens[:10]}  # top-10 get extra treatment
    print(f"\n  Sensitive layers (rank=16 + INT4): {len(sensitive_names)}")

    # Compress all layers with mixed precision
    t_compress = time.time()
    for idx, name in enumerate(layers):
        if (idx + 1) % 70 == 0:
            print(f"  [{idx+1}/{len(layers)}] ({time.time()-t_compress:.1f}s)")

        W_orig = get_weight(mdl, name)
        X_calib = calib_inputs.get(name, torch.randn(64, W_orig.shape[1], device=device))

        if name in sensitive_names:
            rank, bits = 16, 4  # Higher rank for sensitive layers
        else:
            rank, bits = 8, 4   # Standard config

        W_comp, stats = compress_layer_mixed(
            W_orig, X_calib, rank=rank, residual_bits=bits
        )
        set_weight(mdl, name, W_comp.to(W_orig.dtype))

    compress_time = time.time() - t_compress

    # Measure final PPL
    compressed_ppl = compute_perplexity(mdl, tok, encoded)
    ratio = compressed_ppl / baseline_ppl if baseline_ppl > 0 else float("inf")
    total_time = time.time() - t0

    print(f"\n  {'=' * 40}")
    print(f"  Result: PPL={compressed_ppl:.4f} (ratio={ratio:.3f}x)")
    print(f"  Time: {compress_time:.1f}s compress, {total_time:.1f}s total")
    print(f"  {'=' * 40}")

    result = {
        "model": tag,
        "baseline_ppl": round(baseline_ppl, 4),
        "compressed_ppl": round(compressed_ppl, 4),
        "ratio": round(ratio, 4),
        "sensitive_layers": list(sensitive_names),
        "timing": {
            "compress_sec": round(compress_time, 2),
            "total_sec": round(total_time, 2),
        },
    }

    out_file = RESULTS / f"chmc_v5_mixed_{tag}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_file}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT_DIR / "models/smollm-135m"))
    args = parser.parse_args()
    run_chmc_v5_mixed(args.model)
