#!/usr/bin/env python3
"""Test Stage 2: Layerwise calibration — compare calibrated vs non-calibrated compression."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import json
from model_loader import load_model, load_tokenizer
from chmc_v4 import (
    compute_perplexity, get_compressible_layers, allocate_ranks_uniform,
    _get_weight, _set_weight, collect_single_layer_input,
    calibrate_layer, honest_compression_bits, select_best_lowrank_method,
    quantize_symmetric_per_channel,
)

print("Loading model...")
tok = load_tokenizer("HuggingFaceTB/SmolLM-135M")
mdl = load_model("HuggingFaceTB/SmolLM-135M")

baseline_ppl = compute_perplexity(mdl, tok, n_tokens=2000)
print(f"Baseline PPL: {baseline_ppl:.4f}")

# ── Config ────────────────────────────────────────────────────────────
rank = 8
ranks = allocate_ranks_uniform(mdl, rank=rank)
layers = list(ranks.keys())[:5]  # First 5 layers for speed

print(f"\nTesting {len(layers)} layers at rank={rank}")

orig_weights = {}
for name in layers:
    orig_weights[name] = _get_weight(mdl, name).detach().clone()

# ── Step 1: Low-rank + quantized residual WITHOUT calibration ────────
print(f"\n--- Step 1: Non-calibrated (low-rank + INT4 residual) ---")
for i, name in enumerate(layers):
    W_orig = orig_weights[name]
    X_calib = collect_single_layer_input(mdl, name, tok, 1024)
    if X_calib.numel() == 0:
        continue
    
    _, W_lr, err = select_best_lowrank_method(W_orig, X_calib, rank)
    R_full = W_orig - W_lr
    # Quantize residual to INT4 (no optimization)
    R_q, _ = quantize_symmetric_per_channel(R_full, bits=4, dim=0)
    W_comp = W_lr + R_q
    
    recon_err = (W_orig - W_comp).pow(2).sum().sqrt() / W_orig.pow(2).sum().sqrt()
    _set_weight(mdl, name, W_comp)
    print(f"  [{i+1}/{len(layers)}] {name}: recon_err={recon_err:.6f}")

ppl_no_calib = compute_perplexity(mdl, tok, n_tokens=2000)
print(f"No-calibration PPL: {ppl_no_calib:.4f} (ratio={ppl_no_calib/baseline_ppl:.3f})")

# Restore originals for calibration test
for name in layers:
    _set_weight(mdl, name, orig_weights[name])

# ── Step 2: Calibrated compression with INT8 residual ────────────────
print(f"\n--- Step 2: Calibrated (rank={rank}, steps=100, INT8 residual) ---")
for i, name in enumerate(layers):
    W_orig = orig_weights[name]
    X_calib = collect_single_layer_input(mdl, name, tok, 1024)
    if X_calib.numel() == 0:
        continue
    
    print(f"  [{i+1}/{len(layers)}] Calibrating {name}...")
    A, B, R = calibrate_layer(W_orig, X_calib, rank, residual_bits=8, steps=100, lr=1e-3)
    W_calibrated = A @ B.T + R
    
    recon_err = (W_orig - W_calibrated).pow(2).sum().sqrt() / W_orig.pow(2).sum().sqrt()
    Y_orig = X_calib @ W_orig.T
    Y_cal = X_calib @ W_calibrated.T
    cos_sim = torch.nn.functional.cosine_similarity(Y_orig.reshape(1, -1), Y_cal.reshape(1, -1), dim=1).item()
    
    _set_weight(mdl, name, W_calibrated)
    print(f"    recon_err={recon_err:.6f}, cos_sim={cos_sim:.6f}")

ppl_calib_int8 = compute_perplexity(mdl, tok, n_tokens=2000)
print(f"\nCalibration INT8 PPL: {ppl_calib_int8:.4f} (ratio={ppl_calib_int8/baseline_ppl:.3f})")

# Restore originals for INT4 calibration test
for name in layers:
    _set_weight(mdl, name, orig_weights[name])

# ── Step 3: Calibrated compression with INT4 residual ────────────────
print(f"\n--- Step 3: Calibrated (rank={rank}, steps=100, INT4 residual) ---")
for i, name in enumerate(layers):
    W_orig = orig_weights[name]
    X_calib = collect_single_layer_input(mdl, name, tok, 1024)
    if X_calib.numel() == 0:
        continue
    
    print(f"  [{i+1}/{len(layers)}] Calibrating {name} (INT4)...")
    A, B, R = calibrate_layer(W_orig, X_calib, rank, residual_bits=4, steps=100, lr=1e-3)
    
    # The STE quantization is applied during training; for final weight we need to do forward pass
    R_q, _ = quantize_symmetric_per_channel(R, bits=4, dim=0)
    W_calibrated_q4 = A @ B.T + R_q
    
    recon_err = (W_orig - W_calibrated_q4).pow(2).sum().sqrt() / W_orig.pow(2).sum().sqrt()
    
    _set_weight(mdl, name, W_calibrated_q4)
    print(f"    recon_err={recon_err:.6f}")

ppl_calib_int4 = compute_perplexity(mdl, tok, n_tokens=2000)
print(f"\nCalibration INT4 PPL: {ppl_calib_int4:.4f} (ratio={ppl_calib_int4/baseline_ppl:.3f})")

# ── Summary ───────────────────────────────────────────────────────────
# Compute bw for rank=8 with dense INT4 residual
total_comp = 0.0
total_weights = 0
for name in layers:
    W = orig_weights[name]
    out_f, in_f = W.shape
    bi = honest_compression_bits(out_f, in_f, rank)
    total_comp += bi["compressed_bits"]
    total_weights += out_f * in_f
bw = total_comp / total_weights if total_weights else 0

print(f"\n{'='*60}")
print(f"STAGE 2 SUMMARY ({len(layers)} layers, rank={rank})")
print(f"{'='*60}")
print(f"Baseline PPL:         {baseline_ppl:.4f} (ratio=1.000)")
print(f"No-calibration INT4:  {ppl_no_calib:.4f} (ratio={ppl_no_calib/baseline_ppl:.3f})")
print(f"Calibrated INT8:      {ppl_calib_int8:.4f} (ratio={ppl_calib_int8/baseline_ppl:.3f})")
print(f"Calibrated INT4:      {ppl_calib_int4:.4f} (ratio={ppl_calib_int4/baseline_ppl:.3f})")
print(f"BW (rank={rank}):     {bw:.3f}")

improvement_over_no_calib = (ppl_no_calib - ppl_calib_int4) / baseline_ppl
if improvement_over_no_calib > 0:
    print(f"\nCalibration improves PPL by {improvement_over_no_calib:+.2f} over non-calibrated INT4")
else:
    print(f"\nCalibration did NOT improve over non-calibrated (delta={improvement_over_no_calib:+.2f})")

# Save results
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results_v4", "smollm-135m")
import os
os.makedirs(results_dir, exist_ok=True)

result = {
    "baseline_ppl": round(baseline_ppl, 4),
    "no_calib_int4": {"ppl": round(ppl_no_calib, 4), "ratio": round(ppl_no_calib/baseline_ppl, 3)},
    "calib_int8": {"ppl": round(ppl_calib_int8, 4), "ratio": round(ppl_calib_int8/baseline_ppl, 3)},
    "calib_int4": {"ppl": round(ppl_calib_int4, 4), "ratio": round(ppl_calib_int4/baseline_ppl, 3)},
    "rank": rank,
    "residual_bits_no_calib": 4,
    "layers_tested": len(layers),
    "bw": round(bw, 3),
}

with open(os.path.join(results_dir, "stage2_results.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved to {results_dir}/stage2_results.json")

# Restore all originals
for name in layers:
    _set_weight(mdl, name, orig_weights[name])

print("\nDone.")
