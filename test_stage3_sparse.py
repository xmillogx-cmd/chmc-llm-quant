#!/usr/bin/env python3
"""Test Stage 3: Sparse residual with compensation vs without."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import json
import numpy as np
from model_loader import load_model, load_tokenizer
from chmc_v4 import (
    compute_perplexity, get_compressible_layers, allocate_ranks_uniform,
    _get_weight, _set_weight, collect_single_layer_input,
    honest_compression_bits, select_best_lowrank_method,
    quantize_sparse_hessian,
)

print("Loading model...")
tok = load_tokenizer("HuggingFaceTB/SmolLM-135M")
mdl = load_model("HuggingFaceTB/SmolLM-135M")

baseline_ppl = compute_perplexity(mdl, tok, n_tokens=2000)
print(f"Baseline PPL: {baseline_ppl:.4f}")

# ── Config ────────────────────────────────────────────────────────────
rank = 8
densities = [0.05, 0.10, 0.25]
layers = get_compressible_layers(mdl)[:5]  # First 5 layers for speed

print(f"\nTesting {len(layers)} layers at rank={rank}")

orig_weights = {}
for name in layers:
    orig_weights[name] = _get_weight(mdl, name).detach().clone()

# ── Per-layer comparison ─────────────────────────────────────────────
all_results = []

for density in densities:
    print(f"\n{'='*60}")
    print(f"Density={density:.2f} (keep top {int(density*100)}% by hessian-aware magnitude)")
    print(f"{'='*60}")
    
    comp_results = {"with_comp": [], "no_comp": []}
    
    for i, name in enumerate(layers):
        W_orig = orig_weights[name]
        X_calib = collect_single_layer_input(mdl, name, tok, 1024)
        if X_calib.numel() == 0:
            continue
        
        out_f, in_f = W_orig.shape
        _, W_lr, lr_err = select_best_lowrank_method(W_orig, X_calib, rank)
        R_full = W_orig - W_lr
        
        # ── Without compensation (baseline sparse) ────────────────
        R_sparse_nc, mask_nc = quantize_sparse_hessian(R_full, X_calib, bits=4, density=density)
        W_no_comp = W_lr + R_sparse_nc
        err_nc = (W_orig - W_no_comp).pow(2).sum().sqrt() / W_orig.pow(2).sum().sqrt()
        
        Y_orig = X_calib @ W_orig.T
        Y_nc = X_calib @ W_no_comp.T
        cos_nc = torch.nn.functional.cosine_similarity(Y_orig.reshape(1, -1), Y_nc.reshape(1, -1), dim=1).item()
        
        nnz_nc = mask_nc.sum().item()
        comp_results["no_comp"].append({
            "layer": name, "cos_sim": round(float(cos_nc), 6),
            "recon_err": round(float(err_nc), 6), "nnz": int(nnz_nc),
        })
        
        # ── With compensation (optimize A,B for W - R_sparse) ────
        from chmc_v4 import sparse_with_compensation
        W_comp, W_lr_final, R_sparse_c = sparse_with_compensation(
            W_orig, X_calib, rank, density=density, steps=100, lr=1e-3
        )
        err_c = (W_orig - W_comp).pow(2).sum().sqrt() / W_orig.pow(2).sum().sqrt()
        
        Y_c = X_calib @ W_comp.T
        cos_c = torch.nn.functional.cosine_similarity(Y_orig.reshape(1, -1), Y_c.reshape(1, -1), dim=1).item()
        
        nnz_c = (R_sparse_c != 0).sum().item()
        comp_results["with_comp"].append({
            "layer": name, "cos_sim": round(float(cos_c), 6),
            "recon_err": round(float(err_c), 6), "nnz": int(nnz_c),
        })
        
        print(f"  [{i+1}/{len(layers)}] {name.split('.')[-1]}:")
        print(f"    no_comp:   err={err_nc:.6f}, cos={cos_nc:.6f}, nnz={nnz_nc}")
        print(f"    with_comp: err={err_c:.6f}, cos={cos_c:.6f}, nnz={nnz_c}")
        
        improvement = (err_nc - err_c) / err_nc * 100 if err_nc > 0 else 0
        print(f"    improvement: {improvement:+.1f}%")
    
    # Summary for this density
    avg_err_nc = np.mean([r["recon_err"] for r in comp_results["no_comp"]])
    avg_cos_nc = np.mean([r["cos_sim"] for r in comp_results["no_comp"]])
    avg_err_c = np.mean([r["recon_err"] for r in comp_results["with_comp"]])
    avg_cos_c = np.mean([r["cos_sim"] for r in comp_results["with_comp"]])
    
    print(f"\n  Summary density={density:.2f}:")
    print(f"    no_comp:   avg_err={avg_err_nc:.6f}, avg_cos={avg_cos_nc:.6f}")
    print(f"    with_comp: avg_err={avg_err_c:.6f}, avg_cos={avg_cos_c:.6f}")
    
    all_results.append({
        "density": density,
        "no_comp": {"avg_recon_err": round(avg_err_nc, 6), "avg_cos_sim": round(avg_cos_nc, 6), "layers": comp_results["no_comp"]},
        "with_comp": {"avg_recon_err": round(avg_err_c, 6), "avg_cos_sim": round(avg_cos_c, 6), "layers": comp_results["with_comp"]},
    })

# ── Full model PPL test at best density ───────────────────────────────
best_density = 0.10  # Middle ground — good compression ratio
print(f"\n{'='*60}")
print(f"Full model test: all layers with sparse compensation d={best_density}")
print(f"{'='*60}")

# Compress ALL compressible layers with sparse + compensation
all_layers = get_compressible_layers(mdl)
for name in all_layers:
    orig_weights[name] = _get_weight(mdl, name).detach().clone()

compressed_count = 0
for name in all_layers:
    W_orig = orig_weights[name]
    X_calib = collect_single_layer_input(mdl, name, tok, 512)
    if X_calib.numel() == 0:
        continue
    
    _, W_lr, _ = select_best_lowrank_method(W_orig, X_calib, rank)
    R_full = W_orig - W_lr
    R_sparse_c, _ = quantize_sparse_hessian(R_full, X_calib, bits=4, density=best_density)
    
    # Simple compensation: optimize A,B for 20 steps (speed vs quality tradeoff)
    from chmc_v4 import sparse_with_compensation
    W_comp, _, _ = sparse_with_compensation(W_orig, X_calib, rank, density=best_density, steps=20, lr=1e-3)
    
    _set_weight(mdl, name, W_comp)
    compressed_count += 1

ppl_sparse_comp = compute_perplexity(mdl, tok, n_tokens=2000)
print(f"Sparse+compensation PPL: {ppl_sparse_comp:.4f} (ratio={ppl_sparse_comp/baseline_ppl:.3f})")

# Restore originals
for name in all_layers:
    if name in orig_weights:
        _set_weight(mdl, name, orig_weights[name])

# Compute bw for sparse
total_comp = 0.0
total_weights = 0
for name in all_layers:
    W = orig_weights[name]
    out_f, in_f = W.shape
    bi = honest_compression_bits(out_f, in_f, rank, residual_density=best_density)
    total_comp += bi["compressed_bits"]
    total_weights += out_f * in_f
bw_sparse = total_comp / total_weights if total_weights else 0

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"STAGE 3 SUMMARY (rank={rank})")
print(f"{'='*60}")
for r in all_results:
    d = r["density"]
    print(f"Density={d:.2f}:")
    print(f"  no_comp:   err={r['no_comp']['avg_recon_err']:.6f}, cos={r['no_comp']['avg_cos_sim']:.6f}")
    print(f"  with_comp: err={r['with_comp']['avg_recon_err']:.6f}, cos={r['with_comp']['avg_cos_sim']:.6f}")

print(f"\nFull model PPL (d={best_density}): {ppl_sparse_comp:.4f} (ratio={ppl_sparse_comp/baseline_ppl:.3f})")
print(f"BW sparse d={best_density}: {bw_sparse:.3f}")

# Save results
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results_v4", "smollm-135m")
import os
os.makedirs(results_dir, exist_ok=True)

result = {
    "baseline_ppl": round(baseline_ppl, 4),
    "rank": rank,
    "densities": all_results,
    "full_model_sparse_comp": {
        "density": best_density,
        "ppl": round(ppl_sparse_comp, 4),
        "ratio": round(ppl_sparse_comp/baseline_ppl, 3),
        "bw": round(bw_sparse, 3),
        "layers_compressed": compressed_count,
    },
}

with open(os.path.join(results_dir, "stage3_results.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved to {results_dir}/stage3_results.json")
print("\nDone.")
