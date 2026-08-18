#!/usr/bin/env python3
"""Test Stage 4: Rank-1 validation on SmolLM-135M."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import math
from model_loader import load_model, load_tokenizer
from chmc_v4 import (
    compute_perplexity, get_compressible_layers, allocate_ranks_uniform,
    _get_weight, _set_weight, collect_single_layer_input,
    honest_compression_bits, select_best_lowrank_method,
)

print("Loading model...")
tok = load_tokenizer("HuggingFaceTB/SmolLM-135M")
mdl = load_model("HuggingFaceTB/SmolLM-135M")

baseline_ppl = compute_perplexity(mdl, tok, n_tokens=2000)
print(f"Baseline PPL: {baseline_ppl:.4f}")

# ── Find rank-1 candidates via effective_rank ────────────────────────
import torch
layers = get_compressible_layers(mdl)

print(f"\nScanning {len(layers)} layers for rank-1 candidates...")

rank1_candidates = []
for name in layers:
    W = _get_weight(mdl, name).float()
    out_f, in_f = W.shape
    
    # Quick SVD to estimate effective rank
    q = min(64, out_f, in_f)
    U, S, V = torch.svd_lowrank(W, q=q, niter=3)
    
    total_energy = (S ** 2).sum().item()
    if total_energy < 1e-12:
        continue
    
    # Effective rank = exp(entropy of singular value spectrum)
    probs = (S ** 2) / total_energy
    entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
    eff_rank = math.exp(entropy)
    
    # Also check: how much energy is in top-1 singular value?
    top1_energy = (S[0] ** 2).item() / total_energy
    
    if eff_rank < 5.0 or top1_energy > 0.9:
        rank1_candidates.append({
            "layer": name,
            "shape": [out_f, in_f],
            "effective_rank": round(eff_rank, 2),
            "top1_energy": round(top1_energy, 4),
            "d90_approx": int(S[0].item() / total_energy * q) if total_energy > 0 else 1,
        })

print(f"\nFound {len(rank1_candidates)} rank-1 candidates (eff_rank < 5 or top1_energy > 0.9)")

# Sort by effective rank (lowest first)
rank1_candidates.sort(key=lambda x: x["effective_rank"])

for c in rank1_candidates[:20]:  # Show top 20
    print(f"  {c['layer']}: eff_rank={c['effective_rank']}, top1_energy={c['top1_energy']}")

# ── Test rank-1 replacement on candidates ────────────────────────────
import math
orig_weights = {}
results = []

test_count = min(5, len(rank1_candidates))
print(f"\nTesting rank-1 replacement on {test_count} best candidates...")

for i, cand in enumerate(rank1_candidates[:test_count]):
    name = cand["layer"]
    W_orig = _get_weight(mdl, name).detach().float()
    orig_weights[name] = W_orig.clone()
    
    out_f, in_f = W_orig.shape
    
    # Rank-1 approximation via SVD
    u, s, v = torch.svd_lowrank(W_orig, q=1, niter=5)
    W_r1 = (u[:, 0:1] * s[0]) @ v[0:1, :]
    
    recon_err = (W_orig - W_r1).pow(2).sum().sqrt() / W_orig.pow(2).sum().sqrt()
    
    # Bit savings
    orig_bits = 32.0 * out_f * in_f
    r1_bits = 16.0 * (out_f + in_f)  # two vectors at FP16
    cr = orig_bits / r1_bits
    
    # Apply and measure PPL impact
    _set_weight(mdl, name, W_r1)
    ppl_r1 = compute_perplexity(mdl, tok, n_tokens=2000)
    
    results.append({
        "layer": name,
        "shape": [out_f, in_f],
        "effective_rank": cand["effective_rank"],
        "top1_energy": cand["top1_energy"],
        "recon_err": round(float(recon_err), 6),
        "compression_ratio_r1": round(cr, 1),
        "ppl_after_r1": round(ppl_r1, 4),
        "ppl_ratio": round(ppl_r1 / baseline_ppl, 3),
    })
    
    print(f"  [{i+1}/{test_count}] {name.split('.')[-1]}:")
    print(f"    eff_rank={cand['effective_rank']}, top1_energy={cand['top1_energy']}")
    print(f"    recon_err={recon_err:.6f}, CR={cr:.1f}x, PPL_ratio={ppl_r1/baseline_ppl:.3f}")

# Restore originals
for name in orig_weights:
    _set_weight(mdl, name, orig_weights[name])

# ── Test rank-2 and rank-4 for comparison ────────────────────────────
print(f"\n{'='*60}")
print("Comparing rank-1 vs rank-2 vs rank-4 on same layers")
print(f"{'='*60}")

for i, cand in enumerate(rank1_candidates[:3]):
    name = cand["layer"]
    W_orig = _get_weight(mdl, name).detach().float()
    orig_weights[name] = W_orig.clone()
    
    out_f, in_f = W_orig.shape
    
    for rank in [1, 2, 4]:
        U, S, V = torch.svd_lowrank(W_orig, q=rank, niter=5)
        W_lr = (U[:, :rank] * S[:rank]) @ V[:rank, :]
        
        recon_err = (W_orig - W_lr).pow(2).sum().sqrt() / W_orig.pow(2).sum().sqrt()
        
        orig_bits = 32.0 * out_f * in_f
        lr_bits = 16.0 * rank * (out_f + in_f)
        cr = orig_bits / lr_bits
        
        _set_weight(mdl, name, W_lr)
        ppl_lr = compute_perplexity(mdl, tok, n_tokens=2000)
        
        print(f"  {name.split('.')[-1]} rank={rank}: err={recon_err:.6f}, CR={cr:.1f}x, PPL_ratio={ppl_lr/baseline_ppl:.3f}")
    
    # Restore
    _set_weight(mdl, name, orig_weights[name])

# Save results
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results_v4", "smollm-135m")
import os
os.makedirs(results_dir, exist_ok=True)

result = {
    "baseline_ppl": round(baseline_ppl, 4),
    "rank1_candidates_count": len(rank1_candidates),
    "top_candidates": rank1_candidates[:20],
    "replacement_tests": results,
}

with open(os.path.join(results_dir, "stage4_results.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved to {results_dir}/stage4_results.json")
print("\nDone.")
