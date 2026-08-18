#!/usr/bin/env python3
"""Test Stage 5: Shared basis for q/k/v projections."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import json
from collections import defaultdict
from model_loader import load_model, load_tokenizer
from chmc_v4 import (
    compute_perplexity, get_compressible_layers,
    _get_weight, _set_weight, honest_compression_bits,
)

print("Loading model...")
tok = load_tokenizer("HuggingFaceTB/SmolLM-135M")
mdl = load_model("HuggingFaceTB/SmolLM-135M")

baseline_ppl = compute_perplexity(mdl, tok, n_tokens=2000)
print(f"Baseline PPL: {baseline_ppl:.4f}")

# ── Find q/k/v groups ────────────────────────────────────────────────
layers = get_compressible_layers(mdl)

blocks_qkv = defaultdict(dict)
for name in layers:
    if "self_attn" in name and any(suf in name for suf in [".q_proj", ".k_proj", ".v_proj"]):
        parts = name.split(".")
        suffix = parts[-1]
        # Block prefix is everything before the projection name
        block_prefix = ".".join(parts[:-1])
        blocks_qkv[block_prefix][suffix] = name

print(f"\nFound {len(blocks_qkv)} q/k/v groups")

# ── Test shared basis at different ranks ─────────────────────────────
ranks_to_test = [8, 16, 32, 64]
all_results = []

for rank in ranks_to_test:
    print(f"\n{'='*60}")
    print(f"Shared basis rank={rank}")
    print(f"{'='*60}")
    
    block_results = []
    total_err = 0.0
    count = 0
    
    for i, (block_prefix, qkv_map) in enumerate(list(blocks_qkv.items())[:5]):  # First 5 blocks for speed
        if len(qkv_map) < 3:
            continue
        
        W_q = _get_weight(mdl, qkv_map["q_proj"]).detach().float()
        W_k = _get_weight(mdl, qkv_map["k_proj"]).detach().float()
        W_v = _get_weight(mdl, qkv_map["v_proj"]).detach().float()
        
        out_q, in_f_q = W_q.shape[0], W_q.shape[1]
        out_k, in_f_k = W_k.shape[0], W_k.shape[1]
        out_v, in_f_v = W_v.shape[0], W_v.shape[1]
        
        # Skip if input dims differ (shouldn't happen but safety check)
        if in_f_q != in_f_k or in_f_q != in_f_v:
            print(f"  Skipping {block_prefix}: mismatched input dims ({in_f_q}, {in_f_k}, {in_f_v})")
            continue
        
        in_f = in_f_q
        
        # Shared basis: SVD of concatenated [W_q; W_k; W_v]
        total_out = out_q + out_k + out_v
        W_concat = torch.cat([W_q, W_k, W_v], dim=0)  # [total_out, in_f]
        actual_rank = min(rank, total_out, in_f)
        
        U, S, V = torch.svd_lowrank(W_concat, q=actual_rank, niter=5)
        P_shared = V  # [in_f, rank] — shared right basis
        
        # Recover left factors for each projection (respecting actual output dims)
        A_q = (U[:, :actual_rank] * S[:actual_rank])[:out_q]  # [out_q, rank]
        A_k = (U[:, :actual_rank] * S[:actual_rank])[out_q:out_q+out_k]  # [out_k, rank]
        A_v = (U[:, :actual_rank] * S[:actual_rank])[out_q+out_k:]  # [out_v, rank]
        
        # Reconstruct and measure error
        W_q_hat = A_q @ P_shared.T
        W_k_hat = A_k @ P_shared.T
        W_v_hat = A_v @ P_shared.T
        
        err_q = (W_q - W_q_hat).pow(2).sum().sqrt() / W_q.pow(2).sum().sqrt()
        err_k = (W_k - W_k_hat).pow(2).sum().sqrt() / W_k.pow(2).sum().sqrt()
        err_v = (W_v - W_v_hat).pow(2).sum().sqrt() / W_v.pow(2).sum().sqrt()
        
        # Bit comparison
        bits_no_shared = 16.0 * actual_rank * (out_q + in_f) + 16.0 * actual_rank * (out_k + in_f) + 16.0 * actual_rank * (out_v + in_f)
        bits_shared = 16.0 * actual_rank * (out_q + out_k + out_v + in_f)
        
        # For equal-rank q/k/v: no_shared = 3*r*(o+i), shared = r*(3*o + i)
        # Savings = 3r(o+i) - r(3o+i) = 3ro + 3ri - 3ro - ri = 2ri bits saved
        saved_pct = (bits_no_shared - bits_shared) / bits_no_shared * 100 if bits_no_shared > 0 else 0
        
        avg_err = (err_q + err_k + err_v) / 3
        total_err += avg_err
        count += 1
        
        block_results.append({
            "block": block_prefix,
            "rank": actual_rank,
            "err_q": round(float(err_q), 6),
            "err_k": round(float(err_k), 6),
            "err_v": round(float(err_v), 6),
            "avg_err": round(float(avg_err), 6),
            "bits_no_shared": round(bits_no_shared, 0),
            "bits_shared": round(bits_shared, 0),
            "saved_pct": round(saved_pct, 1),
        })
        
        print(f"  [{i+1}] {block_prefix.split('.')[-1]}: avg_err={avg_err:.6f}, saved={saved_pct:.1f}%")
    
    if count > 0:
        avg_total = total_err / count
        print(f"\n  Avg error across {count} blocks: {avg_total:.6f}")
        
        # ── Full model PPL test at this rank ───────────────────────
        # Apply shared basis to all q/k/v groups and measure PPL
        orig_weights = {}
        applied_count = 0
        
        for block_prefix, qkv_map in blocks_qkv.items():
            if len(qkv_map) < 3:
                continue
            
            W_q = _get_weight(mdl, qkv_map["q_proj"]).detach().float()
            W_k = _get_weight(mdl, qkv_map["k_proj"]).detach().float()
            W_v = _get_weight(mdl, qkv_map["v_proj"]).detach().float()
            
            out_q, in_f_q = W_q.shape[0], W_q.shape[1]
            out_k, in_f_k = W_k.shape[0], W_k.shape[1]
            out_v, in_f_v = W_v.shape[0], W_v.shape[1]
            
            if in_f_q != in_f_k or in_f_q != in_f_v:
                continue
            
            in_f = in_f_q
            total_out = out_q + out_k + out_v
            
            orig_weights[qkv_map["q_proj"]] = W_q.clone()
            orig_weights[qkv_map["k_proj"]] = W_k.clone()
            orig_weights[qkv_map["v_proj"]] = W_v.clone()
            
            W_concat = torch.cat([W_q, W_k, W_v], dim=0)
            actual_rank = min(rank, total_out, in_f)
            
            U, S, V = torch.svd_lowrank(W_concat, q=actual_rank, niter=5)
            P_shared = V
            
            A_q = (U[:, :actual_rank] * S[:actual_rank])[:out_q]
            A_k = (U[:, :actual_rank] * S[:actual_rank])[out_q:out_q+out_k]
            A_v = (U[:, :actual_rank] * S[:actual_rank])[out_q+out_k:]
            
            _set_weight(mdl, qkv_map["q_proj"], A_q @ P_shared.T)
            _set_weight(mdl, qkv_map["k_proj"], A_k @ P_shared.T)
            _set_weight(mdl, qkv_map["v_proj"], A_v @ P_shared.T)
            
            applied_count += 1
        
        ppl_shared = compute_perplexity(mdl, tok, n_tokens=2000)
        
        # Restore originals
        for name, W in orig_weights.items():
            _set_weight(mdl, name, W)
        
        print(f"  Full model PPL (rank={rank}, {applied_count} groups): {ppl_shared:.4f} (ratio={ppl_shared/baseline_ppl:.3f})")
        
        all_results.append({
            "rank": rank,
            "avg_error": round(float(avg_total), 6),
            "blocks_tested": count,
            "full_model_ppl": round(ppl_shared, 4),
            "ppl_ratio": round(ppl_shared/baseline_ppl, 3),
            "groups_applied": applied_count,
            "block_details": block_results,
        })

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"STAGE 5 SUMMARY: Shared q/k/v basis")
print(f"{'='*60}")
for r in all_results:
    print(f"Rank={r['rank']}: avg_err={r['avg_error']:.6f}, PPL_ratio={r['ppl_ratio']:.3f} ({r['groups_applied']} groups)")

# Save results
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results_v4", "smollm-135m")
import os
os.makedirs(results_dir, exist_ok=True)

result = {
    "baseline_ppl": round(baseline_ppl, 4),
    "total_qkv_groups": len(blocks_qkv),
    "shared_basis_results": all_results,
}

with open(os.path.join(results_dir, "stage5_results.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved to {results_dir}/stage5_results.json")
print("\nDone.")
