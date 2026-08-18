#!/usr/bin/env python3
"""Test CHMC compression vs scalar baselines on SmolLM-135M."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_loader import load_model, load_tokenizer
from chmc_v4 import (
    compress_model_sequential, compute_perplexity,
    get_compressible_layers, allocate_ranks_uniform, honest_compression_bits
)
import numpy as np

print("Loading model...")
tok = load_tokenizer("HuggingFaceTB/SmolLM-135M")
mdl = load_model("HuggingFaceTB/SmolLM-135M")

# Baseline PPL
baseline_ppl = compute_perplexity(mdl, tok, n_tokens=2000)
print(f"Baseline PPL: {baseline_ppl:.4f}")

# Test CHMC at different ranks — full model (all 210 layers)
for rank in [4, 8]:
    print(f"\n--- CHMC rank={rank} (full model) ---")
    
    # Compute expected bw
    layers = get_compressible_layers(mdl)
    total_comp = 0.0
    total_weights = 0
    for name in layers:
        from chmc_v4 import _get_weight
        W = _get_weight(mdl, name)
        out_f, in_f = W.shape
        bi = honest_compression_bits(out_f, in_f, rank)
        total_comp += bi["compressed_bits"]
        total_weights += out_f * in_f
    
    bw = total_comp / total_weights if total_weights else 0
    print(f"Expected bw: {bw:.3f}, layers={len(layers)}")
    
    # Compress ALL layers
    ranks = allocate_ranks_uniform(mdl, rank=rank)
    
    results = compress_model_sequential(mdl, tok, ranks, method='auto', calib_tokens=512)
    print(f"Compressed {len(results)} layers")
    
    # Measure PPL
    ppl_chmc = compute_perplexity(mdl, tok, n_tokens=2000)
    ratio = ppl_chmc / baseline_ppl if baseline_ppl > 0 else float('inf')
    print(f"CHMC rank={rank}: PPL={ppl_chmc:.4f}, ratio={ratio:.3f}")

print("\nDone.")
