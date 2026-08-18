#!/usr/bin/env python3
"""Quick Stage 0 test — just accounting + single layer, no PPL runs."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("[1/4] Accounting sanity...")
from chmc_v4 import stage0_accounting_sanity
acct = stage0_accounting_sanity()
for tc in acct["test_cases"]:
    print(f"  {tc['shape']} r={tc['rank']}: CR={tc['compression_ratio']}x {'OK' if tc['ok'] else 'BAD'}")
print(f"  all_ok: {acct['all_ok']}")

print()
print("[2/4] Loading model...")
from model_loader import load_model, load_tokenizer
tok = load_tokenizer("HuggingFaceTB/SmolLM-135M")
mdl = load_model("HuggingFaceTB/SmolLM-135M")
print(f"  Loaded {type(mdl).__name__}")

print()
print("[3/4] Single layer debug...")
from chmc_v4 import stage0_single_layer_debug
single = stage0_single_layer_debug(mdl, tok, rank=32)
for k, v in single.items():
    if not isinstance(v, (list, dict)):
        print(f"  {k} = {v}")

print()
print("[4/4] Sequential debug...")
from chmc_v4 import stage0_sequential_debug
seq = stage0_sequential_debug(mdl, tok, n_layers=2)
for pair in seq.get("pairs", []):
    print(f"  {pair['layer_compressed']} -> {pair['layer_observed']}: cos={pair['input_cos_before_after']:.6f} changed={pair['inputs_changed']}")
print(f"  sequential_works: {seq.get('sequential_works')}")

# Quick single PPL run (just to check it works)
print()
print("[bonus] Single baseline PPL run...")
from chmc_v4 import compute_perplexity
ppl = compute_perplexity(mdl, tok, n_tokens=2000)
print(f"  PPL = {ppl:.4f}")

print("\nDone.")