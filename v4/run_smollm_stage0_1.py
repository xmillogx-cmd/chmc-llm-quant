#!/usr/bin/env python3
"""Run CHMC v4 pipeline (Stages 0-1) on SmolLM-135M — baseline, scalar q2-q6, CHMC rank 4/8."""
import sys, json, math
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cpu"
DTYPE = torch.float32

# ── Load model ────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = str(_ROOT / "models" / "smollm-135m")
RESULTS_DIR = _ROOT / "results_v4" / "smollm-135m"

print(f"Loading {MODEL_NAME}...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
)
mdl.eval()
n_params = sum(p.numel() for p in mdl.parameters())
print(f"[OK] Loaded {n_params:,} params -> {DEVICE}")

# ── Import shared utils ───────────────────────────────────────────────
from eval_utils import (
    compute_perplexity,
    collect_calibration_inputs,
    honest_compression_bits,
    weighted_svd_compress,
    get_compressible_layers,
    get_weight,
    set_weight,
)

from chmc_v4 import quantize_symmetric_per_channel


# ── Stage 0: Baseline PPL ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("STAGE 0: Baseline")
print(f"{'='*60}")

layers = get_compressible_layers(mdl)
print(f"Compressible layers: {len(layers)}")

shape_summary = {}
for name, mod in mdl.named_modules():
    if isinstance(mod, nn.Linear):
        shape_key = f"{mod.out_features}x{mod.in_features}"
        shape_summary[shape_key] = shape_summary.get(shape_key, 0) + 1

print("Layer shapes:")
for s, c in sorted(shape_summary.items()):
    print(f"  {s}: {c} layers")

baseline_ppl = compute_perplexity(mdl, tok)
print(f"\nBaseline PPL: {baseline_ppl:.4f}")

# ── Collect ALL calibration inputs in ONE pass ────────────────────────
print(f"\nCollecting calibration inputs (single forward pass)...")
calib_inputs = collect_calibration_inputs(mdl, layers, tok, 2048)

for name, X in calib_inputs.items():
    W = get_weight(mdl, name)
    if X.shape[1] != W.shape[1]:
        print(f"  [WARN] {name}: calib dim={X.shape[1]} vs weight in_f={W.shape[1]}, using random")
        calib_inputs[name] = torch.randn(64, W.shape[1], device=DEVICE)

print(f"[OK] Calibration collected for {len(calib_inputs)} layers (shape: [tokens, in_f])")

# ── Stage 1: Scalar baselines + CHMC comparison ───────────────────────
print(f"\n{'='*60}")
print("STAGE 1: Scalar Baselines + CHMC")
print(f"{'='*60}")

orig_weights = {name: get_weight(mdl, name).detach().clone() for name in layers}

# Scalar quantization baselines
scalar_results = {}
for bits in [2, 3, 4, 5, 6]:
    print(f"\n--- scalar_q{bits} ---")

    for name in layers:
        W = get_weight(mdl, name)
        W_q, scale = quantize_symmetric_per_channel(W, bits=bits, dim=0)
        set_weight(mdl, name, W_q)

    ppl = compute_perplexity(mdl, tok)

    total_comp = 0.0
    total_weights = 0
    for name in layers:
        W = orig_weights[name]
        out_f, in_f = W.shape
        bw = bits + 16.0 / in_f   # per-channel scale overhead
        total_comp += out_f * in_f * bw
        total_weights += out_f * in_f

    actual_bw = total_comp / total_weights if total_weights else 0

    scalar_results[f"scalar_q{bits}"] = {
        "ppl": round(ppl, 4),
        "ppl_ratio": round(ppl/baseline_ppl, 3),
        "bit_per_weight": round(actual_bw, 3),
    }

    print(f"  PPL: {ppl:.4f} (ratio={ppl/baseline_ppl:.3f}), bw={actual_bw:.3f}")

    for name in layers:
        set_weight(mdl, name, orig_weights[name])

# CHMC at different ranks
chmc_results = {}
for rank in [4, 8]:
    print(f"\n--- CHMC rank={rank} ---")

    for name in layers:
        set_weight(mdl, name, orig_weights[name])

    total_comp = 0.0
    total_weights = 0
    compressed_count = 0

    for name in tqdm(layers, desc=f"CHMC rank={rank}"):
        W_orig = orig_weights[name]
        X_calib = calib_inputs.get(name)

        if X_calib is None or X_calib.numel() == 0:
            in_f = W_orig.shape[1]
            X_calib = torch.randn(64, in_f, device=DEVICE)

        W_lr, err = weighted_svd_compress(W_orig, X_calib, rank)
        R_full = W_orig - W_lr
        R_q, _ = quantize_symmetric_per_channel(R_full, bits=4, dim=0)
        W_comp = W_lr + R_q
        set_weight(mdl, name, W_comp)

        out_f, in_f = W_orig.shape
        bi = honest_compression_bits(out_f, in_f, rank)
        total_comp += bi["compressed_bits"]
        total_weights += out_f * in_f
        compressed_count += 1

    bw = total_comp / total_weights if total_weights else 0
    ppl_chmc = compute_perplexity(mdl, tok)

    chmc_results[f"rank_{rank}"] = {
        "ppl": round(ppl_chmc, 4),
        "ratio": round(ppl_chmc/baseline_ppl, 3),
        "bw": round(bw, 3),
        "layers_compressed": compressed_count,
    }

    print(f"  PPL: {ppl_chmc:.4f} (ratio={ppl_chmc/baseline_ppl:.3f}), bw={bw:.3f}")

    for name in layers:
        set_weight(mdl, name, orig_weights[name])

# ── Save results ───────────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

result = {
    "model": "HuggingFaceTB/SmolLM-135M",
    "local_path": MODEL_NAME,
    "baseline_ppl": round(baseline_ppl, 4),
    "total_layers": len(layers),
    "scalar_baselines": scalar_results,
    "chmc_results": chmc_results,
}

with open(RESULTS_DIR / "stage1_results.json", "w") as f:
    json.dump(result, f, indent=2)

# ── Summary ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"STAGE 1 SUMMARY (SmolLM-135M)")
print(f"{'='*60}")
print(f"Baseline PPL: {baseline_ppl:.4f}")

for method, info in scalar_results.items():
    print(f"{method}: PPL={info['ppl']}, ratio={info['ppl_ratio']}x, bw={info['bit_per_weight']}")

for method, info in chmc_results.items():
    print(f"CHMC {method}: PPL={info['ppl']}, ratio={info['ratio']}x, bw={info['bw']}")

if "rank_4" in chmc_results and "scalar_q4" in scalar_results:
    chmc_bw = chmc_results["rank_4"]["bw"]
    sc_bw = scalar_results["scalar_q4"]["bit_per_weight"]
    print(f"\nCHMC rank=4 at bw={chmc_bw:.2f} -> ratio {chmc_results['rank_4']['ratio']}x")
    print(f"scalar_q4 at bw={sc_bw:.2f} -> ratio {scalar_results['scalar_q4']['ppl_ratio']}x")

    if chmc_results["rank_4"]["ratio"] < scalar_results["scalar_q4"]["ppl_ratio"]:
        improvement = scalar_results["scalar_q4"]["ppl_ratio"] / chmc_results["rank_4"]["ratio"]
        print(f"CHMC wins by {improvement:.1f}x in PPL ratio!")

print(f"\nResults saved to {RESULTS_DIR}/stage1_results.json")
print("\nDone.")
