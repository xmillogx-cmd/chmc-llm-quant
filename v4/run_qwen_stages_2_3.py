#!/usr/bin/env python3
"""CHMC v4 Stages 2-3 for Qwen — calibration + sparse compensation."""
import sys, os, json, math
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cpu"
DTYPE = torch.float32
_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = str(_ROOT / "models" / "qwen2.5-0.5b")
RESULTS_DIR = _ROOT / "results_v4" / "qwen2.5-0.5b"

# ── Import shared utils ───────────────────────────────────────────────
from eval_utils import (
    compute_perplexity,
    collect_calibration_inputs,
    weighted_svd_compress,
    get_compressible_layers,
    get_weight,
    set_weight,
)

from chmc_v4 import quantize_symmetric_per_channel


# ── Load model ────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
)
mdl.eval()
print(f"[OK] Loaded {sum(p.numel() for p in mdl.parameters()):,} params")


# ── Load Stage 1 results ──────────────────────────────────────────────
with open(RESULTS_DIR / "stage1_results.json") as f:
    stage1 = json.load(f)

baseline_ppl = stage1["baseline_ppl"]
layers = get_compressible_layers(mdl)
orig_weights = {name: get_weight(mdl, name).detach().clone() for name in layers}
calib_inputs = collect_calibration_inputs(mdl, layers, tok, 2048)

# ── Stage 2: Layerwise Calibration ───────────────────────────────────
print(f"\n{'='*60}")
print("STAGE 2: Layerwise Calibration")
print(f"{'='*60}")

rank = 8
compress_results = {}

for name in tqdm(layers, desc="Compress rank=8"):
    W_orig = orig_weights[name]
    X_calib = calib_inputs.get(name)
    if X_calib is None or X_calib.numel() == 0:
        X_calib = torch.randn(64, W_orig.shape[1], device=DEVICE)

    W_lr, _ = weighted_svd_compress(W_orig, X_calib, rank)
    R_full = W_orig - W_lr
    R_q_int4, _ = quantize_symmetric_per_channel(R_full, bits=4, dim=0)
    R_q_int8, _ = quantize_symmetric_per_channel(R_full, bits=8, dim=0)

    compress_results[name] = {
        "lr": W_lr.detach(),
        "int4": R_q_int4.detach(),
        "int8": R_q_int8.detach(),
    }

# Test variants on subset of layers (first 10 for speed)
test_layers = layers[:10]

variants = {}

# Variant 1: no calibration INT4 (just low-rank + int4 residual)
for name in test_layers:
    set_weight(mdl, name, compress_results[name]["lr"] + compress_results[name]["int4"])

ppl_no_calib_int4 = compute_perplexity(mdl, tok)
variants["no_calib_int4"] = {"ppl": round(ppl_no_calib_int4, 2), "ratio": round(ppl_no_calib_int4/baseline_ppl, 3)}
print(f"  no_calib_int4: PPL={ppl_no_calib_int4:.2f} (ratio={ppl_no_calib_int4/baseline_ppl:.3f})")

# Restore originals
for name in test_layers:
    set_weight(mdl, name, orig_weights[name])

# Variant 2: INT8 residual
for name in test_layers:
    set_weight(mdl, name, compress_results[name]["lr"] + compress_results[name]["int8"])

ppl_int8 = compute_perplexity(mdl, tok)
variants["calib_int8"] = {"ppl": round(ppl_int8, 2), "ratio": round(ppl_int8/baseline_ppl, 3)}
print(f"  calib_int8: PPL={ppl_int8:.2f} (ratio={ppl_int8/baseline_ppl:.3f})")

# Restore originals
for name in test_layers:
    set_weight(mdl, name, orig_weights[name])

stage2_results = {
    "rank": rank,
    "test_layers": len(test_layers),
    "baseline_ppl": baseline_ppl,
    "results": variants,
}

# ── Stage 3: Sparse Compensation ──────────────────────────────────────
print(f"\n{'='*60}")
print("STAGE 3: Sparse Compensation")
print(f"{'='*60}")

def apply_sparse_residual(W_orig, X_calib, rank, density):
    W_lr, _ = weighted_svd_compress(W_orig, X_calib, rank)
    R_full = W_orig - W_lr

    flat = R_full.abs().flatten()
    k = int(density * len(flat))
    threshold, _ = flat.topk(k, largest=True)
    min_val = threshold.min() if len(threshold) else 0

    mask = R_full.abs() >= min_val
    R_sparse = R_full * mask
    R_q, _ = quantize_symmetric_per_channel(R_sparse, bits=4, dim=0)
    return W_lr + R_q

# Per-layer analysis on sample layers
sample_layers = layers[:5]
densities = [0.05, 0.10, 0.25]

sparse_per_layer = {}
for density in densities:
    no_comp_errs, with_comp_errs = [], []
    no_comp_cosims, with_comp_cosims = [], []

    for name in sample_layers:
        W_orig = orig_weights[name].float()
        X_calib = calib_inputs.get(name)
        if X_calib is None:
            continue

        W_lr, _ = weighted_svd_compress(W_orig, X_calib, rank)

        err_no = (W_orig - W_lr).pow(2).mean().sqrt().item()
        cos_no = F.cosine_similarity(W_orig.flatten(), W_lr.flatten(), dim=0).item()
        no_comp_errs.append(err_no)
        no_comp_cosims.append(cos_no)

        W_sparse = apply_sparse_residual(W_orig, X_calib, rank, density)
        err_with = (W_orig - W_sparse).pow(2).mean().sqrt().item()
        cos_with = F.cosine_similarity(W_orig.flatten(), W_sparse.flatten(), dim=0).item()
        with_comp_errs.append(err_with)
        with_comp_cosims.append(cos_with)

    sparse_per_layer[f"density_{density}"] = {
        "no_comp": {"avg_recon_err": round(sum(no_comp_errs)/len(no_comp_errs), 3),
                    "avg_cos_sim": round(sum(no_comp_cosims)/len(no_comp_cosims), 3)},
        "with_comp": {"avg_recon_err": round(sum(with_comp_errs)/len(with_comp_errs), 3),
                      "avg_cos_sim": round(sum(with_comp_cosims)/len(with_comp_cosims), 3)}
    }

print(f"  Per-layer analysis (sample {len(sample_layers)} layers):")
for d, info in sparse_per_layer.items():
    print(f"    {d}: no_comp err={info['no_comp']['avg_recon_err']}, cos_sim={info['no_comp']['avg_cos_sim']}")
    print(f"           with_comp err={info['with_comp']['avg_recon_err']}, cos_sim={info['with_comp']['avg_cos_sim']}")

# Full model test at density=0.1
test_density = 0.1
for name in tqdm(layers, desc=f"Sparse comp density={test_density}"):
    W_orig = orig_weights[name]
    X_calib = calib_inputs.get(name)
    if X_calib is None:
        X_calib = torch.randn(64, W_orig.shape[1], device=DEVICE)

    W_comp = apply_sparse_residual(W_orig, X_calib, rank, test_density)
    set_weight(mdl, name, W_comp)

ppl_sparse_full = compute_perplexity(mdl, tok)
print(f"  Full model density={test_density}: PPL={ppl_sparse_full:.2f} (ratio={ppl_sparse_full/baseline_ppl:.1f})")

# Restore originals
for name in layers:
    set_weight(mdl, name, orig_weights[name])

stage3_results = {
    "baseline_ppl": baseline_ppl,
    "rank": rank,
    "per_layer_analysis": sparse_per_layer,
    "full_model_sparse_comp": {
        "density": test_density,
        "ppl": round(ppl_sparse_full, 2),
        "ratio": round(ppl_sparse_full/baseline_ppl, 1),
    }
}

# ── Save results ──────────────────────────────────────────────────────
all_results = {
    "model": "Qwen/Qwen2.5-0.5B",
    "stage2_calibration": stage2_results,
    "stage3_sparse_compensation": stage3_results,
}

with open(RESULTS_DIR / "stages_2_3_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print("Stages 2-3 complete!")
print(f"{'='*60}")
print(f"Results saved to {RESULTS_DIR}/stages_2_3_results.json")
