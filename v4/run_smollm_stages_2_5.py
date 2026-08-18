#!/usr/bin/env python3
"""CHMC v4 Stages 2-5 for SmolLM-135M — calibration, sparse comp, eff rank, shared basis."""
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
MODEL_NAME = str(_ROOT / "models" / "smollm-135m")
RESULTS_DIR = _ROOT / "results_v4" / "smollm-135m"

# ── Import shared utils ───────────────────────────────────────────────
from eval_utils import (
    compute_perplexity,
    collect_calibration_inputs,
    load_calib_text,
    weighted_svd_compress,
    get_compressible_layers,
    get_module_by_name,
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

for name in test_layers:
    set_weight(mdl, name, compress_results[name]["lr"] + compress_results[name]["int4"])

ppl_no_calib_int4 = compute_perplexity(mdl, tok)
variants["no_calib_int4"] = {"ppl": round(ppl_no_calib_int4, 2), "ratio": round(ppl_no_calib_int4/baseline_ppl, 3)}
print(f"  no_calib_int4: PPL={ppl_no_calib_int4:.2f} (ratio={ppl_no_calib_int4/baseline_ppl:.3f})")

for name in test_layers:
    set_weight(mdl, name, orig_weights[name])

for name in test_layers:
    set_weight(mdl, name, compress_results[name]["lr"] + compress_results[name]["int8"])

ppl_int8 = compute_perplexity(mdl, tok)
variants["calib_int8"] = {"ppl": round(ppl_int8, 2), "ratio": round(ppl_int8/baseline_ppl, 3)}
print(f"  calib_int8: PPL={ppl_int8:.2f} (ratio={ppl_int8/baseline_ppl:.3f})")

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


# ── Stage 4: Effective Rank Distribution ──────────────────────────────
print(f"\n{'='*60}")
print("STAGE 4: Effective Rank Distribution")
print(f"{'='*60}")

eff_ranks = []
top1_energies = []
layer_details = []

for name in tqdm(layers, desc="Effective rank"):
    W = orig_weights[name].float()
    out_f, in_f = W.shape

    sample_rank = min(64, out_f, in_f)
    U, S, V = torch.svd_lowrank(W, q=sample_rank, niter=5)

    s_squared = S ** 2
    total_energy = s_squared.sum()
    if total_energy > 0:
        cumsum = s_squared.sort(descending=True)[0].cumsum(dim=0)
        cumsum_norm = cumsum / total_energy

        mask_95 = cumsum_norm >= 0.95
        if mask_95.any():
            eff_rank = mask_95.nonzero()[0].item() + 1
        else:
            eff_rank = sample_rank

        eff_ranks.append(eff_rank)

        if len(s_squared) > 0:
            top1 = s_squared.sort(descending=True)[0][0] / total_energy
            top1_e = top1.item()
            top1_energies.append(top1_e)

            layer_details.append({
                "layer": name,
                "eff_rank": round(eff_rank, 2),
                "top1_energy": round(top1_e, 4),
                "shape": f"{out_f}x{in_f}"
            })

eff_stats = {
    "min": round(min(eff_ranks), 2) if eff_ranks else 0,
    "median": round(sorted(eff_ranks)[len(eff_ranks)//2], 2) if eff_ranks else 0,
    "max": round(max(eff_ranks), 2) if eff_ranks else 0,
    "top1_energy_median": round(sorted(top1_energies)[len(top1_energies)//2] * 100, 1) if top1_energies else 0,
}

layer_details.sort(key=lambda x: x["eff_rank"])
bottom_10 = layer_details[:10]

print(f"  Min eff_rank: {eff_stats['min']}")
print(f"  Median eff_rank: {eff_stats['median']}")
print(f"  Max eff_rank: {eff_stats['max']}")
print(f"  Top-1 energy median: {eff_stats['top1_energy_median']}%")

stage4_results = {
    "effective_rank_stats": eff_stats,
    "bottom_10_layers_by_eff_rank": bottom_10,
}


# ── Stage 5: Shared Basis Q/K/V ───────────────────────────────────────
print(f"\n{'='*60}")
print("STAGE 5: Shared Basis Q/K/V")
print(f"{'='*60}")

qkv_layers = []
for name in layers:
    if any(s in name for s in ("q_proj", "k_proj", "v_proj")):
        qkv_layers.append(name)

# Group by transformer block
qkv_groups = {}
for name in qkv_layers:
    parts = name.split(".")
    block_id = None
    for i, p in enumerate(parts):
        if p == "layers":
            if i + 1 < len(parts):
                block_id = ".".join(parts[:i+2])
                break

    if block_id is None:
        block_id = name.rsplit(".", 2)[0]

    if block_id not in qkv_groups:
        qkv_groups[block_id] = {}
    proj_type = parts[-1].replace("_proj", "")
    qkv_groups[block_id][proj_type] = name

print(f"  Found {len(qkv_layers)} QKV layers in {len(qkv_groups)} blocks")

# Check if Q and K have matching shapes (required for shared basis)
valid_blocks = {}
for block_id, projs in qkv_groups.items():
    q_name = projs.get("q")
    k_name = projs.get("k")
    if q_name and k_name:
        W_q = orig_weights[q_name]
        W_k = orig_weights[k_name]
        if W_q.shape == W_k.shape:
            valid_blocks[block_id] = projs

print(f"  Valid blocks (Q/K same shape): {len(valid_blocks)} / {len(qkv_groups)}")

if len(valid_blocks) == 0:
    print("  [WARN] No blocks with matching Q/K shapes — shared basis not applicable")
    stage5_results = {
        "total_qkv_layers": len(qkv_layers),
        "total_blocks": len(qkv_groups),
        "valid_blocks_shared_basis": 0,
        "note": "Q and K projections have different output dimensions. Shared basis requires matching shapes.",
    }
else:
    from collections import defaultdict as dd
    inputs_map = dd(list)

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0].detach().float()
            inputs_map[name].append(x.flatten(0, 1))
        return hook

    hooks = []
    for name in qkv_layers:
        mod = get_module_by_name(mdl, name)
        if mod is not None:
            hooks.append(mod.register_forward_hook(make_hook(name)))

    text = load_calib_text(tok, 2048)
    encoded = tok(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        _ = mdl(input_ids, attention_mask=attention_mask, use_cache=False)

    for h in hooks:
        h.remove()

    calib_qkv = {}
    for name in qkv_layers:
        if inputs_map[name]:
            cat = torch.cat(inputs_map[name], dim=0)
            if len(cat) > 2048:
                idx = torch.randperm(len(cat), device=cat.device)[:2048]
                cat = cat[idx]
            calib_qkv[name] = cat

    def get_module_by_name(model, name):
        mod = model
        for p in name.split("."):
            mod = getattr(mod, p, None)
            if mod is None:
                return None
        return mod

    shared_basis_results = {}
    test_ranks = [8, 16, 32]

    for test_rank in test_ranks:
        total_err = 0.0
        count = 0

        for name in layers:
            set_weight(mdl, name, orig_weights[name])

        compressed_count = 0

        for block_id, projs in tqdm(valid_blocks.items(), desc=f"Shared basis rank={test_rank}"):
            q_name = projs.get("q")
            k_name = projs.get("k")
            if not q_name or not k_name:
                continue

            W_q = orig_weights[q_name].float()
            W_k = orig_weights[k_name].float()
            W_shared = (W_q + W_k) / 2

            X_calib = calib_qkv.get(q_name)
            if X_calib is None:
                continue

            diag_c = (X_calib ** 2).mean(dim=0).clamp(min=1e-8)
            W_weighted = W_shared * torch.sqrt(diag_c).unsqueeze(0)
            U, S, V = torch.svd_lowrank(W_weighted, q=test_rank, niter=5)
            W_approx = U @ torch.diag(S) @ V.T
            W_approx = W_approx / torch.sqrt(diag_c).unsqueeze(0)

            err_q = (W_q - W_approx).pow(2).mean().sqrt().item()
            err_k = (W_k - W_approx).pow(2).mean().sqrt().item()
            total_err += (err_q + err_k) / 2
            count += 1

            set_weight(mdl, q_name, W_approx)
            if k_name != q_name:
                set_weight(mdl, k_name, W_approx)

            compressed_count += 1

        avg_err = total_err / count if count > 0 else float("inf")
        ppl_shared = compute_perplexity(mdl, tok)

        shared_basis_results[f"rank_{test_rank}"] = {
            "avg_error": round(avg_err, 2),
            "ppl": round(ppl_shared, 2),
            "ratio": round(ppl_shared/baseline_ppl, 1),
            "blocks_compressed": compressed_count,
        }

        print(f"  rank={test_rank}: avg_err={avg_err:.3f}, PPL={ppl_shared:.0f} (ratio={ppl_shared/baseline_ppl:.0f})")

        for name in layers:
            set_weight(mdl, name, orig_weights[name])

    stage5_results = {
        "total_qkv_layers": len(qkv_layers),
        "total_blocks": len(qkv_groups),
        "valid_blocks_shared_basis": len(valid_blocks),
        "results_by_rank": shared_basis_results,
    }


# ── Save all results ──────────────────────────────────────────────────
all_results = {
    "model": "HuggingFaceTB/SmolLM-135M",
    "baseline_ppl": baseline_ppl,
    "stage2_calibration": stage2_results,
    "stage3_sparse_compensation": stage3_results,
    "stage4_effective_rank": stage4_results,
    "stage5_shared_basis_qkv": stage5_results,
}

with open(RESULTS_DIR / "stages_2_5_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print("Stages 2-5 complete!")
print(f"{'='*60}")
print(f"Results saved to {RESULTS_DIR}/stages_2_5_results.json")
