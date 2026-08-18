#!/usr/bin/env python3
"""
run_third_model.py — Третья модель с MHA (GAP-3 fix)
=====================================================

Проблема v4: Обе модели (SmolLM, Qwen2.5) используют GQA — q_proj и k_proj
разных размеров → shared basis не работает.

Решение: Протестировать модель с классическим MHA где q/k/v одинаковых размеров.
Кандидаты: gpt2 (124M), facebook/opt-125m, EleutherAI/pythia-160m

Usage:
    python v5/run_third_model.py --model gpt2
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).parent.resolve()       # v5/
ROOT_DIR = BASE_DIR.parent                        # cmq_experiment/
RESULTS  = ROOT_DIR / "results_v5"
RESULTS.mkdir(parents=True, exist_ok=True)

from eval_utils_v5 import (
    compute_perplexity, load_wikitext_eval,
    collect_calibration_inputs, weighted_svd_compress,
    get_compressible_layers, get_weight, set_weight,
    quantize_symmetric_per_channel, honest_compression_bits
)


def get_all_compressible_layers(model: AutoModelForCausalLM) -> list[str]:
    """Get all compressible layers including Conv1D (GPT-2 style)."""
    skip_prefixes = ("embed_tokens", "lm_head", "wte", "wpe")
    layers = []
    for name, mod in model.named_modules():
        if any(name.startswith(pfx) for pfx in skip_prefixes):
            continue
        # Support both nn.Linear and Conv1D (GPT-2)
        if isinstance(mod, (nn.Linear,)):
            if mod.in_features > 0 and mod.out_features > 0:
                layers.append(name)
        elif hasattr(mod, "weight") and hasattr(mod, "nf"):  # Conv1D
            weight = mod.weight
            if weight.dim() >= 2 and weight.shape[0] > 0 and weight.shape[1] > 0:
                layers.append(name)
    return layers


def get_weight_generic(model, layer_name):
    """Get weight tensor for both Linear and Conv1D."""
    mod = model
    for p in layer_name.split("."):
        mod = getattr(mod, p, None)
        if mod is None:
            raise ValueError(f"Layer '{layer_name}' not found")
    return mod.weight


def set_weight_generic(model, layer_name, weight_tensor):
    """Set weight tensor for both Linear and Conv1D."""
    mod = model
    for p in layer_name.split("."):
        mod = getattr(mod, p, None)
        if mod is None:
            raise ValueError(f"Layer '{layer_name}' not found")
    with torch.no_grad():
        mod.weight.copy_(weight_tensor.to(mod.weight.dtype))


def get_compressible_layers(model: AutoModelForCausalLM) -> list[str]:
    """Get all compressible layers (Linear and Conv1D for GPT-2)."""
    skip_prefixes = ("embed_tokens", "lm_head", "wte", "wpe")
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        Conv1D = type(None)  # Fallback if not available
    layers = []
    for name, mod in model.named_modules():
        if any(name.startswith(pfx) for pfx in skip_prefixes):
            continue
        if isinstance(mod, nn.Linear):
            if mod.in_features > 0 and mod.out_features > 0:
                layers.append(name)
        elif Conv1D is not None and isinstance(mod, Conv1D):
            # Conv1D weight shape: [nf, nin] where nf=output features
            if mod.nf > 0 and mod.nin > 0:
                layers.append(name)
    return layers


def get_weight_generic(model, layer_name):
    """Get weight tensor for both Linear and Conv1D."""
    mod = model
    for p in layer_name.split("."):
        mod = getattr(mod, p, None)
        if mod is None:
            raise ValueError(f"Layer '{layer_name}' not found")
    return mod.weight


def set_weight_generic(model, layer_name, weight_tensor):
    """Set weight tensor for both Linear and Conv1D."""
    mod = model
    for p in layer_name.split("."):
        mod = getattr(mod, p, None)
        if mod is None:
            raise ValueError(f"Layer '{layer_name}' not found")
    with torch.no_grad():
        mod.weight.copy_(weight_tensor.to(mod.weight.dtype))


def check_mha(model: AutoModelForCausalLM) -> dict:
    """Check if model uses MHA (q/k/v same size) vs GQA."""
    layers = get_compressible_layers(model)

    mha_layers = []
    gqa_layers = []

    for name in layers:
        mod = model
        for p in name.split("."):
            mod = getattr(mod, p, None)
            if mod is None:
                break
        if not isinstance(mod, nn.Linear):
            continue

        # Check if this is a q_proj — look at sibling k_proj/v_proj
        parent_name = ".".join(name.split(".")[:-1])
        parent = model
        for p in parent_name.split("."):
            parent = getattr(parent, p, None)
            if parent is None:
                break

        if parent is not None and "q_proj" in name:
            k_mod = getattr(parent, "k_proj", None)
            v_mod = getattr(parent, "v_proj", None)
            if k_mod and isinstance(k_mod, nn.Linear):
                q_in, q_out = mod.in_features, mod.out_features
                k_in, k_out = k_mod.in_features, k_mod.out_features

                if q_out == k_out:
                    mha_layers.append(name)
                else:
                    gqa_layers.append(name)

    return {
        "mha_count": len(mha_layers),
        "gqa_count": len(gqa_layers),
        "is_mha_model": len(gqa_layers) == 0,
        "sample_mha": mha_layers[:3] if mha_layers else [],
        "sample_gqa": gqa_layers[:3] if gqa_layers else [],
    }


def run_chmc_on_third_model(model_path: str, rank: int = 4) -> dict:
    """Run full CHMC pipeline on a third model (MHA candidate)."""

    print(f"\nLoading model...")
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map="cpu"
    )
    mdl.eval()

    # Check MHA vs GQA
    mha_info = check_mha(mdl)
    print(f"MHA info: {mha_info}")

    encoded = load_wikitext_eval(tok)
    baseline_ppl = compute_perplexity(mdl, tok, encoded)
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    layers = get_compressible_layers(mdl)
    orig_weights = {n: get_weight(mdl, n).detach().clone() for n in layers}
    calib = collect_calibration_inputs(mdl, layers, tok, 1024)

    # Compress all layers with CHMC
    cos_sims = []
    recon_errors = []
    bit_infos = {}

    for name in tqdm(layers, desc=f"CHMC rank={rank}"):
        W = orig_weights[name]
        X = calib.get(name)
        if X is None or X.numel() == 0:
            X = torch.randn(64, W.shape[1])

        # Ensure X has correct input features for this layer
        in_f = W.shape[1]
        if X.dim() < 2 or X.shape[-1] != in_f:
            X = torch.randn(64, in_f)
        else:
            X = X[:, :in_f] if X.shape[1] > in_f else X

        W_lr, lr_err = weighted_svd_compress(W, X, rank)
        R = W - W_lr
        R_q, _ = quantize_symmetric_per_channel(R, bits=4, dim=0)
        W_comp = W_lr + R_q

        set_weight(mdl, name, W_comp)

        recon_err = (W - W_comp).pow(2).sum().sqrt() / W.pow(2).sum().sqrt()
        recon_errors.append(float(recon_err))

        Y_orig = X @ W.T
        Y_comp = X @ W_comp.T
        cos = F.cosine_similarity(Y_orig.reshape(-1), Y_comp.reshape(-1), dim=0).item()
        cos_sims.append(cos)

        out_f, in_f = W.shape
        bit_infos[name] = honest_compression_bits(out_f, in_f, rank)

    full_ppl = compute_perplexity(mdl, tok, encoded)
    ratio = full_ppl / baseline_ppl if baseline_ppl > 0 else float("inf")

    # Compute overall BW
    total_comp = sum(v["compressed_bits"] for v in bit_infos.values())
    total_orig = sum(v["original_bits"] for v in bit_infos.values())
    avg_bw = (total_comp / len(bit_infos)) / (sum(w.shape[0]*w.shape[1] for w in orig_weights.values()) / len(orig_weights))

    print(f"Full model PPL: {full_ppl:.4f} (ratio={ratio:.3f}x)")
    print(f"Avg recon error: {sum(recon_errors)/len(recon_errors):.4f}")
    print(f"Avg cos_sim: {sum(cos_sims)/len(cos_sims):.4f}")

    # Restore
    for name in layers:
        set_weight(mdl, name, orig_weights[name])

    return {
        "model": model_path,
        "mha_info": mha_info,
        "baseline_ppl": round(baseline_ppl, 4),
        "full_model_ppl": round(full_ppl, 4),
        "ratio": round(ratio, 4),
        "avg_recon_error": round(sum(recon_errors) / len(recon_errors), 4),
        "avg_cos_sim": round(sum(cos_sims) / len(cos_sims), 4),
        "rank": rank,
        "layers_compressed": len(layers),
    }


def test_shared_basis(model_path: str, rank: int = 8) -> dict:
    """Test shared basis for q/k/v projections (only works on MHA models)."""

    print(f"\n--- Testing shared q/k/v basis ---")

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map="cpu"
    )
    mdl.eval()

    encoded = load_wikitext_eval(tok)
    baseline_ppl = compute_perplexity(mdl, tok, encoded)

    # Find attention blocks with q/k/v of same size (MHA)
    attn_blocks = []
    for name, mod in mdl.named_modules():
        if hasattr(mod, "q_proj") and hasattr(mod, "k_proj") and hasattr(mod, "v_proj"):
            q_out = mod.q_proj.out_features
            k_out = mod.k_proj.out_features
            v_out = mod.v_proj.out_features
            q_in  = mod.q_proj.in_features

            if q_out == k_out == v_out:
                attn_blocks.append({
                    "name": name,
                    "q_proj": f"{q_out}x{q_in}",
                    "k_proj": f"{k_out}x{q_in}",
                    "v_proj": f"{v_out}x{q_in}",
                })

    if not attn_blocks:
        print("  No MHA blocks found — shared basis not applicable")
        return {"error": "No MHA blocks", "blocks_found": 0}

    print(f"  Found {len(attn_blocks)} MHA attention blocks")

    # Save originals
    orig_weights = {}
    for block in attn_blocks:
        parent = mdl
        for p in block["name"].split("."):
            parent = getattr(parent, p, None)
        for proj in ["q_proj", "k_proj", "v_proj"]:
            key = f"{block['name']}.{proj}"
            orig_weights[key] = getattr(parent, proj).weight.detach().clone()

    # Apply shared basis: SVD of concatenated [W_q; W_k; W_v], project each back
    total_savings = 0
    savings_count = 0

    for block in tqdm(attn_blocks, desc="Shared basis"):
        parent = mdl
        for p in block["name"].split("."):
            parent = getattr(parent, p, None)

        W_q = parent.q_proj.weight.float()
        W_k = parent.k_proj.weight.float()
        W_v = parent.v_proj.weight.float()

        out_f, in_f = W_q.shape  # All same size for MHA
        r = min(rank, out_f, in_f)

        # Concatenate and SVD
        W_concat = torch.cat([W_q, W_k, W_v], dim=0)  # [3*out_f, in_f]
        U, S, V = torch.svd_lowrank(W_concat, q=r, niter=5)
        W_approx = U @ torch.diag(S) @ V.T

        # Split back
        W_q_shared = W_approx[:out_f]
        W_k_shared = W_approx[out_f:2*out_f]
        W_v_shared = W_approx[2*out_f:]

        # Set shared weights
        with torch.no_grad():
            parent.q_proj.weight.copy_(W_q_shared)
            parent.k_proj.weight.copy_(W_k_shared)
            parent.v_proj.weight.copy_(W_v_shared)

        # Bit savings: 3 separate low-rank vs 1 shared
        separate_bits = 3 * (16.0 * r * (out_f + in_f))
        shared_bits = 16.0 * r * (3*out_f + in_f)  # U is bigger, V shared
        savings = separate_bits - shared_bits
        total_savings += abs(savings)
        savings_count += 1

    # Measure PPL
    shared_ppl = compute_perplexity(mdl, tok, encoded)
    ratio = shared_ppl / baseline_ppl if baseline_ppl > 0 else float("inf")

    print(f"Shared basis PPL: {shared_ppl:.4f} (ratio={ratio:.3f}x)")

    # Restore
    for key, W_orig in orig_weights.items():
        parts = key.split(".")
        mod = mdl
        for p in parts[:-1]:
            mod = getattr(mod, p, None)
        proj = parts[-1]
        with torch.no_grad():
            getattr(mod, proj).weight.copy_(W_orig)

    return {
        "blocks_found": len(attn_blocks),
        "baseline_ppl": round(baseline_ppl, 4),
        "shared_basis_ppl": round(shared_ppl, 4),
        "ratio": round(ratio, 4),
        "rank": rank,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--rank", type=int, default=4)
    args = parser.parse_args()

    tag = Path(args.model).name or args.model.replace("/", "_")
    print(f"\n{'=' * 60}")
    print(f"Third model (MHA): {args.model}")
    print(f"{'=' * 60}")

    results = {}

    # CHMC on third model
    r1 = run_chmc_on_third_model(args.model, args.rank)
    results["chmc"] = r1

    # Shared basis test
    r2 = test_shared_basis(args.model, args.rank)
    results["shared_basis"] = r2

    out_file = RESULTS / f"third_model_{tag}_r{args.rank}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {out_file}")
