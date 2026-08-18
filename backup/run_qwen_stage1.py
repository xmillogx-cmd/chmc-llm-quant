#!/usr/bin/env python3
"""Run CHMC v4 pipeline (Stages 0-1) on Qwen2.5-0.5B — local model, single-pass calibration."""
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

# ── Load model (local) ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = str(_ROOT / "models" / "qwen2.5-0.5b")

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

# ── Helpers ────────────────────────────────────────────────────────────
def _get_module_by_name(model, name):
    mod = model
    for p in name.split("."):
        mod = getattr(mod, p, None)
        if mod is None:
            return None
    return mod

def _get_weight(model, layer_name):
    mod = _get_module_by_name(model, layer_name)
    if mod is None or not isinstance(mod, nn.Linear):
        raise ValueError(f"Layer '{layer_name}' not found or not Linear")
    return mod.weight

def _set_weight(model, layer_name, weight):
    mod = _get_module_by_name(model, layer_name)
    with torch.no_grad():
        mod.weight.copy_(weight.to(mod.weight.dtype))

def get_compressible_layers(model):
    skip_prefixes = ("embed_tokens", "lm_head")
    compressible = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.in_features > 0 and mod.out_features > 0:
            if any(name.startswith(pfx) for pfx in skip_prefixes):
                continue
            compressible.append(name)
    return compressible

def honest_compression_bits(out_f, in_f, rank, original_bits=32.0, factor_bits=16.0, residual_bits=4.0, residual_density=1.0):
    original = original_bits * out_f * in_f
    lowrank = factor_bits * rank * (out_f + in_f)
    n_residual = int(residual_density * out_f * in_f)
    residual_vals = n_residual * residual_bits
    is_sparse = residual_density < 0.99
    residual_idx = n_residual * (16.0 * 2) if is_sparse else 0.0
    n_groups = (out_f * in_f + 63) // 64
    scales = n_groups * 16.0
    compressed = lowrank + residual_vals + residual_idx + scales
    return {
        "original_bits": original,
        "compressed_bits": compressed if compressed > 0 else 1.0,
        "compression_ratio": original / (compressed if compressed > 0 else 1.0),
        "bits_per_weight": compressed / (out_f * in_f) if compressed > 0 else 32.0,
    }

def collect_all_calibration_inputs(model, layer_names, tokenizer, n_tokens=2048):
    """Collect input activations for ALL layers in a SINGLE forward pass."""
    inputs_map = {name: [] for name in layer_names}
    
    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0].detach().float()
            x_flat = x.flatten(0, 1)  # [batch*seq, in_f] — same as chmc_v4
            inputs_map[name].append(x_flat)
        return hook
    
    hooks = []
    for name in layer_names:
        mod = _get_module_by_name(model, name)
        if mod is not None:
            hooks.append(mod.register_forward_hook(make_hook(name)))
    
    text = "The quick brown fox jumps over the lazy dog. " * 100
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=min(n_tokens, 2048))
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask, use_cache=False)
    
    for h in hooks:
        h.remove()
    
    result = {}
    for name in layer_names:
        if inputs_map[name]:
            cat = torch.cat(inputs_map[name], dim=0)  # [total_tokens, in_f]
            if len(cat) > n_tokens:
                idx = torch.randperm(len(cat), device=cat.device)[:n_tokens]
                cat = cat[idx]
            result[name] = cat
        else:
            in_f = _get_weight(model, name).shape[1]
            result[name] = torch.randn(64, in_f, device=DEVICE)  # fallback
    
    return result

from chmc_v4 import quantize_symmetric_per_channel

def weighted_svd_compress(W, X, rank):
    W = W.float()
    out_f, in_f = W.shape
    rank = min(rank, out_f, in_f)
    diag_c = (X ** 2).mean(dim=0).clamp(min=1e-8)
    W_weighted = W * torch.sqrt(diag_c).unsqueeze(0)
    U, S, V = torch.svd_lowrank(W_weighted, q=rank, niter=5)
    W_approx = U @ torch.diag(S) @ V.T
    W_approx = W_approx / torch.sqrt(diag_c).unsqueeze(0)
    err = (W - W_approx).pow(2).sum().sqrt() / W.pow(2).sum().sqrt()
    return W_approx, err.item()

def select_best_lowrank_method(W, X, rank):
    wsvd_W, wsvd_err = weighted_svd_compress(W, X, rank)
    return "weighted_svd", wsvd_W, wsvd_err

def compute_perplexity(model, tokenizer, n_tokens=2000, max_len=512, stride=256):
    WIKITEXT_CACHE = _ROOT / ".wikitext_cache.pt"
    
    if WIKITEXT_CACHE.exists():
        try:
            cached = torch.load(WIKITEXT_CACHE, map_location=DEVICE, weights_only=True)
            encoded = cached[:n_tokens]
            
            nll_total = 0.0
            token_count = 0
            with torch.no_grad():
                for i in range(0, len(encoded) - max_len + 1, stride):
                    chunk = encoded[i:i + max_len].unsqueeze(0)
                    attn_mask = (chunk != tokenizer.pad_token_id).to(DEVICE) if tokenizer.pad_token_id is not None else torch.ones_like(chunk)
                    outputs = model(chunk, attention_mask=attn_mask, use_cache=False)
                    logits = outputs.logits
                    shift_logits = logits[:, :-1, :].contiguous().float()
                    shift_labels = encoded[i:i + max_len][None, 1:]
                    ce = F.cross_entropy(shift_logits.flatten(0, 1), shift_labels.flatten(0, 1), reduction="sum")
                    if not torch.isnan(ce):
                        nll_total += ce.item()
                        token_count += (max_len - 1)
            if token_count == 0:
                return float("inf")
            avg_nll = nll_total / token_count
            perplexity = math.exp(min(avg_nll, 50.0))
            if math.isinf(perplexity):
                perplexity = 1e9
            return perplexity
        except Exception:
            pass
    
    print(f"[PPL] Generating {n_tokens} tokens for evaluation...")
    model.eval()
    with torch.no_grad():
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
        seed_id = torch.tensor([[bos_id]], device=DEVICE)
        text_ids = model.generate(seed_id, max_new_tokens=n_tokens - 1, do_sample=False)
    
    encoded = text_ids.flatten().tolist()[:n_tokens]
    
    nll_total = 0.0
    token_count = 0
    with torch.no_grad():
        for i in range(0, len(encoded) - max_len + 1, stride):
            chunk = torch.tensor(encoded[i:i + max_len], device=DEVICE).unsqueeze(0)
            outputs = model(chunk, use_cache=False)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = torch.tensor(encoded[i:i + max_len][1:], device=DEVICE).unsqueeze(0)
            ce = F.cross_entropy(shift_logits.flatten(0, 1), shift_labels.flatten(0, 1), reduction="sum")
            if not torch.isnan(ce):
                nll_total += ce.item()
                token_count += (max_len - 1)
    
    if token_count == 0:
        return float("inf")
    avg_nll = nll_total / token_count
    perplexity = math.exp(min(avg_nll, 50.0))
    return perplexity

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
calib_inputs = collect_all_calibration_inputs(mdl, layers, tok, 2048)
# Verify calibration dimensions match weight input features
for name, X in calib_inputs.items():
    W = _get_weight(mdl, name)
    if X.shape[1] != W.shape[1]:
        print(f"  [WARN] {name}: calib dim={X.shape[1]} vs weight in_f={W.shape[1]}, using random")
        calib_inputs[name] = torch.randn(64, W.shape[1], device=DEVICE)
print(f"[OK] Calibration collected for {len(calib_inputs)} layers (shape: [tokens, in_f])")

# ── Stage 1: Scalar baselines + CHMC comparison ───────────────────────
print(f"\n{'='*60}")
print("STAGE 1: Scalar Baselines + CHMC")
print(f"{'='*60}")

# Save originals once
orig_weights = {}
for name in layers:
    orig_weights[name] = _get_weight(mdl, name).detach().clone()

# Scalar quantization baselines
scalar_results = {}
for bits in [2, 3, 4, 5, 6]:
    print(f"\n--- scalar_q{bits} ---")
    
    for name in layers:
        W = _get_weight(mdl, name)
        W_q, scale = quantize_symmetric_per_channel(W, bits=bits, dim=0)
        _set_weight(mdl, name, W_q)
    
    ppl = compute_perplexity(mdl, tok)
    
    total_comp = 0.0
    total_weights = 0
    for name in layers:
        W = orig_weights[name]
        out_f, in_f = W.shape
        bw = bits + 16.0 / (out_f * in_f)
        total_comp += out_f * in_f * bw
        total_weights += out_f * in_f
    
    actual_bw = total_comp / total_weights if total_weights else 0
    
    scalar_results[f"scalar_q{bits}"] = {
        "ppl": round(ppl, 4),
        "ppl_ratio": round(ppl/baseline_ppl, 3),
        "bit_per_weight": round(actual_bw, 3),
    }
    
    print(f"  PPL: {ppl:.4f} (ratio={ppl/baseline_ppl:.3f}), bw={actual_bw:.3f}")
    
    # Restore originals
    for name in layers:
        _set_weight(mdl, name, orig_weights[name])

# CHMC at different ranks — using pre-collected calibration
chmc_results = {}
for rank in [4, 8]:
    print(f"\n--- CHMC rank={rank} ---")
    
    # Restore originals before each rank
    for name in layers:
        _set_weight(mdl, name, orig_weights[name])
    
    total_comp = 0.0
    total_weights = 0
    compressed_count = 0
    
    for name in tqdm(layers, desc=f"CHMC rank={rank}"):
        W_orig = orig_weights[name]
        X_calib = calib_inputs.get(name)
        
        if X_calib is None or X_calib.numel() == 0:
            in_f = W_orig.shape[1]
            X_calib = torch.randn(64, in_f, device=DEVICE)
        
        _, W_lr, err = select_best_lowrank_method(W_orig, X_calib, rank)
        R_full = W_orig - W_lr
        
        # Quantize residual to INT4
        R_q, _ = quantize_symmetric_per_channel(R_full, bits=4, dim=0)
        W_comp = W_lr + R_q
        
        _set_weight(mdl, name, W_comp)
        
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
    
    # Restore originals for next rank
    for name in layers:
        _set_weight(mdl, name, orig_weights[name])

# ── Save results ───────────────────────────────────────────────────────
results_dir = _ROOT / "results_v4" / "qwen2.5-0.5b"
results_dir.mkdir(parents=True, exist_ok=True)

result = {
    "model": "Qwen/Qwen2.5-0.5B",
    "local_path": MODEL_NAME,
    "baseline_ppl": round(baseline_ppl, 4),
    "total_layers": len(layers),
    "scalar_baselines": scalar_results,
    "chmc_results": chmc_results,
}

with open(results_dir / "stage1_results.json", "w") as f:
    json.dump(result, f, indent=2)

# ── Summary ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"STAGE 1 SUMMARY (Qwen/Qwen2.5-0.5B)")
print(f"{'='*60}")
print(f"Baseline PPL: {baseline_ppl:.4f}")

for method, info in scalar_results.items():
    print(f"{method}: PPL={info['ppl']}, ratio={info['ppl_ratio']}x, bw={info['bit_per_weight']}")

for method, info in chmc_results.items():
    print(f"CHMC {method}: PPL={info['ppl']}, ratio={info['ratio']}x, bw={info['bw']}")

if "rank_4" in chmc_results and "scalar_q4" in scalar_results:
    chmc_bw = chmc_results["rank_4"]["bw"]
    sc_bw = scalar_results["scalar_q4"]["bit_per_weight"]
    print(f"\nCHMC rank=4 at bw={chmc_bw:.2f} → ratio {chmc_results['rank_4']['ratio']}x")
    print(f"scalar_q4 at bw={sc_bw:.2f} → ratio {scalar_results['scalar_q4']['ppl_ratio']}x")
    
    if chmc_results["rank_4"]["ratio"] < scalar_results["scalar_q4"]["ppl_ratio"]:
        improvement = scalar_results["scalar_q4"]["ppl_ratio"] / chmc_results["rank_4"]["ratio"]
        print(f"CHMC wins by {improvement:.1f}x in PPL ratio!")

print(f"\nResults saved to {results_dir}/stage1_results.json")
print("\nDone.")
