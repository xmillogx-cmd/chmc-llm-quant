#!/usr/bin/env python3
"""CHMC v4 Stages 4-5 for Qwen — effective rank + shared basis."""
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
        raise ValueError(f"Layer '{layer_name}' not found")
    return mod.weight

def _set_weight(model, layer_name, weight):
    mod = _get_module_by_name(model, layer_name)
    with torch.no_grad():
        mod.weight.copy_(weight.to(mod.weight.dtype))

def get_compressible_layers(model):
    skip_prefixes = ("embed_tokens", "lm_head")
    return [name for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and mod.in_features > 0 and mod.out_features > 0
            and not any(name.startswith(pfx) for pfx in skip_prefixes)]

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
            return perplexity if not math.isinf(perplexity) else 1e9
        except Exception:
            pass
    
    print(f"[PPL] Generating {n_tokens} tokens...")
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

# ── Load previous results ─────────────────────────────────────────────
with open(RESULTS_DIR / "stage1_results.json") as f:
    stage1 = json.load(f)

baseline_ppl = stage1["baseline_ppl"]
layers = get_compressible_layers(mdl)
orig_weights = {name: _get_weight(mdl, name).detach().clone() for name in layers}

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

# Bottom 10 layers by effective rank
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

# Find QKV projection layers
qkv_layers = []
for name in layers:
    if any(s in name for s in ("q_proj", "k_proj", "v_proj")):
        qkv_layers.append(name)

# Group by transformer block
qkv_groups = {}
for name in qkv_layers:
    parts = name.split(".")
    # Find the transformer layer index
    block_id = None
    for i, p in enumerate(parts):
        if p == "layers":
            # Next part should be layer index
            if i + 1 < len(parts):
                block_id = ".".join(parts[:i+2])
                break
    
    if block_id is None:
        block_id = name.rsplit(".", 2)[0]  # Take up to parent of projection
    
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
        "note": "Q and K projections have different output dimensions in this model architecture, shared basis not applicable",
        "results_by_rank": {},
    }
else:
    # Collect calibration inputs for Q/K/V
    from collections import defaultdict
    inputs_map = defaultdict(list)

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0].detach().float()
            inputs_map[name].append(x.flatten(0, 1))
        return hook

    hooks = []
    for name in qkv_layers:
        mod = _get_module_by_name(mdl, name)
        if mod is not None:
            hooks.append(mod.register_forward_hook(make_hook(name)))

    text = "The quick brown fox jumps over the lazy dog. " * 100
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

    # Test shared basis at different ranks (only on valid blocks with matching Q/K shapes)
    shared_basis_results = {}
    test_ranks = [8, 16, 32]

    for test_rank in test_ranks:
        total_err = 0.0
        count = 0

        # Restore originals
        for name in layers:
            _set_weight(mdl, name, orig_weights[name])

        compressed_count = 0

        for block_id, projs in tqdm(valid_blocks.items(), desc=f"Shared basis rank={test_rank}"):
            q_name = projs.get("q")
            k_name = projs.get("k")
            if not q_name or not k_name:
                continue

            W_q = orig_weights[q_name].float()
            W_k = orig_weights[k_name].float()

            # Average as shared basis (only works when shapes match)
            W_shared = (W_q + W_k) / 2

            X_calib = calib_qkv.get(q_name)
            if X_calib is None:
                continue

            # Weighted SVD on shared basis
            diag_c = (X_calib ** 2).mean(dim=0).clamp(min=1e-8)
            W_weighted = W_shared * torch.sqrt(diag_c).unsqueeze(0)
            U, S, V = torch.svd_lowrank(W_weighted, q=test_rank, niter=5)
            W_approx = U @ torch.diag(S) @ V.T
            W_approx = W_approx / torch.sqrt(diag_c).unsqueeze(0)

            err_q = (W_q - W_approx).pow(2).mean().sqrt().item()
            err_k = (W_k - W_approx).pow(2).mean().sqrt().item()
            total_err += (err_q + err_k) / 2
            count += 1

            _set_weight(mdl, q_name, W_approx)
            if k_name != q_name:
                _set_weight(mdl, k_name, W_approx)

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

        # Restore for next iteration
        for name in layers:
            _set_weight(mdl, name, orig_weights[name])

    stage5_results = {
        "total_qkv_layers": len(qkv_layers),
        "total_blocks": len(qkv_groups),
        "valid_blocks_shared_basis": len(valid_blocks),
        "results_by_rank": shared_basis_results,
    }

# ── Save results ──────────────────────────────────────────────────────
all_results = {
    "model": "Qwen/Qwen2.5-0.5B",
    "baseline_ppl": baseline_ppl,
    "stage4_effective_rank": stage4_results,
    "stage5_shared_basis_qkv": stage5_results,
}

with open(RESULTS_DIR / "stages_4_5_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print("Stages 4-5 complete!")
print(f"{'='*60}")
print(f"Results saved to {RESULTS_DIR}/stages_4_5_results.json")
