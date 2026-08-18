#!/usr/bin/env python3
"""Debug GPTQ — find the exact cause of PPL=trillions."""

import sys, torch, math
from pathlib import Path

if sys.platform == "win32" and not getattr(sys.stdout, 'encoding', '').lower().startswith('utf'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = Path(__file__).parent.parent.resolve()

from eval_utils_v5 import (
    compute_perplexity, load_wikitext_eval, collect_calibration_inputs,
    get_compressible_layers, get_weight, set_weight,
)

model_path = str(ROOT_DIR / "models/smollm-135m")

tok = AutoTokenizer.from_pretrained(model_path)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(DEVICE)
mdl.eval()

encoded = load_wikitext_eval(tok)
baseline_ppl = compute_perplexity(mdl, tok, encoded)
print(f"Baseline PPL: {baseline_ppl:.4f}")

layers = get_compressible_layers(mdl)
device = next(mdl.parameters()).device
calib_inputs = collect_calibration_inputs(mdl, layers, tok, n_tokens=2048)

# ── Test 1: Naive per-channel quantization (NO compensation) ──
print("\n--- Test 1: Naive per-channel INT4 (no compensation) ---")
mdl1 = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(DEVICE)
mdl1.eval()

qmax = 7
for name in layers[:10]:  # first 10 layers only for speed
    W_orig = get_weight(mdl1, name).detach().float()
    out_f, in_f = W_orig.shape
    W_q = torch.zeros_like(W_orig)
    
    for col in range(in_f):
        w_col = W_orig[:, col]
        s = w_col.abs().amax().clamp(min=1e-8) / qmax
        W_q[:, col] = torch.round(w_col / s).clamp(-qmax, qmax) * s
    
    set_weight(mdl1, name, W_q.to(W_orig.dtype))

ppl1 = compute_perplexity(mdl1, tok, encoded)
print(f"After 10 layers naive per-channel: PPL={ppl1:.4f} (ratio={ppl1/baseline_ppl:.3f}x)")

# ── Test 2: Groupwise INT4 (no compensation) ──
print("\n--- Test 2: Groupwise INT4 group_size=128 (no compensation) ---")
mdl2 = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(DEVICE)
mdl2.eval()

group_size = 128
for name in layers[:10]:
    W_orig = get_weight(mdl2, name).detach().float()
    out_f, in_f = W_orig.shape
    W_q = torch.zeros_like(W_orig)
    
    for col in range(in_f):
        w_col = W_orig[:, col]
        for i in range(0, out_f, group_size):
            end = min(i + group_size, out_f)
            block = w_col[i:end]
            s = block.abs().amax().clamp(min=1e-8) / qmax
            W_q[i:end, col] = torch.round(block / s).clamp(-qmax, qmax) * s
    
    set_weight(mdl2, name, W_q.to(W_orig.dtype))

ppl2 = compute_perplexity(mdl2, tok, encoded)
print(f"After 10 layers groupwise: PPL={ppl2:.4f} (ratio={ppl2/baseline_ppl:.3f}x)")

# ── Test 3: Groupwise + error compensation ──
print("\n--- Test 3: Groupwise INT4 + Hessian diagonal compensation ---")
mdl3 = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(DEVICE)
mdl3.eval()

for name in layers[:10]:
    W_orig = get_weight(mdl3, name).detach().float()
    out_f, in_f = W_orig.shape
    X = calib_inputs.get(name, torch.randn(64, in_f, device=device))
    H_diag = (X ** 2).mean(dim=0).clamp(min=1e-8)
    
    W_work = W_orig.clone()
    W_q = torch.zeros_like(W_work)
    
    for col in range(in_f):
        w_col = W_work[:, col].clone()
        
        for i in range(0, out_f, group_size):
            end = min(i + group_size, out_f)
            block = w_col[i:end]
            s = block.abs().amax().clamp(min=1e-8) / qmax
            W_q[i:end, col] = torch.round(block / s).clamp(-qmax, qmax) * s
        
        err = w_col - W_q[:, col]
        
        if col < in_f - 1:
            h_j = H_diag[col].clamp(min=1e-8)
            ratio = (H_diag[col + 1:] / h_j)
            # Check for extreme ratios
            if ratio.max() > 1000:
                print(f"  WARN: {name} col={col}, max_ratio={ratio.max():.1f}, H_diag[{col}]={H_diag[col]:.6f}")
            W_work[:, col + 1:] -= err.unsqueeze(1) * ratio.unsqueeze(0)
    
    # Check weight norm preservation
    orig_norm = W_orig.pow(2).sum().sqrt().item()
    q_norm = W_q.pow(2).sum().sqrt().item()
    if abs(q_norm / orig_norm - 1) > 0.5:
        print(f"  WARN: {name} norm ratio={q_norm/orig_norm:.3f}")
    
    set_weight(mdl3, name, W_q.to(W_orig.dtype))

ppl3 = compute_perplexity(mdl3, tok, encoded)
print(f"After 10 layers groupwise+comp: PPL={ppl3:.4f} (ratio={ppl3/baseline_ppl:.3f}x)")

# ── Test 4: Check if H_diag has extreme values ──
print("\n--- Test 4: Hessian diagonal statistics ---")
for name in layers[:5]:
    X = calib_inputs.get(name, torch.randn(64, 1, device=device))
    H_diag = (X ** 2).mean(dim=0)
    print(f"  {name}: min={H_diag.min():.6f}, max={H_diag.max():.6f}, "
          f"ratio={H_diag.max()/H_diag.min():.1f}x, shape={X.shape}")

# ── Test 5: Full GPTQ on all layers with compensation disabled after extreme ratios ──
print("\n--- Test 5: Full GPTQ (all layers) with safe compensation ---")
mdl5 = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(DEVICE)
mdl5.eval()

for name in layers:
    W_orig = get_weight(mdl5, name).detach().float()
    out_f, in_f = W_orig.shape
    X = calib_inputs.get(name, torch.randn(64, in_f, device=device))
    H_diag = (X ** 2).mean(dim=0).clamp(min=1e-8)
    
    W_work = W_orig.clone()
    W_q = torch.zeros_like(W_work)
    
    for col in range(in_f):
        w_col = W_work[:, col].clone()
        
        for i in range(0, out_f, group_size):
            end = min(i + group_size, out_f)
            block = w_col[i:end]
            s = block.abs().amax().clamp(min=1e-8) / qmax
            W_q[i:end, col] = torch.round(block / s).clamp(-qmax, qmax) * s
        
        err = w_col - W_q[:, col]
        
        if col < in_f - 1:
            h_j = H_diag[col].clamp(min=1e-8)
            ratio = (H_diag[col + 1:] / h_j).clamp(max=10.0)  # SAFETY CLAMP
            W_work[:, col + 1:] -= err.unsqueeze(1) * ratio.unsqueeze(0)
    
    set_weight(mdl5, name, W_q.to(W_orig.dtype))

ppl5 = compute_perplexity(mdl5, tok, encoded)
print(f"Full GPTQ safe comp: PPL={ppl5:.4f} (ratio={ppl5/baseline_ppl:.3f}x)")

# ── Test 6: Full naive per-channel (no compensation at all) ──
print("\n--- Test 6: Full naive per-channel INT4 (all layers, no comp) ---")
mdl6 = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(DEVICE)
mdl6.eval()

for name in layers:
    W_orig = get_weight(mdl6, name).detach().float()
    out_f, in_f = W_orig.shape
    W_q = torch.zeros_like(W_orig)
    
    for col in range(in_f):
        w_col = W_orig[:, col]
        s = w_col.abs().amax().clamp(min=1e-8) / qmax
        W_q[:, col] = torch.round(w_col / s).clamp(-qmax, qmax) * s
    
    set_weight(mdl6, name, W_q.to(W_orig.dtype))

ppl6 = compute_perplexity(mdl6, tok, encoded)
print(f"Full naive per-channel: PPL={ppl6:.4f} (ratio={ppl6/baseline_ppl:.3f}x)")

# ── Test 7: Full groupwise without compensation ──
print("\n--- Test 7: Full groupwise INT4 (all layers, no comp) ---")
mdl7 = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(DEVICE)
mdl7.eval()

for name in layers:
    W_orig = get_weight(mdl7, name).detach().float()
    out_f, in_f = W_orig.shape
    W_q = torch.zeros_like(W_orig)
    
    for col in range(in_f):
        w_col = W_orig[:, col]
        for i in range(0, out_f, group_size):
            end = min(i + group_size, out_f)
            block = w_col[i:end]
            s = block.abs().amax().clamp(min=1e-8) / qmax
            W_q[i:end, col] = torch.round(block / s).clamp(-qmax, qmax) * s
    
    set_weight(mdl7, name, W_q.to(W_orig.dtype))

ppl7 = compute_perplexity(mdl7, tok, encoded)
print(f"Full groupwise no comp: PPL={ppl7:.4f} (ratio={ppl7/baseline_ppl:.3f}x)")

print("\n\n=== SUMMARY ===")
print(f"Baseline:              {baseline_ppl:.4f}")
print(f"T1 naive 10 layers:    {ppl1:.4f} ({ppl1/baseline_ppl:.2f}x)")
print(f"T2 groupwise 10 layers:{ppl2:.4f} ({ppl2/baseline_ppl:.2f}x)")
print(f"T3 groupwise+comp 10l: {ppl3:.4f} ({ppl3/baseline_ppl:.2f}x)")
print(f"T5 full safe comp:     {ppl5:.4f} ({ppl5/baseline_ppl:.2f}x)")
print(f"T6 full naive per-ch:  {ppl6:.4f} ({ppl6/baseline_ppl:.2f}x)")
print(f"T7 full groupwise:     {ppl7:.4f} ({ppl7/baseline_ppl:.2f}x)")
