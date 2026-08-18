#!/usr/bin/env python3
"""Quick parameter sweep for CHMC v5."""
import sys, math
sys.path.insert(0, str(__file__).rsplit("\\", 1)[0].replace("\\", "/"))

from eval_utils_v5 import *
from stabilizers import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "../models/smollm-135m"
tok = AutoTokenizer.from_pretrained(model_path)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float32
    # BUG-10: no device_map="auto" on CPU — adds accelerate overhead
)
mdl.eval()

encoded = load_wikitext_eval(tok)
baseline_ppl = compute_perplexity(mdl, tok, encoded)
print(f"Baseline PPL: {baseline_ppl:.4f}")

layers = get_compressible_layers(mdl)
calib_inputs = collect_calibration_inputs(mdl, layers, tok, n_tokens=2048)
device = next(mdl.parameters()).device

results = []
for damp in [0.005, 0.01, 0.02]:
    for gs in [64, 128]:
        print(f"\nTesting damp={damp}, gs={gs}...")
        sys.stdout.flush()

        mdl2 = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, device_map="auto"
        )
        mdl2.eval()

        for name in layers:
            W_orig = get_weight(mdl2, name)
            X_calib = calib_inputs.get(name, torch.randn(64, W_orig.shape[1], device=device))

            diag_c_raw = (X_calib ** 2).mean(dim=0)
            lam = damp * diag_c_raw.mean().clamp(min=1e-8)
            diag_c = diag_c_raw + lam
            W_weighted = W_orig.float() * torch.sqrt(diag_c).unsqueeze(0)

            out_f, in_f = W_orig.shape
            rank = min(8, out_f, in_f)
            U, S, V = torch.svd_lowrank(W_weighted, q=rank, niter=5)
            W_lr = ((U * S.unsqueeze(0)) @ V.T) / torch.sqrt(diag_c).unsqueeze(0)

            R_full = W_orig.float() - W_lr
            R_q = quantize_with_compensation(R_full, X_calib, bits=4, group_size=gs, dampening=damp)
            W_comp = W_lr + R_q
            set_weight(mdl2, name, W_comp.to(W_orig.dtype))

        ppl = compute_perplexity(mdl2, tok, encoded)
        ratio = ppl / baseline_ppl
        print(f"  damp={damp}, gs={gs}: PPL={ppl:.4f} ratio={ratio:.3f}x")
        results.append((damp, gs, ppl, ratio))

        del mdl2
        torch.cuda.empty_cache()

print("\n\nSummary:")
for damp, gs, ppl, ratio in sorted(results, key=lambda r: r[3]):
    print(f"  damp={r[0]}, gs={r[1]}: PPL={r[2]:.4f} ratio={r[3]:.3f}x")
