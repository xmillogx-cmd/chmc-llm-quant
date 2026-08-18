#!/usr/bin/env python3
"""
full_calibration.py — Full-model calibration test (BUG-3/4 fix)
==============================================================

Проблема v4: Calibration тестировалась на 10 слоях из 210.
Этот скрипт проверяет PPL при сжатии ВСЕХ слоёв.

Варианты:
  1. Independent compression (all layers compressed with same calibration inputs)
  2. Sequential compression (inputs re-collected after each layer)
  3. Layerwise calibration (optimize A,B,R jointly per layer)

Usage:
    python v5/full_calibration.py --model models/smollm-135m --rank 8
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
    compute_perplexity, load_wikitext_eval, load_calib_text,
    collect_calibration_inputs, weighted_svd_compress,
    get_compressible_layers, get_weight, set_weight,
    quantize_symmetric_per_channel, honest_compression_bits
)


def calibrate_layer(
    W_orig: torch.Tensor, X: torch.Tensor, rank: int, residual_bits: int = 4,
    steps: int = 100, lr: float = 1e-3
) -> torch.Tensor:
    """
    Jointly optimize A, B, R for a single layer.

    Minimize ||WX^T - (AB^T + R_q)X^T||_F^2 using Adam on A, B.
    R uses STE (straight-through estimator) for differentiable quantization.
    """
    W = W_orig.float()
    out_f, in_f = W.shape
    rank = min(rank, out_f, in_f)

    # Initialize with weighted SVD
    W_lr, _ = weighted_svd_compress(W, X, rank)
    R_init = W - W_lr

    U, S, V = torch.svd_lowrank(W_lr, q=rank, niter=3)
    A = torch.nn.Parameter((U * S.unsqueeze(0)).clone())
    B = torch.nn.Parameter(V.clone())

    # Quantize residual (fixed scales per channel)
    R_abs_max = R_init.abs().amax(dim=0).clamp(min=1e-8)  # [in_f]
    scale = R_abs_max / (2 ** (residual_bits - 1) - 1)

    optimizer = torch.optim.Adam([A, B], lr=lr)

    for step in range(steps):
        W_hat = A @ B.T
        R_full = W - W_hat

        # Quantize residual with STE
        q = torch.round(R_full / scale).clamp(-(2**(residual_bits-1)-1), 2**(residual_bits-1)-1)
        R_q = q * scale
        W_compressed = W_hat + R_q

        # Loss: output MSE on calibration inputs
        Y_target = X @ W.T
        Y_pred   = X @ W_compressed.T
        loss = F.mse_loss(Y_pred, Y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final reconstruction
    with torch.no_grad():
        W_final_lr = A.detach() @ B.detach().T
        R_full = W - W_final_lr
        q = torch.round(R_full / scale).clamp(-(2**(residual_bits-1)-1), 2**(residual_bits-1)-1)
        W_compressed = W_final_lr + q * scale

    return W_compressed


def run_independent_compression(model_path: str, rank: int = 8, residual_bits: int = 4) -> dict:
    """Compress all layers independently with same calibration inputs."""
    print(f"\n--- Independent compression (rank={rank}) ---")

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map="cpu"
    )
    mdl.eval()

    encoded = load_wikitext_eval(tok)
    baseline_ppl = compute_perplexity(mdl, tok, encoded)
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    layers = get_compressible_layers(mdl)
    orig_weights = {n: get_weight(mdl, n).detach().clone() for n in layers}

    # Collect calibration inputs once
    calib = collect_calibration_inputs(mdl, layers, tok, 2048)

    cos_sims = []
    recon_errors = []

    for name in tqdm(layers, desc="Independent compress"):
        W = orig_weights[name]
        X = calib.get(name)
        if X is None or X.numel() == 0:
            X = torch.randn(64, W.shape[1])

        # Weighted SVD + quantized residual
        W_lr, lr_err = weighted_svd_compress(W, X, rank)
        R = W - W_lr
        R_q, _ = quantize_symmetric_per_channel(R, bits=residual_bits, dim=0)
        W_comp = W_lr + R_q

        set_weight(mdl, name, W_comp)

        # Track metrics
        recon_err = (W - W_comp).pow(2).sum().sqrt() / W.pow(2).sum().sqrt()
        recon_errors.append(float(recon_err))

        Y_orig = X @ W.T
        Y_comp = X @ W_comp.T
        cos = F.cosine_similarity(Y_orig.reshape(-1), Y_comp.reshape(-1), dim=0).item()
        cos_sims.append(cos)

    full_ppl = compute_perplexity(mdl, tok, encoded)
    ratio = full_ppl / baseline_ppl if baseline_ppl > 0 else float("inf")

    print(f"Full model PPL: {full_ppl:.4f} (ratio={ratio:.3f}x)")
    print(f"Avg recon error: {sum(recon_errors)/len(recon_errors):.4f}")
    print(f"Avg cos_sim: {sum(cos_sims)/len(cos_sims):.4f}")

    # Restore
    for name in layers:
        set_weight(mdl, name, orig_weights[name])

    return {
        "method": f"independent_r{rank}_res{residual_bits}",
        "baseline_ppl": round(baseline_ppl, 4),
        "full_model_ppl": round(full_ppl, 4),
        "ratio": round(ratio, 4),
        "avg_recon_error": round(sum(recon_errors) / len(recon_errors), 4),
        "avg_cos_sim": round(sum(cos_sims) / len(cos_sims), 4),
        "rank": rank,
        "residual_bits": residual_bits,
        "layers_compressed": len(layers),
    }


def run_sequential_compression(model_path: str, rank: int = 8, residual_bits: int = 4) -> dict:
    """Compress layers sequentially, re-collecting inputs after each."""
    print(f"\n--- Sequential compression (rank={rank}) ---")

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map="cpu"
    )
    mdl.eval()

    encoded = load_wikitext_eval(tok)
    baseline_ppl = compute_perplexity(mdl, tok, encoded)
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    layers = get_compressible_layers(mdl)
    orig_weights = {n: get_weight(mdl, n).detach().clone() for n in layers}

    cos_sims = []
    recon_errors = []

    for name in tqdm(layers, desc="Sequential compress"):
        W_orig = orig_weights[name]

        # Collect fresh inputs from current (partially compressed) model state
        calib = collect_calibration_inputs(mdl, [name], tok, 1024)
        X = calib.get(name)
        if X is None or X.numel() == 0:
            X = torch.randn(64, W_orig.shape[1])

        W_lr, lr_err = weighted_svd_compress(W_orig, X, rank)
        R = W_orig - W_lr
        R_q, _ = quantize_symmetric_per_channel(R, bits=residual_bits, dim=0)
        W_comp = W_lr + R_q

        set_weight(mdl, name, W_comp)

        recon_err = (W_orig - W_comp).pow(2).sum().sqrt() / W_orig.pow(2).sum().sqrt()
        recon_errors.append(float(recon_err))

        Y_orig = X @ W_orig.T
        Y_comp = X @ W_comp.T
        cos = F.cosine_similarity(Y_orig.reshape(-1), Y_comp.reshape(-1), dim=0).item()
        cos_sims.append(cos)

    full_ppl = compute_perplexity(mdl, tok, encoded)
    ratio = full_ppl / baseline_ppl if baseline_ppl > 0 else float("inf")

    print(f"Full model PPL: {full_ppl:.4f} (ratio={ratio:.3f}x)")
    print(f"Avg recon error: {sum(recon_errors)/len(recon_errors):.4f}")
    print(f"Avg cos_sim: {sum(cos_sims)/len(cos_sims):.4f}")

    # Restore
    for name in layers:
        set_weight(mdl, name, orig_weights[name])

    return {
        "method": f"sequential_r{rank}_res{residual_bits}",
        "baseline_ppl": round(baseline_ppl, 4),
        "full_model_ppl": round(full_ppl, 4),
        "ratio": round(ratio, 4),
        "avg_recon_error": round(sum(recon_errors) / len(recon_errors), 4),
        "avg_cos_sim": round(sum(cos_sims) / len(cos_sims), 4),
        "rank": rank,
        "residual_bits": residual_bits,
        "layers_compressed": len(layers),
    }


def run_calibrated_compression(model_path: str, rank: int = 8, residual_bits: int = 4, calib_steps: int = 50) -> dict:
    """Compress all layers with per-layer calibration (joint A,B,R optimization)."""
    print(f"\n--- Calibrated compression (rank={rank}, steps={calib_steps}) ---")

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map="cpu"
    )
    mdl.eval()

    encoded = load_wikitext_eval(tok)
    baseline_ppl = compute_perplexity(mdl, tok, encoded)
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    layers = get_compressible_layers(mdl)
    orig_weights = {n: get_weight(mdl, n).detach().clone() for n in layers}
    calib_inputs = collect_calibration_inputs(mdl, layers, tok, 1024)

    cos_sims = []
    recon_errors = []

    for name in tqdm(layers, desc="Calibrated compress"):
        W_orig = orig_weights[name]
        X = calib_inputs.get(name)
        if X is None or X.numel() == 0:
            X = torch.randn(64, W_orig.shape[1])

        W_comp = calibrate_layer(W_orig, X, rank, residual_bits, steps=calib_steps)

        set_weight(mdl, name, W_comp)

        recon_err = (W_orig - W_comp).pow(2).sum().sqrt() / W_orig.pow(2).sum().sqrt()
        recon_errors.append(float(recon_err))

        Y_orig = X @ W_orig.T
        Y_comp = X @ W_comp.T
        cos = F.cosine_similarity(Y_orig.reshape(-1), Y_comp.reshape(-1), dim=0).item()
        cos_sims.append(cos)

    full_ppl = compute_perplexity(mdl, tok, encoded)
    ratio = full_ppl / baseline_ppl if baseline_ppl > 0 else float("inf")

    print(f"Full model PPL: {full_ppl:.4f} (ratio={ratio:.3f}x)")
    print(f"Avg recon error: {sum(recon_errors)/len(recon_errors):.4f}")
    print(f"Avg cos_sim: {sum(cos_sims)/len(cos_sims):.4f}")

    # Restore
    for name in layers:
        set_weight(mdl, name, orig_weights[name])

    return {
        "method": f"calibrated_r{rank}_res{residual_bits}_s{calib_steps}",
        "baseline_ppl": round(baseline_ppl, 4),
        "full_model_ppl": round(full_ppl, 4),
        "ratio": round(ratio, 4),
        "avg_recon_error": round(sum(recon_errors) / len(recon_errors), 4),
        "avg_cos_sim": round(sum(cos_sims) / len(cos_sims), 4),
        "rank": rank,
        "residual_bits": residual_bits,
        "calib_steps": calib_steps,
        "layers_compressed": len(layers),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT_DIR / "models/smollm-135m"))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--residual_bits", type=int, default=4)
    parser.add_argument("--calib_steps", type=int, default=50)
    args = parser.parse_args()

    tag = Path(args.model).name

    print(f"\n{'=' * 60}")
    print(f"Full calibration: {tag}, rank={args.rank}")
    print(f"{'=' * 60}")

    results = {}

    # Independent (fast, baseline for full-model)
    r1 = run_independent_compression(args.model, args.rank, args.residual_bits)
    results["independent"] = r1

    # Sequential (re-collect inputs per layer — slower but more accurate)
    r2 = run_sequential_compression(args.model, args.rank, args.residual_bits)
    results["sequential"] = r2

    # Calibrated (joint optimization — slowest but best quality)
    r3 = run_calibrated_compression(args.model, args.rank, args.residual_bits, args.calib_steps)
    results["calibrated"] = r3

    out_file = RESULTS / f"full_calib_{tag}_r{args.rank}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_file}")
