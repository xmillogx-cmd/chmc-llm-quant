#!/usr/bin/env python3
"""
error_accumulation.py — Error accumulation analysis (GAP-4 fix)
===============================================================

Сжимать слои по одному и измерять PPL после каждого.
Показывает, какие слои вносят наибольший вклад в деградацию качества.

Usage:
    python v5/error_accumulation.py --model models/smollm-135m --rank 8
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
    quantize_symmetric_per_channel
)


def measure_error_accumulation(
    model_path: str, rank: int = 8, eval_every: int = 10
) -> dict:
    """
    Compress layers one by one and measure PPL after each batch.

    Returns per-layer contribution to error accumulation, identifying
    the most sensitive layers (largest PPL jump when compressed).
    """
    tag = Path(model_path).name

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
    calib = collect_calibration_inputs(mdl, layers, tok, 1024)

    accumulation = []
    layer_contributions = []  # Per-layer PPL delta

    prev_ppl = baseline_ppl

    for i, name in enumerate(tqdm(layers, desc="Error accumulation")):
        W = orig_weights[name]
        X = calib.get(name)
        if X is None or X.numel() == 0:
            X = torch.randn(64, W.shape[1])

        # Compress this layer
        W_lr, _ = weighted_svd_compress(W, X, rank)
        R = W - W_lr
        R_q, _ = quantize_symmetric_per_channel(R, bits=4, dim=0)
        set_weight(mdl, name, W_lr + R_q)

        # Measure PPL periodically (every N layers or at the end)
        measure = ((i + 1) % eval_every == 0) or (i == len(layers) - 1)

        if measure:
            ppl = compute_perplexity(mdl, tok, encoded)
            delta_vs_prev = ppl - prev_ppl
            ratio = ppl / baseline_ppl if baseline_ppl > 0 else float("inf")

            accumulation.append({
                "layers_compressed": i + 1,
                "layer_name": name,
                "ppl": round(ppl, 4),
                "ratio": round(ratio, 4),
                "delta_vs_prev": round(delta_vs_prev, 4),
            })

            print(f"  {i+1}/{len(layers)} layers: PPL={ppl:.2f} ({ratio:.2f}x, delta={delta_vs_prev:+.2f})")
            prev_ppl = ppl

    # Restore all weights
    for name in layers:
        set_weight(mdl, name, orig_weights[name])

    return {
        "baseline_ppl": round(baseline_ppl, 4),
        "accumulation": accumulation,
        "total_layers": len(layers),
        "rank": rank,
    }


def find_sensitive_layers(
    model_path: str, rank: int = 8, top_n: int = 10
) -> list:
    """Find the top-N layers that cause the largest PPL increase when compressed."""

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map="cpu"
    )
    mdl.eval()

    encoded = load_wikitext_eval(tok)
    baseline_ppl = compute_perplexity(mdl, tok, encoded)

    layers = get_compressible_layers(mdl)
    calib = collect_calibration_inputs(mdl, layers, tok, 512)

    contributions = []

    for name in tqdm(layers, desc="Sensitive layer scan"):
        W_orig = get_weight(mdl, name).detach().clone()
        X = calib.get(name)
        if X is None or X.numel() == 0:
            X = torch.randn(64, W_orig.shape[1])

        # Compress single layer
        W_lr, _ = weighted_svd_compress(W_orig, X, rank)
        R = W_orig - W_lr
        R_q, _ = quantize_symmetric_per_channel(R, bits=4, dim=0)
        set_weight(mdl, name, W_lr + R_q)

        # Measure PPL impact of just this layer
        ppl = compute_perplexity(mdl, tok, encoded)
        delta = ppl - baseline_ppl
        ratio = ppl / baseline_ppl if baseline_ppl > 0 else float("inf")

        contributions.append({
            "layer": name,
            "ppl": round(ppl, 4),
            "delta": round(delta, 4),
            "ratio": round(ratio, 4),
        })

        # Restore immediately
        set_weight(mdl, name, W_orig)

    # Sort by delta (most sensitive first)
    contributions.sort(key=lambda x: x["delta"], reverse=True)
    return contributions[:top_n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT_DIR / "models/smollm-135m"))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=10)
    args = parser.parse_args()

    tag = Path(args.model).name

    print(f"\n{'=' * 60}")
    print(f"Error accumulation: {tag}, rank={args.rank}")
    print(f"{'=' * 60}")

    results = {}

    # Full accumulation curve
    r1 = measure_error_accumulation(args.model, args.rank, args.eval_every)
    results["accumulation"] = r1

    # Sensitive layers (top-10 most impactful)
    print(f"\n--- Finding sensitive layers ---")
    sensitive = find_sensitive_layers(args.model, args.rank)
    results["sensitive_layers_top10"] = sensitive

    for entry in sensitive:
        print(f"  {entry['layer']}: PPL={entry['ppl']:.2f} (delta={entry['delta']:+.2f})")

    out_file = RESULTS / f"error_accum_{tag}_r{args.rank}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_file}")
