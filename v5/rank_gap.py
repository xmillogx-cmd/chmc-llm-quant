#!/usr/bin/env python3
"""
rank_gap.py — Activation vs Weight rank gap analysis (BUG-5 fix)
================================================================

v4 problem: cov_stats shows eff_rank=1.0 for activations, but SVD of weights
shows min effective_rank=29. Why is there such a discrepancy?

Analysis:
  1. Effective rank of WEIGHTS (SVD of the matrix W itself)
  2. Effective rank of ACTIVATIONS (SVD of input covariance X)
  3. Effective rank of the WEIGHTED matrix (W * sqrt(diag(XX^T)))

If activation_rank << weight_rank, we can use a smaller rank for compression
without quality loss — activations live in a low-dimensional subspace.

Usage:
    python v5/rank_gap.py --model models/smollm-135m
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).parent.resolve()       # v5/
ROOT_DIR = BASE_DIR.parent                        # cmq_experiment/
RESULTS  = ROOT_DIR / "results_v5"
RESULTS.mkdir(parents=True, exist_ok=True)

from eval_utils_v5 import (
    get_compressible_layers, get_weight,
    collect_calibration_inputs
)


def effective_rank_from_svd(singular_values: torch.Tensor) -> float:
    """Compute effective rank = exp(entropy of normalized squared singular values)."""
    sv2 = singular_values ** 2
    total = sv2.sum()
    if total < 1e-12:
        return 1.0
    probs = sv2 / total
    entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum().item()
    return float(torch.exp(torch.tensor(entropy)))


def effective_rank_from_eigenvalues(eigenvalues: torch.Tensor) -> float:
    """Compute effective rank from eigenvalues of covariance matrix."""
    ev = eigenvalues.clamp(min=0)
    total = ev.sum()
    if total < 1e-12:
        return 1.0
    probs = ev / total
    entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum().item()
    return float(torch.exp(torch.tensor(entropy)))


def compare_ranks(model_path: str, n_tokens: int = 2048) -> list:
    """Compare effective rank of weights vs activations for each layer."""

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map="cpu"
    )
    mdl.eval()

    layers = get_compressible_layers(mdl)
    calib = collect_calibration_inputs(mdl, layers, tok, n_tokens)

    comparison = []

    for name in tqdm(layers, desc="Comparing ranks"):
        W = get_weight(mdl, name).float()
        X = calib.get(name)
        if X is None or X.numel() == 0:
            continue

        out_f, in_f = W.shape
        k_svd = min(64, out_f, in_f)

        # 1. Effective rank of WEIGHTS (SVD of W itself)
        _, S_w, _ = torch.svd_lowrank(W, q=k_svd, niter=5)
        eff_rank_weight = effective_rank_from_svd(S_w)

        # 2. Effective rank of ACTIVATIONS (eigenvalues of X^T @ X / N)
        k_eig = min(64, in_f)
        C_diag = (X ** 2).mean(dim=0)  # Diagonal approximation of covariance
        eff_rank_activation = effective_rank_from_eigenvalues(C_diag[-k_eig:])

        # Full covariance for smaller layers
        if in_f <= 512:
            C_full = X.T @ X / X.shape[0]
            eigvals = torch.linalg.eigvalsh(C_full)[-k_eig:]
            eff_rank_activation_full = effective_rank_from_eigenvalues(eigvals.clamp(min=0))
        else:
            eff_rank_activation_full = None

        # 3. Effective rank of WEIGHTED matrix (W * sqrt(diag(XX^T)))
        diag_c = (X ** 2).mean(dim=0).clamp(min=1e-8)
        W_weighted = W * torch.sqrt(diag_c).unsqueeze(0)
        _, S_ww, _ = torch.svd_lowrank(W_weighted, q=k_svd, niter=5)
        eff_rank_weighted = effective_rank_from_svd(S_ww)

        # 4. Recommended rank (min of weight and activation ranks, with floor)
        recommended_rank = max(2, min(int(eff_rank_weight), int(eff_rank_activation)))

        comparison.append({
            "layer": name,
            "shape": f"{out_f}x{in_f}",
            "eff_rank_weight": round(eff_rank_weight, 2),
            "eff_rank_activation_diag": round(eff_rank_activation, 2),
            "eff_rank_activation_full": round(eff_rank_activation_full, 2) if eff_rank_activation_full else None,
            "eff_rank_weighted": round(eff_rank_weighted, 2),
            "gap_weight_vs_activation": round(eff_rank_weight - eff_rank_activation, 2),
            "recommended_rank": recommended_rank,
        })

    # Sort by gap (largest first)
    comparison.sort(key=lambda x: x["gap_weight_vs_activation"], reverse=True)

    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT_DIR / "models/smollm-135m"))
    args = parser.parse_args()

    tag = Path(args.model).name
    print(f"\n{'=' * 60}")
    print(f"Activation vs Weight rank gap: {tag}")
    print(f"{'=' * 60}")

    result = compare_ranks(args.model)

    # Top-10 layers with largest gap
    print("\nTop-10 layers with largest weight-vs-activation gap:")
    for entry in result[:10]:
        full_or_diag = entry.get("eff_rank_activation_full") or entry["eff_rank_activation_diag"]
        print(f"  {entry['layer']}:")
        print(f"    weight={entry['eff_rank_weight']:.1f}, "
              f"activation={full_or_diag:.1f}, "
              f"gap={entry['gap_weight_vs_activation']:.1f}, "
              f"recommended_r={entry['recommended_rank']}")

    # Summary statistics
    gaps = [e["gap_weight_vs_activation"] for e in result]
    print(f"\nGap statistics:")
    print(f"  Mean gap: {sum(gaps)/len(gaps):.2f}")
    print(f"  Max gap:  {max(gaps):.2f}")
    print(f"  Min gap:  {min(gaps):.2f}")

    # Layers where activation rank < weight rank (compression opportunity)
    opportunities = [e for e in result if e["gap_weight_vs_activation"] > 5]
    print(f"\nLayers with gap > 5 (compression opportunity): {len(opportunities)}")
    for e in opportunities[:5]:
        print(f"  {e['layer']}: weight_rank={e['eff_rank_weight']:.1f} -> "
              f"use rank={e['recommended_rank']}")

    out_file = RESULTS / f"rank_gap_{tag}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_file}")
