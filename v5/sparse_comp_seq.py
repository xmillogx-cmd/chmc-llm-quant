#!/usr/bin/env python3
"""
sparse_comp_seq.py — Sparse sequential compensation (BUG-4 fix)
================================================================

Проблема v4: Per-layer sparse улучшает cos_sim, но full-model PPL = 2972x.
Ошибки накапливаются через слои.

Решение: После sparse выбора для каждого слоя переоптимизировать low-rank факторы
чтобы компенсировать потерянный остаток. Делать последовательно (sequential),
собирая входы из уже сжатой модели.

Usage:
    python v5/sparse_comp_seq.py --model models/smollm-135m --density 0.25
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


def sparse_with_compensation(
    W_orig: torch.Tensor, X: torch.Tensor, rank: int,
    density: float = 0.10, steps: int = 50, lr: float = 1e-3
) -> torch.Tensor:
    """
    Sparse compensation with low-rank re-optimization.

    1. Weighted SVD → W_lr, R_full = W - W_lr
    2. Hessian-aware top-k from R_full → R_sparse (keep only `density` fraction)
    3. Quantize sparse residual
    4. Re-optimize A,B so that A@B.T ≈ (W - R_sparse)

    This gives the low-rank factors room to compensate for what sparse
    residual couldn't capture, reducing error accumulation across layers.
    """
    W = W_orig.float()
    out_f, in_f = W.shape
    rank = min(rank, out_f, in_f)

    # Step 1: Initial low-rank via weighted SVD
    W_lr_init, _ = weighted_svd_compress(W, X, rank)
    R_full = W - W_lr_init

    # Step 2: Hessian-aware sparse selection
    diag_c = (X ** 2).mean(dim=0).clamp(min=1e-8)  # [in_f]
    importance = R_full.pow(2) * diag_c.unsqueeze(0)  # [out_f, in_f]

    k = max(1, int(density * importance.numel()))
    threshold = torch.topk(importance.flatten(), k, largest=True).values[-1]
    mask = importance >= threshold
    R_sparse = R_full * mask.float()

    # Step 3: Quantize sparse non-zeros
    nz_vals = R_sparse[mask.bool()]
    if nz_vals.numel() > 0:
        scale = nz_vals.abs().max().clamp(min=1e-8) / (2 ** 3 - 1)  # INT4 symmetric
        quantized = torch.round(nz_vals / scale).clamp(-7, 7) * scale
        R_sparse_zeroed = torch.zeros_like(R_sparse)
        R_sparse_zeroed[mask.bool()] = quantized
    else:
        R_sparse_zeroed = R_sparse

    # Step 4: Re-optimize A,B for W_target = W - R_sparse_quantized
    W_target = W - R_sparse_zeroed.float()

    U, S, V = torch.svd_lowrank(W_lr_init, q=rank, niter=3)
    A = torch.nn.Parameter((U * S.unsqueeze(0)).clone())
    B = torch.nn.Parameter(V.clone())

    optimizer = torch.optim.Adam([A, B], lr=lr)

    for step in range(steps):
        W_lr = A @ B.T
        Y_pred = X @ W_lr.T
        Y_target = X @ W_target.T
        loss = F.mse_loss(Y_pred, Y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final reconstruction: optimized low-rank + sparse residual
    with torch.no_grad():
        W_final = A.detach() @ B.detach().T + R_sparse_zeroed.float()

    return W_final


def run_sparse_sequential(
    model_path: str, rank: int = 8, density: float = 0.10, steps: int = 50
) -> dict:
    """Compress all layers with sparse compensation sequentially."""
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

    cos_sims = []
    recon_errors = []

    for name in tqdm(layers, desc=f"Sparse seq d={density:.2f}"):
        W_orig = orig_weights[name]

        # Collect fresh inputs from current model state
        calib = collect_calibration_inputs(mdl, [name], tok, 1024)
        X = calib.get(name)
        if X is None or X.numel() == 0:
            X = torch.randn(64, W_orig.shape[1])

        # Compress with sparse compensation
        W_comp = sparse_with_compensation(W_orig, X, rank, density, steps)
        set_weight(mdl, name, W_comp)

        # Track metrics
        recon_err = (W_orig - W_comp).pow(2).sum().sqrt() / W_orig.pow(2).sum().sqrt()
        recon_errors.append(float(recon_err))

        Y_orig = X @ W_orig.T
        Y_comp = X @ W_comp.T
        cos = F.cosine_similarity(Y_orig.reshape(-1), Y_comp.reshape(-1), dim=0).item()
        cos_sims.append(cos)

    full_ppl = compute_perplexity(mdl, tok, encoded)
    ratio = full_ppl / baseline_ppl if baseline_ppl > 0 else float("inf")

    print(f"Sparse seq: PPL={full_ppl:.2f}, ratio={ratio:.1f}x, avg_cos={sum(cos_sims)/len(cos_sims):.4f}")
    print(f"Avg recon error: {sum(recon_errors)/len(recon_errors):.4f}")

    # Restore
    for name in layers:
        set_weight(mdl, name, orig_weights[name])

    return {
        "baseline_ppl": round(baseline_ppl, 4),
        "full_ppl": round(full_ppl, 4),
        "ratio": round(ratio, 2),
        "avg_cos_sim": round(sum(cos_sims) / len(cos_sims), 4),
        "avg_recon_error": round(sum(recon_errors) / len(recon_errors), 4),
        "density": density,
        "rank": rank,
        "calib_steps": steps,
        "layers": len(layers),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT_DIR / "models/smollm-135m"))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--density", nargs="+", type=float, default=[0.10, 0.25])
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    tag = Path(args.model).name

    all_results = {}
    for density in args.density:
        print(f"\n{'=' * 60}")
        print(f"Sparse sequential: {tag}, d={density:.2f}, rank={args.rank}")
        print(f"{'=' * 60}")

        result = run_sparse_sequential(args.model, args.rank, density, args.steps)
        all_results[f"d{int(density*100)}"] = result

    out_file = RESULTS / f"sparse_seq_{tag}_r{args.rank}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_file}")
