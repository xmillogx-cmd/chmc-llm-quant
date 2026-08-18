#!/usr/bin/env python3
"""generate_cov_stats.py — Generate covariance statistics for budget allocator testing."""

import json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).parent.resolve()       # v5/
ROOT_DIR = BASE_DIR.parent                        # cmq_experiment/
RESULTS  = ROOT_DIR / "results_v5"
RESULTS.mkdir(parents=True, exist_ok=True)

from eval_utils_v5 import (
    get_compressible_layers, collect_calibration_inputs
)


def compute_cov_stats(model_path: str, n_tokens: int = 2048) -> dict:
    """Compute covariance statistics for all compressible layers."""
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map="cpu"
    )
    mdl.eval()

    layers = get_compressible_layers(mdl)
    calib = collect_calibration_inputs(mdl, layers, tok, n_tokens)

    stats = {}
    for name in tqdm(layers, desc="Computing cov stats"):
        W = mdl
        for p in name.split("."):
            W = getattr(W, p, None)
        if W is None or not hasattr(W, "weight"):
            continue

        weight = W.weight.float()
        X = calib.get(name)
        if X is None or X.numel() == 0:
            continue

        out_f, in_f = weight.shape
        n_weights = out_f * in_f

        # Input covariance diagonal
        diag_c = (X ** 2).mean(dim=0)

        # Weighted SVD for spectrum estimation
        W_weighted = weight * torch.sqrt(diag_c.clamp(min=1e-8)).unsqueeze(0)
        q = min(64, in_f, out_f)
        U, S, V = torch.svd_lowrank(W_weighted, q=q, niter=5)

        total_energy = (S ** 2).sum().item()
        if total_energy < 1e-12:
            continue

        cum_energy = torch.cumsum(S ** 2, dim=0) / total_energy

        d90 = min(in_f, out_f)
        d95 = min(in_f, out_f)
        d99 = min(in_f, out_f)
        for idx in range(len(cum_energy)):
            if cum_energy[idx] >= 0.90 and d90 == min(in_f, out_f):
                d90 = idx + 1
            if cum_energy[idx] >= 0.95 and d95 == min(in_f, out_f):
                d95 = idx + 1
            if cum_energy[idx] >= 0.99 and d99 == min(in_f, out_f):
                d99 = idx + 1

        # Effective rank
        probs = (S ** 2) / total_energy
        entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum().item()
        import math
        effective_rank = math.exp(entropy)

        top16_e = cum_energy[min(15, len(cum_energy)-1)].item() if len(cum_energy) >= 16 else cum_energy[-1].item()
        top32_e = cum_energy[min(31, len(cum_energy)-1)].item() if len(cum_energy) >= 32 else cum_energy[-1].item()
        top64_e = cum_energy[min(min(64, len(S))-1, len(cum_energy)-1)].item()

        stats[name] = {
            "out_features": out_f,
            "in_features": in_f,
            "n_weights": n_weights,
            "d90": d90,
            "d95": d95,
            "d99": d99,
            "effective_rank": round(effective_rank, 2),
            "top16_energy": round(top16_e, 4),
            "top32_energy": round(top32_e, 4),
            "top64_energy": round(top64_e, 4),
        }

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT_DIR / "models/smollm-135m"))
    args = parser.parse_args()

    tag = Path(args.model).name
    print(f"Computing cov stats for {tag}...")
    stats = compute_cov_stats(args.model)

    out_file = RESULTS / f"cov_stats_{tag}.json"
    with open(out_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved -> {out_file}")
    print(f"Layers: {len(stats)}")
