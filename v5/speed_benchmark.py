#!/usr/bin/env python3
"""
speed_benchmark.py — Inference speed benchmark (GAP-2 fix)
==========================================================

Измерить latency генерации для:
  1. Baseline (FP32)
  2. Scalar INT4 quantization
  3. CHMC low-rank + residual

Usage:
    python v5/speed_benchmark.py --model models/smollm-135m --rank 4
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).parent.resolve()       # v5/
ROOT_DIR = BASE_DIR.parent                        # cmq_experiment/
RESULTS  = ROOT_DIR / "results_v5"
RESULTS.mkdir(parents=True, exist_ok=True)

from eval_utils_v5 import (
    get_compressible_layers, get_weight, set_weight,
    weighted_svd_compress, collect_calibration_inputs,
    quantize_symmetric_per_channel
)


def measure_latency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    n_tokens: int = 128,
    n_runs: int = 5,
    prompt_text: str = "The meaning of life is",
) -> dict:
    """Measure generation latency (time to generate n_tokens tokens)."""
    prompt = tokenizer(prompt_text, return_tensors="pt")

    # Warmup runs
    with torch.no_grad():
        for _ in range(2):
            model.generate(**prompt, max_new_tokens=5, do_sample=False)

    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**prompt, max_new_tokens=n_tokens, do_sample=False)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        print(f"    Run {i+1}: {elapsed:.3f}s ({n_tokens/elapsed:.1f} tok/s)")

    avg_time = sum(times) / len(times)
    tokens_per_sec = n_tokens / avg_time
    return {
        "avg_time_sec": round(avg_time, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "times": [round(t, 3) for t in times],
    }


def run_speed_benchmark(model_path: str, rank: int = 4) -> dict:
    """Compare baseline vs scalar_q4 vs CHMC speed."""
    tag = Path(model_path).name

    print(f"\nLoading model...")
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map="cpu"
    )
    mdl.eval()

    layers = get_compressible_layers(mdl)
    orig_weights = {n: get_weight(mdl, n).detach().clone() for n in layers}

    results = {}

    # 1. Baseline speed (FP32)
    print("\n--- Baseline (FP32) ---")
    r_base = measure_latency(mdl, tok)
    results["baseline"] = r_base
    print(f"  -> {r_base['tokens_per_sec']:.1f} tok/s")

    # 2. Scalar INT4 speed
    print("\n--- Scalar INT4 ---")
    for name in layers:
        W = orig_weights[name]
        W_q, _ = quantize_symmetric_per_channel(W, bits=4, dim=0)
        set_weight(mdl, name, W_q)

    r_q4 = measure_latency(mdl, tok)
    results["scalar_q4"] = {
        **r_q4,
        "speedup_vs_baseline": round(r_q4["tokens_per_sec"] / r_base["tokens_per_sec"], 2),
    }
    print(f"  -> {r_q4['tokens_per_sec']:.1f} tok/s ({results['scalar_q4']['speedup_vs_baseline']}x)")

    # Restore for next test
    for name in layers:
        set_weight(mdl, name, orig_weights[name])

    # 3. CHMC low-rank speed
    print(f"\n--- CHMC rank={rank} ---")
    calib = collect_calibration_inputs(mdl, layers, tok, 512)
    for name in layers:
        W = orig_weights[name]
        X = calib.get(name)
        if X is None or X.numel() == 0:
            X = torch.randn(64, W.shape[1])
        W_lr, _ = weighted_svd_compress(W, X, rank)
        R = W - W_lr
        R_q, _ = quantize_symmetric_per_channel(R, bits=4, dim=0)
        set_weight(mdl, name, W_lr + R_q)

    r_chmc = measure_latency(mdl, tok)
    results[f"chmc_rank{rank}"] = {
        **r_chmc,
        "speedup_vs_baseline": round(r_chmc["tokens_per_sec"] / r_base["tokens_per_sec"], 2),
        "speedup_vs_scalar_q4": round(r_chmc["tokens_per_sec"] / r_q4["tokens_per_sec"], 2),
    }
    print(f"  -> {r_chmc['tokens_per_sec']:.1f} tok/s ({results[f'chmc_rank{rank}']['speedup_vs_baseline']}x)")

    # Restore
    for name in layers:
        set_weight(mdl, name, orig_weights[name])

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT_DIR / "models/smollm-135m"))
    parser.add_argument("--rank", type=int, default=4)
    args = parser.parse_args()

    tag = Path(args.model).name
    print(f"\n{'=' * 60}")
    print(f"Speed benchmark: {tag}, rank={args.rank}")
    print(f"{'=' * 60}")

    results = run_speed_benchmark(args.model, args.rank)

    out_file = RESULTS / f"speed_{tag}_r{args.rank}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_file}")
