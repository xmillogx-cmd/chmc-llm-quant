#!/usr/bin/env python3
"""
run_benchmark.py — Unified benchmark: CHMC v5 (full_v5) vs GPTQModel
=====================================================================

Usage:
    python v5/run_benchmark.py                          # default 3 models
    python v5/run_benchmark.py --models smollm-135m qwen2.5-0.5b
    python v5/run_benchmark.py --model ../models/smollm-135m   # single model

Pipeline (identical for all models):
  1. CHMC v5 full_v5: rank=8, INT4/INT8 mixed, compensation=True, adaptive_rank
     (chmc_pipeline.run_chmc_v5)
  2. GPTQ baseline: GPTQModel library (manual GPTQ was removed — broken)

Metrics per model:
  - Baseline PPL (FP32)
  - CHMC v5: compressed PPL, ratio, compress_time
  - GPTQ: compressed PPL, ratio, quant_time
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Fix Windows encoding
if sys.platform == "win32" and not getattr(sys.stdout, 'encoding', '').lower().startswith('utf'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch

BASE_DIR = Path(__file__).parent.resolve()  # v5/
ROOT_DIR = BASE_DIR.parent                   # cmq_experiment/
RESULTS  = ROOT_DIR / "results_v5"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from chmc_pipeline import run_chmc_v5
from run_gptq_baseline import try_gptqmodel


# ── Default models ───────────────────────────────────────────────

DEFAULT_MODELS = [
    str(ROOT_DIR / "models/smollm-135m"),
    str(ROOT_DIR / "models/qwen2.5-0.5b"),
    str(ROOT_DIR / "models/tinyllama-1.1b"),
]


# ── GPTQ baseline (library) ──────────────────────────────────────

def run_gptq(model_path: str, baseline_ppl: float) -> Dict[str, Any]:
    """GPTQModel library baseline (no manual fallback)."""
    tag = Path(model_path).name
    print(f"\n{'=' * 60}")
    print(f"GPTQ BASELINE (GPTQModel): {tag}")
    print(f"{'=' * 60}")

    res = try_gptqmodel(model_path, bits=4)
    if res is None:
        print("  GPTQModel unavailable — skipping GPTQ baseline")
        return {"gptq_error": "GPTQModel unavailable"}

    ratio = res["ppl"] / baseline_ppl if baseline_ppl > 0 else float("inf")
    print(f"  Result: PPL={res['ppl']:.4f} (ratio={ratio:.3f}x)")

    return {
        "gptq_ppl": res["ppl"],
        "gptq_ratio": round(ratio, 4),
        "quant_sec": res["quant_time_sec"],
    }


# ── Summary table ────────────────────────────────────────────────

def print_summary(all_results: Dict[str, Any]):
    """Print comparison table."""
    print(f"\n\n{'=' * 90}")
    print("BENCHMARK SUMMARY — CHMC v5 (full_v5) vs GPTQModel")
    print(f"{'=' * 90}")

    header = f"{'Model':<20} {'Baseline':>10} {'CHMC PPL':>10} {'CHMC Ratio':>10} {'GPTQ PPL':>10} {'GPTQ Ratio':>10}"
    print(header)
    print("-" * 90)

    for tag, res in all_results.items():
        baseline = res.get("baseline_ppl", 0)
        chmc_ppl = res.get("chmc_v5_ppl", 0)
        chmc_ratio = res.get("chmc_v5_ratio", 0)
        gptq_ppl = res.get("gptq_ppl", 0)
        gptq_ratio = res.get("gptq_ratio", 0)

        print(f"{tag:<20} {baseline:>10.4f} {chmc_ppl:>10.4f} {chmc_ratio:>9.3f}x {gptq_ppl:>10.4f} {gptq_ratio:>9.3f}x")

    print(f"{'=' * 90}")


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CHMC v5 vs GPTQModel benchmark on N models")
    parser.add_argument("--models", nargs="+", default=None, help="Model paths (default: smollm-135m, qwen2.5-0.5b, gemma-4-e2b)")
    args = parser.parse_args()

    if args.models:
        models = args.models
    else:
        # Filter to existing models only
        models = [p for p in DEFAULT_MODELS if Path(p).exists()]
        if not models:
            print("No models found. Download at least one model first.")
            sys.exit(1)

    print(f"\nModels to benchmark: {[Path(m).name for m in models]}")
    all_results = {}

    for i, model_path in enumerate(models):
        tag = Path(model_path).name
        print(f"\n\n{'#'*60}")
        print(f"# [{i+1}/{len(models)}] Model: {tag}")
        print(f"{'#'*60}")

        # Clean GPU before each model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        res = {}

        # 1. CHMC v5 full pipeline (shared chmc_pipeline)
        chmc_res = run_chmc_v5(model_path, mixed_precision=True, adaptive_rank=True)
        res.update(chmc_res)

        # Clean between pipelines
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # 2. GPTQ baseline (GPTQModel library, fresh model load inside)
        gptq_res = run_gptq(model_path, chmc_res["baseline_ppl"])
        res.update(gptq_res)

        all_results[tag] = res

    # Save results
    out_file = RESULTS / "benchmark_chmc_vs_gptq.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved -> {out_file}")

    # Print summary
    print_summary(all_results)
