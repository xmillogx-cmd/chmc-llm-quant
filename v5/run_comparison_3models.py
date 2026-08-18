#!/usr/bin/env python3
"""
run_comparison_3models.py — CHMC v5 (full_v5) vs GPTQModel on 3 models
=======================================================================

Models:
  1. smollm-135m       (~135M params)
  2. qwen2.5-0.5b      (~500M params)
  3. gemma-4-e2b       (~270M params)

Pipeline (identical for all models, shared chmc_pipeline):
  CHMC v5 full_v5: rank=8, residual_bits=4, group_size=128,
                   compensation=True, mixed_precision=True, adaptive_rank=True
  GPTQ baseline:    GPTQModel library (manual GPTQ was removed — broken)

Metrics per model:
  - Baseline PPL (FP32)
  - CHMC v5 compressed PPL + ratio + compress_time
  - GPTQ compressed PPL + ratio + quant_time
  - GPU memory usage during compression

Usage:
    python v5/run_comparison_3models.py
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

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

# Models to test — adjust paths as needed
MODELS = [
    {"path": str(ROOT_DIR / "models/smollm-135m"),       "name": "smollm-135m"},
    {"path": str(ROOT_DIR / "models/qwen2.5-0.5b"),      "name": "qwen2.5-0.5b"},
    {"path": str(ROOT_DIR / "models/tinyllama-1.1b"),    "name": "tinyllama-1.1b"},
]


def get_gpu_mem_mb() -> Optional[float]:
    """Get current GPU memory usage in MB."""
    if DEVICE == "cuda":
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return None


def run_gptq_baseline(model_path: str, model_name: str, baseline_ppl: float) -> Dict[str, Any]:
    """Run GPTQModel baseline on a single model (library only)."""
    print(f"\n{'=' * 60}")
    print(f"GPTQ BASELINE (GPTQModel): {model_name}")
    print(f"{'=' * 60}")

    res = try_gptqmodel(model_path, bits=4)
    if res is None:
        print("  GPTQModel unavailable — skipping GPTQ baseline")
        return {"gptq_error": "GPTQModel unavailable"}

    ratio = res["ppl"] / baseline_ppl if baseline_ppl > 0 else float("inf")
    print(f"  GPTQModel: PPL={res['ppl']:.4f} (ratio={ratio:.3f}x)")

    return {
        "gptq_ppl": res["ppl"],
        "gptq_ratio": round(ratio, 4),
        "quant_sec": res["quant_time_sec"],
    }


def print_summary_table(all_results: Dict[str, Any]):
    """Print comparison table."""
    print(f"\n\n{'=' * 80}")
    print("COMPARISON SUMMARY — CHMC v5 (full_v5) vs GPTQModel")
    print(f"{'=' * 80}")

    header = f"{'Model':<20} {'Baseline':>10} {'CHMC PPL':>10} {'CHMC Ratio':>10} {'GPTQ PPL':>10} {'GPTQ Ratio':>10}"
    print(header)
    print("-" * 80)

    for model_name, res in all_results.items():
        baseline = res.get("baseline_ppl", 0)
        chmc_ppl = res.get("compressed_ppl", 0)
        chmc_ratio = res.get("ratio", 0)
        gptq_ppl = res.get("gptq_ppl", 0)
        gptq_ratio = res.get("gptq_ratio", 0)

        print(f"{model_name:<20} {baseline:>10.4f} {chmc_ppl:>10.4f} {chmc_ratio:>9.3f}x {gptq_ppl:>10.4f} {gptq_ratio:>9.3f}x")

    print(f"{'=' * 80}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CHMC v5 vs GPTQModel on 3 models")
    parser.add_argument("--models", nargs="+", default=None, help="Override model paths")
    args = parser.parse_args()

    if args.models:
        MODELS = [{"path": p, "name": Path(p).name} for p in args.models]

    all_results = {}

    # Check which models exist
    available_models = []
    for m in MODELS:
        if Path(m["path"]).exists():
            available_models.append(m)
        else:
            print(f"⚠ Model not found, skipping: {m['name']} ({m['path']})")

    if not available_models:
        print("No models available. Download at least one model first.")
        sys.exit(1)

    # Run CHMC v5 + GPTQ for each available model
    for m in available_models:
        print(f"\n\n{'#'*60}")
        print(f"# Model: {m['name']}")
        print(f"{'#'*60}")

        res = {}

        # CHMC v5 full pipeline (shared chmc_pipeline)
        chmc_res = run_chmc_v5(
            m["path"], mixed_precision=True, adaptive_rank=True
        )
        res.update(chmc_res)

        # GPTQ baseline (GPTQModel library, fresh model load inside)
        gptq_res = run_gptq_baseline(m["path"], m["name"], chmc_res["baseline_ppl"])
        res.update(gptq_res)

        all_results[m["name"]] = res

    # Save combined results
    out_file = RESULTS / "comparison_3models.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved -> {out_file}")

    # Print summary table
    print_summary_table(all_results)
