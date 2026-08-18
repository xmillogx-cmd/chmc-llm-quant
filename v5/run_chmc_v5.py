#!/usr/bin/env python3
"""
run_chmc_v5.py — CHMC v5 runner (thin CLI over chmc_pipeline)
=============================================================

Stabilizers (see chmc_pipeline.py / stabilizers.py):
  1. Per-group scales (group_size=128)
  2. Damped covariance in weighted SVD
  3. Act-order column permutation
  4. H⁻¹ error compensation (sequential column-wise GPTQ for residual)
  5. Mixed precision (INT8 for sensitive layers, real-input ranking)
  7. Adaptive rank allocation by sensitivity

STE calibration (old stabilizer 6) was removed — iterative optimization
diverged in the null space of calibration data (PPL 1e8+).

Usage:
    python v5/run_chmc_v5.py --model models/smollm-135m
    python v5/run_chmc_v5.py --model models/smollm-135m --compare
"""

import json
import sys
from pathlib import Path

from chmc_pipeline import run_chmc_v5, ROOT_DIR, RESULTS


def run_comparison(model_path: str):
    """Run CHMC v5 with different stabilizer combinations to measure each contribution."""
    tag = Path(model_path).name
    results = {}

    labeled_configs = [
        ("s1-2_groupwise+damp", {"use_compensation": False}),
        ("s1-4_core", {"use_compensation": True}),
        ("full_v5", {"use_compensation": True, "mixed_precision": True, "adaptive_rank": True}),
    ]

    for label, cfg in labeled_configs:
        print(f"\n\n{'#' * 60}")
        print(f"# Running CHMC v5 variant: {label}")
        print(f"{'#' * 60}")

        results[label] = run_chmc_v5(model_path, **cfg)

    # Save comparison
    out_file = RESULTS / f"chmc_v5_comparison_{tag}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {out_file}")

    # Print summary table
    print(f"\n{'=' * 60}")
    print("CHMC v5 Comparison Summary")
    print(f"{'=' * 60}")
    print(f"{'Method':<25} {'PPL':>8} {'Ratio':>8} {'Time(s)':>8}")
    print("-" * 50)
    for label, r in results.items():
        print(f"{label:<25} {r['compressed_ppl']:>8.4f} {r['ratio']:>7.3f}x {r['timing']['total_sec']:>8.1f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CHMC v5 (closed-form pipeline)")
    parser.add_argument("--model", default=str(ROOT_DIR / "models/smollm-135m"))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--residual-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--no-compensation", action="store_true", help="Disable H⁻¹ error compensation")
    parser.add_argument("--mixed-precision", action="store_true", help="INT8 for sensitive layers")
    parser.add_argument("--adaptive-rank", action="store_true", help="Adaptive rank by sensitivity")
    parser.add_argument("--compare", action="store_true", help="Run all variants and compare")
    args = parser.parse_args()

    if args.compare:
        run_comparison(args.model)
    else:
        tag = Path(args.model).name
        try:
            result = run_chmc_v5(
                args.model,
                rank=args.rank,
                residual_bits=args.residual_bits,
                group_size=args.group_size,
                use_compensation=not args.no_compensation,
                mixed_precision=args.mixed_precision,
                adaptive_rank=args.adaptive_rank,
            )
        except Exception as e:
            # Save the failure as a result too (e.g. gemma4 QAT checkpoint
            # is not loadable by the installed transformers)
            result = {
                "model": tag,
                "error": f"{type(e).__name__}: {e}",
                "baseline_ppl": None,
                "compressed_ppl": None,
                "ratio": None,
            }
            print(f"\n[ERROR] CHMC v5 failed for {tag}: {type(e).__name__}: {e}")

        # Save result
        out_file = RESULTS / f"chmc_v5_{tag}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved -> {out_file}")

        if "error" in result:
            sys.exit(1)
