#!/usr/bin/env python3
"""
stabilize_baseline.py — Deterministic baseline PPL stability check (BUG-1 fix)
=============================================================================

Run the same model multiple times on identical WikiText tokens and verify:
  CV < 0.02  (less than 2% variation between runs)
  SmolLM PPL in [9, 25]
  Qwen   PPL in [6, 20]

Usage:
    python v5/stabilize_baseline.py [--model models/smollm-135m models/qwen2.5-0.5b]
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).parent.resolve()       # v5/
ROOT_DIR = BASE_DIR.parent                        # cmq_experiment/
RESULTS  = ROOT_DIR / "results_v5"
RESULTS.mkdir(parents=True, exist_ok=True)

# Import unified eval utilities
from eval_utils_v5 import compute_perplexity, load_wikitext_eval


def run_stability_check(model_path: str, n_runs: int = 5) -> dict:
    """Run PPL evaluation n_runs times on identical tokens and report stability."""
    print(f"\nLoading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading model (float32, CPU)...")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map="cpu"
    )
    mdl.eval()

    # Load ONCE — same tokens for every run
    encoded = load_wikitext_eval(tok)
    print(f"Evaluation tokens: {len(encoded)}")

    results = []
    for i in range(n_runs):
        ppl = compute_perplexity(mdl, tok, encoded=encoded)
        results.append(ppl)
        print(f"  Run {i + 1}/{n_runs}: PPL = {ppl:.4f}")

    import numpy as np
    mean_ppl = float(np.mean(results))
    std_ppl  = float(np.std(results))
    cv       = std_ppl / mean_ppl if mean_ppl > 0 else 999.0

    report = {
        "model": model_path,
        "runs": [round(r, 4) for r in results],
        "mean": round(mean_ppl, 4),
        "std": round(std_ppl, 4),
        "cv": round(cv, 6),
        "stable": cv < 0.02,
        "n_tokens": len(encoded),
    }

    print(f"\n  Mean={mean_ppl:.4f}, Std={std_ppl:.4f}, CV={cv:.4%}")
    print(f"  Stable: {'✅' if report['stable'] else '❌'}")
    return report


def check_reasonable_range(report: dict, tag: str) -> bool:
    """Verify PPL is in the expected range for this model."""
    mean = report["mean"]
    ranges = {
        "smollm": (9.0, 25.0),
        "qwen":   (6.0, 20.0),
    }
    low, high = ranges.get(tag.lower(), (1.0, 100.0))
    ok = low <= mean <= high
    if not ok:
        print(f"  ⚠️  PPL {mean:.2f} outside expected [{low}, {high}] for {tag}")
        print(f"     Possible causes: tokenizer mismatch, chat template, pad masking")
    return ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=[
        str(ROOT_DIR / "models/smollm-135m"),
        str(ROOT_DIR / "models/qwen2.5-0.5b"),
    ])
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    all_reports = {}
    for model_path in args.model:
        tag = Path(model_path).name
        print(f"\n{'=' * 60}")
        print(f"Baseline stability check: {tag}")
        print(f"{'=' * 60}")

        report = run_stability_check(model_path, n_runs=args.runs)
        all_reports[tag] = report

        # Save
        out_file = RESULTS / f"baseline_stability_{tag}.json"
        with open(out_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Saved -> {out_file}")

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for tag, r in all_reports.items():
        status = "✅ STABLE" if r["stable"] else "❌ UNSTABLE"
        range_ok = check_reasonable_range(r, tag)
        range_status = "✅ RANGE OK" if range_ok else "⚠️  OUT OF RANGE"
        print(f"  {tag:20s}  mean={r['mean']:.4f}  CV={r['cv']:.4%}  {status}  {range_status}")
