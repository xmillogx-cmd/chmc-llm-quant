#!/usr/bin/env python3
"""
run_stat_grid.py — CHMC v6 statistical grid (3-4 reps per config)
==================================================================

Problem: the strict λ=0.05 win on SmolLM (1.1723 vs GPTQ 1.1761, +0.0038)
is at noise level. One run = one point. What is needed:
  1. A grid of configs around the peak (block/strict × λ) — find the TRUE best.
  2. 3-4 reps of each with DIFFERENT seeds (42,43,44[,45]) — measure the noise.
  3. Verdict: does the best config beat GPTQ by more than its own run-to-run
     spread (margin > 2σ => real win, otherwise within noise).

Why different seeds per rep: calibration is deterministic (fixed wikitext,
`cat[:n_tokens]` without randperm), but seed-dependent randomness and CUDA
non-determinism (cuBLAS/atomics) produce a real run-to-run spread. Using one
seed for all reps would give identical numbers and measure nothing.

Grid (SmolLM, equal BPW 4.2875):
  block_damp005   block,  λ=0.05, 3 reps  (missed cell: block > strict at λ=0.01)
  block_damp01    block,  λ=0.1,  3 reps  (missed cell)
  strict_damp004  strict, λ=0.04, 3 reps  (fine sweep around the peak)
  strict_damp006  strict, λ=0.06, 3 reps  (fine sweep around the peak)
  strict_damp005  strict, λ=0.05, 4 reps  (reference, variance)

Total: 16 runs × ~60 s ≈ 16 min on a GPU.

Run by the USER (an automated agent shares the GPU and risks OOM).
Artifacts: results_v6/stat_grid/<model>/

Usage:
    python run_stat_grid.py [--model smollm-135m]

Environment:
    CMQ_MODELS_DIR  model directory override (default: <repo_root>/models)
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import chmc_v6 as C                      # noqa: E402

ROOT = C.ROOT_DIR
# Model directory can be relocated via CMQ_MODELS_DIR (default: <root>/models).
MODELS = Path(os.environ.get("CMQ_MODELS_DIR", str(ROOT / "models")))
OUT_DIR = ROOT / "results_v6" / "stat_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GPTQModel reference (4-bit, group-128)
GPTQ_REF = {
    "smollm-135m":    {"ppl": 25.0595, "ratio": 1.1761},
    "qwen2.5-0.5b":   {"ppl": 17.0373, "ratio": 1.1407},
    "tinyllama-1.1b": {"ppl": 10.185,  "ratio": 1.0807},
}
TARGET_BPW = 4.2875
DEFAULT_MODEL = "smollm-135m"
BASE_SEED = 42
# Per-model output dir, set in main() (results_v6/stat_grid/<model>/).
ACTIVE_OUT_DIR = OUT_DIR

# (tag, overrides, n_reps)
GRID = [
    ("block_damp005",  {"strict_sequential": False, "dampening": 0.05}, 3),
    ("block_damp01",   {"strict_sequential": False, "dampening": 0.1},  3),
    ("strict_damp004", {"strict_sequential": True,  "dampening": 0.04}, 3),
    ("strict_damp006", {"strict_sequential": True,  "dampening": 0.06}, 3),
    ("strict_damp005", {"strict_sequential": True,  "dampening": 0.05}, 4),
]


class _Tee:
    """Duplicate a stream to the original destination and a log file."""

    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, s):
        try:
            self.original.write(s)
        except UnicodeEncodeError:
            enc = getattr(self.original, "encoding", None) or "utf-8"
            self.original.write(s.encode(enc, "replace").decode(enc, "replace"))
        self.log_file.write(s)
        self.log_file.flush()
        return len(s)

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def isatty(self):
        return False


def _setup_logging():
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_dir = ACTIVE_OUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{ts}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_path


def _save(tag, obj):
    out_file = ACTIVE_OUT_DIR / f"{tag}.json"
    with open(out_file, "w") as f:
        json.dump(obj, f, indent=2)
    return out_file


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    """Sample std (n-1 denominator). 0 if n<2."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


def run_config(tag, model_name, overrides, seed):
    """Run one CHMC v6 config at equal BPW with a given seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_path = str(MODELS / model_name)
    cfg = {**C.default_config(), **overrides, "bit_budget_bpw": TARGET_BPW}
    result = C.run_chmc_v6(model_path, cfg, tag=tag)
    torch.cuda.empty_cache()
    return result


def main():
    global ACTIVE_OUT_DIR
    parser = argparse.ArgumentParser(description="CHMC v6 statistical grid")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="model dir name (default: %(default)s)")
    args = parser.parse_args()
    model = args.model
    ACTIVE_OUT_DIR = OUT_DIR / model
    ACTIVE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log_path = _setup_logging()
    summary = {
        "target_bpw": TARGET_BPW,
        "model": model,
        "base_seed": BASE_SEED,
        "configs": {},
        "errors": {},
    }

    print(f"Log file: {log_path}")
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"CUDA: {torch.cuda.is_available()} | device: {C.DEVICE}")
    print(f"Target BPW: {TARGET_BPW} | model: {model}")
    print(f"Stat grid: {len(GRID)} configs, "
          f"{sum(n for _, _, n in GRID)} runs, seeds {BASE_SEED}+")

    # ── grid ──────────────────────────────────────────────────────
    for tag, overrides, n_reps in GRID:
        print("\n" + "=" * 64)
        print(f"CONFIG: {tag}   (overrides={overrides}, reps={n_reps})")
        print("=" * 64)
        ratios = []
        for i in range(n_reps):
            seed = BASE_SEED + i
            run_tag = f"{tag}_rep{i}_seed{seed}"
            print(f"\n  --- rep {i} (seed={seed}) ---")
            try:
                r = run_config(run_tag, model, overrides, seed)
                _save(run_tag, r)
                ratios.append(r["ratio"])
                print(f"  {run_tag}: ratio={r['ratio']:.6f}  "
                      f"ppl={r['compressed_ppl']:.4f}  bpw={r['bpw']}")
            except Exception as e:
                summary["errors"][run_tag] = {
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
                traceback.print_exc()

        if ratios:
            m = _mean(ratios)
            s = _std(ratios)
            summary["configs"][tag] = {
                "overrides": overrides,
                "n": len(ratios),
                "ratios": [round(x, 6) for x in ratios],
                "mean": round(m, 6),
                "std": round(s, 6),
                "min": round(min(ratios), 6),
                "max": round(max(ratios), 6),
            }
            print(f"\n  {tag}: mean={m:.6f}  std={s:.6f}  "
                  f"min={min(ratios):.6f}  max={max(ratios):.6f}  n={len(ratios)}")

    # ── best by mean vs GPTQ ──────────────────────────────────────
    print("\n" + "=" * 64)
    print("BEST (by mean) vs GPTQModel — significance")
    print("=" * 64)
    if summary["configs"]:
        best_tag = min(summary["configs"], key=lambda t: summary["configs"][t]["mean"])
        best = summary["configs"][best_tag]
        gptq = GPTQ_REF[model]
        margin = gptq["ratio"] - best["mean"]
        if best["std"] > 0:
            margin_in_std = margin / best["std"]
            significant = margin_in_std > 2.0
            verdict = (f"margin {margin:+.6f} = {margin_in_std:.1f}σ -> "
                       f"{'REAL WIN' if significant else 'WITHIN NOISE'}")
        else:
            margin_in_std = float("inf")
            significant = margin > 0
            verdict = (f"margin {margin:+.6f}, std=0 (deterministic) -> "
                       f"{'REPRODUCIBLE WIN' if significant else 'LOSES'}")
        step_best = {
            "best_tag": best_tag,
            "best_mean": best["mean"],
            "best_std": best["std"],
            "best_min": best["min"],
            "best_max": best["max"],
            "gptq_ratio": gptq["ratio"],
            "margin": round(margin, 6),
            "margin_in_std": (round(margin_in_std, 3)
                              if margin_in_std != float("inf") else "inf"),
            "beats_gptq": bool(best["mean"] < gptq["ratio"]),
            "significant": bool(significant),
            "verdict": verdict,
        }
        _save("best_vs_gptq", step_best)
        summary["best_vs_gptq"] = step_best
        print(f"  best={best_tag}  mean={best['mean']:.6f}  std={best['std']:.6f}")
        print(f"  GPTQ={gptq['ratio']}  -> {verdict}")
    else:
        summary["errors"]["best"] = "no grid results to compare"

    # ── final summary ─────────────────────────────────────────────
    summary["log_file"] = str(log_path)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary_file = _save("SUMMARY_stat_grid", summary)

    print("\n" + "=" * 64)
    print("SUMMARY (by mean ratio, lower is better)")
    print("=" * 64)
    print(f"  {'config':<18} {'mean':>9} {'std':>9} {'min':>9} {'max':>9} {'n':>3}")
    for tag in sorted(summary["configs"], key=lambda t: summary["configs"][t]["mean"]):
        c = summary["configs"][tag]
        print(f"  {tag:<18} {c['mean']:>9.6f} {c['std']:>9.6f} "
              f"{c['min']:>9.6f} {c['max']:>9.6f} {c['n']:>3}")
    if summary["errors"]:
        print("\n  ERRORS (full tracebacks in SUMMARY json + log file):")
        for tag, err in summary["errors"].items():
            if isinstance(err, dict):
                print(f"    {tag}: {err['error']}")
            else:
                print(f"    {tag}: {err}")
    print(f"\n  Total time: {summary['elapsed_sec']}s")
    print(f"  Summary saved: {summary_file}")
    print(f"  Log file:     {log_path}")
    return summary


if __name__ == "__main__":
    main()
