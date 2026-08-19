#!/usr/bin/env python3
"""
run_variance_diag.py — diagnosing the SOURCE of run-to-run variance
================================================================================

The grid (run_stat_grid.py) showed real variance (std ~0.008) between reps with
DIFFERENT seeds (42,43,44,45). Two candidates for the source:
  A. CUDA non-determinism (cuBLAS/atomics) — variance even with a SINGLE seed.
  B. Hidden seed-dependent randomness — variance only between different seeds.

This runner disambiguates: we run strict_damp005 (strict, λ=0.05) with ONE seed
(42) three times in a row.
  - results identical (range < 1e-6) -> source B: CUDA is deterministic,
    the grid variance came from DIFFERENT seeds (hidden seed randomness exists).
  - results differ (range >= 1e-6)   -> source A: CUDA non-determinism present,
    the variance is fundamental, seed does not control it.

3 runs x ~60s = ~3 min.

Run by the USER (the agent shares the GPU and risks OOM).
Artifacts: results_v6/variance_diag/

Usage:
    python run_variance_diag.py
"""

import json
import sys
import time
import traceback
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import chmc_v6 as C                      # noqa: E402

ROOT = C.ROOT_DIR
MODELS = ROOT / "models"
OUT_DIR = ROOT / "results_v6" / "variance_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BPW = 4.2875
MODEL = "smollm-135m"
SEED = 42
N_RUNS = 3
OVERRIDES = {"strict_sequential": True, "dampening": 0.05}
IDENTICAL_EPS = 1e-6


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
    log_dir = OUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{ts}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_path


def _save(tag, obj):
    out_file = OUT_DIR / f"{tag}.json"
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


def run_once(run_tag, seed):
    """Run strict_damp005 once with a given seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_path = str(MODELS / MODEL)
    cfg = {**C.default_config(), **OVERRIDES, "bit_budget_bpw": TARGET_BPW}
    result = C.run_chmc_v6(model_path, cfg, tag=run_tag)
    torch.cuda.empty_cache()
    return result


def main():
    t0 = time.time()
    log_path = _setup_logging()
    summary = {
        "model": MODEL,
        "seed": SEED,
        "overrides": OVERRIDES,
        "n_runs": N_RUNS,
        "identical_eps": IDENTICAL_EPS,
        "ratios": [],
        "errors": {},
    }

    print(f"Log file: {log_path}")
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"CUDA: {torch.cuda.is_available()} | device: {C.DEVICE}")
    print(f"Variance diagnostic: {MODEL}, seed={SEED} x {N_RUNS} runs")
    print(f"Config: {OVERRIDES}")

    for i in range(N_RUNS):
        run_tag = f"strict_damp005_seed{SEED}_run{i}"
        print(f"\n  --- run {i} (seed={SEED}) ---")
        try:
            r = run_once(run_tag, SEED)
            _save(run_tag, r)
            summary["ratios"].append(r["ratio"])
            print(f"  {run_tag}: ratio={r['ratio']:.6f}  "
                  f"ppl={r['compressed_ppl']:.4f}  bpw={r['bpw']}")
        except Exception as e:
            summary["errors"][run_tag] = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
            traceback.print_exc()

    ratios = summary["ratios"]
    if len(ratios) >= 2:
        mn = min(ratios)
        mx = max(ratios)
        rng = mx - mn
        s = _std(ratios)
        identical = rng < IDENTICAL_EPS
        if identical:
            verdict = ("IDENTICAL (range < 1e-6) -> CUDA deterministic for this "
                       "seed; the grid variance came from DIFFERENT seeds "
                       "(hidden seed-dependent randomness).")
        else:
            verdict = (f"DIFFERENT (range={rng:.6f}, std={s:.6f}) -> "
                       "CUDA non-determinism present; variance is inherent, "
                       "seed does not control it.")
        summary["diagnosis"] = {
            "min": round(mn, 6),
            "max": round(mx, 6),
            "range": round(rng, 6),
            "std": round(s, 6),
            "identical": bool(identical),
            "verdict": verdict,
        }
        print("\n" + "=" * 64)
        print("DIAGNOSIS")
        print("=" * 64)
        print(f"  ratios: {[round(x, 6) for x in ratios]}")
        print(f"  min={mn:.6f}  max={mx:.6f}  range={rng:.6f}  std={s:.6f}")
        print(f"  -> {verdict}")
    else:
        summary["errors"]["diagnosis"] = "not enough successful runs"

    summary["log_file"] = str(log_path)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary_file = _save("SUMMARY_variance_diag", summary)

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
