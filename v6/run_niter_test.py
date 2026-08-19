#!/usr/bin/env python3
"""
run_niter_test.py - effect of niter (power iterations in svd_lowrank) on variance and quality
================================================================================

Context: the variance diagnostics (run_variance_diag) showed that the pipeline
is deterministic for a fixed seed, and the grid spread (~0.008) comes from
seed randomness. Source identified: torch.svd_lowrank - randomized SVD,
which generates a random projection Omega from the global random state. niter=5 -
too few power iterations, so the result depends on Omega.

Hypothesis: more niter -> less influence of Omega ->
  1. spread between seeds decreases,
  2. mean approaches the true (deterministic) low-rank SVD
     and may IMPROVE (closer to the optimum).

Test: TinyLlama-1.1b (the model with a real win), best config block_damp01
(strict_sequential=False, dampening=0.1), niter in {5, 10, 20} x 3 seeds
(42, 43, 44). Equal BPW 4.2875.

Self-check: here niter=5 must SECONDARILY reproduce the grid numbers
(same seeds, determinism confirmed):
  grid niter=5: ratios [1.058485, 1.057963, 1.059282], mean 1.058577.
If it does not match - something changed in the pipeline and the result is invalid.

Total of 9 runs x ~60-90s = ~10-15 min.

Run by the USER (the agent shares the GPU and risks OOM).
Artifacts: results_v6/niter_test/

Usage:
    python run_niter_test.py
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
OUT_DIR = ROOT / "results_v6" / "niter_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "tinyllama-1.1b"
GPTQ_RATIO = 1.0807
TARGET_BPW = 4.2875
BASE_SEED = 42
N_REPS = 3

# best TinyLlama config from the grid
OVERRIDES = {"strict_sequential": False, "dampening": 0.1}

# (niter, n_reps)
NITER_GRID = [
    (5,  N_REPS),
    (10, N_REPS),
    (20, N_REPS),
]

# self-check: niter=5 from stat_grid (same seeds 42,43,44)
GRID_NITER5 = {
    "ratios": [1.058485, 1.057963, 1.059282],
    "mean": 1.058577,
    "std": 0.000664,
}


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


def run_config(tag, niter, seed):
    """Run one CHMC v6 config at equal BPW with a given niter and seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_path = str(MODELS / MODEL)
    cfg = {**C.default_config(), **OVERRIDES,
           "niter": niter, "bit_budget_bpw": TARGET_BPW}
    result = C.run_chmc_v6(model_path, cfg, tag=tag)
    torch.cuda.empty_cache()
    return result


def main():
    t0 = time.time()
    log_path = _setup_logging()
    summary = {
        "model": MODEL,
        "overrides": OVERRIDES,
        "target_bpw": TARGET_BPW,
        "gptq_ratio": GPTQ_RATIO,
        "seeds": list(range(BASE_SEED, BASE_SEED + N_REPS)),
        "niter_grid": {},
        "errors": {},
    }

    print(f"Log file: {log_path}")
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"CUDA: {torch.cuda.is_available()} | device: {C.DEVICE}")
    print(f"Model: {MODEL} | config: {OVERRIDES} | BPW: {TARGET_BPW}")
    print(f"niter grid: {[n for n, _ in NITER_GRID]} x {N_REPS} reps")

    # ── niter grid ────────────────────────────────────────────────
    for niter, n_reps in NITER_GRID:
        print("\n" + "=" * 64)
        print(f"NITER: {niter}   (reps={n_reps})")
        print("=" * 64)
        ratios = []
        for i in range(n_reps):
            seed = BASE_SEED + i
            run_tag = f"niter{niter}_rep{i}_seed{seed}"
            print(f"\n  --- rep {i} (seed={seed}) ---")
            try:
                r = run_config(run_tag, niter, seed)
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
            summary["niter_grid"][str(niter)] = {
                "n": len(ratios),
                "ratios": [round(x, 6) for x in ratios],
                "mean": round(m, 6),
                "std": round(s, 6),
                "min": round(min(ratios), 6),
                "max": round(max(ratios), 6),
            }
            print(f"\n  niter={niter}: mean={m:.6f}  std={s:.6f}  "
                  f"min={min(ratios):.6f}  max={max(ratios):.6f}  n={len(ratios)}")

    # ── self-check: niter=5 vs stat_grid ──────────────────────────
    print("\n" + "=" * 64)
    print("SELF-CHECK: niter=5 vs stat_grid (same seeds, deterministic)")
    print("=" * 64)
    self_check = {"grid": GRID_NITER5, "reproduced": None, "match": None}
    n5 = summary["niter_grid"].get("5")
    if n5:
        max_diff = max(abs(a - b) for a, b in zip(n5["ratios"], GRID_NITER5["ratios"]))
        match = max_diff < 1e-5
        self_check["reproduced"] = n5["ratios"]
        self_check["max_diff"] = round(max_diff, 8)
        self_check["match"] = bool(match)
        print(f"  grid:     {GRID_NITER5['ratios']}")
        print(f"  now:      {n5['ratios']}")
        print(f"  max_diff: {max_diff:.2e}  -> {'MATCH' if match else 'MISMATCH (pipeline changed!)'}")
    else:
        print("  niter=5 missing — cannot self-check")
    _save("self_check", self_check)
    summary["self_check"] = self_check

    # ── best by mean vs GPTQ ──────────────────────────────────────
    print("\n" + "=" * 64)
    print("BEST (by mean) vs GPTQModel — significance")
    print("=" * 64)
    if summary["niter_grid"]:
        best_key = min(summary["niter_grid"], key=lambda k: summary["niter_grid"][k]["mean"])
        best = summary["niter_grid"][best_key]
        margin = GPTQ_RATIO - best["mean"]
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
            "best_niter": int(best_key),
            "best_mean": best["mean"],
            "best_std": best["std"],
            "best_min": best["min"],
            "best_max": best["max"],
            "gptq_ratio": GPTQ_RATIO,
            "margin": round(margin, 6),
            "margin_in_std": (round(margin_in_std, 3)
                              if margin_in_std != float("inf") else "inf"),
            "beats_gptq": bool(best["mean"] < GPTQ_RATIO),
            "significant": bool(significant),
            "verdict": verdict,
        }
        _save("best_vs_gptq", step_best)
        summary["best_vs_gptq"] = step_best
        print(f"  best_niter={best_key}  mean={best['mean']:.6f}  std={best['std']:.6f}")
        print(f"  GPTQ={GPTQ_RATIO}  -> {verdict}")

        # ── variance trend ─────────────────────────────────────────
        print("\n  Variance trend (std by niter):")
        for k in sorted(summary["niter_grid"], key=int):
            c = summary["niter_grid"][k]
            print(f"    niter={k:>3}: mean={c['mean']:.6f}  std={c['std']:.6f}")
    else:
        summary["errors"]["best"] = "no niter results to compare"

    # ── final summary ─────────────────────────────────────────────
    summary["log_file"] = str(log_path)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary_file = _save("SUMMARY_niter_test", summary)

    print("\n" + "=" * 64)
    print("SUMMARY (by mean ratio, lower is better)")
    print("=" * 64)
    print(f"  {'niter':<8} {'mean':>9} {'std':>9} {'min':>9} {'max':>9} {'n':>3}")
    for k in sorted(summary["niter_grid"], key=int):
        c = summary["niter_grid"][k]
        print(f"  {k:<8} {c['mean']:>9.6f} {c['std']:>9.6f} "
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
