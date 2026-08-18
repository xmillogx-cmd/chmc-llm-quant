#!/usr/bin/env python3
"""
run_turbo_weights.py — runs ALL TurboQuant patch steps in order and saves results.
===============================================================================

This is the single runner the user launches (NOT the agent — the agent shares
the GPU and would risk OOM / resource contention). It:

  1. Smoke test (synthetic) — sanity check of all 4 patches
  2. Patch 4 (IP metric)      on SmolLM   [eval-only, baseline compression]
  3. Patch 3 (Lloyd-Max)      on SmolLM   [hadamard + lloyd_max]
  4. Patch 5 (Cone-Aware)     on SmolLM   [cone_aware + hadamard]
  5. Patch 2 (QJL)            on SmolLM   [qjl, rank reinvested]
  6. Combo 3+5                on SmolLM   [hadamard + lloyd_max + cone_aware]
  7. Combo 3+5+2              on SmolLM   [hadamard + lloyd_max + cone_aware + qjl]
  8. Best vs GPTQModel        on SmolLM   [equal BPW 4.2875]
  9. Best config              on Qwen-0.5B + TinyLlama-1.1B

Every config is run at EQUAL BPW (target 4.2875 = GPTQModel 4-bit/group-128).
Each step is wrapped in try/except so a single failure (e.g. OOM) does not stop
the whole run. All artifacts land in results_v6/patch_turbo_weights/.

Usage:
    python run_turbo_weights.py
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
import smoke_test_turbo_weights as ST    # noqa: E402

ROOT = C.ROOT_DIR
MODELS = ROOT / "models"
OUT_DIR = ROOT / "results_v6" / "patch_turbo_weights"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GPTQModel reference (4-bit, group-128) from results_v5/gptq_awq_baselines.json
GPTQ_REF = {
    "smollm-135m":  {"ppl": 25.0595, "ratio": 1.1761},
    "qwen2.5-0.5b": {"ppl": 17.0373, "ratio": 1.1407},
    "tinyllama-1.1b": {"ppl": 10.185, "ratio": 1.0807},
}
TARGET_BPW = 4.2875


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
    """Tee stdout+stderr to a timestamped log file under the results dir."""
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


def run_config(tag, model_name, overrides):
    """Run one CHMC v6 config at equal BPW and save its result JSON."""
    # Seed so randomized svd_lowrank starts are reproducible across configs
    # (otherwise identical compression settings give slightly different PPL).
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model_path = str(MODELS / model_name)
    cfg = {**C.default_config(), **overrides, "bit_budget_bpw": TARGET_BPW}
    result = C.run_chmc_v6(model_path, cfg, tag=tag)
    out_file = _save(tag, result)
    print(f"  -> saved {out_file}")
    torch.cuda.empty_cache()
    return result


def main():
    t0 = time.time()
    log_path = _setup_logging()
    summary = {"target_bpw": TARGET_BPW, "steps": {}, "errors": {}}

    print(f"Log file: {log_path}")
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"CUDA: {torch.cuda.is_available()} | device: {C.DEVICE}")
    print(f"Target BPW: {TARGET_BPW}")

    # ── STEP 1: smoke test (synthetic, no model load) ─────────────
    print("\n" + "=" * 64)
    print("STEP 1: Smoke test (synthetic) — sanity check of all 4 patches")
    print("=" * 64)
    try:
        smoke = ST.main()
        summary["steps"]["step1_smoke"] = {"all_pass": smoke["all_pass"]}
    except Exception as e:
        summary["errors"]["step1_smoke"] = {
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
        traceback.print_exc()

    # ── STEPS 2-7: single patches + combos on SmolLM ──────────────
    # (tag, overrides, is_numbered_step)
    smollm_configs = [
        ("ref_baseline",          {},                                              False),
        ("ref_baseline_hadamard", {"hadamard": True},                             False),
        ("step2_patch4_ip",       {"ip_metric": True},                            True),
        ("step3_patch3_lloydmax", {"hadamard": True, "lloyd_max": True},          True),
        ("step4_patch5_cone",     {"cone_aware": True, "hadamard": True},         True),
        ("step5_patch2_qjl",      {"qjl": True},                                  True),
        ("step6_combo_3_5",       {"hadamard": True, "lloyd_max": True,
                                   "cone_aware": True},                           True),
        ("step7_combo_3_5_2",     {"hadamard": True, "lloyd_max": True,
                                   "cone_aware": True, "qjl": True},              True),
    ]
    smollm_results = {}
    for tag, overrides, is_step in smollm_configs:
        print("\n" + "=" * 64)
        print(f"{'STEP' if is_step else 'REF'}: {tag}   (overrides={overrides})")
        print("=" * 64)
        try:
            r = run_config(tag, "smollm-135m", overrides)
            smollm_results[tag] = r
            summary["steps"][tag] = {
                "ratio": r["ratio"], "bpw": r["bpw"],
                "baseline_ppl": r["baseline_ppl"],
                "compressed_ppl": r["compressed_ppl"],
            }
        except Exception as e:
            summary["errors"][tag] = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
            traceback.print_exc()

    # ── STEP 8: best vs GPTQModel at equal BPW (SmolLM) ───────────
    print("\n" + "=" * 64)
    print("STEP 8: Best config vs GPTQModel (equal BPW, SmolLM)")
    print("=" * 64)
    best_tag, best = None, None
    if smollm_results:
        best_tag = min(smollm_results, key=lambda t: smollm_results[t]["ratio"])
        best = smollm_results[best_tag]
        gptq = GPTQ_REF["smollm-135m"]
        step8 = {
            "best_tag": best_tag,
            "best_ratio": best["ratio"],
            "best_bpw": best["bpw"],
            "gptq_ratio": gptq["ratio"],
            "gptq_ppl": gptq["ppl"],
            "beats_gptq": bool(best["ratio"] < gptq["ratio"]),
            "margin": round(gptq["ratio"] - best["ratio"], 6),
        }
        _save("step8_best_vs_gptq", step8)
        summary["steps"]["step8_best_vs_gptq"] = step8
        print(f"  best={best_tag}  ratio={best['ratio']:.6f}  vs GPTQ {gptq['ratio']}  "
              f"-> {'WINS' if step8['beats_gptq'] else 'loses'} by {step8['margin']:.6f}")
    else:
        summary["errors"]["step8"] = "no SmolLM results to compare"

    # ── STEP 9: best config on Qwen-0.5B + TinyLlama-1.1B ─────────
    print("\n" + "=" * 64)
    print("STEP 9: Best config on Qwen-0.5B + TinyLlama-1.1B")
    print("=" * 64)
    if best is not None:
        best_overrides = {k: v for k, v in best["config"].items()
                          if k in C.default_config() and v is not None
                          and k not in ("bit_budget_bpw", "rank")}
        # reconstruct the active patch flags from the best config
        best_overrides = {
            "hadamard": bool(best["config"].get("hadamard")),
            "lloyd_max": bool(best["config"].get("lloyd_max")),
            "cone_aware": bool(best["config"].get("cone_aware")),
            "qjl": bool(best["config"].get("qjl")),
            "qjl_n_projections": best["config"].get("qjl_n_projections", 0),
            "ip_metric": bool(best["config"].get("ip_metric")),
        }
        for model_name in ["qwen2.5-0.5b", "tinyllama-1.1b"]:
            tag = f"step9_best_{model_name}"
            print(f"\n  -> {model_name}")
            try:
                r = run_config(tag, model_name, best_overrides)
                gptq = GPTQ_REF[model_name]
                step9 = {
                    "model": model_name, "best_tag": best_tag,
                    "ratio": r["ratio"], "bpw": r["bpw"],
                    "gptq_ratio": gptq["ratio"],
                    "beats_gptq": bool(r["ratio"] < gptq["ratio"]),
                }
                _save(tag, {**r, "gptq_ratio": gptq["ratio"],
                            "beats_gptq": step9["beats_gptq"]})
                summary["steps"][tag] = step9
                print(f"     ratio={r['ratio']:.6f}  vs GPTQ {gptq['ratio']}  "
                      f"-> {'WINS' if step9['beats_gptq'] else 'loses'}")
            except Exception as e:
                summary["errors"][tag] = {
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
                traceback.print_exc()
    else:
        summary["errors"]["step9"] = "no best config to propagate"

    # ── final summary ─────────────────────────────────────────────
    summary["log_file"] = str(log_path)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary_file = _save("SUMMARY_turbo_weights", summary)

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    for tag, s in summary["steps"].items():
        if "ratio" in s:
            print(f"  {tag:<28} ratio={s['ratio']:.6f}  bpw={s.get('bpw')}")
        else:
            print(f"  {tag:<28} {s}")
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
