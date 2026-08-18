#!/usr/bin/env python3
"""
run_scale_test.py — CHMC v6: scale hypothesis (Qwen2.5-3B + Qwen3-4B)
======================================================================

Context: v6 showed model-dependence — TinyLlama-1.1B win 33σ, Qwen-0.5B
loss −2.8σ, SmolLM-135M neutral. Size is NOT monotonic at 0.5-1.1B.
Scale hypothesis: bigger model = more redundancy = bigger CHMC win.
Testing with Qwen2.5-3B (single GPU) + Qwen3-4B (multi-GPU).

Setup:
  - Qwen2.5-3B: FP32, single GPU (12 GB fits 16 GB).
  - Qwen3-4B:   FP32, multi-GPU via device_map="auto" (16 GB > 16 GB single).
  - GPTQ ref:   GPTQModel 4-bit, group-128, desc_act=False, sym=True.
                4B uses device_map (RISK — untested); falls back to BF16.
  - CHMC grid:  5 configs × 3 reps (seeds 42,43,44), equal BPW 4.2875.

Comparison:
  - Per model: CHMC best vs GPTQ (margin, significance in σ).
  - Cross-model: 0.5B (loss) → 3B (?) → 4B (?) — does the win grow with size?

The user runs this (agent shares GPU and risks OOM).
Artifacts: results_v6/scale_test/

Usage:
    python run_scale_test.py
    python run_scale_test.py --model qwen2.5-3b
    python run_scale_test.py --skip-gptq   # reuse cached GPTQ ref
"""

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
V6_DIR = BASE_DIR.parent / "v6"
sys.path.insert(0, str(V6_DIR))

import chmc_v6 as C                      # noqa: E402

ROOT = C.ROOT_DIR
MODELS = ROOT / "models"
OUT_DIR = ROOT / "results_v6" / "scale_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BPW = 4.2875
BASE_SEED = 42
N_REPS = 3

# ── models ──────────────────────────────────────────────────────────
# device_map: None = single GPU (.to(DEVICE)); "auto" = multi-GPU.
# Qwen2.5-3B: 3B × 4 bytes = 12 GB → fits single 16 GB GPU; "auto" lets
#             the runtime split across both cards if a single card is tight.
# Qwen3-4B:   4B × 4 bytes = 16 GB → does NOT fit single 16 GB GPU.
MODELS_CFG = {
    "qwen2.5-3b": {
        "device_map": "auto",
        "gptq_ratio": None,   # computed at runtime
    },
    "qwen3-4b": {
        "device_map": "auto",
        "gptq_ratio": None,   # computed at runtime
    },
}

# ── CHMC grid (same as run_stat_grid.py) ────────────────────────────
# (tag, overrides, n_reps)
GRID = [
    ("block_damp005",  {"strict_sequential": False, "dampening": 0.05}, 3),
    ("block_damp01",   {"strict_sequential": False, "dampening": 0.1},  3),
    ("strict_damp004", {"strict_sequential": True,  "dampening": 0.04}, 3),
    ("strict_damp006", {"strict_sequential": True,  "dampening": 0.06}, 3),
    ("strict_damp005", {"strict_sequential": True,  "dampening": 0.05}, 3),
]

# ── prior results (for cross-model comparison) ──────────────────────
PRIOR = {
    "qwen2.5-0.5b": {"chmc_best": 1.1787, "gptq": 1.1407, "margin": -0.038,
                     "margin_in_std": -2.8, "verdict": "REAL LOSS"},
    "tinyllama-1.1b": {"chmc_best": 1.0586, "gptq": 1.0807, "margin": +0.0221,
                       "margin_in_std": 33.0, "verdict": "REAL WIN"},
    "smollm-135m": {"chmc_best": 1.1793, "gptq": 1.1761, "margin": -0.003,
                    "margin_in_std": -0.34, "verdict": "NEUTRAL"},
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


def _save(tag, obj, out_dir=None):
    d = out_dir or OUT_DIR
    out_file = d / f"{tag}.json"
    with open(out_file, "w") as f:
        json.dump(obj, f, indent=2)
    return out_file


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


# ──────────────────────────────────────────────────────────────────────
# GPTQ reference
# ──────────────────────────────────────────────────────────────────────
def compute_gptq_ref(model_name, device_map=None):
    """Compute GPTQ reference (4-bit, group-128) for a model.

    Returns dict with ppl, baseline_ppl, ratio, method, device_map.
    Falls back to BF16 if device_map fails (OOM or API error).
    """
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer
    from eval_utils_v6 import (
        compute_perplexity, load_wikitext_eval, load_calib_text,
    )

    model_path = str(MODELS / model_name)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Calibration data — TRAIN split (never the test/eval split)
    calib_text = load_calib_text(tok, n_tokens=4096)
    calib_enc_ids = tok(calib_text, return_tensors="pt",
                        truncation=True, max_length=4096)["input_ids"][0]
    calib_enc = [{"input_ids": calib_enc_ids.unsqueeze(0)}]

    quant_config = QuantizeConfig(bits=4, group_size=128, desc_act=False, sym=True)
    encoded = load_wikitext_eval(tok)

    # Try the requested device_map first; fall back to BF16 single-GPU.
    attempts = []
    if device_map:
        attempts.append(("fp32_" + str(device_map), device_map, torch.float32))
    attempts.append(("fp32_single", None, torch.float32))
    if device_map:
        # BF16 fallback (half the VRAM, fits single GPU)
        attempts.append(("bf16_single", None, torch.bfloat16))

    last_err = None
    for label, dm, dtype in attempts:
        try:
            print(f"  [GPTQ ref] attempt: {label} (dtype={dtype})")
            if dm:
                model = GPTQModel.from_pretrained(
                    model_path, quantize_config=quant_config, device_map=dm)
            else:
                model = GPTQModel.from_pretrained(
                    model_path, quantize_config=quant_config)

            t0 = time.time()
            model.quantize(calib_enc)
            quant_time = time.time() - t0
            print(f"  [GPTQ ref] quantized in {quant_time:.1f}s")

            import tempfile
            tmpdir = tempfile.mkdtemp(dir=str(OUT_DIR))
            model.save_quantized(tmpdir)
            print(f"  [GPTQ ref] saved to {tmpdir}")

            # Free the GPTQ model before loading the next one (avoid OOM).
            del model
            gc.collect()
            torch.cuda.empty_cache()

            if dm:
                model_q = GPTQModel.from_quantized(tmpdir, device_map=dm)
            else:
                dev = "cuda:0" if torch.cuda.is_available() else "cpu"
                model_q = GPTQModel.from_quantized(tmpdir, device=dev)

            ppl = compute_perplexity(model_q, tok, encoded)
            del model_q
            gc.collect()
            torch.cuda.empty_cache()

            # baseline PPL (FP32 or BF16, same dtype as the quantized load)
            from transformers import AutoModelForCausalLM
            if dm:
                mdl = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=dtype, device_map=dm)
            else:
                mdl = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=dtype)
                mdl = mdl.to("cuda" if torch.cuda.is_available() else "cpu")
            mdl.eval()
            baseline_ppl = compute_perplexity(mdl, tok, encoded)
            del mdl
            gc.collect()
            torch.cuda.empty_cache()

            ratio = ppl / baseline_ppl if baseline_ppl > 0 else float("inf")
            print(f"  [GPTQ ref] baseline={baseline_ppl:.4f}  "
                  f"quantized={ppl:.4f}  ratio={ratio:.6f}")
            return {
                "ppl": round(ppl, 4),
                "baseline_ppl": round(baseline_ppl, 4),
                "ratio": round(ratio, 6),
                "method": f"gptqmodel_4bit_{label}",
                "device_map": dm,
                "dtype": str(dtype),
                "quant_time_sec": round(quant_time, 2),
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"  [GPTQ ref] {label} FAILED: {last_err}")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            continue

    raise RuntimeError(f"GPTQ ref failed for {model_name}: {last_err}")


# ──────────────────────────────────────────────────────────────────────
# CHMC grid
# ──────────────────────────────────────────────────────────────────────
def run_chmc_config(model_name, overrides, seed, device_map=None):
    """Run one CHMC v6 config at equal BPW with a given seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_path = str(MODELS / model_name)
    cfg = {**C.default_config(), **overrides,
           "bit_budget_bpw": TARGET_BPW, "device_map": device_map}
    result = C.run_chmc_v6(model_path, cfg,
                           tag=f"{model_name}_{overrides}_{seed}")
    # run_chmc_v6 already frees the model (del + gc.collect + empty_cache);
    # this is a safety net in case a reference cycle survived.
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_chmc_grid(model_name, device_map=None, out_dir=None):
    """Run the full CHMC grid for one model. Returns summary dict."""
    d = out_dir or (OUT_DIR / model_name)
    d.mkdir(parents=True, exist_ok=True)

    configs = {}
    errors = {}

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
                r = run_chmc_config(model_name, overrides, seed, device_map)
                _save(run_tag, r, d)
                ratios.append(r["ratio"])
                print(f"  {run_tag}: ratio={r['ratio']:.6f}  "
                      f"ppl={r['compressed_ppl']:.4f}  bpw={r['bpw']}")
            except Exception as e:
                errors[run_tag] = {
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
                traceback.print_exc()

        if ratios:
            m = _mean(ratios)
            s = _std(ratios)
            configs[tag] = {
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

    return {"configs": configs, "errors": errors}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CHMC v6 scale hypothesis test")
    parser.add_argument("--model", default=None,
                        help="run a single model (default: all)")
    parser.add_argument("--skip-gptq", action="store_true",
                        help="skip GPTQ ref computation (reuse cached)")
    args = parser.parse_args()

    t0 = time.time()
    log_path = _setup_logging()
    summary = {
        "target_bpw": TARGET_BPW,
        "base_seed": BASE_SEED,
        "n_reps": N_REPS,
        "models": {},
    }

    print(f"Log file: {log_path}")
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
    print(f"Target BPW: {TARGET_BPW} | reps: {N_REPS}")
    print(f"Models: {list(MODELS_CFG)}")

    models_to_run = [args.model] if args.model else list(MODELS_CFG)

    for model_name in models_to_run:
        if model_name not in MODELS_CFG:
            print(f"\n[WARN] Unknown model '{model_name}', skipping")
            continue
        spec = MODELS_CFG[model_name]
        device_map = spec["device_map"]
        model_dir = OUT_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 64)
        print(f"MODEL: {model_name}  (device_map={device_map})")
        print("=" * 64)

        model_summary = {
            "device_map": device_map,
            "gptq": None,
            "chmc": None,
            "best_vs_gptq": None,
        }

        # ── GPTQ reference ────────────────────────────────────────
        gptq_file = model_dir / "gptq_ref.json"
        if args.skip_gptq and gptq_file.exists():
            gptq = json.loads(gptq_file.read_text())
            print(f"  [GPTQ ref] loaded from cache: ratio={gptq['ratio']}")
        else:
            print("\n  --- GPTQ reference ---")
            try:
                gptq = compute_gptq_ref(model_name, device_map)
                _save("gptq_ref", gptq, model_dir)
            except Exception as e:
                gptq = {"error": f"{type(e).__name__}: {e}"}
                _save("gptq_ref", gptq, model_dir)
                traceback.print_exc()
        model_summary["gptq"] = gptq

        # ── CHMC grid ─────────────────────────────────────────────
        print("\n  --- CHMC grid ---")
        chmc = run_chmc_grid(model_name, device_map, model_dir)
        model_summary["chmc"] = chmc

        # ── best vs GPTQ ──────────────────────────────────────────
        configs = chmc.get("configs", {})
        if configs and "ratio" in gptq:
            best_tag = min(configs, key=lambda t: configs[t]["mean"])
            best = configs[best_tag]
            gptq_ratio = gptq["ratio"]
            margin = gptq_ratio - best["mean"]
            if best["std"] > 0:
                margin_in_std = margin / best["std"]
                significant = margin_in_std > 2.0
                verdict = (f"margin {margin:+.6f} = {margin_in_std:.1f}σ -> "
                           f"{'REAL WIN' if significant else 'WITHIN NOISE'}")
            else:
                margin_in_std = float("inf")
                significant = margin > 0
                verdict = (f"margin {margin:+.6f}, std=0 -> "
                           f"{'REPRODUCIBLE WIN' if significant else 'LOSES'}")
            step_best = {
                "best_tag": best_tag,
                "best_mean": best["mean"],
                "best_std": best["std"],
                "gptq_ratio": gptq_ratio,
                "margin": round(margin, 6),
                "margin_in_std": (round(margin_in_std, 3)
                                  if margin_in_std != float("inf") else "inf"),
                "beats_gptq": bool(best["mean"] < gptq_ratio),
                "significant": bool(significant),
                "verdict": verdict,
            }
            _save("best_vs_gptq", step_best, model_dir)
            model_summary["best_vs_gptq"] = step_best
            print(f"\n  best={best_tag}  mean={best['mean']:.6f}  "
                  f"std={best['std']:.6f}")
            print(f"  GPTQ={gptq_ratio}  -> {verdict}")
        else:
            print("\n  [WARN] Cannot compare (no CHMC configs or GPTQ ref missing)")

        summary["models"][model_name] = model_summary

    # ── cross-model comparison ─────────────────────────────────────
    print("\n" + "=" * 64)
    print("CROSS-MODEL: scale hypothesis (bigger = bigger win?)")
    print("=" * 64)
    cross = {}
    # prior results
    for m, p in PRIOR.items():
        cross[m] = {**p, "source": "prior"}
    # new results
    for m, ms in summary["models"].items():
        bv = ms.get("best_vs_gptq")
        if bv:
            cross[m] = {
                "chmc_best": bv["best_mean"],
                "gptq": bv["gptq_ratio"],
                "margin": bv["margin"],
                "margin_in_std": bv["margin_in_std"],
                "verdict": bv["verdict"],
                "source": "this_run",
            }
    # print sorted by margin
    print(f"  {'model':<20} {'CHMC':>9} {'GPTQ':>9} {'margin':>10} "
          f"{'σ':>8}  verdict")
    for m in sorted(cross, key=lambda x: cross[x].get("margin", 0)):
        c = cross[m]
        mi = c.get("margin_in_std", float("nan"))
        mi_s = f"{mi:.1f}" if mi != float("nan") else "n/a"
        print(f"  {m:<20} {c.get('chmc_best', float('nan')):>9.6f} "
              f"{c.get('gptq', float('nan')):>9.6f} "
              f"{c.get('margin', float('nan')):>+10.6f} "
              f"{mi_s:>8}  {c.get('verdict', 'n/a')}")
    _save("cross_model", cross)
    summary["cross_model"] = cross

    # ── final summary ──────────────────────────────────────────────
    summary["log_file"] = str(log_path)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary_file = _save("SUMMARY_scale_test", summary)

    print(f"\n  Total time: {summary['elapsed_sec']}s")
    print(f"  Summary saved: {summary_file}")
    print(f"  Log file:     {log_path}")
    return summary


if __name__ == "__main__":
    main()
