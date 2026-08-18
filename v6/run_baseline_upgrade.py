#!/usr/bin/env python3
"""
run_baseline_upgrade.py — апгрейд plain CHMC v6 baseline (без TurboQuant-патчей)
================================================================================

Цель: закрыть разрыв до GPTQModel (SmolLM 1.2146 -> < 1.1761) улучшением САМОГО
baseline, а не патчами. Два рычага:

  A1  strict_sequential  — настоящий GPTQ (колонка-за-колонкой, полный H^-1,
                           error-feedback во ВСЕ оставшиеся колонки).
                           Сейчас baseline использует block-компенсацию (BLOCK=32)
                           — приближение: внутри блока колонки не компенсируют
                           друг друга. Strict точнее.
  A4  dampening grid     — подбор λ (демпфирование H = X^T X/n + λI) под данные.
                           Сейчас λ=0.01 фиксирован.

Все конфиги — при РАВНОМ BPW (target 4.2875 = GPTQModel 4-bit/group-128).
QJL НЕ используется (фундаментально не подходит для весов — см. REPORT).

Порядок:
  1. ref_baseline          (block,  λ=0.01)  — референс (должен дать ~1.2146)
  2. strict_gptq           (strict, λ=0.01)  — A1
  3. strict_gptq_damp0001  (strict, λ=0.001) — A4
  4. strict_gptq_damp005   (strict, λ=0.05)  — A4
  5. strict_gptq_damp01    (strict, λ=0.1)   — A4
  6. Лучший vs GPTQ (SmolLM)
  7. Лучший на Qwen-0.5B + TinyLlama-1.1B

Запускает ПОЛЬЗОВАТЕЛЬ (агент разделяет GPU и рискует OOM).
Артефакты: results_v6/baseline_upgrade/

Usage:
    python run_baseline_upgrade.py
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
OUT_DIR = ROOT / "results_v6" / "baseline_upgrade"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GPTQModel reference (4-bit, group-128)
GPTQ_REF = {
    "smollm-135m":   {"ppl": 25.0595, "ratio": 1.1761},
    "qwen2.5-0.5b":  {"ppl": 17.0373, "ratio": 1.1407},
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
    print(f"Baseline upgrade: A1 strict_sequential + A4 dampening grid")

    # ── SmolLM configs ────────────────────────────────────────────
    # (tag, overrides)
    smollm_configs = [
        ("ref_baseline",          {"strict_sequential": False, "dampening": 0.01}),
        ("strict_gptq",           {"strict_sequential": True,  "dampening": 0.01}),
        ("strict_gptq_damp0001",  {"strict_sequential": True,  "dampening": 0.001}),
        ("strict_gptq_damp005",   {"strict_sequential": True,  "dampening": 0.05}),
        ("strict_gptq_damp01",    {"strict_sequential": True,  "dampening": 0.1}),
        # group_dim=1 (GPTQ orientation: groups along input cols)
        ("strict_gptq_gd1",       {"strict_sequential": True,  "dampening": 0.01,  "group_dim": 1}),
        ("strict_gptq_gd1_d0001", {"strict_sequential": True,  "dampening": 0.001, "group_dim": 1}),
    ]
    smollm_results = {}
    for tag, overrides in smollm_configs:
        print("\n" + "=" * 64)
        print(f"CONFIG: {tag}   (overrides={overrides})")
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

    # ── Best vs GPTQ (SmolLM) ─────────────────────────────────────
    print("\n" + "=" * 64)
    print("BEST vs GPTQModel (equal BPW, SmolLM)")
    print("=" * 64)
    best_tag, best = None, None
    if smollm_results:
        best_tag = min(smollm_results, key=lambda t: smollm_results[t]["ratio"])
        best = smollm_results[best_tag]
        gptq = GPTQ_REF["smollm-135m"]
        step_best = {
            "best_tag": best_tag,
            "best_ratio": best["ratio"],
            "best_bpw": best["bpw"],
            "gptq_ratio": gptq["ratio"],
            "gptq_ppl": gptq["ppl"],
            "beats_gptq": bool(best["ratio"] < gptq["ratio"]),
            "margin": round(gptq["ratio"] - best["ratio"], 6),
        }
        _save("best_vs_gptq", step_best)
        summary["steps"]["best_vs_gptq"] = step_best
        print(f"  best={best_tag}  ratio={best['ratio']:.6f}  vs GPTQ {gptq['ratio']}  "
              f"-> {'WINS' if step_best['beats_gptq'] else 'loses'} by {step_best['margin']:.6f}")
    else:
        summary["errors"]["best"] = "no SmolLM results to compare"

    # ── Best on Qwen-0.5B + TinyLlama-1.1B ────────────────────────
    print("\n" + "=" * 64)
    print("BEST on Qwen-0.5B + TinyLlama-1.1B")
    print("=" * 64)
    if best is not None:
        # reconstruct the active flags from the best config
        best_overrides = {
            "strict_sequential": bool(best["config"].get("strict_sequential")),
            "dampening": best["config"].get("dampening", 0.01),
            "group_dim": best["config"].get("group_dim", 0),
        }
        for model_name in ["qwen2.5-0.5b", "tinyllama-1.1b"]:
            tag = f"best_{model_name}"
            print(f"\n  -> {model_name}")
            try:
                r = run_config(tag, model_name, best_overrides)
                gptq = GPTQ_REF[model_name]
                step = {
                    "model": model_name, "best_tag": best_tag,
                    "ratio": r["ratio"], "bpw": r["bpw"],
                    "gptq_ratio": gptq["ratio"],
                    "beats_gptq": bool(r["ratio"] < gptq["ratio"]),
                }
                _save(tag, {**r, "gptq_ratio": gptq["ratio"],
                            "beats_gptq": step["beats_gptq"]})
                summary["steps"][tag] = step
                print(f"     ratio={r['ratio']:.6f}  vs GPTQ {gptq['ratio']}  "
                      f"-> {'WINS' if step['beats_gptq'] else 'loses'}")
            except Exception as e:
                summary["errors"][tag] = {
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
                traceback.print_exc()
    else:
        summary["errors"]["other_models"] = "no best config to propagate"

    # ── final summary ─────────────────────────────────────────────
    summary["log_file"] = str(log_path)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary_file = _save("SUMMARY_baseline_upgrade", summary)

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
