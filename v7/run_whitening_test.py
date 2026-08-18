#!/usr/bin/env python3
"""
run_whitening_test.py — CHMC v7: полная ковариационная whitening (Lever 1)
==========================================================================

Контекст: v6 показал модель-зависимость — TinyLlama win 33σ, Qwen loss −2.8σ,
SmolLM neutral. Hypothesis (Lever 1): текущий damping использует ТОЛЬКО
ДИАГОНАЛЬ ковариации (diag_c), игнорируя корреляции между input-фичами.
Полная whitening (C^{1/2}, data-adaptive) учитывает корреляции.

Математика (в chmc_v6.py, ветка whitening=True):
  Цель: минимизировать ||(W - W_lr) @ C^{1/2}||_F,  C = X^T X / n.
  W_weighted = W @ C^{1/2};  top-rank SVD;  W_lr = (U S Vt) @ C^{-1/2}.
  При диагональной C это РОВНО текущий код (diag damping) — строгое обобщение.

Риск (по niter-тесту): более data-adaptive трансформ → возможный overfit к
калибровке. Проверяем эмпирически.

Тест: 3 модели × (whitening=False baseline, whitening=True) × 3 seed (42,43,44).
Лучший конфиг каждой модели из v6-сетки. Равный BPW 4.2875.
Сравнение per-seed (whitened - baseline) изолирует эффект whitening.

Self-check: TinyLlama baseline (whitening=False, block_damp01) должен
воспроизвести niter-тест (niter=5, seeds 42,43,44):
  [1.058485, 1.057963, 1.059282]. Если нет — путь baseline сломан.

Итого 18 прогонов, ~25-30 мин.

Запускает ПОЛЬЗОВАТЕЛЬ (агент разделяет GPU и рискует OOM).
Артефакты: results_v6/whitening_test/

Usage:
    python run_whitening_test.py
"""

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
OUT_DIR = ROOT / "results_v6" / "whitening_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BPW = 4.2875
BASE_SEED = 42
N_REPS = 3

# лучший конфиг каждой модели из v6-сетки + GPTQ reference
MODELS_CFG = {
    "smollm-135m": {
        "overrides": {"strict_sequential": True, "dampening": 0.05},
        "gptq_ratio": 1.1761,
    },
    "qwen2.5-0.5b": {
        "overrides": {"strict_sequential": True, "dampening": 0.06},
        "gptq_ratio": 1.1407,
    },
    "tinyllama-1.1b": {
        "overrides": {"strict_sequential": False, "dampening": 0.1},
        "gptq_ratio": 1.0807,
    },
}

# self-check: TinyLlama block_damp01, niter=5, seeds 42,43,44 (из niter_test)
SELF_CHECK = {
    "model": "tinyllama-1.1b",
    "ratios": [1.058485, 1.057963, 1.059282],
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
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


def run_config(tag, model, overrides, whitening, seed):
    """Run one CHMC v7 config at equal BPW with a given whitening flag and seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_path = str(MODELS / model)
    cfg = {**C.default_config(), **overrides,
           "whitening": whitening, "bit_budget_bpw": TARGET_BPW}
    result = C.run_chmc_v6(model_path, cfg, tag=tag)
    # run_chmc_v6 already frees the model (del + gc.collect + empty_cache);
    # this is a safety net in case a reference cycle survived.
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    t0 = time.time()
    log_path = _setup_logging()
    summary = {
        "target_bpw": TARGET_BPW,
        "seeds": list(range(BASE_SEED, BASE_SEED + N_REPS)),
        "models": {},
        "errors": {},
    }

    print(f"Log file: {log_path}")
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"CUDA: {torch.cuda.is_available()} | device: {C.DEVICE}")
    print(f"BPW: {TARGET_BPW} | reps: {N_REPS} | models: {list(MODELS_CFG)}")

    for model, spec in MODELS_CFG.items():
        overrides = spec["overrides"]
        gptq = spec["gptq_ratio"]
        print("\n" + "=" * 64)
        print(f"MODEL: {model}  (config={overrides}, GPTQ={gptq})")
        print("=" * 64)

        model_summary = {
            "overrides": overrides,
            "gptq_ratio": gptq,
            "baseline": None,
            "whitened": None,
            "delta_per_seed": None,
            "verdict": None,
        }

        for label, whitening in [("baseline", False), ("whitened", True)]:
            print(f"\n  --- {label} (whitening={whitening}) ---")
            ratios = []
            for i in range(N_REPS):
                seed = BASE_SEED + i
                run_tag = f"{model}_{label}_seed{seed}"
                try:
                    r = run_config(run_tag, model, overrides, whitening, seed)
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
                model_summary[label] = {
                    "n": len(ratios),
                    "ratios": [round(x, 6) for x in ratios],
                    "mean": round(_mean(ratios), 6),
                    "std": round(_std(ratios), 6),
                    "min": round(min(ratios), 6),
                    "max": round(max(ratios), 6),
                }
                m = _mean(ratios)
                s = _std(ratios)
                print(f"  {label}: mean={m:.6f}  std={s:.6f}  "
                      f"min={min(ratios):.6f}  max={max(ratios):.6f}")

        # ── per-seed delta (whitened - baseline) ──────────────────
        base = model_summary.get("baseline")
        white = model_summary.get("whitened")
        if base and white and base["n"] == white["n"]:
            delta = [w - b for b, w in zip(base["ratios"], white["ratios"])]
            model_summary["delta_per_seed"] = [round(d, 6) for d in delta]
            mean_delta = _mean(delta)
            # negative delta = whitening improved (lower ratio)
            if mean_delta < -1e-4:
                verdict = f"WHITENING HELPS (mean delta {mean_delta:+.6f})"
            elif mean_delta > 1e-4:
                verdict = f"WHITENING HURTS (mean delta {mean_delta:+.6f})"
            else:
                verdict = f"NO EFFECT (mean delta {mean_delta:+.6f})"
            model_summary["verdict"] = verdict
            print(f"\n  delta/seed (white-base): {model_summary['delta_per_seed']}")
            print(f"  -> {verdict}")

        # ── vs GPTQ ───────────────────────────────────────────────
        if white:
            margin_white = gptq - white["mean"]
            print(f"  whitened vs GPTQ: margin {margin_white:+.6f} "
                  f"({'WINS' if white['mean'] < gptq else 'loses'})")
        if base:
            margin_base = gptq - base["mean"]
            print(f"  baseline vs GPTQ: margin {margin_base:+.6f} "
                  f"({'WINS' if base['mean'] < gptq else 'loses'})")

        summary["models"][model] = model_summary

    # ── self-check ────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("SELF-CHECK: TinyLlama baseline vs niter_test (same seeds)")
    print("=" * 64)
    sc_model = SELF_CHECK["model"]
    sc = {"expected": SELF_CHECK["ratios"], "reproduced": None,
          "max_diff": None, "match": None}
    tl_base = summary["models"].get(sc_model, {}).get("baseline")
    if tl_base:
        max_diff = max(abs(a - b) for a, b in
                       zip(tl_base["ratios"], SELF_CHECK["ratios"]))
        match = max_diff < 1e-5
        sc["reproduced"] = tl_base["ratios"]
        sc["max_diff"] = round(max_diff, 8)
        sc["match"] = bool(match)
        print(f"  expected: {SELF_CHECK['ratios']}")
        print(f"  now:      {tl_base['ratios']}")
        print(f"  max_diff: {max_diff:.2e}  -> "
              f"{'MATCH' if match else 'MISMATCH (baseline path broken!)'}")
    else:
        print("  TinyLlama baseline missing — cannot self-check")
    _save("self_check", sc)
    summary["self_check"] = sc

    # ── final summary ─────────────────────────────────────────────
    summary["log_file"] = str(log_path)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary_file = _save("SUMMARY_whitening_test", summary)

    print("\n" + "=" * 64)
    print("SUMMARY (mean ratio, lower is better)")
    print("=" * 64)
    print(f"  {'model':<16} {'base':>9} {'white':>9} {'delta':>9} "
          f"{'GPTQ':>9}  verdict")
    for model, ms in summary["models"].items():
        b = ms.get("baseline", {}).get("mean", float("nan"))
        w = ms.get("whitened", {}).get("mean", float("nan"))
        d = (w - b) if (b == b and w == w) else float("nan")
        g = ms["gptq_ratio"]
        print(f"  {model:<16} {b:>9.6f} {w:>9.6f} {d:>+9.6f} "
              f"{g:>9.6f}  {ms.get('verdict', 'n/a')}")
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
