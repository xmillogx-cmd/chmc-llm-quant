#!/usr/bin/env python3
"""
run_step12.py — CHMC v6 Step 1 (baseline) + Step 2 (A1/A2/A3 single)
=====================================================================

Per the directive: "Начни с Шага 1 и Шага 2. Не переходи к комбинациям
пока не проверены одиночные техники."

All runs are at an EQUAL bit budget where a fair comparison is intended:
  - Step 1a: v5-style baseline (rank=8)  -> reproduces ~1.188x, BPW ~4.50
  - Step 1b: baseline at GPTQ's BPW (4.2875) -> the FAIR baseline
  - Step 2 : A1 / A2 / A3, each at GPTQ's BPW (4.2875) -> isolate each gain

GPTQ reference (same model, same BPW): ratio 1.1761 at 4.2875 BPW.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chmc_v6 import run_chmc_v6, RESULTS, GPTQ_BPW, ROOT_DIR

MODEL = str(ROOT_DIR / "models" / "smollm-135m")
GPTQ_RATIO = 1.1761  # from results_v5/gptq_awq_baselines.json

# ── configs ──────────────────────────────────────────────────────
CONFIGS = [
    # (label, config-overrides, description)
    ("step1_baseline_rank8",
     {"rank": 8, "bit_budget_bpw": None},
     "v5-style baseline (rank=8), reproduces ~1.188x, BPW~4.50 (UNFAIR vs GPTQ)"),
    ("step1_baseline_matched",
     {"bit_budget_bpw": GPTQ_BPW},
     "baseline at GPTQ BPW (4.2875) — FAIR baseline"),
    ("step2_A1_strict",
     {"bit_budget_bpw": GPTQ_BPW, "strict_sequential": True},
     "A1 strict sequential (one col at a time, full H^-1) at matched BPW"),
    ("step2_A2_groupdim1",
     {"bit_budget_bpw": GPTQ_BPW, "group_dim": 1},
     "A2 GPTQ group orientation (dim=1) at matched BPW"),
    ("step2_A3_hess8",
     {"bit_budget_bpw": GPTQ_BPW, "hessian_batches": 8},
     "A3 Hessian accumulation (8 batches) at matched BPW"),
]


def main():
    matrix = {}
    baseline_ppls = []

    for label, cfg, desc in CONFIGS:
        print(f"\n\n{'#' * 64}")
        print(f"# {label}")
        print(f"# {desc}")
        print(f"{'#' * 64}")
        t0 = time.time()
        try:
            res = run_chmc_v6(MODEL, config=cfg, tag=f"smollm-135m__{label}")
            matrix[label] = res
            if res.get("baseline_ppl"):
                baseline_ppls.append((label, res["baseline_ppl"]))
        except Exception as e:
            import traceback
            traceback.print_exc()
            matrix[label] = {
                "model": "smollm-135m", "label": label,
                "error": f"{type(e).__name__}: {e}",
            }
            print(f"[ERROR] {label} failed: {e}")

    # ── HARD RULE 2: baseline PPL must be consistent (<1% spread) ──
    if len(baseline_ppls) >= 2:
        ppls = [p for _, p in baseline_ppls]
        spread = (max(ppls) - min(ppls)) / (sum(ppls) / len(ppls))
        print(f"\n[BASELINE CHECK] baseline PPLs: {baseline_ppls}")
        print(f"[BASELINE CHECK] relative spread = {spread * 100:.3f}% "
              f"({'OK <1%' if spread < 0.01 else 'FAIL >=1% — STOP'})")

    # ── save ─────────────────────────────────────────────────────
    out = RESULTS / "ablation_matrix.json"
    with open(out, "w") as f:
        json.dump(matrix, f, indent=2, default=str)
    print(f"\nSaved -> {out}")

    # ── summary table ────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("CHMC v6 — STEP 1 + STEP 2 SUMMARY  (SmolLM-135M)")
    print(f"{'=' * 72}")
    print(f"{'config':<24} {'PPL':>9} {'ratio':>9} {'BPW':>7} {'ΔBPW':>7}  verdict")
    print("-" * 72)
    for label, _cfg, _desc in CONFIGS:
        r = matrix.get(label, {})
        if "error" in r:
            print(f"{label:<24} {'ERR':>9} {r['error'][:40]}")
            continue
        bpw = r.get("bpw", float("nan"))
        delta = r.get("bpw_delta_vs_gptq", float("nan"))
        ratio = r.get("ratio", float("nan"))
        # verdict vs GPTQ (only fair when BPW matched)
        if abs(bpw - GPTQ_BPW) < 0.02:
            verdict = "BEATS GPTQ" if ratio < GPTQ_RATIO else "loses to GPTQ"
        else:
            verdict = "(unfair BPW)"
        print(f"{label:<24} {r.get('compressed_ppl', float('nan')):>9.4f} "
              f"{ratio:>9.6f} {bpw:>7.4f} {delta:>+7.4f}  {verdict}")
    print("-" * 72)
    print(f"{'GPTQModel (ref)':<24} {'25.0595':>9} {GPTQ_RATIO:>9.6f} "
          f"{GPTQ_BPW:>7.4f} {'':>7}  reference")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
