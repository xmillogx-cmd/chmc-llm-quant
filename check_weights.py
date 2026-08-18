"""
check_weights.py — Low-rank структура весов (SVD energy).
Результат → results/geometry_weights.json
"""

import json
from pathlib import Path

import torch
from tqdm import tqdm

from model_loader import load_model, BASE_DIR, DEVICE, DTYPE

RESULTS = BASE_DIR / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("  CMQ — Weight Low-Rank Structure")
    print("=" * 60)

    # ── Загрузка модели (с прогрессом + ретраями) ─────────────
    model = load_model()

    records = []
    skip_reasons = {"non_2d": 0, "embedding": 0, "too_small": 0, "svd_fail": 0}

    # Собираем список матриц
    matrices = []
    for name, param in model.named_parameters():
        if param.ndim != 2:
            skip_reasons["non_2d"] += 1
            continue
        if "embed" in name or "lm_head" in name:
            skip_reasons["embedding"] += 1
            continue
        if min(param.shape) < 64:
            skip_reasons["too_small"] += 1
            continue
        matrices.append((name, param))

    print(f"\n  Matrices to analyze: {len(matrices)}")
    print(f"  Skipped: non_2d={skip_reasons['non_2d']}, embed={skip_reasons['embedding']}, small={skip_reasons['too_small']}")

    # ── SVD по каждой матрице ─────────────────────────────────
    for name, param in tqdm(matrices, desc="SVD", unit="matrix", ncols=80):
        w = param.float()
        try:
            S = torch.linalg.svdvals(w)
        except Exception:
            skip_reasons["svd_fail"] += 1
            continue

        total = float((S ** 2).sum())
        if total == 0:
            continue

        energy = {}
        for k in (8, 16, 32, 64, 128, 256):
            ek = min(k, len(S))
            energy[f"e{ek}"] = round(float((S[:ek] ** 2).sum()) / total, 6)

        cum = torch.cumsum(S ** 2, dim=0) / total
        r90 = min(int(torch.searchsorted(cum, 0.90).item()) + 1, min(param.shape))
        r95 = min(int(torch.searchsorted(cum, 0.95).item()) + 1, min(param.shape))

        records.append({
            "name": name,
            "shape": list(param.shape),
            **energy,
            "r90": r90, "r95": r95,
        })

    # ── Агрегация ─────────────────────────────────────────────
    if records:
        n = len(records)
        summary = {
            "model": "HuggingFaceTB/SmolLM-135M",
            "n_matrices": n,
        }
        for key in ("e8", "e16", "e32", "e64", "e128", "e256"):
            vals = [r[key] for r in records if key in r]
            summary[f"mean_{key}"] = round(sum(vals) / len(vals), 6)
            summary[f"min_{key}"] = round(min(vals), 6)
            summary[f"max_{key}"] = round(max(vals), 6)

        ranks90 = sorted(r["r90"] for r in records)
        summary["rank90"] = {
            "mean": round(sum(ranks90) / len(ranks90), 1),
            "median": ranks90[len(ranks90) // 2],
            "min": min(ranks90), "max": max(ranks90),
        }

        e64 = summary["mean_e64"]
        if e64 > 0.6:
            summary["verdict"] = "[OK] very_low_rank - rank 64 покрывает энергию"
        elif e64 > 0.3:
            summary["verdict"] = "[WARN] moderate - нужен больший rank"
        else:
            summary["verdict"] = "[FAIL] weak low-rank"
    else:
        summary = {"n_matrices": 0, "verdict": "no matrices found"}

    print("\n" + "=" * 60)
    for k in ("mean_e16", "mean_e32", "mean_e64", "mean_e128"):
        if k in summary:
            print(f"  {k:20s}: {summary[k]:.4f}")
    if "rank90" in summary:
        r = summary["rank90"]
        print(f"  rank90              : mean={r['mean']}, median={r['median']}")
    print(f"  verdict             : {summary.get('verdict', 'N/A')}")

    out = RESULTS / "geometry_weights.json"
    with open(out, "w") as f:
        json.dump({**summary, "matrices": records}, f, indent=2)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
