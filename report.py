"""
report.py — Final report from all results.
Reads results/*.json → report.md + a table in the console.
"""

import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
RESULTS = BASE_DIR / "results"


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ga = load("geometry_activations.json")   # activations
    gw = load("geometry_weights.json")       # weights
    bp = load("ppl_base.json")               # baseline PPL
    qs = load("quant_scalar.json") or []      # scalar quantization
    lr = load("lowrank_eval.json") or []      # low-rank only
    lq = load("lowrank_quant.json") or []     # low-rank + quant

    base_ppl = bp.get("perplexity") if bp else None

    lines = []
    lines.append(f"# CMQ Quick-Check Report\n")
    lines.append(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n")
    lines.append(f"**Model:** HuggingFaceTB/SmolLM-135M\n")

    # ── Activation geometry ────────────────────────────────────
    lines.append("\n## 1. Activation Geometry\n")
    if ga:
        lines.append("| Metric | Value | Interpretation |")
        lines.append("|---|---|---|")
        ag = ga["anisotropy_global"]
        cone = "🔥 strong" if ag > 0.8 else "✅ moderate" if ag > 0.5 else "⚠️ weak" if ag > 0.2 else "❌ isotropic"
        lines.append(f"| anisotropy_global | {ag:.4f} | {cone} |")
        lines.append(f"| anisotropy_pairwise | {ga['anisotropy_pairwise']:.4f} | {'strong' if ga['anisotropy_pairwise'] > 0.3 else 'moderate' if ga['anisotropy_pairwise'] > 0.1 else 'weak'} |")
        lines.append(f"| PCA d90 | {ga['pca_d90']} / {ga['hidden_dim']} | ratio={ga.get('low_dim_ratio', '?')} |")
        lines.append(f"| top10% variance | {ga.get('top10_pct_variance', '?'):.4f} | |")

        if ag > 0.5 and ga.get("low_dim_ratio", 1) < 0.33:
            lines.append("\n✅ **Geometry CONFIRMED** — strong anisotropy + low effective dimension\n")
        elif ag > 0.2 or ga.get("low_dim_ratio", 1) < 0.5:
            lines.append("\n⚠️ **Partially confirmed**\n")
        else:
            lines.append("\n❌ **Geometry WEAK** — activations nearly isotropic\n")
    else:
        lines.append("_No data_\n")

    # ── Weight structure ────────────────────────────────────────
    lines.append("## 2. Weight Low-Rank Structure\n")
    if gw:
        lines.append("| Metric | Value | Interpretation |")
        lines.append("|---|---|---|")
        e64 = gw.get("mean_e64", 0)
        ver = "✅ very low-rank" if e64 > 0.6 else "⚠️ moderate" if e64 > 0.3 else "❌ weak"
        lines.append(f"| mean_energy_top16 | {gw.get('mean_e16', '?'):.4f} | |")
        lines.append(f"| mean_energy_top32 | {gw.get('mean_e32', '?'):.4f} | |")
        lines.append(f"| mean_energy_top64 | {e64:.4f} | {ver} |")
        lines.append(f"| mean_energy_top128 | {gw.get('mean_e128', '?'):.4f} | |")
        if "rank90" in gw:
            r = gw["rank90"]
            lines.append(f"| rank90 stats | mean={r['mean']}, median={r['median']} | |")
    else:
        lines.append("_No data_\n")

    # ── Compression summary table ────────────────────────────────
    lines.append("\n## 3. Compression Results\n")
    lines.append("| Method | Config | PPL | Δ% | ratio | compression | vs scalar |")
    lines.append("|---|---|---:|---:|---:|---:|---|")

    if base_ppl is not None:
        lines.append(f"| **baseline** | FP32/BF16 | **{base_ppl:.2f}** | — | 1.00 | 1× | — |")

    for r in qs:
        d = f"{r.get('ppl_degradation_pct', '?'):+.1f}%" if base_ppl else "?"
        lines.append(f"| scalar | {r['bits']}-bit | {r['perplexity']:.2f} | {d} | {r.get('ppl_ratio', '?')} | {r.get('compression_ratio', '?'):.1f}× | — |")

    for r in lr:
        d = f"{r.get('ppl_degradation_pct', '?'):+.1f}%" if base_ppl else "?"
        lines.append(f"| low-rank | rank {r['rank']} | {r['perplexity']:.2f} | {d} | {r.get('ppl_ratio', '?')} | {r.get('mean_compression_ratio', '?'):.1f}× | — |")

    for r in lq:
        d = f"{r.get('ppl_degradation_pct', '?'):+.1f}%" if base_ppl else "?"
        vs = ""
        if r.get("better_than_scalar"):
            vs = f"✅ ({r.get('vs_scalar_pct', ''):+.1f}%)"
        elif "vs_scalar_pct" in r:
            vs = f"✗ ({r.get('vs_scalar_pct', ''):+.1f}%)"
        lines.append(f"| **LR+Q** | r{r['rank']} q{r['bits']}b | {r['perplexity']:.2f} | {d} | {r.get('ppl_ratio', '?')} | {r.get('mean_compression_ratio', '?'):.1f}× | {vs} |")

    # ── Conclusions ────────────────────────────────────────────────
    lines.append("\n## 4. Conclusions\n")

    if lq:
        best = min(lq, key=lambda r: r["perplexity"])
        lines.append(f"**Best LR+Q:** `{best['method']}` — PPL={best['perplexity']:.2f}, compression={best.get('mean_compression_ratio', '?'):.1f}×")

        if base_ppl and best["perplexity"] / base_ppl < 1.5:
            lines.append("✅ Acceptable degradation (< 50%)\n")
        elif base_ppl:
            ratio = best["perplexity"] / base_ppl
            lines.append(f"⚠️ PPL ratio = {ratio:.2f}× ({'acceptable' if ratio < 2 else 'significant degradation'})\n")

    # Final verdict
    confirmed = False
    if ga and gw:
        ag = ga.get("anisotropy_global", 0)
        e64 = gw.get("mean_e64", 0)
        ld = ga.get("low_dim_ratio", 1.0)
        if ag > 0.5 and e64 > 0.3:
            confirmed = True

    lines.append("\n### Overall Verdict\n")
    if confirmed and lq:
        best = min(lq, key=lambda r: r["perplexity"])
        if base_ppl and best["perplexity"] / base_ppl < 2:
            lines.append("🟢 **CMQ CONFIRMED** — geometry supports compression, LR+Q achieves acceptable quality.")
            lines.append("\nNext steps:")
            lines.append("- QAT (Quantization-Aware Training) for 2-bit")
            lines.append("- LoRA-QAT — fine-tune only the low-rank factors")
        else:
            lines.append("🟡 **Geometry confirmed, but quality degrades too much.** Try QAT.")
    elif not confirmed:
        lines.append("🔴 **CMQ NOT strongly supported** by this model.")
        lines.append("\nNext steps:")
        lines.append("- Try other models (Qwen, LLaMA)")
        lines.append("- Check the intermediate layers")

    # Save + print
    report = "\n".join(lines)
    out = BASE_DIR / "report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
