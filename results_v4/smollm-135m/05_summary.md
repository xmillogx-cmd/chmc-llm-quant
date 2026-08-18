# CHMC v4 — SmolLM-135M Experiment Summary

## Model Info
| Parameter | Value |
|---|---|
| Model | HuggingFaceTB/SmolLM-135M |
| Parameters | ~135M (FP16) |
| Baseline PPL | **21.28** (corrected, post-fix) |
| Compressible layers | 210 |
| Vocab size | 49,152 |

## Stage 0 — Scalar Quantization Baselines

| Method | PPL | Ratio vs baseline | BW |
|---|---|---|---|
| FP16 (baseline) | 21.28 | 1.0x | 16.0 |
| Q2 | 9,444,688,669 | **443Mx** | 2.02 |
| Q3 | 50,731 | **2,384x** | 3.02 |
| Q4 | 350.71 | **16.48x** | 4.02 |
| Q5 | 30.64 | **1.44x** | 5.02 |
| Q6 | 23.48 | **1.10x** | 6.02 |

**Finding**: Only q5-q6 are usable; q2-q4 destroy quality completely.

## Stage 1 — CHMC vs Scalar (POST-FIX DATA)

| Method | PPL | Ratio | BW |
|---|---|---|---|
| FP16 baseline | 21.28 | 1.0x | 16.0 |
| scalar_q4 | 350.71 | **16.48x** | 4.02 |
| **CHMC rank=4** | **30.24** | **1.42x** | **4.21** |
| **CHMC rank=8** | **27.53** | **1.29x** | **4.39** |

### CHMC vs Scalar at ~4 bit/weight
| Metric | scalar_q4 | CHMC rank=4 |
|---|---|---|
| BW | 4.02 | 4.21 |
| PPL ratio | 16.48x | **1.42x** |
| Improvement | — | **11.6x better** |

CHMC rank=4 is also slightly better than scalar_q5 (ratio 1.42 vs 1.44) at a LOWER bandwidth (4.21 vs 5.02).

## Stage 2 — Calibration (POST-FIX DATA)

| Variant | PPL | Ratio |
|---|---|---|
| FP16 baseline | 21.28 | 1.0x |
| no_calib_int4 (rank=8, 10 layers) | 21.65 | **1.018x** |
| calib_int8 (rank=8, 10 layers) | 21.27 | **1.0x** |

**Finding**: Calibration has minimal impact at rank=8. INT4 without calibration already within 1.8% of baseline; INT8 with calibration matches baseline exactly.

## Stage 3 — Sparse Compensation (POST-FIX DATA)

### Per-layer analysis (sample 5 layers, rank=8)
| Density | no_comp err | no_comp cos_sim | with_comp err | with_comp cos_sim |
|---|---|---|---|---|
| 0.05 | 0.131 | 0.542 | 0.096 | **0.775** |
| 0.10 | 0.131 | 0.540 | 0.080 | **0.845** |
| 0.25 | 0.132 | 0.535 | 0.053 | **0.936** |

### Full model (density=0.1, all 210 layers)
| Metric | Value |
|---|---|
| PPL | **63,224** |
| Ratio to baseline | **2,972x** |

**Finding**: Sparse compensation improves per-layer cosine similarity significantly (0.54 -> 0.85 at density=0.1) but full-model PPL is catastrophic (2,972x). Not viable as standalone technique — the error accumulates across layers.

## Stage 4 — Effective Rank (POST-FIX DATA)

| Stat | Value |
|---|---|
| Min eff_rank | **29** |
| Median eff_rank | **59** |
| Max eff_rank | **60** |
| Top-1 energy median | **4.3%** |

Lowest effective rank layers:
1. `model.layers.0.self_attn.k_proj` — eff_rank=29 (shape 192x576)
2. `model.layers.0.self_attn.q_proj` — eff_rank=41 (shape 576x576)
3. `model.layers.0.self_attn.o_proj` — eff_rank=50 (shape 576x576)

**Finding**: No layer has eff_rank < 30. Rank-1 structural replacement is NOT viable for SmolLM-135M. k_proj layers consistently have the lowest effective rank across all blocks.

## Stage 5 — Shared Basis Q/K/V (POST-FIX DATA)

| Metric | Value |
|---|---|
| Total QKV layers | 90 |
| Total blocks | 30 |
| Valid blocks for shared basis | **0** |

**Finding**: Q and K projections have different output dimensions (q_proj: 576 out, k_proj: 192 out). Shared basis requires matching shapes — not applicable for this architecture. Zero valid blocks found.

## Overall Conclusions

1. **CHMC wins over scalar quantization** — rank=4 CHMC gives 11.6x better PPL ratio than scalar_q4 at matched bandwidth
2. **Calibration is unnecessary** for this model at moderate ranks (>=8) — no_calib_int4 is only 1.018x vs baseline
3. **Rank-1 replacement impossible** — all layers have eff_rank >= 29
4. **Shared Q/K/V basis not applicable** — architectural mismatch (Q: 576 out, K: 192 out)
5. **Sparse compensation not viable alone** — per-layer improvements don't translate to full-model quality (PPL ratio 2,972x)

## Data Files (5-file format)
| # | File | Content | Status |
|---|---|---|---|
| 01 | `01_baseline.json` | Baseline PPL + scalar q2-q6 | ✅ Post-fix |
| 02 | `02_chmc_comparison.json` | CHMC vs scalar comparison | ✅ Post-fix |
| 03 | `03_calibration_and_sparse.json` | Stage 2-3 | ✅ Post-fix (fresh run) |
| 04 | `04_advanced_techniques.json` | Stage 4-5 | ✅ Post-fix (fresh run) |
| 05 | `05_summary.md` | This file | ✅ Updated |
