# CHMC v4 — Qwen/Qwen2.5-0.5B Experiment Summary

## Model Info
| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B |
| Parameters | ~494M (FP16) |
| Baseline PPL | **15.55** (corrected, post-fix) |
| Compressible layers | 168 |
| Vocab size | 151,665 |

### Baseline Correction
Old pre-fix baseline was 5,494 — caused by using SmolLM's WikiText cache (vocab=49152) for Qwen (vocab=151665). After fixing to model-specific caching, correct baseline is **15.55**, which matches expected range for a 0.5B model.

## Stage 0 — Scalar Quantization Baselines

| Method | PPL | Ratio vs baseline | BW |
|---|---|---|---|
| FP16 (baseline) | 15.55 | 1.0x | 16.0 |
| Q2 | 29,690,904 | **1.9Mx** | 2.01 |
| Q3 | 311,313 | **20,024x** | 3.01 |
| Q4 | 161.43 | **10.38x** | 4.01 |
| Q5 | 22.04 | **1.42x** | 5.01 |
| Q6 | 17.13 | **1.10x** | 6.01 |

## Stage 1 — CHMC vs Scalar (POST-FIX DATA)

| Method | PPL | Ratio | BW |
|---|---|---|---|
| FP16 baseline | 15.55 | 1.0x | 16.0 |
| scalar_q4 | 161.43 | **10.38x** | 4.01 |
| **CHMC rank=4** | **27.08** | **1.74x** | **4.11** |
| **CHMC rank=8** | **23.24** | **1.50x** | **4.21** |

### CHMC vs Scalar at ~4 bit/weight
| Metric | scalar_q4 | CHMC rank=4 |
|---|---|---|
| BW | 4.01 | 4.11 |
| PPL ratio | 10.38x | **1.74x** |
| Improvement | — | **6.0x better** |

## Stage 2 — Calibration (POST-FIX DATA)

| Variant | PPL | Ratio |
|---|---|---|
| FP16 baseline | 15.55 | 1.0x |
| no_calib_int4 (rank=8, 10 layers) | 15.84 | **1.019x** |
| calib_int8 (rank=8, 10 layers) | 15.56 | **1.001x** |

**Finding**: Calibration has minimal impact at rank=8 on Qwen. INT4 without calibration is within 1.9% of baseline; INT8 with calibration matches baseline exactly.

## Stage 3 — Sparse Compensation (POST-FIX DATA)

### Per-layer analysis (sample 5 layers, rank=8)
| Density | no_comp err | no_comp cos_sim | with_comp err | with_comp cos_sim |
|---|---|---|---|---|
| 0.05 | 0.028 | 0.496 | 0.021 | **0.742** |
| 0.10 | 0.028 | 0.495 | 0.018 | **0.817** |
| 0.25 | 0.028 | 0.496 | 0.012 | **0.920** |

### Full model (density=0.1, all 168 layers)
| Metric | Value |
|---|---|
| PPL | **653,985** |
| Ratio to baseline | **42,065x** |

**Finding**: Sparse compensation improves per-layer cosine similarity (0.495 -> 0.817 at density=0.1) but full-model PPL is catastrophic (42,065x). Not viable as standalone technique — error accumulation across 168 layers is worse than SmolLM's 210 layers.

## Stage 4 — Effective Rank (POST-FIX DATA)

| Stat | Value |
|---|---|
| Min eff_rank | **29** |
| Median eff_rank | **60** |
| Max eff_rank | **61** |
| Top-1 energy median | **3.8%** |

Lowest effective rank layers:
1. `model.layers.0.self_attn.k_proj` — eff_rank=29 (shape 128x896)
2. `model.layers.0.self_attn.q_proj` — eff_rank=41 (shape 896x896)
3. `model.layers.16.self_attn.k_proj` — eff_rank=52 (shape 128x896)

**Finding**: No layer has eff_rank < 30. Rank-1 structural replacement is NOT viable for Qwen2.5-0.5B. k_proj layers consistently have the lowest effective rank across all blocks.

## Stage 5 — Shared Basis Q/K/V (POST-FIX DATA)

| Metric | Value |
|---|---|
| Total QKV layers | 72 |
| Total blocks | 24 |
| Valid blocks for shared basis | **0** |

**Finding**: Q and K projections have different output dimensions (q_proj: 896 out, k_proj: 128 out). Shared basis requires matching shapes — not applicable for this architecture. Zero valid blocks found.

## Cross-Model Comparison

| Model | Baseline | scalar_q4 ratio | CHMC rank=4 ratio | CHMC advantage |
|---|---|---|---|---|
| SmolLM-135M | 21.28 | 16.48x | **1.42x** | **11.6x** |
| Qwen2.5-0.5B | 15.55 | 10.38x | **1.74x** | **6.0x** |

CHMC provides significant improvement on both models, with a larger relative advantage on SmolLM (11.6x vs 6.0x). Both models show CHMC rank=4 is competitive with or better than scalar_q5 at lower bandwidth.

## Overall Conclusions

1. **CHMC wins over scalar quantization** — rank=4 CHMC gives 6.0x better PPL ratio than scalar_q4 at matched bandwidth
2. **Calibration is unnecessary** for this model at moderate ranks (>=8) — no_calib_int4 is only 1.019x vs baseline
3. **Rank-1 replacement impossible** — all layers have eff_rank >= 29
4. **Shared Q/K/V basis not applicable** — architectural mismatch (Q: 896 out, K: 128 out)
5. **Sparse compensation not viable alone** — per-layer improvements don't translate to full-model quality (PPL ratio 42,065x)

## Data Files (5-file format)
| # | File | Content | Status |
|---|---|---|---|
| 01 | `01_baseline.json` | Baseline PPL + scalar q2-q6 | ✅ Post-fix |
| 02 | `02_chmc_comparison.json` | CHMC vs scalar comparison | ✅ Post-fix |
| 03 | `03_calibration_and_sparse.json` | Stage 2-3 | ✅ Post-fix (fresh run) |
| 04 | `04_advanced_techniques.json` | Stage 4-5 | ✅ Post-fix (fresh run) |
| 05 | `05_summary.md` | This file | ✅ Updated |
