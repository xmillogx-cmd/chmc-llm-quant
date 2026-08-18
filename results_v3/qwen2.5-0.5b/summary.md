# CHMC v2 Summary — qwen2.5-0.5b

**Baseline PPL:** 4.518224439777543
**Best method:** `adaptive_dense_hybrid`
- PPL: 8.1677
- PPL ratio: 1.81x
- Honest compression ratio: 3.18x
- Bits per weight: 5.0323
- Cosine similarity: 0.9855

## Results Table
| Method | PPL | PPL/Base | Honest CR | B/W | Cos Sim | Status |
|---|---|---|---|---|---|---|
| adaptive_dense_hybrid | 8.17 | 1.8x | 3.2x | 5.03 | 0.986 | ✅ ok |
| adaptive_dense_eff_rank | 8.17 | 1.8x | 3.2x | 5.03 | 0.986 | ✅ ok |
| ablation_lowrank_cov_proj_dense | 8.17 | 1.8x | 3.2x | 5.03 | 0.986 | ✅ ok |
| ablation_lowrank_auto_dense | 8.17 | 1.8x | 3.2x | 5.03 | 0.986 | ✅ ok |
| scalar_q4 | 10.21 | 2.3x | 3.8x | 4.25 | 0.984 | ✅ ok |
| adaptive_sparse_hess_r0.1 | 5003.77 | 1107.5x | 5.8x | 2.78 | 0.878 | ⚠️ degraded |
| adaptive_sparse_mag_r0.1 | 9404.68 | 2081.5x | 5.8x | 2.78 | 0.889 | ⚠️ degraded |
| adaptive_sequential_sparse_r0.1 | 9404.68 | 2081.5x | 5.8x | 2.78 | 0.889 | ⚠️ degraded |
| adaptive_sparse_r0.10_q4 | 9404.68 | 2081.5x | 5.8x | 2.78 | 0.889 | ⚠️ degraded |
| adaptive_sparse_hess_r0.05 | 11447.49 | 2533.6x | 9.0x | 1.78 | 0.864 | ⚠️ degraded |
| adaptive_sparse_mag_r0.25 | 21997.03 | 4868.5x | 2.8x | 5.78 | 0.899 | ⚠️ degraded |
| adaptive_sequential_sparse_r0.25 | 21997.03 | 4868.5x | 2.8x | 5.78 | 0.899 | ⚠️ degraded |
| adaptive_sparse_hess_r0.25 | 24399.34 | 5400.2x | 2.8x | 5.78 | 0.897 | ⚠️ degraded |
| adaptive_sparse_mag_r0.05 | 27282.61 | 6038.4x | 9.0x | 1.78 | 0.883 | ⚠️ degraded |
| adaptive_sparse_r0.10_q3 | 31043.85 | 6870.8x | 6.0x | 2.68 | 0.869 | ⚠️ degraded |
| adaptive_outlier_protected | 40819.26 | 9034.4x | 12.3x | 1.30 | 0.880 | ⚠️ degraded |
| scalar_q3 | 151791.33 | 33595.3x | 4.9x | 3.25 | 0.932 | ⚠️ degraded |
| scalar_q2 | 18004799.49 | 3984928.1x | 7.1x | 2.25 | 0.631 | ❌ collapse |
| chmc_v1_uniform | 11758299003494.88 | 2602415873805.9x | 3.2x | 5.03 | -0.007 | ❌ collapse |
| ablation_policy_uniform_dense | 11758299003494.88 | 2602415873805.9x | 3.2x | 5.03 | -0.007 | ❌ collapse |
| ablation_policy_eff_rank_dense | 11758299003494.88 | 2602415873805.9x | 3.2x | 5.03 | -0.007 | ❌ collapse |
| ablation_policy_d90_dense | 11758299003494.88 | 2602415873805.9x | 3.2x | 5.03 | -0.007 | ❌ collapse |
| ablation_policy_hybrid_dense | 11758299003494.88 | 2602415873805.9x | 3.2x | 5.03 | -0.007 | ❌ collapse |
| ablation_lowrank_weighted_svd_dense | 11758299003494.88 | 2602415873805.9x | 3.2x | 5.03 | -0.007 | ❌ collapse |

## Rank Allocation Stats
- Min rank: 4
- Max rank: 64
- Median rank: 32
- Mean rank: 29.7

## Verdict: 🟠 WEAK RESULT (better than v1, needs improvement)

## vs CHMC v1
- v1 best PPL ratio: 3.95x (weighted_svd_residual_q4, rank=8)
- v2 best PPL ratio: 1.81x (adaptive_dense_hybrid)
- Improvement over v1: 2.2x