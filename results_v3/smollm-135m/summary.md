# CHMC v2 Summary — smollm-135m

**Baseline PPL:** 5.103066751400459
**Best method:** `scalar_q4`
- PPL: 9.3947
- PPL ratio: 1.84x
- Honest compression ratio: 3.76x
- Bits per weight: 4.25
- Cosine similarity: 0.9879

## Results Table
| Method | PPL | PPL/Base | Honest CR | B/W | Cos Sim | Status |
|---|---|---|---|---|---|---|
| scalar_q4 | 9.39 | 1.8x | 3.8x | 4.25 | 0.988 | ✅ ok |
| adaptive_dense_hybrid | 11.99 | 2.4x | 2.8x | 5.83 | 0.987 | ✅ ok |
| adaptive_dense_eff_rank | 11.99 | 2.4x | 2.8x | 5.83 | 0.987 | ✅ ok |
| ablation_lowrank_cov_proj_dense | 11.99 | 2.4x | 2.8x | 5.83 | 0.987 | ✅ ok |
| ablation_lowrank_auto_dense | 11.99 | 2.4x | 2.8x | 5.83 | 0.987 | ✅ ok |
| adaptive_sparse_hess_r0.1 | 10153.31 | 1989.7x | 4.5x | 3.58 | 0.878 | ⚠️ degraded |
| adaptive_sparse_hess_r0.05 | 11892.65 | 2330.5x | 6.2x | 2.58 | 0.859 | ⚠️ degraded |
| adaptive_sparse_mag_r0.1 | 15034.92 | 2946.2x | 4.5x | 3.58 | 0.892 | ⚠️ degraded |
| adaptive_sequential_sparse_r0.1 | 15034.92 | 2946.2x | 4.5x | 3.58 | 0.892 | ⚠️ degraded |
| adaptive_sparse_r0.10_q4 | 15034.92 | 2946.2x | 4.5x | 3.58 | 0.892 | ⚠️ degraded |
| adaptive_sparse_mag_r0.05 | 16130.30 | 3160.9x | 6.2x | 2.58 | 0.878 | ⚠️ degraded |
| adaptive_outlier_protected | 34556.42 | 6771.7x | 7.6x | 2.10 | 0.866 | ⚠️ degraded |
| adaptive_sparse_hess_r0.25 | 91444.58 | 17919.5x | 2.4x | 6.58 | 0.915 | ⚠️ degraded |
| adaptive_sparse_mag_r0.25 | 108216.80 | 21206.2x | 2.4x | 6.58 | 0.920 | ⚠️ degraded |
| adaptive_sequential_sparse_r0.25 | 108216.80 | 21206.2x | 2.4x | 6.58 | 0.920 | ⚠️ degraded |
| scalar_q3 | 245704.13 | 48148.3x | 4.9x | 3.25 | 0.949 | ⚠️ degraded |
| adaptive_sparse_r0.10_q3 | 2181388.76 | 427466.2x | 4.6x | 3.48 | 0.864 | ❌ collapse |
| scalar_q2 | 8924493272.16 | 1748849017.1x | 7.1x | 2.25 | 0.683 | ❌ collapse |
| chmc_v1_uniform | 59093626120.35 | 11580022170.8x | 2.8x | 5.83 | 0.284 | ❌ collapse |
| ablation_policy_uniform_dense | 59093626120.35 | 11580022170.8x | 2.8x | 5.83 | 0.284 | ❌ collapse |
| ablation_policy_eff_rank_dense | 59093626120.35 | 11580022170.8x | 2.8x | 5.83 | 0.284 | ❌ collapse |
| ablation_policy_d90_dense | 59093626120.35 | 11580022170.8x | 2.8x | 5.83 | 0.284 | ❌ collapse |
| ablation_policy_hybrid_dense | 59093626120.35 | 11580022170.8x | 2.8x | 5.83 | 0.284 | ❌ collapse |
| ablation_lowrank_weighted_svd_dense | 59093626120.35 | 11580022170.8x | 2.8x | 5.83 | 0.284 | ❌ collapse |

## Rank Allocation Stats
- Min rank: 4
- Max rank: 64
- Median rank: 32
- Mean rank: 29.3

## Verdict: 🟠 WEAK RESULT (better than v1, needs improvement)

## vs CHMC v1
- v1 best PPL ratio: 3.95x (weighted_svd_residual_q4, rank=8)
- v2 best PPL ratio: 1.84x (scalar_q4)
- Improvement over v1: 2.1x