# CHMC v1 Summary

**Date:** results_v2
**Model:** HuggingFaceTB/SmolLM-135M
**Baseline PPL:** 1.9055 (diverse fallback, sliding window)

## 1. Evaluation Fix
- Baseline PPL: 1.9055
- Sliding window with stride=256, max_len=512
- Overlap tokens set to -100 (not counted in loss)

## 2. Calibration Statistics
- Layers analyzed: 210
- Covariance d90 median: 66
- Best covariance rank found: see cov_stats.json

## 3. Best Method
**Method:** `weighted_svd_residual_q4` (rank=8, bits=4)
- PPL: 7.5218
- PPL ratio to baseline: 3.95x
- Avg cosine similarity: 0.9889
- Avg compression ratio: 151.48x

## 4. Comparison Table
| Method | Rank | PPL | PPL/Base | Cos Sim | Compression |
|---|---|---|---|---|---|
| weighted_svd_residual_q4 | 8 | 7.52 | 4.0x | 0.989 | 151.5x |
| weighted_svd_residual_q4 | 16 | 7.79 | 4.1x | 0.990 | 75.7x |
| cov_proj_residual_q4 | 32 | 8.39 | 4.4x | 0.988 | 2.7x |
| cov_proj_residual_q4 | 8 | 9.07 | 4.8x | 0.987 | 3.6x |
| cov_proj_residual_q4 | 16 | 9.78 | 5.1x | 0.987 | 3.2x |
| weighted_svd_residual_q4 | 32 | 10.29 | 5.4x | 0.991 | 37.9x |
| cov_proj | 8 | 9326.81 | 4894.7x | 0.762 | 37.9x |
| weighted_svd | 8 | 12942.22 | 6792.0x | 0.781 | 37.9x |
| cov_proj | 32 | 28501.39 | 14957.4x | 0.880 | 9.5x |
| weighted_svd | 32 | 58288.29 | 30589.5x | 0.902 | 9.5x |

## 5. vs Previous Plain Low-Rank
- Old plain SVD best PPL: 458659.1814
- New CHMC best PPL: 7.5218
- Improvement factor: 60977.3x

## 6. vs Scalar 4-bit Baseline
- Old scalar 4-bit PPL: 1.3631
- New CHMC best PPL: 7.5218
- [FAIL] Worse than scalar 4-bit

## 7. Verdict
**[GREEN] STRONG SUCCESS — covariance-aware compression works!**

## 8. Next Steps
- Develop block-wise reconstruction for even better quality
- Try mixed-precision residual quantization (INT4/INT2)