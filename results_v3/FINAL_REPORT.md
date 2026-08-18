# CHMC v2 — Final Report

**Date:** results_v3
**Device:** cpu

## 1. Tested Models
- **smollm-135m**: baseline PPL=5.103066751400459, best=scalar_q4 (PPL ratio=1.84x, honest CR=3.76x)
- **qwen2.5-0.5b**: baseline PPL=4.518224439777543, best=adaptive_dense_hybrid (PPL ratio=1.81x, honest CR=3.18x)

## 2. Compression Accounting Fix
- ✅ Honest bit counting implemented
- ✅ Separate tracking of low-rank, residual, scale bits
- ✅ Sparse residual accounting with index overhead
- ✅ Dense INT4 residual capped at ~3-4x (correct)

## 3. Adaptive Rank vs Uniform Rank
- smollm-135m: uniform=11580022170.81x, adaptive=2.35x → adaptive better
- qwen2.5-0.5b: uniform=2602415873805.91x, adaptive=1.81x → adaptive better

## 4. Sparse vs Dense Residual (same bit budget)
- smollm-135m:
  Dense best PPL ratio: 2.35x (CR=2.8x)
  Sparse best CR: 2.4x (PPL ratio=21206.23x)
- qwen2.5-0.5b:
  Dense best PPL ratio: 1.81x (CR=3.2x)
  Sparse best CR: 2.8x (PPL ratio=4868.51x)

## 5. Hessian-Aware vs Magnitude Sparse Selection
- smollm-135m: magnitude=2946.25x, hessian=1989.65x → hessian better
- qwen2.5-0.5b: magnitude=2081.5x, hessian=1107.46x → hessian better

## 6. Sequential vs Independent Calibration
- smollm-135m: independent=2.35x, sequential=2946.25x
- qwen2.5-0.5b: independent=1.81x, sequential=2081.5x

## 7. Best Configuration Per Model
- **smollm-135m**: `scalar_q4` PPL=9.3947 (ratio=1.84x, CR=3.76x)
- **qwen2.5-0.5b**: `adaptive_dense_hybrid` PPL=8.1677 (ratio=1.81x, CR=3.18x)

## 8. Cross-Model Scaling Effect
- ✅ Effect generalizes across models

## 9. Verdict
🔴 **NEEDS IMPROVEMENT** — Post-training adaptive CHMC insufficient without local calibration or QAT

## 10. Next Steps for CHMC v3
- Block-wise reconstruction (compress entire transformer blocks)
- Quantization-Aware Training with covariance-aware initialization
- Mixed-precision residual quantization (INT4/INT2 per-block)
- Codebook-based vector quantization for residuals
- Learnable rank allocation via differentiable relaxation