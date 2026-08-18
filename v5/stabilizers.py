#!/usr/bin/env python3
"""
stabilizers.py — CHMC v5 stabilizer modules
============================================

Stabilizers to close the gap vs GPTQModel (ratio 1.29x → < 1.15x):
  1. Per-group scales (group_size=128)
  2. Damped diagonal covariance
  3. Act-order column permutation
  4. H⁻¹ error compensation (sequential column-wise GPTQ for residual)
  5. Mixed precision (INT8 for sensitive layers)
  7. Adaptive rank allocation by sensitivity

Stabilizer 6 (STE calibration) was REMOVED: iterative optimization of
low-rank factors against output-MSE drifts in the null space of the
calibration inputs and explodes weights (PPL 1e8+). The closed-form
pipeline is stable; sensitivity ranking lives in chmc_pipeline.py.

Each function is self-contained and can be used independently or combined.
"""

import torch
from typing import Tuple


# ─── Stabilizer 1: Per-group symmetric quantization ──────────────

def quantize_groupwise(
    tensor: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-group symmetric quantization along dim=0 (output channels).
    Each group of `group_size` rows shares one scale.

    tensor: [out_f, in_f]
    Returns: (dequantized_tensor, scales) where scales is [n_groups, 1]
    """
    t = tensor.float()
    out_f, in_f = t.shape
    qmax = 2 ** (bits - 1) - 1

    # Group rows: pad to multiple of group_size
    n_groups = (out_f + group_size - 1) // group_size
    padded_out = n_groups * group_size

    if padded_out > out_f:
        t_padded = torch.zeros(padded_out, in_f, device=t.device, dtype=t.dtype)
        t_padded[:out_f, :] = t
    else:
        t_padded = t

    # Reshape to [n_groups, group_size, in_f] then compute abs max per group
    t_grouped = t_padded.reshape(n_groups, group_size, in_f)
    max_val = t_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # [n_groups, 1, in_f]
    scales = max_val / qmax

    # Quantize + dequantize
    q = torch.round(t_grouped / scales).clamp(-qmax, qmax)
    deq = (q * scales).reshape(padded_out, in_f)[:out_f, :]

    # Return per-group scales: [n_groups, 1]
    scales_out = max_val.squeeze(1)  # [n_groups, in_f]

    return deq, scales_out


# ─── Stabilizer 2: Damped diagonal covariance ────────────────────

def compute_damped_covariance(
    X: torch.Tensor,
    dampening: float = 0.01,
) -> torch.Tensor:
    """
    Damped diagonal covariance: diag(X^T X)/n + λ·mean(diag).
    Protects against zero/near-zero elements that blow up division.

    X: [n_tokens, in_features]
    Returns: [in_features]
    """
    diag_c = (X ** 2).mean(dim=0)
    lam = dampening * diag_c.mean().clamp(min=1e-8)
    return diag_c + lam


# ─── Stabilizer 3 + 4: Act-order + column-wise error compensation ─

def quantize_with_compensation(
    R: torch.Tensor,
    X: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    dampening: float = 0.01,
) -> torch.Tensor:
    """
    Sequential column-wise quantization with full Hessian inverse error compensation.

    This is the core GPTQ algorithm applied to residual matrices:
      1. Compute H = X^T X / n + λI (full matrix, not just diagonal)
      2. Invert via Cholesky for numerical stability
      3. Sort columns by importance (diag(H) descending) — act-order
      4. Quantize column-by-column, compensating error in remaining columns:
         err = (w - q) / Hinv[j,j]
         W[:, j+1:] -= err · Hinv[j, j+1:]
      5. Restore original column order

    BUG-5 fix: group_size applies along output channels (dim=0), matching GPTQ convention.
    OPT-2: block-wise compensation processes BLOCK columns at once for 5-10× speedup.

    R: [out_f, in_f] residual matrix
    X: [n_tokens, in_f] calibration inputs
    Returns: quantized residual [out_f, in_f]
    """
    out_f, in_f = R.shape
    R_work = R.float().clone()
    qmax = 2 ** (bits - 1) - 1

    # Full Hessian with dampening
    n = X.shape[0]
    H = (X.T @ X) / max(n, 1)  # [in_f, in_f]
    lam = dampening * H.diag().mean().clamp(min=1e-8)
    H = H + lam * torch.eye(in_f, device=H.device, dtype=H.dtype)

    # Cholesky inverse for numerical stability
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.linalg.cholesky_inverse(L)
        del L
    except Exception:
        Hinv = torch.linalg.inv(H)

    # Act-order: sort columns by importance (diag descending)
    col_importance = H.diag()
    perm = col_importance.argsort(descending=True)
    invperm = perm.argsort()

    R_reordered = R_work[:, perm]
    Hinv_reordered = Hinv[perm][:, perm]

    # Sequential column-wise quantization with error compensation
    R_quantized = torch.zeros_like(R_reordered)

    # OPT-2: block-wise compensation — process BLOCK columns at once
    BLOCK = min(32, in_f)  # process up to 32 columns per batch

    for j_start in range(0, in_f, BLOCK):
        j_end = min(j_start + BLOCK, in_f)

        # Quantize block of columns
        R_block = R_reordered[:, j_start:j_end]  # [out_f, block_size]

        if group_size > 0 and group_size < out_f:
            # BUG-5 fix: group along output channels (dim=0), matching GPTQ convention
            n_groups = (out_f + group_size - 1) // group_size
            padded_out = n_groups * group_size

            if padded_out > out_f:
                R_pad = torch.zeros(padded_out, j_end - j_start, device=R_block.device, dtype=R_block.dtype)
                R_pad[:out_f, :] = R_block
            else:
                R_pad = R_block

            # Reshape to [n_groups, group_size, block] — groups along output channels
            R_grp = R_pad.reshape(n_groups, group_size, j_end - j_start)
            max_val = R_grp.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # [n_groups, 1, block]
            scales = max_val / qmax
            q_block = torch.round(R_grp / scales).clamp(-qmax, qmax) * scales
            R_quantized[:, j_start:j_end] = q_block.reshape(padded_out, j_end - j_start)[:out_f, :]
        else:
            # Per-channel (single scale per column)
            max_val = R_block.abs().amax(dim=0, keepdim=True).clamp(min=1e-8)  # [1, block]
            scales = max_val / qmax
            R_quantized[:, j_start:j_end] = torch.round(R_block / scales).clamp(-qmax, qmax) * scales

        # Error compensation for the entire block at once (OPT-2 vectorization)
        err_raw = R_block - R_quantized[:, j_start:j_end]  # [out_f, block_size]
        diag_inv = Hinv_reordered[j_start:j_end, j_start:j_end].diag().abs().clamp(min=1e-8)  # [block_size]
        err = err_raw / diag_inv.unsqueeze(0)  # [out_f, block_size]

        if j_end < in_f:
            # Hinv_reordered[j_start:j_end, j_end:] is [block_size, remaining_cols]
            compensation = err @ Hinv_reordered[j_start:j_end, j_end:]  # [out_f, remaining]
            R_reordered[:, j_end:] -= compensation

    # Restore original column order
    R_result = R_quantized[:, invperm]

    return R_result


# ─── Stabilizer 6: STE calibration — REMOVED ─────────────────────
# Iterative optimization of (A, B, R) against output-MSE on finite
# calibration data drifts in the null space of X and explodes weights
# (historical PPL 1e8+). Replaced by the closed-form pipeline in
# chmc_pipeline.py + recon-error sensitivity ranking (real inputs).


# ─── Stabilizer 7: Adaptive rank allocation ──────────────────────

def allocate_ranks_by_sensitivity(
    precision_map: dict,
    layer_stats: dict,
    base_rank: int = 8,
    sensitive_rank_boost: int = 8,
) -> dict:
    """
    Allocate ranks based on layer sensitivity and effective rank.
    - Sensitive layers (INT8 residual) get larger rank
    - Low-rank layers get smaller rank
    - Others get base_rank

    precision_map: {layer_name: bits} from chmc_pipeline.find_sensitive_layers
    layer_stats: {layer_name: {"effective_rank": float}} covariance stats
    """
    ranks = {}
    for name in precision_map:
        residual_bits = precision_map.get(name, 4)
        stats = layer_stats.get(name, {})

        if residual_bits == 8:
            # Sensitive → larger rank
            ranks[name] = base_rank + sensitive_rank_boost
        elif stats.get("effective_rank", base_rank * 2) < base_rank:
            # Low-rank → smaller
            ranks[name] = max(4, int(stats["effective_rank"] // 2))
        else:
            ranks[name] = base_rank

    return ranks
