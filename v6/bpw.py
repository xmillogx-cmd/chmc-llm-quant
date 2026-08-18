#!/usr/bin/env python3
"""
bpw.py — Exact bits-per-weight (BPW) accounting for CHMC vs GPTQ
================================================================

The v5 comparison was UNFAIR: CHMC v5 used ~4.5 BPW (low-rank overhead)
while GPTQModel used 4.2875 BPW, yet CHMC had a WORSE PPL ratio. This module
makes the bit budget explicit so CHMC and GPTQ are compared at EQUAL BPW.

Storage layout (what actually gets stored, in bits):
  CHMC:  lowrank  = 16 * rank * (out_f + in_f)          [U,V in FP16]
         residual = residual_bits * out_f * in_f         [INT4/INT8 indices]
         scales   = 16 * n_groups * in_f                 [one scale per (group, col)]
  GPTQ:  residual = bits * out_f * in_f
         scales   = 16 * out_f * (in_f / group_size)     [one scale per (row, group)]
         (GPTQModel reports 4.2875 BPW total for 4-bit / group-128)

NOTE: scale overhead DEPENDS on group orientation (the ceil() makes the two
      differ whenever out_f and in_f are not both multiples of group_size):
        group_dim=0 (CHMC): n_groups = ceil(out_f/gs), scales = 16*n_groups*in_f
        group_dim=1 (GPTQ): n_groups = ceil(in_f/gs),  scales = 16*out_f*n_groups
      They coincide only when out_f and in_f are both multiples of group_size.
      The only free parameter is the low-rank rank.
"""

import math
from typing import Dict, List


def chmc_bpw(out_f: int, in_f: int, rank: int,
             residual_bits: int = 4, group_size: int = 128,
             factor_bits: int = 16, group_dim: int = 0) -> Dict[str, float]:
    """Exact BPW for one CHMC layer (lowrank + residual + scales).

    Scale count depends on group orientation:
      group_dim=0 (CHMC): groups along output rows -> ceil(out_f/gs) groups, each with in_f scales
      group_dim=1 (GPTQ): groups along input cols  -> ceil(in_f/gs) groups, each with out_f scales
    """
    if group_dim == 0:
        n_groups = math.ceil(out_f / group_size)
        scales = factor_bits * n_groups * in_f
    else:
        n_groups = math.ceil(in_f / group_size)
        scales = factor_bits * out_f * n_groups
    lowrank = factor_bits * rank * (out_f + in_f)
    residual = residual_bits * out_f * in_f
    total = lowrank + residual + scales
    n_w = out_f * in_f
    return {
        "out_f": out_f, "in_f": in_f, "rank": rank, "group_dim": group_dim,
        "lowrank_bpw": lowrank / n_w,
        "residual_bpw": residual / n_w,
        "scale_bpw": scales / n_w,
        "bpw": total / n_w,
    }


def gptq_bpw(out_f: int, in_f: int, bits: int = 4, group_size: int = 128) -> float:
    """GPTQ BPW (residual + scales). GPTQModel reports 4.2875 for 4-bit/128."""
    n_groups = math.ceil(in_f / group_size)
    residual = bits * out_f * in_f
    scales = 16.0 * out_f * n_groups
    n_w = out_f * in_f
    return (residual + scales) / n_w


def rank_for_bpw(out_f: int, in_f: int, target_bpw: float,
                 residual_bits: int = 4, group_size: int = 128,
                 factor_bits: int = 16, group_dim: int = 0) -> int:
    """Largest integer rank whose CHMC BPW <= target_bpw (bit-budget match).

    group_dim must match the quantization's group orientation (0=CHMC rows,
    1=GPTQ cols) so the fixed scale overhead is counted correctly.
    """
    if group_dim == 0:
        n_groups = math.ceil(out_f / group_size)
        scales = factor_bits * n_groups * in_f
    else:
        n_groups = math.ceil(in_f / group_size)
        scales = factor_bits * out_f * n_groups
    fixed = residual_bits * out_f * in_f + scales
    budget = target_bpw * out_f * in_f
    if budget <= fixed:
        return 0
    rank = (budget - fixed) / (factor_bits * (out_f + in_f))
    return max(0, int(math.floor(rank + 1e-9)))


def chmc_bpw_qjl(out_f: int, in_f: int, rank: int,
                 n_projections: int, factor_bits: int = 16) -> Dict[str, float]:
    """Exact BPW for one CHMC layer with a QJL (1-bit sign) residual.

    Storage layout (what actually gets stored, in bits):
      lowrank  = 16 * rank * (out_f + in_f)          [U,V in FP16]
      residual = 1  * out_f * n_projections          [1-bit signs of random projections]
      scale    = 16                                  [one FP16 scale per matrix]
    The QJL residual replaces the INT4/INT8 groupwise residual (and its
    per-group scales), so no group-scale term is counted here.
    """
    lowrank = factor_bits * rank * (out_f + in_f)
    residual = 1 * out_f * n_projections
    scale = factor_bits
    total = lowrank + residual + scale
    n_w = out_f * in_f
    return {
        "out_f": out_f, "in_f": in_f, "rank": rank,
        "n_projections": n_projections,
        "lowrank_bpw": lowrank / n_w,
        "residual_bpw": residual / n_w,
        "scale_bpw": scale / n_w,
        "bpw": total / n_w,
    }


def cone_aware_bpw_overhead(out_f: int, in_f: int,
                            factor_bits: int = 16) -> float:
    """BPW overhead for storing the Cone-Aware cone component (rank-1).

    The cone component W_cone = (W @ axis) * axis^T is rank-1, so it is stored
    as two vectors: axis [in_f] and W@axis [out_f], each in FP16.
      cone_bits = 16 * (out_f + in_f)
    This is added on top of the CHMC BPW of the perpendicular component.
    """
    cone_bits = factor_bits * (out_f + in_f)
    n_w = out_f * in_f
    return cone_bits / n_w


def rank_for_bpw_qjl(out_f: int, in_f: int, target_bpw: float,
                     n_projections: int, factor_bits: int = 16) -> int:
    """Largest integer rank whose QJL-CHMC BPW <= target_bpw.

    The QJL residual is much cheaper than INT4 (1-bit signs), so the freed
    budget is reinvested in a HIGHER low-rank rank (the whole point of Patch 2).
    """
    residual = 1 * out_f * n_projections
    scale = factor_bits
    fixed = residual + scale
    budget = target_bpw * out_f * in_f
    if budget <= fixed:
        return 0
    rank = (budget - fixed) / (factor_bits * (out_f + in_f))
    return max(0, int(math.floor(rank + 1e-9)))


def model_bpw(layer_shapes: List[tuple], rank: int,
              residual_bits: int = 4, group_size: int = 128) -> Dict[str, float]:
    """Parameter-weighted average BPW across all layers of a model."""
    tot_bits = 0.0
    tot_w = 0
    per = {}
    for name, out_f, in_f in layer_shapes:
        d = chmc_bpw(out_f, in_f, rank, residual_bits, group_size)
        per[name] = d["bpw"]
        tot_bits += d["bpw"] * out_f * in_f
        tot_w += out_f * in_f
    return {"avg_bpw": tot_bits / tot_w, "per_layer": per}


# ── SmolLM-135M concrete shapes ─────────────────────────────────
SMOLLM_LAYERS = [
    ("q_proj", 576, 576),
    ("k_proj", 576, 192),
    ("v_proj", 576, 192),
    ("o_proj", 576, 576),
    ("gate_proj", 1536, 576),
    ("up_proj", 1536, 576),
    ("down_proj", 576, 1536),
]


if __name__ == "__main__":
    GPTQ_BPW = 4.2875  # from GPTQModel log: "Estimated Quantization BPW: 4.2875"

    print("=" * 64)
    print("BPW UNFAIRNESS ANALYSIS — SmolLM-135M")
    print("=" * 64)

    # 1) CHMC v5 actual BPW (rank 8, INT4, group 128)
    v5 = model_bpw(SMOLLM_LAYERS, rank=8, residual_bits=4, group_size=128)
    print(f"\n[CHMC v5] rank=8, INT4, group=128")
    print(f"  Parameter-weighted avg BPW = {v5['avg_bpw']:.4f}")
    for name, bpw in v5["per_layer"].items():
        print(f"    {name:<10} {bpw:.4f} BPW")

    # 2) GPTQ BPW
    g = gptq_bpw(576, 576, bits=4, group_size=128)
    print(f"\n[GPTQ] 4-bit, group=128")
    print(f"  Formula BPW (residual+scales) = {g:.4f}")
    print(f"  GPTQModel REPORTED BPW        = {GPTQ_BPW:.4f}")

    # 3) Unfairness
    print(f"\n[UNFAIRNESS]")
    print(f"  CHMC v5 uses {v5['avg_bpw']:.4f} BPW  vs  GPTQ {GPTQ_BPW:.4f} BPW")
    print(f"  CHMC spends +{v5['avg_bpw'] - GPTQ_BPW:.4f} BPW "
          f"(+{(v5['avg_bpw'] / GPTQ_BPW - 1) * 100:.2f}% more bits) "
          f"and STILL loses on PPL ratio.")

    # 4) Bit-budget match: rank per layer so CHMC BPW == GPTQ BPW
    print(f"\n[BIT-BUDGET MATCH] rank per layer so CHMC BPW <= {GPTQ_BPW}")
    matched = []
    for name, out_f, in_f in SMOLLM_LAYERS:
        r = rank_for_bpw(out_f, in_f, GPTQ_BPW, residual_bits=4, group_size=128)
        d = chmc_bpw(out_f, in_f, r, residual_bits=4, group_size=128)
        matched.append((name, out_f, in_f, r, d["bpw"]))
        print(f"    {name:<10} {out_f}x{in_f:<5} -> rank={r}  (BPW={d['bpw']:.4f})")

    # weighted avg of matched
    tb = sum(bpw * o * i for _, o, i, r, bpw in matched)
    tw = sum(o * i for _, o, i, r, bpw in matched)
    print(f"    Matched avg BPW = {tb / tw:.4f}  (target {GPTQ_BPW})")
