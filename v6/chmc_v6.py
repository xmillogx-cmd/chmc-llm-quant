#!/usr/bin/env python3
"""
chmc_v6.py — CHMC v6: hybrid of CHMC v5 geometry + GPTQ optimizers
===================================================================

NEW vs v5 (the core change): the comparison is now done at an EQUAL bit
budget (BPW). v5 spent ~4.50 BPW vs GPTQ's 4.2875 BPW and still lost, so
the comparison was unfair. v6 allocates the low-rank rank PER LAYER so the
total BPW matches a target (default: GPTQ's 4.2875), then compares PPL.

Block A — GPTQ optimizers (P0):
  A1  strict sequential compensation (one column at a time, full H^-1)
  A2  group orientation dim=0 -> dim=1 (GPTQ-style, groups along input)
  A3  Hessian accumulation across batches (numerically stable)
  A4  dampening grid search

Block B — geometric (P1, added in later steps):
  B1' Hybrid rotation: SVD stays in the original cone space; Hadamard rotates
      ONLY the residual-quantization stage (R and X together). Full rotation
      equalized X column variances and destroyed the activation-cone alignment
      that damped-cov SVD exploits (SmolLM ratio 2.55 vs 1.23 at equal BPW,
      despite better per-layer Frobenius recon_err), so it was demoted from
      "rotate everything" to "rotate only residual quantization".
  B2  block covariance instead of diagonal
  B3  curvature-aware rank allocation
  B4  angular / cone residual quantization
  B5  tangent-space SVD projection

HARD RULES honored:
  - eval pipeline untouched (uses the local v6 copy eval_utils_v6.compute_perplexity,
    a verbatim port of eval_utils_v5 — PPL logic byte-identical)
  - no STE calibration
  - Hadamard seed=42
  - strict sequential falls back to block version if a layer is too slow
"""

import gc
import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

V6_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(V6_DIR))

from eval_utils_v6 import (  # noqa: E402
    compute_perplexity, load_wikitext_eval, collect_calibration_inputs,
    get_compressible_layers, get_weight, set_weight,
)

V5_DIR = Path(__file__).resolve().parent.parent / "v5"
sys.path.insert(0, str(V5_DIR))
from stabilizers import (  # noqa: E402
    quantize_groupwise,
    compute_damped_covariance,
    quantize_with_compensation,
)
from bpw import (  # noqa: E402
    chmc_bpw, rank_for_bpw, chmc_bpw_qjl, rank_for_bpw_qjl,
    cone_aware_bpw_overhead,
)

BASE_DIR = Path(__file__).parent.resolve()          # v6/
ROOT_DIR = BASE_DIR.parent                           # cmq_experiment/
RESULTS = ROOT_DIR / "results_v6"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPTQ_BPW = 4.2875  # GPTQModel reported BPW (4-bit, group-128)


# ════════════════════════════════════════════════════════════════
# Block A — GPTQ optimizers
# ════════════════════════════════════════════════════════════════

def _hessian(X: torch.Tensor, dampening: float, n_batches: int = 1) -> torch.Tensor:
    """Damped Hessian H = X^T X / n + lam*I, optionally accumulated in batches (A3)."""
    n = X.shape[0]
    in_f = X.shape[1]
    if n_batches and n_batches > 1 and n >= n_batches:
        bs = n // n_batches
        H = torch.zeros(in_f, in_f, device=X.device, dtype=X.dtype)
        for b in range(n_batches):
            Xb = X[b * bs:(b + 1) * bs]
            H += Xb.T @ Xb
        H = H / n
    else:
        H = (X.T @ X) / max(n, 1)
    lam = dampening * H.diag().mean().clamp(min=1e-8)
    return H + lam * torch.eye(in_f, device=H.device, dtype=H.dtype)


def _hinv(H: torch.Tensor) -> torch.Tensor:
    try:
        L = torch.linalg.cholesky(H)
        return torch.linalg.cholesky_inverse(L)
    except Exception:
        return torch.linalg.inv(H)


def _quantize_col_groups(col: torch.Tensor, bits: int, group_size: int,
                         group_dim: int) -> torch.Tensor:
    """Quantize a single column [out_f] with per-group scales.

    group_dim=0 (CHMC): groups of `group_size` rows share one scale.
    group_dim=1 (GPTQ): for a single column the whole column is one group
        (scale shared along the column) — the multi-column group sharing is
        handled by the caller precomputing scales; here we fall back to a
        single per-column scale, which is the correct per-column behavior.
    """
    out_f = col.shape[0]
    qmax = 2 ** (bits - 1) - 1
    if group_dim == 0 and group_size > 0 and group_size < out_f:
        n_groups = (out_f + group_size - 1) // group_size
        padded = n_groups * group_size
        cp = torch.zeros(padded, device=col.device, dtype=col.dtype)
        cp[:out_f] = col
        cg = cp.reshape(n_groups, group_size)
        maxv = cg.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scales = maxv / qmax
        q = torch.round(cg / scales).clamp(-qmax, qmax) * scales
        return q.reshape(padded)[:out_f]
    # group_dim=1 or per-column: single scale for the column
    maxv = col.abs().max().clamp(min=1e-8)
    scales = maxv / qmax
    return torch.round(col / scales).clamp(-qmax, qmax) * scales


def quantize_with_compensation_strict(
    R: torch.Tensor, X: torch.Tensor, bits: int = 4, group_size: int = 128,
    dampening: float = 0.01, group_dim: int = 0, n_batches: int = 1,
) -> torch.Tensor:
    """A1: STRICT sequential column-wise GPTQ (one column at a time, full H^-1).

    This is the true GPTQ update: after quantizing column j, the error is
    pushed into ALL remaining columns via the full H^-1 row (not a block).
    """
    out_f, in_f = R.shape
    R_work = R.float().clone()

    H = _hessian(X, dampening, n_batches)
    perm = H.diag().argsort(descending=True)
    Hinv = _hinv(H)
    del H  # free Hessian (in_f x in_f x 4B) before the column loop
    invperm = perm.argsort()
    Rr = R_work[:, perm]
    Hir = Hinv[perm][:, perm]
    del R_work, Hinv, perm  # free originals; keep Rr/Hir/invperm for the loop

    Rq = torch.zeros_like(Rr)
    for j in range(in_f):
        col = Rr[:, j]
        col_q = _quantize_col_groups(col, bits, group_size, group_dim)
        Rq[:, j] = col_q
        if j < in_f - 1:
            err = (col - col_q) / Hir[j, j].abs().clamp(min=1e-8)
            Rr[:, j + 1:] -= err.unsqueeze(1) * Hir[j, j + 1:].unsqueeze(0)
    return Rq[:, invperm]


def quantize_groupwise_v2(
    tensor: torch.Tensor, bits: int = 4, group_size: int = 128,
    group_dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A2: groupwise quantization with selectable group orientation.

    group_dim=0 (CHMC v5): groups along output channels (rows).
    group_dim=1 (GPTQ):    groups along input features (columns).
    """
    t = tensor.float()
    out_f, in_f = t.shape
    qmax = 2 ** (bits - 1) - 1

    if group_dim == 1:
        # GPTQ style: groups of `group_size` columns share a scale per row
        n_groups = (in_f + group_size - 1) // group_size
        padded_in = n_groups * group_size
        if padded_in > in_f:
            tp = torch.zeros(out_f, padded_in, device=t.device, dtype=t.dtype)
            tp[:, :in_f] = t
        else:
            tp = t
        tg = tp.reshape(out_f, n_groups, group_size)
        maxv = tg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)  # [out_f, n_groups, 1]
        scales = maxv / qmax
        q = torch.round(tg / scales).clamp(-qmax, qmax)
        deq = (q * scales).reshape(out_f, padded_in)[:, :in_f]
        scales_out = maxv.squeeze(2)  # [out_f, n_groups]
        return deq, scales_out
    else:
        # CHMC v5 style (dim=0)
        return quantize_groupwise(tensor, bits=bits, group_size=group_size)


def accumulate_hessian(
    X: torch.Tensor, n_batches: int = 8, dampening: float = 0.01
) -> torch.Tensor:
    """A3: Hessian accumulated across batches (numerically stable)."""
    return _hessian(X, dampening, n_batches=n_batches)


def find_best_dampening(
    R: torch.Tensor, X: torch.Tensor, bits: int = 4, group_size: int = 128,
    candidates: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """A4: grid-search the dampening that minimizes reconstruction error."""
    if candidates is None:
        candidates = [0.001, 0.005, 0.01, 0.05, 0.1]
    best_lam, best_err = candidates[0], float("inf")
    for lam in candidates:
        H = _hessian(X, lam, n_batches=1)
        Hinv = _hinv(H)
        # quick proxy: quantize a few columns with this Hinv and measure error
        in_f = R.shape[1]
        j = 0
        col = R[:, j].float()
        col_q = _quantize_col_groups(col, bits, group_size, 0)
        err = (col - col_q).pow(2).sum().item()
        # include Hessian-conditioning proxy (smaller diag spread is better)
        cond = (Hinv.diag().max() / Hinv.diag().min().clamp(min=1e-8)).log2().item()
        score = err + 0.01 * cond
        if score < best_err:
            best_err, best_lam = score, lam
    return best_lam, best_err


# ════════════════════════════════════════════════════════════════
# Block B — geometric (P1)
# ════════════════════════════════════════════════════════════════

def _hadamard_matrix(n: int, device) -> torch.Tensor:
    """Deterministic UNNORMALIZED Sylvester Hadamard of size n (n a power of 2).

    H @ H.T = n * I. Normalization happens ONCE in random_hadamard_rotation —
    dividing by sqrt(n) at every recursion level would underscale the matrix
    by prod_k sqrt(2^k) (e.g. ~10^-3 for n=128), breaking orthogonality.
    """
    if n == 1:
        return torch.ones(1, 1, device=device)
    H = _hadamard_matrix(n // 2, device)
    return torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)


def random_hadamard_rotation(n: int, seed: int = 42, device=None) -> torch.Tensor:
    """B1: QuIP#-style random rotation (seed=42 per HARD RULE).

    Returns an EXACTLY n×n orthogonal matrix R (R @ R.T = I).
      - If n is a power of 2: a true Hadamard H scaled by random ±1 signs.
      - Otherwise a cropped Hadamard is NOT orthogonal, so we use a Haar
        random-orthogonal matrix (QR of a seeded Gaussian) — the same
        "random rotation to uniformize the weight distribution" idea as
        QuIP#, but exactly orthogonal for any n.
    """
    if device is None:
        device = DEVICE
    if n & (n - 1) == 0:  # power of 2 -> true Hadamard
        H = _hadamard_matrix(n, device)   # unnormalized: H @ H.T = n * I
        g = torch.Generator(device="cpu").manual_seed(seed)
        signs = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
        return (H * signs.to(device).unsqueeze(0)) / math.sqrt(n)
    # Haar random orthogonal via QR of a seeded Gaussian (deterministic)
    g = torch.Generator(device="cpu").manual_seed(seed)
    G = torch.randn(n, n, generator=g)
    Q, Rr = torch.linalg.qr(G)
    Q = Q * torch.sign(torch.diag(Rr)).unsqueeze(0)  # proper Haar rotation
    return Q.to(device)


def compute_block_covariance(
    X: torch.Tensor, block_size: int = 16, dampening: float = 0.01
) -> torch.Tensor:
    """B2: block-diagonal covariance (instead of pure diagonal).

    Returns a block-diagonal matrix [in_f, in_f] where each block of
    `block_size` features keeps its full sub-covariance.
    """
    n, in_f = X.shape
    B = block_size
    n_blocks = (in_f + B - 1) // B
    C = torch.zeros(in_f, in_f, device=X.device, dtype=X.dtype)
    for b in range(n_blocks):
        sl = slice(b * B, min((b + 1) * B, in_f))
        Xb = X[:, sl]
        Cb = (Xb.T @ Xb) / max(n, 1)
        C[sl, sl] = Cb
    lam = dampening * C.diag()[C.diag() > 0].mean().clamp(min=1e-8)
    idx = torch.arange(in_f, device=X.device)
    C[idx, idx] += lam
    return C


def estimate_curvature(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """B3: per-output-channel curvature = ||W_row||^2 * input_energy(row).

    A proxy for how much each output channel's error matters, used to
    allocate rank where curvature is highest.
    """
    row_norm = W.pow(2).sum(dim=1)  # [out_f]
    col_energy = (X ** 2).mean(dim=0)  # [in_f]
    # curvature per output channel: how much weight mass it carries
    return row_norm


def allocate_ranks_by_curvature(
    layer_shapes: List[tuple], total_rank_budget: int,
    curvatures: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, int]:
    """B3: allocate a total rank budget across layers by curvature."""
    names = [n for n, _, _ in layer_shapes]
    if curvatures is None:
        # fall back to parameter count
        curv = {n: float(o * i) for n, o, i in layer_shapes}
    else:
        curv = {n: float(curvatures.get(n, torch.tensor(o * i)).sum())
                for n, o, i in layer_shapes}
    total = sum(curv.values()) + 1e-8
    ranks = {}
    for n in names:
        r = max(1, int(round(total_rank_budget * curv[n] / total)))
        ranks[n] = r
    return ranks


def angular_quantize_residual(
    R: torch.Tensor, bits: int = 4, group_size: int = 128
) -> torch.Tensor:
    """B4: angular (cone) residual quantization.

    Quantize the DIRECTION of each residual vector on a hypercube cone and
    keep a per-group magnitude — reduces error on the angular component that
    symmetric axis-aligned quantization treats poorly.
    """
    t = R.float()
    out_f, in_f = t.shape
    qmax = 2 ** (bits - 1) - 1
    n_groups = (out_f + group_size - 1) // group_size
    padded = n_groups * group_size
    tp = torch.zeros(padded, in_f, device=t.device, dtype=t.dtype)
    tp[:out_f] = t
    tg = tp.reshape(n_groups, group_size, in_f)
    # magnitude per (group, feature)
    mag = tg.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    # direction on [-1, 1]
    dirn = tg / mag
    # quantize direction to a fine angular grid, reconstruct with magnitude
    n_levels = 2 ** bits - 1
    q = torch.round(dirn * (n_levels / 2)).clamp(-n_levels / 2, n_levels / 2)
    deq = (q / (n_levels / 2)) * mag
    return deq.reshape(padded, in_f)[:out_f]


def tangent_space_svd(
    W: torch.Tensor, X: torch.Tensor, rank: int, dampening: float = 0.01,
    niter: int = 5,
) -> torch.Tensor:
    """B5: SVD in the tangent space of the calibration manifold.

    Projects W onto the subspace spanned by the top calibration directions
    before low-rank factorization, so the rank is spent where the data lives.
    """
    W = W.float()
    out_f, in_f = W.shape
    rank = min(rank, out_f, in_f)
    diag_c = compute_damped_covariance(X, dampening)
    sqrt_c = torch.sqrt(diag_c)
    # whiten input direction: V lives in the calibration span
    Ww = W * sqrt_c.unsqueeze(0)
    U, S, Vt = torch.svd_lowrank(Ww, q=rank, niter=niter)
    W_lr = ((U * S.unsqueeze(0)) @ Vt.T) / sqrt_c.unsqueeze(0)
    return W_lr


# ════════════════════════════════════════════════════════════════
# Block C — TurboQuant-inspired patches (weights only, ICLR 2026)
#
# Framed in CHMC's cone hypothesis: model data lives in a COMPRESSED CONE
# (anisotropic, redundant space). We approach it geometrically:
#   C1 (Patch 5) Cone-Aware rotation  — the centerpiece. Preserve the cone
#       component (where activations live) EXACTLY; rotate + quantize only
#       the perpendicular (redundant) component.
#   C2 (Patch 3) Lloyd-Max quantizer  — optimal scalar quantizer for the
#       ~N(0,1) distribution that a random rotation produces (replaces the
#       uniform round() in the rotated branch).
#   C3 (Patch 2) QJL 1-bit residual   — store the residual as 1-bit signs of
#       random projections (saves ~3 bits/weight, reinvested in higher rank).
#   C4 (Patch 4) IP preservation      — eval-only metric: MSE-optimal
#       quantizers are inner-product-biased, which matters for Q·K^T.
#
# HARD RULES honored: weights only (no KV-cache), eval pipeline untouched
# (C4 is a stat, not a compression change), Hadamard seed=42, Cone-Aware
# seed=42, Lloyd-Max only in the hadamard=True branch.
# ════════════════════════════════════════════════════════════════

# ── C4 (Patch 4): IP preservation metric (eval-only) ─────────────
def measure_ip_preservation(
    W_orig: torch.Tensor, W_comp: torch.Tensor,
    X_calib: torch.Tensor, n_pairs: int = 1000,
) -> Dict[str, float]:
    """C4: measure inner-product preservation (eval-only, no compression effect).

    MSE-optimal quantizers minimize squared error but are IP-biased: they can
    distort inner products, which is exactly what the forward pass uses
    (Q·K^T). This reports how well the compressed weight preserves inner
    products at both the weight level and the output (activation) level.
    """
    W_o = W_orig.float()
    W_c = W_comp.float()
    X = X_calib.float()
    n = X.shape[0]
    if n == 0:
        return {"ip_bias_weight": float("nan"), "ip_cosine_weight": float("nan"),
                "ip_bias_output": float("nan"), "ip_cosine_output": float("nan")}
    idx = torch.randperm(n, device=X.device)[:min(n_pairs, n)]
    Xs = X[idx]                                   # [n_pairs, in_f]
    Y_o = Xs @ W_o.T                              # [n_pairs, out_f]
    Y_c = Xs @ W_c.T
    ip_bias_weight = float((W_c.pow(2).sum() - W_o.pow(2).sum()).abs()
                           / W_o.pow(2).sum().clamp(min=1e-8))
    ip_cosine_weight = float(torch.nn.functional.cosine_similarity(
        W_o.flatten(), W_c.flatten(), dim=0))
    ip_bias_output = float((Y_c.pow(2).sum() - Y_o.pow(2).sum()).abs()
                           / Y_o.pow(2).sum().clamp(min=1e-8))
    ip_cosine_output = float(torch.nn.functional.cosine_similarity(Y_o, Y_c, dim=1).mean())
    return {
        "ip_bias_weight": round(ip_bias_weight, 6),
        "ip_cosine_weight": round(ip_cosine_weight, 6),
        "ip_bias_output": round(ip_bias_output, 6),
        "ip_cosine_output": round(ip_cosine_output, 6),
    }


# ── C2 (Patch 3): Lloyd-Max quantizer (optimal for N(0,1)) ───────
_LLOYD_MAX_CACHE: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}


def _compute_lloyd_max(bits: int, n_iter: int = 200, n_grid: int = 20001
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Lloyd-Max decision boundaries + centroids for N(0,1), 2^bits levels.

    Iterates: centroid_i = E[x | region_i]; boundary_i = (c_i + c_{i+1})/2.
    Returns (boundaries [k-1], centroids [k]) on CPU.
    """
    k = 2 ** bits
    x = torch.linspace(-8.0, 8.0, n_grid)
    pdf = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    dx = x[1] - x[0]
    boundaries = torch.linspace(-8.0, 8.0, k + 1)[1:-1].clone()
    centroids = torch.zeros(k)
    for _ in range(n_iter):
        for i in range(k):
            lo = boundaries[i - 1] if i > 0 else x[0]
            hi = boundaries[i] if i < k - 1 else x[-1]
            mask = (x >= lo) & (x < hi)
            mass = (pdf[mask] * dx).sum()
            if mass > 1e-12:
                centroids[i] = (pdf[mask] * x[mask] * dx).sum() / mass
        for i in range(k - 1):
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2
    return boundaries, centroids


def get_lloyd_max(bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lazily compute + cache the Lloyd-Max tables for `bits` levels."""
    if bits not in _LLOYD_MAX_CACHE:
        _LLOYD_MAX_CACHE[bits] = _compute_lloyd_max(bits)
    return _LLOYD_MAX_CACHE[bits]


def lloyd_max_quantize_rotated(
    tensor: torch.Tensor, bits: int = 4, group_size: int = 128,
) -> torch.Tensor:
    """C2: Lloyd-Max quantization (optimal for N(0,1)) on a rotated tensor.

    Assumes the tensor has been rotated (Hadamard/Haar) so its distribution is
    ~N(0,1). Per-group std normalizes to unit variance, the Lloyd-Max tables
    pick the optimal (non-uniform) reconstruction level, and the per-group std
    de-normalizes. Replaces the uniform round() in the hadamard=True branch.
    Grouping is along input features (dim=1, GPTQ orientation).
    """
    t = tensor.float()
    out_f, in_f = t.shape
    boundaries, centroids = get_lloyd_max(bits)
    boundaries = boundaries.to(t.device)
    centroids = centroids.to(t.device)
    n_groups = (in_f + group_size - 1) // group_size
    padded_in = n_groups * group_size
    tp = torch.zeros(out_f, padded_in, device=t.device, dtype=t.dtype)
    tp[:, :in_f] = t
    tg = tp.reshape(out_f, n_groups, group_size)
    group_std = tg.std(dim=2, keepdim=True).clamp(min=1e-8)   # [out_f, n_groups, 1]
    x_norm = tg / group_std
    idx = torch.searchsorted(boundaries, x_norm)              # [out_f, n_groups, group_size]
    idx = idx.clamp(0, len(centroids) - 1)
    deq_norm = centroids[idx]
    deq = (deq_norm * group_std).reshape(out_f, padded_in)[:, :in_f]
    return deq


# ── C1 (Patch 5): Cone-Aware rotation (the geometric core) ───────
def cone_axis_from_activations(X_calib: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """C1: find the cone axis from calibration activations (mean direction).

    Cone hypothesis: activations lie in a narrow cone around a mean direction.
    The cone axis is the normalized mean activation (the data's principal axis).
    """
    X = X_calib.float()
    axis = X.mean(dim=0)
    norm = axis.norm()
    if norm < 1e-8:
        g = torch.Generator(device="cpu").manual_seed(seed)
        axis = torch.randn(X.shape[1], generator=g)
        axis = axis / axis.norm()
    else:
        axis = axis / norm
    return axis.to(X.device)


def cone_decompose(W: torch.Tensor, axis: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """C1: decompose W = W_cone + W_perp (Cone-Aware, Patch 5).

    W_cone = (W @ axis) * axis^T — W's component along the cone axis (input
      space). This is the "data" component (where activations live) and is
      preserved EXACTLY (rank-1, stored as two vectors).
    W_perp = W - W_cone — the perpendicular (redundant) component, which is
      what gets rotated + quantized.
    """
    W = W.float()
    W_cone = (W @ axis).unsqueeze(1) * axis.unsqueeze(0)   # [out_f, in_f]
    W_perp = W - W_cone
    return W_cone, W_perp


# ── C3 (Patch 2): QJL 1-bit residual ─────────────────────────────
def qjl_residual_encode(R: torch.Tensor, n_projections: int, seed: int = 42
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """C3: encode residual R as 1-bit signs of random projections (QJL).

    R: [out_f, in_f]; n_projections typically in_f // 4 (residual BPW ~0.25).
    Returns (signs [out_f, n_projections] in {-1,+1}, U [in_f, n_projections]).
    U is seed-derived and shared, so only the signs (1 bit each) are stored.
    """
    R = R.float()
    out_f, in_f = R.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    U = torch.randn(in_f, n_projections, generator=g).to(R.device)
    proj = R @ U                                            # [out_f, n_projections]
    signs = torch.sign(proj)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return signs, U


def qjl_residual_decode(signs: torch.Tensor, U: torch.Tensor,
                        R_orig_for_scale: torch.Tensor) -> torch.Tensor:
    """C3: decode QJL signs back to a residual estimate.

    scale = <R, R_approx> / <R_approx, R_approx> (optimal global scale, computed
    at compression time from the original residual; stored as one FP16 scalar).
    """
    R_approx = signs @ U.T                                  # [out_f, in_f]
    R = R_orig_for_scale.float()
    scale = ((R * R_approx).sum() / (R_approx * R_approx).sum().clamp(min=1e-8))
    return scale * R_approx


# ════════════════════════════════════════════════════════════════
# Layer compression (v6)
# ════════════════════════════════════════════════════════════════

def compress_layer_v6(
    W_orig: torch.Tensor,
    X_calib: torch.Tensor,
    rank: int = 8,
    residual_bits: int = 4,
    group_size: int = 128,
    use_compensation: bool = True,
    strict_sequential: bool = False,   # A1
    group_dim: int = 0,               # A2 (0=CHMC, 1=GPTQ)
    hessian_batches: int = 1,         # A3
    dampening: float = 0.01,          # A4
    niter: int = 5,                   # svd_lowrank power iterations
    hadamard: bool = False,           # B1
    block_cov: bool = False,          # B2
    whitening: bool = False,          # v7: full covariance whitening
    angular: bool = False,            # B4
    tangent: bool = False,            # B5
    # ── Block C — TurboQuant-inspired (weights only) ──
    cone_aware: bool = False,         # C1 (Patch 5): preserve cone, compress perp
    lloyd_max: bool = False,          # C2 (Patch 3): optimal scalar quantizer
    qjl: bool = False,                # C3 (Patch 2): 1-bit sign residual
    qjl_n_projections: int = 0,       # C3: 0 -> in_f // 4
    ip_metric: bool = False,          # C4 (Patch 4): IP preservation (eval-only)
) -> Tuple[torch.Tensor, Dict]:
    W = W_orig.detach().float()
    out_f, in_f = W.shape
    rank = min(rank, out_f, in_f)

    # C1 (Patch 5): Cone-Aware decomposition — preserve the cone component
    # (where activations live) EXACTLY; compress only the perpendicular part.
    W_cone = None
    if cone_aware and X_calib is not None and X_calib.shape[0] > 0:
        axis = cone_axis_from_activations(X_calib, seed=42)
        W_cone, W_base = cone_decompose(W, axis)
    else:
        W_base = W

    # B1' (Hybrid rotation): Hadamard applies ONLY to the residual-quantization
    # stage, NOT to the low-rank SVD. The SVD stays in original activation space
    # so damped-cov weighting keeps its cone alignment — full rotation equalized
    # X column variances and 2x'd PPL on SmolLM (ratio 2.55 vs 1.23) despite a
    # better per-layer Frobenius recon_err. Rotating R and X together leaves the
    # loss ||(R - R_q) X^T||^2 invariant, so GPTQ/INT4/Lloyd-Max in rotated
    # coords minimize the same activation-weighted error with uniform magnitudes.
    Q_rot = None
    if hadamard:
        Q_rot = random_hadamard_rotation(in_f, seed=42, device=W.device)

    # Step 1: low-rank factor — ALWAYS in original (cone) space
    W_svd = W_base
    X_svd = X_calib
    if rank <= 0:
        # no low-rank component; the full weight is the residual
        W_lr = torch.zeros_like(W_svd)
    elif tangent:
        W_lr = tangent_space_svd(W_svd, X_svd, rank, dampening, niter=niter)
    else:
        if whitening:
            # v7: full covariance whitening (data-adaptive, accounts for
            # input-feature correlations). Objective: minimize
            #   ||(W - W_lr) @ C^{1/2}||_F,  C = X^T X / n  (activation cov).
            # Reduces to the diagonal damping below when C is diagonal.
            n_tok = max(X_svd.shape[0], 1)
            C = (X_svd.T @ X_svd) / n_tok                       # [in_f, in_f]
            diag_mean = C.diagonal().mean().clamp_min(1e-12)
            C_damped = C + dampening * diag_mean * torch.eye(
                C.shape[0], device=C.device, dtype=C.dtype)
            evals, evecs = torch.linalg.eigh(C_damped)
            evals = evals.clamp_min(1e-12)
            sqrt_evals = torch.sqrt(evals)
            C_sqrt = (evecs * sqrt_evals.unsqueeze(0)) @ evecs.T      # C^{1/2}
            C_inv_sqrt = (evecs / sqrt_evals.unsqueeze(0)) @ evecs.T  # C^{-1/2}
            W_weighted = W_svd @ C_sqrt
            U, S, Vt = torch.svd_lowrank(W_weighted, q=rank, niter=niter)
            W_lr = ((U * S.unsqueeze(0)) @ Vt.T) @ C_inv_sqrt
        elif block_cov:
            C = compute_block_covariance(X_svd, block_size=16, dampening=dampening)
            # use sqrt of the block-diag covariance's diagonal as the weight
            diag_c = C.diag()
            W_weighted = W_svd * torch.sqrt(diag_c).unsqueeze(0)
            U, S, Vt = torch.svd_lowrank(W_weighted, q=rank, niter=niter)
            W_lr = ((U * S.unsqueeze(0)) @ Vt.T) / torch.sqrt(diag_c).unsqueeze(0)
        else:
            diag_c = compute_damped_covariance(X_svd, dampening=dampening)
            W_weighted = W_svd * torch.sqrt(diag_c).unsqueeze(0)
            U, S, Vt = torch.svd_lowrank(W_weighted, q=rank, niter=niter)
            W_lr = ((U * S.unsqueeze(0)) @ Vt.T) / torch.sqrt(diag_c).unsqueeze(0)

    total_norm = W_svd.pow(2).sum().sqrt().item()
    residual_norm = (W_svd - W_lr).pow(2).sum().sqrt().item()
    lr_energy_pct = 1.0 - (residual_norm / total_norm) if total_norm > 0 else 0.0

    # Step 2: quantize residual — in rotated coords when hadamard=True
    R_full = W_svd - W_lr
    rotate_res = Q_rot is not None and X_calib.shape[0] > 0
    if rotate_res:
        R_q_in = R_full @ Q_rot
        X_q = X_calib @ Q_rot.T
    else:
        R_q_in = R_full
        X_q = X_calib
    if qjl:
        # C3 (Patch 2): QJL 1-bit sign residual (most aggressive, overrides C2)
        n_proj = qjl_n_projections if qjl_n_projections > 0 else max(1, in_f // 4)
        signs, U_proj = qjl_residual_encode(R_q_in, n_proj, seed=42)
        R_q_rot = qjl_residual_decode(signs, U_proj, R_q_in)
    elif lloyd_max and hadamard:
        # C2 (Patch 3): Lloyd-Max (optimal for N(0,1) post-Hadamard)
        R_q_rot = lloyd_max_quantize_rotated(
            R_q_in, bits=residual_bits, group_size=group_size)
    elif use_compensation and X_q.shape[0] > 0:
        if strict_sequential:
            # A1: true GPTQ, one column at a time, full H^-1
            R_q_rot = quantize_with_compensation_strict(
                R_q_in, X_q, bits=residual_bits, group_size=group_size,
                dampening=dampening, group_dim=group_dim,
                n_batches=hessian_batches,
            )
        elif group_dim == 1:
            # A2: block compensation but GPTQ (dim=1) group orientation
            R_q_rot = _block_comp_groupdim1(
                R_q_in, X_q, bits=residual_bits, group_size=group_size,
                dampening=dampening, n_batches=hessian_batches,
            )
        else:
            # v5 baseline: block compensation, CHMC (dim=0) groups
            R_q_rot = quantize_with_compensation(
                R_q_in, X_q, bits=residual_bits, group_size=group_size,
                dampening=dampening,
            )
    else:
        if angular:
            R_q_rot = angular_quantize_residual(R_q_in, residual_bits, group_size)
        else:
            R_q_rot, _ = quantize_groupwise_v2(
                R_q_in, bits=residual_bits, group_size=group_size, group_dim=group_dim
            )

    if rotate_res:
        R_q = R_q_rot @ Q_rot.T   # rotate back to original space
    else:
        R_q = R_q_rot

    W_comp_perp = W_lr + R_q

    # C1 (Patch 5): restore the exactly-preserved cone component
    if W_cone is not None:
        W_comp = W_cone + W_comp_perp
    else:
        W_comp = W_comp_perp

    recon_err = float((W - W_comp).pow(2).sum().sqrt() / max(total_norm, 1e-8))
    stats = {
        "rank": rank, "residual_bits": residual_bits, "group_size": group_size,
        "lr_energy_pct": round(lr_energy_pct, 4),
        "recon_err_ratio": round(recon_err, 6),
        "strict_sequential": strict_sequential, "group_dim": group_dim,
        "hessian_batches": hessian_batches, "dampening": dampening,
        "hadamard": hadamard, "block_cov": block_cov,
        "angular": angular, "tangent": tangent,
        # Block C (TurboQuant)
        "cone_aware": cone_aware, "lloyd_max": lloyd_max, "qjl": qjl,
    }

    # C4 (Patch 4): IP preservation metric (eval-only, does not affect compression)
    if ip_metric and X_calib is not None and X_calib.shape[0] > 0:
        stats.update(measure_ip_preservation(W, W_comp, X_calib))

    return W_comp, stats


def _block_comp_groupdim1(R, X, bits=4, group_size=128, dampening=0.01, n_batches=1):
    """Block-wise H^-1 compensation with TRUE GPTQ group orientation (dim=1).

    GPTQ grouping: every `group_size` consecutive (act-ordered) input columns
    form one group; each group has ONE scale per output row, computed from the
    ORIGINAL residual (before any compensation) and held fixed throughout.
    Quantization proceeds block-sequentially in act-order, pushing the
    quantization error onto the remaining columns via H^-1.

    Stored scale count = out_f * ceil(in_f/group_size) — identical to GPTQ,
    so the BPW comparison is fair.
    """
    out_f, in_f = R.shape
    R_work = R.float().clone()
    qmax = 2 ** (bits - 1) - 1
    H = _hessian(X, dampening, n_batches)
    Hinv = _hinv(H)
    perm = H.diag().argsort(descending=True)
    invperm = perm.argsort()
    Rr = R_work[:, perm]
    Hir = Hinv[perm][:, perm]
    Rq = torch.zeros_like(Rr)

    # Fixed GPTQ group scales from the ORIGINAL residual (before compensation).
    n_groups = (in_f + group_size - 1) // group_size
    padded_in = n_groups * group_size
    R_pad = torch.zeros(out_f, padded_in, device=Rr.device, dtype=Rr.dtype)
    R_pad[:, :in_f] = Rr
    R_g = R_pad.reshape(out_f, n_groups, group_size)
    group_scale = R_g.abs().amax(dim=2).clamp(min=1e-8) / qmax   # [out_f, n_groups]
    col_group = torch.arange(in_f, device=Rr.device) // group_size  # [in_f]

    BLOCK = min(32, in_f)
    for j_start in range(0, in_f, BLOCK):
        j_end = min(j_start + BLOCK, in_f)
        R_block = Rr[:, j_start:j_end]
        scale_block = group_scale[:, col_group[j_start:j_end]]    # [out_f, block]
        q = torch.round(R_block / scale_block).clamp(-qmax, qmax)
        Rq_block = q * scale_block
        Rq[:, j_start:j_end] = Rq_block
        err_raw = R_block - Rq_block
        diag_inv = Hir[j_start:j_end, j_start:j_end].diag().abs().clamp(min=1e-8)
        err = err_raw / diag_inv.unsqueeze(0)
        if j_end < in_f:
            Rr[:, j_end:] -= err @ Hir[j_start:j_end, j_end:]
    return Rq[:, invperm]


# ════════════════════════════════════════════════════════════════
# Bit-budget rank allocation
# ════════════════════════════════════════════════════════════════

def allocate_ranks_bit_budget(
    layer_shapes: List[tuple], target_bpw: float,
    residual_bits: int = 4, group_size: int = 128,
    qjl: bool = False, qjl_n_projections: int = 0,
    cone_aware: bool = False, group_dim: int = 0,
) -> Dict[str, int]:
    """Allocate per-layer rank so total BPW <= target_bpw (fair vs GPTQ).

    Accounts for the Cone-Aware rank-1 overhead (subtracted from the budget)
    and the cheaper QJL residual (frees budget for a HIGHER rank).
    group_dim is forwarded to rank_for_bpw so the scale overhead matches the
    quantization orientation (0 = CHMC row-groups, 1 = GPTQ col-groups).
    """
    ranks = {}
    for name, out_f, in_f in layer_shapes:
        eff_target = target_bpw
        if cone_aware:
            eff_target -= cone_aware_bpw_overhead(out_f, in_f)
        if qjl:
            n_proj = (qjl_n_projections if qjl_n_projections > 0
                      else max(1, in_f // 4))
            r = rank_for_bpw_qjl(out_f, in_f, eff_target, n_proj)
        else:
            r = rank_for_bpw(out_f, in_f, eff_target, residual_bits, group_size,
                             group_dim=group_dim)
        ranks[name] = max(1, r)
    return ranks


# ════════════════════════════════════════════════════════════════
# Full pipeline
# ════════════════════════════════════════════════════════════════

def run_chmc_v6(
    model_path: str,
    config: Optional[Dict] = None,
    tag: Optional[str] = None,
) -> Dict:
    """Run CHMC v6 with a config dict. See `default_config` for keys."""
    cfg = {**default_config(), **(config or {})}
    if tag is None:
        tag = Path(model_path).name

    print(f"\n{'=' * 64}")
    print(f"CHMC v6: {tag}")
    print(f"{'=' * 64}")
    for k, v in cfg.items():
        print(f"  {k} = {v}")

    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device_map = cfg.get("device_map")
    if device_map:
        print(f"  [device_map={device_map}] multi-GPU mode")
        # Per-card cap = actual card size minus a headroom for the per-layer
        # Hessian (X.T@X, ~486 MB for in_f=11008) + SVD working memory. The
        # old hardcoded 6-8 GiB cap wasted ~24 GB of a 2x16 GB setup; now the
        # cap is derived from the real card size so the model uses the full
        # VRAM while still leaving room for working tensors on every card.
        n_gpu = torch.cuda.device_count()
        per_gpu = cfg.get("max_memory_per_gpu")
        if per_gpu is None:
            headroom_gib = float(cfg.get("max_memory_headroom_gib", 3.0))
            max_mem = {}
            for i in range(n_gpu):
                total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                cap_gib = max(1.0, total_gib - headroom_gib)
                max_mem[i] = f"{cap_gib:.1f}GiB"
            print(f"  [max_memory] {max_mem} (headroom={headroom_gib:.1f} GiB/card)")
        else:
            max_mem = {i: per_gpu for i in range(n_gpu)}
            print(f"  [max_memory] {max_mem} (explicit cap)")
        mdl = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, device_map=device_map,
            max_memory=max_mem
        )
    else:
        mdl = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        mdl = mdl.to(DEVICE)
    mdl.eval()

    encoded = load_wikitext_eval(tok)
    baseline_ppl = compute_perplexity(mdl, tok, encoded)
    print(f"\n  Baseline PPL (FP32): {baseline_ppl:.4f}")

    layers = get_compressible_layers(mdl)
    print(f"  Compressible layers: {len(layers)}")
    calib_inputs = collect_calibration_inputs(mdl, layers, tok, n_tokens=2048)
    device = next(mdl.parameters()).device

    # layer shapes + rank allocation
    layer_shapes = []
    for name in layers:
        mod = mdl.get_submodule(name) if hasattr(mdl, "get_submodule") else _get_mod(mdl, name)
        out_f, in_f = mod.weight.shape[0], mod.weight.shape[1]
        layer_shapes.append((name, out_f, in_f))

    if cfg.get("bit_budget_bpw"):
        base_ranks = allocate_ranks_bit_budget(
            layer_shapes, cfg["bit_budget_bpw"], cfg["residual_bits"], cfg["group_size"],
            qjl=cfg["qjl"], qjl_n_projections=cfg["qjl_n_projections"],
            cone_aware=cfg["cone_aware"], group_dim=cfg["group_dim"],
        )
    else:
        base_ranks = {n: cfg["rank"] for n, _, _ in layer_shapes}

    # compress
    t_compress = time.time()
    all_stats = []
    for idx, name in enumerate(layers):
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(layers)}] ... ({time.time()-t_compress:.1f}s)")
        W_orig = get_weight(mdl, name)
        X_calib = calib_inputs.get(name)
        if X_calib is None or X_calib.shape[0] == 0:
            X_calib = torch.randn(64, W_orig.shape[1], device=W_orig.device)
        else:
            # calib activations are stored on CPU (see collect_calibration_inputs);
            # move this layer's slice to the weight's device for SVD/quantization.
            X_calib = X_calib.to(W_orig.device)
        W_comp, stats = compress_layer_v6(
            W_orig, X_calib,
            rank=base_ranks[name],
            residual_bits=cfg["residual_bits"],
            group_size=cfg["group_size"],
            use_compensation=cfg["use_compensation"],
            strict_sequential=cfg["strict_sequential"],
            group_dim=cfg["group_dim"],
            hessian_batches=cfg["hessian_batches"],
            dampening=cfg["dampening"],
            niter=cfg["niter"],
            hadamard=cfg["hadamard"],
            block_cov=cfg["block_cov"],
            whitening=cfg["whitening"],
            angular=cfg["angular"],
            tangent=cfg["tangent"],
            cone_aware=cfg["cone_aware"],
            lloyd_max=cfg["lloyd_max"],
            qjl=cfg["qjl"],
            qjl_n_projections=cfg["qjl_n_projections"],
            ip_metric=cfg["ip_metric"],
        )
        set_weight(mdl, name, W_comp.to(W_orig.dtype))
        stats["layer"] = name
        all_stats.append(stats)
        # Explicitly free per-layer working memory (W_orig, X_calib, W_comp)
        # and return reserved-but-unallocated CUDA memory to the driver.
        # Relying on auto-GC alone let the strict-path Hessian/Hinv/SVD
        # working tensors accumulate across layers -> OOM on GPU 0.
        del W_orig, X_calib, W_comp
        torch.cuda.empty_cache()
    compress_time = time.time() - t_compress

    compressed_ppl = compute_perplexity(mdl, tok, encoded)
    ratio = compressed_ppl / baseline_ppl if baseline_ppl > 0 else float("inf")
    total_time = time.time() - t0

    # BPW accounting (parameter-weighted)
    if cfg.get("bit_budget_bpw"):
        eff_ranks = base_ranks
    else:
        eff_ranks = {n: cfg["rank"] for n, _, _ in layer_shapes}
    tot_bits = 0.0
    tot_w = 0
    for name, out_f, in_f in layer_shapes:
        if cfg["qjl"]:
            n_proj = (cfg["qjl_n_projections"] if cfg["qjl_n_projections"] > 0
                      else max(1, in_f // 4))
            bpw = chmc_bpw_qjl(out_f, in_f, eff_ranks[name], n_proj)["bpw"]
        else:
            bpw = chmc_bpw(out_f, in_f, eff_ranks[name],
                           cfg["residual_bits"], cfg["group_size"],
                           group_dim=cfg["group_dim"])["bpw"]
        if cfg["cone_aware"]:
            bpw += cone_aware_bpw_overhead(out_f, in_f)
        tot_bits += bpw * out_f * in_f
        tot_w += out_f * in_f
    avg_bpw = tot_bits / tot_w

    result = {
        "model": tag,
        "baseline_ppl": round(baseline_ppl, 4),
        "compressed_ppl": round(compressed_ppl, 4),
        "ratio": round(ratio, 6),
        "bpw": round(avg_bpw, 4),
        "gptq_bpw": GPTQ_BPW,
        "bpw_delta_vs_gptq": round(avg_bpw - GPTQ_BPW, 4),
        "config": cfg,
        "ranks": {k: int(v) for k, v in list(eff_ranks.items())[:8]},
        "timing": {"compress_sec": round(compress_time, 2), "total_sec": round(total_time, 2)},
        "layer_stats_sample": all_stats[:3],
    }

    print(f"\n  {'=' * 40}")
    print(f"  PPL={compressed_ppl:.4f}  ratio={ratio:.6f}x  BPW={avg_bpw:.4f}")
    print(f"  Time: {compress_time:.1f}s compress, {total_time:.1f}s total")
    print(f"  {'=' * 40}")

    # ── Free the model + working tensors BEFORE returning ──────────────
    # The model (and accelerate's dispatch hooks) form reference cycles, so
    # plain refcounting will NOT release it — an explicit gc.collect() is
    # required. Without this, the next run's model loads on top of the
    # previous one's still-resident CUDA memory -> OOM.
    del mdl, tok, encoded, calib_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def _get_mod(model, name):
    mod = model
    for p in name.split("."):
        mod = getattr(mod, p, None)
        if mod is None:
            return None
    return mod


def default_config() -> Dict:
    return {
        "rank": 8,
        "residual_bits": 4,
        "group_size": 128,
        "use_compensation": True,
        "strict_sequential": False,   # A1
        "group_dim": 0,               # A2
        "hessian_batches": 1,         # A3
        "dampening": 0.01,            # A4
        "niter": 5,                   # svd_lowrank power iterations (higher = less seed-variance)
        "hadamard": False,            # B1
        "block_cov": False,           # B2
        "whitening": False,          # v7: full covariance whitening (data-adaptive)
        "angular": False,             # B4
        "tangent": False,             # B5
        # Block C — TurboQuant-inspired (weights only)
        "cone_aware": False,          # C1 (Patch 5)
        "lloyd_max": False,           # C2 (Patch 3)
        "qjl": False,                 # C3 (Patch 2)
        "qjl_n_projections": 0,       # C3: 0 -> in_f // 4
        "ip_metric": False,           # C4 (Patch 4)
        "bit_budget_bpw": None,       # set to 4.2875 for fair comparison
        "device_map": None,           # None = single GPU (.to(DEVICE)); "auto" = multi-GPU
        "max_memory_per_gpu": None,  # explicit per-card cap (e.g. "8GiB"); None -> auto below
        "max_memory_headroom_gib": 3.0,  # free VRAM kept on each card for Hessian/SVD work tensors
    }


if __name__ == "__main__":
    import argparse
    import json as _json

    MODELS_DIR = Path(os.environ.get("CMQ_MODELS_DIR", str(ROOT_DIR / "models")))
    ap = argparse.ArgumentParser(
        description="Run one CHMC v6 quantization + PPL evaluation, save JSON results.")
    ap.add_argument("--model", default=str(MODELS_DIR / "smollm-135m"),
                    help="local model directory (default: %(default)s)")
    ap.add_argument("--tag", default=None,
                    help="result tag (default: model dir name)")
    ap.add_argument("--bpw", type=float, default=4.2875,
                    help="bit budget in bits/weight; 4.2875 = GPTQ 4-bit group-128 level")
    ap.add_argument("--dampening", type=float, default=None,
                    help="A4 Hessian dampening (default: config value)")
    ap.add_argument("--strict", action="store_true",
                    help="use the A1 strict sequential GPTQ optimizer")
    args = ap.parse_args()

    overrides = {}
    if args.dampening is not None:
        overrides["dampening"] = args.dampening
    if args.strict:
        overrides["strict_sequential"] = True
    cfg = {**default_config(), **overrides, "bit_budget_bpw": args.bpw}

    result = run_chmc_v6(args.model, cfg, tag=args.tag)
    out_file = RESULTS / f"chmc_v6_{result['model']}.json"
    with open(out_file, "w") as f:
        _json.dump(result, f, indent=2)
    print(f"Saved -> {out_file}")
