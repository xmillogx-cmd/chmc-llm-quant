#!/usr/bin/env python3
"""
chmc_pipeline.py — Единый пайплайн CHMC v5 (единая точка правды)
==================================================================

Все раннеры (run_chmc_v5.py, run_benchmark.py, run_comparison_3models.py)
используют этот модуль вместо собственных копий пайплайна.

Пайплайн (чисто алгебраический, без итеративной оптимизации):
  1. Weighted SVD с демпфированной ковариансой → low-rank факторизация
  2. Residual = W - W_lr
  3. Groupwise квантизация residual (INT4/INT8, group_size=128),
     опционально с H⁻¹ error compensation (GPTQ-style sequential columns)
  4. Mixed precision: top-k чувствительных слоёв (по recon error на
     РЕАЛЬНЫХ calibration-входах) получают INT8 residual + rank boost

Историческая заметка: STE-калибровка (итеративная оптимизация A, B, R
против output-MSE) удалена — параметры дрейфуют в нулевом подпространстве
кал. данных и взрывают веса (PPL 1e8+). Закрытая форма стабильнее.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval_utils_v5 import (
    compute_perplexity, load_wikitext_eval, collect_calibration_inputs,
    get_compressible_layers, get_weight, set_weight,
)
from stabilizers import (
    quantize_groupwise,
    compute_damped_covariance,
    quantize_with_compensation,
)

BASE_DIR = Path(__file__).parent.resolve()   # v5/
ROOT_DIR = BASE_DIR.parent                   # cmq_experiment/
RESULTS  = ROOT_DIR / "results_v5"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Слой ─────────────────────────────────────────────────────────

def compress_layer_chmc(
    W_orig: torch.Tensor,
    X_calib: torch.Tensor,
    rank: int = 8,
    residual_bits: int = 4,
    group_size: int = 128,
    use_compensation: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compress one layer: weighted SVD + groupwise quantized residual.

    Returns: (compressed_weight, stats_dict)
    """
    W = W_orig.detach().float()
    out_f, in_f = W.shape
    rank = min(rank, out_f, in_f)

    # Step 1: Weighted SVD with damped covariance
    diag_c = compute_damped_covariance(X_calib, dampening=0.01)
    W_weighted = W * torch.sqrt(diag_c).unsqueeze(0)

    U, S, Vt = torch.svd_lowrank(W_weighted, q=rank, niter=5)
    # torch.svd_lowrank returns V [in_f, rank], not Vt
    W_lr = ((U * S.unsqueeze(0)) @ Vt.T) / torch.sqrt(diag_c).unsqueeze(0)

    total_norm = W.pow(2).sum().sqrt().item()
    residual_norm = (W - W_lr).pow(2).sum().sqrt().item()
    lr_energy_pct = 1.0 - (residual_norm / total_norm) if total_norm > 0 else 0.0

    # Step 2: Quantize residual
    R_full = W - W_lr
    if use_compensation and X_calib.shape[0] > 0:
        R_q = quantize_with_compensation(
            R_full, X_calib, bits=residual_bits, group_size=group_size
        )
    else:
        R_q, _ = quantize_groupwise(R_full, bits=residual_bits, group_size=group_size)

    W_comp = W_lr + R_q
    recon_err = float((W - W_comp).pow(2).sum().sqrt() / max(total_norm, 1e-8))

    stats = {
        "rank": rank,
        "residual_bits": residual_bits,
        "group_size": group_size,
        "lr_energy_pct": round(lr_energy_pct, 4),
        "recon_err_ratio": round(recon_err, 6),
        "use_compensation": use_compensation,
    }
    return W_comp, stats


# ── Чувствительность слоёв ───────────────────────────────────────

def find_sensitive_layers(
    layers: List[str],
    calib_inputs: Dict[str, torch.Tensor],
    mdl,
    device,
    top_k: int = 5,
    rank_test: int = 4,
) -> Set[str]:
    """
    Top-k sensitive layers by reconstruction error on REAL calibration inputs.

    Cheaper and deterministic vs per-layer PPL forward passes: one closed-form
    compression per layer, ranked by relative reconstruction error.
    """
    sensitivities = {}
    for name in layers:
        W_orig = get_weight(mdl, name).detach().float()
        X_calib = calib_inputs.get(name)
        if X_calib is None or X_calib.shape[0] == 0:
            X_calib = torch.randn(64, W_orig.shape[1], device=device)
        _, stats = compress_layer_chmc(W_orig, X_calib, rank=rank_test, residual_bits=4)
        sensitivities[name] = stats["recon_err_ratio"]

    sorted_layers = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    return {n for n, _ in sorted_layers[:top_k]}


# ── Полный пайплайн ──────────────────────────────────────────────

def run_chmc_v5(
    model_path: str,
    rank: int = 8,
    residual_bits: int = 4,
    group_size: int = 128,
    use_compensation: bool = True,
    mixed_precision: bool = False,
    adaptive_rank: bool = False,
    top_k_sensitive: int = 5,
) -> Dict:
    """
    Run the full CHMC v5 pipeline on a model.

    Args:
        model_path: path to model directory
        rank: base low-rank for SVD compression
        residual_bits: INT bits for residual (4 or 8 for sensitive layers)
        group_size: per-group quantization size (128 = GPTQModel default)
        use_compensation: H⁻¹ error compensation on residual
        mixed_precision: INT8 for top-k sensitive layers
        adaptive_rank: sensitive layers get rank + 4
        top_k_sensitive: how many layers get extra precision

    Returns: result dict with PPL, timing, per-layer stats
    """
    tag = Path(model_path).name
    print(f"\n{'=' * 60}")
    print(f"CHMC v5: {tag}")
    print(f"{'=' * 60}")
    print(f"  rank={rank}, residual_bits={residual_bits}, group_size={group_size}")
    print(f"  compensation={use_compensation}")
    print(f"  mixed_precision={mixed_precision}, adaptive_rank={adaptive_rank}")

    t0 = time.time()

    # Load model
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    )
    mdl = mdl.to(DEVICE)
    mdl.eval()

    # Baseline PPL
    encoded = load_wikitext_eval(tok)
    baseline_ppl = compute_perplexity(mdl, tok, encoded)
    print(f"\n  Baseline PPL (FP32): {baseline_ppl:.4f}")

    # Layers + calibration inputs (train split, single forward pass)
    layers = get_compressible_layers(mdl)
    print(f"  Compressible layers: {len(layers)}")
    calib_inputs = collect_calibration_inputs(mdl, layers, tok, n_tokens=2048)
    device = next(mdl.parameters()).device

    # Sensitive layers (real inputs, recon-error ranking)
    if mixed_precision:
        print("\n  [S5] Identifying sensitive layers (recon error, real inputs)...")
        sensitive_names = find_sensitive_layers(
            layers, calib_inputs, mdl, device, top_k=top_k_sensitive
        )
        print(f"    Sensitive layers (INT8): {sorted(sensitive_names)}")
    else:
        sensitive_names = set()

    # Adaptive rank: sensitive → rank + 4
    if adaptive_rank and mixed_precision:
        layer_ranks = {n: (rank + 4 if n in sensitive_names else rank) for n in layers}
    else:
        layer_ranks = {n: rank for n in layers}

    # Compress each layer
    t_compress = time.time()
    all_stats = []

    for idx, name in enumerate(layers):
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t_compress
            print(f"  [{idx+1}/{len(layers)}] Compressing... ({elapsed:.1f}s)")

        W_orig = get_weight(mdl, name)
        X_calib = calib_inputs.get(name)
        if X_calib is None or X_calib.shape[0] == 0:
            X_calib = torch.randn(64, W_orig.shape[1], device=device)
        r_bits = residual_bits if name not in sensitive_names else 8
        r_rank = layer_ranks[name]

        W_comp, stats = compress_layer_chmc(
            W_orig, X_calib, rank=r_rank,
            residual_bits=r_bits, group_size=group_size,
            use_compensation=use_compensation,
        )
        set_weight(mdl, name, W_comp.to(W_orig.dtype))
        stats["layer"] = name
        all_stats.append(stats)

    compress_time = time.time() - t_compress

    # PPL after compression
    compressed_ppl = compute_perplexity(mdl, tok, encoded)
    ratio = compressed_ppl / baseline_ppl if baseline_ppl > 0 else float("inf")
    total_time = time.time() - t0

    result = {
        "model": tag,
        "baseline_ppl": round(baseline_ppl, 4),
        "compressed_ppl": round(compressed_ppl, 4),
        "ratio": round(ratio, 4),
        "config": {
            "rank": rank,
            "residual_bits": residual_bits,
            "group_size": group_size,
            "use_compensation": use_compensation,
            "mixed_precision": mixed_precision,
            "adaptive_rank": adaptive_rank,
        },
        "timing": {
            "compress_sec": round(compress_time, 2),
            "total_sec": round(total_time, 2),
        },
        "layer_stats": all_stats[:5],  # first 5 for debugging
    }

    print(f"\n  {'=' * 40}")
    print(f"  Result: PPL={compressed_ppl:.4f} (ratio={ratio:.3f}x)")
    print(f"  Time: {compress_time:.1f}s compress, {total_time:.1f}s total")
    print(f"  {'=' * 40}")

    return result
