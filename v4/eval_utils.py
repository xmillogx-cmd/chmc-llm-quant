#!/usr/bin/env python3
"""
eval_utils.py — Общие утилиты для CHMC v4 пайплайна
====================================================

Исправления:
  1. PPL без double-counting overlap токенов (ignore_index=-100)
  2. WikiText eval data вместо self-generated fallback
  3. honest_compression_bits с original_bits=16.0 (FP16 baseline)
  4. Calibration на WikiText вместо повторяющегося текста
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


BASE_DIR = Path(__file__).parent.resolve()  # v4/
ROOT_DIR = BASE_DIR.parent                  # cmq_experiment/

WIKITEXT_CACHE = ROOT_DIR / ".wikitext_cache.pt"  # legacy, kept for compat
CALIB_TEXT_CACHE = ROOT_DIR / ".calib_text_cache.txt"

DEVICE = "cpu"
DTYPE = torch.float32


# ──────────────────────────────────────────────────────────────────────
# WikiText eval data
# ──────────────────────────────────────────────────────────────────────
def load_wikitext_eval(tokenizer: PreTrainedTokenizer, n_tokens: int = 12000) -> torch.Tensor:
    """
    Загрузить или скачать WikiText test split для оценки PPL.

    FIX #2: НИКОГДА не использовать model.generate() для eval —
    модель идеально предсказывает свои же токены → PPL ≈ 1.0 (артефакт).

    FIX v5: кэш привязан к vocab-размеру токенизатора, чтобы SmolLM и Qwen
    не использовали чужие токены (разные vocabulary → разные ID).
    """
    # Model-specific cache key based on vocab size
    vocab_size = len(tokenizer)
    cache_path = ROOT_DIR / f".wikitext_cache_v{vocab_size}.pt"

    if cache_path.exists():
        cached = torch.load(cache_path, map_location=DEVICE, weights_only=True)
        return cached[:n_tokens]

    # Download from datasets
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        print(f"[eval_utils] Downloading wikitext-2-raw-v1 test split...")
        parquet_path = hf_hub_download(
            repo_id="Salesforce/wikitext",
            filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(parquet_path)
        text_col = "text" if "text" in df.columns else df.columns[0]
        text = "\n".join(df[text_col].dropna().astype(str))
        encoded = tokenizer(text, return_tensors="pt").input_ids[0]

        # Pad/truncate to n_tokens
        if len(encoded) < n_tokens:
            # Repeat with BOS separator to reach target length
            bos_id = tokenizer.bos_token_id or 1
            full = encoded.clone()
            while len(full) < n_tokens:
                full = torch.cat([full, torch.tensor([bos_id]), encoded])
            encoded = full[:n_tokens]

        torch.save(encoded, cache_path)
        print(f"[eval_utils] WikiText cached ({vocab_size} vocab): {len(encoded)} tokens")
        return encoded[:n_tokens]
    except Exception as e:
        print(f"[eval_utils] WikiText download failed: {e}")
        # Last resort: generate from a fixed seed (NOT the model being evaluated)
        # This is still imperfect but better than self-eval
        raise RuntimeError(
            "WikiText unavailable and no fallback. Install huggingface_hub + pandas.\n"
            f"Or pre-cache tokens at {cache_path}"
        )


def load_calib_text(tokenizer: PreTrainedTokenizer, n_tokens: int = 2048) -> str:
    """
    Загрузить текст для калибровки (использует WikiText).

    FIX #4: calibration на разнообразном тексте вместо повторяющейся фразы.
    """
    if CALIB_TEXT_CACHE.exists():
        with open(CALIB_TEXT_CACHE, "r", encoding="utf-8") as f:
            return f.read()

    encoded = load_wikitext_eval(tokenizer, n_tokens * 2)
    text = tokenizer.decode(encoded[:n_tokens])

    with open(CALIB_TEXT_CACHE, "w", encoding="utf-8") as f:
        f.write(text)

    return text


# ──────────────────────────────────────────────────────────────────────
# PPL evaluation (FIXED — no double-counting)
# ──────────────────────────────────────────────────────────────────────
def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    encoded: Optional[torch.Tensor] = None,
    n_tokens: int = 2000,
    max_len: int = 512,
    stride: int = 256,
) -> float:
    """
    Вычислить perplexity на WikiText с маскированием overlap токенов.

    FIX #1: при overlap (stride < max_len) первые (max_len - stride - 1) токенов
    в каждом чанке маскируются ignore_index=-100, чтобы не считать loss дважды.
    """
    if encoded is None:
        encoded = load_wikitext_eval(tokenizer, n_tokens)

    model.eval()
    nll_total = 0.0
    token_count = 0
    pad_id = tokenizer.pad_token_id

    with torch.no_grad():
        for i in range(0, len(encoded) - max_len + 1, stride):
            chunk = encoded[i:i + max_len].unsqueeze(0).to(DEVICE)

            # Build attention mask for padding tokens
            attn_mask = None
            if pad_id is not None:
                attn_mask = (chunk != pad_id).to(DEVICE)

            outputs = model(chunk, attention_mask=attn_mask, use_cache=False)
            logits = outputs.logits.float()

            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = encoded[i + 1:i + max_len].unsqueeze(0).to(DEVICE).clone()

            # FIX #1: mask overlap tokens that were already counted in prev chunk
            if i > 0 and stride < max_len:
                n_overlap = max_len - stride - 1
                shift_labels[:, :n_overlap] = -100

            # Mask padding tokens
            if pad_id is not None:
                shift_labels[shift_labels == pad_id] = -100

            ce = F.cross_entropy(
                shift_logits.flatten(0, 1),
                shift_labels.flatten(0, 1),
                ignore_index=-100,
                reduction="sum",
            )

            if not torch.isnan(ce):
                nll_total += ce.item()
                token_count += (shift_labels != -100).sum().item()

    if token_count == 0:
        return float("inf")

    avg_nll = nll_total / token_count
    perplexity = math.exp(min(avg_nll, 50.0))

    if math.isinf(perplexity) or math.isnan(perplexity):
        return 1e9

    return perplexity


# ──────────────────────────────────────────────────────────────────────
# Calibration inputs collection (single-pass, FIX #4)
# ──────────────────────────────────────────────────────────────────────
def collect_calibration_inputs(
    model: PreTrainedModel,
    layer_names: List[str],
    tokenizer: PreTrainedTokenizer,
    n_tokens: int = 2048,
) -> Dict[str, torch.Tensor]:
    """
    Собрать input activations для всех слоёв за один forward pass.

    FIX #4: использует WikiText вместо повторяющегося текста.
    Использует flatten(0,1) для получения [batch*seq, in_f] формы.
    """
    inputs_map = {name: [] for name in layer_names}

    def _get_module(name):
        mod = model
        for p in name.split("."):
            mod = getattr(mod, p, None)
            if mod is None:
                return None
        return mod

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0].detach().float()
            inputs_map[name].append(x.flatten(0, 1))
        return hook

    hooks = []
    for name in layer_names:
        mod = _get_module(name)
        if mod is not None and isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    # Load calibration text (WikiText-based)
    calib_text = load_calib_text(tokenizer, n_tokens)
    encoded = tokenizer(calib_text, return_tensors="pt", truncation=True, max_length=n_tokens)
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask, use_cache=False)

    for h in hooks:
        h.remove()

    result = {}
    for name in layer_names:
        if inputs_map[name]:
            cat = torch.cat(inputs_map[name], dim=0)  # [total_tokens, in_f]
            if len(cat) > n_tokens:
                idx = torch.randperm(len(cat), device=cat.device)[:n_tokens]
                cat = cat[idx]
            result[name] = cat
        else:
            mod = _get_module(name)
            if isinstance(mod, nn.Linear):
                in_f = mod.in_features
            else:
                in_f = 256  # fallback
            result[name] = torch.randn(64, in_f, device=DEVICE)

    return result


# ──────────────────────────────────────────────────────────────────────
# Honest bit accounting (FIXED — original_bits=16.0)
# ──────────────────────────────────────────────────────────────────────
def honest_compression_bits(
    out_f: int,
    in_f: int,
    rank: int,
    original_bits: float = 16.0,     # FIX #3: FP16 baseline (not FP32)
    factor_bits: float = 16.0,        # low-rank factors as FP16
    residual_bits: float = 4.0,       # INT4 residual
    residual_density: float = 1.0,    # 1.0 = dense, <1.0 = sparse
    index_bits: float = 16.0,         # CoO index bits per non-zero (sparse only)
    scale_bits: float = 16.0,         # per-channel scale
    group_size: int = None,           # None = per-channel (default), >0 = per-group
) -> Dict[str, float]:
    """
    Честный подсчёт бит для сжатого веса.

    FIX #3: original_bits=16.0 (FP16 baseline), как в индустрии (GPTQ, AWQ).
    FIX v2: dense residual НЕ получает index_bits overhead.
    FIX v3: per-channel scales по умолчанию (group_size=None), как quantize_symmetric_per_channel.
    """
    original = original_bits * out_f * in_f

    # Low-rank factors: A(out×rank) + B(in×rank)
    lowrank = factor_bits * rank * (out_f + in_f)

    # Residual values
    n_residual = int(residual_density * out_f * in_f)
    residual_vals = n_residual * residual_bits

    # Index overhead — ТОЛЬКО для sparse residual
    is_sparse = residual_density < 0.99
    if is_sparse:
        residual_idx = n_residual * (index_bits * 2)
    else:
        residual_idx = 0.0

    # Scales — per-channel by default, or per-group if group_size is set
    if group_size is None:
        n_groups = out_f  # per-channel: 1 scale per output channel
    else:
        n_groups = (out_f * in_f + group_size - 1) // group_size
    scales = n_groups * scale_bits

    compressed = lowrank + residual_vals + residual_idx + scales
    if compressed <= 0:
        compressed = 1.0

    return {
        "original_bits": original,
        "compressed_bits": compressed,
        "compression_ratio": original / compressed,
        "bits_per_weight": compressed / (out_f * in_f),
        "lowrank_bits": lowrank,
        "residual_val_bits": residual_vals,
        "residual_idx_bits": residual_idx,
        "scale_bits": scales,
    }


# ──────────────────────────────────────────────────────────────────────
# Weighted SVD compression (shared)
# ──────────────────────────────────────────────────────────────────────
def weighted_svd_compress(
    W: torch.Tensor, X: torch.Tensor, rank: int
) -> Tuple[torch.Tensor, float]:
    """
    Low-rank аппроксимация с взвешиванием по ковариации входов.

    W: [out_f, in_f] — оригинальный вес
    X: [tokens, in_f] — калибровочные входы
    rank: целевой ранг
    """
    W = W.float()
    out_f, in_f = W.shape
    rank = min(rank, out_f, in_f)

    # Covariance diagonal (input importance)
    diag_c = (X ** 2).mean(dim=0).clamp(min=1e-8)

    # Weight and compress
    W_weighted = W * torch.sqrt(diag_c).unsqueeze(0)
    U, S, V = torch.svd_lowrank(W_weighted, q=rank, niter=5)
    W_approx = U @ torch.diag(S) @ V.T

    # Un-weight
    W_approx = W_approx / torch.sqrt(diag_c).unsqueeze(0)

    err = (W - W_approx).pow(2).sum().sqrt() / W.pow(2).sum().sqrt()
    return W_approx, err.item()


# ──────────────────────────────────────────────────────────────────────
# Layer helpers (shared)
# ──────────────────────────────────────────────────────────────────────
def get_compressible_layers(model: PreTrainedModel) -> List[str]:
    """Получить список compressible Linear слоёв (без embed_tokens и lm_head)."""
    skip_prefixes = ("embed_tokens", "lm_head")
    return [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and mod.in_features > 0
        and mod.out_features > 0
        and not any(name.startswith(pfx) for pfx in skip_prefixes)
    ]


def get_module_by_name(model: PreTrainedModel, name: str) -> Optional[nn.Module]:
    """Найти модуль по точному пути (model.layers.0.self_attn.q_proj)."""
    mod = model
    for p in name.split("."):
        mod = getattr(mod, p, None)
        if mod is None:
            return None
    return mod


def get_weight(model: PreTrainedModel, layer_name: str) -> torch.Tensor:
    """Получить weight тензор слоя."""
    mod = get_module_by_name(model, layer_name)
    if mod is None or not isinstance(mod, nn.Linear):
        raise ValueError(f"Layer '{layer_name}' not found or not Linear")
    return mod.weight


def set_weight(model: PreTrainedModel, layer_name: str, weight: torch.Tensor):
    """Установить weight тензор слоя."""
    mod = get_module_by_name(model, layer_name)
    with torch.no_grad():
        mod.weight.copy_(weight.to(mod.weight.dtype))
