#!/usr/bin/env python3
"""
eval_utils_v5.py — Unified evaluation utilities for CHMC v5
==========================================================

Fixes over v4:
  1. Deterministic PPL with proper overlap masking (BUG-1 fix)
  2. Model-specific token cache keyed by vocab size
  3. Consistent API — all v5 scripts import ONLY from this module
  4. Honest bit accounting matches actual storage layout

Usage:
    from eval_utils_v5 import (
        compute_perplexity, load_wikitext_eval, load_calib_text,
        collect_calibration_inputs, weighted_svd_compress,
        get_compressible_layers, get_weight, set_weight,
        honest_compression_bits, quantize_symmetric_per_channel,
    )
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

BASE_DIR = Path(__file__).parent.resolve()       # v5/
ROOT_DIR = BASE_DIR.parent                        # cmq_experiment/

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

if DEVICE == "cuda":
    print(f"[eval_v5] GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("[eval_v5] CPU mode (no CUDA)")


# ──────────────────────────────────────────────────────────────────────
# WikiText eval data (model-specific cache)
# ──────────────────────────────────────────────────────────────────────
def load_wikitext_eval(tokenizer: PreTrainedTokenizer, n_tokens: int = 12000) -> torch.Tensor:
    """
    Load WikiText test split tokens. Cache is keyed by vocab size so different
    tokenizers don't share cached encodings.

    FIX BUG-1: NEVER use model.generate() for eval data.
    """
    vocab_size = len(tokenizer)
    cache_path = ROOT_DIR / f".wikitext_cache_v{vocab_size}.pt"

    if cache_path.exists():
        cached = torch.load(cache_path, map_location=DEVICE, weights_only=True)
        return cached[:n_tokens]

    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        print(f"[eval_v5] Downloading wikitext-2-raw-v1 test split...")
        parquet_path = hf_hub_download(
            repo_id="Salesforce/wikitext",
            filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(parquet_path)
        text_col = "text" if "text" in df.columns else df.columns[0]
        text = "\n".join(df[text_col].dropna().astype(str))
        encoded = tokenizer(text, return_tensors="pt").input_ids[0]

        if len(encoded) < n_tokens:
            # BUG-6 fix: use train split as supplement instead of repeating test text
            # Repeating causes model to "memorize" patterns → PPL underestimation
            try:
                from datasets import load_dataset
                print("[eval_v5] Test split too short, loading train split...")
                ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                text_train = "\n\n".join([x for x in ds_train["text"] if len(x.strip()) > 80][:300])
                enc_train = tokenizer(text_train, return_tensors="pt").input_ids[0]
                encoded = torch.cat([encoded, enc_train])[:n_tokens]
            except Exception:
                # Last resort: repeat with BOS separator (warns about potential PPL bias)
                import warnings
                warnings.warn(
                    "WikiText test split too short and train unavailable. "
                    "Repeating text may underestimate PPL.", UserWarning
                )
                bos_id = tokenizer.bos_token_id or 1
                full = encoded.clone()
                while len(full) < n_tokens:
                    full = torch.cat([full, torch.tensor([bos_id]), encoded])
                encoded = full[:n_tokens]

        torch.save(encoded, cache_path)
        print(f"[eval_v5] Cached ({vocab_size} vocab): {len(encoded)} tokens")
        return encoded[:n_tokens]
    except Exception as e:
        # Try training split as fallback
        try:
            from datasets import load_dataset
            print(f"[eval_v5] Test split failed ({e}), trying train+test...")
            ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join([x for x in ds_test["text"] if len(x.strip()) > 80])
            encoded = tokenizer(text, return_tensors="pt").input_ids[0]

            if len(encoded) < n_tokens:
                ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                text_train = "\n\n".join([x for x in ds_train["text"] if len(x.strip()) > 80][:200])
                enc_train = tokenizer(text_train, return_tensors="pt").input_ids[0]
                encoded = torch.cat([encoded, enc_train])[:n_tokens]

            torch.save(encoded, cache_path)
            return encoded[:n_tokens]
        except Exception as e2:
            raise RuntimeError(
                f"WikiText unavailable: {e} / {e2}. "
                f"Pre-cache tokens at {cache_path}"
            )


def load_calib_text(tokenizer: PreTrainedTokenizer, n_tokens: int = 2048) -> str:
    """Load calibration text from the WikiText TRAIN split.

    CONTA-MINATION fix: calibration must never use the test split (that is
    the PPL eval set). Previously this function decoded test-split tokens,
    so calibration and eval shared the same data.

    Cache is keyed by vocab size (`.calib_text_train_v{vocab}.txt`) so
    different tokenizers don't share cached text. The new cache name also
    guarantees old test-split caches (`.calib_text_cache_v*.txt`) are not
    reused.
    """
    vocab_size = len(tokenizer)
    calib_cache = ROOT_DIR / f".calib_text_train_v{vocab_size}.txt"

    if calib_cache.exists():
        with open(calib_cache, "r", encoding="utf-8") as f:
            return f.read()

    # Primary: datasets library, train split
    try:
        from datasets import load_dataset
        print("[eval_v5] Loading wikitext-2-raw-v1 TRAIN split for calibration...")
        ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        lines = [x for x in ds_train["text"] if len(x.strip()) > 80]
        # Take enough lines to reach n_tokens after tokenization
        text = "\n\n".join(lines)
        encoded = tokenizer(text, return_tensors="pt").input_ids[0]
        if len(encoded) < n_tokens:
            raise RuntimeError(f"train split too short: {len(encoded)} < {n_tokens}")
        text = tokenizer.decode(encoded[:n_tokens])
    except Exception as e:
        # Fallback: hf_hub_download of the train parquet
        print(f"[eval_v5] datasets failed ({e}), trying hf_hub_download train parquet...")
        from huggingface_hub import hf_hub_download
        import pandas as pd
        parquet_path = hf_hub_download(
            repo_id="Salesforce/wikitext",
            filename="wikitext-2-raw-v1/train-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(parquet_path)
        text_col = "text" if "text" in df.columns else df.columns[0]
        text = "\n".join(df[text_col].dropna().astype(str))
        encoded = tokenizer(text, return_tensors="pt").input_ids[0]
        if len(encoded) < n_tokens:
            raise RuntimeError(f"train split too short: {len(encoded)} < {n_tokens}")
        text = tokenizer.decode(encoded[:n_tokens])

    with open(calib_cache, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[eval_v5] Calib text cached (train split, {vocab_size} vocab): {len(text)} chars")
    return text


# ──────────────────────────────────────────────────────────────────────
# Perplexity (deterministic, FIX BUG-1)
# ──────────────────────────────────────────────────────────────────────
def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    encoded: Optional[torch.Tensor] = None,
    n_tokens: int = 12000,
    max_len: int = 512,
    stride: int = 256,
) -> float:
    """
    Sliding window PPL with overlap masking.

    FIX BUG-1: Deterministic — same encoded input always produces same PPL.
    Overlap tokens masked with ignore_index=-100 to prevent double-counting.
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

            attn_mask = None
            if pad_id is not None:
                attn_mask = (chunk != pad_id).to(DEVICE)

            outputs = model(chunk, attention_mask=attn_mask, use_cache=False)
            logits = outputs.logits.float()

            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = encoded[i + 1:i + max_len].unsqueeze(0).to(DEVICE).clone()

            # Mask overlap tokens (already counted in previous chunk)
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
# Calibration inputs collection
# ──────────────────────────────────────────────────────────────────────
def collect_calibration_inputs(
    model: PreTrainedModel,
    layer_names: List[str],
    tokenizer: PreTrainedTokenizer,
    n_tokens: int = 2048,
) -> Dict[str, torch.Tensor]:
    """Single-pass forward hook to collect pre-activation inputs per layer."""
    inputs_map: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}

    def _get_mod(name):
        mod = model
        for p in name.split("."):
            mod = getattr(mod, p, None)
            if mod is None:
                return None
        return mod

    hooks = []
    for name in layer_names:
        mod = _get_mod(name)
        if mod is not None and isinstance(mod, nn.Linear):
            captured = inputs_map[name]
            def hook(m, inp, out, store=captured):
                x = inp[0].detach().float()
                store.append(x.flatten(0, 1))
            hooks.append(mod.register_forward_hook(hook))

    calib_text = load_calib_text(tokenizer, n_tokens)
    enc = tokenizer(calib_text, return_tensors="pt", truncation=True, max_length=n_tokens)
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask, use_cache=False)

    for h in hooks:
        h.remove()

    result = {}
    for name in layer_names:
        if inputs_map[name]:
            cat = torch.cat(inputs_map[name], dim=0)
            if len(cat) > n_tokens:
                # Deterministic subsample (first n_tokens rows, token order).
                # Old torch.randperm made runs non-reproducible.
                cat = cat[:n_tokens]
            result[name] = cat
        else:
            mod = _get_mod(name)
            in_f = mod.in_features if isinstance(mod, nn.Linear) else 256
            result[name] = torch.randn(64, in_f, device=DEVICE)

    return result


# ──────────────────────────────────────────────────────────────────────
# Honest bit accounting
# ──────────────────────────────────────────────────────────────────────
def honest_compression_bits(
    out_f: int, in_f: int, rank: int,
    original_bits: float = 16.0,
    factor_bits: float = 16.0,
    residual_bits: float = 4.0,
    residual_density: float = 1.0,
) -> Dict[str, float]:
    """Bit accounting: original vs compressed (lowrank + residual + scales)."""
    original = original_bits * out_f * in_f
    lowrank = factor_bits * rank * (out_f + in_f)

    n_residual = int(residual_density * out_f * in_f)
    residual_vals = n_residual * residual_bits

    is_sparse = residual_density < 0.99
    residual_idx = n_residual * (16.0 * 2) if is_sparse else 0.0

    # Per-channel scales (default, no group_size overhead for dense)
    n_groups = out_f
    scales = n_groups * 16.0

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
        "scale_bits": scales,
    }


# ──────────────────────────────────────────────────────────────────────
# Weighted SVD compression
# ──────────────────────────────────────────────────────────────────────
def weighted_svd_compress(
    W: torch.Tensor, X: torch.Tensor, rank: int
) -> Tuple[torch.Tensor, float]:
    """Covariance-weighted SVD: W_weighted = W * sqrt(diag(XX^T)).

    OPT-4 fix: use full SVD for small matrices (<256×256) where lowrank is slower.
    """
    W = W.float()
    out_f, in_f = W.shape
    rank = min(rank, out_f, in_f)

    diag_c = (X ** 2).mean(dim=0).clamp(min=1e-8)
    W_weighted = W * torch.sqrt(diag_c).unsqueeze(0)

    # OPT-4: full SVD is faster for small matrices (<256×256)
    if min(out_f, in_f) <= 256:
        U_full, S_full, Vh = torch.linalg.svd(W_weighted, full_matrices=False)
        U = U_full[:, :rank]
        S = S_full[:rank]
        V = Vh[:rank, :].T
    else:
        U, S, V = torch.svd_lowrank(W_weighted, q=rank, niter=5)

    W_approx = (U * S.unsqueeze(0)) @ V.T  # broadcasting instead of diag(S)
    W_approx = W_approx / torch.sqrt(diag_c).unsqueeze(0)

    err = (W - W_approx).pow(2).sum().sqrt() / W.pow(2).sum().sqrt()
    return W_approx, err.item()


# ──────────────────────────────────────────────────────────────────────
# Quantization
# ──────────────────────────────────────────────────────────────────────
def quantize_symmetric_per_channel(
    tensor: torch.Tensor, bits: int = 4, dim: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel quantization. Returns (reconstructed, scales)."""
    t = tensor.float()
    max_val = t.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
    scales = max_val / (2 ** (bits - 1) - 1)
    q = torch.round(t / scales).clamp(-(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1)
    return q * scales, scales


# ──────────────────────────────────────────────────────────────────────
# Layer helpers
# ──────────────────────────────────────────────────────────────────────
def get_compressible_layers(model: PreTrainedModel) -> List[str]:
    """All Linear layers excluding embed_tokens and lm_head."""
    skip_prefixes = ("embed_tokens", "lm_head")
    return [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and mod.in_features > 0
        and mod.out_features > 0
        and not any(name.startswith(pfx) for pfx in skip_prefixes)
    ]


def get_module_by_name(model: PreTrainedModel, name: str) -> Optional[nn.Module]:
    """Navigate dotted path to find module."""
    mod = model
    for p in name.split("."):
        mod = getattr(mod, p, None)
        if mod is None:
            return None
    return mod


def get_weight(model: PreTrainedModel, layer_name: str) -> torch.Tensor:
    """Get weight tensor of a layer."""
    mod = get_module_by_name(model, layer_name)
    if mod is None or not isinstance(mod, nn.Linear):
        raise ValueError(f"Layer '{layer_name}' not found or not Linear")
    return mod.weight


def set_weight(model: PreTrainedModel, layer_name: str, weight: torch.Tensor):
    """Replace weight tensor in-place."""
    mod = get_module_by_name(model, layer_name)
    with torch.no_grad():
        mod.weight.copy_(weight.to(mod.weight.dtype))
