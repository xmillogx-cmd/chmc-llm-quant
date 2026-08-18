#!/usr/bin/env python3
"""
run_gptq_baseline.py — GPTQ/AWQ baselines (library-only)
=========================================================

Сравнение CHMC с индустриальными методами (только библиотеки):
  - GPTQ 4-bit (GPTQModel)
  - AutoGPTQ 4-bit (если установлена)
  - AWQ 4-bit (если установлена)

Ручная реализация GPTQ удалена: формула компенсации
W[:,j+1:] -= err·(H_diag[j+1:]/H_diag[j]) не выводится из GPTQ и давала
PPL 1e8+. Работаем только с библиотечной GPTQ (GPTQModel).

Критерий приёмки:
  GPTQ/AWQ PPL ratio < 2.0x для обеих моделей

Usage:
    python v5/run_gptq_baseline.py --model models/smollm-135m models/qwen2.5-0.5b
"""

# Fix Windows cp1251 encoding — GPTQModel/logbar need UTF-8
import sys
if sys.platform == "win32" and not getattr(sys.stdout, 'encoding', '').lower().startswith('utf'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).parent.resolve()       # v5/
ROOT_DIR = BASE_DIR.parent                        # cmq_experiment/
RESULTS  = ROOT_DIR / "results_v5"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from eval_utils_v5 import (
    compute_perplexity, load_wikitext_eval, load_calib_text,
)


def try_gptqmodel(model_path: str, bits: int = 4) -> Optional[Dict[str, Any]]:
    """GPTQModel library quantization (primary baseline)."""
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except (ImportError, RuntimeError):
        print("  [GPTQModel] Import failed, skipping")
        return None

    import tempfile
    try:
        tok = AutoTokenizer.from_pretrained(model_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Calibration data — TRAIN split (never the test/eval split)
        calib_text = load_calib_text(tok, n_tokens=4096)
        calib_enc_ids = tok(calib_text, return_tensors="pt",
                            truncation=True, max_length=4096)["input_ids"][0]
        calib_enc = [{"input_ids": calib_enc_ids.unsqueeze(0)}]

        quant_config = QuantizeConfig(bits=bits, group_size=128, desc_act=False, sym=True)

        print("  [GPTQModel] Loading model...")
        model = GPTQModel.from_pretrained(model_path, quantize_config=quant_config)

        t0 = time.time()
        print("  [GPTQModel] Quantizing...")
        model.quantize(calib_enc)
        quant_time = time.time() - t0

        # Save quantized model and reload to avoid meta tensor issues
        tmpdir = tempfile.mkdtemp(dir=str(RESULTS))
        print(f"  [GPTQModel] Saving to {tmpdir}...")
        model.save_quantized(tmpdir)

        # Reload as quantized model
        device = "cuda:0" if DEVICE == "cuda" else "cpu"
        print("  [GPTQModel] Reloading quantized model...")
        model_q = GPTQModel.from_quantized(tmpdir, device=device)

        encoded = load_wikitext_eval(tok)
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.to(device)

        ppl = compute_perplexity(model_q, tok, encoded)

        result = {
            "method": f"gptqmodel_{bits}bit",
            "ppl": round(ppl, 4),
            "quant_time_sec": round(quant_time, 2),
            "group_size": 128,
        }
        # Cleanup GPU memory
        del model, model_q
        torch.cuda.empty_cache()
        return result
    except Exception as e:
        print(f"  [GPTQModel] Error: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()
        return None


def try_auto_gptq(model_path: str, bits: int = 4) -> Optional[Dict[str, Any]]:
    """Try loading auto-gptq library."""
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        return None

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        calib_texts = [x for x in ds["text"] if len(x.strip()) > 80][:32]
    except Exception:
        return None

    quant_config = BaseQuantizeConfig(bits=bits, group_size=128, desc_act=False)
    model = AutoGPTQForCausalLM.from_pretrained(model_path, quant_config)

    calib_enc = [
        tok(t, return_tensors="pt", max_length=512, truncation=True)
        for t in calib_texts
    ]

    t0 = time.time()
    model.quantize(calib_enc)
    quant_time = time.time() - t0

    encoded = load_wikitext_eval(tok)
    ppl = compute_perplexity(model.model, tok, encoded)

    result = {
        "method": f"auto_gptq_{bits}bit",
        "ppl": round(ppl, 4),
        "quant_time_sec": round(quant_time, 2),
    }
    del model
    torch.cuda.empty_cache()
    return result


def try_awq(model_path: str, bits: int = 4) -> Optional[Dict[str, Any]]:
    """Try loading autoawq library."""
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        return None

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoAWQForCausalLM.from_pretrained(model_path)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": bits,
        "version": "GEMM",
    }

    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        calib_texts = [x for x in ds["text"] if len(x.strip()) > 80][:32]
    except Exception:
        return None

    t0 = time.time()
    model.quantize(tok, quant_config=quant_config, calib_data=calib_texts)
    quant_time = time.time() - t0

    encoded = load_wikitext_eval(tok)
    ppl = compute_perplexity(model.model, tok, encoded)

    result = {
        "method": f"awq_{bits}bit",
        "ppl": round(ppl, 4),
        "quant_time_sec": round(quant_time, 2),
    }
    del model
    torch.cuda.empty_cache()
    return result


def run_gptq_baseline_for_model(model_path: str) -> Dict[str, Any]:
    """Run all available library GPTQ/AWQ baselines for one model."""
    tag = Path(model_path).name

    print(f"\nLoading model for GPTQ baseline...")
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    )
    mdl = mdl.to(DEVICE)
    mdl.eval()

    encoded = load_wikitext_eval(tok)
    baseline_ppl = compute_perplexity(mdl, tok, encoded)
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    results = {"baseline_ppl": round(baseline_ppl, 4)}

    # Library-based GPTQ/AWQ only (manual GPTQ was removed)
    for lib_fn in [try_gptqmodel, try_auto_gptq, try_awq]:
        res = lib_fn(model_path)
        if res:
            ratio = res["ppl"] / baseline_ppl if baseline_ppl > 0 else float("inf")
            results[res["method"]] = {**res, "ratio": round(ratio, 4)}
            print(f"  {res['method']}: PPL={res['ppl']:.4f} (ratio={ratio:.3f}x)")

    if not any(k.startswith("gptq") or k.startswith("awq") for k in results):
        results["error"] = "No GPTQ/AWQ library available (GPTQModel/AutoGPTQ/AWQ)"
        print("  No GPTQ/AWQ library available — install gptqmodel to run the baseline")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=[
        str(ROOT_DIR / "models/smollm-135m"),
        str(ROOT_DIR / "models/qwen2.5-0.5b"),
    ])
    args = parser.parse_args()

    all_results = {}
    for model_path in args.model:
        tag = Path(model_path).name
        print(f"\n{'=' * 60}")
        print(f"GPTQ/AWQ baselines: {tag}")
        print(f"{'=' * 60}")

        try:
            result = run_gptq_baseline_for_model(model_path)
        except Exception as e:
            # Save the failure as a result too (e.g. gemma4 QAT checkpoint
            # is not loadable by the installed transformers)
            result = {"baseline_ppl": None, "error": f"{type(e).__name__}: {e}"}
            print(f"  [ERROR] {type(e).__name__}: {e}")
        all_results[tag] = result

        out_file = RESULTS / f"gptq_baselines_{tag}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved -> {out_file}")

    # Combined summary (merge with existing so multi-model runs accumulate)
    combined_file = RESULTS / "gptq_awq_baselines.json"
    combined = {}
    if combined_file.exists():
        try:
            combined = json.loads(combined_file.read_text())
        except Exception:
            combined = {}
    combined.update(all_results)
    with open(combined_file, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results -> {combined_file}")

    # Non-zero exit if any model failed (so batch runners can detect it)
    if any("error" in r for r in all_results.values()):
        sys.exit(1)
