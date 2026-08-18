#!/usr/bin/env python3
"""
collect_activations.py — TDA data collection (GPU, run by the USER)
====================================================================

What it does:
  1. Loads the model in FP32 (like run_chmc_v6).
  2. Captures block-level activations (decoder-layer outputs, residual stream
     N x hidden_size) on calibration tokens BEFORE compression.
  3. Runs THE SAME CHMC v6 compression loop at the target BPW (default:
     strict_damp005 from the stat grid — the best config), saving per layer:
       - singular values of the original weight (sv_pre);
       - singular values of the compressed weight W_comp (sv_post).
  4. Captures the same block activations AFTER compression on the same tokens.
  5. Computes baseline/compressed PPL (so the artifact is self-contained).
  6. Saves ONE .pt file into results_v6/tda_analysis/.

Memory: after every layer del + torch.cuda.empty_cache() (the v7 OOM lesson).
At the end an explicit model del + gc.collect() + empty_cache().

Usage (from the repo root):
    python tda_analysis/collect_activations.py --model smollm-135m
Options:
    --n-tokens 2048          calibration tokens (as in collect_calibration_inputs)
    --bpw 4.2875             target BPW (= GPTQ level)
    --overrides JSON         config overrides, default the best from the stat grid:
                             {"strict_sequential": true, "dampening": 0.05}
Time estimate (SmolLM): ~3-6 min (2x PPL + compression + SVD of all layers on CPU).

Environment:
    CMQ_MODELS_DIR           model directory override (default: <repo_root>/models)

Logging: all output is duplicated to tda_analysis/logs/collect_<model>_<time>.log
(console + file, including the traceback on failure).
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent          # tda_analysis/
ROOT_DIR = BASE_DIR.parent                          # cmq_experiment/
V6_DIR = ROOT_DIR / "v6"
sys.path.insert(0, str(V6_DIR))
sys.path.insert(0, str(BASE_DIR))

import chmc_v6 as C                                  # noqa: E402
import tda_log                                       # noqa: E402

OUT_DIR = ROOT_DIR / "results_v6" / "tda_analysis"
DEFAULT_OVERRIDES = {"strict_sequential": True, "dampening": 0.05}


def _get_blocks(mdl):
    """Список decoder-layer блоков (residual stream). Llama/Qwen/TinyLlama: mdl.model.layers."""
    core = getattr(mdl, "model", None) or getattr(mdl, "transformer", None) \
        or getattr(mdl, "language_model", None)
    layers = getattr(core, "layers", None) if core is not None else None
    if layers is None:
        raise RuntimeError("Не удалось найти decoder layers (mdl.model.layers).")
    blocks = list(layers)
    return [f"layers.{i}" for i in range(len(blocks))], blocks


def _capture_block_outputs(mdl, blocks, input_ids, attention_mask):
    """Один forward с хуками на выходы блоков -> {idx: (N, d) CPU fp32}."""
    captured = {}

    def make_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # (B, N, d) -> (N, d): batch=1 в наших прогонах
            captured[idx] = h[0].detach().float().cpu()
        return hook

    hooks = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(blocks)]
    with torch.no_grad():
        _ = mdl(input_ids, attention_mask=attention_mask, use_cache=False)
    for h in hooks:
        h.remove()
    return captured


def main():
    parser = argparse.ArgumentParser(description="TDA data collection (GPU)")
    parser.add_argument("--model", default="smollm-135m")
    parser.add_argument("--n-tokens", type=int, default=2048)
    parser.add_argument("--bpw", type=float, default=4.2875)
    parser.add_argument("--overrides", default=json.dumps(DEFAULT_OVERRIDES))
    args = parser.parse_args()

    _log_file = tda_log.start_logging(
        tda_log.log_file_for("collect", args.model))  # noqa: F841 (дожитие до конца процесса)

    overrides = json.loads(args.overrides)
    cfg = {**C.default_config(), **overrides, "bit_budget_bpw": args.bpw}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | device: {C.DEVICE}")
    print(f"Model: {args.model} | n_tokens={args.n_tokens} | bpw={args.bpw}")
    print(f"Overrides: {overrides}")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    models_dir = Path(os.environ.get("CMQ_MODELS_DIR", str(C.ROOT_DIR / "models")))
    model_path = str(models_dir / args.model)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device_map = cfg.get("device_map")
    if device_map:
        print(f"[collect] multi-GPU mode (device_map={device_map})")
        n_gpu = torch.cuda.device_count()
        per_gpu = cfg.get("max_memory_per_gpu")
        if per_gpu is None:
            headroom_gib = float(cfg.get("max_memory_headroom_gib", 3.0))
            max_mem = {}
            for i in range(n_gpu):
                total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                cap_gib = max(1.0, total_gib - headroom_gib)
                max_mem[i] = f"{cap_gib:.1f}GiB"
            print(f"[collect] max_memory: {max_mem}")
        else:
            max_mem = {i: per_gpu for i in range(n_gpu)}
        mdl = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, device_map=device_map,
            max_memory=max_mem)
    else:
        mdl = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        mdl = mdl.to(C.DEVICE)
    mdl.eval()

    # ── входные токены (те же, что в collect_calibration_inputs) ──────
    from eval_utils_v6 import load_calib_text
    calib_text = load_calib_text(tok, args.n_tokens)
    enc = tok(calib_text, return_tensors="pt", truncation=True, max_length=args.n_tokens)
    input_ids = enc["input_ids"].to(C.DEVICE if device_map is None else "cpu")
    attention_mask = enc["attention_mask"].to(input_ids.device)

    block_names, blocks = _get_blocks(mdl)
    print(f"Blocks: {len(block_names)} | hidden={mdl.config.hidden_size}")

    # ── PPL baseline + pre-активации ────────────────────────────────
    encoded_eval = C.load_wikitext_eval(tok)
    baseline_ppl = C.compute_perplexity(mdl, tok, encoded_eval)
    print(f"Baseline PPL (FP32): {baseline_ppl:.4f}")

    pre_act = _capture_block_outputs(mdl, blocks, input_ids, attention_mask)
    first = next(iter(pre_act.values()))
    n_tokens_actual = int(first.shape[0])
    d_hidden = int(first.shape[1])
    print(f"Pre-activations captured: {len(pre_act)} blocks x ({n_tokens_actual}, {d_hidden})")

    # ── цикл сжатия (реплика run_chmc_v6) + SVD весов до/после ───────
    layers = C.get_compressible_layers(mdl)
    print(f"Compressible layers: {len(layers)}")
    calib_inputs = C.collect_calibration_inputs(mdl, layers, tok, n_tokens=args.n_tokens)

    layer_shapes = []
    for name in layers:
        mod = mdl.get_submodule(name) if hasattr(mdl, "get_submodule") else None
        out_f, in_f = mod.weight.shape[0], mod.weight.shape[1]
        layer_shapes.append((name, out_f, in_f))

    base_ranks = C.allocate_ranks_bit_budget(
        layer_shapes, cfg["bit_budget_bpw"], cfg["residual_bits"], cfg["group_size"],
        qjl=cfg["qjl"], qjl_n_projections=cfg["qjl_n_projections"],
        cone_aware=cfg["cone_aware"], group_dim=cfg["group_dim"])

    sv_pre, sv_post = {}, {}
    t_comp = time.time()
    for idx, name in enumerate(layers):
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(layers)}] compress... ({time.time()-t_comp:.1f}s)")
        W_orig = C.get_weight(mdl, name)
        X_calib = calib_inputs.get(name)
        if X_calib is None or X_calib.shape[0] == 0:
            X_calib = torch.randn(64, W_orig.shape[1], device=W_orig.device)
        else:
            X_calib = X_calib.to(W_orig.device)

        sv_pre[name] = torch.linalg.svdvals(W_orig.detach().cpu().float())

        W_comp, _stats = C.compress_layer_v6(
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
        C.set_weight(mdl, name, W_comp.to(W_orig.dtype))
        sv_post[name] = torch.linalg.svdvals(W_comp.detach().cpu().float())

        del W_orig, X_calib, W_comp
        torch.cuda.empty_cache()

    print(f"Compression done in {time.time()-t_comp:.1f}s")

    # ── post-активации + PPL после сжатия ───────────────────────────
    if device_map is None:
        input_ids = input_ids.to(C.DEVICE)
        attention_mask = attention_mask.to(C.DEVICE)
    post_act = _capture_block_outputs(mdl, blocks, input_ids, attention_mask)
    compressed_ppl = C.compute_perplexity(mdl, tok, encoded_eval)
    ratio = compressed_ppl / baseline_ppl if baseline_ppl > 0 else float("inf")
    print(f"Compressed PPL: {compressed_ppl:.4f} | ratio={ratio:.6f}")

    # ── сохранение ОДНОГО .pt ───────────────────────────────────────
    payload = {
        "model": args.model,
        "config": cfg,
        "n_tokens": int(n_tokens_actual),
        "hidden_size": int(d_hidden),
        "blocks": block_names,
        "pre": [pre_act[i] for i in range(len(block_names))],
        "post": [post_act[i] for i in range(len(block_names))],
        "weight_sv_pre": {k: v.cpu() for k, v in sv_pre.items()},
        "weight_sv_post": {k: v.cpu() for k, v in sv_post.items()},
        "ranks": {k: int(v) for k, v in base_ranks.items()},
        "ppl": {"baseline": float(baseline_ppl), "compressed": float(compressed_ppl),
                "ratio": float(ratio)},
    }
    out_file = OUT_DIR / f"tda_activations_{args.model}.pt"
    torch.save(payload, out_file)
    print(f"\nSaved: {out_file}")

    # ── освобождение памяти (урок OOM v7) ───────────────────────────
    del mdl, tok, encoded_eval, calib_inputs, pre_act, post_act, sv_pre, sv_post
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Total: {time.time()-t0:.1f}s")
    print("Дальше (CPU): venv\\Scripts\\python.exe tda_analysis\\analyze.py")


if __name__ == "__main__":
    main()
