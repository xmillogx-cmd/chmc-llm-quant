#!/usr/bin/env python3
"""
step1_per_layer_drift.py — CHMC v6.4, STEP 1: per-layer drift decomposition
===========================================================================

Hypothesis (CHMC v6.4): the total model drift is high-dimensional
(d90 = 227-792 on blocks, results_v6/drift_correction), but the PER-LAYER
contribution of each layer may be low-rank: every layer adds a small noise,
and the sum over 30+ layers produces a high-dimensional residual. If so —
SELECTIVE compression is possible (compress only "safe" layers, keep the rest FP16).

Two drift measures on each of the 210 compressible layers of SmolLM-135M:

  LOCAL (the main test of the hypothesis):
      D_loc,l = X_fp,l @ E_l.T,   E_l = W_fp,l - W_comp,l
    The pure contribution of layer l on a clean FP input. No extra forward
    passes — only weight difference x calibration input (the same X_calib as in
    quantization). This is exactly what the hypothesis requires: "each layer adds
    small noise". Computed INSIDE the quantization loop (before set_weight).

  ACCUMULATED (secondary, to separate local vs total):
      D_acc_in,l  = post_in_l - pre_in_l   (state ENTERING layer l)
      D_acc_out,l = post_out_l - pre_out_l (output of layer l)
    One forward of the FP model + one forward of the compressed model, hooks on
    inputs/outputs. Shows how drift dimensionality grows with depth (summing over layers).

The quantization reproduces the strict_damp005 seed42 baseline EXACTLY as
run_chmc_v6: same layer order, same calibration (collect logic from
eval_utils_v6), a single torch.manual_seed(42) at the start. PPL is not computed
(inference does not consume RNG -> the random stream for svd_lowrank is identical).
CPU runs: weights will differ from GPU rep0 at the level of BLAS/SVD non-determinism
— this is normal, the drift structure does not depend on it.

STEP 1 GATE (declared BEFORE seeing data):
    >= 50% of layers with local d90 < 30 -> PASS, proceed to Step 2.
    otherwise                            -> STOP, negative result.

Artifacts:
    results_v6/per_layer_drift/per_layer_drift_smollm.json
    results_v6/per_layer_drift/chmc_weights_seed42.pt   (W_comp per layer)
"""

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2] / "v6"
sys.path.insert(0, str(BASE_DIR))

import numpy as np                      # noqa: E402
import torch                            # noqa: E402
import torch.nn as nn                   # noqa: E402

import chmc_v6 as C                     # noqa: E402
from eval_utils_v6 import (            # noqa: E402
    get_compressible_layers, get_weight, set_weight, load_calib_text,
)

OUT_DIR = BASE_DIR.parent / "results_v6" / "per_layer_drift"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(BASE_DIR.parent / "models" / "smollm-135m")
SEED = 42
N_TOKENS = 2048
TARGET_BPW = 4.2875
D90_GATE = 30          # порог d90_layer (объявлен до данных)
GATE_FRAC = 0.50       # доля слоёв, прошедших порог


def _get_mod(model, name):
    mod = model
    for p in name.split("."):
        mod = getattr(mod, p, None)
        if mod is None:
            return None
    return mod


def collect_inputs_outputs(model, layer_names, tokenizer, n_tokens=N_TOKENS):
    """Один forward с hooks на ВХОДЫ и ВЫХОДы каждого слоя.

    Логика токенизации/субсемплирования идентична eval_utils_v6.
    collect_calibration_inputs (тот же load_calib_text + truncation=2048 +
    cat[:n_tokens]), плюс захват выходов для accumulated drift.
    """
    store_in = {name: [] for name in layer_names}
    store_out = {name: [] for name in layer_names}

    hooks = []
    for name in layer_names:
        mod = _get_mod(model, name)
        if mod is None or not isinstance(mod, nn.Linear):
            continue
        si, so = store_in[name], store_out[name]

        def hook(m, inp, out, _si=si, _so=so):
            x = inp[0].detach().float()
            _si.append(x.flatten(0, 1).cpu())
            _so.append(out.detach().float().flatten(0, 1).cpu())

        hooks.append(mod.register_forward_hook(hook))

    calib_text = load_calib_text(tokenizer, n_tokens)
    enc = tokenizer(calib_text, return_tensors="pt", truncation=True,
                    max_length=n_tokens)
    input_ids = enc["input_ids"].to(C.DEVICE)
    attention_mask = enc["attention_mask"].to(C.DEVICE)

    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask, use_cache=False)

    for h in hooks:
        h.remove()

    ins, outs = {}, {}
    for name in layer_names:
        if store_in[name]:
            cat_i = torch.cat(store_in[name], dim=0)[:n_tokens]
            cat_o = torch.cat(store_out[name], dim=0)[:n_tokens]
            ins[name] = cat_i
            outs[name] = cat_o
    return ins, outs


def drift_metrics(D: np.ndarray) -> dict:
    """d90 + спектр для матрицы дрейфа D [n_tokens, out_f].

    Центрирование по токенам (конвенция TDA/drift_correction): SVD от
    D - mean(D, axis=0). d90 = мин. k с cumsum(s^2)/sum(s^2) >= 0.90.
    """
    Dc = D - D.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Dc, compute_uv=False)
    e = s ** 2
    tot = e.sum()
    if tot <= 0:
        return {"d90": 0, "n_components": int(s.size),
                "spectrum_top32": [], "cum_energy_at": {}}
    cum = np.cumsum(e) / tot
    d90 = int(np.searchsorted(cum, 0.90) + 1)
    cum_at = {}
    for k in (1, 5, 10, 20, 30, 50, 100):
        if k < s.size:
            cum_at[str(k)] = round(float(cum[k - 1]), 6)
    return {
        "d90": d90,
        "n_components": int(s.size),
        "spectrum_top32": [round(float(x), 8) for x in s[:32]],
        "cum_energy_at": cum_at,
    }


def rel_disps(D: np.ndarray, X: np.ndarray) -> dict:
    """rel_disp в двух конвенциях.

    user:   mean(||d_i||) / mean(||x_i||)          (формула из промпта)
    token:  mean( ||d_i|| / ||x_i|| )              (конвенция TDA)
    """
    dn = np.linalg.norm(D, axis=1)
    xn = np.linalg.norm(X, axis=1).clip(min=1e-12)
    return {
        "rel_disp_user": round(float(dn.mean() / max(xn.mean(), 1e-12)), 8),
        "rel_disp_tok": round(float((dn / xn).mean()), 8),
    }


def block_of(name: str) -> int:
    # model.layers.N.xxx -> N
    return int(name.split(".")[2])


def main():
    t0 = time.time()
    torch.manual_seed(SEED)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[step1] model={MODEL_PATH} seed={SEED}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32).to(C.DEVICE)
    mdl.eval()

    layers = get_compressible_layers(mdl)
    print(f"[step1] compressible layers: {len(layers)}", flush=True)

    # ── FP forward: входы + выходы всех слоёв на чистой FP-модели ───
    t_fp = time.time()
    pre_in, pre_out = collect_inputs_outputs(mdl, layers, tok, N_TOKENS)
    print(f"[step1] FP pass done ({time.time()-t_fp:.1f}s)", flush=True)

    # ── ranks + квантизация (точно как run_chmc_v6 baseline) ───────
    layer_shapes = []
    for name in layers:
        mod = _get_mod(mdl, name)
        out_f, in_f = mod.weight.shape[0], mod.weight.shape[1]
        layer_shapes.append((name, out_f, in_f))

    ranks = C.allocate_ranks_bit_budget(
        layer_shapes, TARGET_BPW, residual_bits=4, group_size=128,
        group_dim=0)

    t_q = time.time()
    comp_weights = {}
    records = []
    for idx, name in enumerate(layers):
        if (idx + 1) % 30 == 0:
            print(f"[step1] quant [{idx+1}/{len(layers)}] "
                  f"({time.time()-t_q:.0f}s)", flush=True)
        W_orig = get_weight(mdl, name).detach().cpu()
        X_calib = pre_in[name].to(W_orig.device)

        # LOCAL drift: D_loc = X @ (W - W_comp).T — до set_weight
        W_comp, stats = C.compress_layer_v6(
            W_orig, X_calib,
            rank=ranks[name],
            residual_bits=4, group_size=128, use_compensation=True,
            strict_sequential=True, group_dim=0, hessian_batches=1,
            dampening=0.05, niter=5, hadamard=False, block_cov=False,
            whitening=False, angular=False, tangent=False, cone_aware=False,
            lloyd_max=False, qjl=False, qjl_n_projections=0, ip_metric=False,
        )
        E = (W_orig - W_comp).numpy()
        X_np = X_calib.numpy()
        D_loc = X_np @ E.T
        rec = {
            "name": name,
            "block": block_of(name),
            "out_f": int(W_orig.shape[0]),
            "in_f": int(W_orig.shape[1]),
            "rank": int(ranks[name]),
            "lr_energy_pct": round(float(stats.get("lr_energy_pct", -1)), 6),
            "recon_err_ratio": round(float(stats.get("recon_err_ratio", -1)), 6),
            "local": drift_metrics(D_loc),
            **rel_disps(D_loc, X_np),
        }
        records.append(rec)

        set_weight(mdl, name, W_comp.to(W_orig.dtype))
        comp_weights[name] = W_comp.detach().cpu()
        del W_orig, W_comp, E, D_loc
    print(f"[step1] quantization + local drift done "
          f"({time.time()-t_q:.0f}s)", flush=True)

    # ── compressed forward: accumulated drift ───────────────────────
    t_cp = time.time()
    post_in, post_out = collect_inputs_outputs(mdl, layers, tok, N_TOKENS)
    print(f"[step1] compressed pass done ({time.time()-t_cp:.1f}s)", flush=True)

    # ── accumulated metrics (по слоям, с освобождением памяти) ──────
    t_m = time.time()
    for idx, rec in enumerate(records):
        if (idx + 1) % 30 == 0:
            print(f"[step1] acc-metrics [{idx+1}/{len(records)}] "
                  f"({time.time()-t_m:.0f}s)", flush=True)
        name = rec["name"]
        D_in = (post_in[name].numpy() - pre_in[name].numpy())
        D_out = (post_out[name].numpy() - pre_out[name].numpy())
        X_np = pre_in[name].numpy()
        m_in, m_out = drift_metrics(D_in), drift_metrics(D_out)
        rec["acc_in"] = {**m_in, **rel_disps(D_in, X_np)}
        rec["acc_out"] = {**m_out, **rel_disps(D_out, pre_out[name].numpy())}
        del D_in, D_out

    # ── summary + gate ──────────────────────────────────────────────
    def _stats(vals):
        a = np.array(vals, dtype=float)
        return {
            "min": int(a.min()), "max": int(a.max()),
            "median": round(float(np.median(a)), 1),
            "mean": round(float(a.mean()), 1),
            "p25": round(float(np.percentile(a, 25)), 1),
            "p75": round(float(np.percentile(a, 75)), 1),
        }

    loc = [r["local"]["d90"] for r in records]
    ain = [r["acc_in"]["d90"] for r in records]
    aout = [r["acc_out"]["d90"] for r in records]
    frac_loc = sum(1 for d in loc if d < D90_GATE) / len(loc)

    summary = {
        "n_layers": len(records),
        "gate": {"d90_threshold": D90_GATE, "required_frac": GATE_FRAC},
        "local_d90": _stats(loc),
        "acc_in_d90": _stats(ain),
        "acc_out_d90": _stats(aout),
        "frac_local_lt_gate": round(frac_loc, 4),
    }

    result = {
        "model": "smollm-135m",
        "task": "CHMC v6.4 step1: per-layer drift decomposition",
        "seed": SEED,
        "n_tokens_calib": N_TOKENS,
        "target_bpw": TARGET_BPW,
        "config": {
            "strict_sequential": True, "dampening": 0.05,
            "residual_bits": 4, "group_size": 128, "group_dim": 0,
            "hessian_batches": 1, "niter": 5, "use_compensation": True,
        },
        "definitions": {
            "local": "D_loc = X_fp @ (W - W_comp).T — вклад слоя на чистом FP-входе",
            "acc_in": "post_in - pre_in — состояние, входящее в слой (сумма всех предыдущих)",
            "acc_out": "post_out - pre_out — выход слоя в сжатой vs FP модели",
            "d90": "мин. k: cumsum(s^2)/sum(s^2) >= 0.90, SVD от центрированной D (float64)",
        },
        "summary": summary,
        "verdict_gate1": ("PASS" if frac_loc >= GATE_FRAC else "FAIL"),
        "layers": records,
    }

    out_json = OUT_DIR / "per_layer_drift_smollm.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    torch.save(comp_weights, OUT_DIR / "chmc_weights_seed42.pt")

    print("\n" + "=" * 64)
    print(f"LOCAL d90   : {summary['local_d90']}")
    print(f"ACC_IN  d90 : {summary['acc_in_d90']}")
    print(f"ACC_OUT d90 : {summary['acc_out_d90']}")
    print(f"\nGATE: frac(local d90 < {D90_GATE}) = {frac_loc:.3f} "
          f"(need >= {GATE_FRAC}) -> {result['verdict_gate1']}")
    print(f"JSON: {out_json}")
    print(f"[step1] total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
