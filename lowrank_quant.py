"""
lowrank_quant.py — Low-rank + quantization combination (CMQ).
Result → results/lowrank_quant.json
"""

import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset

from model_loader import load_model, load_tokenizer, BASE_DIR, DEVICE, DTYPE

RESULTS = BASE_DIR / "results"
RESULTS.mkdir(exist_ok=True)

EXPERIMENTS = [
    {"rank": 16, "bits": 4},
    {"rank": 32, "bits": 4},
    {"rank": 64, "bits": 4},
    {"rank": 32, "bits": 3},
    {"rank": 64, "bits": 3},
    {"rank": 16, "bits": 2},
    {"rank": 32, "bits": 2},
    {"rank": 64, "bits": 2},
]


def get_eval_text() -> str:
    try:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
        parts = [x for x in data["text"] if len(x.strip()) > 80][:200]
        return "\n\n".join(parts)
    except Exception:
        return ("Conic manifold quantization combines low-rank decomposition with quantization. " * 200)


def compute_perplexity(model, tokenizer, text, max_len=1024):
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[0]
    seq_len = ids.size(0)

    total_loss, total_tok = 0.0, 0
    for b in range(0, seq_len - max_len + 1, max_len):
        e = min(b + max_len, seq_len)
        if e - b < 32:
            break
        chunk = ids[b:e].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(chunk, labels=chunk)
        total_loss += out.loss.item() * (e - b)
        total_tok += e - b

    if seq_len % max_len > 32:
        r = seq_len % max_len
        chunk = ids[-r:].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(chunk, labels=chunk)
        total_loss += out.loss.item() * r
        total_tok += r

    return math.exp(total_loss / max(1, total_tok))


def quantize_symmetric(w, bits):
    if bits >= 16:
        return w
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    wf = w.float()
    scale = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(wf / scale).clamp(qmin, qmax) * scale


def lowrank_quant_weight(w, rank, bits):
    """SVD → quantization of the factors → reconstruction."""
    if rank >= min(w.shape):
        return w
    wf = w.float().cpu()
    U, S, Vh = torch.linalg.svd(wf, full_matrices=False)

    A = U[:, :rank] * S[:rank].unsqueeze(0)   # m × r
    B = Vh[:rank, :]                          # r × n

    A_q = quantize_symmetric(A, bits)
    B_q = quantize_symmetric(B, bits)

    return (A_q @ B_q).to(w.dtype).to(w.device)


def main():
    print("=" * 60)
    print("  CMQ — Low-Rank + Quantization")
    print("=" * 60)

    tokenizer = load_tokenizer()
    text = get_eval_text()

    # Determine the allowed ranks
    probe = load_model(max_retries=2)
    min_dim = None
    for p in probe.parameters():
        if p.ndim == 2:
            d = min(p.shape)
            if min_dim is None or d < min_dim:
                min_dim = d
    del probe
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    valid = [e for e in EXPERIMENTS if e["rank"] < (min_dim or 256)]
    print(f"\n  Min dim: {min_dim}, experiments: {len(valid)}")

    # Baselines
    base_ppl = None
    bp = RESULTS / "ppl_base.json"
    if bp.exists():
        with open(bp) as f:
            base_ppl = json.load(f)["perplexity"]

    scalar_ppls = {}
    sp = RESULTS / "quant_scalar.json"
    if sp.exists():
        with open(sp) as f:
            for r in json.load(f):
                scalar_ppls[r["bits"]] = r["perplexity"]

    results = []

    for exp in valid:
        rank, bits = exp["rank"], exp["bits"]
        print(f"\n{'-' * 50}")
        print(f"  rank={rank}, {bits}-bit")

        model = load_model()

        matrices = []
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "embed" not in name and "lm_head" not in name:
                if min(mod.weight.shape) > rank:
                    matrices.append((name, mod))

        print(f"  Compressing {len(matrices)} matrices...")
        ratios = []

        with torch.no_grad():
            for name, mod in tqdm(matrices, desc=f"LRQ{rank}/{bits}b", ncols=80):
                w = mod.weight.data
                m, n = w.shape
                orig = m * n * 16
                comp = (m * rank + rank * n) * bits + (m + rank) * 16
                ratios.append(orig / max(1, comp))
                mod.weight.data = lowrank_quant_weight(w, rank, bits).to(DEVICE)

        ppl = compute_perplexity(model, tokenizer, text)
        mean_ratio = sum(ratios) / max(1, len(ratios))

        entry = {
            "method": f"lowrank{rank}_q{bits}",
            "rank": rank, "bits": bits,
            "perplexity": round(ppl, 4),
            "mean_compression_ratio": round(mean_ratio, 3),
            "compressed_matrices": len(matrices),
        }
        if base_ppl:
            entry["ppl_degradation_pct"] = round((ppl - base_ppl) / base_ppl * 100, 2)
            entry["ppl_ratio"] = round(ppl / base_ppl, 4)

        vs_scalar = ""
        if bits in scalar_ppls:
            sppl = scalar_ppls[bits]
            better = ppl < sppl
            entry["scalar_baseline_ppl"] = sppl
            entry["better_than_scalar"] = better
            entry["vs_scalar_pct"] = round((ppl - sppl) / sppl * 100, 2)
            vs_scalar = f"  vs scalar {bits}b: {'[OK]' if better else '[FAIL]'} ({entry['vs_scalar_pct']:+.1f}%)"

        results.append(entry)
        print(f"  PPL={ppl:.2f}  comp={mean_ratio:.2f}x{'  delta=' + f'{(ppl-base_ppl)/base_ppl*100:+.1f}' + '%' if base_ppl else ''}{vs_scalar}")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    out = RESULTS / "lowrank_quant.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
