"""
quantize_scalar.py — Naive scalar quantization (baseline).
Result → results/quant_scalar.json
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
BITS = [8, 4, 3, 2]


def get_eval_text() -> str:
    try:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
        parts = [x for x in data["text"] if len(x.strip()) > 80][:200]
        return "\n\n".join(parts)
    except Exception:
        return ("Neural network quantization reduces precision to compress models. " * 200)


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
        n = e - b
        total_loss += out.loss.item() * n
        total_tok += n

    if seq_len % max_len > 32:
        r = seq_len % max_len
        chunk = ids[-r:].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(chunk, labels=chunk)
        total_loss += out.loss.item() * r
        total_tok += r

    return math.exp(total_loss / max(1, total_tok))


def quantize_symmetric(w, bits):
    """Symmetric per-channel quantization."""
    if bits >= 16:
        return w
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    wf = w.float()
    scale = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(wf / scale).clamp(qmin, qmax) * scale


def main():
    print("=" * 60)
    print("  CMQ — Scalar Quantization Baseline")
    print("=" * 60)

    tokenizer = load_tokenizer()
    text = get_eval_text()

    # Baseline PPL
    base_path = RESULTS / "ppl_base.json"
    base_ppl = None
    if base_path.exists():
        with open(base_path) as f:
            base_ppl = json.load(f)["perplexity"]
        print(f"\n  Baseline PPL from file: {base_ppl}")

    results = []

    for bits in BITS:
        print(f"\n{'-' * 50}")
        print(f"  {bits}-bit scalar quantization")

        model = load_model()

        orig_bits, new_bits = 0, 0

        # Progress across layers
        layers = list(model.named_modules())
        linear_count = sum(1 for _, m in layers if isinstance(m, torch.nn.Linear))

        print(f"  Quantizing {linear_count} Linear layers...")
        with torch.no_grad():
            for _name, module in tqdm(layers, desc=f"Q{bits}", ncols=80):
                if not isinstance(module, torch.nn.Linear):
                    continue
                w = module.weight.data
                orig_bits += w.numel() * 16
                new_bits += w.numel() * bits + w.shape[0] * 16  # scales
                module.weight.data = quantize_symmetric(w, bits)

        ppl = compute_perplexity(model, tokenizer, text)
        comp = orig_bits / max(1, new_bits)

        entry = {
            "method": f"scalar_{bits}bit",
            "bits": bits,
            "perplexity": round(ppl, 4),
            "compression_ratio": round(comp, 3),
        }
        if base_ppl:
            entry["ppl_degradation_pct"] = round((ppl - base_ppl) / base_ppl * 100, 2)
            entry["ppl_ratio"] = round(ppl / base_ppl, 4)

        results.append(entry)
        print(f"  PPL={ppl:.2f}  compression={comp:.2f}x{'  delta=' + f'{(ppl-base_ppl)/base_ppl*100:+.1f}' + '%' if base_ppl else ''}")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    out = RESULTS / "quant_scalar.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
