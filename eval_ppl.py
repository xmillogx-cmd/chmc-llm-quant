"""
eval_ppl.py — Базовая perplexity без сжатия (точка отсчёта).
Результат → results/ppl_base.json
"""

import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from model_loader import load_model, BASE_DIR, DEVICE, DTYPE

RESULTS = BASE_DIR / "results"
RESULTS.mkdir(exist_ok=True)


def get_eval_text() -> str:
    """wikitext-2 test split или fallback."""
    try:
        print("\n  ↓ Loading wikitext-2...")
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
        parts = [x for x in data["text"] if len(x.strip()) > 80][:200]
        text = "\n\n".join(parts)
        print(f"     {len(text):,} chars")
        return text
    except Exception as e:
        print(f"     wikitext failed ({e}), using fallback")
        return ("Artificial intelligence is a field of computer science focused on building intelligent systems. " * 80)


def compute_perplexity(model, tokenizer, text, max_len=1024):
    """Perplexity с overlapping windows + прогресс-бар."""
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[0]
    seq_len = ids.size(0)

    total_nll = 0.0
    total_tok = 0
    chunks = []

    for begin in range(0, seq_len - max_len + 1, max_len):
        end = min(begin + max_len, seq_len)
        if end - begin >= 32:
            chunks.append((begin, end))

    # Последний чанк
    if chunks and chunks[-1][1] < seq_len:
        remain = seq_len - chunks[-1][1]
        if remain >= 32:
            chunks.append((chunks[-1][1], seq_len))

    print(f"\n  Evaluating {len(chunks)} chunks ({seq_len:,} tokens)...")

    for begin, end in tqdm(chunks, desc="PPL", unit="chunk", ncols=80):
        chunk = ids[begin:end].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(chunk, labels=chunk)
        n = end - begin
        total_nll += out.loss.item() * n
        total_tok += n

    avg = total_nll / max(1, total_tok)
    return math.exp(avg)


def main():
    print("=" * 60)
    print("  CMQ — Baseline Perplexity")
    print("=" * 60)

    # ── Загрузка модели (с прогрессом + ретраями) ─────────────
    from model_loader import load_tokenizer
    tokenizer = load_tokenizer()
    model = load_model()

    text = get_eval_text()
    ppl = compute_perplexity(model, tokenizer, text)

    result = {
        "model": "HuggingFaceTB/SmolLM-135M",
        "device": DEVICE,
        "perplexity": round(ppl, 4),
        "eval_text_length": len(text),
    }

    print("\n" + "=" * 60)
    print(f"  Baseline PPL: {ppl:.4f}")

    out = RESULTS / "ppl_base.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
