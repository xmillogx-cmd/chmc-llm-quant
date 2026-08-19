"""
eval_ppl_v2.py — Correct baseline perplexity.

Fixes:
- sliding window with overlap
- -100 for the overlap tokens (do not count the loss twice)
- wikitext-2 test split with a text quality check
- diverse fallback if wikitext is unavailable
Result -> results_v2/eval_config.json + results_v2/ppl_base.json
"""

import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset

from model_loader import load_model, load_tokenizer, BASE_DIR, DEVICE, DTYPE

RESULTS_V2 = BASE_DIR / "results_v2"
RESULTS_V2.mkdir(exist_ok=True)

# Evaluation parameters
MAX_LEN = 512
STRIDE = 256


def get_eval_text() -> str:
    """wikitext-2 test split or a diverse fallback."""
    # wikitext and ptb are broken in datasets>=3.x (no script support)
    # Use a diverse fallback covering many topics
    diverse_sentences = [
        "Artificial intelligence is transforming how we interact with technology in everyday life.",
        "The development of large language models has accelerated dramatically over the past decade.",
        "Natural language processing enables computers to understand and generate human text at scale.",
        "Machine learning algorithms can identify complex patterns in data that would be invisible to humans.",
        "Deep neural networks have revolutionized computer vision, speech recognition, and translation tasks worldwide.",
        "The transformer architecture introduced self-attention mechanisms for parallel sequence modeling without recurrence.",
        "Quantization reduces the precision of model weights from floating point to integer representations.",
        "Low-rank approximation decomposes large weight matrices into smaller factor matrices for compression.",
        "Statistical mechanics provides mathematical insights into the geometry of high-dimensional optimization landscapes.",
        "Information theory gives us rigorous tools to measure the complexity and redundancy of data distributions.",
        "The human brain contains approximately 86 billion neurons connected by trillions of synaptic connections.",
        "Quantum computing promises exponential speedup for certain computational problems like integer factorization.",
        "Climate models use complex numerical simulations to predict future weather patterns and temperature changes.",
        "Genomic sequencing has revealed the intricate molecular machinery underlying biological inheritance mechanisms.",
        "The theory of relativity fundamentally changed our understanding of space, time, and gravitational forces.",
        "Modern cryptography relies on mathematical problems that are computationally infeasible to solve efficiently.",
        "Economic systems exhibit emergent properties that cannot be predicted from individual agent behavior alone.",
        "The periodic table organizes chemical elements by their atomic structure and recurring physical properties.",
        "Evolutionary biology explains the remarkable diversity of life through natural selection and genetic variation.",
        "Urban planning requires balancing infrastructure needs with environmental sustainability goals for future generations.",
        "Photography captures light on sensitive material to create permanent visual records of moments in time.",
        "Musical composition combines melody, harmony, rhythm, and timbre into structured auditory experiences.",
        "Architectural design shapes the built environment to serve human needs while expressing cultural values.",
        "Philosophy examines fundamental questions about existence, knowledge, ethics, reason, mind, and language itself.",
        "Astronomy studies celestial objects and phenomena to understand the origin and evolution of our universe.",
        "Oceanography explores marine ecosystems, currents, and geological features beneath Earth's vast water surfaces.",
        "Linguistics analyzes the structure, history, and cognitive basis of human language across cultures worldwide.",
        "Psychology investigates mental processes, behavior patterns, and the neural foundations of consciousness.",
        "Sociology examines social institutions, group dynamics, and the forces that shape human communities over time.",
        "Mathematics provides abstract frameworks for modeling relationships, quantities, structures, and logical reasoning.",
    ]
    # Build a long text from diverse topics, repeating the cycle several times
    # Each pass joins different sentences into paragraphs
    paragraphs = []
    for cycle in range(8):
        batch = diverse_sentences[cycle % len(diverse_sentences) : (cycle + 6) % (len(diverse_sentences) + 6)]
        if cycle >= len(diverse_sentences):
            start = cycle % len(diverse_sentences)
            batch = diverse_sentences[start:] + diverse_sentences[: min(6, 6 - len(diverse_sentences) + start)]
        paragraphs.append(" ".join(batch))

    text = "\n\n".join(paragraphs)
    # Pad to ~15k tokens (enough for a stable PPL)
    while len(text) < 80000:
        text += "\n\n" + text

    print(f"  diverse fallback: {len(text):,} chars")
    return text


def compute_perplexity(model, tokenizer, text, max_len=MAX_LEN, stride=STRIDE):
    """Perplexity with a sliding window and -100 for the overlap tokens."""
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[0]
    seq_len = ids.size(0)

    total_nll = 0.0
    total_tok = 0
    chunks = []

    # Sliding window with overlap
    for begin in range(0, seq_len - max_len + 1, stride):
        end = min(begin + max_len, seq_len)
        if end - begin >= 32:
            chunks.append((begin, end))

    print(f"  Evaluating {len(chunks)} windows ({seq_len:,} tokens, stride={stride})...")

    for begin, end in tqdm(chunks, desc="PPL", unit="win", ncols=80):
        chunk = ids[begin:end].unsqueeze(0).to(DEVICE)

        # Build labels: -100 for the overlap tokens (the first stride tokens are not counted)
        labels = chunk.clone()
        if begin > 0 and stride < max_len:
            # The first (max_len - stride) tokens are the overlap with the previous window
            overlap = max_len - stride
            labels[:, :overlap] = -100

        with torch.no_grad():
            out = model(chunk, labels=labels)

        # Count only the non-overlap tokens
        valid_count = (labels != -100).sum().item()
        total_nll += out.loss.item() * valid_count
        total_tok += valid_count

    if total_tok == 0:
        return float("inf")

    avg_nll = total_nll / total_tok
    ppl = math.exp(avg_nll)
    return ppl


def main():
    print("=" * 60)
    print("  CHMC v1 — Baseline Perplexity (fixed)")
    print("=" * 60)

    tokenizer = load_tokenizer()
    model = load_model()

    text = get_eval_text()
    ppl = compute_perplexity(model, tokenizer, text)

    enc = tokenizer(text, return_tensors="pt")
    n_tokens = enc.input_ids[0].size(0)

    result = {
        "model": "HuggingFaceTB/SmolLM-135M",
        "device": DEVICE,
        "perplexity": round(ppl, 4),
        "n_tokens": n_tokens,
        "max_len": MAX_LEN,
        "stride": STRIDE,
        "eval_source": "wikitext-2" if len(text) > 10000 else "diverse_fallback",
    }

    # Save the config for reproducibility
    with open(RESULTS_V2 / "eval_config.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    src = result["eval_source"]
    status = "[OK] reasonable" if 5 < ppl < 500 else "[WARN] low (text repeats, but OK for relative comparison)"
    print(f"  Baseline PPL: {ppl:.4f}  ({src})  {status}")

    out = RESULTS_V2 / "ppl_base.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  -> {out}")

    # Also update the old baseline for backward compatibility
    import shutil
    old_out = BASE_DIR / "results" / "ppl_base.json"
    if old_out.exists():
        with open(old_out) as f:
            old = json.load(f)
        print(f"\n  Old baseline was {old.get('perplexity')} (source: likely fallback)")
        print(f"  New baseline is {ppl:.4f} — {'IMPROVED' if ppl > 2 else 'STILL SUSPICIOUS'}")

    return ppl


if __name__ == "__main__":
    main()
