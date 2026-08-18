"""
lowrank_eval.py — Чистое low-rank сжатие без квантования.
Результат → results/lowrank_eval.json
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


def get_eval_text() -> str:
    try:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
        parts = [x for x in data["text"] if len(x.strip()) > 80][:200]
        return "\n\n".join(parts)
    except Exception:
        return ("Low-rank approximation decomposes weight matrices into smaller factors. " * 200)


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


def low_rank_approx(w, rank):
    """Truncated SVD: W ≈ U[:,:r]·S[:r] @ Vh[:r,:]"""
    if rank >= min(w.shape):
        return w
    wf = w.float().cpu()
    U, S, Vh = torch.linalg.svd(wf, full_matrices=False)
    return ((U[:, :rank] * S[:rank]) @ Vh[:rank, :]).to(w.dtype).to(w.device)


def main():
    print("=" * 60)
    print("  CMQ — Low-Rank Evaluation")
    print("=" * 60)

    tokenizer = load_tokenizer()
    text = get_eval_text()

    # Определяем допустимые ранги
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

    all_ranks = [8, 16, 32, 64, 128]
    ranks = [r for r in all_ranks if r < (min_dim or 256)]
    print(f"\n  Min matrix dim: {min_dim}, testing ranks: {ranks}")

    # Baseline PPL
    base_path = RESULTS / "ppl_base.json"
    base_ppl = None
    if base_path.exists():
        with open(base_path) as f:
            base_ppl = json.load(f)["perplexity"]
        print(f"  Baseline PPL: {base_ppl}")

    results = []

    for rank in ranks:
        print(f"\n{'-' * 50}")
        print(f"  rank={rank}")

        model = load_model()

        # Собираем матрицы для сжатия
        matrices = []
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "embed" not in name and "lm_head" not in name:
                if min(mod.weight.shape) > rank:
                    matrices.append((name, mod))

        print(f"  Compressing {len(matrices)} matrices...")
        ratios = []

        with torch.no_grad():
            for name, mod in tqdm(matrices, desc=f"LR{rank}", ncols=80):
                w = mod.weight.data
                m, n = w.shape
                orig_params = m * n
                lr_params = rank * (m + n)
                ratios.append(orig_params / max(1, lr_params))
                mod.weight.data = low_rank_approx(w, rank).to(DEVICE)

        ppl = compute_perplexity(model, tokenizer, text)
        mean_ratio = sum(ratios) / max(1, len(ratios))

        entry = {
            "method": f"lowrank_{rank}",
            "rank": rank,
            "perplexity": round(ppl, 4),
            "mean_compression_ratio": round(mean_ratio, 3),
            "compressed_matrices": len(matrices),
        }
        if base_ppl:
            entry["ppl_degradation_pct"] = round((ppl - base_ppl) / base_ppl * 100, 2)
            entry["ppl_ratio"] = round(ppl / base_ppl, 4)

        results.append(entry)
        print(f"  PPL={ppl:.2f}  compression={mean_ratio:.2f}x{'  delta=' + f'{(ppl-base_ppl)/base_ppl*100:+.1f}' + '%' if base_ppl else ''}")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    out = RESULTS / "lowrank_eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
