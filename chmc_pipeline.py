"""
chmc_pipeline.py — Covariance/Hessian-Aware Manifold Compression v1

Implements 4 compression methods:
  1. Plain SVD (baseline for comparison)
  2. Covariance Projection
  3. Weighted SVD
  4. Covariance Low-Rank + Residual Quantization

Results -> results_v2/
"""

import csv
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from model_loader import load_model, load_tokenizer, BASE_DIR, DEVICE, DTYPE

RESULTS_V2 = BASE_DIR / "results_v2"
RESULTS_V2.mkdir(exist_ok=True)


# ============================================================
# Evaluation (copy from eval_ppl_v2.py)
# ============================================================

MAX_LEN = 512
STRIDE = 256


def get_eval_text() -> str:
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
    paragraphs = []
    n = len(diverse_sentences)
    for cycle in range(8):
        start = (cycle * 4) % n
        batch = diverse_sentences[start:] + diverse_sentences[:n]
        paragraphs.append(" ".join(batch[:6]))
    text = "\n\n".join(paragraphs)
    while len(text) < 80000:
        text += "\n\n" + text
    return text


def compute_perplexity(model, tokenizer, text):
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[0]
    seq_len = ids.size(0)

    total_nll = 0.0
    total_tok = 0

    for begin in range(0, seq_len - MAX_LEN + 1, STRIDE):
        end = min(begin + MAX_LEN, seq_len)
        if end - begin < 32:
            break
        chunk = ids[begin:end].unsqueeze(0).to(DEVICE)
        labels = chunk.clone()
        if begin > 0 and STRIDE < MAX_LEN:
            overlap = MAX_LEN - STRIDE
            labels[:, :overlap] = -100
        with torch.no_grad():
            out = model(chunk, labels=labels)
        valid_count = (labels != -100).sum().item()
        total_nll += out.loss.item() * valid_count
        total_tok += valid_count

    if total_tok == 0:
        return float("inf")
    return math.exp(total_nll / total_tok)


# ============================================================
# Calibration — collect inputs via hooks
# ============================================================

def collect_calibration_inputs(model, tokenizer, num_sequences=128):
    """Collect inputs for each Linear layer."""
    texts = [
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
    ]

    inputs_by_name = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            x = input[0].detach().float().cpu()
            x = x.reshape(-1, x.shape[-1])
            if name not in inputs_by_name:
                inputs_by_name[name] = []
            inputs_by_name[name].append(x)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "embed" in name or "lm_head" in name:
                continue
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run the calibration data through the model
    all_text = "\n\n".join(texts * (num_sequences // len(texts) + 1))
    enc = tokenizer(all_text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        # Split into batches to avoid overflowing memory
        batch_size = 32
        ids = enc.input_ids[0]
        for b in range(0, len(ids), batch_size):
            e = min(b + batch_size, len(ids))
            chunk = ids[b:e].unsqueeze(0).to(DEVICE)
            model(chunk)

    for h in hooks:
        h.remove()

    # Concatenate and sample down to 50k tokens per layer
    MAX_TOKENS = 50000
    final_inputs = {}
    total_layers = len(inputs_by_name)
    print(f"  Collected inputs for {total_layers} layers")

    for name, tensors in tqdm(inputs_by_name.items(), desc="Calib", ncols=80):
        X = torch.cat(tensors, dim=0)
        if X.shape[0] > MAX_TOKENS:
            idx = torch.randperm(X.shape[0])[:MAX_TOKENS]
            X = X[idx]
        final_inputs[name] = X

    return final_inputs


# ============================================================
# Covariance statistics
# ============================================================

def compute_cov_stats(inputs_by_name, lam=1e-3):
    """Compute covariance spectra for each layer."""
    stats = []

    for name, X in inputs_by_name.items():
        C = X.T @ X / X.shape[0]
        C = C + lam * torch.eye(C.shape[0], device=C.device)

        vals = torch.linalg.eigvalsh(C).sort(descending=True).values
        total_var = vals.sum()
        cumvar = torch.cumsum(vals, dim=0) / (total_var + 1e-8)

        d90 = int((cumvar >= 0.90).float().argmax().item()) + 1
        d95 = int((cumvar >= 0.95).float().argmax().item()) + 1
        d99 = int((cumvar >= 0.99).float().argmax().item()) + 1

        top16_e = float(vals[:16].sum() / total_var) if len(vals) >= 16 else 1.0
        top32_e = float(vals[:32].sum() / total_var) if len(vals) >= 32 else 1.0
        top64_e = float(vals[:64].sum() / total_var) if len(vals) >= 64 else 1.0

        # Effective rank: exp(H) where H is the entropy of the normalized eigenvalues
        probs = (vals / (total_var + 1e-8)).clamp(min=1e-10)
        eff_rank = float(torch.exp(-(probs * probs.log()).sum()))

        stats.append({
            "layer": name,
            "in_features": X.shape[1],
            "n_tokens": X.shape[0],
            "d90": d90,
            "d95": d95,
            "d99": d99,
            "top16_energy": round(top16_e, 4),
            "top32_energy": round(top32_e, 4),
            "top64_energy": round(top64_e, 4),
            "effective_rank": round(eff_rank, 1),
        })

    return stats


# ============================================================
# Compression methods
# ============================================================

def plain_svd(W, rank):
    """Method 1: plain SVD."""
    if rank >= min(W.shape):
        return W.clone()
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    A = U[:, :rank] * S[:rank].unsqueeze(0)
    return (A @ Vh[:rank, :]).to(W.dtype)


def covariance_projection(W, X, rank, lam=1e-3):
    """Method 2: Covariance Projection — W_hat = W P_r P_r^T."""
    if rank >= min(W.shape):
        return W.clone()

    C = X.T @ X / X.shape[0]
    C = C + lam * torch.eye(C.shape[0], device=C.device)

    vals, vecs = torch.linalg.eigh(C.float())
    idx = torch.argsort(vals, descending=True)
    P = vecs[:, idx[:rank]]  # [in_features, rank]

    W_hat = W.float() @ P @ P.T
    return W_hat.to(W.dtype)


def weighted_svd_compress(W, X, rank, lam=1e-3):
    """Method 3: Weighted SVD — SVD(W * C_x^{1/2})."""
    if rank >= min(W.shape):
        return W.clone()

    C = X.T @ X / X.shape[0]
    C = C + lam * torch.eye(C.shape[0], device=C.device)

    vals, vecs = torch.linalg.eigh(C.float())
    vals = vals.clamp(min=1e-8)

    sqrt_vals = torch.sqrt(vals)
    inv_sqrt_vals = 1.0 / sqrt_vals

    sqrtC = vecs @ torch.diag(sqrt_vals) @ vecs.T
    inv_sqrtC = vecs @ torch.diag(inv_sqrt_vals) @ vecs.T

    M = W.float() @ sqrtC
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    A_r = U[:, :rank] * S[:rank].unsqueeze(0)  # [out, rank]
    W_hat = (A_r @ Vh[:rank, :] @ inv_sqrtC)
    return W_hat.to(W.dtype)


def quantize_symmetric(w, bits):
    """Symmetric per-channel quantization."""
    if bits >= 16:
        return w.clone()
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    wf = w.float()
    scale = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(wf / scale).clamp(qmin, qmax) * scale


def cov_proj_residual_q(W, X, rank, bits, lam=1e-3):
    """Method 4: Covariance Projection + Residual Quantization."""
    W_low = covariance_projection(W, X, rank, lam)
    R = W.float() - W_low
    R_q = quantize_symmetric(R, bits)
    return (W_low + R_q).to(W.dtype)


def weighted_svd_residual_q(W, X, rank, bits, lam=1e-3):
    """Method 4b: Weighted SVD + Residual Quantization."""
    W_low = weighted_svd_compress(W, X, rank, lam)
    R = W.float() - W_low
    R_q = quantize_symmetric(R, bits)
    return (W_low + R_q).to(W.dtype)


# ============================================================
# Layer metrics
# ============================================================

def compute_layer_metrics(W_orig, W_compressed, X):
    """Compute the layerwise metrics."""
    Y_true = W_orig.float() @ X.T  # [out, N]
    Y_hat = W_compressed.float() @ X.T

    # Output reconstruction error
    output_err = float(((Y_true - Y_hat).norm() / (Y_true.norm() + 1e-8)).item())

    # Cosine similarity of the outputs
    cos_sims = []
    for i in range(min(100, Y_true.shape[1])):
        c = float((Y_true[:, i] * Y_hat[:, i]).sum() /
                  ((Y_true[:, i].norm() + 1e-8) * (Y_hat[:, i].norm() + 1e-8)))
        cos_sims.append(c)
    cos_sim_mean = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0

    # Relative weight error
    weight_err = float(((W_orig - W_compressed).norm() / (W_orig.norm() + 1e-8)).item())

    return output_err, cos_sim_mean, weight_err


def compute_compression_bits(W_orig, method_info):
    """Count the bits of the compressed representation."""
    out_f, in_f = W_orig.shape
    orig_bits = out_f * in_f * 16

    if method_info["method"] == "plain_svd" or method_info["method"] == "cov_proj":
        rank = method_info["rank"]
        # low-rank: out*r + r*in stored in FP32
        comp_bits = (out_f * rank + rank * in_f) * 16
    elif method_info["method"].startswith("weighted_svd"):
        rank = method_info["rank"]
        bits = method_info.get("bits", 16)
        # low-rank factors in bits + covariance basis storage
        comp_bits = (out_f * rank + rank * in_f) * bits
    elif method_info["method"].startswith("cov_proj_residual"):
        rank = method_info["rank"]
        bits = method_info.get("bits", 4)
        # W_low in FP16 + residual in bits
        comp_bits = (out_f * rank + rank * in_f) * 16 + out_f * in_f * bits
    elif method_info["method"].startswith("weighted_svd_residual"):
        rank = method_info["rank"]
        bits = method_info.get("bits", 4)
        comp_bits = (out_f * rank + rank * in_f) * 16 + out_f * in_f * bits
    else:
        comp_bits = orig_bits

    return orig_bits, comp_bits


# ============================================================
# Main loop
# ============================================================

def apply_compression(model, inputs_by_name, method_fn, method_info):
    """Apply the compression method to all layers."""
    matrices_compressed = 0
    layer_metrics = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "embed" in name or "lm_head" in name:
            continue
        if name not in inputs_by_name:
            continue

        W_orig = module.weight.data.clone()
        X = inputs_by_name[name].to(W_orig.device)

        W_compressed = method_fn(W_orig, X, method_info)

        out_err, cos_sim, w_err = compute_layer_metrics(W_orig, W_compressed, X)

        orig_bits, comp_bits = compute_compression_bits(W_orig, method_info)
        ratio = orig_bits / max(1, comp_bits)

        layer_metrics.append({
            "layer": name,
            "output_error": round(out_err, 4),
            "cos_sim": round(cos_sim, 4),
            "weight_error": round(w_err, 4),
            "compression_ratio": round(ratio, 2),
        })

        module.weight.data = W_compressed.to(DEVICE)
        matrices_compressed += 1

    return matrices_compressed, layer_metrics


def main():
    print("=" * 60)
    print("  CHMC v1 — Covariance-Aware Manifold Compression")
    print("=" * 60)

    # Load the model and tokenizer
    tokenizer = load_tokenizer()
    eval_text = get_eval_text()

    # Baseline PPL
    base_path = RESULTS_V2 / "ppl_base.json"
    if not base_path.exists():
        print("\n  [WARN] No baseline found, run eval_ppl_v2.py first")
        return

    with open(base_path) as f:
        base_ppl = json.load(f)["perplexity"]
    print(f"\n  Baseline PPL: {base_ppl}")

    # ── Collecting calibration inputs ──────────────────────────────
    print("\n[1/4] Collecting calibration inputs...")
    model = load_model()
    inputs_by_name = collect_calibration_inputs(model, tokenizer, num_sequences=256)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── Covariance spectra ────────────────────────────────
    print("\n[2/4] Computing covariance statistics...")
    cov_stats = compute_cov_stats(inputs_by_name)

    with open(RESULTS_V2 / "cov_stats.json", "w") as f:
        json.dump(cov_stats, f, indent=2)

    # Statistics across all layers
    d90_vals = [s["d90"] for s in cov_stats]
    eff_ranks = [s["effective_rank"] for s in cov_stats]
    print(f"  Layer count: {len(cov_stats)}")
    print(f"  d90 range: [{min(d90_vals)}, {max(d90_vals)}], median={sorted(d90_vals)[len(d90_vals)//2]}")
    print(f"  effective_rank range: [{min(eff_ranks):.1f}, {max(eff_ranks):.1f}]")

    # ── Compression methods ────────────────────────────────────────
    RANKS = [8, 16, 32]
    BITS_RESIDUAL = [4]  # can be extended to [4, 3, 2]

    methods = []
    for r in RANKS:
        methods.append(("plain_svd", {"rank": r, "bits": 16}))
        methods.append(("cov_proj", {"rank": r, "bits": 16}))
        methods.append(("weighted_svd", {"rank": r, "bits": 16}))

    for r in RANKS:
        for b in BITS_RESIDUAL:
            methods.append(("cov_proj_residual_q" + str(b), {"rank": r, "bits": b}))
            methods.append(("weighted_svd_residual_q" + str(b), {"rank": r, "bits": b}))

    all_results = []
    csv_rows = []

    print(f"\n[3/4] Testing {len(methods)} method configurations...")

    for method_name, info in tqdm(methods, desc="Methods", ncols=80):
        info["method"] = method_name  # needed by compute_compression_bits
        model = load_model()

        if method_name == "plain_svd":
            fn = lambda W, X, r: plain_svd(W, r["rank"])
        elif method_name == "cov_proj":
            fn = lambda W, X, r: covariance_projection(W, X, r["rank"])
        elif method_name.startswith("weighted_svd") and not "residual" in method_name:
            fn = lambda W, X, r: weighted_svd_compress(W, X, r["rank"])
        elif method_name.startswith("cov_proj_residual_q"):
            bits = info["bits"]
            fn = lambda W, X, r, _b=bits: cov_proj_residual_q(W, X, r["rank"], _b)
        elif method_name.startswith("weighted_svd_residual_q"):
            bits = info["bits"]
            fn = lambda W, X, r, _b=bits: weighted_svd_residual_q(W, X, r["rank"], _b)

        n_matrices, layer_metrics = apply_compression(model, inputs_by_name, fn, info)

        # PPL evaluation
        ppl = compute_perplexity(model, tokenizer, eval_text)

        # Average metrics across layers
        avg_output_err = sum(m["output_error"] for m in layer_metrics) / max(1, len(layer_metrics))
        avg_cos_sim = sum(m["cos_sim"] for m in layer_metrics) / max(1, len(layer_metrics))
        avg_comp_ratio = sum(m["compression_ratio"] for m in layer_metrics) / max(1, len(layer_metrics))

        entry = {
            "method": method_name,
            "rank": info["rank"],
            "bits": info.get("bits", 16),
            "perplexity": round(ppl, 4),
            "ppl_ratio_to_base": round(ppl / base_ppl, 2),
            "avg_output_error": round(avg_output_err, 4),
            "avg_cos_sim": round(avg_cos_sim, 4),
            "avg_compression_ratio": round(avg_comp_ratio, 2),
            "compressed_matrices": n_matrices,
        }

        all_results.append(entry)
        csv_rows.append({
            "method": method_name,
            "rank": info["rank"],
            "bits": info.get("bits", 16),
            "compression_ratio": round(avg_comp_ratio, 2),
            "layer_output_error": round(avg_output_err, 4),
            "cos_sim": round(avg_cos_sim, 4),
            "ppl": round(ppl, 4),
            "delta_ppl_pct": round((ppl - base_ppl) / base_ppl * 100, 2),
        })

        status = "[OK]" if ppl < base_ppl * 10 else "[WARN]" if ppl < base_ppl * 100 else "[FAIL]"
        print(f"  {status} {method_name} r={info['rank']} b={info.get('bits',16)}: "
              f"PPL={ppl:.2f} ({entry['ppl_ratio_to_base']:.1f}x) "
              f"cos_sim={avg_cos_sim:.3f} comp={avg_comp_ratio:.1f}x")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ── Saving results ────────────────────────────────
    print("\n[4/4] Saving results...")

    with open(RESULTS_V2 / "compression_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # CSV
    if csv_rows:
        with open(RESULTS_V2 / "compression_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    # Layerwise metrics for the best method
    best = min(all_results, key=lambda e: e["ppl_ratio_to_base"])
    print(f"\n  Best method: {best['method']} PPL={best['perplexity']} "
          f"ratio={best['ppl_ratio_to_base']}x cos_sim={best['avg_cos_sim']:.3f}")

    # ── Summary ───────────────────────────────────────────────
    generate_summary(all_results, base_ppl, cov_stats)


def generate_summary(results, base_ppl, cov_stats):
    """Generate summary.md."""
    best = min(results, key=lambda e: e["ppl_ratio_to_base"])

    # Old results for comparison
    old_lr_path = BASE_DIR / "results" / "lowrank_eval.json"
    old_scalar_path = BASE_DIR / "results" / "quant_scalar.json"

    old_lr_best = None
    if old_lr_path.exists():
        with open(old_lr_path) as f:
            for e in json.load(f):
                if old_lr_best is None or e["perplexity"] < old_lr_best["perplexity"]:
                    old_lr_best = e

    old_scalar_4b = None
    if old_scalar_path.exists():
        with open(old_scalar_path) as f:
            for e in json.load(f):
                if e["bits"] == 4:
                    old_scalar_4b = e

    d90_vals = [s["d90"] for s in cov_stats]
    median_d90 = sorted(d90_vals)[len(d90_vals) // 2]

    lines = []
    lines.append("# CHMC v1 Summary")
    lines.append("")
    lines.append(f"**Date:** {Path(RESULTS_V2).name}")
    lines.append(f"**Model:** HuggingFaceTB/SmolLM-135M")
    lines.append(f"**Baseline PPL:** {base_ppl} (diverse fallback, sliding window)")
    lines.append("")

    lines.append("## 1. Evaluation Fix")
    lines.append("- Baseline PPL: " + f"{base_ppl}")
    lines.append("- Sliding window with stride=256, max_len=512")
    lines.append("- Overlap tokens set to -100 (not counted in loss)")
    lines.append("")

    lines.append("## 2. Calibration Statistics")
    lines.append(f"- Layers analyzed: {len(cov_stats)}")
    lines.append(f"- Covariance d90 median: {median_d90}")
    lines.append(f"- Best covariance rank found: see cov_stats.json")
    lines.append("")

    lines.append("## 3. Best Method")
    lines.append(f"**Method:** `{best['method']}` (rank={best['rank']}, bits={best['bits']})")
    lines.append(f"- PPL: {best['perplexity']}")
    lines.append(f"- PPL ratio to baseline: {best['ppl_ratio_to_base']}x")
    lines.append(f"- Avg cosine similarity: {best['avg_cos_sim']}")
    lines.append(f"- Avg compression ratio: {best['avg_compression_ratio']}x")
    lines.append("")

    lines.append("## 4. Comparison Table")
    lines.append("| Method | Rank | PPL | PPL/Base | Cos Sim | Compression |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda e: e["ppl_ratio_to_base"])[:10]:
        lines.append(f"| {r['method']} | {r['rank']} | {r['perplexity']:.2f} "
                     f"| {r['ppl_ratio_to_base']:.1f}x | {r['avg_cos_sim']:.3f} "
                     f"| {r['avg_compression_ratio']:.1f}x |")
    lines.append("")

    if old_lr_best:
        lines.append("## 5. vs Previous Plain Low-Rank")
        lines.append(f"- Old plain SVD best PPL: {old_lr_best['perplexity']}")
        lines.append(f"- New CHMC best PPL: {best['perplexity']}")
        improvement = old_lr_best["perplexity"] / max(1, best["perplexity"])
        lines.append(f"- Improvement factor: {improvement:.1f}x")
        lines.append("")

    if old_scalar_4b:
        lines.append("## 6. vs Scalar 4-bit Baseline")
        lines.append(f"- Old scalar 4-bit PPL: {old_scalar_4b['perplexity']}")
        lines.append(f"- New CHMC best PPL: {best['perplexity']}")
        better = best["perplexity"] < old_scalar_4b["perplexity"]
        lines.append(f"- {'[OK] Better than scalar 4-bit' if better else '[FAIL] Worse than scalar 4-bit'}")
        lines.append("")

    # Verdict
    lines.append("## 7. Verdict")
    ratio = best["ppl_ratio_to_base"]
    cos_sim = best["avg_cos_sim"]

    if ratio < 10 and cos_sim > 0.95:
        verdict = "[GREEN] STRONG SUCCESS — covariance-aware compression works!"
    elif ratio < old_lr_best.get("perplexity", 999) / base_ppl * 100:
        verdict = "[YELLOW] MODERATE — better than plain SVD, but still needs QAT"
    else:
        verdict = "[RED] COVARIANCE insufficient alone — need QAT or distillation"

    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append("## 8. Next Steps")
    if ratio < 10:
        lines.append("- Develop block-wise reconstruction for even better quality")
        lines.append("- Try mixed-precision residual quantization (INT4/INT2)")
    else:
        lines.append("- Implement Quantization-Aware Training (QAT) with covariance-aware initialization")
        lines.append("- Try knowledge distillation from full model to compressed model")

    out = RESULTS_V2 / "summary.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
