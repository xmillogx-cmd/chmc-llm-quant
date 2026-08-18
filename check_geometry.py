"""
check_geometry.py — Геометрия активаций (конус, анизотропия, PCA).
Без обучения. Результат → results/geometry_activations.json
"""

# UTF-8 output для Windows консоли
import sys; sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model_loader import load_model_and_tokenizer, BASE_DIR, DEVICE, DTYPE

RESULTS = BASE_DIR / "results"
RESULTS.mkdir(exist_ok=True)

TEXTS = [
    "Artificial intelligence is a field of computer science focused on building systems that can perform tasks requiring human-like intelligence.",
    "Large language models are trained on massive amounts of text data and learn statistical patterns of language through attention mechanisms.",
    "Quantization reduces the precision of neural network weights and activations to make models smaller and faster for deployment.",
    "The geometry of hidden states may contain low-dimensional structures such as cones or manifolds that enable extreme compression.",
    "Machine learning models optimize parameters by minimizing a loss function over a training dataset using gradient descent.",
    "Natural language processing enables computers to understand, interpret, manipulate, and generate human language in various forms.",
    "Neural networks consist of layers of interconnected artificial neurons that progressively transform input representations into outputs.",
    "Extreme compression is possible if the model contains significant redundancy or if its internal representations lie on a low-dimensional manifold.",
    "Transformers use self-attention mechanisms to capture long-range dependencies in sequential data without recurrent connections.",
    "Deep learning has revolutionized computer vision, speech recognition, and natural language processing through hierarchical feature learning.",
    # Дополнительные тексты для достаточного количества сэмплов
    "The development of artificial neural networks was inspired by the structure of biological brains, where neurons are connected through synapses that can strengthen or weaken over time.",
    "Backpropagation is an algorithm for calculating gradients in neural networks by applying the chain rule of calculus to compute how each weight contributes to the final error.",
    "Recurrent neural networks process sequential data by maintaining a hidden state that captures information from previous time steps, enabling them to model temporal dependencies.",
    "Convolutional neural networks apply learnable filters across spatial dimensions, making them highly effective for image recognition and other grid-structured data tasks.",
    "Transfer learning involves taking a pre-trained model on one task and fine-tuning it on a different but related task, often requiring far less training data from scratch.",
    "Attention mechanisms allow models to weigh the importance of different parts of the input when producing each element of the output, enabling better handling of long-range dependencies.",
    "Regularization techniques such as dropout, weight decay, and early stopping help prevent overfitting by constraining model complexity or introducing noise during training.",
    "Batch normalization normalizes layer inputs to reduce internal covariate shift, allowing higher learning rates and faster convergence during training deep networks.",
    "The vanishing gradient problem occurs when gradients become extremely small during backpropagation through many layers, making it difficult for early layers to learn effectively.",
    "Transformer architectures have largely replaced recurrent models for sequence tasks due to their parallelizable self-attention mechanism and superior performance on long sequences.",
    "Pre-training involves training a model on a large unlabeled dataset using self-supervised objectives like masked language modeling or next sentence prediction before fine-tuning.",
    "Embedding layers map discrete tokens to continuous vector spaces where semantic relationships between words are captured through geometric proximity in the embedding space.",
    "Positional encodings provide transformers with information about token order since the self-attention mechanism itself is permutation-invariant and cannot distinguish sequences from sets.",
    "Layer normalization normalizes activations across the feature dimension within each sample, improving training stability especially for transformer architectures without batch dependencies.",
]


def main():
    print("=" * 60)
    print("  CMQ — Activation Geometry")
    print("=" * 60)

    # ── Загрузка модели (с прогрессом + ретраями) ──────────────
    tokenizer, model = load_model_and_tokenizer()

    # ── Forward pass ───────────────────────────────────────────
    print("\n[1/3]  Forward pass -> сбор активаций...")
    enc = tokenizer(TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)

    h = out.hidden_states[-1].float().cpu().numpy()[0]
    mask = enc.attention_mask[0].bool().cpu().numpy()
    h = h[mask]

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    hidden_dim = h.shape[1]
    print(f"       {h.shape[0]} токенов × {hidden_dim} dim")

    # ── Сэмплирование ──────────────────────────────────────────
    n_samples = min(5000, h.shape[0])
    idx = np.random.choice(h.shape[0], size=n_samples, replace=h.shape[0] < n_samples)
    X = h[idx]

    # ── Анизотропия ────────────────────────────────────────────
    print("[2/3]  Анизотропия...")
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms

    u = Xn.mean(axis=0)
    u_norm = np.linalg.norm(u)
    anis_global = float(np.mean(Xn @ (u / u_norm))) if u_norm > 1e-8 else 0.0

    n_pairs = min(10000, Xn.shape[0] ** 2 // 2)
    pair_idx = np.random.randint(0, Xn.shape[0], size=(n_pairs, 2))
    anis_pairwise = float(np.mean(np.sum(Xn[pair_idx[:, 0]] * Xn[pair_idx[:, 1]], axis=1)))

    # ── PCA ────────────────────────────────────────────────────
    print("[3/3]  PCA decomposition...")
    max_comp = min(512, X.shape[0], X.shape[1])  # не больше сэмплов и фичей
    from sklearn.decomposition import PCA
    pca = PCA(n_components=max_comp, svd_solver="full").fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    d90 = int(np.searchsorted(cumvar, 0.90) + 1)
    d95 = int(np.searchsorted(cumvar, 0.95) + 1)
    d99 = int(np.searchsorted(cumvar, 0.99) + 1)

    top10 = float(pca.explained_variance_ratio_[:max(1, max_comp // 10)].sum())
    top25 = float(pca.explained_variance_ratio_[:max(1, max_comp // 4)].sum())

    # ── Результат ──────────────────────────────────────────────
    cone = "[FIRE] strong" if anis_global > 0.8 else "[OK] moderate" if anis_global > 0.5 else "[WARN] weak" if anis_global > 0.2 else "[FAIL] isotropic"

    result = {
        "model": "HuggingFaceTB/SmolLM-135M",
        "device": DEVICE,
        "hidden_dim": hidden_dim,
        "n_tokens_used": n_samples,
        "anisotropy_global": round(anis_global, 6),
        "anisotropy_pairwise": round(anis_pairwise, 6),
        "pca_d90": d90, "pca_d95": d95, "pca_d99": d99,
        "pca_components_used": max_comp,
        "top10_pct_variance": round(top10, 6),
        "top25_pct_variance": round(top25, 6),
        "cone_strength": cone,
        "low_dim_ratio": round(d90 / hidden_dim, 4),
    }

    print("\n" + "=" * 60)
    print(f"  anisotropy_global   : {anis_global:.4f}  ({cone})")
    print(f"  anisotropy_pairwise : {anis_pairwise:.4f}")
    print(f"  PCA d90             : {d90}/{hidden_dim}  ratio={d90/hidden_dim:.3f}")
    print(f"  PCA d95/d99         : {d95} / {d99}")
    print(f"  top10% variance     : {top10:.4f}")
    print(f"  top25% variance     : {top25:.4f}")

    ag, ld = anis_global, d90 / hidden_dim
    if ag > 0.5 and ld < 0.33:
        print("\n  [OK] Geometry CONFIRMED")
    elif ag > 0.2 or ld < 0.5:
        print("\n  [WARN] Partially confirmed")
    else:
        print("\n  [FAIL] Geometry WEAK")

    out = RESULTS / "geometry_activations.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
