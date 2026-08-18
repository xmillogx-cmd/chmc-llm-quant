import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_article_figures import _act2d, _token_idx, TDA_DIR, N_PTS, MODELS

for model, _ in MODELS:
    payload = torch.load(TDA_DIR / f"tda_activations_{model}.pt", map_location="cpu")
    pre_t, post_t = payload["pre"], payload["post"]
    top3, cosm = [], []
    for bi in range(len(pre_t)):
        A, B = _act2d(pre_t[bi]), _act2d(post_t[bi])
        idx = _token_idx(A.shape[0], N_PTS)
        X = np.vstack([A[idx], B[idx]])
        S = np.linalg.svd(X - X.mean(0), full_matrices=False)[1]
        top3.append(float((S[:3] ** 2).sum() / (S ** 2).sum()))
        Pn = A[idx] / np.maximum(np.linalg.norm(A[idx], axis=1, keepdims=True), 1e-12)
        G = Pn @ Pn.T
        cosm.append(float((G.sum() - len(G)) / (len(G) * (len(G) - 1))))
    print(f"{model}: top3 peak blk{int(np.argmax(top3))}={max(top3):.4f} | "
          f"cos peak blk{int(np.argmax(cosm))}={max(cosm):.4f} | edges top3[0]={top3[0]:.3f}, top3[-1]={top3[-1]:.3f}")
