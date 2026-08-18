#!/usr/bin/env python3
"""Sanity: CPU svd_lowrank + local model load + numpy SVD speed."""
import sys, time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "v6"
sys.path.insert(0, str(BASE))

import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())

t0 = time.time()
A = torch.randn(1536, 576)
U, S, V = torch.svd_lowrank(A, q=4, niter=5)
err = (A - (U * S.unsqueeze(0)) @ V.T).norm().item() / A.norm().item()
print(f"svd_lowrank CPU ok: {time.time()-t0:.2f}s  rel_err={err:.3e}")

import numpy as np
D = np.random.randn(2048, 1536)
t0 = time.time()
s = np.linalg.svd(D, compute_uv=False)
print(f"numpy SVD 2048x1536: {time.time()-t0:.2f}s")

from transformers import AutoTokenizer, AutoModelForCausalLM
p = str(Path(__file__).resolve().parents[2] / "models" / "smollm-135m")
tok = AutoTokenizer.from_pretrained(p)
mdl = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32)
n_lin = sum(1 for m in mdl.modules() if isinstance(m, torch.nn.Linear))
print(f"model ok: vocab={len(tok)}  linear_modules={n_lin}")
