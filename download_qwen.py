"""Download and verify Qwen2.5-0.5B for CHMC v2."""
import sys
import json

print(">>> Starting Qwen download...", flush=True)
sys.stdout.flush()

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("[1/3] Downloading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", resume_download=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"  Tokenizer vocab size: {tok.vocab_size}", flush=True)

print("[2/3] Downloading model weights...", flush=True)
mdl = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.float32,
    device_map="cpu",
    output_hidden_states=True,
    resume_download=True,
)
mdl.eval()

n_params = sum(p.numel() for p in mdl.parameters())
print(f"[3/3] Params: {n_params:,}", flush=True)

cfg = mdl.config
print(f"  hidden_size: {cfg.hidden_size}")
print(f"  num_hidden_layers: {cfg.num_hidden_layers}")
print(f"  intermediate_size: {getattr(cfg, 'intermediate_size', 'N/A')}")
print(f"  num_attention_heads: {cfg.num_attention_heads}", flush=True)

# Quick PPL test
import math
text = "The quick brown fox jumps over the lazy dog. Artificial intelligence is transforming technology."
ids = tok(text, return_tensors="pt").input_ids
out = mdl(ids)
ppl = math.exp(out.loss.item())
print(f"  Quick PPL: {ppl:.2f}", flush=True)

# Save checkpoint
from pathlib import Path
_CHECK = Path(__file__).resolve().parent / "models" / "qwen_check.json"
_CHECK.parent.mkdir(parents=True, exist_ok=True)
with open(_CHECK, "w") as f:
    json.dump({
        "ok": True,
        "params": n_params,
        "hidden_size": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
    }, f)

print("[OK] Qwen2.5-0.5B downloaded and verified!", flush=True)
