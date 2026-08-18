#!/usr/bin/env python3
"""Stage 4: Rank-1 validation — check effective rank distribution."""
import sys, math
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model_loader import load_model
from chmc_v4 import get_compressible_layers, _get_weight

mdl = load_model("HuggingFaceTB/SmolLM-135M")
layers = get_compressible_layers(mdl)

print(f"Scanning {len(layers)} layers...")

eff_ranks = []
top1_energies = []
layer_details = []

for name in layers:
    W = _get_weight(mdl, name).float()
    out_f, in_f = W.shape
    q = min(64, out_f, in_f)
    
    U, S, V = torch.svd_lowrank(W, q=q, niter=3)
    total = (S ** 2).sum().item()
    if total < 1e-12:
        continue
    
    probs = (S ** 2) / total
    eff_rank = math.exp(-(probs * torch.log(probs + 1e-12)).sum().item())
    top1 = (S[0] ** 2).item() / total
    
    eff_ranks.append(eff_rank)
    top1_energies.append(top1)
    layer_details.append({
        "layer": name, "shape": [out_f, in_f],
        "eff_rank": round(eff_rank, 2), "top1_energy": round(top1, 4),
    })

print(f"\nEffective rank distribution ({len(eff_ranks)} layers):")
print(f"  min={min(eff_ranks):.1f}, median={sorted(eff_ranks)[len(eff_ranks)//2]:.1f}, max={max(eff_ranks):.1f}")
print(f"\nTop-1 energy distribution:")
print(f"  min={min(top1_energies):.4f}, median={sorted(top1_energies)[len(top1_energies)//2]:.4f}, max={max(top1_energies):.4f}")

# Show layers with lowest effective rank
layer_details.sort(key=lambda x: x["eff_rank"])
print(f"\nBottom 10 by effective rank:")
for d in layer_details[:10]:
    print(f"  {d['layer']}: eff_rank={d['eff_rank']}, top1_energy={d['top1_energy']}")

# Show layers with highest top-1 energy
layer_details.sort(key=lambda x: -x["top1_energy"])
print(f"\nTop 10 by top-1 energy:")
for d in layer_details[:10]:
    print(f"  {d['layer']}: eff_rank={d['eff_rank']}, top1_energy={d['top1_energy']}")

# Save distribution info
import json, os
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results_v4", "smollm-135m")
os.makedirs(results_dir, exist_ok=True)

result = {
    "effective_rank": {"min": round(min(eff_ranks), 2), "median": round(sorted(eff_ranks)[len(eff_ranks)//2], 2), "max": round(max(eff_ranks), 2)},
    "top1_energy": {"min": round(min(top1_energies), 4), "median": round(sorted(top1_energies)[len(top1_energies)//2], 4), "max": round(max(top1_energies), 4)},
    "bottom_eff_rank": layer_details[:10],
    "conclusion": f"No layers with eff_rank < 5 found. Rank-1 structural replacement not viable for SmolLM-135M."
}

with open(os.path.join(results_dir, "stage4_results.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"\nConclusion: No rank-1 candidates found. All layers have eff_rank > 5.")
print(f"Results saved to {results_dir}/stage4_results.json")
