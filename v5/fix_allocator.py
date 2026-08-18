#!/usr/bin/env python3
"""
fix_allocator.py — Исправленный budget allocator (BUG-2 fix)
============================================================

Проблема v4: Для target_bw ≤ 4.25 все ранги застревали на min_rank=2.
Корень бага: energy_at_rank интерполировал неточно для малых рангов,
gain_per_bit возвращал 0 или отрицательное значение.

Исправление:
  1. Точная интерполяция энергии через (rank, energy) точки из cov_stats
  2. Greedy allocator с +2/-2 шагами вместо +4/-2
  3. Фаза увеличения рангов до достижения target_bw
  4. Фаза уменьшения только при перерасходе

Usage:
    python v5/fix_allocator.py --cov results_v4/smollm-135m/cov_stats_v4.json
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).parent.resolve()       # v5/
ROOT_DIR = BASE_DIR.parent                        # cmq_experiment/
RESULTS  = ROOT_DIR / "results_v5"
RESULTS.mkdir(parents=True, exist_ok=True)


def load_cov_stats(path: str) -> Dict[str, dict]:
    """Load covariance statistics from JSON (v4 format)."""
    with open(path) as f:
        raw = json.load(f)
    return raw


def energy_at_rank_from_spectrum(stats: dict, r: int) -> float:
    """
    Точная интерполяция энергии по рангу.

    Используем реальные точки из cov_stats: top16/32/64 energy и d90/d95/d99.
    Линейная интерполяция между соседними точками (rank, cumulative_energy).
    """
    r = max(1, r)
    d90  = stats.get("d90", 64)
    d95  = stats.get("d95", 96)
    d99  = stats.get("d99", 128)
    e16  = stats.get("top16_energy", 0.5)
    e32  = stats.get("top32_energy", 0.65)
    e64  = stats.get("top64_energy", 0.8)
    in_f = stats.get("in_features", 576)

    # Build interpolation points: (rank, cumulative_energy)
    points = [
        (1, max(0.005, e16 / 16)),   # rank=1 captures ~1/16 of top-16 energy
        (16, e16),
        (32, e32),
        (64, min(e64, 1.0)),
        (d90, 0.90),
        (d95, 0.95),
        (d99, 0.99),
        (in_f, 1.0),
    ]

    # Deduplicate and sort by rank
    seen = set()
    unique = []
    for rank, energy in points:
        rk = int(rank)
        if rk not in seen:
            seen.add(rk)
            unique.append((rk, energy))
    unique.sort(key=lambda x: x[0])

    # Clamp r to range
    if r <= unique[0][0]:
        return unique[0][1]
    if r >= unique[-1][0]:
        return 1.0

    # Linear interpolation between bracketing points
    for i in range(len(unique) - 1):
        r0, e0 = unique[i]
        r1, e1 = unique[i + 1]
        if r0 <= r <= r1:
            t = (r - r0) / max(r1 - r0, 1)
            return e0 + t * (e1 - e0)

    return 0.5


def honest_bits(out_f: int, in_f: int, rank: int,
                factor_bits: float = 16.0, residual_bits: float = 4.0) -> Tuple[float, float]:
    """Return (original_bits, compressed_bits) for a layer."""
    original = 16.0 * out_f * in_f
    lowrank  = factor_bits * rank * (out_f + in_f)
    residual = residual_bits * out_f * in_f
    scales   = 16.0 * out_f  # per-channel scale
    compressed = lowrank + residual + scales
    return original, max(compressed, 1.0)


def allocate_ranks_fixed(
    cov_stats: dict,
    target_bw: float,
    min_rank: int = 2,
    max_rank: int = 64,
    factor_bits: float = 16.0,
    residual_bits: float = 4.0,
) -> Tuple[Dict[str, int], float]:
    """
    Greedy rank allocator with proper energy interpolation.

    Phase 1: Start all at min_rank, increase where gain/bit is highest.
    Phase 2: If over budget, decrease where loss/bit is lowest.
    Step size: +2/-2 (finer granularity than v4's +4).
    """
    layers = list(cov_stats.keys())

    # Handle nested format from v4 ({"layers": {...}} or flat)
    if "layers" in cov_stats and isinstance(cov_stats["layers"], dict):
        layer_data = cov_stats["layers"]
    else:
        layer_data = cov_stats

    ranks = {name: min_rank for name in layers}

    def compute_bw() -> float:
        total_comp = 0.0
        total_weights = 0
        for name in layers:
            s = layer_data.get(name, {})
            out_f = s.get("out_features", 576)
            in_f  = s.get("in_features", 576)
            _, comp = honest_bits(out_f, in_f, ranks[name], factor_bits, residual_bits)
            total_comp += comp
            total_weights += out_f * in_f
        return total_comp / total_weights if total_weights else 99

    def gain_per_bit(name: str) -> float:
        """Energy gained per bit spent when increasing rank by +2."""
        s = layer_data.get(name, {})
        r = ranks[name]
        if r >= max_rank:
            return -1.0
        out_f = s.get("out_features", 576)
        in_f  = s.get("in_features", 576)
        e_now  = energy_at_rank_from_spectrum(s, r)
        e_next = energy_at_rank_from_spectrum(s, r + 2)
        delta_e = e_next - e_now
        delta_bits = factor_bits * 2.0 * (out_f + in_f)
        return delta_e / delta_bits if delta_bits > 0 else 0.0

    def loss_per_bit(name: str) -> float:
        """Bits saved per unit energy lost when decreasing rank by -2."""
        s = layer_data.get(name, {})
        r = ranks[name]
        if r <= min_rank:
            return 1e9  # can't decrease further → "expensive" to skip
        out_f = s.get("out_features", 576)
        in_f  = s.get("in_features", 576)
        e_now   = energy_at_rank_from_spectrum(s, r)
        e_prev  = energy_at_rank_from_spectrum(s, max(r - 2, min_rank))
        delta_e = e_now - e_prev
        saved_bits = factor_bits * 2.0 * (out_f + in_f)
        # High ratio = we save many bits for little quality loss → good candidate to decrease
        return saved_bits / delta_e if delta_e > 1e-6 else 1e9

    bw = compute_bw()
    iteration = 0

    # Phase 1: Increase ranks until budget is met
    while bw < target_bw - 0.05 and iteration < 5000:
        best = max(layers, key=gain_per_bit)
        if gain_per_bit(best) <= 0:
            break
        ranks[best] = min(ranks[best] + 2, max_rank)
        bw = compute_bw()
        iteration += 1

    # Phase 2: Decrease ranks if over budget
    while bw > target_bw + 0.05 and iteration < 10000:
        best = max(layers, key=loss_per_bit)
        if ranks[best] <= min_rank:
            break
        ranks[best] = max(ranks[best] - 2, min_rank)
        bw = compute_bw()
        iteration += 1

    return ranks, compute_bw()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cov", default=str(ROOT_DIR / "results_v4/smollm-135m/cov_stats_v4.json"))
    parser.add_argument("--targets", nargs="+", type=float, default=[3.25, 4.25, 5.00])
    args = parser.parse_args()

    cov = load_cov_stats(args.cov)

    print(f"\nLoaded cov stats from: {args.cov}")
    layers = list(cov.keys()) if "layers" not in cov else list(cov["layers"].keys())
    print(f"Layers: {len(layers)}")

    all_results = {}
    for target in args.targets:
        ranks, actual_bw = allocate_ranks_fixed(cov, target)
        vals = list(ranks.values())
        avg_rank = sum(vals) / len(vals) if vals else 0

        print(f"\n  target={target:.2f}:")
        print(f"    actual_bw  = {actual_bw:.3f}")
        print(f"    min_rank   = {min(vals)}")
        print(f"    max_rank   = {max(vals)}")
        print(f"    avg_rank   = {avg_rank:.1f}")
        print(f"    unique_ranks = {len(set(vals))}")

        # Verify criteria
        bw_ok = abs(actual_bw - target) <= 0.5
        diverse = max(vals) > min(vals)
        print(f"    BW within ±0.5: {'✅' if bw_ok else '❌'}")
        print(f"    Ranks diverse:  {'✅' if diverse else '❌'}")

        all_results[f"bw{target:.2f}"] = {
            "ranks": ranks,
            "actual_bw": round(actual_bw, 3),
            "min_rank": min(vals),
            "max_rank": max(vals),
            "avg_rank": round(avg_rank, 1),
            "unique_ranks": len(set(vals)),
        }

    # Save
    out_file = RESULTS / f"rank_alloc_fixed.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved → {out_file}")
