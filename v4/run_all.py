#!/usr/bin/env python3
"""
run_all.py — Master runner for CHMC v4 on both models
=====================================================

Usage:
    python run_all.py                    # Run everything (both models, all stages)
    python run_all.py --model smollm     # SmolLM-135M only
    python run_all.py --model qwen       # Qwen2.5-0.5B only
    python run_all.py --stage 0          # Stage 0 only (baseline PPL)
    python run_all.py --stages 0,1       # Stages 0 and 1
    python run_all.py --model smollm --stages 0,1
"""

import subprocess
import sys
import argparse
from pathlib import Path

V4_DIR = Path(__file__).parent.resolve()

MODELS = {
    "smollm": {
        "name": "HuggingFaceTB/SmolLM-135M",
        "stage0_1": "run_smollm_stage0_1.py",
        "stages_2_5": "run_smollm_stages_2_5.py",
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-0.5B",
        "stage0_1": "run_qwen_stage1.py",
        "stages_2_3": "run_qwen_stages_2_3.py",
        "stages_4_5": "run_qwen_stages_4_5.py",
    },
}


def run_script(script_name, model_key):
    """Run a script and return its exit code."""
    script_path = V4_DIR / script_name
    if not script_path.exists():
        print(f"[ERR] Script not found: {script_path}")
        return 1

    cmd = [sys.executable, str(script_path)]
    print(f"\n{'='*60}")
    print(f"RUNNING: {model_key} → {script_name}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(V4_DIR))

    if result.returncode != 0:
        print(f"\n[ERR] Script failed with exit code {result.returncode}")
        return result.returncode
    return 0


def main():
    parser = argparse.ArgumentParser(description="CHMC v4 Master Runner")
    parser.add_argument("--model", choices=["smollm", "qwen", "both"], default="both")
    parser.add_argument("--stages", default="all", help="Stages to run: all, 0, 1, 2, 3, 4, 5, or comma-separated (e.g. '0,1')")

    args = parser.parse_args()

    # Parse stages
    if args.stages == "all":
        do_0_1 = True
        do_2_3 = True
        do_4_5 = True
    else:
        stage_list = [s.strip() for s in args.stages.split(",")]
        do_0_1 = "0" in stage_list or "1" in stage_list
        do_2_3 = "2" in stage_list or "3" in stage_list
        do_4_5 = "4" in stage_list or "5" in stage_list

    models = [args.model] if args.model != "both" else ["smollm", "qwen"]

    for model_key in models:
        info = MODELS[model_key]
        print(f"\n{'#'*60}")
        print(f"# MODEL: {info['name']} ({model_key})")
        print(f"{'#'*60}")

        # Stage 0-1 (baseline + scalar + CHMC)
        if do_0_1:
            rc = run_script(info["stage0_1"], model_key)
            if rc != 0:
                print(f"[ABORT] {model_key} stage 0-1 failed, skipping remaining stages")
                continue

        # Stage 2-3 (calibration + sparse) — Qwen has separate script
        if do_2_3 and "stages_2_3" in info:
            rc = run_script(info["stages_2_3"], model_key)
            if rc != 0:
                print(f"[ABORT] {model_key} stages 2-3 failed")
                continue

        # Stage 4-5 (eff rank + shared basis) — Qwen has separate script
        if do_4_5 and "stages_4_5" in info:
            rc = run_script(info["stages_4_5"], model_key)
            if rc != 0:
                print(f"[ABORT] {model_key} stages 4-5 failed")
                continue

        # Stages 2-5 combined — SmolLM has one script for all
        if "stages_2_5" in info and ("stages_2_3" not in info or "stages_4_5" not in info):
            if do_2_3 or do_4_5:
                rc = run_script(info["stages_2_5"], model_key)
                if rc != 0:
                    print(f"[ABORT] {model_key} stages 2-5 failed")
                    continue

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
