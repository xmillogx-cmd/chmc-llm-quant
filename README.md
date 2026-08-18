# CHMC v6 — Hybrid Low-Rank + Grouped Quantization vs GPTQ at Equal Bit Budget

CHMC (Compressed Hybrid Matrix Compression) v6 is a weight-compression method for
small LLMs that combines **per-layer low-rank SVD** with **grouped residual
quantization and GPTQ-style error compensation**. The central experiment compares
CHMC against the industrial GPTQ baseline **at an equal bit budget (BPW = 4.2875)**:

- v5 spent ~4.50 BPW vs GPTQ's 4.2875 and still lost, so the comparison was unfair;
- v6 allocates the low-rank rank **per layer** so total BPW matches a target
  (default 4.2875), then compares perplexity (PPL) on WikiText-2.

The full results write-up is in [`results_v6/article/index.html`](results_v6/article/index.html).

## Headline results (BPW = 4.2875, PPL ratio = compressed / baseline, lower is better)

| Model | CHMC v6 best | GPTQ ref | Verdict |
|---|---|---|---|
| SmolLM2-135M | 1.179–1.186 (stat grid mean ~1.174, strict λ=0.05) | 1.1761 | parity / within noise on single runs; best config margin ≈ +0.0038 |
| Qwen2.5-0.5B | 1.174* | 1.1407 | CHMC loses |
| TinyLlama-1.1B | 1.059 | 1.0807 | **real win** (margin > 30σ of run-to-run noise) |

\* single-run value; see `results_v6/stat_grid/` for the multi-rep grid.

Follow-up negative results (documented, directions closed):
- **Drift correction v6.2**: post-quantization drift is *not* a low-rank field along
  the activation cone — total drift d90 = 227–792 per block (`results_v6/drift_correction/`).
- **Per-layer drift decomposition v6.4**: even per-layer local drift is high-dimensional
  (median local d90 ≈ 230; only `L0.o_proj` passes the gate) — selective compression
  not viable on SmolLM (`results_v6/per_layer_drift/`).

## Repository layout

```
cmq_experiment/
├── v6/                      # CURRENT pipeline (the one to reproduce)
│   ├── chmc_v6.py           # core: run_chmc_v6() + CLI for a single run
│   ├── eval_utils_v6.py     # PPL / WikiText-2 loading (byte-identical port of v5)
│   └── run_stat_grid.py     # 16-run statistical grid, 3–4 reps per config
├── v5/                      # GPTQ/AWQ baselines (library-only: gptqmodel)
│   └── run_gptq_baseline.py
├── tda_analysis/            # TDA analysis of compression (CPU) — "exactly 5 artifacts per model"
│   ├── collect_activations.py   # GPU: pre/post block activations + weight SVDs -> one .pt
│   ├── analyze.py               # CPU: Betti / W1 / effective rank -> csv+json+png+md
│   ├── project_3d.py            # CPU: 3D PCA projection of activation clouds
│   └── tda_core.py, tda_log.py
├── drift_correction/        # v6.2 step-1 drift diagnostics (CPU, reads TDA .pt)
│   └── drift_diagnostics.py
├── results_v6/per_layer_drift/  # v6.4 per-layer drift decomposition
│   ├── step1_per_layer_drift.py # full quantization + local/accumulated drift per layer
│   └── _sanity.py               # cheap sanity check of the drift math
├── v7/                      # post-hoc analysis over all results_v6 JSONs (no torch)
│   ├── correlation_analysis.py  # Pearson/Spearman param×outcome matrices + effect tables
│   └── visualize_correlations.py# heatmaps + self-contained HTML report
├── download_hf_model.py     # generic HF downloader with Range-resume (bypasses stalling snapshot_download)
├── download_tinyllama_direct.py  # TinyLlama single-file downloader (resume-capable)
├── requirements.txt
└── results_v6/              # all artifacts: JSONs, CSVs, PNGs, logs, article HTML
```

Legacy material kept for history only: top-level `test_stage*.py`, `chmc_pipeline.py`,
`chmc_v2.py`, `lowrank_*.py`, `eval_ppl*.py`, `check_*.py`, directories `v4/`, `backup/`,
and result folders `results/` … `results_v5/`. They are not needed to reproduce the
article results.

## Models used (Hugging Face IDs)

Models are **not** bundled in this repository. Download them into `models/<name>/`:

| Local dir name | Hugging Face repo | Size |
|---|---|---|
| `smollm-135m`    | `HuggingFaceTB/SmolLM2-135M`            | 135 M params |
| `qwen2.5-0.5b`   | `Qwen/Qwen2.5-0.5B`                     | ~494 M params |
| `tinyllama-1.1b` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`    | 1.1 B params |

Additional models used only in the scale test (`results_v6/scale_test/`):
`Qwen/Qwen2.5-3B`, `Qwen/Qwen3-4B`, and a Gemma multimodal checkpoint ("gemma4").

Download commands (from the repo root; both scripts resume interrupted downloads):

```bash
python download_hf_model.py HuggingFaceTB/SmolLM2-135M models/smollm-135m
python download_hf_model.py Qwen/Qwen2.5-0.5B       models/qwen2.5-0.5b
python download_tinyllama_direct.py --dest-dir models/tinyllama-1.1b   # or: python download_hf_model.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 models/tinyllama-1.1b
```

All entry points resolve the model directory as `<repo_root>/models` by default and
honor the `CMQ_MODELS_DIR` environment variable for relocation:

```bash
set CMQ_MODELS_DIR=D:\my-models        # Windows cmd
export CMQ_MODELS_DIR=/data/models     # Linux/macOS
```

## Data (auto-downloaded, no manual step)

Evaluation and calibration use **WikiText-2** (`wikitext-2-raw-v1`), fetched
automatically on first run via `huggingface_hub`:

- eval set: 12 000 tokens → cached in `.wikitext_cache_v<vocab_size>.pt`;
- calibration text: 2 048 tokens from the train split → cached in `.calib_text_train_v<vocab_size>.txt`.

The caches are keyed by tokenizer vocab size, so different models never collide.
Delete them to force a re-download; they are gitignored.

## Installation

Python ≥ 3.10 recommended (tested with 3.12). Create and activate a venv:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS
```

**GPU (CUDA) install** — for the quantization runs themselves:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`requirements.txt` is written so that a plain `pip install -r requirements.txt`
also works on CPU-only machines (it does not pin the CUDA wheel index).

**CPU-only install** — enough for every analysis step (TDA, drift diagnostics,
correlation analysis, figure generation) and for smoke-testing the pipeline:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Optional, only for the GPTQ baseline: `gptqmodel` (plus `autoawq` / `autogptq` if you
want those baselines too). The baseline script skips missing libraries gracefully.

## Reproduction guide

All commands run from the **repo root**. GPU steps are marked; everything else is CPU.

### 1. Single CHMC v6 run (GPU) — one model, default config at BPW 4.2875

```bash
python v6/chmc_v6.py --model models/smollm-135m [--tag myrun] [--bpw 4.2875] \
                     [--dampening 0.05] [--strict]
```

Saves `results_v6/chmc_v6_<model>.json` with baseline/compressed PPL, ratio, achieved
BPW, per-layer ranks and timing. Default config: rank=8, residual_bits=4, group_size=128,
use_compensation=true, strict_sequential=false, dampening=0.01 — override via the flags
or by calling `run_chmc_v6(model_path, cfg)` from Python (see `v6/run_stat_grid.py` for
the canonical call pattern).

### 2. Statistical grid (GPU) — the main SmolLM experiment

```bash
python v6/run_stat_grid.py --model smollm-135m
```

Runs 5 configs × 3–4 reps with different seeds (42, 43, 44[, 45]) around the peak:
`block_damp005`, `block_damp01`, `strict_damp004/006/005`. ~16 min on a GPU.
Artifacts in `results_v6/stat_grid/<model>/`: one JSON per rep, `SUMMARY_stat_grid.json`
(with mean/std/min/max per config and the margin-vs-GPTQ verdict), logs in `logs/`.

### 3. GPTQ baseline (GPU) — the comparison reference

```bash
python v5/run_gptq_baseline.py                      # default: smollm-135m + qwen2.5-0.5b
python v5/run_gptq_baseline.py --model models/tinyllama-1.1b
```

Uses `gptqmodel` (4-bit, group 128) and saves `results_v5/gptq_baselines_<tag>.json` plus
the combined `results_v5/gptq_awq_baselines.json`. Reference ratios used throughout the
analysis: SmolLM 1.1761 · Qwen2.5-0.5B 1.1407 · TinyLlama 1.0807 (BPW 4.2875).

### 4. TDA pipeline — "did compression break the topology of activations?"

```bash
# GPU, ~3–6 min per model: one .pt with pre/post block activations + weight SVDs
python tda_analysis/collect_activations.py --model smollm-135m \
    [--n-tokens 2048] [--bpw 4.2875] \
    [--overrides '{"strict_sequential": true, "dampening": 0.05}']

# CPU, ~2–5 min: exactly 4 analysis artifacts next to the .pt (5 files per model total)
python tda_analysis/analyze.py --data results_v6/tda_analysis/tda_activations_smollm-135m.pt

# CPU, seconds: 3D PCA projection of pre/post activation clouds
python tda_analysis/project_3d.py --data results_v6/tda_analysis/tda_activations_smollm-135m.pt
```

Outputs per model in `results_v6/tda_analysis/`: `tda_layer_table_<model>.csv`,
`tda_summary_<model>.json`, `tda_diagrams_<model>.png`, `tda_report_<model>.md`.
The 3D figure goes to the separate folder `results_v6/tda_3d/`.

### 5. Drift diagnostics (v6.2 step 1, CPU) — reads the TDA .pt files

```bash
python drift_correction/drift_diagnostics.py --data results_v6/tda_analysis/tda_activations_smollm-135m.pt
python drift_correction/drift_diagnostics.py --all      # every tda_activations_*.pt found
```

Writes `results_v6/drift_correction/step1_diagnostics/drift_pca_<model>.json` and the
combined spectrum plot. Result: **negative** — total drift is high-dimensional, not a
low-rank cone-aligned field (d90 = 227–792).

### 6. Per-layer drift decomposition (v6.4 step 1) — CPU or GPU

```bash
python results_v6/per_layer_drift/step1_per_layer_drift.py   # full quantization + per-layer drift, ~30 min on CPU for SmolLM
python results_v6/per_layer_drift/_sanity.py                 # cheap math sanity check (seconds)
```

Writes `results_v6/per_layer_drift/per_layer_drift_smollm.json` and the compressed-weight
dump `chmc_weights_seed42.pt`. Result: **negative** — median local d90 ≈ 230, gate failed.

### 7. Correlation analysis over all runs (CPU, no torch)

```bash
python v7/correlation_analysis.py      # -> results_v6/correlation_analysis/*.csv + stdout report
python v7/visualize_correlations.py    # -> heatmaps + self-contained CORRELATION_REPORT.html
```

Collects every per-run JSON under `results_v6/`, builds Pearson/Spearman matrices of
config parameters × outcomes (ratio, margin vs GPTQ) and per-parameter effect tables.
Key finding: `hadamard` is the strongest negative driver (r = −0.74 with margin); all
experimental flags off + dampening 0.1 give the best mean ratio.

### 8. Article figures / report

The self-contained HTML report with embedded figures lives at
[`results_v6/article/index.html`](results_v6/article/index.html) (source draft:
`draft_source.txt`, figure diagnostics in `_diag_*.py`). No regeneration step is needed —
the HTML embeds all images as data URIs.

## Environment variables & output locations

| Variable | Default | Used by |
|---|---|---|
| `CMQ_MODELS_DIR` | `<repo_root>/models` | `v6/chmc_v6.py`, `v6/run_stat_grid.py`, `tda_analysis/collect_activations.py` |

All results are written under the repo root, next to the code:
`results_v5/` (GPTQ baselines), `results_v6/<experiment>/` (everything else).
Every long-running script tees its console output into a timestamped log file in the
corresponding `<experiment>/logs/` directory.

## Notes on determinism

- Calibration is deterministic (fixed WikiText, no shuffling); run-to-run spread comes
  from seed-dependent randomness and CUDA non-determinism — which is why the stat grid
  uses different seeds per rep.
- CPU runs of the same quantization differ from GPU runs at the BLAS/SVD level; PPL
  ratios agree to ~1e-3, drift structure is unaffected (documented in v6.4).
