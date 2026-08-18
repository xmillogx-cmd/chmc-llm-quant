# tda_analysis — TDA topology check of CHMC v6 compression

Formalizes the hypothesis from the "topological path" plan: quantization may break
the topology of the activation manifold (Betti numbers, H_1 cycles) and the weight
spectrum (effective rank) even when MSE/cosines look fine. We compare BEFORE/AFTER
CHMC v6 compression at equal BPW = 4.2875 (= GPTQ level).

## Package files

| File | Role | Run by |
|---|---|---|
| `collect_activations.py` | GPU: pre/post block activations + weight SVDs before/after compression -> one .pt | user (GPU session) |
| `analyze.py` | CPU: persistent homology, Betti at eps*, topo-rank, W_1, ranks -> 4 artifacts | user (after collect) |
| `tda_core.py` | TDA math (kNN VR, H_0 union-find + GUDHI, H_1 GUDHI, W_1, eff-rank/d90) | — |
| `tda_log.py` | logging: all collect/analyze output is duplicated to `logs/` (console + file) | — |
| `tests/test_tda_core.py` | CPU tests on synthetic data (clusters/circle/cone/W_1/ranks/Tee) | anyone, no GPU |

## Running

**Easiest — batch script:** double-click `tda_analysis\run_tda_all.bat`
(or run it from a console). It runs, in order: core CPU tests, then for each model
smollm-135m -> qwen2.5-0.5b -> tinyllama-1.1b does collect (GPU) + analyze (CPU).
Stops on the first error. Subset of models: `tda_analysis\run_tda_all.bat qwen2.5-0.5b tinyllama-1.1b`.
Do NOT run in parallel with other GPU jobs.

Manual variant (from the repo root):

```bat
:: 1) core tests (CPU, ~1 min, any time):
python tda_analysis\tests\test_tda_core.py

:: 2) data collection on GPU (~3-6 min for SmolLM; do NOT run in parallel with other GPU jobs):
python tda_analysis\collect_activations.py --model smollm-135m

:: 3) analysis (CPU, ~2-5 min):
python tda_analysis\analyze.py
```

Other models: `--model qwen2.5-0.5b` / `tinyllama-1.1b` for collect, then
`analyze.py --data results_v6/tda_analysis/tda_activations_<model>.pt`.

## Logs

All output of each step is duplicated to a file (console + file, including the traceback
on failure): `tda_analysis/logs/<step>_<model>_<YYYYmmdd_HHMMSS>.log`, where
step = `collect` or `analyze`. Logs are NOT artifacts: `results_v6/tda_analysis/` still
contains exactly 5 files per model.

## Default compression config in collect

Best from the stat grid: `strict_sequential=true, dampening=0.05`, BPW=4.2875
(override with the flag `--overrides "{\"...\": ...}"`).

## Artifacts (exactly 5 files per model in `results_v6/tda_analysis/`)

1. `tda_activations_<model>.pt` — raw data (collect)
2. `tda_layer_table_<model>.csv` — per block + weight aggregates
3. `tda_summary_<model>.json` — aggregates, method, verdict
4. `tda_diagrams_<model>.png` — 4 panels (H0 diagrams, b_1(eps*) per block, W_1(H1)+cosine, eff-rank of weights)
5. `tda_report_<model>.md` — report

Datasets for different models coexist in one folder (names carry the `<model>` suffix).

## Method in one paragraph

Reduced VR: kNN graph (k=32), triangles = 3-cliques; persistence is computed on
512 points (seed=42). H_0 — union-find (exact) + GUDHI cross-check;
H_1 — GUDHI SimplexTree (filtration = max vertex/edge weight). eps* = median NN distance;
Betti and topo-rank(eps*) = b_0+b_1 are read at s=eps*. W_1 — exact computation over the
top-64 intervals by persistence (L_inf, with p/2 on the diagonal).
effective_rank/d90 — v5 conventions (exp-entropy of normalized sigma^2), so the numbers
are comparable with early measurements (min ~17 / median ~60).

## Memory and risks

- collect keeps the model FP32 + calibration inputs on CPU; after every layer
  `del` + `empty_cache()` (the v7 OOM lesson); explicit `gc.collect()` at the end.
- SVD of all layers is computed on CPU inside the compression loop (+1-2 min).
- If GUDHI turns out slow on larger models: reduce SAMPLE/K_NN in analyze.py
  (constants at the top) — pre/post comparison accuracy does not suffer, since both sides
  are computed under identical conditions.
