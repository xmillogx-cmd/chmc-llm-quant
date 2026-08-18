"""Correlation analysis of CHMC v6/v7 runs vs GPTQ at equal BPW (~4.2875).

Collects every per-run JSON under results_v6, attributes each run to its model,
flattens config params + outcomes into one dataset and builds:
  * Pearson & Spearman correlation matrices over all numeric columns (params x outcomes)
  * param-effect tables: mean/std/min/max ratio + %beats_GPTQ per parameter value

Pure stdlib — no torch/GPU. Outputs -> results_v6/correlation_analysis/:
runs_dataset.csv, corr_pearson_matrix.csv, corr_spearman_matrix.csv,
param_effects_overall.csv, param_effects_per_model.csv (+ stdout report).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results_v6"
OUT_DIR = RESULTS_DIR / "correlation_analysis"

GPTQ_REF = {  # GPTQ reference ratios per model (lower is better), from each test's SUMMARY
    "smollm-135M": 1.1761,
    "qwen2.5-0.5B": 1.1407,
    "tinyllama-1.1B": 1.0807,
    "qwen2.5-3b": 1.085498,
    "qwen3-4b": 1.091636,
}

N_PARAMS = {  # approx model sizes — context column only
    "smollm-135M": 0.135e9,
    "qwen2.5-0.5B": 0.494e9,
    "tinyllama-1.1B": 1.1e9,
}

# Directory-level model defaults when path/tag doesn't name a model explicitly.
DIR_MODEL_DEFAULT = {
    # self_check in run_whitening_test.py compares against niter-test TinyLlama ratios
    "niter_test": "tinyllama-1.1B",
    # baseline_ppl=21.3072 identical to smollm grid runs -> SmolLM-135M
    "baseline_upgrade": "smollm-135M",
    "patch_turbo_weights": "smollm-135M",
    # top-level stat_grid files are the smollm grid (SUMMARY says model=smollm-135m)
    "stat_grid": "smollm-135M",
    "variance_diag": "smollm-135M",  # same-seed noise repeats on SmolLM baseline_ppl=21.3072
}

HEALTHY_MAX_RATIO = 3.0  # catastrophic outliers (qjl run ratio~12541) excluded from 'healthy' subset


def _attr_model(relpath_str, tag):
    """Canonical model id from path/tag; falls back to DIR_MODEL_DEFAULT by experiment dir."""
    raw = (str(relpath_str) + " / " + str(tag)).lower()
    norm = re.sub(r"[^a-z0-9]", "_", raw).strip("_")  # qwen2.5-3b -> qwen2_5_3b (adjacent alnum stay adjacent)

    if "smollm" in norm:                                # SmolLM runs (no collision with tinyllama)
        return "smollm-135M"
    if "qwen2_5_0_5b" in norm:                          # qwen2.5-0.5B -> qwen2_5_0_5b (before generic checks)
        return "qwen2.5-0.5B"
    if "tinyllama" in raw:                              # TinyLlama-1.1B
        return "tinyllama-1.1B"
    if "3_4b" in norm and "qwen" in raw:                # Qwen3-4B -> qwen3_4b
        return "qwen3-4b"
    if "2_5_3b" in norm and "qwen" in raw:              # qwen2.5-3B -> qwen2_5_3b (after the -0.5b check above)
        return "qwen2.5-3b"

    parts = [p.lower() for p in str(relpath_str).replace("\\", "/").split("/") if p]
    exp_dir = None  # experiment dir under results_v6/... (e.g. niter_test, stat_grid)
    for i, part in enumerate(parts):
        if "results" in part and len(parts) > i + 1:
            exp_dir = parts[i + 1]
            break
    return DIR_MODEL_DEFAULT.get(exp_dir or (parts[0] if parts else ""), None)


def collect_runs():
    """Walk results_v6, load per-run JSONs, flatten to rows; dedupe identical blobs."""
    seen = set()
    runs = []

    def walk(d):
        for p in sorted(d.iterdir()):
            if p.is_dir():
                name_l = p.name.lower()
                if not (name_l == "logs" or name_l.startswith("tmp") or name_l.endswith(".git")):
                    walk(p)
            elif p.suffix.lower() == ".json":
                try:
                    with open(str(p), "r", encoding="utf-8-sig") as f:
                        obj = json.load(f)
                except Exception:
                    continue
                if not isinstance(obj, dict):  # SUMMARY files are dicts too — filter by keys below
                    continue
                cfg = obj.get("config")
                ratio = obj.get("ratio")
                if (not isinstance(cfg, dict)) or ("dampening" not in cfg and "niter" not in cfg \
                        and "strict_sequential" not in cfg):
                    continue  # skip summaries / GPTQ refs without a CHMC config block
                try:
                    ratio = float(ratio)
                    baseline_ppl = float(obj.get("baseline_ppl"))
                    compressed_ppl = float(obj.get("compressed_ppl"))
                except (TypeError, ValueError):
                    continue  # no numeric outcomes -> not a per-run result

                tag = str(obj.get("model") or p.stem)
                blob_key = hashlib.sha256(
                    json.dumps([tag, baseline_ppl, ratio], sort_keys=True).encode()).hexdigest()
                if blob_key in seen:  # exact duplicate copy of an already-collected run
                    continue
                seen.add(blob_key)

                model = _attr_model(str(p.relative_to(ROOT)), tag) or "unknown"
                row = {
                    "source": str(Path("results_v6") / p.relative_to(RESULTS_DIR).as_posix()),
                    "tag": tag,
                    "model": model,
                    "n_params_approx": N_PARAMS.get(model),
                    "baseline_ppl": baseline_ppl,
                    "compressed_ppl": compressed_ppl,
                    "ratio": ratio,
                }

                def num(v):  # scalar numeric only; bools become 0/1 flags
                    if isinstance(v, bool):
                        return int(v)
                    if type(v) in (int, float) and not math.isnan(float(v)):
                        return v
                    return None

                for k, v in cfg.items():
                    nv = num(v)
                    row[k] = nv  # non-scalar config values -> None column value
                ref = GPTQ_REF.get(model)
                if ref is not None:
                    row["gptq_ref"] = float(ref)
                    row["margin_vs_gptq"] = round(float(ref) - ratio, 6)  # >0 means beats GPTQ
                    row["beats_gptq"] = int(row["margin_vs_gptq"] > 0)
                else:
                    row["gptq_ref"] = None
                    row["margin_vs_gptq"] = None
                    row["beats_gptq"] = None
                runs.append(row)

    walk(RESULTS_DIR)
    return runs


def _mean_ranks(vals):  # average ranks for ties (needed by Spearman)
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # mean of positions i..j, 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    vy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if vx == 0 or vy == 0:
        return None
    r = cov / (vx * vy)
    return max(-1.0, min(1.0, round(r, 6)))


def spearman(xs, ys):
    p = pearson(_mean_ranks(list(xs)), _mean_ranks(list(ys)))
    if p is None:
        # ranks can be constant even when values vary (all-equal) -> undefined too
        return None
    return p


def corr_matrix(rows, cols, rank=False):
    """Pairwise matrix over rows; cell=None where pair has <3 valid points or zero variance.

    Returns (matrix, pair_n dict {(i,j)->n_pairs}). Diagonal is 1.0 for defined columns."""
    m = len(cols)
    mat = [[None] * m for _ in range(m)]
    pair_n = {}
    colvals = {c: [r.get(c) if isinstance(r.get(c), (int, float)) else None for r in rows]
               for c in cols}

    def aligned(a, b):  # non-None pairs only; both numeric required
        xs, ys = [], []
        va, vb = colvals[a], colvals[b]
        for i in range(len(rows)):
            if va[i] is not None and vb[i] is not None:
                xs.append(float(va[i]))
                ys.append(float(vb[i]))
        return xs, ys

    defined = set()  # columns with >=3 valid values AND variance > 0
    for ci, c in enumerate(cols):
        nums = [float(v) for v in colvals[c] if v is not None]
        if len(nums) >= 3 and stdev(nums) > 0:
            defined.add(ci)

    for i in range(m):
        mat[i][i] = 1.0 if i in defined else None
    for i in range(m):
        for j in range(i + 1, m):
            xs, ys = aligned(cols[i], cols[j])
            n_pairs = len(xs)
            r_val = (spearman(xs, ys) if rank else pearson(xs, ys)) \
                if i in defined and j in defined else None
            pair_n[(i, j)] = n_pairs
            mat[i][j] = mat[j][i] = r_val
    return mat, pair_n


def effects_table(subset_rows):  # -> (header, rows) sorted best-first per param value
    if not subset_rows:
        return [], []
    skip = {"source", "tag", "model", "n_params_approx", "baseline_ppl", "compressed_ppl",
            "ratio", "gptq_ref", "margin_vs_gptq", "beats_gptq"}

    header = ["param", "value", "n_runs", "mean_ratio", "std_ratio", "min_ratio", "max_ratio",
              "%beats_GPTQ"]  # lower mean ratio is better; % >0 means beats GPTQ ref for that model
    out_rows = []

    all_keys = set()
    for r0 in subset_rows:
        all_keys.update(k for k, v in r0.items() if type(v) in (int, float))
    param_cols = [k for k in sorted(all_keys - skip)]  # numeric config knobs only; outcomes excluded

    def group_stats(key):
        groups = {}
        for r1 in subset_rows:
            v = r1.get(key)
            if type(v) not in (int, float):
                continue
            groups.setdefault(str(v), []).append(r1)
        stats = []
        for val, grp in sorted(groups.items(), key=lambda kv: mean(g["ratio"] for g in kv[1])):  # best first
            ratios = [g["ratio"] for g in grp]
            margins = [g.get("margin_vs_gptq") for g in grp if type(g.get("margin_vs_gptq")) is float or \
                       isinstance(g.get("margin_vs_gptq"), (int, float))]
            beats_pct = round(100.0 * sum(1 for m2 in margins if m2 > 0) / len(margins), 1) \
                if margins else None
            stats.append([key, val, len(grp), round(mean(ratios), 6),
                          round(stdev(ratios), 6) if len(ratios) > 1 else 0.0,
                          min(ratios), max(ratios), beats_pct])
        return stats

    for key in param_cols:
        vals = [r2.get(key) for r2 in subset_rows]
        distinct = {v for v in vals if type(v) in (int, float)}  # noqa — set of scalars only
        n_valid = sum(1 for v in vals if type(v) in (int, float))
        if not (n_valid >= 4 and 2 <= len(distinct) <= 8):
            continue  # skip constants / near-unique ids; keep interpretable knobs
        out_rows.extend(group_stats(key))

    return header, out_rows


def write_csv(path, header, data_rows):
    with open(str(path), "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig for Excel
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for r in data_rows:
            w.writerow(r)


def _fmt(x):  # CSV cell: None -> empty, floats compact
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


OUTCOME_COLS = ["ratio", "margin_vs_gptq"]  # margin >0 means CHMC beats GPTQ at equal BPW


def _top_pairs(cols, mat, pair_n, top_k=15):
    """Top |r| pairs where one side is an outcome column; with rough t-stat."""
    triples = []  # (neg_abs_r for sorting, col_a, col_b, r, n_pairs, abs_t) — best-first after sort
    for i in range(len(cols)):
        if cols[i] not in OUTCOME_COLS:  # only report pairs touching an outcome column
            continue
        js = list(range(0, i)) + list(range(i + 1, len(cols)))
        for j in js:
            r_val = mat[i][j]
            n_pairs = pair_n.get((min(i, j), max(i, j)), 0) or 0
            if not isinstance(r_val, float):
                continue
            t_stat = None
            if abs(r_val) < 1.0 and n_pairs > 3:
                denom_sq = (n_pairs - 2) / max(1e-9, 1.0 - r_val * r_val)
                t_stat = round(abs(math.sqrt(denom_sq)), 3)  # |t| ~ significance proxy; >~4 => p<0.05-ish at n>=6
            triples.append((-abs(r_val), cols[i], cols[j], round(r_val, 4), int(n_pairs), t_stat))

    def _other_is_outcome(a, b):
        return (a in OUTCOME_COLS) != (b in OUTCOME_COLS) or True  # keep all pairs; filter below by relevance

    triples = [t for t in triples if any(c not in ("ratio", "margin_vs_gptq") for c in (t[1], t[2]))]
    return sorted(triples)[:top_k]


def _dataset_header(rows):  # stable column order: fixed identity/outcome cols first, then config knobs alphabetical
    fixed = ["source", "tag", "model", "n_params_approx", "baseline_ppl", "compressed_ppl",
             "ratio", "gptq_ref", "margin_vs_gptq", "beats_gptq"]  # identity + outcomes up front for readability
    rest = sorted({k for r in rows if isinstance(r, dict) for k in r} - set(fixed))
    return fixed + [c for c in rest]


def _select_corr_cols(rows):  # numeric cols with variance>0 and >=3 valid values; outcomes appended last
    skip_id = {"source", "tag", "model", "n_params_approx", "baseline_ppl", "compressed_ppl"}
    keys = set()
    for r in rows:
        keys.update(k for k, v in r.items() if type(v) in (int, float))  # bools already converted to int earlier
    cols = []
    for key in sorted(keys - skip_id):
        vals = [float(r[key]) for r in rows if type(r.get(key)) in (int, float)]
        if len(vals) >= 3 and stdev(vals) > 0:
            cols.append(key)
    return cols + [c for c in OUTCOME_COLS if c not in cols]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = collect_runs()  # all per-run results found under results_v6/...
    if not rows:
        print("ERROR: no runs collected — check RESULTS_DIR")
        return

    model_counts = {}  # sanity-check attribution before trusting anything downstream
    for r in rows:
        model_counts[r["model"]] = model_counts.get(r["model"], 0) + 1
    print(f"collected {len(rows)} runs; models={sorted(model_counts.items(), key=lambda kv: -kv[1])}")

    healthy_rows = [r for r in rows if isinstance(r.get("ratio"), (int, float)) and \
                    r["ratio"] <= HEALTHY_MAX_RATIO]  # drop catastrophic outliers from stats core set
    healthy_ids = {id(r) for r in healthy_rows}
    bad_runs = [r for r in rows if id(r) not in healthy_ids]
    print(f"healthy runs (ratio<={HEALTHY_MAX_RATIO}): {len(healthy_rows)}; catastrophic: {len(bad_runs)}")
    for b in bad_runs:  # list outliers explicitly so they can be traced back to their source files
        print(f"  OUTLIER ratio={b['ratio']:.4f} tag={b['tag']} src={b['source']} model={b['model']}")

    # --- dataset CSV (all runs, every field) ---
    header = _dataset_header(rows)
    write_csv(OUT_DIR / "runs_dataset.csv", header,
              [[_fmt(r.get(c)) for c in header] for r in rows])

    # --- correlation matrices: healthy subset (primary) + full set ---
    report_lines = []  # collected here and printed at the end in one go
    for label, subset in (("healthy", healthy_rows), ("fullset", rows)):
        cols = _select_corr_cols(subset)
        if len(cols) < 2:
            print(f"WARNING: not enough varied numeric columns for '{label}' set")
            continue
        mat_p, pair_n = corr_matrix(subset, cols)
        mat_s, _ = corr_matrix(subset, cols, rank=True)

        def matrix_rows(mat):  # build CSV rows with column labels prepended
            return [[cols[i]] + [_fmt(v) for v in row] for i, row in enumerate(mat)]

        write_csv(OUT_DIR / f"corr_pearson_matrix_{label}.csv", [""] + list(cols), matrix_rows(mat_p))
        write_csv(OUT_DIR / f"corr_spearman_matrix_{label}.csv", [""] + list(cols), matrix_rows(mat_s))
        if label == "healthy":  # pair-counts file only for the primary set (identical structure otherwise)
            counts = [[0] * len(cols) for _ in range(len(cols))]
            for i in range(len(cols)):
                counts[i][i] = sum(1 for r in subset if type(r.get(cols[i])) in (int, float))
                for j in range(i + 1, len(cols)):
                    n_pairs = pair_n.get((i, j), 0) or 0
                    counts[i][j] = counts[j][i] = int(n_pairs)
            write_csv(OUT_DIR / "corr_pair_counts_healthy.csv", [""] + list(cols),
                      [[cols[i]] + [str(v) for v in row] for i, row in enumerate(counts)])

        report_lines.append(f"--- top correlations [{label}] (n_runs={len(subset)}, cols={len(cols)}) ---")
        tops = _top_pairs(cols, mat_p, pair_n, top_k=12)
        if not tops:
            report_lines.append("  (no defined pairs)")
        for neg_r, a, b, r_val, n_pairs, t_stat in tops:
            t_txt = f" |t|={t_stat}" if t_stat is not None else ""
            report_lines.append(f"  {a} <-> {b}: pearson r={r_val:+.4f} n={n_pairs}{t_txt}")

    # --- param-effect tables (healthy set only; outliers stay visible in dataset CSV + report above) ---
    eff_header, eff_rows = effects_table(healthy_rows)
    write_csv(OUT_DIR / "param_effects_overall.csv", eff_header, eff_rows)

    pm_rows = []  # same stats but restricted to one model at a time; 'model' column prepended
    for m in sorted(model_counts):
        if model_counts[m] < 4:  # too few runs per model to be meaningful on its own
            continue
        h2, r2 = effects_table([r for r in healthy_rows if r["model"] == m])
        pm_rows.extend([m] + row for row in r2)
    write_csv(OUT_DIR / "param_effects_per_model.csv", ["model"] + eff_header, pm_rows)

    # --- human-readable summary: best value per parameter (first row of each param = lowest mean ratio) ---
    report_lines.append("--- best config values by mean ratio (healthy set) ---")
    seen_params = []  # keep first-appearance order; rows already sorted best-first within each param
    for row in eff_rows:
        if row[0] not in seen_params:
            seen_params.append(row[0])
            pct = f"{row[7]:.0f}% beat GPTQ" if isinstance(row[7], (int, float)) else "n/a vs GPTQ"
            report_lines.append(f"  {row[0]}={row[1]}: mean ratio {row[3]:.4f} +/- {row[4]:.4f} "
                                f"(n={row[2]}, min {row[5]:.4f}, max {row[6]:.4f}, {pct})")

    print("\n".join(report_lines))
    print(f"\noutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
