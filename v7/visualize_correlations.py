"""Visualize CHMC v6/v7 vs GPTQ correlation matrices with explanations.

Reads the CSV artifacts from results_v6/correlation_analysis/ (produced by
correlation_analysis.py), draws 4 annotated heatmaps (PNG) and writes a
self-contained HTML report CORRELATION_REPORT.html: per-matrix "how to read"
notes, top-pairs tables with n and |t|, parameter glossary ("what and why")
and numbered footnotes.

No GPU / no torch — matplotlib + stdlib only.
Run from the repo root:  python v7/visualize_correlations.py
"""
from __future__ import annotations

import base64
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: PNG files only, no window
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results_v6" / "correlation_analysis"
FIG_DIR = OUT_DIR / "figures"
HEALTHY_MAX_RATIO = 3.0
OUTCOME_COLS = ("ratio", "margin_vs_gptq")

GPTQ_REF = {  # same refs as correlation_analysis.py (from each test's SUMMARY)
    "smollm-135M": 1.1761, "qwen2.5-0.5B": 1.1407, "tinyllama-1.1B": 1.0807,
    "qwen2.5-3b": 1.085498, "qwen3-4b": 1.091636,
}

# ---------------------------------------------------------------- CSV I/O

def load_matrix(path):
    """corr_*_matrix_*.csv -> (cols, mat[list of list float|None])."""
    with open(str(path), encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    cols = [c for c in rows[0] if c]
    mat = [[float(c) if c.strip() else None for c in r[1:]] for r in rows[1:]]
    return cols, mat


def load_dataset():
    """runs_dataset.csv -> list of dict rows; empty cells stay as ''."""
    with open(str(OUT_DIR / "runs_dataset.csv"), encoding="utf-8-sig") as f:
        out = []
        for row in csv.DictReader(f):
            d = {}
            for k, v in row.items():
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    d[k] = v
            out.append(d)
    return out


def _num(x):  # numeric-and-sane check for pair counting
    return isinstance(x, float) and not math.isnan(x)


def pair_n(rows, a, b):  # aligned non-missing pairs for two columns
    return sum(1 for r in rows if _num(r.get(a)) and _num(r.get(b)))


def t_stat(r, n):  # |t| proxy: sqrt((n-2)/(1-r^2)); >~4 at n>=6 ~ p<0.05 [8]
    if abs(r) >= 1.0 or n <= 3:
        return None
    return round(abs(math.sqrt((n - 2) / max(1e-9, 1.0 - r * r))), 3)


def top_pairs(cols, mat, rows, k=8):
    """Top |r| pairs where one side is an outcome column -> [(a,b,r,n,t)]."""
    out = []
    for i in range(len(cols)):
        if cols[i] not in OUTCOME_COLS:
            continue
        for j in range(len(cols)):
            if j == i or cols[j] in OUTCOME_COLS:  # each pair once, no outcome-outcome
                continue
            r_val = mat[i][j]
            if isinstance(r_val, float):
                n = pair_n(rows, cols[i], cols[j])
                out.append((cols[i], cols[j], r_val, n, t_stat(r_val, n)))
    return sorted(out, key=lambda x: -abs(x[2]))[:k]

# ---------------------------------------------------------------- heatmap

def draw_heatmap(fname, cols, mat, title, subtitle, cbar_label, footnotes):
    m = len(cols)
    data = np.array([[0.0 if v is None else v for v in row] for row in mat])
    mask = np.array([[v is None for v in row] for row in mat])

    fig = plt.figure(figsize=(10.5, 9.4))
    ax = fig.add_axes([0.17, 0.18, 0.66, 0.64])
    im = ax.imshow(np.ma.masked_where(mask, data), cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    # black frame around top-3 |r| pairs touching an outcome column [9]
    cand = []
    for i in range(m):
        if cols[i] not in OUTCOME_COLS:
            continue
        for j in range(m):
            v = mat[i][j]
            if isinstance(v, float) and abs(v) > 0.35:
                cand.append((abs(v), i, j))
    for _, i, j in sorted(cand, reverse=True)[:3]:
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="black", lw=2.2))

    fs = max(6.5, min(9.0, 110 / m))  # cell font shrinks with matrix size
    for i in range(m):
        for j in range(m):
            v = mat[i][j]
            if v is None:
                ax.text(j, i, "–", ha="center", va="center", fontsize=fs, color="#9a9a9a")
            else:  # number + color: never rely on color alone (anti-pattern)
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=fs,
                        color="white" if abs(v) > 0.6 else "#111111")

    ax.set_xticks(range(m), cols, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(m), cols, fontsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=13, weight="bold")
    fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=9.5, color="#333333")
    y = 0.118  # footnotes block at the bottom of the figure
    for line in footnotes:
        fig.text(0.17, y, line, ha="left", va="top", fontsize=7.4, color="#444444")
        y -= 0.022

    path = FIG_DIR / fname
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    return path

# ---------------------------------------------------------------- HTML report

GLOSSARY = {  # column -> "what this is and why it matters in the analysis"
    "ratio": "Quality loss from quantization: compressed_ppl / baseline_ppl. Lower is better; compared against GPTQ at equal BPW [1].",
    "margin_vs_gptq": "gptq_ref − ratio; >0 means CHMC beats GPTQ at equal BPW 4.2875 [3].",
    "beats_gptq": "Binary indicator 'beats GPTQ' (margin>0); its correlations ≈ the sign of margin.",
    "gptq_ref": "The model's GPTQ reference — a constant, not a parameter; its correlations reflect the mix of models in the dataset [6].",
    "dampening": "Hessian regularization λ·diag(H) before inversion. Larger = more conservative error compensation; 0.1 is best on average (77% beat GPTQ).",
    "niter": "Number of column-wise GPTQ-style error refinement iterations; tested only in the dedicated experiments (n=57) [7].",
    "strict_sequential": "1 — strictly row-by-row, 0 — in blocks. The effect is model-dependent: better on SmolLM, worse on qwen3-4b.",
    "group_dim": "Quantization grouping: 0 (CHMC) — along output channels, 1 (GPTQ) — along input features.",
    "hadamard": "B1 variant: SVD in the original cone + Hadamard rotation of the residual. Strongest negative effect on margin (r=−0.74).",
    "cone_aware": "Cone-Aware variant accounting for BPW overhead; worsens the result (r=+0.49 with ratio).",
    "qjl": "C3 patch 2: residual as 1-bit signs of random QJL projections. Catastrophic in the tests (ratio up to 12541) [4].",
    "lloyd_max": "Lloyd–Max quantizer instead of uniform round() on the rotated distribution.",
    "ip_metric": "Experimental variant of the error metric; no significant effect found.",
    "whitening": "v7: whitening of the Hessian data by the full covariance; worsens the result on all tested models (n=48) [7].",
}

FOOTNOTES = [
    "ratio = compressed_ppl / baseline_ppl — PPL degradation from quantization at BPW≈4.2875; lower is better.",
    "Per-model GPTQ references (from each test's SUMMARY, same BPW 4.2875): smollm-135M 1.1761 · qwen2.5-0.5B 1.1407 · tinyllama-1.1B 1.0807 · qwen2.5-3b 1.085498 · qwen3-4b 1.091636.",
    "margin_vs_gptq = gptq_ref − ratio; >0 — CHMC beats GPTQ at equal BPW.",
    "Healthy subset: ratio ≤ 3.0 — excludes the two catastrophic qjl runs (ratio 12541.68 and 3515.69, SmolLM) that would have dominated all correlations.",
    "Empty cell = pair undefined (<3 common points or zero variance in one of the columns).",
    "gptq_ref is a per-model constant: its correlation with ratio (r=+0.56, healthy) reflects the mix of models in the dataset, not a tuning effect. For parameter effects see param_effects_per_model.csv.",
    "niter (n=57), whitening (n=48) and qjl (fullset only) were tested on dedicated subsets — correlations with them are less reliable; check n and |t| in the tables.",
    "|t| = sqrt((n−2)/(1−r²)) — a rough significance proxy: at n≥6, |t|>~4 ≈ p<0.05.",
    "Black frame on the heatmap = top-3 pairs touching an outcome column (ratio / margin_vs_gptq).",
]


def _esc(s):  # minimal HTML escape for generated text
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_html(figures, rows_all):
    """figures: list of dicts (key,title,subtitle,path,cols,mat,tops,lead) -> html string."""
    n_all = len(rows_all)
    counts = {}
    for r in rows_all:
        m = r.get("model")
        if isinstance(m, str):
            counts[m] = counts.get(m, 0) + 1

    p = []
    p.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             "<title>CHMC vs GPTQ — correlation matrices</title>"
             "<style>"
             "body{font-family:system-ui,Segoe UI,sans-serif;max-width:1020px;margin:auto;"
             "padding:24px;color:#1a1a1a;line-height:1.5}"
             "h1{font-size:1.5em}h2{font-size:1.2em;border-bottom:2px solid #ddd;padding-bottom:4px}"
             "table{border-collapse:collapse;margin:10px 0;font-size:.92em}"
             "th,td{border:1px solid #ccc;padding:5px 10px;text-align:left}"
             "th{background:#f3f3f3}img{max-width:100%;border:1px solid #ddd}"
             ".lead{color:#444}.fn{font-size:.85em;color:#555;column-count:2}"
             "sup{color:#b35900}</style></head><body>")

    p.append("<h1>CHMC v6/v7 vs GPTQ — parameter × outcome correlation matrices</h1>"
             f"<p class='lead'>Data: all per-run JSON under <code>results_v6/</code> — {n_all} runs "
             "at equal BPW≈4.2875 (CHMC against the GPTQ reference of the same model) [1][2]. "
             f"healthy = ratio≤3.0 ({sum(1 for r in rows_all if _num(r.get('ratio')) and r['ratio']<=HEALTHY_MAX_RATIO)} runs, "
             "the 2 catastrophic qjl outliers excluded) [4]. Matrices: Pearson and Spearman (rank-based, robust to outliers). "
             "Red = positive association, blue = negative; numbers in cells — r/ρ.</p>")

    p.append("<h2>Models in the dataset</h2><table><tr><th>model</th><th>runs</th>"
             "<th>GPTQ ref ratio [2]</th></tr>")
    for m in sorted(counts, key=lambda k: -counts[k]):
        p.append(f"<tr><td>{_esc(m)}</td><td>{counts[m]}</td>"
                 f"<td>{GPTQ_REF.get(m, '–')}</td></tr>")
    p.append("</table>")

    for fig in figures:
        b64 = base64.b64encode(fig["path"].read_bytes()).decode()
        p.append(f"<h2>{_esc(fig['title'])}</h2>"
                 f"<img src='data:image/png;base64,{b64}' alt='{_esc(fig['key'])}'>"
                 f"<p class='lead'>{_esc(fig['lead'])}</p>")
        p.append("<table><tr><th>pair</th><th>r</th><th>n pairs [5]</th>"
                 "<th>|t| [8]</th></tr>")
        for a, b, r_val, n, t in fig["tops"]:
            p.append(f"<tr><td>{_esc(a)} ↔ {_esc(b)}</td><td>{r_val:+.4f}</td>"
                     f"<td>{n}</td><td>{'–' if t is None else t}</td></tr>")
        p.append("</table>")

    cols_all = []
    for fig in figures:
        for c in fig["cols"]:
            if c not in cols_all:
                cols_all.append(c)
    p.append("<h2>Parameter glossary — what and why</h2>"
             "<p class='lead'>All matrix columns with an explanation of what the parameter does in the pipeline "
             "(v6/chmc_v6.py: low-rank SVD + group quantization with GPTQ-style error compensation via the Hessian) "
             "and why it is in the analysis.</p>"
             "<table><tr><th>column</th><th>what this is / why</th></tr>")
    for c in cols_all:
        p.append(f"<tr><td><b>{_esc(c)}</b></td><td>{_esc(GLOSSARY.get(c, '—'))}</td></tr>")
    p.append("</table>")

    p.append("<h2>Footnotes</h2><ol class='fn'>")
    for fn in FOOTNOTES:
        p.append(f"<li>{_esc(fn)}</li>")
    p.append("</ol></body></html>")
    return "\n".join(p)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows_all = load_dataset()
    if not rows_all:
        print("ERROR: runs_dataset.csv missing — run correlation_analysis.py first")
        return
    rows_h = [r for r in rows_all if _num(r.get("ratio")) and r["ratio"] <= HEALTHY_MAX_RATIO]

    fn_common = [
        f"Source: results_v6/** — {len(rows_all)} CHMC v6/v7 runs, BPW≈4.2875; "
        f"healthy = ratio≤{HEALTHY_MAX_RATIO:.0f} [4].",
        "Empty cells — pair undefined (<3 common points) [5]. Red = +, blue = −.",
        "Black frame — top-3 pairs with outcome columns (ratio / margin_vs_gptq) [9].",
    ]

    specs = [  # (key, csv file, subset rows for pair-n, title, subtitle, cbar label, lead text)
        ("pearson_healthy", "corr_pearson_matrix_healthy.csv", rows_h,
         "Parameter × outcome correlations — healthy (n=124)",
         "Main drivers: hadamard and dampening; all experimental flags off = better",
         "Pearson r",
         "Strongest association with the outcome — hadamard (r=−0.74 with margin): enabling the flag sharply worsens the result. "
         "dampening=0.1 and cone_aware=0 give the best mean ratio; whitening hurts (n=48) [7]."),
        ("spearman_healthy", "corr_spearman_matrix_healthy.csv", rows_h,
         "Rank correlations (Spearman) — healthy (n=124)",
         "Conclusions match Pearson's: the method is robust to outliers, of which there are almost none here [4]",
         "Spearman ρ",
         "Rank version of the same data: the pair ordering barely changes, confirming the robustness of "
         "the conclusion about hadamard/dampening/cone_aware."),
        ("pearson_fullset", "corr_pearson_matrix_fullset.csv", rows_all,
         "Correlations — fullset (n=126, including outliers)",
         "qjl dominates the association with ratio (r=+0.87) — two catastrophic runs [4]",
         "Pearson r",
         "In the full dataset qjl becomes the main correlation (r=+0.87): both of its runs gave ratio 12541 and 3515. "
         "That is why the main analysis is done on the healthy subset [4]."),
        ("spearman_fullset", "corr_spearman_matrix_fullset.csv", rows_all,
         "Rank correlations (Spearman) — fullset (n=126)",
         "Even by ranks qjl stays in first place — the effect is not an artifact of a single outlier point",
         "Spearman ρ",
         "Spearman confirms: the qjl↔ratio association survives even after compressing values into ranks."),
    ]

    figures = []
    for key, fname_csv, sub_rows, title, subtitle, cbar_label, lead in specs:
        cols, mat = load_matrix(OUT_DIR / fname_csv)
        path = draw_heatmap(f"heatmap_{key}.png", cols, mat, title, subtitle, cbar_label, fn_common)
        figures.append({"key": key, "title": title, "subtitle": subtitle, "path": path,
                        "cols": cols, "mat": mat, "tops": top_pairs(cols, mat, sub_rows),
                        "lead": lead})
        print(f"  {path.name}: top3 = " + ", ".join(
            f"{a}↔{b} r={r:+.2f}" for a, b, r, _, _ in figures[-1]["tops"][:3]))

    html_path = OUT_DIR / "CORRELATION_REPORT.html"
    html_path.write_text(build_html(figures, rows_all), encoding="utf-8")
    print(f"\nHTML report: {html_path}")
    print(f"figures dir:  {FIG_DIR}")


if __name__ == "__main__":
    main()
