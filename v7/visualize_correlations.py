"""Visualize CHMC v6/v7 vs GPTQ correlation matrices with explanations.

Reads the CSV artifacts from results_v6/correlation_analysis/ (produced by
correlation_analysis.py), draws 4 annotated heatmaps (PNG) and writes a
self-contained HTML report CORRELATION_REPORT.html: per-matrix "how to read"
notes, top-pairs tables with n and |t|, parameter glossary ("что и зачем")
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

GLOSSARY = {  # column -> "что это и зачем в анализе"
    "ratio": "Потеря качества от квантизации: compressed_ppl / baseline_ppl. Ниже — лучше; сравнивается с GPTQ при равном BPW [1].",
    "margin_vs_gptq": "gptq_ref − ratio; >0 означает, что CHMC бьёт GPTQ при равном BPW 4.2875 [3].",
    "beats_gptq": "Бинарный индикатор «бьёт GPTQ» (margin>0); его корреляции ≈ знаку margin.",
    "gptq_ref": "GPTQ-референс модели — константа, не параметр; его корреляции отражают смешение моделей в наборе [6].",
    "dampening": "Регуляризация Hessian'а λ·diag(H) перед инверсией. Больше = консервативнее компенсация ошибки; 0.1 лучший в среднем (77% бьют GPTQ).",
    "niter": "Число итераций уточнения ошибок в стиле GPTQ по колонкам; тестировался только в dedicated-экспериментах (n=57) [7].",
    "strict_sequential": "1 — строго построчно, 0 — блоками. Эффект модель-зависим: лучше на SmolLM, хуже на qwen3-4b.",
    "group_dim": "Группировка квантизации: 0 (CHMC) — вдоль выходных каналов, 1 (GPTQ) — вдоль входных признаков.",
    "hadamard": "B1-вариант: SVD в исходном конусе + Hadamard-вращение остатка. Сильнейший негативный эффект на margin (r=−0.74).",
    "cone_aware": "Cone-Aware вариант с учётом BPW-overhead; ухудшает результат (r=+0.49 c ratio).",
    "qjl": "C3 patch 2: остаток как 1-битные знаки случайных QJL-проекций. Катастрофа в тестах (ratio до 12541) [4].",
    "lloyd_max": "Квантователь Ллойда–Макса вместо равномерного round() на повёрнутом распределении.",
    "ip_metric": "Экспериментальный вариант метрики ошибки; значимого эффекта не найдено.",
    "whitening": "v7: whitening данных Hessian'а по полной ковариации; ухудшает результат на всех тестировавшихся моделях (n=48) [7].",
}

FOOTNOTES = [
    "ratio = compressed_ppl / baseline_ppl — деградация PPL от квантизации при BPW≈4.2875; ниже лучше.",
    "GPTQ-референсы по моделям (из SUMMARY каждого теста, тот же BPW 4.2875): smollm-135M 1.1761 · qwen2.5-0.5B 1.1407 · tinyllama-1.1B 1.0807 · qwen2.5-3b 1.085498 · qwen3-4b 1.091636.",
    "margin_vs_gptq = gptq_ref − ratio; >0 — CHMC бьёт GPTQ при равном BPW.",
    "healthy поднабор: ratio ≤ 3.0 — исключает два катастрофических qjl-прогона (ratio 12541.68 и 3515.69, SmolLM), которые доминировали бы во всех корреляциях.",
    "Пустая ячейка = пара не определена (<3 общих точек или нулевая дисперсия в одном из столбцов).",
    "gptq_ref — константа на модель: его корреляция с ratio (r=+0.56, healthy) отражает смешение моделей в наборе, а не эффект настройки. Для эффектов параметров смотрите param_effects_per_model.csv.",
    "niter (n=57), whitening (n=48) и qjl (только fullset) тестировались на dedicated-подмножествах — корреляции с ними менее надёжны; сверяйтесь с n и |t| в таблицах.",
    "|t| = sqrt((n−2)/(1−r²)) — грубая прокси значимости: при n≥6 |t|>~4 ≈ p<0.05.",
    "Чёрная рамка на heatmap = топ-3 пары, затрагивающие колонки результата (ratio / margin_vs_gptq).",
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
    p.append("<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
             "<title>CHMC vs GPTQ — корреляционные матрицы</title>"
             "<style>"
             "body{font-family:system-ui,Segoe UI,sans-serif;max-width:1020px;margin:auto;"
             "padding:24px;color:#1a1a1a;line-height:1.5}"
             "h1{font-size:1.5em}h2{font-size:1.2em;border-bottom:2px solid #ddd;padding-bottom:4px}"
             "table{border-collapse:collapse;margin:10px 0;font-size:.92em}"
             "th,td{border:1px solid #ccc;padding:5px 10px;text-align:left}"
             "th{background:#f3f3f3}img{max-width:100%;border:1px solid #ddd}"
             ".lead{color:#444}.fn{font-size:.85em;color:#555;column-count:2}"
             "sup{color:#b35900}</style></head><body>")

    p.append("<h1>CHMC v6/v7 vs GPTQ — корреляционные матрицы параметров × результата</h1>"
             f"<p class='lead'>Данные: все per-run JSON под <code>results_v6/</code> — {n_all} прогонов "
             "при равном BPW≈4.2875 (CHMC против GPTQ-референса той же модели) [1][2]. "
             f"healthy = ratio≤3.0 ({sum(1 for r in rows_all if _num(r.get('ratio')) and r['ratio']<=HEALTHY_MAX_RATIO)} прогонов, "
             "исключены 2 катастрофических qjl-выброса) [4]. Матрицы: Пирсон и Спирмен (ранговая, устойчива к выбросам). "
             "Красное = положительная связь, синее = отрицательная; числа в ячейках — r/ρ.</p>")

    p.append("<h2>Модели в наборе</h2><table><tr><th>модель</th><th>прогонов</th>"
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
        p.append("<table><tr><th>пара</th><th>r</th><th>n пар [5]</th>"
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
    p.append("<h2>Словарь параметров — что и зачем</h2>"
             "<p class='lead'>Все столбцы матриц с пояснением, что делает параметр в пайплайне "
             "(v6/chmc_v6.py: low-rank SVD + групповая квантизация с GPTQ-стилевой компенсацией ошибки через Hessian) "
             "и почему он в анализе.</p>"
             "<table><tr><th>столбец</th><th>что это / зачем</th></tr>")
    for c in cols_all:
        p.append(f"<tr><td><b>{_esc(c)}</b></td><td>{_esc(GLOSSARY.get(c, '—'))}</td></tr>")
    p.append("</table>")

    p.append("<h2>Сноски</h2><ol class='fn'>")
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
        f"Источник: results_v6/** — {len(rows_all)} прогонов CHMC v6/v7, BPW≈4.2875; "
        f"healthy = ratio≤{HEALTHY_MAX_RATIO:.0f} [4].",
        "Пустые ячейки — пара не определена (<3 общих точек) [5]. Красное = +, синее = −.",
        "Чёрная рамка — топ-3 пары с колонками результата (ratio / margin_vs_gptq) [9].",
    ]

    specs = [  # (key, csv file, subset rows for pair-n, title, subtitle, cbar label, lead text)
        ("pearson_healthy", "corr_pearson_matrix_healthy.csv", rows_h,
         "Корреляции параметров × результата — healthy (n=124)",
         "Главные драйверы: hadamard и dampening; все экспериментальные флаги выключены = лучше",
         "Pearson r",
         "Сильнейшая связь с результатом — hadamard (r=−0.74 c margin): включение флага резко ухудшает результат. "
         "dampening=0.1 и cone_aware=0 дают лучшие средние ratio; whitening вредит (n=48) [7]."),
        ("spearman_healthy", "corr_spearman_matrix_healthy.csv", rows_h,
         "Ранговые корреляции (Спирмен) — healthy (n=124)",
         "Выводы совпадают с Пирсоном: метод устойчив к выбросам, которых здесь почти нет [4]",
         "Spearman ρ",
         "Ранговая версия тех же данных: порядок пар практически не меняется, что подтверждает устойчивость "
         "вывода о hadamard/dampening/cone_aware."),
        ("pearson_fullset", "corr_pearson_matrix_fullset.csv", rows_all,
         "Корреляции — fullset (n=126, включая выбросы)",
         "qjl доминирует связь с ratio (r=+0.87) — два катастрофических прогона [4]",
         "Pearson r",
         "В полном наборе qjl становится главной корреляцией (r=+0.87): оба его прогона дали ratio 12541 и 3515. "
         "Поэтому основной анализ ведётся на healthy-подмножестве [4]."),
        ("spearman_fullset", "corr_spearman_matrix_fullset.csv", rows_all,
         "Ранговые корреляции (Спирмен) — fullset (n=126)",
         "Даже по рангам qjl остаётся на первом месте — эффект не артефакт выброса одной точки",
         "Spearman ρ",
         "Спирмен подтверждает: связь qjl↔ratio выживает даже после сжатия значений в ранги."),
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
