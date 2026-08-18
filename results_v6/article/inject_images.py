#!/usr/bin/env python3
"""
inject_images.py — собирает index.html из template.html + base64-картинок.

Подстановка 7 плейсхолдеров @@IMG_*@@ в data:image/png;base64,...:
  IMG_SMOLLM_3D / IMG_QWEN_3D / IMG_TINYLLAMA_3D — results_v6/tda_3d/*_activations_3d.png
  IMG_BOTTLENECK / IMG_VERSIONS / IMG_PPL        — results_v6/article/figures/*.png
  IMG_DRIFT                                       — results_v6/drift_correction/step1_diagnostics/drift_spectrum_plots.png

Usage:
    venv\\Scripts\\python.exe results_v6\\article\\inject_images.py
"""

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent          # results_v6/article/
ROOT = HERE.parents[1]                          # cmq_experiment/

MAP = {
    "@@IMG_SMOLLM_3D@@": ROOT / "results_v6" / "tda_3d" / "smollm-135m_activations_3d.png",
    "@@IMG_QWEN_3D@@": ROOT / "results_v6" / "tda_3d" / "qwen2.5-0.5b_activations_3d.png",
    "@@IMG_TINYLLAMA_3D@@": ROOT / "results_v6" / "tda_3d" / "tinyllama-1.1b_activations_3d.png",
    "@@IMG_BOTTLENECK@@": HERE / "figures" / "fig_bottleneck.png",
    "@@IMG_VERSIONS@@": HERE / "figures" / "fig_versions.png",
    "@@IMG_PPL@@": HERE / "figures" / "fig_ppl.png",
    "@@IMG_DRIFT@@": ROOT / "results_v6" / "drift_correction" / "step1_diagnostics" / "drift_spectrum_plots.png",
}

html = (HERE / "template.html").read_text(encoding="utf-8")
for ph, png in MAP.items():
    assert png.exists(), f"нет файла: {png}"
    b64 = base64.b64encode(png.read_bytes()).decode("ascii")
    n = html.count(ph)
    assert n == 1, f"{ph}: найдено {n} (ожидалось 1)"
    html = html.replace(ph, f"data:image/png;base64,{b64}")

leftover = [tok for tok in ("@@IMG", "@@") if tok in html]
assert not leftover, f"остались плейсхолдеры: {leftover}"

out = HERE / "index.html"
out.write_text(html, encoding="utf-8")
print(f"[ok] {out}  ({out.stat().st_size / 1e6:.2f} MB)")
print(f"[ok] <img>={html.count('<img')} data:image/png={html.count('data:image/png;base64,')}")
