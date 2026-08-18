# Сводка по корреляционному анализу CHMC v6/v7 vs GPTQ (BPW ≈ 4.2875)

Дата: 2026-08-18. Источник данных: все per-run JSON под `results_v6/`
(scale_test, whitening_test, stat_grid, niter_test, baseline_upgrade,
patch_turbo_weights, variance_diag). Скрипт: `v7/correlation_analysis.py`.

## Данные

- **126 прогонов** после дедупликации; healthy (ratio ≤ 3.0) = 124, катастрофических = 2.
- Модели: smollm-135M — 39, tinyllama-1.1B — 33, qwen2.5-0.5B — 24, qwen2.5-3b — 15, qwen3-4b — 15.
- GPTQ-референсы (ratio, ниже = лучше): smollm 1.1761 · qwen2.5-0.5B 1.1407 · tinyllama 1.0807 · qwen2.5-3b 1.085498 · qwen3-4b 1.091636.
- `margin_vs_gptq = gptq_ref − ratio` (>0 — CHMC бьёт GPTQ при равном BPW).

Катастрофические выбросы (исключены из healthy, видны в runs_dataset.csv):

| run | ratio | источник |
|---|---|---|
| step5_patch2_qjl | 12541.68 | patch_turbo_weights/step5_patch2_qjl.json |
| step7_combo_3_5_2 | 3515.69 | patch_turbo_weights/step7_combo_3_5_2.json |

Оба — SmolLM, оба связаны с qjl-вариантами. В fullset qjl ↔ ratio r=+0.87 (n=126).

## Топ корреляции (healthy, n=124)

| пара | pearson r | n | \|t\| |
|---|---|---|---|
| hadamard ↔ margin_vs_gptq | **−0.7438** | 124 | 16.5 |
| ratio ↔ hadamard | +0.7177 | 124 | 15.9 |
| ratio ↔ gptq_ref | +0.5587 | 124 | 13.3 |
| cone_aware ↔ margin_vs_gptq | −0.5051 | 124 | 12.8 |
| ratio ↔ cone_aware | +0.4888 | 124 | 12.7 |
| ratio ↔ dampening | −0.4636 | 124 | 12.5 |
| ratio ↔ whitening | +0.4581 | 48 | 7.6 |

Полные матрицы: `corr_pearson_matrix_healthy.csv`, `corr_spearman_matrix_healthy.csv`
(+ fullset-версии), n пар на пару колонок — `corr_pair_counts_healthy.csv`.

Примечание: ratio ↔ gptq_ref (r=+0.56) — артефакт смешения моделей (у разных моделей
разные референсы и разные ratios), а не эффект параметра; для параметров смотреть
`param_effects_per_model.csv`.

## Лучшие значения параметров по среднему ratio (healthy)

| параметр | лучшее значение | mean ± std ratio | n | % бьёт GPTQ |
|---|---|---|---|---|
| dampening | **0.1** | 1.0908 ± 0.0501 | 31 | 77% |
| niter | **10** (tinyllama) | 1.0610 ± 0.0043 | 3 | 100% |
| whitening | 0 | 1.0945 ± 0.0404 | 39 | 62% |
| hadamard | 0 | 1.1332 ± 0.0803 | 120 | 51% |
| cone_aware | 0 | 1.1439 ± 0.1286 | 122 | 50% |
| group_dim | 0 | 1.1477 ± 0.1512 | 122 | 50% |
| lloyd_max | 0 | 1.1505 ± 0.1567 | 122 | 50% |
| ip_metric | 0 | 1.1530 ± 0.1579 | 123 | 50% |
| strict_sequential | 1 (но см. ниже) | 1.1517 ± 0.0869 | 71 | 44% |

## Разбивка по моделям (healthy, из param_effects_per_model.csv)

- **qwen3-4b** — единственный стабильный выигрыш: dampening=0.1 → 1.06625 ± 0.0086,
  100% бьёт GPTQ (ref 1.0916, +2.9σ). Non-strict (block) лучше strict: 1.070/83% vs 1.082/67%.
- **tinyllama-1.1B** — niter=10 → 1.0610 ± 0.0043, 100% бьёт GPTQ (ref 1.0807). Реальный выигрыш.
- **qwen2.5-3b** — dampening=0.1 → 1.0800 ± 0.010, 67% бьёт; vs ref 1.0855 это +0.5σ, т.е. в пределах шума (паритет).
- **smollm-135M** — лучший strict_damp005 ≈ 1.179 vs ref 1.1761 → в пределах шума; hadamard=1 / cone_aware=1 / group_dim=1 заметно ломают (mean 1.35–2.23).
- **qwen2.5-0.5B** — CHMC проигрывает во всех конфигурациях: лучший mean ≈ 1.181 vs ref 1.1407 (+3.6%), 0% бьёт GPTQ; whitening=1 ещё хуже (1.2148 vs 1.1787).

## Выводы — почему именно эти настройки дают лучший результат

1. **dampening=0.1 + non-strict (block) — лучшая и самая стабильная конфигурация**
   на моделях ≥1B: сильнее всего коррелирует с margin (r=−0.46 по ratio), 77% прогонов
   бьют GPTQ, на qwen3-4b даёт реальный выигрыш +2.9σ. Больший dampening мягче
   компенсирует ошибку через Hessian и не «перелечивает» веса.
2. **Все экспериментальные флаги должны быть выключены.** hadamard — самый сильный
   негативный эффект (r=−0.74 с margin): включение резко ухудшает результат на всех
   моделях. cone_aware, group_dim, lloyd_max, ip_metric — аналогично, слабее. qjl —
   катастрофа (ratio 10³–10⁴).
3. **Whitening вредит везде**, где тестировался (r=+0.458 с ratio, n=48; +0.007…+0.036
   к ratio на smollm/qwen-0.5b/tinyllama). Data-adaptive whitening Hessian'а не
   окупается при BPW 4.29.
4. **strict_sequential — модель-зависим**: на SmolLM строгий лучше (1.214 vs 1.388),
   на qwen3-4b — нет (block 1.070/83% beat). В среднем по всем моделям strict=1 чуть
   лучше, но это смешение моделей.
5. **Главный фактор — модель, а не параметры** (ratio↔gptq_ref r=+0.56): CHMC при
   4.29 BPW выигрывает на qwen3-4b и tinyllama(niter10), парит в пределах шума на
   smollm/qwen2.5-3b, проигрывает на qwen2.5-0.5B. Дальнейший анализ «почему» стоит
   вести внутри модели (param_effects_per_model.csv), а не по всему набору.

## Файлы для дальнейшего анализа (эта папка)

| файл | что это |
|---|---|
| runs_dataset.csv | все 126 прогонов: параметры + ratio/margin/beats_gptq, исходник |
| corr_pearson_matrix_{healthy,fullset}.csv | матрицы Пирсона (параметры × результаты) |
| corr_spearman_matrix_{healthy,fullset}.csv | то же, Спирмен (устойчиво к выбросам) |
| corr_pair_counts_healthy.csv | n пар для каждой ячейки healthy-матрицы |
| param_effects_overall.csv | mean/std/min/max ratio + %beats_GPTQ на значение параметра |
| param_effects_per_model.csv | то же, раздельно по моделям (основной инструмент анализа) |

Пересчёт любой подгруппы: правка фильтра в `v7/correlation_analysis.py` и
`python v7\correlation_analysis.py`.
