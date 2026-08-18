# Conic Manifold Quantization — Quick-Check Experiment

## Цель
Быстро и без обучения проверить гипотезу: активации LLM лежат на конусе/низкоразмерном многообразии → сжатие через low-rank + quantization.

## Архитектура

```
cmq_experiment/
├── PROJECT.md              ← этот файл
├── requirements.txt        ← зависимости
├── setup.bat              ← установка venv + pip install
├── model_loader.py         ← загрузка модели с прогрессом, watchdog, ретраями
├── check_geometry.py       ← геометрия активаций (конус, PCA)
├── check_weights.py        ← low-rank структура весов (SVD energy)
├── eval_ppl.py            ← базовая perplexity до сжатия
├── quantize_scalar.py     ← скалярное квантование baseline
├── lowrank_eval.py        ← чистое low-rank сжатие
├── lowrank_quant.py       ← low-rank + quantization (CMQ)
├── report.py              ← финальный отчёт → report.md
├── run_all.bat            ← запуск всего пайплайна одной командой
└── results/               ← JSON-результаты
```

## model_loader.py — ключевой модуль

Каждый скрипт грузит модель через `model_loader`:

- **tqdm прогресс-бар** — байты + % для каждого файла
- **SpeedWatchdog** — фоновый поток мониторит скорость; если < 50 KB/s дольше 30s → stall flag
- **Auto-retry** — при network error или stall: до 3 попыток с exponential backoff (5s, 10s, 15s)
- **Local cache** — после первой загрузки модель кэшируется в `models/smollm-135m`

## Порядок запуска

```batch
setup.bat                                    # 1. Установка зависимостей
python check_geometry.py                     # 2. Геометрия активаций (сам загрузит модель)
python check_weights.py                      # 3. Low-rank структура весов
python eval_ppl.py                           # 4. Базовая perplexity
python quantize_scalar.py                    # 5. Scalar quantization baseline
python lowrank_eval.py                       # 6. Чистое low-rank
python lowrank_quant.py                      # 7. Low-rank + quantization (CMQ)
python report.py                             # 8. Финальный отчёт

# Или одной командой:
run_all.bat                                  # Запуск всего пайплайна
```

## Критерии успеха

### Геометрия подтверждена
- `anisotropy_global > 0.5` — активации анизотропны (конус)
- `pca_d90 < hidden_dim / 3` — низкая эффективная размерность
- `mean_energy_top64 > 0.3` — веса имеют low-rank структуру

### Сжатие работает
- `lowrank64_q4`: compression > 8×, perplexity degradation < 30%
- `lowrank + quant` лучше чистого scalar при тех же битах

### Гипотеза слабая
- `anisotropy_global < 0.3` — изотропные активации
- `pca_d90 > hidden_dim / 2` — полная размерность
- `mean_energy_top64 < 0.2` — low-rank бесполезен

## Модель
**HuggingFaceTB/SmolLM-135M** — CPU-friendly, ~270 MB safetensors, быстрый SVD.
