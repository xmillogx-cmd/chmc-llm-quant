@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
rem ============================================================
rem  TDA-прогон по моделям CHMC v6 (BPW=4.2875)
rem  Запуск: двойной клик ИЛИ из консоли:
rem     tda_analysis\run_tda_all.bat                          - все 3 модели
rem     tda_analysis\run_tda_all.bat smollm-135m              - одна модель
rem     tda_analysis\run_tda_all.bat qwen2.5-0.5b tinyllama-1.1b - несколько
rem  Не запускать параллельно с другими GPU-задачами.
rem ============================================================

cd /d "%~dp0.."
set "PY=venv\Scripts\python.exe"
set "OUTDIR=results_v6\tda_analysis"

echo ============================================================
echo  Логи шагов: tda_analysis\logs (collect_ и analyze_, имя с моделью и временем)
echo ============================================================
echo.
echo ============================================================
echo  ШАГ 0: CPU-тесты TDA-ядра (~30 сек)
echo ============================================================
%PY% tda_analysis\tests\test_tda_core.py
if errorlevel 1 (
    echo.
    echo ОШИБКА: тесты ядра не прошли - GPU-прогон НЕ запускается.
    pause
    exit /b 1
)

if "%*"=="" (
    set "MODELS=smollm-135m qwen2.5-0.5b tinyllama-1.1b"
) else (
    set "MODELS=%*"
)

for %%M in (%MODELS%) do (
    call :RUN_MODEL %%M
    if errorlevel 1 exit /b 1
)

echo.
echo ============================================================
echo  ВСЁ ГОТОВО. Артефакты в %OUTDIR% (5 файлов на модель):
echo    tda_activations_*.pt, tda_layer_table_*.csv,
echo    tda_summary_*.json, tda_diagrams_*.png, tda_report_*.md
echo ============================================================
pause
exit /b 0

rem ------------------------------------------------------------
:RUN_MODEL
set "MODEL=%~1"
echo.
echo ============================================================
echo  МОДЕЛЬ: %MODEL%
echo  [1/2] collect_activations.py (GPU, ~3-8 мин)
echo ============================================================
%PY% tda_analysis\collect_activations.py --model %MODEL%
if errorlevel 1 (
    echo ОШИБКА: collect для %MODEL% не удался - дальше не перехожу.
    exit /b 1
)
echo ============================================================
echo  [2/2] analyze.py (CPU, ~2-5 мин)
echo ============================================================
%PY% tda_analysis\analyze.py --data "%OUTDIR%\tda_activations_%MODEL%.pt"
if errorlevel 1 (
    echo ОШИБКА: analyze для %MODEL% не удался.
    exit /b 1
)
exit /b 0
