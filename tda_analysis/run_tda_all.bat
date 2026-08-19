@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
rem ============================================================
rem  TDA run over the CHMC v6 models (BPW=4.2875)
rem  Launch: double-click OR from the console:
rem     tda_analysis\run_tda_all.bat                          - all 3 models
rem     tda_analysis\run_tda_all.bat smollm-135m              - one model
rem     tda_analysis\run_tda_all.bat qwen2.5-0.5b tinyllama-1.1b - several
rem  Do not run in parallel with other GPU jobs.
rem ============================================================

cd /d "%~dp0.."
set "PY=venv\Scripts\python.exe"
set "OUTDIR=results_v6\tda_analysis"

echo ============================================================
echo  Step logs: tda_analysis\logs (collect_ and analyze_, name includes model and time)
echo ============================================================
echo.
echo ============================================================
echo  STEP 0: CPU tests of the TDA core (~30 sec)
echo ============================================================
%PY% tda_analysis\tests\test_tda_core.py
if errorlevel 1 (
    echo.
    echo ERROR: core tests failed - the GPU run will NOT be started.
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
echo  ALL DONE. Artifacts in %OUTDIR% (5 files per model):
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
echo  MODEL: %MODEL%
echo  [1/2] collect_activations.py (GPU, ~3-8 min)
echo ============================================================
%PY% tda_analysis\collect_activations.py --model %MODEL%
if errorlevel 1 (
    echo ERROR: collect for %MODEL% failed - not proceeding further.
    exit /b 1
)
echo ============================================================
echo  [2/2] analyze.py (CPU, ~2-5 min)
echo ============================================================
%PY% tda_analysis\analyze.py --data "%OUTDIR%\tda_activations_%MODEL%.pt"
if errorlevel 1 (
    echo ERROR: analyze for %MODEL% failed.
    exit /b 1
)
exit /b 0
