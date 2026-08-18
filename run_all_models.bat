@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo  CHMC v5 (full_v5) vs GPTQModel - all 3 models
echo  3 different architectures: SmolLM / Qwen2.5 / TinyLlama
echo  Per-model isolation: a failure on one model does not stop
echo  the rest; every run (success or failure) saves its own JSON
echo  Results: results_v5\chmc_v5_*.json, results_v5\gptq_baselines_*.json
echo ============================================================

set "FAILED="

echo.
echo ============================================================
echo [1/3] smollm-135m   (GQA, 210 compressible layers)
echo ============================================================

echo --- CHMC v5 full_v5 ---
python v5\run_chmc_v5.py --model "models\smollm-135m" --mixed-precision --adaptive-rank
if errorlevel 1 set "FAILED=%FAILED% smollm-135m:CHMC;"

echo --- GPTQModel 4bit ---
python v5\run_gptq_baseline.py --model "models\smollm-135m"
if errorlevel 1 set "FAILED=%FAILED% smollm-135m:GPTQ;"

echo.
echo ============================================================
echo [2/3] qwen2.5-0.5b  (GQA)
echo ============================================================

echo --- CHMC v5 full_v5 ---
python v5\run_chmc_v5.py --model "models\qwen2.5-0.5b" --mixed-precision --adaptive-rank
if errorlevel 1 set "FAILED=%FAILED% qwen2.5-0.5b:CHMC;"

echo --- GPTQModel 4bit ---
python v5\run_gptq_baseline.py --model "models\qwen2.5-0.5b"
if errorlevel 1 set "FAILED=%FAILED% qwen2.5-0.5b:GPTQ;"

echo.
echo ============================================================
echo [3/3] tinyllama-1.1b  (LLaMA arch, GQA, 1.1B params)
echo ============================================================

echo --- CHMC v5 full_v5 ---
python v5\run_chmc_v5.py --model "models\tinyllama-1.1b" --mixed-precision --adaptive-rank
if errorlevel 1 set "FAILED=%FAILED% tinyllama-1.1b:CHMC;"

echo --- GPTQModel 4bit ---
python v5\run_gptq_baseline.py --model "models\tinyllama-1.1b"
if errorlevel 1 set "FAILED=%FAILED% tinyllama-1.1b:GPTQ;"

echo.
echo ============================================================
if defined FAILED (
    echo  FINISHED WITH ERRORS:
    echo    %FAILED%
) else (
    echo  ALL 6 RUNS COMPLETED SUCCESSFULLY
)
echo  Results: results_v5\chmc_v5_*.json, results_v5\gptq_baselines_*.json
echo ============================================================
pause
