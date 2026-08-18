@echo off
setlocal enabledelayedexpansion

echo ============================================
echo CMQ Experiment — Full Pipeline
echo ============================================
echo.

REM Activate venv
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: venv not found. Run setup.bat first!
    pause
    exit /b 1
)

set MODEL_PATH=models\smollm-135m

REM Check model exists
if not exist "%MODEL_PATH%" (
    echo Model not found at %MODEL_PATH%
    echo Download with: huggingface-cli download HuggingFaceTB/SmolLM-135M --local-dir %MODEL_PATH%
    pause
    exit /b 1
)

echo [1/7] Checking activation geometry...
python check_geometry.py
if errorlevel 1 ( echo FAILED at step 1 & pause & exit /b 1 )
echo.

echo [2/7] Analyzing weight low-rank structure...
python check_weights.py
if errorlevel 1 ( echo FAILED at step 2 & pause & exit /b 1 )
echo.

echo [3/7] Computing baseline perplexity...
python eval_ppl.py
if errorlevel 1 ( echo FAILED at step 3 & pause & exit /b 1 )
echo.

echo [4/7] Scalar quantization baseline...
python quantize_scalar.py
if errorlevel 1 ( echo FAILED at step 4 & pause & exit /b 1 )
echo.

echo [5/7] Pure low-rank evaluation...
python lowrank_eval.py
if errorlevel 1 ( echo FAILED at step 5 & pause & exit /b 1 )
echo.

echo [6/7] Low-rank + quantization...
python lowrank_quant.py
if errorlevel 1 ( echo FAILED at step 6 & pause & exit /b 1 )
echo.

echo [7/7] Generating final report...
python report.py
if errorlevel 1 ( echo FAILED at step 7 & pause & exit /b 1 )
echo.

echo ============================================
echo Experiment complete!
echo Results: results\*
echo Report: report.md
echo ============================================
pause
