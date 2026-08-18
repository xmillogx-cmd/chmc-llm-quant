@echo off
REM ============================================================
REM  run_scale_test.bat  —  RUN THE WHOLE PIPELINE, ALL MODELS
REM
REM  One entry point. Runs, for EVERY model in MODELS_CFG
REM  (qwen2.5-3b + qwen3-4b):
REM      1) GPTQ reference  (GPTQModel 4-bit, group-128)
REM      2) CHMC grid       (5 configs x 3 reps, seeds 42/43/44)
REM      3) best-vs-GPTQ verdict + cross-model summary
REM
REM  No --model flag  =>  all models.  No --skip-gptq  =>  GPTQ
REM  ref is computed if missing, reused if already on disk.
REM
REM  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  reduces
REM  CUDA fragmentation (the OOM log showed 547-634 MiB reserved
REM  but unallocated). Keeps the 3B run inside the 16 GB card.
REM
REM  Run this YOURSELF (not the agent) to avoid GPU contention.
REM  Results: ..\results_v6\scale_test\  (+ SUMMARY json + logs)
REM ============================================================

setlocal
cd /d "%~dp0"

REM --- OOM fragmentation fix (harmless if not needed) ---
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

REM --- project Python if present, else PATH ---
if exist "C:\Python314\python.exe" (
    set "PY=C:\Python314\python.exe"
) else (
    set "PY=python"
)

echo.
echo ============================================================
echo  CHMC v6 scale test  -  FULL PIPELINE, ALL MODELS
echo  Models   : qwen2.5-3b + qwen3-4b
echo  Per model: GPTQ ref + 5 CHMC configs x 3 reps
echo  Python   : %PY%
echo  CWD      : %cd%
echo ============================================================
echo.

REM -u = unbuffered stdout so the log tracks live progress
"%PY%" -u run_scale_test.py
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Runner finished with exit code %RC%
echo  Results : ..\results_v6\scale_test\
echo  Log     : ..\results_v6\scale_test\logs\run_*.log
echo ============================================================

endlocal
pause
