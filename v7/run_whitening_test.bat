@echo off
REM ============================================================
REM  run_whitening_test.bat
REM  CHMC v7 — Lever 1: full covariance whitening test.
REM
REM  Context: v6 showed model-dependence (TinyLlama win 33sigma,
REM  Qwen loss -2.8sigma, SmolLM neutral). Current damping uses
REM  only the DIAGONAL of the activation covariance (ignores
REM  correlations). Full whitening (C^{1/2}, data-adaptive)
REM  accounts for correlations. Hypothesis: helps where we lose
REM  (Qwen), risk of overfit where we win (per niter finding).
REM
REM  3 models x (baseline, whitened) x 3 seeds (42,43,44).
REM  18 runs, ~25-30 min. Self-check: TinyLlama baseline must
REM  reproduce the niter_test numbers.
REM
REM  Run this YOURSELF (not the agent) to avoid GPU contention.
REM  Results: ..\results_v6\whitening_test\  (+ SUMMARY json)
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo  CHMC v7 whitening test (3 models, on/off, 3 seeds)
echo  Working dir: %cd%
echo ============================================================
echo.

REM Use the project Python (C:\Python314) if present, else PATH python
if exist "C:\Python314\python.exe" (
    set "PY=C:\Python314\python.exe"
) else (
    set "PY=python"
)

echo  Using Python: %PY%
echo.

"%PY%" run_whitening_test.py
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Runner finished with exit code %RC%
echo  Results: ..\results_v6\whitening_test\
echo  Log:     ..\results_v6\whitening_test\logs\run_*.log
echo ============================================================

endlocal
pause
