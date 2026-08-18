@echo off
REM ============================================================
REM  run_stat_grid_others.bat
REM  Runs the statistical grid (run_stat_grid.py) on the OTHER two
REM  models (Qwen-0.5B + TinyLlama-1.1b), 3-4 reps each, different
REM  seeds, to check whether the run-to-run dynamics (variance,
REM  win/loss vs GPTQ) differ from SmolLM.
REM
REM  Why: the SmolLM grid showed the "win" was noise (mean 1.1793 >
REM  GPTQ 1.1761). TinyLlama "won" by +0.0155 and Qwen lost by
REM  -0.033 in the single-run baseline_upgrade, but those were ONE
REM  run each. This measures their real variance.
REM
REM  Run this YOURSELF (not the agent) to avoid GPU contention.
REM
REM  Per model (equal BPW 4.2875), 16 runs x ~60s = ~16 min:
REM    block_damp005   block,  damp=0.05, 3 reps
REM    block_damp01    block,  damp=0.1,  3 reps
REM    strict_damp004  strict, damp=0.04, 3 reps
REM    strict_damp006  strict, damp=0.06, 3 reps
REM    strict_damp005  strict, damp=0.05, 4 reps
REM
REM  Results: results_v6\stat_grid\qwen2.5-0.5b\
REM           results_v6\stat_grid\tinyllama-1.1b\
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo  CHMC v6 statistical grid on Qwen-0.5B + TinyLlama-1.1B
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

echo ============================================================
echo  [1/2] Qwen-0.5B
echo ============================================================
"%PY%" run_stat_grid.py --model qwen2.5-0.5b
set "RC1=%ERRORLEVEL%"
echo  Qwen exit code: %RC1%
echo.

echo ============================================================
echo  [2/2] TinyLlama-1.1B
echo ============================================================
"%PY%" run_stat_grid.py --model tinyllama-1.1b
set "RC2=%ERRORLEVEL%"
echo  TinyLlama exit code: %RC2%
echo.

echo ============================================================
echo  Done. Qwen RC=%RC1%, TinyLlama RC=%RC2%
echo  Results:
echo    ..\results_v6\stat_grid\qwen2.5-0.5b\SUMMARY_stat_grid.json
echo    ..\results_v6\stat_grid\tinyllama-1.1b\SUMMARY_stat_grid.json
echo ============================================================

endlocal
pause
