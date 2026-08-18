@echo off
REM ============================================================
REM  run_stat_grid.bat
REM  Launches the statistical-grid runner (3-4 reps per config,
REM  different seeds) to test whether the SmolLM win over GPTQ
REM  is real or within run-to-run noise.
REM  Run this YOURSELF (not the agent) to avoid GPU contention.
REM
REM  Grid (SmolLM, all at equal BPW 4.2875):
REM    block_damp005   block,  damp=0.05, 3 reps  (missing cell)
REM    block_damp01    block,  damp=0.1,  3 reps  (missing cell)
REM    strict_damp004  strict, damp=0.04, 3 reps  (fine around peak)
REM    strict_damp006  strict, damp=0.06, 3 reps  (fine around peak)
REM    strict_damp005  strict, damp=0.05, 4 reps  (reference, variance)
REM
REM  16 runs x ~60s = ~16 min.
REM  Results: results_v6\stat_grid\  (+ SUMMARY_stat_grid.json)
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo  CHMC v6 statistical grid runner (3-4 reps/config, seeds 42+)
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

"%PY%" run_stat_grid.py
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Runner finished with exit code %RC%
echo  Results: ..\results_v6\stat_grid\
echo  Log:     ..\results_v6\stat_grid\logs\run_*.log
echo ============================================================

endlocal
pause
