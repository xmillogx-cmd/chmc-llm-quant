@echo off
REM ============================================================
REM  run_niter_test.bat
REM  Tests whether higher niter (svd_lowrank power iterations)
REM  reduces seed-to-seed variance and improves mean quality.
REM
REM  Context: variance diagnostic showed the pipeline is
REM  deterministic per seed; the ~0.008 grid variance comes from
REM  torch.svd_lowrank (randomized SVD) consuming the global
REM  random state. niter=5 is low; more iterations should make
REM  the result closer to the true low-rank SVD.
REM
REM  Model: tinyllama-1.1b (real win), config block_damp01.
REM  niter in {5, 10, 20} x 3 seeds (42,43,44). 9 runs, ~10-15 min.
REM  Self-check: niter=5 must reproduce the stat_grid numbers.
REM
REM  Run this YOURSELF (not the agent) to avoid GPU contention.
REM  Results: results_v6\niter_test\  (+ SUMMARY_niter_test.json)
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo  CHMC v6 niter test (TinyLlama, block_damp01)
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

"%PY%" run_niter_test.py
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Runner finished with exit code %RC%
echo  Results: ..\results_v6\niter_test\
echo  Log:     ..\results_v6\niter_test\logs\run_*.log
echo ============================================================

endlocal
pause
