@echo off
REM ============================================================
REM  run_variance_diag.bat
REM  Diagnoses the SOURCE of run-to-run variance:
REM    - identical results (same seed x3) -> hidden seed-dependent randomness
REM    - different results (same seed x3) -> CUDA non-determinism
REM  Run this YOURSELF (not the agent) to avoid GPU contention.
REM
REM  Config: strict_damp005 (strict, damp=0.05), seed=42, 3 runs.
REM  ~3 min.
REM  Results: results_v6\variance_diag\  (+ SUMMARY_variance_diag.json)
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo  CHMC v6 variance diagnostic (same seed x3)
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

"%PY%" run_variance_diag.py
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Runner finished with exit code %RC%
echo  Results: ..\results_v6\variance_diag\
echo  Log:     ..\results_v6\variance_diag\logs\run_*.log
echo ============================================================

endlocal
pause
