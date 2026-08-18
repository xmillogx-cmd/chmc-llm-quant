@echo off
REM ============================================================
REM  run_baseline_upgrade.bat
REM  Launches the baseline-upgrade runner (A1 strict GPTQ + A4 dampening grid).
REM  Run this YOURSELF (not the agent) to avoid GPU contention.
REM
REM  Configs (all at equal BPW 4.2875):
REM    1. ref_baseline          (block,  damp=0.01)  - reference (~1.2146)
REM    2. strict_gptq           (strict, damp=0.01)  - A1 true GPTQ
REM    3. strict_gptq_damp0001  (strict, damp=0.001) - A4
REM    4. strict_gptq_damp005   (strict, damp=0.05)  - A4
REM    5. strict_gptq_damp01    (strict, damp=0.1)   - A4
REM    6. Best vs GPTQModel     on SmolLM
REM    7. Best config           on Qwen-0.5B + TinyLlama-1.1B
REM
REM  Results: results_v6\baseline_upgrade\  (+ SUMMARY_baseline_upgrade.json)
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo  CHMC v6 baseline upgrade runner (A1 strict GPTQ + A4 damp grid)
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

"%PY%" run_baseline_upgrade.py
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Runner finished with exit code %RC%
echo  Results: ..\results_v6\baseline_upgrade\
echo  Log:     ..\results_v6\baseline_upgrade\logs\run_*.log
echo ============================================================

endlocal
pause
