@echo off
REM ============================================================
REM  run_turbo_weights.bat
REM  Launches the full TurboQuant patch runner (Steps 1-9).
REM  Run this YOURSELF (not the agent) to avoid GPU contention.
REM
REM  Steps:
REM    1. Smoke test (synthetic)
REM    2. Patch 4 (IP metric)      on SmolLM
REM    3. Patch 3 (Lloyd-Max)      on SmolLM
REM    4. Patch 5 (Cone-Aware)     on SmolLM
REM    5. Patch 2 (QJL)            on SmolLM
REM    6. Combo 3+5                on SmolLM
REM    7. Combo 3+5+2              on SmolLM
REM    8. Best vs GPTQModel        on SmolLM (equal BPW 4.2875)
REM    9. Best config              on Qwen-0.5B + TinyLlama-1.1B
REM
REM  Results: results_v6\patch_turbo_weights\  (+ SUMMARY_turbo_weights.json)
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo  CHMC v6.1 TurboQuant patches runner
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

"%PY%" run_turbo_weights.py
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Runner finished with exit code %RC%
echo  Results: ..\results_v6\patch_turbo_weights\
echo  Log:     ..\results_v6\patch_turbo_weights\logs\run_*.log
echo ============================================================

endlocal
pause
