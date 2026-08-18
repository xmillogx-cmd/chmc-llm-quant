@echo off
setlocal enabledelayedexpansion

echo ============================================
echo CMQ Experiment — Setup (Windows)
echo ============================================

REM Create venv if not exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

REM Activate
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip wheel

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Check torch
echo.
echo Verifying PyTorch installation:
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo.
echo ============================================
echo Setup complete!
echo Next: download model with:
echo   huggingface-cli download HuggingFaceTB/SmolLM-135M --local-dir models\smollm-135m
echo ============================================
