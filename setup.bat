@echo off
REM HippoRAG PDF System - Setup Script for Windows
REM This script automates the installation process

echo ============================================================
echo    HippoRAG PDF System - Automated Setup
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ and try again
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

echo [4/5] Installing dependencies...
echo This may take several minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo [5/5] Creating .env file...
if not exist .env (
    copy .env.example .env
    echo Created .env file from template
) else (
    echo .env file already exists, skipping...
)

echo.
echo ============================================================
echo    Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo.
echo 1. Add PDF files to the 'pdfs' folder
echo.
echo 2. Start vLLM server (in a NEW terminal):
echo    venv\Scripts\activate
echo    vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
echo.
echo 3. Run the system (in this terminal):
echo    python main.py
echo.
echo For testing without vLLM:
echo    python example_simple.py
echo.
echo See README.md for detailed documentation
echo See QUICKSTART.md for quick start guide
echo ============================================================
echo.
pause
