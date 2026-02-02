@echo off
REM Quick run script for HippoRAG PDF System

echo Starting HippoRAG PDF System...
echo.

REM Check if virtual environment exists
if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if .env exists
if not exist .env (
    echo Warning: .env file not found, using defaults
)

REM Run the main script
python main.py

pause
