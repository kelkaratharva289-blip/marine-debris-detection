@echo off
REM ==========================================================================
REM  SonicSweep — Start FastAPI Backend
REM ==========================================================================
REM  Activates the Python virtual environment and starts the FastAPI server
REM  with hot-reload on http://localhost:8000
REM
REM  Prerequisites:
REM    1. Python 3.11+ installed
REM    2. backend\.env file exists (copy from .env.example)
REM    3. pip install -r backend\requirements.txt (run setup first)
REM ==========================================================================

setlocal

set "ROOT=%~dp0"
set "BACKEND=%ROOT%backend"
set "VENV=%ROOT%.venv"

REM --- Check virtual environment exists ---
if not exist "%VENV%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at %VENV%
    echo Run this first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r backend\requirements.txt
    exit /b 1
)

REM --- Check .env exists ---
if not exist "%BACKEND%\.env" (
    echo [WARN] backend\.env not found. Copying from .env.example...
    copy "%ROOT%.env.example" "%BACKEND%\.env" >nul
)

REM --- Activate venv and start backend ---
call "%VENV%\Scripts\activate.bat"
cd /d "%BACKEND%"
echo Starting FastAPI backend on http://localhost:8000 ...
python main.py

endlocal
