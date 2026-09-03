@echo off
REM ==========================================================================
REM  SonicSweep — Initial Project Setup
REM ==========================================================================
REM  One-time setup: creates Python venv, installs backend dependencies,
REM  installs frontend npm packages, and copies the .env file.
REM
REM  Run this once after cloning the repository.
REM ==========================================================================

setlocal

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"

echo ============================================================
echo  SonicSweep — Project Setup
echo ============================================================
echo.

REM --- 1. Create Python virtual environment ---
echo [1/4] Creating Python virtual environment...
if not exist "%VENV%\Scripts\python.exe" (
    python -m venv "%VENV%"
    echo      Created %VENV%
) else (
    echo      Virtual environment already exists at %VENV%
)

REM --- 2. Install backend dependencies ---
echo [2/4] Installing backend Python packages...
call "%VENV%\Scripts\activate.bat"
pip install -r "%ROOT%backend\requirements.txt"
if exist "%ROOT%backend\requirements-edge.txt" (
    echo      Installing optional edge-deployment packages...
    pip install -r "%ROOT%backend\requirements-edge.txt"
)

REM --- 3. Install frontend dependencies ---
echo [3/4] Installing frontend npm packages...
cd /d "%ROOT%frontend"
if not exist "node_modules" (
    npm install
) else (
    echo      node_modules already exists, skipping
)

REM --- 4. Copy .env files ---
echo [4/4] Setting up environment files...
if not exist "%ROOT%backend\.env" (
    copy "%ROOT%.env.example" "%ROOT%backend\.env" >nul
    echo      Created backend\.env from .env.example
) else (
    echo      backend\.env already exists
)

if not exist "%ROOT%frontend\.env.local" (
    echo NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1 > "%ROOT%frontend\.env.local"
    echo      Created frontend\.env.local
) else (
    echo      frontend\.env.local already exists
)

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  Next steps:
echo    1. Place your real sonar data in dataset\images and dataset\labels
echo    2. Start services:    run_all.bat
echo    3. Train the model:   cd backend ^& python train.py
echo ============================================================

cd /d "%ROOT%"
endlocal
