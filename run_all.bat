@echo off
REM ==========================================================================
REM  SonicSweep — Start All Services
REM ==========================================================================
REM  Starts both the FastAPI backend (port 8000) and the Next.js frontend
REM  (port 3000) in separate windows.
REM
REM  Prerequisites:
REM    1. Python 3.11+ and Node.js 18+ installed
REM    2. Virtual environment set up (setup.bat)
REM    3. npm packages installed (setup.bat)
REM ==========================================================================

setlocal

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"

echo ============================================================
echo  SonicSweep — Starting All Services
echo ============================================================
echo.

REM --- Start Backend in a new window ---
REM Build a wrapper batch so paths with spaces work with cmd /k.
echo Starting backend (port 8000)...

if exist "%VENV%\Scripts\activate.bat" (
    > "%TEMP%\sonicsweep_backend.bat" (
        echo @echo off
        echo cd /d "%ROOT%backend"
        echo call "%VENV%\Scripts\activate.bat"
        echo python main.py
    )
) else (
    > "%TEMP%\sonicsweep_backend.bat" (
        echo @echo off
        echo cd /d "%ROOT%backend"
        echo python main.py
    )
)
start "SonicSweep Backend" cmd /k ""%TEMP%\sonicsweep_backend.bat""

REM --- Small delay to let backend window open ---
timeout /t 2 /nobreak >nul

REM --- Start Frontend in a new window ---
echo Starting frontend (port 3000)...
> "%TEMP%\sonicsweep_frontend.bat" (
    echo @echo off
    echo cd /d "%ROOT%frontend"
    echo if not exist node_modules npm install
    echo npm run dev
)
start "SonicSweep Frontend" cmd /k ""%TEMP%\sonicsweep_frontend.bat""

echo.
echo ============================================================
echo  Both services starting:
echo    Backend:  http://localhost:8000
echo    Frontend: http://localhost:3000
echo    API Docs: http://localhost:8000/docs
echo ============================================================
echo.
echo Close the individual windows to stop each service.

endlocal
