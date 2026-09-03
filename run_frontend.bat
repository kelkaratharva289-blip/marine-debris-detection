@echo off
REM ==========================================================================
REM  SonicSweep — Start React Frontend (Next.js)
REM ==========================================================================
REM  Installs npm dependencies if needed, then starts the Next.js dev server
REM  on http://localhost:3000
REM
REM  Prerequisites:
REM    1. Node.js 18+ installed
REM    2. npm packages installed (run setup first)
REM ==========================================================================

setlocal

set "ROOT=%~dp0"
set "FRONTEND=%ROOT%frontend"

REM --- Check Node.js is available ---
where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org
    exit /b 1
)

REM --- Install dependencies if node_modules is missing ---
if not exist "%FRONTEND%\node_modules" (
    echo Installing frontend dependencies...
    cd /d "%FRONTEND%"
    npm install
)

REM --- Start Next.js dev server ---
cd /d "%FRONTEND%"
echo Starting Next.js frontend on http://localhost:3000 ...
npm run dev

endlocal
