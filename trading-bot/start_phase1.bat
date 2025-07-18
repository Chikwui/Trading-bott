@echo off
REM ====================================================
REM Trading Bot - Phase 1 Startup Script
REM ====================================================

SETLOCAL

REM Set Python executable (update path if needed)
SET PYTHON=python

REM Set script directory
SET SCRIPT_DIR=%~dp0

REM Change to script directory
cd /d "%SCRIPT_DIR%"

echo ====================================================
echo  Trading Bot - Phase 1 Initialization
echo ====================================================

REM Check if .env file exists
if not exist ".env" (
    echo [ERROR] .env file not found. Please create it from .env.example
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

REM Create required directories
for %%d in (data logs backtest_results config strategies) do (
    if not exist "%%~d" (
        mkdir "%%~d"
        echo Created directory: %%~d
    )
)

REM Install/update dependencies
echo.
echo [INFO] Checking and installing dependencies...
%PYTHON% -m pip install --upgrade pip
%PYTHON% -m pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

REM Check Redis server
echo.
echo [INFO] Checking Redis server...
%PYTHON% -c "import redis; redis.Redis().ping()" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Redis server is not running or not installed
    echo [INFO] Please install and start Redis server:
    echo [INFO] 1. Download from: https://github.com/tporadowski/redis/releases
    echo [INFO] 2. Run redis-server.exe
    echo.
    echo [WARNING] Continuing without Redis - some features may be limited
    timeout /t 5 /nobreak >nul
)

REM Run setup script
echo.
echo [INFO] Running setup...
%PYTHON% setup.py

if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Setup completed with warnings
    timeout /t 3 /nobreak >nul
)

REM Start Phase 1
echo.
echo ====================================================
echo  Starting Phase 1 Trading Bot...
echo ====================================================

echo [INFO] Starting bot in PAPER TRADING mode
echo [INFO] Press Ctrl+C to stop

title Trading Bot - Phase 1 (Paper Trading)

%PYTHON% run_phase1.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Trading bot stopped with errors
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

echo.
echo [INFO] Trading bot stopped successfully
ENDLOCAL
