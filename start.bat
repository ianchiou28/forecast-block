@echo off
chcp 65001 >nul
echo ============================================
echo    A股板块涨停预测系统 - 快速启动
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [Error] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

REM Check dependencies
echo [1/3] Checking dependencies...
python -c "import akshare" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    python -m pip install -r requirements.txt
)

echo.
echo [2/3] Select Mode:
echo   1. Predict Now (predict)
echo   2. Fetch Data (fetch)
echo   3. Train Model (train)
echo   4. Full Process (full)
echo   5. Daemon Mode (daemon)
echo.

set /p choice="Enter option (1-5): "

if "%choice%"=="1" (
    echo.
    echo [3/3] Running Prediction...
    python main.py --mode predict
) else if "%choice%"=="2" (
    echo.
    echo [3/3] Fetching Data...
    python main.py --mode fetch
) else if "%choice%"=="3" (
    echo.
    echo [3/3] Training Model...
    python main.py --mode train --force-train
) else if "%choice%"=="4" (
    echo.
    echo [3/3] Running Full Process...
    python main.py --mode full
) else if "%choice%"=="5" (
    echo.
    echo [3/3] Starting Daemon...
    python main.py --mode daemon
) else (
    echo Invalid Option
)

pause
