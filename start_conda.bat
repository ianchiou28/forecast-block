@echo off
chcp 65001 >nul
echo ============================================
echo    A股板块涨停预测系统 - 快速启动 (Conda)
echo ============================================
echo.

REM Check Conda
call conda --version >nul 2>&1
if errorlevel 1 (
    echo [Error] Conda not found. Please install Anaconda or Miniconda.
    pause
    exit /b 1
)

echo [1/2] Select Mode:
echo   1. Predict Now (predict)
echo   2. Fetch Data (fetch)
echo   3. Train Model (train)
echo   4. Full Process (full)
echo   5. Daemon Mode (daemon)
echo.

set /p choice="Enter option (1-5): "

if "%choice%"=="1" (
    echo.
    echo [2/2] Running Prediction...
    call conda run -n forecast-block python main.py --mode predict
) else if "%choice%"=="2" (
    echo.
    echo [2/2] Fetching Data...
    call conda run -n forecast-block python main.py --mode fetch
) else if "%choice%"=="3" (
    echo.
    echo [2/2] Training Model...
    call conda run -n forecast-block python main.py --mode train --force-train
) else if "%choice%"=="4" (
    echo.
    echo [2/2] Running Full Process...
    call conda run -n forecast-block python main.py --mode full
) else if "%choice%"=="5" (
    echo.
    echo [2/2] Starting Daemon...
    call conda run -n forecast-block python main.py --mode daemon
) else (
    echo Invalid Option
)

pause
