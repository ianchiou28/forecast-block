@echo off
chcp 65001 >nul
echo ============================================
echo    A股板块涨停预测系统 - 快速启动
echo ============================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.9+
    pause
    exit /b 1
)

REM 检查依赖
echo [1/3] 检查依赖包...
pip show akshare >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖包...
    pip install -r requirements.txt
)

echo.
echo [2/3] 选择运行模式:
echo   1. 立即预测 (predict)
echo   2. 获取数据 (fetch)
echo   3. 训练模型 (train)
echo   4. 完整流程 (full)
echo   5. 守护模式 (daemon) - 自动定时执行
echo.

set /p choice="请输入选项 (1-5): "

if "%choice%"=="1" (
    echo.
    echo [3/3] 执行预测...
    python main.py --mode predict
) else if "%choice%"=="2" (
    echo.
    echo [3/3] 获取数据...
    python main.py --mode fetch
) else if "%choice%"=="3" (
    echo.
    echo [3/3] 训练模型...
    python main.py --mode train --force-train
) else if "%choice%"=="4" (
    echo.
    echo [3/3] 执行完整流程...
    python main.py --mode full
) else if "%choice%"=="5" (
    echo.
    echo [3/3] 启动守护模式...
    echo 系统将在每日8:00自动执行预测
    python main.py --mode daemon
) else (
    echo 无效选项
)

echo.
pause
