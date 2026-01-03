@echo off
chcp 65001 >nul
echo ============================================================
echo   A股板块涨停预测系统 - 环境安装脚本
echo ============================================================
echo.

:: 检查 conda 是否安装
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 conda，请先安装 Anaconda 或 Miniconda
    echo 下载地址: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [1/3] 检测到 conda，开始创建环境...
echo.

:: 检查环境是否已存在
conda env list | findstr /C:"forecast" >nul 2>&1
if %errorlevel% equ 0 (
    echo [提示] 环境 'forecast' 已存在
    set /p choice="是否删除并重新创建? (y/n): "
    if /i "%choice%"=="y" (
        echo 正在删除旧环境...
        conda env remove -n forecast -y
    ) else (
        echo 跳过环境创建，使用现有环境
        goto activate
    )
)

:: 创建环境
echo [2/3] 正在创建 conda 环境 (这可能需要几分钟)...
conda env create -f environment.yml
if %errorlevel% neq 0 (
    echo [错误] 环境创建失败
    pause
    exit /b 1
)

:activate
echo.
echo [3/3] 环境创建成功！
echo.
echo ============================================================
echo   使用方法:
echo ============================================================
echo   1. 激活环境:    conda activate forecast
echo   2. 获取数据:    python main.py --mode fetch
echo   3. 运行预测:    python main.py --mode predict
echo   4. 守护模式:    python main.py --mode daemon
echo   5. 查看回测:    python main.py --mode backtest
echo ============================================================
echo.
pause
