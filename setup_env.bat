@echo off
chcp 65001 >nul
echo ============================================
echo    A股板块涨停预测系统 - 环境初始化
echo ============================================

echo [1/3] 创建 Conda 环境 (forecast-block)...
call conda create -n forecast-block python=3.10 -y

echo [2/3] 安装依赖...
call conda run -n forecast-block pip install -r requirements.txt
call conda run -n forecast-block pip install -r requirements_web.txt

echo [3/3] 环境配置完成!
echo.
echo 请使用 start.bat 或 run_web.bat 启动程序。
pause
