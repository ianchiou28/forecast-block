@echo off
chcp 65001 >nul
echo ============================================
echo    A股板块涨停预测系统 - 一键启动
echo ============================================

echo [1/3] 检查环境...
call conda list -n forecast-block >nul 2>&1
if errorlevel 1 (
    echo [警告] 环境未找到，正在尝试创建...
    call setup_env.bat
)

echo [2/3] 启动 Web 前端...
start "Forecast Web Server" cmd /k "conda run -n forecast-block python web_server.py"

echo [3/3] 启动 预测守护进程...
start "Forecast Daemon" cmd /k "conda run -n forecast-block python main.py --mode daemon"

echo.
echo 系统已启动！
echo Web 界面: http://127.0.0.1:8000
echo.
pause
