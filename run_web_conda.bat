@echo off
echo Starting Web Server in Conda environment 'forecast-block'...
call conda run -n forecast-block python web_server.py
pause
