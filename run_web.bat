@echo off
echo Installing web dependencies...
pip install -r requirements_web.txt

echo Starting Web Server...
python web_server.py
pause
