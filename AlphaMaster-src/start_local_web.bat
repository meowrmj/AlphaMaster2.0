@echo off
title AlphaMaster Web Server
cd /d "%~dp0"

echo ============================================================
echo   AlphaMaster - Quant Factor Mining Center
echo ============================================================
echo.
echo Starting web server on http://127.0.0.1:8765
echo Press Ctrl+C in this window to stop the server.
echo.

set HTTP_PROXY=http://127.0.0.1:7897
set HTTPS_PROXY=http://127.0.0.1:7897
set ALL_PROXY=http://127.0.0.1:7897
set http_proxy=http://127.0.0.1:7897
set https_proxy=http://127.0.0.1:7897
set all_proxy=http://127.0.0.1:7897

start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8765"
".venv\Scripts\python.exe" run_web.py --host 127.0.0.1 --port 8765

echo.
echo Server has stopped.
pause
