@echo off
title Stop AlphaMaster Web Server
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8765" ^| findstr "LISTENING"') do (
  taskkill /PID %%a /F
)
echo AlphaMaster web server stop command sent.
pause
