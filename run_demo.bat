@echo off
title AI Product Inspector — Demo Mode

echo.
echo  ============================================
echo   AI Product Inspector — Test Mode
echo  ============================================
echo.
echo  Starting AI server...
echo  Models will load — this takes 1-2 minutes on first run.
echo.

:: Start the AI server in a new window
start "AI Inspector Server" cmd /k "cd /d %~dp0 && python demo_server.py"

:: Wait for server to come up before starting tunnel
echo  Waiting for server to start...
timeout /t 10 /nobreak >nul

:: Start Cloudflare Tunnel
echo.
echo  Starting Cloudflare Tunnel...
echo  A public URL will appear below — share it with your client.
echo.
"%~dp0cloudflared.exe" tunnel --url http://localhost:8000

pause
