@echo off
title AI Product Inspector — Demo Mode

echo.
echo  ============================================
echo   AI Product Inspector — Test Mode
echo  ============================================
echo.

:: Install dependencies on first run only
if not exist "%~dp0.installed" (
    echo  First run detected — installing Python dependencies...
    echo  This may take several minutes. Do not close this window.
    echo.
    pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo.
        echo  ERROR: pip install failed. Check your Python and internet connection.
        pause
        exit /b 1
    )
    echo. > "%~dp0.installed"
    echo.
    echo  Dependencies installed successfully.
    echo.
)

echo  Starting AI server...
echo  Models will load on first run — this can take 1-2 minutes.
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
