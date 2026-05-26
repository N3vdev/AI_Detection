@echo off
title AI Product Inspector — Conveyor UI

echo.
echo  ============================================
echo   AI Product Inspector — Conveyor Mode
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

echo  Starting Conveyor UI...
echo  AI models will load on first run — this can take 1-2 minutes.
echo  The window will open automatically once models are ready.
echo.

python "%~dp0run_conveyor_ui.py" %*

pause
