@echo off
title Camera ADB Setup — IP Webcam over USB

echo.
echo  ============================================
echo   Camera Setup — Android Phones via USB
echo  ============================================
echo.
echo  Prerequisites:
echo    1. Install "IP Webcam" app on each Android phone
echo    2. Open IP Webcam app and tap "Start server"
echo    3. Connect each phone via USB cable
echo    4. Enable "USB Debugging" in phone Developer Options
echo.

:: Check ADB is available
where adb >nul 2>&1
if errorlevel 1 (
    echo  ERROR: adb not found. Install Android Platform Tools and add to PATH.
    echo  Download: https://developer.android.com/tools/releases/platform-tools
    pause
    exit /b 1
)

echo  Connected devices:
echo  ------------------
adb devices
echo.

:: Forward ports — one per phone (IP Webcam runs on port 8080 by default)
echo  Forwarding ports...
echo.

set PORT=8080
set CAM_NUM=1

for /f "skip=1 tokens=1" %%D in ('adb devices') do (
    if not "%%D"=="" (
        echo  Phone %%D  →  localhost:%PORT%
        adb -s %%D forward tcp:%PORT% tcp:8080
        set /a PORT=%PORT%+1
        set /a CAM_NUM=%CAM_NUM%+1
    )
)

echo.
echo  Done. Update CAMERA_INDICES in live\conveyor_config.py:
echo.
echo    CAMERA_INDICES = [
echo        "http://localhost:8080/video",   ^# Phone 1
echo        "http://localhost:8081/video",   ^# Phone 2
echo        "http://localhost:8082/video",   ^# Phone 3
echo    ]
echo.
echo  Then run run_conveyor_ui.bat
echo.
pause
