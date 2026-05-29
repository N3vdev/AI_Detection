@echo off
setlocal EnableDelayedExpansion
title AI Inspector — Build EXE

echo.
echo ============================================================
echo   AI Inspector — EXE Builder
echo ============================================================
echo.

:: ── Working directory = this script's folder ─────────────────────────────────
cd /d "%~dp0"

:: ── 1. Check Python ───────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Install Python 3.13+ from https://python.org and re-run.
    pause & exit /b 1
)

:: ── 2. Ensure PyInstaller is installed ───────────────────────────────────────
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [Setup] Installing PyInstaller...
    python -m pip install --upgrade pyinstaller
    if errorlevel 1 (
        echo [ERROR] PyInstaller install failed.
        pause & exit /b 1
    )
)

:: ── 3. Download Python 3.13 embedded runtime ─────────────────────────────────
set EMBED_ZIP=python313_embed.zip
set EMBED_URL=https://www.python.org/ftp/python/3.13.3/python-3.13.3-embed-amd64.zip

if not exist "%EMBED_ZIP%" (
    echo [Download] Fetching Python 3.13.3 embedded runtime...
    python -c "import urllib.request; urllib.request.urlretrieve('%EMBED_URL%', '%EMBED_ZIP%'); print('Done.')"
    if errorlevel 1 (
        echo [ERROR] Download failed. Check your internet connection.
        pause & exit /b 1
    )
    echo [OK] %EMBED_ZIP% downloaded.
) else (
    echo [OK] %EMBED_ZIP% already present.
)

:: ── 4. Download get-pip.py ────────────────────────────────────────────────────
set GETPIP=get_pip.py
set GETPIP_URL=https://bootstrap.pypa.io/get-pip.py

if not exist "%GETPIP%" (
    echo [Download] Fetching get-pip.py...
    python -c "import urllib.request; urllib.request.urlretrieve('%GETPIP_URL%', '%GETPIP%'); print('Done.')"
    if errorlevel 1 (
        echo [ERROR] get-pip.py download failed.
        pause & exit /b 1
    )
    echo [OK] %GETPIP% downloaded.
) else (
    echo [OK] %GETPIP% already present.
)

:: ── 5. Clean previous build artifacts ────────────────────────────────────────
if exist "build\AI_Inspector"    rmdir /s /q "build\AI_Inspector"
if exist "dist\AI_Inspector.exe" del /q "dist\AI_Inspector.exe"
if exist "AI_Inspector.spec"     del /q "AI_Inspector.spec"

:: ── 6. Build EXE ──────────────────────────────────────────────────────────────
echo.
echo [Build] Running PyInstaller...
echo.

pyinstaller ^
    --onefile ^
    --windowed ^
    --name AI_Inspector ^
    --add-data "%EMBED_ZIP%;." ^
    --add-data "%GETPIP%;." ^
    launcher.py

if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller build failed. Check output above.
    pause & exit /b 1
)

:: ── 7. Assemble distribution folder ──────────────────────────────────────────
echo.
echo [Package] Assembling distribution folder...

set DIST_DIR=dist\AI_Inspector_dist

if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
mkdir "%DIST_DIR%"

:: Core EXE
copy "dist\AI_Inspector.exe" "%DIST_DIR%\" >nul

:: App source files (needed by the venv-launched run_conveyor_ui.py)
xcopy /e /i /q "src"              "%DIST_DIR%\src\"              >nul
xcopy /e /i /q "live"             "%DIST_DIR%\live\"             >nul
xcopy /e /i /q "conveyor_ui"      "%DIST_DIR%\conveyor_ui\"      >nul
xcopy /e /i /q "models"           "%DIST_DIR%\models\"           >nul 2>&1

copy "run_conveyor_ui.py"         "%DIST_DIR%\" >nul
copy "requirements_app.txt"       "%DIST_DIR%\" >nul

if exist "live\conveyor_config.py" (
    copy "live\conveyor_config.py" "%DIST_DIR%\live\" >nul 2>&1
)

echo.
echo ============================================================
echo   BUILD COMPLETE
echo ============================================================
echo.
echo   Distribution folder:  %DIST_DIR%\
echo.
echo   Contents:
echo     AI_Inspector.exe    ^<-- send this to the client
echo     src\                Application source
echo     live\               Session / worker logic
echo     conveyor_ui\        UI widgets + QSS
echo     models\             Local model files (barcode, CRNN)
echo     run_conveyor_ui.py  Main app entry point
echo     requirements_app.txt  Package list (used by launcher)
echo.
echo   On the client machine:
echo     1. Copy the entire AI_Inspector_dist\ folder anywhere
echo     2. Double-click AI_Inspector.exe
echo     3. First run: setup window appears, installs Python env
echo        + packages + downloads Qwen / YOLO  (~10-20 min)
echo     4. Subsequent runs: launches directly in ~3 seconds
echo.
pause
