@echo off
rem Double-click to start the whole AYATLA system (hub, vision, bluetooth, game).
rem Extra options are passed on, e.g.:  AYATLA.bat --camera 1
rem On a new computer it runs Setup.bat first (installs Python libraries once).
cd /d "%~dp0"
set "READY="
if exist ".venv\ayatla_setup_ok" ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1 && set "READY=1"
if not defined READY (
    echo First start on this computer: running Setup.bat first...
    call Setup.bat --quiet
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" run_ayatla.py %*
pause
