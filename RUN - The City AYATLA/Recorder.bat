@echo off
rem Double-click to open the AYATLA Gesture Recorder (record samples for every gesture).
rem Close the game first: both need the camera.  Options, e.g.:  Recorder.bat --camera 1
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
".venv\Scripts\python.exe" record_templates.py %*
