@echo off
rem ======================================================================
rem  AYATLA - one-time setup for a new computer (safe to run again).
rem  1. finds Python 3.10 - 3.12 (offers to install 3.12 if missing)
rem  2. creates the project's own Python environment in .venv
rem  3. installs the libraries from requirements.txt and checks them
rem ======================================================================
setlocal
title AYATLA setup
cd /d "%~dp0"
set "PYGAME_HIDE_SUPPORT_PROMPT=1"
echo ==================================================
echo   AYATLA - setup
echo   Creates .venv and installs the libraries.
echo ==================================================
echo.

rem ---- 1. Keep a .venv only if it works on THIS computer ---------------
rem      (a .venv copied from another PC points to that PC's Python)
if not exist ".venv\Scripts\python.exe" goto findpython
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1
if not errorlevel 1 goto install
echo The existing .venv does not work on this computer - recreating it...
rmdir /s /q ".venv"

rem ---- 2. Find Python 3.12, 3.11 or 3.10 -------------------------------
:findpython
set "PY="
py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11"
if not defined PY py -3.10 -c "import sys" >nul 2>&1 && set "PY=py -3.10"
if not defined PY python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1 && set "PY=python"
if defined PY goto makevenv

echo Python 3.10, 3.11 or 3.12 was not found on this computer.
echo MediaPipe - used for the camera - does not support newer Python versions yet.
echo.
where winget >nul 2>&1 || goto nopython
choice /c YN /m "Install Python 3.12 now"
if errorlevel 2 goto nopython
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
py -3.12 -c "import sys" >nul 2>&1 || goto restart
set "PY=py -3.12"

:makevenv
echo Creating .venv with %PY% ...
%PY% -m venv .venv || goto fail

rem ---- 3. Install and check the libraries ------------------------------
:install
echo.
echo Installing libraries. The first time this downloads a few hundred MB
echo and takes a few minutes - please wait...
".venv\Scripts\python.exe" -m pip install --upgrade pip --disable-pip-version-check -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt --disable-pip-version-check || goto fail
echo.
echo Checking the libraries...
".venv\Scripts\python.exe" -c "import mediapipe, cv2, dollarpy, pygame, bleak; print('All libraries OK - MediaPipe', mediapipe.__version__, '- OpenCV', cv2.__version__)" || goto fail
echo ok> ".venv\ayatla_setup_ok"
echo.
echo ==================================================
echo   Setup complete. Start the game with AYATLA.bat
echo ==================================================
if /i not "%~1"=="--quiet" pause
exit /b 0

:nopython
echo Please install Python 3.12 from python.org - tick "Add python.exe to PATH" -
echo then run Setup.bat again. Opening the download page...
start "" "https://www.python.org/downloads/release/python-31210/"
goto fail

:restart
echo Python was installed. Close this window and run Setup.bat again
echo so Windows can find the new Python.
goto fail

:fail
echo.
echo Setup did NOT finish - see the messages above.
if /i not "%~1"=="--quiet" pause
exit /b 1
