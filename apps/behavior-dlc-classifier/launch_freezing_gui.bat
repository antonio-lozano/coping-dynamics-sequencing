@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1

where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo Starting Behavior Studio with the locked environment...
    uv run --locked --extra ml --extra video freezing-dlc-gui
    goto finished
)

if exist "%~dp0.venv\Scripts\python.exe" (
    set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"
) else (
    set "APP_PYTHON=python"
)
echo Starting Behavior Studio with %APP_PYTHON%...
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
"%APP_PYTHON%" -m freezing_dlc.gui

:finished
set "LAUNCH_CODE=%errorlevel%"
if not "%LAUNCH_CODE%"=="0" (
    echo.
    echo Behavior Studio could not start ^(exit %LAUNCH_CODE%^). See the error above.
    echo Follow Installation in GUIDE.md for dependencies and Tk-enabled Python.
    pause
)
exit /b %LAUNCH_CODE%
