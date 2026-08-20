@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo Starting the behavior GUI with uv...
    uv run freezing-dlc-gui
    goto finished
)

echo uv was not found, using the active Python environment instead.
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
python -m freezing_dlc.gui

:finished
if errorlevel 1 (
    echo.
    echo The GUI could not start.
    echo Install uv, then run this file again. It will set everything up for you.
    echo Installation instructions are in GUIDE.md.
    pause
)
