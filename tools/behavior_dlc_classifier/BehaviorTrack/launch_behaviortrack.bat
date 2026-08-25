@echo off
setlocal
cd /d "%~dp0"

set "APP=src\bh_app.py"
set "CONFIG=config\behaviortrack_config.yaml"

if exist "..\..\..\.venv\Scripts\python.exe" (
    "..\..\..\.venv\Scripts\python.exe" -c "import tkinter as tk; r=tk.Tk(); r.withdraw(); r.destroy()" >nul 2>nul
    if not errorlevel 1 (
        "..\..\..\.venv\Scripts\python.exe" "%APP%" --config "%CONFIG%" %*
        goto finished
    )
)

if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" -c "import tkinter as tk; r=tk.Tk(); r.withdraw(); r.destroy()" >nul 2>nul
    if not errorlevel 1 (
        "..\.venv\Scripts\python.exe" "%APP%" --config "%CONFIG%" %*
        goto finished
    )
)

if exist "%USERPROFILE%\anaconda3\python.exe" (
    "%USERPROFILE%\anaconda3\python.exe" -c "import tkinter as tk; r=tk.Tk(); r.withdraw(); r.destroy()" >nul 2>nul
    if not errorlevel 1 (
        "%USERPROFILE%\anaconda3\python.exe" "%APP%" --config "%CONFIG%" %*
        goto finished
    )
)

if exist "%USERPROFILE%\miniconda3\python.exe" (
    "%USERPROFILE%\miniconda3\python.exe" -c "import tkinter as tk; r=tk.Tk(); r.withdraw(); r.destroy()" >nul 2>nul
    if not errorlevel 1 (
        "%USERPROFILE%\miniconda3\python.exe" "%APP%" --config "%CONFIG%" %*
        goto finished
    )
)

where uv >nul 2>nul
if %errorlevel% equ 0 (
    uv run --project .. --extra ml --extra video python -c "import tkinter as tk; r=tk.Tk(); r.withdraw(); r.destroy()" >nul 2>nul
    if not errorlevel 1 (
        uv run --project .. --extra ml --extra video python "%APP%" --config "%CONFIG%" %*
        goto finished
    )
)

echo No Python with a working Tk runtime was found.
echo Install Anaconda/Miniconda, or use a Python installation that includes Tcl/Tk.
goto failed

:finished
if not errorlevel 1 exit /b 0

:failed
echo.
echo BehaviorTrack could not start. See README.md for environment setup.
pause
exit /b 1

