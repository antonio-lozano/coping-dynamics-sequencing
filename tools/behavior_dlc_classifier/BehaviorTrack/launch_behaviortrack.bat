@echo off
setlocal
cd /d "%~dp0"

set "APP=src\bh_app.py"
set "CONFIG=config\behaviortrack_config.yaml"

REM What the interface itself needs: a working Tk runtime plus the packages the
REM steps import in process. Probing only for tkinter used to pick an
REM interpreter that opened the window fine and then died partway through
REM tracking with "No module named cv2", after the user had already waited on
REM CLAHE. Failing the probe instead moves on to the next candidate.
REM
REM DeepLabCut is deliberately absent from this list. It lives in .venv-dlc and
REM is driven out of process, which is the whole reason the heavy stack is not
REM required here.
set "PROBE=import tkinter as tk, cv2, numpy, pandas, xgboost, sklearn; r=tk.Tk(); r.withdraw(); r.destroy()"

REM The tool's own environment is tried first: it is the one GUIDE.md tells you
REM to create, and the only one guaranteed to carry the tool's extras. A .venv
REM further up the tree belongs to whichever project this tool is nested in and
REM knows nothing about these dependencies.
set "PYEXE="
call :probe "..\.venv\Scripts\python.exe"
if not defined PYEXE call :probe "..\..\..\.venv\Scripts\python.exe"
if not defined PYEXE call :probe "%USERPROFILE%\anaconda3\python.exe"
if not defined PYEXE call :probe "%USERPROFILE%\miniconda3\python.exe"

if defined PYEXE (
    "%PYEXE%" "%APP%" --config "%CONFIG%" %*
    goto finished
)

where uv >nul 2>nul
if %errorlevel% equ 0 (
    uv run --project .. --extra ml --extra video python -c "%PROBE%" >nul 2>nul
    if not errorlevel 1 (
        uv run --project .. --extra ml --extra video python "%APP%" --config "%CONFIG%" %*
        goto finished
    )
)

echo No Python was found that has both a working Tk runtime and the packages
echo BehaviorTrack needs (tkinter, opencv-python, numpy, pandas, xgboost,
echo scikit-learn). Create the tool's own environment with:
echo.
echo     cd ..
echo     uv sync --extra ml --extra video
goto failed

:probe
if not exist %1 exit /b 0
%1 -c "%PROBE%" >nul 2>nul
if errorlevel 1 exit /b 0
set "PYEXE=%~1"
exit /b 0

:finished
if not errorlevel 1 exit /b 0

:failed
echo.
echo BehaviorTrack could not start. See GUIDE.md for environment setup.
pause
exit /b 1
