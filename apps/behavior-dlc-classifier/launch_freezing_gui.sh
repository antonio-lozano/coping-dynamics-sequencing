#!/usr/bin/env bash
# Start Behavior Studio from any working directory.
set -u
cd "$(dirname "$0")" || exit 1

if command -v uv >/dev/null 2>&1; then
    echo "Starting Behavior Studio with the locked environment..."
    uv run --locked --extra ml --extra video freezing-dlc-gui
    launch_code=$?
else
    if [ -x .venv/bin/python ]; then
        app_python=.venv/bin/python
    elif command -v python3 >/dev/null 2>&1; then
        app_python=python3
    else
        echo "Python and uv were not found. Follow Installation in GUIDE.md."
        exit 127
    fi
    echo "Starting Behavior Studio with $app_python..."
    PYTHONPATH="$PWD/src:${PYTHONPATH:-}" "$app_python" -m freezing_dlc.gui
    launch_code=$?
fi

if [ "$launch_code" -ne 0 ]; then
    echo
    echo "Behavior Studio could not start (exit $launch_code). See the error above."
    echo "For dependency setup, follow Installation in GUIDE.md."
    echo "If the error mentions tkinter, install a Tk-enabled Python; uv alone"
    echo "does not supply the system Tk packages required on some platforms."
fi

exit "$launch_code"
