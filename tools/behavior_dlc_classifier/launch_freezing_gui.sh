#!/usr/bin/env bash
# Start the behavior GUI. Run it with: bash launch_freezing_gui.sh
set -u
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
    echo "Starting the behavior GUI with uv..."
    uv run freezing-dlc-gui
    status=$?
else
    echo "uv was not found, using the active Python environment instead."
    PYTHONPATH="$PWD/src:${PYTHONPATH:-}" python3 -m freezing_dlc.gui
    status=$?
fi

if [ "$status" -ne 0 ]; then
    echo
    echo "The GUI could not start."
    echo "Install uv, then run this file again. It will set everything up for you."
    echo "If you are not using uv, the interface also needs Tkinter, which several"
    echo "Linux distributions ship as a separate package (for example python3-tk)."
    echo "Installation instructions are in GUIDE.md."
fi

exit "$status"
