#!/usr/bin/env bash
# Start BehaviorTrack from this folder.
set -u
cd "$(dirname "$0")"

app="src/bh_app.py"
config="config/behaviortrack_config.yaml"

# What the interface itself needs: a working Tk runtime plus the packages the
# steps import in process. Checking only that an interpreter exists used to pick
# one that opened the window fine and then died partway through tracking with
# "No module named cv2", after the user had already waited on CLAHE. Failing the
# probe instead moves on to the next candidate.
#
# DeepLabCut is deliberately absent from this list. It lives in .venv-dlc and is
# driven out of process, which is the whole reason the heavy stack is not
# required here.
probe='import tkinter as tk, cv2, numpy, pandas, xgboost, sklearn; r=tk.Tk(); r.withdraw(); r.destroy()'

usable() {
    [ -x "$1" ] && "$1" -c "$probe" >/dev/null 2>&1
}

# The tool's own environment is tried first: it is the one GUIDE.md tells you to
# create, and the only one guaranteed to carry the tool's extras. A .venv
# further up the tree belongs to whichever project this tool is nested in and
# knows nothing about these dependencies.
python_exe=""
for candidate in "../.venv/bin/python" "../../../.venv/bin/python"; do
    if usable "$candidate"; then
        python_exe="$candidate"
        break
    fi
done

if [ -n "$python_exe" ]; then
    "$python_exe" "$app" --config "$config" "$@"
    status=$?
elif command -v uv >/dev/null 2>&1 &&
        uv run --project .. --extra ml --extra video python -c "$probe" >/dev/null 2>&1; then
    uv run --project .. --extra ml --extra video python "$app" --config "$config" "$@"
    status=$?
else
    echo "No Python was found that has both a working Tk runtime and the packages"
    echo "BehaviorTrack needs (tkinter, opencv-python, numpy, pandas, xgboost,"
    echo "scikit-learn). Create the tool's own environment with:"
    echo
    echo "    cd .. && uv sync --extra ml --extra video"
    status=1
fi

if [ "$status" -ne 0 ]; then
    echo "BehaviorTrack could not start. See GUIDE.md for environment setup."
fi
exit "$status"
