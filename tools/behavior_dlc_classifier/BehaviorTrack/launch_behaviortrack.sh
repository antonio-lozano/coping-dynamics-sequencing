#!/usr/bin/env bash
# Start BehaviorTrack from this folder.
set -u
cd "$(dirname "$0")"

app="src/bh_app.py"
config="config/behaviortrack_config.yaml"

if [ -x "../../../.venv/bin/python" ]; then
    ../../../.venv/bin/python "$app" --config "$config"
    status=$?
elif [ -x "../.venv/bin/python" ]; then
    ../.venv/bin/python "$app" --config "$config"
    status=$?
elif command -v uv >/dev/null 2>&1; then
    uv run --project .. --extra ml --extra video python "$app" --config "$config"
    status=$?
else
    PYTHONPATH="$PWD/../src:${PYTHONPATH:-}" python3 "$app" --config "$config"
    status=$?
fi

if [ "$status" -ne 0 ]; then
    echo "BehaviorTrack could not start. See README.md for environment setup."
fi
exit "$status"

