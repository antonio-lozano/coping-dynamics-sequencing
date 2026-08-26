"""The workflow's step order, shared by the launcher and by every step window.

The launcher builds its cards from this list and each step window finds its own
successor in it, so "what follows step 3" is written down once. It lives here
rather than in ``bh_app`` because the dependency only runs one way: the launcher
imports the step windows, and the step windows import the runner, so a step
window cannot import the launcher back without a cycle.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

#: (key, title, blurb, script). The key names the step in status lookups, the
#: title is what the launcher's button reads, and the script is the module that
#: opens the window.
STEPS: list[tuple[str, str, str, str]] = [
    (
        "setup",
        "1. Select videos and name sessions",
        "Pick the folder holding your recordings, confirm how animal IDs and session types are read, and save sessions.csv.",
        "bh_setup.py",
    ),
    (
        "track",
        "2. Run DeepLabCut tracking",
        "Track every recording with the bundled network. Needs the DeepLabCut environment; existing tracking is reused.",
        "bh_track.py",
    ),
    (
        "classify",
        "3. Compute features and classify behaviors",
        "Reproduce the archived 719-feature schema, then use one frozen model for all seven behaviors.",
        "bh_features.py",
    ),
    (
        "results",
        "4. Ethograms, totals and summaries",
        "Per-video ethograms, per-class seconds and percentages, bout tables and the combined workbook.",
        "bh_results.py",
    ),
    (
        "model",
        "5. Validate the seven behaviors",
        "Create blinded annotation templates and measure every class on independent target recordings.",
        "bh_model.py",
    ),
]

#: The step scripts live beside ``utils``, in ``src``.
SRC_DIR = Path(__file__).resolve().parents[1]


def step_number(script: str) -> int | None:
    """1-based position of a step, or None when the script is not a step."""
    for index, entry in enumerate(STEPS):
        if entry[3] == script:
            return index + 1
    return None


def next_step(script: str) -> tuple[int, tuple[str, str, str, str]] | None:
    """The step after this one as ``(number, entry)``, or None at the end.

    An unknown script also yields None, so a window that is not part of the
    numbered workflow simply shows no Next button instead of raising.
    """
    number = step_number(script)
    if number is None or number >= len(STEPS):
        return None
    return number + 1, STEPS[number]


def launch_step(script: str, config_path: Path) -> None:
    """Open one step in its own process.

    Each step is a separate process rather than a Toplevel so that a step which
    wedges - DeepLabCut import, a long ffmpeg read - cannot take the launcher
    down with it. Raises FileNotFoundError or OSError; callers report it in the
    idiom of whichever window they are.
    """
    path = SRC_DIR / script
    if not path.is_file():
        raise FileNotFoundError(path)
    subprocess.Popen(
        [sys.executable, str(path), "--config", str(config_path)],
        cwd=str(path.parent),
    )
