# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""The reproducibility checker is itself the repository's largest test: it
verifies required files, manifest hashes, retired paths, and text hygiene.
Running it under pytest makes ``pytest`` a single entry point for everything.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_check_reproducibility_passes():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_reproducibility.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"check_reproducibility.py failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
