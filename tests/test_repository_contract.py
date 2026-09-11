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


def test_checkout_requires_no_symlink_privileges():
    result = subprocess.run(
        ["git", "ls-files", "-s"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    links = [line for line in result.stdout.splitlines() if line.startswith("120000 ")]
    assert not links, f"Committed symlinks require OS-specific checkout privileges: {links}"


def test_legacy_manuscript_check_requires_explicit_source():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/check_manuscript.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Legacy tool" in result.stdout
    assert "all agree" not in result.stdout
