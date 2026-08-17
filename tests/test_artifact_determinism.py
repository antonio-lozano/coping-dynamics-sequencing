# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""Determinism regression tests (marked ``slow``): building the same artifact
twice must produce identical bytes on the same machine.

These rebuild real tracked artifacts in place, exactly like ``run_all.py``
does; on the reference machine the bytes also match the committed ones.  They
run in the CI pipeline job and via ``pytest -m slow``, not in the default
fast selection.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.slow


def _run(script: str) -> None:
    # Mirror the environment run_all_figures.py gives every figure subprocess:
    # without SOURCE_DATE_EPOCH the PDF/SVG metadata carries the wall clock and
    # no two renders can ever match.
    env = os.environ.copy()
    env["BATCH_MODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env.setdefault("SOURCE_DATE_EPOCH", "0")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert result.returncode == 0, (
        f"{script} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _normalize_eol(paths: list[Path]) -> None:
    # Mirror update_manifest.normalize_line_endings: generators write platform
    # line endings, the repository stores LF, and the pipeline rewrites text
    # artifacts to LF before hashing. Doing the same here leaves the tree in
    # the state a run_all.py pass leaves it.
    for path in paths:
        if path.suffix.lower() in {".csv", ".md", ".txt", ".json", ".svg"}:
            data = path.read_bytes()
            if b"\r\n" in data:
                path.write_bytes(data.replace(b"\r\n", b"\n"))


def _digests(paths: list[Path]) -> dict[str, str]:
    _normalize_eol(paths)
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_raw_data_workbook_double_build_identical():
    target = [REPO_ROOT / "report" / "raw_data.xlsx"]
    _run("scripts/build_raw_data_workbook.py")
    first = _digests(target)
    _run("scripts/build_raw_data_workbook.py")
    assert _digests(target) == first, "raw_data.xlsx is not byte-deterministic"


def test_supplementary_figure_4_double_render_identical():
    stem = REPO_ROOT / "figures" / "supplementary_figure4"
    targets = [stem.with_suffix(ext) for ext in (".pdf", ".png", ".svg")]
    _run("scripts/generate_figures/supplementary_figure_4_classifier_shap.py")
    first = _digests(targets)
    _run("scripts/generate_figures/supplementary_figure_4_classifier_shap.py")
    assert _digests(targets) == first, (
        "Supplementary Figure 4 is not render-deterministic; check that the "
        "SHAP sampling seed is still applied"
    )
