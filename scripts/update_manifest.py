#!/usr/bin/env python
"""Write MANIFEST.csv with hashes for publication artifacts."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "MANIFEST.csv"
ARTIFACT_DIRS = ("data", "figure_source_data", "figures", "statistics", "report", "classifier")
EXCLUDED_DIRS = {"__pycache__", "manuscript_final"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def category(path: Path) -> str:
    parts = path.parts
    if parts[:2] == ("data", "raw"):
        return "raw_input"
    if parts[:2] == ("data", "processed"):
        return "processed_data"
    if parts[0] == "data":
        return "data_documentation"
    if parts[0] == "figure_source_data":
        return "figure_source_data"
    if parts[0] == "figures":
        return "manuscript_figure"
    if parts[0] == "statistics":
        return "statistical_output"
    if parts[0] == "report":
        return "manuscript_report"
    if parts[0] == "classifier":
        return "trained_model"
    return "artifact"


@lru_cache(maxsize=1)
def clean_clone_paths() -> frozenset[str] | None:
    """Repository-relative paths a fresh clone contains, or None outside a checkout.

    Tracked files, plus untracked files Git is not ignoring. Scratch folders that
    sit inside the tree but never reach the repository are left out, so a working
    machine and a continuous integration runner agree on which files exist.

    None means Git could not answer, as in an archive unpacked from a data
    repository. Callers then read the directory tree as it stands, which is the
    right answer there, because such an archive carries no ignored files.
    """
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    names = completed.stdout.decode("utf-8", "surrogateescape").split("\0")
    return frozenset(name for name in names if name)


def iter_artifacts() -> list[Path]:
    """Every publication artifact, in the one order both the writer and the checker use."""
    included = clean_clone_paths()
    paths: list[Path] = []
    for dirname in ARTIFACT_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("~$"):
                continue
            rel = path.relative_to(ROOT)
            if EXCLUDED_DIRS.intersection(rel.parts):
                continue
            if included is not None and rel.as_posix() not in included:
                continue
            paths.append(path)
    return sorted(paths, key=lambda p: p.as_posix().lower())


# Text artifacts the generators rewrite. pandas, csv, and matplotlib's SVG
# backend default to the platform's line endings, so on Windows a regenerated
# artifact lands on disk with CRLF while the repository stores LF and every
# clean clone checks out LF.
NORMALIZE_SUFFIXES = {".csv", ".md", ".txt", ".json", ".svg"}


def normalize_line_endings(paths: list[Path]) -> int:
    """Rewrite CRLF as LF in text artifacts before hashing.

    Hashing the bytes as the generators wrote them would record hashes that no
    clean clone can reproduce, and Git would renormalize the files on the next
    touch anyway. Doing it here keeps the manifest a record of what a clone
    contains, whichever platform regenerated the artifacts.
    """
    changed = 0
    for path in paths:
        if path.suffix.lower() not in NORMALIZE_SUFFIXES:
            continue
        data = path.read_bytes()
        if b"\r\n" in data:
            path.write_bytes(data.replace(b"\r\n", b"\n"))
            changed += 1
    return changed


def main() -> None:
    artifacts = iter_artifacts()
    normalized = normalize_line_endings(artifacts)
    if normalized:
        print(f"Normalized line endings to LF in {normalized} text artifacts")

    rows = []
    for path in artifacts:
        rel = path.relative_to(ROOT).as_posix()
        rows.append(
            {
                "path": rel,
                "category": category(Path(rel)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    # csv.writer ends rows with CRLF by default, which Git then rewrites to LF on
    # the way in and hands back as LF on the way out, leaving this file reported as
    # modified after every run. Writing LF directly makes the output match what the
    # repository stores, so regenerating a manifest that has not changed is a no-op.
    with OUT.open("w", newline="\n", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "category", "bytes", "sha256"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT.relative_to(ROOT)} with {len(rows)} artifacts")


if __name__ == "__main__":
    main()
