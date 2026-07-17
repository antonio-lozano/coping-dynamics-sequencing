#!/usr/bin/env python
"""Write MANIFEST.csv with hashes for publication artifacts."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "MANIFEST.csv"
ARTIFACT_DIRS = ("data", "figures", "report", "results")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def category(path: Path) -> str:
    parts = path.parts
    if parts[0] == "data":
        return "source_data"
    if parts[0] == "figures":
        return "manuscript_figure"
    if parts[0] == "report":
        return "manuscript_report"
    if parts[:2] == ("results", "figure_data"):
        return "intermediate_figure_data"
    if parts[:2] == ("results", "statistical_reports"):
        return "statistical_output"
    if parts[:2] == ("results", "behavior_classifier"):
        return "trained_model"
    return "result"


def iter_artifacts() -> list[Path]:
    paths: list[Path] = []
    for dirname in ARTIFACT_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        paths.extend(p for p in base.rglob("*") if p.is_file())
    return sorted(paths, key=lambda p: p.as_posix().lower())


def main() -> None:
    rows = []
    for path in iter_artifacts():
        rel = path.relative_to(ROOT).as_posix()
        rows.append(
            {
                "path": rel,
                "category": category(Path(rel)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "category", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT.relative_to(ROOT)} with {len(rows)} artifacts")


if __name__ == "__main__":
    main()
