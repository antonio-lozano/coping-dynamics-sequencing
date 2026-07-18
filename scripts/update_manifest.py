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
    if parts[:2] == ("data", "raw"):
        return "raw_input"
    if parts[:2] == ("data", "derived"):
        return "derived_data"
    if parts[0] == "data":
        return "data_documentation"
    if parts[0] == "figures":
        return "manuscript_figure"
    if parts[0] == "report":
        return "manuscript_report"
    if parts[:2] == ("results", "source_data"):
        return "figure_source_data"
    if parts[:2] == ("results", "statistics"):
        return "statistical_output"
    if parts[:2] == ("results", "intermediate"):
        return "intermediate_output"
    if parts[:2] == ("results", "models"):
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
