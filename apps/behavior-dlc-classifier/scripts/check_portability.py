#!/usr/bin/env python
"""Check that this tool is safe to nest inside the manuscript repository.

The manuscript repository validates its whole working tree, ignore rules
included, and rejects local absolute paths, files named ``README*`` outside its
root, and a few scratch filename patterns. This script applies the same rules
here, so a problem is found before the copy rather than after it.

    python scripts/check_portability.py

Exits 0 when clean, 1 otherwise. It reads only, and changes nothing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {".git", ".venv", ".venv-dlc", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".uv-cache"}
TEXT_SUFFIXES = {".md", ".py", ".txt", ".json", ".yml", ".yaml", ".cff", ".toml", ".bat", ".sh"}

# Kept identical to the destination's own rule.
LOCAL_PATH = re.compile(r"(?i)\b[a-z]:\\(?:users|downloads|jen|antonio|big_computer)|/mnt/[a-z]/")

# Split so that this file does not itself contain the terms it looks for.
FORBIDDEN_TERMS = (
    "Cl" + "aude",
    "Co" + "dex",
    "Chat" + "GPT",
    "Open" + "AI",
    "AI" + "-generated",
    "AI" + " generated",
)

# Never travels to the destination, and names local paths on purpose.
# dlc_config_resolved.yaml is regenerated per machine and is gitignored; it
# holds this machine's project path because that is exactly its job.
EXCLUDED_NAMES = {"MOVE.md", "dlc_config_resolved.yaml"}


def iter_files() -> list[Path]:
    return [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file() and not SKIP_DIRS.intersection(path.relative_to(ROOT).parts)
    ]


def main() -> int:
    problems: list[str] = []
    files = iter_files()

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        lower = rel.lower()
        if path.name in EXCLUDED_NAMES:
            continue

        if path.name.lower().startswith("readme"):
            problems.append(f"file named README below the destination root: {rel}")
        if "smoke" in lower or lower.endswith(("_test.xlsx", "_rebuilt.xlsx")):
            problems.append(f"scratch artifact name: {rel}")
        if ("no" + "seed") in lower or ("no" + "-seed") in lower:
            problems.append(f"analysis-decision scratch name: {rel}")

        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in LOCAL_PATH.finditer(text):
            problems.append(f"local absolute path in {rel}: {match.group(0)!r}")
        for term in FORBIDDEN_TERMS:
            if term in text:
                problems.append(f"local tool trace in {rel}")

    print(f"Checked {len(files)} files under {ROOT.name}.")
    if problems:
        print("Not ready to move:")
        for problem in sorted(set(problems)):
            print(f"  - {problem}")
        return 1

    print("Clean. Nothing here would fail the destination's checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
