"""Rebuild the full manuscript reproducibility package from tracked inputs.

This is the top-level command reviewers should use when they want to regenerate
the repository artifacts from `data/raw`.

Run: python scripts/run_all.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

TASKS = [
    (
        "build compact freezing prediction inputs",
        ["scripts/derive_tables/freezing_predictions_light.py"],
    ),
    ("derive processed cluster tables", ["scripts/derive_tables/cluster_tables.py"]),
    ("derive processed tracking-control tables", ["scripts/derive_tables/tracking_exclusions.py"]),
    ("derive Figure 6 resilience statistics", ["scripts/derive_tables/fig6_resilience_stats.py"]),
    ("regenerate data-derived figures", ["scripts/run_all_figures.py"]),
    ("build raw-data workbook", ["scripts/build_raw_data_workbook.py"]),
    ("build statistical report workbook", ["scripts/build_statistical_report.py"]),
    ("update artifact manifest", ["scripts/update_manifest.py"]),
    ("check reproducibility package", ["scripts/check_reproducibility.py"]),
    # Verification, so it runs last and against the finished artifacts.
    ("check manuscript, figures and report agree", ["scripts/check_manuscript.py"]),
]


def run(label: str, command: list[str], *, verbose: bool) -> float:
    print(f"\n{'=' * 72}\n  {label}\n  python {' '.join(command)}\n{'=' * 72}")
    start = time.time()
    env = os.environ.copy()
    env["BATCH_MODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env.setdefault("PYTHONWARNINGS", "ignore")
    cmd = [sys.executable, *command]
    if verbose:
        subprocess.run(cmd, cwd=REPO, env=env, check=True)
    else:
        result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        if result.returncode:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            raise subprocess.CalledProcessError(result.returncode, cmd)
    elapsed = time.time() - start
    print(f"[OK] {label}: {elapsed:.1f}s")
    return elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip slow figure rendering but still rebuild tables, reports, manifest, and checks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream each child script's output instead of showing only task-level progress.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_start = time.time()
    timings = []
    for label, command in TASKS:
        if args.skip_figures and command == ["scripts/run_all_figures.py"]:
            print(f"\n[SKIP] {label}")
            continue
        timings.append((label, run(label, command, verbose=args.verbose)))

    print(f"\n{'Task':<44} {'Time':>10}")
    print("-" * 57)
    for label, elapsed in timings:
        print(f"{label:<44} {elapsed:>9.1f}s")
    print("-" * 57)
    print(f"{'Total':<44} {time.time() - total_start:>9.1f}s")


if __name__ == "__main__":
    main()
