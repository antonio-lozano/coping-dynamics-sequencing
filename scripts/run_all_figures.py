#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""
Run All Figures - regenerate every data-derived manuscript figure.

Each script writes the clean manuscript figure (figureN.pdf/.svg/.png) to the
top-level ``figures/`` directory. Figure source data go to
``figure_source_data/``; statistical tables go to ``statistics/``; generated
analysis tables go to ``data/processed/``.

Usage:
    python scripts/run_all_figures.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "generate_figures"

# (label, script filename) in manuscript order.
FIGURE_SCRIPTS = [
    ("Figure 2 - Validation (keypoint-MoSeq)", "figure_2_validation.py"),
    ("Figure 3 - Behavior clusters over time", "figure_3_behavior_clusters.py"),
    ("Figure 4 - Diversity dynamics", "figure_4_diversity_dynamics.py"),
    ("Figure 5 - Resilience dynamics", "figure_5_resilience_dynamics.py"),
    ("Figure 6 - Resilience diversity", "figure_6_resilience_diversity.py"),
    ("Figure 7 - Resilience prediction", "figure_7_resilience_prediction.py"),
    ("Supplementary Figure 1 - Tracking clusters", "supplementary_figure_1_tracking_clusters.py"),
    ("Supplementary Figure 3 - Distance metrics", "supplementary_figure_3_distances.py"),
    ("Supplementary Figure 4 - Classifier SHAP", "supplementary_figure_4_classifier_shap.py"),
]


def run_script(label: str, script_path: Path) -> tuple[str, float]:
    print(f"\n{'=' * 70}\n  {label}\n  Script: {script_path.name}\n{'=' * 70}\n")
    start = time.time()
    env = os.environ.copy()
    env["BATCH_MODE"] = "1"  # skip plt.show()
    env["MPLBACKEND"] = "Agg"
    # Freeze the dates matplotlib embeds in PDF (CreationDate) and SVG
    # (dc:date) output; together with svg.hashsalt in the repository
    # matplotlibrc this makes an unchanged figure an unchanged file.
    env.setdefault("SOURCE_DATE_EPOCH", "0")
    try:
        subprocess.run(
            [sys.executable, str(script_path)], cwd=script_path.parents[2], check=True, env=env
        )
        duration = time.time() - start
        print(f"\n[OK] {label} completed in {duration:.1f}s")
        return "SUCCESS", duration
    except subprocess.CalledProcessError as exc:
        duration = time.time() - start
        print(f"\n[FAIL] {label} (exit {exc.returncode}) after {duration:.1f}s")
        return "FAILED", duration


def main() -> None:
    print("\n" + "=" * 70 + "\n  GENERATING ALL MANUSCRIPT FIGURES\n" + "=" * 70)
    results = []
    total_start = time.time()
    for label, script_name in FIGURE_SCRIPTS:
        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            print(f"\n[WARN] Script not found: {script_path}")
            results.append((label, "NOT FOUND", 0.0))
            continue
        status, duration = run_script(label, script_path)
        results.append((label, status, duration))

    print("\n" + "=" * 70 + "\n  SUMMARY\n" + "=" * 70)
    print(f"\n{'Figure':<45} {'Status':<10} {'Time':>8}")
    print("-" * 65)
    n_failed = 0
    for label, status, duration in results:
        if status == "FAILED":
            n_failed += 1
        time_str = f"{duration:.1f}s" if duration else "-"
        print(f"{label:<45} {status:<10} {time_str:>8}")
    print("-" * 65)
    print(f"Total time: {time.time() - total_start:.1f}s")

    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
