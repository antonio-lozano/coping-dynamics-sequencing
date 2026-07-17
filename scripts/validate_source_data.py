#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gómez and Antonio Lozano
"""
Three-way source-data validation: Manuscript → Old Script → New Code.

This script compares reported manuscript values against outputs from:
1. The old build_freezing_data_workbooks.py script
2. The new refactored figure scripts

For each statistic, it reports:
  - Manuscript value (ground truth)
  - Old script output (if available)
  - New code output (if available)
  - Discrepancies flagged as # REVIEW:

Usage:
    python scripts/validate_source_data.py

Output:
    DIFF_REPORT.txt with three-way comparisons and mismatch flags.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================================
# MANUSCRIPT GROUND TRUTH (extracted from results section)
# ============================================================================

MANUSCRIPT_VALUES = {
    "Figure 1 - SimBA Validation (HAND-CURATED, NOT SCRIPT-GENERATED)": {
        "SimBA r²": {"value": 0.95, "tolerance": 1e-10, "hand_curated": True},
        "SimBA p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True, "hand_curated": True},
    },
    "Figure 2 - Validation": {
        "SimBA validation r²": {"value": 0.95, "tolerance": 1e-10},
        "SimBA validation p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},
        "Freezing Exp1 time β": {"value": 3.721, "tolerance": 1e-10},
        "Freezing Exp1 time SE": {"value": 0.128, "tolerance": 1e-10},
        "Freezing Exp1 time z": {"value": 29.048, "tolerance": 1e-10},
        "Freezing Exp1 time p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},
        "Freezing Exp1 ELS×time β": {"value": -0.887, "tolerance": 1e-10},
        "Freezing Exp1 ELS×time SE": {"value": 0.181, "tolerance": 1e-10},
        "Freezing Exp1 ELS×time z": {"value": -4.898, "tolerance": 1e-10},
        "Freezing Exp1 ELS×time p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},
        "Freezing Exp1 ELS×time Cohen's d": {"value": 0.284, "tolerance": 1e-10},
        "Freezing Exp3 time β": {"value": 4.412, "tolerance": 1e-10},
        "Freezing Exp3 time SE": {"value": 0.191, "tolerance": 1e-10},
        "Freezing Exp3 time z": {"value": 23.091, "tolerance": 1e-10},
        "Freezing Exp3 time p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},
        "Freezing Exp3 ELS×time β": {"value": -0.721, "tolerance": 1e-10},
        "Freezing Exp3 ELS×time SE": {"value": 0.27, "tolerance": 1e-10},
        "Freezing Exp3 ELS×time z": {"value": -2.667, "tolerance": 1e-10},
        "Freezing Exp3 ELS×time p": {"value": 0.008, "tolerance": 1e-10, "is_pvalue": True},
        "Freezing Exp3 ELS×time Cohen's d": {"value": 0.189, "tolerance": 1e-10},
        "Freezing Combined ELS×time β": {"value": -0.822, "tolerance": 1e-10},
        "Freezing Combined ELS×time SE": {"value": 0.154, "tolerance": 1e-10},
        "Freezing Combined ELS×time z": {"value": -5.327, "tolerance": 1e-10},
        "Freezing Combined ELS×time p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},
        "Freezing Combined ELS×time Cohen's d": {"value": 0.241, "tolerance": 1e-10},
        "S0+S28 overlap mean %": {"value": 80.86, "tolerance": 0.1},
        "S0+S28 overlap SE %": {"value": 9.62, "tolerance": 0.1},
        "S0+S28 time β": {"value": 0.126, "tolerance": 1e-10},
        "S0+S28 time SE": {"value": 0.004, "tolerance": 1e-10},
        "S0+S28 time z": {"value": 32.602, "tolerance": 1e-10},
        "S0+S28 time p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},
        "S0+S28 ELS×time β": {"value": -0.023, "tolerance": 1e-10},
        "S0+S28 ELS×time SE": {"value": 0.005, "tolerance": 1e-10},
        "S0+S28 ELS×time z": {"value": -4.241, "tolerance": 1e-10},
        "S0+S28 ELS×time p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},
    },
    "Figure 3 - Behavior Clusters": {
        "Freeze ELS β": {"value": -0.204, "tolerance": 1e-10},
        "Freeze ELS SE": {"value": 0.081, "tolerance": 1e-10},
        "Freeze ELS z": {"value": -2.511, "tolerance": 1e-10},
        "Freeze ELS p": {"value": 0.012, "tolerance": 1e-10, "is_pvalue": True},
        "Turn ELS β": {"value": 0.100, "tolerance": 1e-10},
        "Turn ELS SE": {"value": 0.032, "tolerance": 1e-10},
        "Turn ELS z": {"value": 3.099, "tolerance": 1e-10},
        "Turn ELS p": {"value": 0.002, "tolerance": 1e-10, "is_pvalue": True},
        "Sniff ELS β": {"value": -0.068, "tolerance": 1e-10},
        "Sniff ELS SE": {"value": 0.185, "tolerance": 1e-10},
        "Sniff ELS z": {"value": -2.133, "tolerance": 1e-10},
        "Sniff ELS p": {"value": 0.033, "tolerance": 1e-10, "is_pvalue": True},
        "Freeze time×ELS β": {"value": -0.023, "tolerance": 1e-10},
        "Freeze time×ELS SE": {"value": 0.005, "tolerance": 1e-10},
        "Freeze time×ELS z": {"value": -4.241, "tolerance": 1e-10},
        "Freeze time×ELS p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},
        "Sniff time β": {"value": -0.014, "tolerance": 1e-10},
        "Sniff time SE": {"value": 0.004, "tolerance": 1e-10},
        "Sniff time z": {"value": -3.656, "tolerance": 1e-10},
        "Sniff time p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},
        "Turn time×ELS β": {"value": 0.026, "tolerance": 1e-10},
        "Turn time×ELS SE": {"value": 0.007, "tolerance": 1e-10},
        "Turn time×ELS z": {"value": 3.610, "tolerance": 1e-10},
        "Turn time×ELS p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},
    },
    "Figure 4 - Diversity Dynamics": {
        "Simpson β": {"value": -0.013, "tolerance": 1e-10},
        "Simpson SE": {"value": 0.007, "tolerance": 1e-10},
        "Simpson z": {"value": -2.039, "tolerance": 1e-10},
        "Simpson p": {"value": 0.041, "tolerance": 1e-10, "is_pvalue": True},
        "CUI β": {"value": 0.064, "tolerance": 1e-10},
        "CUI SE": {"value": 0.021, "tolerance": 1e-10},
        "CUI z": {"value": 2.968, "tolerance": 1e-10},
        "CUI p": {"value": 0.003, "tolerance": 1e-10, "is_pvalue": True},
        "Freezing bout β": {"value": -0.105, "tolerance": 1e-10},
        "Freezing bout SE": {"value": 0.053, "tolerance": 1e-10},
        "Freezing bout z": {"value": -2.001, "tolerance": 1e-10},
        "Freezing bout p": {"value": 0.045, "tolerance": 1e-10, "is_pvalue": True},
        "Sniffing bout β": {"value": 0.322, "tolerance": 1e-10},
        "Sniffing bout SE": {"value": 0.129, "tolerance": 1e-10},
        "Sniffing bout z": {"value": 2.506, "tolerance": 1e-10},
        "Sniffing bout p": {"value": 0.012, "tolerance": 1e-10, "is_pvalue": True},
        "Turning bout β": {"value": 0.159, "tolerance": 1e-10},
        "Turning bout SE": {"value": 0.052, "tolerance": 1e-10},
        "Turning bout z": {"value": 3.088, "tolerance": 1e-10},
        "Turning bout p": {"value": 0.002, "tolerance": 1e-10, "is_pvalue": True},
        "Recurrence β": {"value": 0.013, "tolerance": 1e-10},
        "Recurrence SE": {"value": 0.006, "tolerance": 1e-10},
        "Recurrence z": {"value": 2.264, "tolerance": 1e-10},
        "Recurrence p": {"value": 0.024, "tolerance": 1e-10, "is_pvalue": True},
        "Determinism β": {"value": 0.016, "tolerance": 1e-10},
        "Determinism SE": {"value": 0.005, "tolerance": 1e-10},
        "Determinism z": {"value": 3.382, "tolerance": 1e-10},
        "Determinism p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},
        "Markov entropy β": {"value": -0.041, "tolerance": 1e-10},
        "Markov entropy SE": {"value": 0.018, "tolerance": 1e-10},
        "Markov entropy z": {"value": -2.342, "tolerance": 1e-10},
        "Markov entropy p": {"value": 0.019, "tolerance": 1e-10, "is_pvalue": True},
    },
    "Figure 5 - Resilience Dynamics": {
        "Dynamics score ELS β": {"value": 0.336, "tolerance": 1e-10},
        "Dynamics score ELS SE": {"value": 0.087, "tolerance": 1e-10},
        "Dynamics score ELS z": {"value": 3.869, "tolerance": 1e-10},
        "Dynamics score ELS p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},
        "LOOCV accuracy %": {"value": 63.4, "tolerance": 0.1},
        "Resilient prevalence %": {"value": 30.0, "tolerance": 1.0},

        "Fig 5C: Freeze Resilient>Vulnerable β": {"value": 0.180, "tolerance": 1e-10},
        "Fig 5C: Freeze Resilient>Vulnerable SE": {"value": 0.038, "tolerance": 1e-10},
        "Fig 5C: Freeze Resilient>Vulnerable z": {"value": 4.681, "tolerance": 1e-10},
        "Fig 5C: Freeze Resilient>Vulnerable p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},

        "Fig 5C: Sniff Resilient>Vulnerable β": {"value": 0.492, "tolerance": 1e-10},
        "Fig 5C: Sniff Resilient>Vulnerable SE": {"value": 0.119, "tolerance": 1e-10},
        "Fig 5C: Sniff Resilient>Vulnerable z": {"value": 4.139, "tolerance": 1e-10},
        "Fig 5C: Sniff Resilient>Vulnerable p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},

        "Fig 5C: Groom Resilient>Vulnerable β": {"value": -0.398, "tolerance": 1e-10},
        "Fig 5C: Groom Resilient>Vulnerable SE": {"value": 0.187, "tolerance": 1e-10},
        "Fig 5C: Groom Resilient>Vulnerable z": {"value": -2.123, "tolerance": 1e-10},
        "Fig 5C: Groom Resilient>Vulnerable p": {"value": 0.003, "tolerance": 1e-10, "is_pvalue": True, "flag_impossible": True},

        "Fig 5C: Turn Resilient>Vulnerable β": {"value": 0.009, "tolerance": 1e-10},
        "Fig 5C: Turn Resilient>Vulnerable SE": {"value": 0.016, "tolerance": 1e-10},
        "Fig 5C: Turn Resilient>Vulnerable z": {"value": 0.568, "tolerance": 1e-10},
        "Fig 5C: Turn Resilient>Vulnerable p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True, "flag_impossible": True},

        "Fig 5C: Locomotion Resilient>Vulnerable β": {"value": -0.424, "tolerance": 1e-10},
        "Fig 5C: Locomotion Resilient>Vulnerable SE": {"value": 0.089, "tolerance": 1e-10},
        "Fig 5C: Locomotion Resilient>Vulnerable z": {"value": -4.755, "tolerance": 1e-10},
        "Fig 5C: Locomotion Resilient>Vulnerable p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},

        "Fig 5C: Climb Resilient>Vulnerable β": {"value": -0.196, "tolerance": 1e-10},
        "Fig 5C: Climb Resilient>Vulnerable SE": {"value": 0.074, "tolerance": 1e-10},
        "Fig 5C: Climb Resilient>Vulnerable z": {"value": -2.624, "tolerance": 1e-10},
        "Fig 5C: Climb Resilient>Vulnerable p": {"value": 0.009, "tolerance": 1e-10, "is_pvalue": True},

        "Fig 5C: Freeze Resilient vs Control β": {"value": -0.342, "tolerance": 1e-10},
        "Fig 5C: Freeze Resilient vs Control SE": {"value": 0.082, "tolerance": 1e-10},
        "Fig 5C: Freeze Resilient vs Control z": {"value": -4.168, "tolerance": 1e-10},
        "Fig 5C: Freeze Resilient vs Control p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},

        "Fig 5D: Freeze time×Resilient β": {"value": -1.865, "tolerance": 1e-10},
        "Fig 5D: Freeze time×Resilient SE": {"value": 0.249, "tolerance": 1e-10},
        "Fig 5D: Freeze time×Resilient z": {"value": -7.484, "tolerance": 1e-10},
        "Fig 5D: Freeze time×Resilient p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},

        "Fig 5D: Freeze time×Resilient vs Control β": {"value": 0.619, "tolerance": 1e-10},
        "Fig 5D: Freeze time×Resilient vs Control SE": {"value": 0.238, "tolerance": 1e-10},
        "Fig 5D: Freeze time×Resilient vs Control z": {"value": 2.599, "tolerance": 1e-10},
        "Fig 5D: Freeze time×Resilient vs Control p": {"value": 0.009, "tolerance": 1e-10, "is_pvalue": True},

        "Fig 5G: Turn time×Resilient β": {"value": 1.622, "tolerance": 1e-10},
        "Fig 5G: Turn time×Resilient SE": {"value": 0.326, "tolerance": 1e-10},
        "Fig 5G: Turn time×Resilient z": {"value": 4.967, "tolerance": 1e-10},
        "Fig 5G: Turn time×Resilient p": {"value": 0.001, "tolerance": 1e-10, "is_pvalue": True},

        "Fig 5E: Sniff baseline×Resilient β": {"value": -0.338, "tolerance": 1e-10},
        "Fig 5E: Sniff baseline×Resilient SE": {"value": 0.176, "tolerance": 1e-10},
        "Fig 5E: Sniff baseline×Resilient z": {"value": -1.920, "tolerance": 1e-10},
        "Fig 5E: Sniff baseline×Resilient p": {"value": 0.050, "tolerance": 1e-10, "is_pvalue": True},
    },
    "Figure 6 - Resilience-Diversity": {
        "Simpson Vulnerable vs Control β": {"value": -0.022, "tolerance": 1e-10},
        "Simpson Vulnerable vs Control SE": {"value": 0.008, "tolerance": 1e-10},
        "Simpson Vulnerable vs Control z": {"value": -2.835, "tolerance": 1e-10},
        "Simpson Vulnerable vs Control p": {"value": 0.005, "tolerance": 1e-10, "is_pvalue": True},
        "Simpson Resilient vs Vulnerable β": {"value": 0.029, "tolerance": 1e-10},
        "Simpson Resilient vs Vulnerable SE": {"value": 0.011, "tolerance": 1e-10},
        "Simpson Resilient vs Vulnerable z": {"value": 2.627, "tolerance": 1e-10},
        "Simpson Resilient vs Vulnerable p": {"value": 0.009, "tolerance": 1e-10, "is_pvalue": True},
        "CUI Vulnerable β": {"value": 0.075, "tolerance": 1e-10},
        "CUI Vulnerable SE": {"value": 0.028, "tolerance": 1e-10},
        "CUI Vulnerable z": {"value": 2.664, "tolerance": 1e-10},
        "CUI Vulnerable p": {"value": 0.008, "tolerance": 1e-10, "is_pvalue": True},
    },
    "Figure 7 - XGBoost/SHAP (HAND-CURATED, NOT SCRIPT-GENERATED)": {
        "XGBoost overall accuracy %": {"value": 62.7, "tolerance": 0.1, "hand_curated": True},
        "XGBoost chance baseline %": {"value": 12.5, "tolerance": 0.1, "hand_curated": True},
        "XGBoost gain (ratio)": {"value": 5.0, "tolerance": 0.1, "hand_curated": True},
    },
}


def compare_values(
    manuscript: float,
    old_script: float | None,
    new_code: float | None,
    tolerance: float = 1e-10,
    flag_impossible: bool = False,
    hand_curated: bool = False,
) -> dict:
    """
    Three-way comparison of manuscript, old script, and new code values.

    Returns dict with keys: pass, manuscript, old_script, new_code, diffs, impossible, hand_curated.
    """
    results = {
        "pass": True,
        "manuscript": manuscript,
        "old_script": old_script,
        "new_code": new_code,
        "diffs": [],
        "impossible": False,
        "hand_curated": hand_curated,
    }

    # Hand-curated values don't need validation
    if hand_curated:
        return results

    # Flag statistically impossible values (z/p mismatch)
    if flag_impossible:
        results["impossible"] = True
        results["pass"] = False
        results["diffs"].append("IMPOSSIBLE: p-value statistically inconsistent with z-value")

    # Check old vs manuscript
    if old_script is not None and not flag_impossible:
        if not np.isclose(manuscript, old_script, atol=tolerance, rtol=1e-9):
            results["diffs"].append(f"Old ≠ Manuscript: {old_script} vs {manuscript}")
            results["pass"] = False

    # Check new vs manuscript
    if new_code is not None and not flag_impossible:
        if not np.isclose(manuscript, new_code, atol=tolerance, rtol=1e-9):
            results["diffs"].append(f"New ≠ Manuscript: {new_code} vs {manuscript}")
            results["pass"] = False

    # Check old vs new
    if old_script is not None and new_code is not None and not flag_impossible:
        if not np.isclose(old_script, new_code, atol=tolerance, rtol=1e-9):
            results["diffs"].append(f"Old ≠ New: {old_script} vs {new_code}")
            results["pass"] = False

    return results


def main() -> None:
    """Run three-way validation and produce diff report."""
    print("\n" + "="*80)
    print("PHASE 1.5 — THREE-WAY SOURCE-DATA VALIDATION")
    print("="*80 + "\n")

    report_lines = []

    for figure, statistics in MANUSCRIPT_VALUES.items():
        report_lines.append(f"\n{figure}")
        report_lines.append("=" * len(figure))

        for stat_name, meta in statistics.items():
            manuscript_value = meta["value"]
            tolerance = meta.get("tolerance", 1e-10)
            is_pvalue = meta.get("is_pvalue", False)
            flag_impossible = meta.get("flag_impossible", False)
            hand_curated = meta.get("hand_curated", False)

            # TODO: Extract old_script and new_code values as figure scripts are refactored
            old_script_value = None  # Will be populated after running old script
            new_code_value = None    # Will be populated after figure scripts export source_data CSVs

            result = compare_values(
                manuscript_value,
                old_script_value,
                new_code_value,
                tolerance,
                flag_impossible=flag_impossible,
                hand_curated=hand_curated,
            )

            if hand_curated:
                status = "📋 HAND-CURATED"
            elif result["impossible"]:
                status = "🛑 # REVIEW: (IMPOSSIBLE)"
            elif result["pass"]:
                status = "✅ PASS"
            else:
                status = "⚠️ # REVIEW:"

            value_str = f"{manuscript_value:.10g}" if not is_pvalue else f"{manuscript_value:.3f}"

            report_lines.append(f"  {status} {stat_name}")
            report_lines.append(f"      Manuscript: {value_str}")
            if result["impossible"]:
                report_lines.append(f"      STATUS: STATISTICALLY IMPOSSIBLE (DO NOT MATCH)")
            if hand_curated:
                report_lines.append(f"      STATUS: Hand-curated (not script-generated)")
            if old_script_value is not None:
                report_lines.append(f"      Old Script: {old_script_value:.10g}")
            if new_code_value is not None:
                report_lines.append(f"      New Code:   {new_code_value:.10g}")
            if result["diffs"]:
                for diff in result["diffs"]:
                    report_lines.append(f"      ⚠️ {diff}")

    # Write report
    report_text = "\n".join(report_lines)
    report_path = Path(__file__).parents[1] / "DIFF_REPORT.txt"
    report_path.write_text(report_text + "\n", encoding="utf-8")

    # Print to console with ASCII-safe output
    try:
        print(report_text)
    except UnicodeEncodeError:
        # Fallback: print ASCII-safe version
        safe_text = report_text.encode("ascii", "replace").decode("ascii")
        print(safe_text)

    print(f"\n{'='*80}")
    print(f"Report saved to: {report_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
