#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Check that the manuscript, the figures and the statistical report agree.

Four sources have to carry the same numbers:

    computation   statistics/*.csv                 written by the analysis scripts
    report        report/statistical_report.xlsx   the submitted workbook
    manuscript    the Results text of a .docx
    figures       the significance markers drawn in figures/

They agree by construction only if nothing has drifted, so this script proves it
instead of assuming it. It runs five checks:

    1. manuscript vs computation    every quoted statistic against its CSV row
    2. manuscript vs report         the same statistic against its workbook row
    3. computation vs report        so a stale export cannot hide behind a
                                    manuscript that happens to quote the workbook
    4. figures vs computation       every asterisk drawn on Figures 4 and 6
    5. coverage                     every statistic in the .docx is claimed by the
                                    registry below, and every registry claim is
                                    found in the .docx

Claims are tied to their sources by the explicit registry below, not by fuzzy
search: several panels report the same coefficient at different time units (per
30 s bin, per second, per minute) and a fuzzy matcher pairs them with the wrong
row without complaining.

Check 5 is what keeps the registry honest. Adding a statistic to the manuscript
without adding it here fails the run, so no claim can go unchecked.

Run:
    python scripts/check_manuscript.py [MANUSCRIPT.docx]

Exits non-zero on any disagreement.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import docx
import openpyxl
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
STATS = REPO / "statistics"
WORKBOOK = REPO / "report" / "statistical_report.xlsx"
SHIPPED_MS = (
    REPO / "report" / "Sanguino-Gómez_coping_strategies_Nature_Neuroscience_highlighted.docx"
)

TOL = 0.0011  # the manuscript quotes three decimals
CROSS_TOL = 5e-4  # computation vs report, both unrounded
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def C(panel, ms, *, scale=1, csv_scale=1, csv=None, wb=None, note=""):
    """One manuscript claim and where it comes from.

    ``ms``        the (beta, SE, z) triple as the manuscript prints it, or None
                  when the panel carries no statistic in the Results text.
    ``scale``     divides the workbook value to reach the manuscript's unit.
    ``csv_scale`` does the same for the statistics/ export, which does not always
                  store the workbook's unit: the timecourse exports are per
                  second and the workbook rescales them to per minute.
    z and p are scale-invariant.
    """
    return dict(panel=panel, ms=ms, scale=scale, csv_scale=csv_scale, csv=csv, wb=wb, note=note)


# csv: (filename, {column: value, ...}, coef_col, se_col, z_col, p_col)
# wb:  (sheet fragment, section fragment, row-label fragment, occurrence index)
CLAIMS = [
    C(
        "Fig 2A time",
        (3.721, 0.128, 29.048),
        scale=2,
        csv=(
            "stats_figure2_ground_truth_MixedLM.csv",
            {"section": "Sanguino-Gomez & Krugers", "parameter": "Time (per minute)"},
            "coef",
            "se",
            "z",
            "p_value",
        ),
        csv_scale=2,
        wb=("Fig.2A-C", "Sanguino-Gomez & Kru", "Time (per minute)", 0),
        note="per 30 s bin",
    ),
    C(
        "Fig 2A ELS x time",
        (-0.887, 0.181, -4.898),
        scale=2,
        csv=(
            "stats_figure2_ground_truth_MixedLM.csv",
            {"section": "Sanguino-Gomez & Krugers", "parameter": "Stress x time (per minute)"},
            "coef",
            "se",
            "z",
            "p_value",
        ),
        csv_scale=2,
        wb=("Fig.2A-C", "Sanguino-Gomez & Kru", "Stress x time", 0),
        note="per 30 s bin",
    ),
    C(
        "Fig 2B time",
        (4.412, 0.191, 23.091),
        scale=2,
        csv=(
            "stats_figure2_ground_truth_MixedLM.csv",
            {"section": "Sanguino-Gomez et al.", "parameter": "Time (per minute)"},
            "coef",
            "se",
            "z",
            "p_value",
        ),
        csv_scale=2,
        wb=("Fig.2A-C", "Sanguino-Gomez et al", "Time (per minute)", 0),
        note="per 30 s bin",
    ),
    C(
        "Fig 2B ELS x time",
        (-0.721, 0.270, -2.667),
        scale=2,
        csv=(
            "stats_figure2_ground_truth_MixedLM.csv",
            {"section": "Sanguino-Gomez et al.", "parameter": "Stress x time (per minute)"},
            "coef",
            "se",
            "z",
            "p_value",
        ),
        csv_scale=2,
        wb=("Fig.2A-C", "Sanguino-Gomez et al", "Stress x time", 0),
        note="per 30 s bin",
    ),
    C(
        "Fig 2C ELS x time",
        (-0.822, 0.154, -5.327),
        scale=2,
        csv=(
            "stats_figure2_ground_truth_MixedLM.csv",
            {"section": "Combined full dataset", "parameter": "Stress x time (per minute)"},
            "coef",
            "se",
            "z",
            "p_value",
        ),
        csv_scale=2,
        wb=("Fig.2A-C", "Combined full", "Stress x time", 0),
        note="per 30 s bin",
    ),
    C(
        "Fig 2H time",
        (0.126, 0.004, 32.620),
        scale=60,
        csv=(
            "stats_figure2h_syllables_MixedLM.csv",
            {
                "section": "Syllables 0 and 28 - Combined full dataset",
                "parameter": "Time (per minute)",
            },
            "coef",
            "se",
            "z",
            "p_value",
        ),
        csv_scale=60,
        wb=("Fig.2H", "Syllables 0 and 28", "Time (per minute)", 0),
        note="per second",
    ),
    C(
        "Fig 2H ELS x time",
        (-0.023, 0.005, -4.262),
        scale=60,
        csv=(
            "stats_figure2h_syllables_MixedLM.csv",
            {
                "section": "Syllables 0 and 28 - Combined full dataset",
                "parameter": "Stress x time (per minute)",
            },
            "coef",
            "se",
            "z",
            "p_value",
        ),
        csv_scale=60,
        wb=("Fig.2H", "Syllables 0 and 28", "Stress x time", 0),
        note="per second",
    ),
    C(
        "Fig 3A freeze",
        (-0.204, 0.081, -2.510),
        csv=(
            "stats_figure3A_GEE.csv",
            {"cluster": "Freeze", "parameter": "C(group, Treatment('Control'))[T.ELS]"},
            "beta",
            "se",
            "z",
            "p_value",
        ),
        wb=("Fig.3A", "Freeze", "Stress", 0),
    ),
    C(
        "Fig 3A sniff",
        (0.623, 0.231, 2.696),
        csv=(
            "stats_figure3A_GEE.csv",
            {"cluster": "Sniff", "parameter": "C(group, Treatment('Control'))[T.ELS]"},
            "beta",
            "se",
            "z",
            "p_value",
        ),
        wb=("Fig.3A", "Sniff", "Stress", 0),
    ),
    C(
        "Fig 3A turn",
        (0.101, 0.032, 3.099),
        csv=(
            "stats_figure3A_GEE.csv",
            {"cluster": "Turn", "parameter": "C(group, Treatment('Control'))[T.ELS]"},
            "beta",
            "se",
            "z",
            "p_value",
        ),
        wb=("Fig.3A", "Turn", "Stress", 0),
    ),
    # The overtime export already stores coefficients per second, which is the
    # unit the text quotes; the workbook rescales the same fit to per minute.
    C(
        "Fig 3B freeze x time",
        (-0.023, 0.005, -4.262),
        scale=60,
        csv=(
            "stats_figure3_overtime.csv",
            {
                "cluster": "Freeze",
                "analysis": "Combined",
                "parameter": "group[T.ELS]:time_bin_numeric",
            },
            "coef",
            "std_err",
            "z",
            "p_value",
        ),
        csv_scale=1,
        wb=("Fig.3B-H", "Freeze", "Stress x time", 0),
        note="per second",
    ),
    C(
        "Fig 3C sniff x time",
        (-0.014, 0.004, -3.656),
        scale=60,
        csv=(
            "stats_figure3_overtime.csv",
            {
                "cluster": "Sniff",
                "analysis": "Combined",
                "parameter": "group[T.ELS]:time_bin_numeric",
            },
            "coef",
            "std_err",
            "z",
            "p_value",
        ),
        csv_scale=1,
        wb=("Fig.3B-H", "Sniff", "Stress x time", 0),
        note="per second",
    ),
    C(
        "Fig 3E turn x time",
        (0.026, 0.007, 3.610),
        scale=60,
        csv=(
            "stats_figure3_overtime.csv",
            {
                "cluster": "Turn",
                "analysis": "Combined",
                "parameter": "group[T.ELS]:time_bin_numeric",
            },
            "coef",
            "std_err",
            "z",
            "p_value",
        ),
        csv_scale=1,
        wb=("Fig.3B-H", "Turn", "Stress x time", 0),
        note="per second",
    ),
    C(
        "Fig 3F locomotion x time",
        (0.005, 0.002, 2.553),
        scale=60,
        csv=(
            "stats_figure3_overtime.csv",
            {
                "cluster": "Locomotion",
                "analysis": "Combined",
                "parameter": "group[T.ELS]:time_bin_numeric",
            },
            "coef",
            "std_err",
            "z",
            "p_value",
        ),
        csv_scale=1,
        wb=("Fig.3B-H", "Locomotion", "Stress x time", 0),
        note="per second",
    ),
    C(
        "Fig 4E Simpson",
        (-0.013, 0.007, -1.893),
        csv=(
            "stats_figure4_diversity_MixedLM.csv",
            {"metric": "Simpson Index"},
            "coef",
            "se",
            "z",
            "p_value",
        ),
        wb=("Fig.4E-H", "Simpson", "Stress", 0),
    ),
    C(
        "Fig 4H CUI",
        (0.064, 0.030, 2.096),
        csv=(
            "stats_figure4_diversity_MixedLM.csv",
            {"metric": "Cumulative usage index"},
            "coef",
            "se",
            "z",
            "p_value",
        ),
        wb=("Fig.4E-H", "Cumulative usage", "Stress", 0),
    ),
    C(
        "Fig 4K freeze bout",
        None,
        csv=(
            "stats_figure4_bouts_MixedLM.csv",
            {"cluster": "Freezing"},
            "coef",
            "se",
            "z",
            "p_value",
        ),
        wb=("Fig.4J-Q", "Freezing", "Stress", 0),
        note="not quoted in the text",
    ),
    C(
        "Fig 4L sniff bout",
        (0.321, 0.147, 2.182),
        csv=(
            "stats_figure4_bouts_MixedLM.csv",
            {"cluster": "Sniffing"},
            "coef",
            "se",
            "z",
            "p_value",
        ),
        wb=("Fig.4J-Q", "Sniffing", "Stress", 0),
    ),
    C(
        "Fig 4N turn bout",
        (0.164, 0.071, 2.318),
        csv=("stats_figure4_bouts_MixedLM.csv", {"cluster": "Turn"}, "coef", "se", "z", "p_value"),
        wb=("Fig.4J-Q", "Turn", "Stress", 0),
    ),
    C(
        "Fig 4U recurrence",
        (0.013, 0.007, 1.893),
        csv=(
            "stats_figure4_transition_MixedLM.csv",
            {"metric": "Recurrence rate"},
            "coef",
            "se",
            "z",
            "p_value",
        ),
        wb=("Fig.4T-W", "Recurrence", "Stress", 0),
    ),
    C(
        "Fig 4V determinism",
        (0.016, 0.009, 1.778),
        csv=(
            "stats_figure4_transition_MixedLM.csv",
            {"metric": "Determinism"},
            "coef",
            "se",
            "z",
            "p_value",
        ),
        wb=("Fig.4T-W", "Determinism", "Stress", 0),
    ),
    C(
        "Fig 4W Markov entropy",
        None,
        csv=(
            "stats_figure4_transition_MixedLM.csv",
            {"metric": "Markov entropy"},
            "coef",
            "se",
            "z",
            "p_value",
        ),
        wb=("Fig.4T-W", "Markov", "Stress", 0),
        note="not quoted in the text",
    ),
    C(
        "Fig 5B dynamics score",
        (0.336, 0.104, 3.222),
        csv=(
            "stats_figure5_dynamics.csv",
            {"section": "Euclidean Distance - Combined full dataset", "parameter": "Stress"},
            "coef",
            "se",
            "z",
            "p_value",
        ),
        csv_scale=1,
        wb=("Fig.5A-B", "Euclidean", "Stress", 0),
    ),
    C(
        "Fig 5C freeze res-vs-vuln",
        (0.409, 0.118, 3.475),
        csv=(
            "fig5c_frequency_gee.csv",
            {"cluster": "Freeze", "contrast": "ELS/Resilient"},
            "beta",
            "SE",
            "z",
            "p",
        ),
        wb=("Fig.5C", "Freeze", "ELS resilient - ELS vulnerab", 0),
    ),
    C(
        "Fig 5C turn res-vs-vuln",
        (-0.145, 0.033, -4.437),
        csv=(
            "fig5c_frequency_gee.csv",
            {"cluster": "Turn", "contrast": "ELS/Resilient"},
            "beta",
            "SE",
            "z",
            "p",
        ),
        wb=("Fig.5C", "Turn", "ELS resilient - ELS vulnerab", 0),
    ),
    # The timecourse export stores coefficients per minute; the text reports per 30 s bin.
    C(
        "Fig 5D freeze res-vs-vuln x time",
        (1.865, 0.249, 7.484),
        scale=2,
        csv=(
            "stats_figure5_timecourse_MixedLM.csv",
            {
                "section": "Freeze - Combined full dataset",
                "parameter": "ELS resilient - ELS vulnerable x time (per minute)",
            },
            "coef",
            "se",
            "z",
            "p_value",
        ),
        csv_scale=2,
        wb=("Fig.5D-J", "Freeze", "ELS resilient - ELS vulnerab", 1),
        note="per 30 s bin",
    ),
    C(
        "Fig 5D freeze res-vs-ctrl x time",
        (0.619, 0.238, 2.599),
        scale=2,
        csv=(
            "stats_figure5_timecourse_MixedLM.csv",
            {
                "section": "Freeze - Combined full dataset",
                "parameter": "ELS resilient - Control x time (per minute)",
            },
            "coef",
            "se",
            "z",
            "p_value",
        ),
        csv_scale=2,
        wb=("Fig.5D-J", "Freeze", "ELS resilient - Control x", 0),
        note="per 30 s bin",
    ),
    C(
        "Fig 5E sniff res-vs-vuln x time",
        (0.338, 0.176, 1.920),
        scale=2,
        csv=(
            "stats_figure5_timecourse_MixedLM.csv",
            {
                "section": "Sniff - Combined full dataset",
                "parameter": "ELS resilient - ELS vulnerable x time (per minute)",
            },
            "coef",
            "se",
            "z",
            "p_value",
        ),
        csv_scale=2,
        wb=("Fig.5D-J", "Sniff", "ELS resilient - ELS vulnerab", 1),
        note="per 30 s bin",
    ),
    C(
        "Fig 5G turn res-vs-vuln x time",
        (-1.622, 0.326, -4.967),
        scale=2,
        csv=(
            "stats_figure5_timecourse_MixedLM.csv",
            {
                "section": "Turn - Combined full dataset",
                "parameter": "ELS resilient - ELS vulnerable x time (per minute)",
            },
            "coef",
            "se",
            "z",
            "p_value",
        ),
        csv_scale=2,
        wb=("Fig.5D-J", "Turn", "ELS resilient - ELS vulnerab", 1),
        note="per 30 s bin",
    ),
    C(
        "Fig 6G Simpson vuln-vs-ctrl",
        (-0.022, 0.007, -3.017),
        csv=(
            "fig6_diversity_resilience_stats.csv",
            {"metric": "Simpson", "contrast": "vuln_vs_control"},
            "beta",
            "SE",
            "z",
            "p",
        ),
        wb=("Fig.6G-K", "Simpson", "ELS vulnerable - Control", 0),
    ),
    C(
        "Fig 6G Simpson res-vs-vuln",
        (0.029, 0.010, 3.091),
        csv=(
            "fig6_diversity_resilience_stats.csv",
            {"metric": "Simpson", "contrast": "resilient_vs_vuln"},
            "beta",
            "SE",
            "z",
            "p",
        ),
        wb=("Fig.6G-K", "Simpson", "ELS resilient - ELS vulnerab", 0),
    ),
    C(
        "Fig 6M freeze bout res-vs-vuln",
        (0.263, 0.100, 2.644),
        csv=(
            "fig6_bout_resilience_stats.csv",
            {"cluster": "Freezing", "contrast": "resilient_vs_vuln"},
            "beta",
            "SE",
            "z",
            "p",
        ),
        wb=("Fig.6L-S", "Freeze", "ELS resilient - ELS vulnerab", 0),
    ),
    C(
        "Fig 6N sniff bout vuln-vs-ctrl",
        (0.420, 0.178, 2.364),
        csv=(
            "fig6_bout_resilience_stats.csv",
            {"cluster": "Sniffing", "contrast": "vuln_vs_control"},
            "beta",
            "SE",
            "z",
            "p",
        ),
        wb=("Fig.6L-S", "Sniff", "ELS vulnerable - Control", 0),
    ),
    C(
        "Fig 6P turn bout res-vs-vuln",
        (-0.201, 0.103, -1.963),
        csv=(
            "fig6_bout_resilience_stats.csv",
            {"cluster": "Turn", "contrast": "resilient_vs_vuln"},
            "beta",
            "SE",
            "z",
            "p",
        ),
        wb=("Fig.6L-S", "Turn", "ELS resilient - ELS vulnerab", 0),
    ),
    C(
        "Fig 6W Lempel-Ziv vuln-vs-ctrl",
        (-8.965, 5.224, -1.716),
        csv=(
            "fig6_transition_resilience_stats.csv",
            {"metric": "LZ_complexity", "contrast": "vuln_vs_control"},
            "beta",
            "SE",
            "z",
            "p",
        ),
        wb=("Fig.6W-Z", "Lempel-Ziv", "ELS vulnerable - Control", 0),
    ),
    C(
        "Fig 6X recurrence vuln-vs-ctrl",
        (0.026, 0.008, 3.237),
        csv=(
            "fig6_transition_resilience_stats.csv",
            {"metric": "Recurrence", "contrast": "vuln_vs_control"},
            "beta",
            "SE",
            "z",
            "p",
        ),
        wb=("Fig.6W-Z", "Recurrence", "ELS vulnerable - Control", 0),
    ),
    C(
        "Fig 6X recurrence res-vs-vuln",
        (-0.033, 0.010, -3.253),
        csv=(
            "fig6_transition_resilience_stats.csv",
            {"metric": "Recurrence", "contrast": "resilient_vs_vuln"},
            "beta",
            "SE",
            "z",
            "p",
        ),
        wb=("Fig.6W-Z", "Recurrence", "ELS resilient - ELS vulnerab", 0),
    ),
    C(
        "Fig 6Y determinism vuln-vs-ctrl",
        (0.010, 0.004, 2.204),
        csv=(
            "fig6_transition_resilience_stats.csv",
            {"metric": "Determinism", "contrast": "vuln_vs_control"},
            "beta",
            "SE",
            "z",
            "p",
        ),
        wb=("Fig.6W-Z", "Determinism", "ELS vulnerable - Control", 0),
    ),
    C(
        "Fig 6Z Markov vuln-vs-ctrl",
        (-0.044, 0.029, -1.502),
        csv=(
            "fig6_transition_resilience_stats.csv",
            {"metric": "MarkovEntropyIdx", "contrast": "vuln_vs_control"},
            "beta",
            "SE",
            "z",
            "p",
        ),
        wb=("Fig.6W-Z", "Markov", "ELS vulnerable - Control", 0),
    ),
]


# ---------------------------------------------------------------------------
# Figure significance markers
#
# The asterisks in the figure scripts are literals in the plotting calls, not
# values read from statistics/. That is fine for layout control, but it means a
# model change can silently leave a figure claiming significance the statistics
# no longer support - which is what happened when the unidentifiable animal
# random intercept was dropped (see coping_dynamics.statistics.fit_grouped_or_ols). The
# specification below must be kept in step with the figure scripts.
# ---------------------------------------------------------------------------
# panel -> (statistics file, key column, row key, starred?); two groups, so a
# star means the Condition[T.ELS] row is p < ALPHA.
FIG4_STARS = [
    ("4E  Simpson index", "stats_figure4_diversity_MixedLM.csv", "metric", "Simpson Index", False),
    (
        "4F  Shannon entropy",
        "stats_figure4_diversity_MixedLM.csv",
        "metric",
        "Shannon entropy",
        False,
    ),
    ("4G  Evenness", "stats_figure4_diversity_MixedLM.csv", "metric", "Evenness", False),
    (
        "4H  Cumulative usage index",
        "stats_figure4_diversity_MixedLM.csv",
        "metric",
        "Cumulative usage index",
        True,
    ),
    ("4K  Freeze bout", "stats_figure4_bouts_MixedLM.csv", "cluster", "Freezing", False),
    ("4L  Sniff bout", "stats_figure4_bouts_MixedLM.csv", "cluster", "Sniffing", True),
    ("4M  Groom bout", "stats_figure4_bouts_MixedLM.csv", "cluster", "Grooming", False),
    ("4N  Turn bout", "stats_figure4_bouts_MixedLM.csv", "cluster", "Turn", True),
    ("4O  Locomotion bout", "stats_figure4_bouts_MixedLM.csv", "cluster", "Locomotion", False),
    ("4P  Climb bout", "stats_figure4_bouts_MixedLM.csv", "cluster", "Climbing", False),
    ("4Q  Jump bout", "stats_figure4_bouts_MixedLM.csv", "cluster", "Jump", False),
    (
        "4T  Lempel-Ziv",
        "stats_figure4_transition_MixedLM.csv",
        "metric",
        "Lempel-Ziv complexity",
        False,
    ),
    ("4U  Recurrence", "stats_figure4_transition_MixedLM.csv", "metric", "Recurrence rate", False),
    ("4V  Determinism", "stats_figure4_transition_MixedLM.csv", "metric", "Determinism", False),
    (
        "4W  Markov entropy",
        "stats_figure4_transition_MixedLM.csv",
        "metric",
        "Markov entropy",
        False,
    ),
]

# Three groups. Bracket (0,1) is vulnerable-vs-control and (1,2) is
# resilient-vs-vulnerable, matching GROUP_ORDER in the figure script.
BRACKET = {(0, 1): "vuln_vs_control", (1, 2): "resilient_vs_vuln"}
FIG6_STARS = [
    (
        "6G  Simpson index",
        "fig6_diversity_resilience_stats.csv",
        "metric",
        "Simpson",
        [(0, 1), (1, 2)],
    ),
    ("6H  Shannon entropy", "fig6_diversity_resilience_stats.csv", "metric", "Entropy", []),
    ("6I  Evenness", "fig6_diversity_resilience_stats.csv", "metric", "Evenness", []),
    ("6J  Cumulative usage", "fig6_diversity_resilience_stats.csv", "metric", "CUI", []),
    ("6M  Freeze bout", "fig6_bout_resilience_stats.csv", "cluster", "Freezing", [(0, 1), (1, 2)]),
    ("6N  Sniff bout", "fig6_bout_resilience_stats.csv", "cluster", "Sniffing", [(0, 1)]),
    ("6O  Groom bout", "fig6_bout_resilience_stats.csv", "cluster", "Grooming", []),
    ("6P  Turn bout", "fig6_bout_resilience_stats.csv", "cluster", "Turn", [(0, 1), (1, 2)]),
    ("6Q  Locomotion bout", "fig6_bout_resilience_stats.csv", "cluster", "Locomotion", []),
    ("6R  Climb bout", "fig6_bout_resilience_stats.csv", "cluster", "Climbing", []),
    ("6S  Jump bout", "fig6_bout_resilience_stats.csv", "cluster", "Jump", []),
    ("6W  Lempel-Ziv", "fig6_transition_resilience_stats.csv", "metric", "LZ_complexity", []),
    (
        "6X  Recurrence",
        "fig6_transition_resilience_stats.csv",
        "metric",
        "Recurrence",
        [(0, 1), (1, 2)],
    ),
    ("6Y  Determinism", "fig6_transition_resilience_stats.csv", "metric", "Determinism", [(0, 1)]),
    (
        "6Z  Markov entropy",
        "fig6_transition_resilience_stats.csv",
        "metric",
        "MarkovEntropyIdx",
        [],
    ),
]

# The "Overall" bout panels average across clusters. That model is fitted only by
# the report builder, so it has no row in statistics/ and is checked against the
# workbook instead. Leaving these panels out is how a wrong asterisk survived on
# Figure 6L, so every panel a figure draws belongs in one of these three tables.
# (sheet fragment, section fragment, row-label fragment, occurrence), starred?
OVERALL_STARS = [
    ("4J  Overall bout", ("Fig.4J-Q", "Overall - Combined", "Stress", 0), False),
    (
        "6L  Overall bout [vuln_vs_control]",
        ("Fig.6L-S", "Overall - Combined", "ELS vulnerable - Control", 0),
        False,
    ),
    (
        "6L  Overall bout [resilient_vs_vuln]",
        ("Fig.6L-S", "Overall - Combined", "ELS resilient - ELS vulnerab", 0),
        False,
    ),
]


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
# "beta = -0.013, SE = 0.007, z = -1.893, p = 0.058" with p optional, because a
# few sentences give the p value after a semicolon or omit it entirely. The
# BH-FDR value, when quoted, is checked against the workbook's P_BH_FDR column;
# the text writes it as "BH_FDR p = 0.055", "BH-FDR p < 0.001" or "BH_FDR = 0.002".
STAT_RE = re.compile(
    r"β\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"SE\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"z\s*=\s*(-?\d+(?:\.\d+)?)"
    r"(?:\s*[,;]\s*p\s*(=|<)\s*(\d+(?:\.\d+)?))?"
    r"(?:\s*[,;]\s*BH[-_]FDR\s*(?:p\s*)?(=|<)\s*(\d+(?:\.\d+)?))?"
)
FIG_RE = re.compile(r"Fig\.\s*(\d+[A-Z])")


def read_manuscript(path: Path) -> list[dict]:
    """Every (beta, SE, z[, p]) statistic in the document, in reading order."""
    text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    text = text.replace("−", "-").replace(" ", " ").replace(" ", " ")
    found = []
    for m in STAT_RE.finditer(text):
        beta, se, z, op, p, bh_op, bh = m.groups()
        tail = text[m.end() : m.end() + 130]
        tag = FIG_RE.search(tail)
        found.append(
            dict(
                beta=float(beta),
                se=float(se),
                z=float(z),
                p_op=op,
                p=float(p) if p is not None else None,
                bh_op=bh_op,
                bh=float(bh) if bh is not None else None,
                fig=tag.group(1) if tag else "?",
                context=text[max(0, m.start() - 60) : m.end() + 40].replace("\n", " "),
            )
        )
    return found


def index_workbook(path: Path) -> list[dict]:
    """Flatten every coefficient block in the workbook into labelled rows."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        grid = [list(r) for r in ws.iter_rows(values_only=True)]
        heads = [
            (ri, ci)
            for ri, row in enumerate(grid)
            for ci, v in enumerate(row)
            if isinstance(v, str) and v.strip() in ("Coef.", "coef")
        ]
        for ri, ci in heads:
            section = ""
            for rj in range(ri - 1, -1, -1):
                for cj in range(ci, max(-1, ci - 12), -1):
                    if (
                        cj < len(grid[rj])
                        and isinstance(grid[rj][cj], str)
                        and grid[rj][cj].strip()
                    ):
                        section = grid[rj][cj].strip()
                        break
                if section:
                    break
            has_bh = ci + 4 < len(grid[ri]) and grid[ri][ci + 4] == "P_BH_FDR"
            rj = ri + 1
            while rj < len(grid):
                row = grid[rj]
                vals = [row[ci + k] if ci + k < len(row) else None for k in range(4)]
                if not isinstance(vals[0], (int, float)):
                    break
                label = row[ci - 1] if ci > 0 and ci - 1 < len(row) else None
                bh = row[ci + 4] if has_bh and ci + 4 < len(row) else None
                out.append(
                    dict(
                        sheet=ws.title,
                        section=section,
                        label=str(label).strip(),
                        coef=vals[0],
                        se=vals[1],
                        z=vals[2],
                        p=vals[3],
                        bh=bh if isinstance(bh, (int, float)) else None,
                    )
                )
                rj += 1
    return out


def wb_lookup(index, spec):
    sheet, section, label, occurrence = spec
    hits = [
        r
        for r in index
        if sheet in r["sheet"]
        and section.lower() in r["section"].lower()
        and label.lower() in r["label"].lower()
    ]
    return hits[occurrence] if len(hits) > occurrence else None


def csv_lookup(spec):
    filename, filters, c_col, s_col, z_col, p_col = spec
    table = pd.read_csv(STATS / filename)
    mask = pd.Series(True, index=table.index)
    for column, value in filters.items():
        mask &= table[column].astype(str).str.strip() == value
    sub = table[mask]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return dict(
        coef=float(row[c_col]), se=float(row[s_col]), z=float(row[z_col]), p=float(row[p_col])
    )


def close(a, b, tol=TOL):
    return a is not None and b is not None and abs(a - b) < tol


def p_agrees(claim_p_op, claim_p, source_p):
    """The manuscript writes p either as '= 0.058' or as '< 0.001'."""
    if claim_p is None or source_p is None:
        return True
    if claim_p_op == "<":
        return source_p < claim_p
    return abs(round(source_p, 3) - claim_p) < TOL


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_statistics(index, ms_stats, fails):
    """Checks 1-3 and 5: the registry against manuscript, computation and report."""
    unclaimed = list(ms_stats) if ms_stats is not None else None
    rows = []

    for claim in CLAIMS:
        panel, scale, cscale = claim["panel"], claim["scale"], claim["csv_scale"]
        mb, mse, mz = claim["ms"] if claim["ms"] else (None, None, None)

        # 5 = coverage: find and consume this claim's statistic in the .docx
        quoted = None
        if unclaimed is not None and claim["ms"] is not None:
            for cand in unclaimed:
                if close(cand["beta"], mb) and close(cand["se"], mse) and close(cand["z"], mz):
                    quoted = cand
                    unclaimed.remove(cand)
                    break
            if quoted is None:
                fails.append(f"{panel}: {mb}/{mse}/{mz} is not written in the manuscript")

        wb_row = wb_lookup(index, claim["wb"]) if claim["wb"] else None
        csv_row = csv_lookup(claim["csv"]) if claim["csv"] else None

        # 2 = report
        if claim["wb"] is None:
            report = "n/a"
        elif wb_row is None:
            report = "MISSING"
            fails.append(f"{panel}: not found in the workbook")
        else:
            ok = claim["ms"] is None or (
                close(round(wb_row["coef"] / scale, 3), mb)
                and close(round(wb_row["se"] / scale, 3), mse)
                and close(round(wb_row["z"], 3), mz)
            )
            if quoted is not None and not p_agrees(quoted["p_op"], quoted["p"], wb_row["p"]):
                ok = False
            # A quoted BH-FDR value must match the workbook's P_BH_FDR column.
            if (
                quoted is not None
                and quoted["bh"] is not None
                and wb_row.get("bh") is not None
                and not p_agrees(quoted["bh_op"], quoted["bh"], wb_row["bh"])
            ):
                ok = False
                fails.append(
                    f"{panel}: manuscript quotes BH_FDR {quoted['bh_op']} {quoted['bh']} "
                    f"but the workbook has {wb_row['bh']:.4f}"
                )
            report = "OK" if ok else "MISMATCH"
            if not ok:
                fails.append(
                    f"{panel}: manuscript {mb}/{mse}/{mz} vs report "
                    f"{wb_row['coef'] / scale:.4f}/{wb_row['se'] / scale:.4f}/{wb_row['z']:.4f}"
                )

        # 1 = computation
        if claim["csv"] is None:
            computation = "n/a"
        elif csv_row is None:
            computation = "MISSING"
            fails.append(f"{panel}: not found in statistics/")
        else:
            ok = claim["ms"] is None or (
                close(round(csv_row["coef"] / cscale, 3), mb)
                and close(round(csv_row["se"] / cscale, 3), mse)
                and close(round(csv_row["z"], 3), mz)
            )
            if quoted is not None and not p_agrees(quoted["p_op"], quoted["p"], csv_row["p"]):
                ok = False
            computation = "OK" if ok else "MISMATCH"
            if not ok:
                fails.append(
                    f"{panel}: manuscript {mb}/{mse}/{mz} vs computation "
                    f"{csv_row['coef'] / cscale:.4f}/{csv_row['se'] / cscale:.4f}/{csv_row['z']:.4f}"
                )

        # 3 = computation vs report, both converted to the manuscript's unit
        if csv_row and wb_row:
            ok = (
                close(csv_row["coef"] / cscale, wb_row["coef"] / scale, CROSS_TOL)
                and close(csv_row["se"] / cscale, wb_row["se"] / scale, CROSS_TOL)
                and close(csv_row["z"], wb_row["z"], 5e-3)
            )
            cross = "OK" if ok else "MISMATCH"
            if not ok:
                fails.append(f"{panel}: statistics/ disagrees with the workbook")
        else:
            cross = "n/a"

        if claim["ms"] is None or unclaimed is None:
            manuscript = "n/a"
        else:
            manuscript = "OK" if quoted else "MISSING"
        rows.append((panel, manuscript, computation, report, cross, claim["note"]))

    if unclaimed:
        for cand in unclaimed:
            fails.append(
                f"Fig {cand['fig']}: manuscript quotes {cand['beta']}/{cand['se']}/"
                f'{cand["z"]} but no registry entry checks it -- "...{cand["context"]}..."'
            )

    width = max(len(r[0]) for r in rows)
    print(
        f"{'claim':<{width}}  {'manuscript':>10}  {'computation':>11}  {'report':>8}  {'comp=report':>11}  unit"
    )
    print("-" * (width + 56))
    for panel, manuscript, computation, report, cross, note in rows:
        print(
            f"{panel:<{width}}  {manuscript:>10}  {computation:>11}  {report:>8}  {cross:>11}  {note}"
        )
    print(f"\n{len(rows)} claims checked.")


def check_figure_stars(index, fails):
    """Check 4: the asterisks drawn on Figures 4 and 6 against the statistics."""
    checked = 0
    before = len(fails)

    for panel, filename, key, row_key, starred in FIG4_STARS:
        table = pd.read_csv(STATS / filename)
        row = table[table[key] == row_key]
        if row.empty:
            fails.append(f"{panel}: no row '{row_key}' in {filename}")
            continue
        p = float(row["p_value"].iloc[0])
        checked += 1
        if (p < ALPHA) != starred:
            fails.append(
                f"{panel}: figure draws {'a star' if starred else 'no star'} but p={p:.4f}"
            )

    for panel, filename, key, row_key, brackets in FIG6_STARS:
        table = pd.read_csv(STATS / filename)
        sub = table[table[key] == row_key]
        if sub.empty:
            fails.append(f"{panel}: no row '{row_key}' in {filename}")
            continue
        drawn = {BRACKET[b] for b in brackets}
        for contrast in BRACKET.values():
            hit = sub[sub["contrast"] == contrast]
            if hit.empty:
                continue
            p = float(hit["p"].iloc[0])
            checked += 1
            if (p < ALPHA) != (contrast in drawn):
                fails.append(
                    f"{panel} [{contrast}]: figure draws "
                    f"{'a star' if contrast in drawn else 'no star'} but p={p:.4f}"
                )

    for panel, spec, starred in OVERALL_STARS:
        row = wb_lookup(index, spec)
        if row is None:
            fails.append(f"{panel}: not found in the workbook")
            continue
        p = float(row["p"])
        checked += 1
        if (p < ALPHA) != starred:
            fails.append(
                f"{panel}: figure draws {'a star' if starred else 'no star'} but p={p:.4f}"
            )

    if len(fails) == before:
        print(f"  OK - all {checked} markers match the statistics at alpha={ALPHA}")
    else:
        for line in fails[before:]:
            print(f"  {line}")


def main() -> int:
    ms_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SHIPPED_MS
    ms_stats = None
    if ms_path.exists():
        ms_stats = read_manuscript(ms_path)
        note = f"{len(ms_stats)} statistics read"
    else:
        note = "not found - manuscript checks skipped"

    print(f"Manuscript : {ms_path.name} ({note})")
    print(f"Report     : {WORKBOOK.relative_to(REPO)}")
    print("Computation: statistics/*.csv")
    print("Figures    : significance markers declared in this script\n")

    fails: list[str] = []
    index = index_workbook(WORKBOOK)
    check_statistics(index, ms_stats, fails)

    print("\nFigure significance markers:")
    check_figure_stars(index, fails)

    if fails:
        print(f"\nFAIL - {len(fails)} problem(s):")
        for line in fails:
            print(f"  {line}")
        return 1
    print("\nOK - manuscript, figures, computation and report all agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
