#!/usr/bin/env python
"""Compare manuscript Results claims against regenerated statistical outputs.

The audit encodes the Results values that must be checked for the current
manuscript and reads the corresponding regenerated CSV rows from `statistics/`.
It can also read a manuscript `.docx` to confirm that the file is accessible
during local checks, but the repository output never records local paths.

Run:
    python scripts/audit_manuscript_results.py
    python scripts/audit_manuscript_results.py --manuscript path/to/manuscript.docx
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
STATISTICS_DIR = ROOT / "statistics"
OUT = STATISTICS_DIR / "manuscript_consistency_audit.csv"


@dataclass(frozen=True)
class Claim:
    section: str
    claim: str
    repo_source: str
    filters: tuple[tuple[str, str], ...]
    manuscript_beta: str = ""
    manuscript_se: str = ""
    manuscript_z: str = ""
    manuscript_p: str = ""
    coef_column: str = "coef"
    se_column: str = "se"
    z_column: str = "z"
    p_column: str = "p_value"
    scale: float = 1.0
    note: str = ""
    markers: tuple[str, ...] = ()


CLAIMS = [
    Claim(
        section="Figure 1",
        claim="Classifier validation r=0.95, R2=0.90, p<0.001",
        repo_source="data/raw/simba_validation_manual_vs_automatic.csv",
        filters=(),
        manuscript_beta="0.95",
        manuscript_p="<0.001",
        markers=("R=0.95",),
    ),
    Claim(
        "Figure 3A",
        "Freeze ELS vs Control frequency",
        "statistics/stats_figure3A_GEE.csv",
        (("cluster", "Freeze"), ("parameter", "C(group, Treatment('Control'))[T.ELS]")),
        "-0.204",
        "0.081",
        "-2.510",
        "0.012",
        coef_column="beta",
        markers=("Freeze", "-0.204", "0.012"),
    ),
    Claim(
        "Figure 3A",
        "Sniff ELS vs Control frequency",
        "statistics/stats_figure3A_GEE.csv",
        (("cluster", "Sniff"), ("parameter", "C(group, Treatment('Control'))[T.ELS]")),
        "0.621",
        "0.231",
        "2.694",
        "0.007",
        coef_column="beta",
        markers=("Sniff", "0.621", "0.007"),
    ),
    Claim(
        "Figure 3A",
        "Turn ELS vs Control frequency",
        "statistics/stats_figure3A_GEE.csv",
        (("cluster", "Turn"), ("parameter", "C(group, Treatment('Control'))[T.ELS]")),
        "0.101",
        "0.032",
        "3.099",
        "0.002",
        coef_column="beta",
        markers=("Turn", "0.101", "0.002"),
    ),
    Claim(
        "Figure 4E",
        "Simpson diversity ELS vs Control",
        "statistics/stats_figure4_diversity_MixedLM.csv",
        (("metric", "simpson"), ("parameter", "Condition[T.ELS]")),
        "-0.013",
        "0.007",
        "-1.915",
        "0.056",
        note="Repo remains near-threshold and non-significant.",
        markers=("Simpson", "-0.013", "0.056"),
    ),
    Claim(
        "Figure 4H",
        "Cumulative usage index ELS vs Control",
        "statistics/stats_figure4_diversity_MixedLM.csv",
        (("metric", "cui"), ("parameter", "Condition[T.ELS]")),
        "0.064",
        "0.028",
        "2.250",
        "0.025",
        markers=("cumulative", "0.064", "0.025"),
    ),
    Claim(
        "Figure 4K",
        "Freezing bout duration ELS vs Control",
        "statistics/stats_figure4_bouts_MixedLM.csv",
        (("cluster", "Freezing"), ("parameter", "Condition[T.ELS]")),
        "-0.094",
        "0.033",
        "-2.824",
        "0.005",
        note="Repo remains significant but less strong than the manuscript value.",
        markers=("Freezing", "-0.094", "0.005"),
    ),
    Claim(
        "Figure 4L",
        "Sniffing bout duration ELS vs Control",
        "statistics/stats_figure4_bouts_MixedLM.csv",
        (("cluster", "Sniffing"), ("parameter", "Condition[T.ELS]")),
        "0.321",
        "0.126",
        "2.540",
        "0.011",
        markers=("Sniffing", "0.321", "0.011"),
    ),
    Claim(
        "Figure 4N",
        "Turn bout duration ELS vs Control",
        "statistics/stats_figure4_bouts_MixedLM.csv",
        (("cluster", "Turn"), ("parameter", "Condition[T.ELS]")),
        "0.164",
        "0.058",
        "2.824",
        "0.005",
        markers=("Turn", "0.164", "0.005"),
    ),
    Claim(
        "Figure 4U",
        "Recurrence ELS vs Control",
        "statistics/stats_figure4_transition_MixedLM.csv",
        (("metric", "recurrence"), ("parameter", "Condition[T.ELS]")),
        "0.013",
        "0.007",
        "1.908",
        "0.056",
        note="Repo remains near-threshold and non-significant.",
        markers=("Recurrence", "0.013", "0.056"),
    ),
    Claim(
        "Figure 4V",
        "Determinism ELS vs Control",
        "statistics/stats_figure4_transition_MixedLM.csv",
        (("metric", "determinism"), ("parameter", "Condition[T.ELS]")),
        "0.016",
        "0.007",
        "2.167",
        "0.030",
        markers=("Determinism", "0.016", "0.030"),
    ),
    Claim(
        "Figure 4W",
        "Markov entropy ELS vs Control",
        "statistics/stats_figure4_transition_MixedLM.csv",
        (("metric", "markov"), ("parameter", "Condition[T.ELS]")),
        "-0.041",
        "0.025",
        "-1.638",
        "0.102",
        note="Repo remains non-significant.",
        markers=("Markov",),
    ),
    Claim(
        "Figure 5C",
        "Freeze resilient vs vulnerable ELS frequency",
        "statistics/fig5c_frequency_gee.csv",
        (("cluster", "Freeze"), ("contrast", "ELS/Resilient")),
        "0.409",
        "0.118",
        "3.476",
        "<0.001",
        coef_column="beta",
        se_column="SE",
        p_column="p",
        markers=("Freeze", "0.409", "0.118"),
    ),
    Claim(
        "Figure 5C",
        "Turn resilient vs vulnerable ELS frequency",
        "statistics/fig5c_frequency_gee.csv",
        (("cluster", "Turn"), ("contrast", "ELS/Resilient")),
        "-0.145",
        "0.033",
        "-4.437",
        "<0.001",
        coef_column="beta",
        se_column="SE",
        p_column="p",
        markers=("Turn", "-0.145", "-4.437"),
    ),
    Claim(
        "Figure 5D",
        "Freeze time resilient vs vulnerable ELS",
        "statistics/fig5_timecourse_mixedlm.csv",
        (("cluster", "Freeze"), ("parameter", "C(group_ext, Treatment('ELS'))[T.ELS resilient]:time_bin_numeric")),
        "1.865",
        "0.249",
        "7.484",
        "<0.001",
        scale=30.0,
        p_column="p",
        note="Repo coefficient and standard error are scaled per 30 s bin for the audit.",
        markers=("Freeze", "1.865", "7.484"),
    ),
    Claim(
        "Figure 6G",
        "Simpson vulnerable ELS vs Control",
        "statistics/fig6_diversity_resilience_stats.csv",
        (("metric", "Simpson"), ("contrast", "vuln_vs_control")),
        "-0.022",
        "0.008",
        "-2.835",
        "0.005",
        coef_column="beta",
        se_column="SE",
        p_column="p",
        markers=("Simpson", "-0.022", "0.005"),
    ),
    Claim(
        "Figure 6M",
        "Freezing bout resilient vs vulnerable ELS",
        "statistics/fig6_bout_resilience_stats.csv",
        (("cluster", "Freezing"), ("contrast", "resilient_vs_vuln")),
        "0.263",
        "0.089",
        "2.969",
        "0.003",
        coef_column="beta",
        se_column="SE",
        p_column="p",
        markers=("Freezing", "0.263", "0.003"),
    ),
    Claim(
        "Figure 6W",
        "Lempel-Ziv vulnerable ELS vs Control",
        "statistics/fig6_transition_resilience_stats.csv",
        (("metric", "LZ_complexity"), ("contrast", "vuln_vs_control")),
        "-8.965",
        "2.658",
        "-3.37",
        "<0.001",
        coef_column="beta",
        se_column="SE",
        p_column="p",
        note="Repo p-value is significant at 0.05 but does not support p<0.001.",
        markers=("Lempel", "-8.965", "-3.37"),
    ),
    Claim(
        "Figure 6X",
        "Recurrence vulnerable ELS vs Control",
        "statistics/fig6_transition_resilience_stats.csv",
        (("metric", "Recurrence"), ("contrast", "vuln_vs_control")),
        "0.026",
        "0.008",
        "2.798",
        "0.005",
        coef_column="beta",
        se_column="SE",
        p_column="p",
        markers=("Recurrence", "0.026", "0.005"),
    ),
    Claim(
        "Figure 6X",
        "Recurrence resilient vs vulnerable ELS",
        "statistics/fig6_transition_resilience_stats.csv",
        (("metric", "Recurrence"), ("contrast", "resilient_vs_vuln")),
        "-0.029",
        "0.011",
        "-2.727",
        "0.006",
        coef_column="beta",
        se_column="SE",
        p_column="p",
        markers=("Recurrence", "-0.029", "0.006"),
    ),
    Claim(
        "Figure 6Y",
        "Determinism vulnerable ELS vs Control",
        "statistics/fig6_transition_resilience_stats.csv",
        (("metric", "Determinism"), ("contrast", "vuln_vs_control")),
        "0.010",
        "0.003",
        "2.860",
        "0.004",
        coef_column="beta",
        se_column="SE",
        p_column="p",
        markers=("Determinism", "0.010", "0.004"),
    ),
    Claim(
        "Figure 6Z",
        "Markov entropy vulnerable ELS vs Control",
        "statistics/fig6_transition_resilience_stats.csv",
        (("metric", "MarkovEntropyIdx"), ("contrast", "vuln_vs_control")),
        "-0.044",
        "0.022",
        "-1.99",
        "0.046",
        coef_column="beta",
        se_column="SE",
        p_column="p",
        markers=("Markov", "-0.044", "0.046"),
    ),
]


def load_csv(rel: str) -> list[dict[str, str]]:
    with (ROOT / rel).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def find_row(rel: str, filters: tuple[tuple[str, str], ...]) -> dict[str, str]:
    rows = load_csv(rel)
    for row in rows:
        if all(str(row.get(key, "")) == value for key, value in filters):
            return row
    terms = ", ".join(f"{key}={value}" for key, value in filters)
    raise KeyError(f"Could not find row in {rel}: {terms}")


def as_float(value: str) -> float:
    return float(value) if value not in ("", "nan", "NaN") else math.nan


def stated_float(value: str) -> float:
    if not value:
        return math.nan
    return float(value.lstrip("<"))


def decimals(value: str) -> int:
    token = value.lstrip("<")
    return len(token.split(".", 1)[1]) if "." in token else 0


def rounded_match(observed: float, stated: str) -> bool:
    if not stated:
        return True
    places = decimals(stated)
    tolerance = 0.5 * 10 ** (-places)
    return abs(observed - stated_float(stated)) <= tolerance + 1e-12


def p_match(observed: float, stated: str) -> bool:
    if not stated:
        return True
    if stated.startswith("<"):
        return observed < stated_float(stated)
    return rounded_match(observed, stated)


def conclusion_from_p(value: float | str) -> str:
    if isinstance(value, str):
        if value.startswith("<"):
            return "significant" if stated_float(value) <= 0.05 else "not_significant"
        value = stated_float(value)
    if value < 0.05:
        return "significant"
    if value < 0.1:
        return "trend_or_near_threshold"
    return "not_significant"


def format_number(value: float) -> str:
    if math.isnan(value):
        return ""
    if value != 0 and abs(value) < 0.0001:
        return f"{value:.3g}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def classify(claim: Claim, beta: float, se: float, z: float, p: float) -> str:
    numeric_values = (
        rounded_match(beta, claim.manuscript_beta),
        rounded_match(se, claim.manuscript_se),
        rounded_match(z, claim.manuscript_z),
    )
    same_conclusion = conclusion_from_p(p) == conclusion_from_p(claim.manuscript_p)

    if all(numeric_values) and p_match(p, claim.manuscript_p):
        return "match"
    if not same_conclusion:
        return "major_mismatch"
    if claim.manuscript_p.startswith("<") and not p_match(p, claim.manuscript_p):
        return "major_mismatch"
    if conclusion_from_p(p) == "trend_or_near_threshold":
        return "minor_mismatch_same_interpretation"
    if any(not flag for flag in numeric_values):
        return "numeric_mismatch_same_conclusion"
    return "rounding_difference"


def audit_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for claim in CLAIMS:
        if claim.section == "Figure 1":
            validation = pd.read_csv(ROOT / claim.repo_source)
            fit = stats.linregress(validation["manual_percent"], validation["automatic_percent"])
            repo_r = float(fit.rvalue)
            repo_p = float(fit.pvalue)
            rows.append(
                {
                    "section": claim.section,
                    "claim": claim.claim,
                    "repo_source": claim.repo_source,
                    "manuscript_beta": claim.manuscript_beta,
                    "manuscript_se": claim.manuscript_se,
                    "manuscript_z": claim.manuscript_z,
                    "manuscript_p": claim.manuscript_p,
                    "repo_beta": format_number(repo_r),
                    "repo_se": "",
                    "repo_z": "",
                    "repo_p": format_number(repo_p),
                    "status": classify(claim, repo_r, math.nan, math.nan, repo_p),
                    "note": "repo_beta stores Pearson r for this validation row.",
                }
            )
            continue

        if claim.repo_source == "not present":
            rows.append(
                {
                    "section": claim.section,
                    "claim": claim.claim,
                    "repo_source": claim.repo_source,
                    "manuscript_beta": claim.manuscript_beta,
                    "manuscript_se": claim.manuscript_se,
                    "manuscript_z": claim.manuscript_z,
                    "manuscript_p": claim.manuscript_p,
                    "repo_beta": "",
                    "repo_se": "",
                    "repo_z": "",
                    "repo_p": "",
                    "status": "not_reproducible_from_repo",
                    "note": claim.note,
                }
            )
            continue

        row = find_row(claim.repo_source, claim.filters)
        beta = as_float(row[claim.coef_column]) * claim.scale
        se = as_float(row[claim.se_column]) * claim.scale
        z = as_float(row[claim.z_column])
        p = as_float(row[claim.p_column])
        rows.append(
            {
                "section": claim.section,
                "claim": claim.claim,
                "repo_source": claim.repo_source,
                "manuscript_beta": claim.manuscript_beta,
                "manuscript_se": claim.manuscript_se,
                "manuscript_z": claim.manuscript_z,
                "manuscript_p": claim.manuscript_p,
                "repo_beta": format_number(beta),
                "repo_se": format_number(se),
                "repo_z": format_number(z),
                "repo_p": format_number(p),
                "status": classify(claim, beta, se, z, p),
                "note": claim.note,
            }
        )
    return rows


def extract_docx_text(path: Path) -> str:
    namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespaces):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespaces))
        if text.strip():
            paragraphs.append(text)
    return "\n".join(paragraphs)


def check_manuscript_text(path: Path) -> list[str]:
    text = extract_docx_text(path)
    normalized = normalize_marker_text(text)
    missing = []
    for claim in CLAIMS:
        markers = [normalize_marker_text(marker) for marker in claim.markers if marker]
        if markers and not all(marker in normalized for marker in markers):
            missing.append(f"{claim.section}: {claim.claim}")
    return missing


def normalize_marker_text(value: str) -> str:
    value = value.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    value = value.replace("\u00a0", " ")
    value = value.replace("R\u00b2", "R2").replace("r\u00b2", "r2")
    value = re.sub(r"\s+", "", value)
    return value.lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manuscript",
        type=Path,
        help="Optional .docx file used to confirm that audited claim markers are present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = audit_rows()

    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "section",
                "claim",
                "repo_source",
                "manuscript_beta",
                "manuscript_se",
                "manuscript_z",
                "manuscript_p",
                "repo_beta",
                "repo_se",
                "repo_z",
                "repo_p",
                "status",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT.relative_to(ROOT)} with {len(rows)} audited claims")

    if args.manuscript:
        if not args.manuscript.is_file():
            print("Manuscript file not found.", file=sys.stderr)
            return 1
        missing = check_manuscript_text(args.manuscript)
        if missing:
            print("Manuscript marker check failed:", file=sys.stderr)
            for item in missing:
                print(f"  - {item}", file=sys.stderr)
            return 1
        print("Manuscript marker check passed.")

    statuses = {row["status"] for row in rows}
    if "major_mismatch" in statuses:
        print("Major manuscript/statistics mismatch present; see audit CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
