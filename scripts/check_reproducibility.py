#!/usr/bin/env python
"""Lightweight publication-repository reproducibility checks.

This script intentionally uses only the Python standard library. It verifies the
tracked artifact layout, required data/report/figure files, absence of common
scratch artifacts, and MANIFEST.csv checksums.
"""
from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "environment.yml",
    "REPRODUCIBILITY.md",
    "DATA_AVAILABILITY.md",
    "CODE_AVAILABILITY.md",
    "MANIFEST.csv",
    "data/README.md",
    "data/source/animal_groups.csv",
    "data/source/bfl_scores.xlsx",
    "data/source/cluster_frequency_per_animal.csv",
    "data/source/cluster_timecourse_per_animal.csv",
    "data/source/freezing_overlap_by_group.csv",
    "data/source/moseq_syllables_per_frame.csv.gz",
    "data/source/s0s28_timecourse_per_animal.csv",
    "data/source/supplementary_figure1_tracking_clusters.csv",
    "data/source/syllable_classification_metrics.csv",
    "data/source/syllable_usage_per_timebin_30s.csv",
    "data/source/syllable_usage_per_timebin_250ms.csv",
    "data/source/tracking_exclusions_per_animal.csv",
    "data/source/updated_results.pkl.gz",
    "report/RAW_DATA.xlsx",
    "report/STATISTICAL_REPORT_FINAL.xlsx",
    "report/STATISTICAL_REPORT_TEMPLATE.xlsx",
    "scripts/run_all_figures.py",
    "scripts/build_statistical_report.py",
]

FIGURE_STEMS = [
    "figure2",
    "figure3",
    "figure4",
    "figure5",
    "figure6",
    "figure7",
    "supplementary_figure1",
    "supplementary_figure3",
]
FIGURE_EXTS = (".pdf", ".svg", ".png")

INTERMEDIATE_CSVS = [
    "results/figure_data/source_data_figure2.csv",
    "results/figure_data/source_data_figure3.csv",
    "results/figure_data/source_data_figure4.csv",
    "results/figure_data/source_data_figure5.csv",
    "results/figure_data/source_data_figure6.csv",
    "results/figure_data/stats_figure3A_GEE.csv",
    "results/figure_data/stats_figure3_overtime.csv",
    "results/figure_data/stats_figure4_bouts_MixedLM.csv",
    "results/figure_data/stats_figure4_diversity_MixedLM.csv",
    "results/figure_data/stats_figure4_transition_MixedLM.csv",
    "results/statistical_reports/fig4_bout_cluster_per_animal.csv",
    "results/statistical_reports/fig4_diversity_per_animal.csv",
    "results/statistical_reports/fig4_transition_per_animal.csv",
    "results/statistical_reports/fig5_timecourse_mixedlm.csv",
    "results/statistical_reports/fig6_bout_resilience_stats.csv",
    "results/statistical_reports/fig6_diversity_resilience_stats.csv",
    "results/statistical_reports/fig6_transition_resilience_stats.csv",
    "results/statistical_reports/results_final.md",
]

FORBIDDEN_PATHS = [
    "dataset",
    "external",
    "data/Raw_data.xlsx",
    "data/source/figure7.pdf",
    "results/figure_data/Raw_data.xlsx",
    "results/figure_data/Statistical_report.xlsx",
    "results/figure_data/figure_7_source.pdf",
    "results/figure_data/figure_7_imported.pdf",
    "scripts/generate_figures/figure_7_import_pdf.py",
]

TEXT_SUFFIXES = {".md", ".py", ".txt", ".json", ".yml", ".yaml", ".cff", ".toml"}
LOCAL_PATH_PATTERN = re.compile(r"(?i)\b[a-z]:\\(?:users|downloads|jen|antonio|big_computer)|/mnt/[a-z]/")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def check_required(errors: list[str]) -> None:
    for rel in REQUIRED_FILES + INTERMEDIATE_CSVS:
        path = ROOT / rel
        if not path.is_file():
            fail(f"missing required file: {rel}", errors)
        elif path.stat().st_size == 0:
            fail(f"empty required file: {rel}", errors)

    for stem in FIGURE_STEMS:
        for ext in FIGURE_EXTS:
            rel = f"figures/{stem}{ext}"
            path = ROOT / rel
            if not path.is_file():
                fail(f"missing figure export: {rel}", errors)

    pred_dir = ROOT / "data/source/freezing_predictions"
    predictions = sorted(pred_dir.glob("*.csv")) if pred_dir.exists() else []
    if len(predictions) != 98:
        fail(f"expected 98 freezing prediction CSVs, found {len(predictions)}", errors)


def check_forbidden(errors: list[str]) -> None:
    for rel in FORBIDDEN_PATHS:
        if (ROOT / rel).exists():
            fail(f"forbidden scratch/artifact path exists: {rel}", errors)

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.startswith("~$") or name.endswith(("_test.xlsx", "_rebuilt.xlsx")) or "smoke" in name:
            fail(f"scratch artifact found: {path.relative_to(ROOT).as_posix()}", errors)


def check_local_paths(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in LOCAL_PATH_PATTERN.finditer(text):
            fail(f"local absolute path in {rel}: {match.group(0)}", errors)


def check_manifest(errors: list[str]) -> None:
    manifest = ROOT / "MANIFEST.csv"
    if not manifest.exists():
        fail("missing MANIFEST.csv", errors)
        return

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        fail("MANIFEST.csv has no artifact rows", errors)
        return

    seen = set()
    for row in rows:
        rel = row.get("path", "")
        seen.add(rel)
        path = ROOT / rel
        if not path.is_file():
            fail(f"manifest path missing: {rel}", errors)
            continue
        actual_size = path.stat().st_size
        if int(row.get("bytes", -1)) != actual_size:
            fail(f"manifest size mismatch: {rel}", errors)
        actual_sha = sha256(path)
        if row.get("sha256") != actual_sha:
            fail(f"manifest sha256 mismatch: {rel}", errors)

    for rel in REQUIRED_FILES:
        if rel.startswith(("data/", "figures/", "report/", "results/")) and rel not in seen:
            fail(f"required artifact absent from manifest: {rel}", errors)


def main() -> int:
    errors: list[str] = []
    check_required(errors)
    check_forbidden(errors)
    check_local_paths(errors)
    check_manifest(errors)

    if errors:
        print("Reproducibility checks failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Reproducibility checks passed.")
    print("Verified required data, figures, reports, intermediate CSVs, and MANIFEST.csv hashes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
