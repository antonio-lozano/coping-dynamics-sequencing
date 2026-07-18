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
    "CITATION.cff",
    ".gitattributes",
    ".gitignore",
    ".github/workflows/reproducibility.yml",
    "MANIFEST.csv",
    "docs/DATA_DICTIONARY.md",
    "docs/REPOSITORY_LAYOUT.md",
    "data/raw/animal_groups.csv",
    "data/raw/bfl_scores.xlsx",
    "data/raw/freezing_overlap_by_group.csv",
    "data/raw/freezing_predictions_index.csv",
    "data/raw/freezing_predictions_light.csv.gz",
    "data/raw/moseq_syllables_per_frame.csv.gz",
    "data/raw/syllable_classification_metrics.csv",
    "data/raw/syllable_usage_per_timebin_30s.csv",
    "data/raw/syllable_usage_per_timebin_250ms.csv",
    "data/raw/updated_results.pkl.gz",
    "data/derived/cluster_frequency_per_animal.csv",
    "data/derived/cluster_timecourse_per_animal.csv",
    "data/derived/s0s28_timecourse_per_animal.csv",
    "data/derived/supplementary_figure1_tracking_clusters.csv",
    "data/derived/tracking_exclusions_per_animal.csv",
    "report/raw_data.xlsx",
    "report/statistical_report.xlsx",
    "scripts/run_all.py",
    "scripts/run_all_figures.py",
    "scripts/derive_tables/freezing_predictions_light.py",
    "scripts/build_raw_data_workbook.py",
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
    "results/source_data/source_data_figure2.csv",
    "results/source_data/source_data_figure3.csv",
    "results/source_data/source_data_figure4.csv",
    "results/source_data/source_data_figure5.csv",
    "results/source_data/source_data_figure6.csv",
    "results/statistics/stats_figure3A_GEE.csv",
    "results/statistics/stats_figure3_overtime.csv",
    "results/statistics/stats_figure4_bouts_MixedLM.csv",
    "results/statistics/stats_figure4_diversity_MixedLM.csv",
    "results/statistics/stats_figure4_transition_MixedLM.csv",
    "results/statistics/fig4_bout_cluster_per_animal.csv",
    "results/statistics/fig4_diversity_per_animal.csv",
    "results/statistics/fig4_transition_per_animal.csv",
    "results/statistics/fig5_timecourse_mixedlm.csv",
    "results/statistics/fig6_bout_resilience_stats.csv",
    "results/statistics/fig6_diversity_resilience_stats.csv",
    "results/statistics/fig6_transition_resilience_stats.csv",
    "results/statistics/manuscript_results_text.md",
    "results/intermediate/tables/figure_5_dynamics_scores.csv",
    "results/intermediate/tables/supplementary_figure1_time_summary.csv",
    "results/models/behavior_classifier/fig7_behavior_xgb.joblib",
]

FORBIDDEN_PATHS = [
    "dataset",
    "external",
    "data/source",
    "results/figure_data",
    "figures/README.txt",
    "report/README.md",
    "results/behavior_classifier",
    "results/statistical_reports",
    "results/README.md",
    "results/intermediate/README.md",
    "results/models/README.md",
    "results/source_data/README.md",
    "results/statistics/README.md",
    "scripts/README.md",
    "scripts/derive_tables/README.md",
    "scripts/generate_figures/README.md",
    "data/Raw_data.xlsx",
    "report/RAW_DATA.xlsx",
    "report/STATISTICAL_REPORT.xlsx",
    "report/STATISTICAL_REPORT_TEMPLATE.xlsx",
    "data/raw/figure7.pdf",
    "results/figure_data/Raw_data.xlsx",
    "results/figure_data/Statistical_report.xlsx",
    "results/figure_data/figure_7_source.pdf",
    "results/figure_data/figure_7_imported.pdf",
    "scripts/generate_figures/figure_7_import_pdf.py",
]

TEXT_SUFFIXES = {".md", ".py", ".txt", ".json", ".yml", ".yaml", ".cff", ".toml"}
LOCAL_PATH_PATTERN = re.compile(r"(?i)\b[a-z]:\\(?:users|downloads|jen|antonio|big_computer)|/mnt/[a-z]/")
SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def path_exists_exact_case(rel: str) -> bool:
    """Return True only when every path component exists with matching case."""
    current = ROOT
    for part in Path(rel).parts:
        if not current.is_dir():
            return False
        entries = {child.name: child for child in current.iterdir()}
        if part not in entries:
            return False
        current = entries[part]
    return True


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

    pred_dir = ROOT / "data/raw/freezing_predictions"
    predictions = sorted(pred_dir.glob("*.csv")) if pred_dir.exists() else []
    if len(predictions) != 98:
        fail(f"expected 98 freezing prediction CSVs, found {len(predictions)}", errors)


def check_forbidden(errors: list[str]) -> None:
    for rel in FORBIDDEN_PATHS:
        if path_exists_exact_case(rel):
            fail(f"forbidden scratch/artifact path exists: {rel}", errors)

    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_DIRS.intersection(path.parts):
            continue
        name = path.name.lower()
        if name.startswith("~$") or name.endswith(("_test.xlsx", "_rebuilt.xlsx")) or "smoke" in name:
            fail(f"scratch artifact found: {path.relative_to(ROOT).as_posix()}", errors)


def check_local_paths(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if SKIP_DIRS.intersection(path.parts) or not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
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
