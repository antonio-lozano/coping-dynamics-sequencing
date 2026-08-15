#!/usr/bin/env python
"""Validate the publication repository from a clean clone."""
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
    "CITATION.cff",
    ".gitattributes",
    ".gitignore",
    ".github/workflows/reproducibility.yml",
    "MANIFEST.csv",
    "REPRODUCIBILITY.md",
    "DATA_AVAILABILITY.md",
    "CODE_AVAILABILITY.md",
    "docs/data_dictionary.md",
    "docs/figure4_coping_provenance.md",
    "docs/behavior_classifier.md",
    "docs/workbook_match_audit.md",
    "data/raw/animal_groups.csv",
    "data/raw/behavioral_flexibility_scores.xlsx",
    "data/raw/freezing_overlap_by_group.csv",
    "data/raw/freezing_predictions_index.csv",
    "data/raw/freezing_predictions_light.csv.gz",
    "data/raw/moseq_syllables_per_frame.csv.gz",
    "data/raw/simba_validation_manual_vs_automatic.csv",
    "data/raw/syllable_classification_metrics.csv",
    "data/raw/syllable_usage_per_timebin_30s.csv",
    "data/raw/syllable_usage_per_timebin_250ms.csv",
    "data/raw/updated_results.pkl.gz",
    "data/raw/manuscript_tables/statistical_report/provenance.md",
    "data/raw/manuscript_tables/statistical_report/reported_cells.jsonl",
    "data/raw/manuscript_tables/statistical_report/reported_layout.json",
    "data/processed/cluster_frequency_per_animal.csv",
    "data/processed/cluster_timecourse_per_animal.csv",
    "data/processed/s0s28_timecourse_per_animal.csv",
    "data/processed/supplementary_figure1_tracking_clusters.csv",
    "data/processed/supplementary_figure1_time_summary.csv",
    "data/processed/supplementary_figure3_distance_scores.csv",
    "data/processed/supplementary_figure3_distance_summary.csv",
    "data/processed/supplementary_figure3_threshold_audit.csv",
    "data/processed/tracking_exclusions_per_animal.csv",
    "data/processed/transition_metrics_per_animal.csv",
    "data/processed/figure5_dynamics_scores.csv",
    "data/processed/figure5_resilience_threshold_audit.csv",
    "figure_source_data/figure2.csv",
    "figure_source_data/figure3.csv",
    "figure_source_data/figure4.csv",
    "figure_source_data/figure5.csv",
    "figure_source_data/figure6.csv",
    "statistics/stats_figure3A_GEE.csv",
    "statistics/stats_figure3_overtime.csv",
    "statistics/stats_figure4_bouts_MixedLM.csv",
    "statistics/stats_figure4_diversity_MixedLM.csv",
    "statistics/stats_figure4_transition_MixedLM.csv",
    "statistics/fig4_bout_cluster_per_animal.csv",
    "statistics/fig4_bout_overall_per_animal.csv",
    "statistics/fig4_diversity_per_animal.csv",
    "statistics/fig4_transition_per_animal.csv",
    "statistics/fig5_timecourse_mixedlm.csv",
    "statistics/fig3_frequency_gee_summary.csv",
    "statistics/fig5c_frequency_contrast_audit.csv",
    "statistics/fig5c_frequency_contrasts.csv",
    "statistics/fig5c_frequency_gee.csv",
    "statistics/fig6_bout_resilience_stats.csv",
    "statistics/fig6_diversity_resilience_stats.csv",
    "statistics/fig6_transition_resilience_stats.csv",
    "statistics/manuscript_consistency_audit.csv",
    "classifier/figure7_behavior_classifier.joblib",
    "report/raw_data.xlsx",
    "report/statistical_report.xlsx",
    "scripts/run_all.py",
    "scripts/run_all_figures.py",
    "scripts/update_manifest.py",
    "scripts/audit_manuscript_results.py",
    "scripts/build_raw_data_workbook.py",
    "scripts/build_statistical_report.py",
    "scripts/compare_workbook_data.py",
    "scripts/import_legacy_statistical_report.py",
    "scripts/derive_tables/freezing_predictions_light.py",
    "scripts/derive_tables/cluster_tables.py",
    "scripts/derive_tables/tracking_exclusions.py",
    "scripts/derive_tables/fig6_resilience_stats.py",
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

FORBIDDEN_TOP_LEVEL = {"results", "dataset", "external"}
FORBIDDEN_PATHS = {"data/" + "derived"}
TEXT_SUFFIXES = {".md", ".py", ".txt", ".json", ".yml", ".yaml", ".cff", ".toml"}
LOCAL_PATH_PATTERN = re.compile(r"(?i)\b[a-z]:\\(?:users|downloads|jen|antonio|big_computer)|/mnt/[a-z]/")
SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".uv-cache"}
FORBIDDEN_TEXT_TERMS = (
    "Cl" + "aude",
    "Co" + "dex",
    "Chat" + "GPT",
    "Open" + "AI",
    "AI" + "-generated",
    "AI" + " generated",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def path_exists_exact_case(rel: str) -> bool:
    current = ROOT
    for part in Path(rel).parts:
        if not current.is_dir():
            return False
        entries = {child.name: child for child in current.iterdir()}
        if part not in entries:
            return False
        current = entries[part]
    return True


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if SKIP_DIRS.intersection(path.parts) or not path.is_file():
            continue
        files.append(path)
    return files


def check_required(errors: list[str]) -> None:
    for rel in REQUIRED_FILES:
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


def check_layout(errors: list[str]) -> None:
    for name in FORBIDDEN_TOP_LEVEL:
        if (ROOT / name).exists():
            fail(f"forbidden top-level path exists: {name}", errors)
    for rel in FORBIDDEN_PATHS:
        if path_exists_exact_case(rel):
            fail(f"forbidden retired path exists: {rel}", errors)

    for path in iter_files():
        rel = path.relative_to(ROOT).as_posix()
        lower = rel.lower()
        if path.name.lower().startswith("readme") and rel != "README.md":
            fail(f"subfolder README found: {rel}", errors)
        if lower.startswith("data/raw/") and path.suffix.lower() in {".pdf", ".png", ".svg"}:
            fail(f"figure artifact found in raw data: {rel}", errors)
        if lower.endswith(("_test.xlsx", "_rebuilt.xlsx")) or "smoke" in lower:
            fail(f"scratch artifact found: {rel}", errors)
        if ("no" + "seed") in lower or ("no" + "-seed") in lower:
            fail(f"analysis-decision scratch name found: {rel}", errors)


def check_text_clean(errors: list[str]) -> None:
    for path in iter_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in LOCAL_PATH_PATTERN.finditer(text):
            fail(f"local absolute path in {rel}: {match.group(0)}", errors)
        for term in FORBIDDEN_TEXT_TERMS:
            if term in text:
                fail(f"tool-specific text trace in {rel}", errors)


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

    artifact_prefixes = ("data/", "figure_source_data/", "figures/", "statistics/", "report/", "classifier/")
    for rel in REQUIRED_FILES:
        if rel.startswith(artifact_prefixes) and rel not in seen:
            fail(f"required artifact absent from manifest: {rel}", errors)


def main() -> int:
    errors: list[str] = []
    check_required(errors)
    check_layout(errors)
    check_text_clean(errors)
    check_manifest(errors)

    if errors:
        print("Reproducibility checks failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Reproducibility checks passed.")
    print("Verified required inputs, processed data, source data, figures, reports, classifier artifact, and hashes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
