# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""Centralized path configuration for coping-dynamics-sequencing.

The repository separates immutable inputs from generated artifacts:

- data/raw: tracked inputs needed to rebuild the analyses
- data/derived: analysis-ready tables generated from data/raw
- results/source_data: figure source-data CSVs
- results/statistics: statistical model outputs
- results/intermediate: helper tables and non-canonical renders
"""
from pathlib import Path
import os


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default

# Project root defaults to this repository root.
PROJECT_ROOT = _env_path("COPING_DYNAMICS_ROOT", Path(__file__).resolve().parents[1])

DATA_DIR = _env_path("COPING_DYNAMICS_DATA", PROJECT_ROOT / "data")
RAW_DATA_DIR = _env_path("COPING_DYNAMICS_RAW_DIR", DATA_DIR / "raw")
DERIVED_DATA_DIR = _env_path("COPING_DYNAMICS_DERIVED_DIR", DATA_DIR / "derived")

FREEZING_DIR = _env_path("COPING_DYNAMICS_FREEZING_DIR", RAW_DATA_DIR / "freezing_predictions")

# Optional development directories (not required for repository reproduction)
DLC_DIR = _env_path("COPING_DYNAMICS_DLC_DIR", DATA_DIR / "CSVs_all")
PROCESSED_DIR = _env_path("COPING_DYNAMICS_PROCESSED_DIR", DATA_DIR / "processed")
METADATA_DIR = _env_path("COPING_DYNAMICS_METADATA_DIR", DATA_DIR / "metadata")

# Processed data artifacts (optional)
RESULTS_CLUSTERS_PKL = _env_path("COPING_DYNAMICS_RESULTS_CLUSTERS_PKL", PROCESSED_DIR / "new_results_clusters.pkl")
RESULTS_RAW_PKL = _env_path("COPING_DYNAMICS_RESULTS_RAW_PKL", PROCESSED_DIR / "new_results.pkl")
POSE_FEATURES_PARQUET = _env_path("COPING_DYNAMICS_POSE_FEATURES_PARQUET", PROCESSED_DIR / "pose_features_with_clusters.parquet")

# Metadata
INDEX_CSV = _env_path("COPING_DYNAMICS_INDEX_CSV", DATA_DIR / "index.csv")
CONFIG_YML = _env_path("COPING_DYNAMICS_CONFIG_YML", METADATA_DIR / "config.yml")

# Outputs
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Manuscript figure source data
# ==============================================================================
# Source files live under data/raw/ for self-contained operation.
SOURCE_DATA_DIR = _env_path("COPING_DYNAMICS_SOURCE_DIR", RAW_DATA_DIR)

# Raw input files (all bundled in data/raw/ for reproducibility)
SYLLABLE_TIMEBIN_30S = SOURCE_DATA_DIR / "syllable_usage_per_timebin_30s.csv"
SYLLABLE_TIMEBIN_250MS = SOURCE_DATA_DIR / "syllable_usage_per_timebin_250ms.csv"
BFL_SCORES_XLSX = SOURCE_DATA_DIR / "bfl_scores.xlsx"
UPDATED_RESULTS_PKL = SOURCE_DATA_DIR / "updated_results.pkl.gz"
MOSEQ_DF = SOURCE_DATA_DIR / "moseq_syllables_per_frame.csv.gz"  # Per-frame syllable table

# Generated analysis-ready data derived from data/raw/.
CLUSTER_FREQUENCY_CSV = DERIVED_DATA_DIR / "cluster_frequency_per_animal.csv"
CLUSTER_TIMECOURSE_CSV = DERIVED_DATA_DIR / "cluster_timecourse_per_animal.csv"
S0S28_TIMECOURSE_CSV = DERIVED_DATA_DIR / "s0s28_timecourse_per_animal.csv"
TRACKING_EXCLUSIONS_CSV = DERIVED_DATA_DIR / "tracking_exclusions_per_animal.csv"
SUPPLEMENTARY_TRACKING_CSV = DERIVED_DATA_DIR / "supplementary_figure1_tracking_clusters.csv"

# Optional hand-curated syllable-to-cluster JSON; scripts fall back to built-in maps.
CLUSTER_JSON = _env_path("COPING_DYNAMICS_CLUSTER_JSON", SOURCE_DATA_DIR / "Behavioral_clusters.json")

# Supporting outputs (kept out of figures/).
RESULTS_SOURCE_DATA_DIR = RESULTS_DIR / "source_data"
RESULTS_STATISTICS_DIR = RESULTS_DIR / "statistics"
RESULTS_INTERMEDIATE_DIR = RESULTS_DIR / "intermediate"
RESULTS_INTERMEDIATE_FIGURES_DIR = RESULTS_INTERMEDIATE_DIR / "figures"
RESULTS_INTERMEDIATE_TABLES_DIR = RESULTS_INTERMEDIATE_DIR / "tables"
RESULTS_MODELS_DIR = RESULTS_DIR / "models"

# Backward-compatible alias for older helper modules.
FIGURE_DATA_DIR = RESULTS_SOURCE_DATA_DIR

# Constants
FPS = 25
BIN_SECONDS = 30
BEHAVIOR_MAPPING = {
    1: "Freezing",
    2: "Sniffing",
    3: "Grooming",
    4: "Turn",
    5: "Locomotion",
    6: "Climbing",
    7: "Jump",
}
CODES = list(BEHAVIOR_MAPPING.keys())

# Color palette for consistent visualization across all figures
PALETTE = {
    "Control": "#F8C650",      # Gold/amber
    "ELS": "#C37B9F",          # Mauve/pink
    "ELS resilient": "#6B9F78", # Sage green
    "ELS vulnerable": "#C37B9F", # Same as ELS (mauve/pink)
}
# Ordered list for plotting
GROUP_ORDER = ["Control", "ELS"]
GROUP_ORDER_WITH_RESILIENT = ["Control", "ELS", "ELS resilient"]
GROUP_ORDER_SUBGROUPS = ["Control", "ELS vulnerable", "ELS resilient"]
