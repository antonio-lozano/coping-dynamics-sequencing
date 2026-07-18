# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gómez and Antonio Lozano
"""Centralized path configuration for coping-dynamics-sequencing.

The repository separates immutable inputs from generated artifacts:

- data/raw: tracked inputs needed to rebuild the analyses
- data/derived: analysis-ready tables generated from data/raw
- results/source_data: figure source-data CSVs
- results/statistics: statistical model outputs
- results/intermediate: helper tables and non-canonical renders
"""
import os
from pathlib import Path


def _first_existing(candidates, fallback):
    """Return first existing path from candidates, or fallback if none exist."""
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return Path(fallback)

# Project root (defaults to repo root); override with COPING_DYNAMICS_ROOT
PROJECT_ROOT = Path(os.getenv("COPING_DYNAMICS_ROOT", Path(__file__).resolve().parents[1]))

# Data directory; override with COPING_DYNAMICS_DATA
DATA_DIR = Path(os.getenv("COPING_DYNAMICS_DATA", PROJECT_ROOT / "data"))
RAW_DATA_DIR = Path(os.getenv("COPING_DYNAMICS_RAW_DIR", DATA_DIR / "raw"))
DERIVED_DATA_DIR = Path(os.getenv("COPING_DYNAMICS_DERIVED_DIR", DATA_DIR / "derived"))

# Freezing predictions directory; override with COPING_DYNAMICS_FREEZING_DIR
# (Bundled in data/raw/freezing_predictions/ for self-contained operation)
FREEZING_DIR = Path(os.getenv("COPING_DYNAMICS_FREEZING_DIR", RAW_DATA_DIR / "freezing_predictions"))

# Optional external directories (for reproducibility extensions, not required)
DLC_DIR = Path(os.getenv("COPING_DYNAMICS_DLC_DIR", DATA_DIR / "CSVs_all"))
PROCESSED_DIR = Path(os.getenv("COPING_DYNAMICS_PROCESSED_DIR", DATA_DIR / "processed"))
METADATA_DIR = Path(os.getenv("COPING_DYNAMICS_METADATA_DIR", DATA_DIR / "metadata"))

# Processed data artifacts (optional)
RESULTS_CLUSTERS_PKL = Path(os.getenv("COPING_DYNAMICS_RESULTS_CLUSTERS_PKL", PROCESSED_DIR / "new_results_clusters.pkl"))
RESULTS_RAW_PKL = Path(os.getenv("COPING_DYNAMICS_RESULTS_RAW_PKL", PROCESSED_DIR / "new_results.pkl"))
POSE_FEATURES_PARQUET = Path(os.getenv("COPING_DYNAMICS_POSE_FEATURES_PARQUET", PROCESSED_DIR / "pose_features_with_clusters.parquet"))

# Metadata
INDEX_CSV = Path(os.getenv("COPING_DYNAMICS_INDEX_CSV", DATA_DIR / "index.csv"))
CONFIG_YML = Path(os.getenv("COPING_DYNAMICS_CONFIG_YML", METADATA_DIR / "config.yml"))

# Outputs
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Manuscript figure source data
# ==============================================================================
# Extracted source files live under data/raw/ (all figures are self-contained
# and do not require external data). Original archives are searched in Downloads
# as a fallback only. Override with:
#   COPING_DYNAMICS_RAW_DIR     (tracked raw input directory)
#   COPING_DYNAMICS_SOURCE_DIR  (legacy alias for tracked raw input directory)
#   COPING_DYNAMICS_DOWNLOADS   (downloads directory for archives)
#   COPING_DATA_ZIP, COPING_DATA2_ZIP  (specific archive locations)
#   COPING_DYNAMICS_CLUSTER_JSON  (optional hand-curated cluster map)
SOURCE_DATA_DIR = Path(os.getenv("COPING_DYNAMICS_SOURCE_DIR", RAW_DATA_DIR))

_downloads_dir = Path(os.getenv("COPING_DYNAMICS_DOWNLOADS", Path.home() / "Downloads"))

COPING_DATA_ZIP = _first_existing(
    [os.getenv("COPING_DATA_ZIP"), _downloads_dir / "Coping_data.zip"],
    _downloads_dir / "Coping_data.zip",
)
COPING_DATA2_ZIP = _first_existing(
    [os.getenv("COPING_DATA2_ZIP"), _downloads_dir / "Coping_data2.zip"],
    _downloads_dir / "Coping_data2.zip",
)

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

# Optional hand-curated syllable→cluster JSON; scripts fall back to built-in maps.
CLUSTER_JSON = _first_existing(
    [os.getenv("COPING_DYNAMICS_CLUSTER_JSON"), SOURCE_DATA_DIR / "Behavioral_clusters.json"],
    SOURCE_DATA_DIR / "Behavioral_clusters.json",
)

# Archive member paths (used when reading directly from the zips).
COPING2_MEMBER_TIMEBIN_30S = "COping/Syllable_per_timebin_final(30s).csv"
COPING2_MEMBER_TIMEBIN_250MS = "COping/Syllable_per_timebin_final(250ms).csv"
COPING2_MEMBER_UPDATED_RESULTS = "COping/updated_results.pkl"
COPING_MEMBER_BFL_SCORES = "CSVs/BFL_scores.xlsx"

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
