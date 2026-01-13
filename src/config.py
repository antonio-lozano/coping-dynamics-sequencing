"""
Centralized path configuration for coping-dynamics-sequencing.
Uses environment variables to allow relocation of data without code edits.
"""
import os
from pathlib import Path

# Project root (defaults to repo root); override with COPING_DYNAMICS_ROOT
PROJECT_ROOT = Path(os.getenv("COPING_DYNAMICS_ROOT", Path(__file__).resolve().parents[1]))

# Data directory (default: <root>/data); override with COPING_DYNAMICS_DATA
DATA_DIR = Path(os.getenv("COPING_DYNAMICS_DATA", PROJECT_ROOT / "data"))
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
FREEZING_DIR = DATA_DIR / "freezing_predictions"

# Processed data artifacts
RESULTS_CLUSTERS_PKL = PROCESSED_DIR / "new_results_clusters.pkl"
RESULTS_RAW_PKL = PROCESSED_DIR / "new_results.pkl"
POSE_FEATURES_PARQUET = PROCESSED_DIR / "pose_features_with_clusters.parquet"

# Metadata
INDEX_CSV = METADATA_DIR / "index.csv"
CONFIG_YML = METADATA_DIR / "config.yml"

# Outputs
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
