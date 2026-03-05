"""
Centralized path configuration for coping-dynamics-sequencing.
Uses environment variables to allow relocation of data without code edits.
"""
import os
from pathlib import Path


def _first_existing_dir(candidates, fallback):
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return Path(fallback)


def _first_existing_path(candidates, fallback):
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return Path(fallback)

# Project root (defaults to repo root); override with COPING_DYNAMICS_ROOT
PROJECT_ROOT = Path(os.getenv("COPING_DYNAMICS_ROOT", Path(__file__).resolve().parents[1]))

# Freezing predictions directory (default: Freezing_predictions_light); override with COPING_DYNAMICS_FREEZING_DIR
_freezing_env = os.getenv("COPING_DYNAMICS_FREEZING_DIR")
_freezing_candidates = [
    _freezing_env,
    r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\Freezing_predictions_light",
    r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\Freezing_predictions_light",
]
FREEZING_DIR = _first_existing_dir(_freezing_candidates, _freezing_candidates[1])
_freezing_root = FREEZING_DIR.parent if FREEZING_DIR.exists() else None

# Data directory (default: equipo_project_data); override with COPING_DYNAMICS_DATA
_data_env = os.getenv("COPING_DYNAMICS_DATA")
_data_candidates = [
    _data_env,
    _freezing_root,
    _freezing_root / "equipo_project_data" if _freezing_root else None,
    r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\equipo_project_data",
    r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\equipo_project_data",
    r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project",
]
DATA_DIR = _first_existing_dir(_data_candidates, _data_candidates[3])

# DLC CSV directory (raw DeepLabCut pose tracking outputs); override with COPING_DYNAMICS_DLC_DIR
_dlc_env = os.getenv("COPING_DYNAMICS_DLC_DIR")
_dlc_candidates = [
    _dlc_env,
    DATA_DIR / "CSVs_all",
    r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\equipo_project_data\CSVs_all",
    r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\2025_01_20-23_00_27\results",
]
DLC_DIR = _first_existing_dir(_dlc_candidates, DATA_DIR / "CSVs_all")

_processed_candidates = [
    DATA_DIR / "processed",
    _freezing_root / "processed" if _freezing_root else None,
    r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\processed",
]
PROCESSED_DIR = _first_existing_dir(_processed_candidates, DATA_DIR / "processed")

_metadata_candidates = [
    DATA_DIR / "metadata",
    _freezing_root / "metadata" if _freezing_root else None,
    r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\metadata",
]
METADATA_DIR = _first_existing_dir(_metadata_candidates, DATA_DIR / "metadata")

# Processed data artifacts
RESULTS_CLUSTERS_PKL = _first_existing_path(
    [
        PROCESSED_DIR / "new_results_clusters.pkl",
        DATA_DIR / "new_results_clusters.pkl",
        r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\equipo_project_data\new_results_clusters.pkl",
        r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\new_results_clusters.pkl",
        r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\processed\new_results_clusters.pkl",
    ],
    PROCESSED_DIR / "new_results_clusters.pkl",
)
RESULTS_RAW_PKL = _first_existing_path(
    [
        PROCESSED_DIR / "new_results.pkl",
        DATA_DIR / "new_results.pkl",
        r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\processed\new_results.pkl",
        r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\equipo_project_data\new_results.pkl",
        r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\new_results.pkl",
        # Use clusters pkl as fallback since it may contain the same syllable data
        RESULTS_CLUSTERS_PKL,
    ],
    PROCESSED_DIR / "new_results.pkl",
)
POSE_FEATURES_PARQUET = _first_existing_path(
    [
        PROCESSED_DIR / "pose_features_with_clusters.parquet",
        r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\processed\pose_features_with_clusters.parquet",
    ],
    PROCESSED_DIR / "pose_features_with_clusters.parquet",
)

# Metadata
INDEX_CSV = _first_existing_path(
    [
        DATA_DIR / "index.csv",
        METADATA_DIR / "index.csv",
        r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\index.csv",
    ],
    DATA_DIR / "index.csv",
)
CONFIG_YML = _first_existing_path(
    [
        METADATA_DIR / "config.yml",
        DATA_DIR / "config.yml",
        r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\metadata\config.yml",
    ],
    METADATA_DIR / "config.yml",
)

# Outputs
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Publication figure outputs live in a dedicated subfolder.
MANUSCRIPT_FIGURES_DIR = RESULTS_DIR / "manuscript_figures"
MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ML model training outputs are saved in-repo for easier tracking/versioning.
MODEL_TRAINING_DIR = RESULTS_DIR / "model_training"
MODEL_TRAINING_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = MODEL_TRAINING_DIR / "models" / "xgb_behavior"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TO_PREDICT_DIR = DATA_DIR / "to_predict"
TO_PREDICT_DIR.mkdir(parents=True, exist_ok=True)

# Prediction outputs are saved in-repo under results/ for easier review/sharing.
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

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
