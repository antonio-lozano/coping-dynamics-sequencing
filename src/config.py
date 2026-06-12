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
    PROJECT_ROOT / "data",
    r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\equipo_project_data",
    r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\equipo_project_data",
    r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project",
]
# Fall back to the repo's own data/ directory so importing config never tries to
# create directories under an absent external (e.g. another machine's) path.
DATA_DIR = _first_existing_dir(_data_candidates, PROJECT_ROOT / "data")

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

# ==============================================================================
# Manuscript figure source data
# ==============================================================================
# Extracted source files live under data/source/ (preferred). The original
# archives (Coping_data.zip / Coping_data2.zip) are searched in the user's
# Downloads folder as a fallback. Override any of these with the matching env
# var (COPING_DYNAMICS_SOURCE_DIR, COPING_DYNAMICS_DOWNLOADS, COPING_DATA_ZIP,
# COPING_DATA2_ZIP, COPING_DYNAMICS_CLUSTER_JSON).
SOURCE_DATA_DIR = Path(os.getenv("COPING_DYNAMICS_SOURCE_DIR") or (DATA_DIR / "source"))

_downloads_dir = Path(os.getenv("COPING_DYNAMICS_DOWNLOADS") or (Path.home() / "Downloads"))

COPING_DATA_ZIP = _first_existing_path(
    [os.getenv("COPING_DATA_ZIP"), _downloads_dir / "Coping_data.zip"],
    _downloads_dir / "Coping_data.zip",
)
COPING_DATA2_ZIP = _first_existing_path(
    [os.getenv("COPING_DATA2_ZIP"), _downloads_dir / "Coping_data2.zip"],
    _downloads_dir / "Coping_data2.zip",
)

# Extracted source files (may be absent; scripts fall back to the archives).
SYLLABLE_TIMEBIN_30S = SOURCE_DATA_DIR / "syllable_usage_per_timebin_30s.csv"
SYLLABLE_TIMEBIN_250MS = SOURCE_DATA_DIR / "syllable_usage_per_timebin_250ms.csv"
BFL_SCORES_XLSX = SOURCE_DATA_DIR / "bfl_scores.xlsx"
UPDATED_RESULTS_PKL = SOURCE_DATA_DIR / "updated_results.pkl.gz"

# Optional hand-curated syllable->cluster JSON; scripts fall back to the
# built-in CLUSTER_MAP in each figure script when this is absent.
CLUSTER_JSON = _first_existing_path(
    [os.getenv("COPING_DYNAMICS_CLUSTER_JSON"), SOURCE_DATA_DIR / "Behavioral_clusters.json"],
    SOURCE_DATA_DIR / "Behavioral_clusters.json",
)

# Archive member paths (used when reading directly from the zips).
COPING2_MEMBER_TIMEBIN_30S = "COping/Syllable_per_timebin_final(30s).csv"
COPING2_MEMBER_TIMEBIN_250MS = "COping/Syllable_per_timebin_final(250ms).csv"
COPING2_MEMBER_UPDATED_RESULTS = "COping/updated_results.pkl"
COPING_MEMBER_BFL_SCORES = "CSVs/BFL_scores.xlsx"

# Supporting figure tables / audits / descriptive renders (kept out of figures/).
FIGURE_DATA_DIR = RESULTS_DIR / "figure_data"

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
