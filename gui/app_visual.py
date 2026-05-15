#!/usr/bin/env python
"""Minimal PyQtGraph comparison viewer for videos, bodyparts, and features."""
from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocessing.map_aya_to_jen import JEN_BODY_ORDER, _resolve_default_input_dir, first_scorer, load_tracking_df
from src.ml.pose_features import DEFAULT_IMPORTANT_FEATURES, compute_kinematic_features

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg

    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
    QT_AVAILABLE = True
    QT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised via runtime dependency checks
    QtCore = QtGui = QtWidgets = pg = None
    QT_AVAILABLE = False
    QT_IMPORT_ERROR = exc


FALLBACK_FPS = 25.0
SLIDER_SCALE = 1000
VIDEO_FILTER = "Video Files (*.avi *.mp4 *.mov);;All Files (*)"
POSE_FILTER = "Pose Files (*.csv *.h5);;All Files (*)"
FEATURE_FILTER = "Feature Tables (*.csv *.parquet);;All Files (*)"
DISPLAY_BODY_MODE_MATCHED = "Matched/JEN"
DISPLAY_BODY_MODE_ALL = "All available"
BODY_MODE_OPTIONS = [DISPLAY_BODY_MODE_MATCHED, DISPLAY_BODY_MODE_ALL]

SKELETON_EDGES = [
    ("nose", "H1R"),
    ("H1R", "H2R"),
    ("H2R", "B1R"),
    ("B1R", "B2R"),
    ("B2R", "B3R"),
    ("nose", "H1L"),
    ("H1L", "H2L"),
    ("H2L", "B1L"),
    ("B1L", "B2L"),
    ("B2L", "B3L"),
    ("B3R", "S2"),
    ("B3L", "S2"),
    ("S1", "S2"),
    ("S2", "tail"),
]

BODY_COLORS = {
    "nose": "#E63946",
    "H1R": "#F4A261",
    "H2R": "#D0D4DB",
    "H1L": "#FF9F1C",
    "H2L": "#D0D4DB",
    "B1R": "#A8B0BA",
    "B2R": "#2A9D8F",
    "B3R": "#98A4B3",
    "B1L": "#A8B0BA",
    "B2L": "#1D7874",
    "B3L": "#98A4B3",
    "tail": "#3A86FF",
    "S2": "#7B9EA8",
    "S1": "#8D5A97",
    "right_ear": "#F4A261",
    "left_ear": "#FFBF69",
    "right_lateral": "#2A9D8F",
    "left_lateral": "#1D7874",
    "tail_base": "#3A86FF",
    "tail_end": "#8B919B",
    "Centroid": "#8D5A97",
}

TRACE_COLORS = ["#0A84FF", "#34C759", "#FF9F0A"]
EXCLUDED_FEATURE_COLUMNS = {"recording", "frame"}
VIDEO_DIR_NAMES = [
    "Videos_all",
    "Videos_experiments_1_and_3",
    "Videos_experiment_5",
    "Videos_two",
    "Videos_tiny",
    "Videos_ratones_malos",
]
LEFT_POSE_DIR = REPO_ROOT / "data" / "to_predict" / "test"
RIGHT_POSE_DIR = REPO_ROOT / "data" / "to_predict" / "matched_dlc_aya_to_jen"
LEFT_FEATURE_DIR = REPO_ROOT / "results" / "predictions" / "test_norm_v2" / "2026-02-10_final_gpu_3min_norm_v2"
RIGHT_FEATURE_DIR = REPO_ROOT / "results" / "predictions" / "matched_dlc_aya_to_jen" / "aya_robust_v3"
RIGHT_VIDEO_DIR = (
    REPO_ROOT
    / "results"
    / "predictions"
    / "matched_dlc_aya_to_jen"
    / "2026-02-10_final_gpu_3min_norm_v2"
    / "figures"
    / "videos"
)
LEFT_VIDEO_DIR_CANDIDATES = [
    REPO_ROOT.parent / "keypoint_moseq_project" / "equipo_project_data" / "Videos_all",
    REPO_ROOT.parent / "keypoint_moseq_project" / "equipo_project_data" / "Videos_experiments_1_and_3",
    REPO_ROOT.parent / "keypoint_moseq_project" / "equipo_project_data" / "Videos_tiny",
]


def normalize_recording_token(name: str) -> str:
    token = Path(str(name)).stem.lower()
    token = token.replace("_matched_to_jen", "")
    token = token.replace("_behavior_overlay", "")
    token = token.replace("_overlay", "")
    token = token.replace("_behavior", "")
    return "".join(ch for ch in token if ch.isalnum())


def duration_from_frame_count(frame_count: int, fps: float) -> float:
    if frame_count <= 1 or fps <= 0:
        return 0.0
    return float(frame_count - 1) / float(fps)


def frame_from_time(time_s: float, fps: float, frame_count: int) -> int:
    if frame_count <= 0 or fps <= 0:
        return 0
    frame = int(round(float(time_s) * float(fps)))
    return max(0, min(int(frame_count) - 1, frame))


def list_bodyparts_from_tracking_df(df: pd.DataFrame) -> list[str]:
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected DLC MultiIndex columns (scorer/bodyparts/coords).")
    bodyparts: list[str] = []
    for value in df.columns.get_level_values(1).unique().tolist():
        bp = str(value).strip()
        if bp.lower() in {"bodyparts", "unnamed: 0", ""}:
            continue
        bodyparts.append(bp)
    return bodyparts


def flatten_tracking_xy_df(df: pd.DataFrame) -> pd.DataFrame:
    scorer = first_scorer(df)
    flat = pd.DataFrame(index=df.index)
    for bp in list_bodyparts_from_tracking_df(df):
        x_col = (scorer, bp, "x")
        y_col = (scorer, bp, "y")
        if x_col in df.columns and y_col in df.columns:
            flat[f"{bp}_x"] = pd.to_numeric(df[x_col], errors="coerce")
            flat[f"{bp}_y"] = pd.to_numeric(df[y_col], errors="coerce")
    return flat.reset_index(drop=True)


def axis_bounds_from_xy(xy_by_bodypart: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    rows = []
    for arr in xy_by_bodypart.values():
        if arr.size == 0:
            continue
        finite = np.isfinite(arr).all(axis=1)
        if finite.any():
            rows.append(arr[finite])
    if not rows:
        return (-1.0, 1.0, -1.0, 1.0)
    merged = np.vstack(rows)
    x_min = float(np.nanmin(merged[:, 0]))
    x_max = float(np.nanmax(merged[:, 0]))
    y_min = float(np.nanmin(merged[:, 1]))
    y_max = float(np.nanmax(merged[:, 1]))
    pad = max(x_max - x_min, y_max - y_min, 1.0) * 0.12
    return (x_min - pad, x_max + pad, y_min - pad, y_max + pad)


def choose_bodyparts_for_mode(bodyparts: list[str], mode: str) -> list[str]:
    if mode == DISPLAY_BODY_MODE_MATCHED:
        matched = [bp for bp in JEN_BODY_ORDER if bp in bodyparts]
        if matched:
            return matched
    return list(bodyparts)


def match_recording_slice(df: pd.DataFrame, recording_stem: str) -> pd.DataFrame:
    if "recording" not in df.columns:
        return df.copy().reset_index(drop=True)
    token = normalize_recording_token(recording_stem)
    rec = df["recording"].astype(str)
    rec_tokens = rec.map(normalize_recording_token)
    mask = rec_tokens == token
    if not mask.any():
        mask = rec_tokens.map(lambda value: token in value or value in token)
    if not mask.any():
        unique_recs = rec.drop_duplicates().tolist()
        if len(unique_recs) == 1:
            mask = rec == unique_recs[0]
        else:
            raise ValueError(f"No recording match found for '{recording_stem}'.")
    return df.loc[mask].copy().reset_index(drop=True)


def load_feature_table_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported feature table type: {path.suffix}")


def extract_feature_columns(df: pd.DataFrame | None) -> list[str]:
    if df is None or df.empty:
        return []
    cols: list[str] = []
    for col in df.columns:
        if str(col) in EXCLUDED_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(str(col))
    return cols


def pick_default_features(feature_cols: list[str], n: int = 3) -> list[str]:
    if not feature_cols:
        return []
    chosen: list[str] = []
    for name in DEFAULT_IMPORTANT_FEATURES:
        if name in feature_cols and name not in chosen:
            chosen.append(name)
    if len(chosen) < n:
        for name in feature_cols:
            if name not in chosen:
                chosen.append(name)
            if len(chosen) >= n:
                break
    return chosen[:n]


def feature_axis_seconds(feature_df: pd.DataFrame, fps: float) -> np.ndarray:
    fps_value = float(fps) if float(fps) > 0 else FALLBACK_FPS
    if "frame" in feature_df.columns:
        frames = pd.to_numeric(feature_df["frame"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        frames = np.arange(len(feature_df), dtype=float)
    return frames / max(fps_value, 1e-8)


def load_fps_manifest(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "recording" not in df.columns or "fps" not in df.columns:
        return {}
    mapping: dict[str, float] = {}
    for _, row in df.iterrows():
        fps = pd.to_numeric(row["fps"], errors="coerce")
        if pd.notna(fps) and float(fps) > 0:
            mapping[normalize_recording_token(str(row["recording"]))] = float(fps)
    return mapping


def discover_nearby_fps_mappings(paths: list[Path | None]) -> list[dict[str, float]]:
    mappings: list[dict[str, float]] = []
    seen: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        for parent in [path.parent, path.parent.parent]:
            manifest = parent / "fps_manifest.csv"
            if manifest in seen or not manifest.exists():
                continue
            seen.add(manifest)
            mapping = load_fps_manifest(manifest)
            if mapping:
                mappings.append(mapping)
    return mappings


def resolve_fps(recording_stem: str, manifest_maps: list[dict[str, float]], video_fps: float | None, fallback: float = FALLBACK_FPS) -> float:
    token = normalize_recording_token(recording_stem)
    for mapping in manifest_maps:
        if token in mapping and mapping[token] > 0:
            return float(mapping[token])
    if video_fps is not None and float(video_fps) > 0:
        return float(video_fps)
    return float(fallback)


def find_feature_suggestions(recording_stem: str, search_root: Path | None = None) -> list[Path]:
    root = search_root or (REPO_ROOT / "results" / "predictions")
    if not root.exists():
        return []
    token = normalize_recording_token(recording_stem)
    suggestions: list[Path] = []
    for summary_path in sorted(root.rglob("predictions_summary.csv")):
        try:
            summary_df = pd.read_csv(summary_path, usecols=["recording"])
        except Exception:
            continue
        if "recording" not in summary_df.columns:
            continue
        rec_tokens = summary_df["recording"].astype(str).map(normalize_recording_token)
        if not rec_tokens.map(lambda value: value == token or token in value or value in token).any():
            continue
        run_dir = summary_path.parent
        for name in (
            "computed_features_raw.parquet",
            "computed_features_raw.csv",
            "computed_features_model_aligned.parquet",
            "computed_features_model_aligned.csv",
        ):
            candidate = run_dir / name
            if candidate.exists() and candidate not in suggestions:
                suggestions.append(candidate)
    return suggestions


def first_existing_dir(candidates: list[Path | None]) -> Path:
    for candidate in candidates:
        if candidate is not None and candidate.exists() and candidate.is_dir():
            return candidate
    return REPO_ROOT


def right_video_dir_candidates() -> list[Path | None]:
    raw_source_dir = _resolve_default_input_dir()
    return [
        raw_source_dir,
        Path(os.getenv("COPING_DYNAMICS_AYA_RAW_DIR", "")) if os.getenv("COPING_DYNAMICS_AYA_RAW_DIR") else None,
        Path(os.getenv("AYA_BEHAVIOR_ENCODED_DIR", "")) if os.getenv("AYA_BEHAVIOR_ENCODED_DIR") else None,
        Path(os.getenv("COPING_DYNAMICS_AYA_VIDEO_DIR", "")) if os.getenv("COPING_DYNAMICS_AYA_VIDEO_DIR") else None,
        REPO_ROOT / "data" / "aya_videos",
        REPO_ROOT.parent / "keypoint_moseq_project" / "aya_videos",
        RIGHT_VIDEO_DIR,
        REPO_ROOT / "results" / "predictions" / "matched_dlc_aya_to_jen" / "2026-02-10_final_gpu_3min_v1" / "figures" / "videos",
        REPO_ROOT / "results" / "predictions" / "matched_dlc_aya_to_jen" / "2026-02-10_baseline_v1" / "figures" / "videos",
    ]


def default_video_search_dir(side_name: str) -> Path:
    if side_name.lower() == "right":
        return first_existing_dir(right_video_dir_candidates())
    return first_existing_dir(LEFT_VIDEO_DIR_CANDIDATES)


def default_pose_search_dir(side_name: str) -> Path:
    if side_name.lower() == "right":
        return first_existing_dir([RIGHT_POSE_DIR])
    return first_existing_dir([LEFT_POSE_DIR])


def default_feature_search_dir(side_name: str) -> Path:
    if side_name.lower() == "right":
        return first_existing_dir([RIGHT_FEATURE_DIR, REPO_ROOT / "results" / "predictions" / "matched_dlc_aya_to_jen"])
    return first_existing_dir([LEFT_FEATURE_DIR, REPO_ROOT / "results" / "predictions" / "test_norm_v2"])


def find_matching_file(recording_stem: str, search_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    if not search_dir.exists() or not search_dir.is_dir():
        return None
    token = normalize_recording_token(recording_stem)
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(search_dir.glob(pattern)))
    if not candidates:
        return None

    exact: list[Path] = []
    contains: list[Path] = []
    for candidate in candidates:
        cand_token = normalize_recording_token(candidate.stem)
        if cand_token == token:
            exact.append(candidate)
        elif token in cand_token or cand_token in token:
            contains.append(candidate)
    if exact:
        return exact[0]
    if contains:
        return contains[0]
    return None


@dataclass
class PoseBundle:
    path: Path
    tracking_df: pd.DataFrame
    flat_xy_df: pd.DataFrame
    scorer: str
    bodyparts: list[str]
    xy: dict[str, np.ndarray]
    frame_count: int
    fps: float
    axis_bounds: tuple[float, float, float, float]

    @property
    def duration_s(self) -> float:
        return duration_from_frame_count(self.frame_count, self.fps)


@dataclass
class FeatureBundle:
    path: Path | None
    df: pd.DataFrame
    feature_cols: list[str]
    source_label: str
    recording_stem: str
    fps: float

    @property
    def frame_count(self) -> int:
        return int(len(self.df))

    @property
    def duration_s(self) -> float:
        return duration_from_frame_count(self.frame_count, self.fps)

    def time_axis(self) -> np.ndarray:
        return feature_axis_seconds(self.df, self.fps)

    def row_index_for_time(self, time_s: float) -> int:
        return frame_from_time(time_s, self.fps, self.frame_count)


def build_pose_bundle(path: Path, fps: float) -> PoseBundle:
    tracking_df = load_tracking_df(path)
    if not isinstance(tracking_df.columns, pd.MultiIndex):
        raise ValueError("Expected DLC-style MultiIndex pose columns.")
    flat_xy = flatten_tracking_xy_df(tracking_df)
    scorer = first_scorer(tracking_df)
    bodyparts = list_bodyparts_from_tracking_df(tracking_df)
    frame_count = int(len(tracking_df))
    xy: dict[str, np.ndarray] = {}
    for bp in bodyparts:
        x_col = (scorer, bp, "x")
        y_col = (scorer, bp, "y")
        arr = np.full((frame_count, 2), np.nan, dtype=float)
        if x_col in tracking_df.columns and y_col in tracking_df.columns:
            arr[:, 0] = pd.to_numeric(tracking_df[x_col], errors="coerce").to_numpy(dtype=float)
            arr[:, 1] = pd.to_numeric(tracking_df[y_col], errors="coerce").to_numpy(dtype=float)
        xy[bp] = arr
    fps_value = float(fps) if float(fps) > 0 else FALLBACK_FPS
    return PoseBundle(
        path=path,
        tracking_df=tracking_df,
        flat_xy_df=flat_xy,
        scorer=scorer,
        bodyparts=bodyparts,
        xy=xy,
        frame_count=frame_count,
        fps=fps_value,
        axis_bounds=axis_bounds_from_xy(xy),
    )


def build_feature_bundle(
    feature_path: Path | None,
    *,
    pose_bundle: PoseBundle,
    recording_stem: str,
    fps: float,
) -> FeatureBundle:
    if feature_path is None:
        raw_df = compute_kinematic_features(pose_bundle.flat_xy_df, fps=max(1, int(round(float(fps)))))
        feature_df = raw_df.copy()
        feature_df.insert(0, "frame", np.arange(len(feature_df), dtype=int))
        feature_df.insert(0, "recording", recording_stem)
        source_label = "Computed from pose"
    else:
        feature_df = load_feature_table_file(feature_path)
        if feature_df.empty:
            raise ValueError(f"Feature table is empty: {feature_path}")
        feature_df = match_recording_slice(feature_df, recording_stem)
        source_label = feature_path.name
        if "frame" not in feature_df.columns:
            feature_df.insert(0, "frame", np.arange(len(feature_df), dtype=int))
        else:
            feature_df["frame"] = pd.to_numeric(feature_df["frame"], errors="coerce").fillna(0).astype(int)
            feature_df = feature_df.sort_values("frame").reset_index(drop=True)
        if "recording" not in feature_df.columns:
            feature_df.insert(0, "recording", recording_stem)
    feature_cols = extract_feature_columns(feature_df)
    return FeatureBundle(
        path=feature_path,
        df=feature_df.reset_index(drop=True),
        feature_cols=feature_cols,
        source_label=source_label,
        recording_stem=recording_stem,
        fps=float(fps) if float(fps) > 0 else FALLBACK_FPS,
    )


class VideoReader:
    """Small OpenCV video reader with a tiny frame cache around the cursor."""

    def __init__(self, path: Path, cache_size: int = 8) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        self.frame_count = max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.metadata_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.cache_size = max(2, int(cache_size))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    @property
    def duration_s(self) -> float:
        fps = self.metadata_fps if self.metadata_fps > 0 else FALLBACK_FPS
        return duration_from_frame_count(self.frame_count, fps)

    def get_frame(self, frame_idx: int) -> np.ndarray | None:
        if self.frame_count <= 0:
            return None
        frame_idx = max(0, min(self.frame_count - 1, int(frame_idx)))
        if frame_idx in self._cache:
            frame = self._cache.pop(frame_idx)
            self._cache[frame_idx] = frame
            return frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        self._cache[frame_idx] = frame
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._cache.clear()


@dataclass
class SideState:
    name: str
    video_path: Path | None = None
    pose_path: Path | None = None
    feature_path: Path | None = None
    video: VideoReader | None = None
    pose: PoseBundle | None = None
    features: FeatureBundle | None = None
    fps: float = FALLBACK_FPS
    bodypart_mode: str = DISPLAY_BODY_MODE_MATCHED
    selected_features: list[str] = field(default_factory=list)
    feature_suggestions: list[Path] = field(default_factory=list)
    status_message: str = ""
    widgets: dict[str, Any] = field(default_factory=dict)

    def recording_stem(self) -> str:
        for path in (self.pose_path, self.video_path, self.feature_path):
            if path is not None:
                return path.stem
        return self.name

    def metadata_fps(self) -> float | None:
        if self.video is None:
            return None
        return self.video.metadata_fps if self.video.metadata_fps > 0 else None

    def refresh_fps(self) -> None:
        manifests = discover_nearby_fps_mappings([self.pose_path, self.video_path])
        self.fps = resolve_fps(self.recording_stem(), manifests, self.metadata_fps(), fallback=FALLBACK_FPS)
        if self.pose is not None:
            self.pose.fps = self.fps
        if self.features is not None:
            self.features.fps = self.fps

    def duration_s(self) -> float:
        durations: list[float] = []
        if self.video is not None and self.video.frame_count > 0:
            durations.append(duration_from_frame_count(self.video.frame_count, self.fps))
        if self.pose is not None and self.pose.frame_count > 0:
            durations.append(duration_from_frame_count(self.pose.frame_count, self.fps))
        if self.features is not None and self.features.frame_count > 0:
            durations.append(duration_from_frame_count(self.features.frame_count, self.fps))
        if not durations:
            return 0.0
        return min(durations)

    def close(self) -> None:
        if self.video is not None:
            self.video.close()
            self.video = None


if QT_AVAILABLE:
    class AppVisualWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("app_visual")
            self.resize(1880, 1120)
            self.setMinimumSize(1500, 900)

            self.left = SideState(name="Left")
            self.right = SideState(name="Right")
            self.current_time_s = 0.0
            self.playing = False
            self._updating_slider = False

            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(33)
            self.timer.timeout.connect(self._on_play_tick)

            self._build_ui()
            self._apply_styles()
            self._update_shared_controls()
            self._render_all()

        def _build_ui(self) -> None:
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)

            root = QtWidgets.QVBoxLayout(central)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(14)

            root.addLayout(self._build_top_bar())

            columns = QtWidgets.QHBoxLayout()
            columns.setSpacing(16)
            columns.addWidget(self._build_side_column(self.left), 1)
            columns.addWidget(self._build_side_column(self.right), 1)
            root.addLayout(columns, 1)

            footer = QtWidgets.QVBoxLayout()
            footer.setSpacing(8)
            self.time_readout = QtWidgets.QLabel("Time 00:00.000 / 00:00.000")
            footer.addWidget(self.time_readout)
            self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.time_slider.setRange(0, 0)
            self.time_slider.setSingleStep(1)
            self.time_slider.valueChanged.connect(self._on_slider_changed)
            footer.addWidget(self.time_slider)
            root.addLayout(footer)

        def _build_top_bar(self) -> QtWidgets.QHBoxLayout:
            layout = QtWidgets.QHBoxLayout()
            layout.setSpacing(18)

            left_box = QtWidgets.QLabel("Left side: explicit video, pose, and optional feature table.")
            right_box = QtWidgets.QLabel("Right side: explicit video, pose, and optional feature table.")
            left_box.setWordWrap(True)
            right_box.setWordWrap(True)

            self.play_button = QtWidgets.QPushButton("Play")
            self.play_button.clicked.connect(self._toggle_play)
            self.step_back_button = QtWidgets.QPushButton("Step -1")
            self.step_back_button.clicked.connect(lambda: self._step_frames(-1))
            self.step_forward_button = QtWidgets.QPushButton("Step +1")
            self.step_forward_button.clicked.connect(lambda: self._step_frames(1))

            self.speed_combo = QtWidgets.QComboBox()
            for label in ("0.25x", "0.5x", "1x", "2x", "4x"):
                self.speed_combo.addItem(label)
            self.speed_combo.setCurrentText("1x")

            playback = QtWidgets.QHBoxLayout()
            playback.setSpacing(10)
            playback.addWidget(self.step_back_button)
            playback.addWidget(self.play_button)
            playback.addWidget(self.step_forward_button)
            playback.addWidget(QtWidgets.QLabel("Speed"))
            playback.addWidget(self.speed_combo)

            playback_wrap = QtWidgets.QWidget()
            playback_wrap.setLayout(playback)

            layout.addWidget(left_box, 1)
            layout.addWidget(playback_wrap, 0)
            layout.addWidget(right_box, 1)
            return layout

        def _build_side_column(self, side: SideState) -> QtWidgets.QWidget:
            container = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(container)
            layout.setSpacing(10)

            title = QtWidgets.QLabel(side.name)
            title.setObjectName("SideTitle")
            layout.addWidget(title)

            control_row = QtWidgets.QHBoxLayout()
            control_row.setSpacing(8)

            load_video = QtWidgets.QPushButton("Load Video")
            load_video.clicked.connect(lambda _checked=False, current=side: self._choose_video(current))
            load_pose = QtWidgets.QPushButton("Load Pose")
            load_pose.clicked.connect(lambda _checked=False, current=side: self._choose_pose(current))
            load_features = QtWidgets.QPushButton("Load Features")
            load_features.clicked.connect(lambda _checked=False, current=side: self._choose_features(current))

            body_mode = QtWidgets.QComboBox()
            body_mode.addItems(BODY_MODE_OPTIONS)
            body_mode.setCurrentText(DISPLAY_BODY_MODE_MATCHED)
            body_mode.currentTextChanged.connect(lambda value, current=side: self._on_body_mode_changed(current, value))

            control_row.addWidget(load_video)
            control_row.addWidget(load_pose)
            control_row.addWidget(load_features)
            control_row.addStretch(1)
            control_row.addWidget(QtWidgets.QLabel("Bodyparts"))
            control_row.addWidget(body_mode)
            layout.addLayout(control_row)

            video_label = QtWidgets.QLabel("Video: not loaded")
            pose_label = QtWidgets.QLabel("Pose: not loaded")
            feature_label = QtWidgets.QLabel("Features: not loaded")
            suggestion_label = QtWidgets.QLabel("Feature suggestion: none")
            for label in (video_label, pose_label, feature_label, suggestion_label):
                label.setWordWrap(True)
                layout.addWidget(label)

            feature_select_row = QtWidgets.QHBoxLayout()
            feature_select_row.setSpacing(8)
            feature_combos: list[QtWidgets.QComboBox] = []
            for idx in range(3):
                feature_select_row.addWidget(QtWidgets.QLabel(f"Feature {idx + 1}"))
                combo = QtWidgets.QComboBox()
                combo.addItem("")
                combo.currentTextChanged.connect(lambda _value="", current=side: self._on_feature_selection_changed(current))
                feature_select_row.addWidget(combo, 1)
                feature_combos.append(combo)
            layout.addLayout(feature_select_row)

            video_plot, video_image, video_text = self._create_video_plot()
            pose_plot, pose_scatter = self._create_pose_plot()
            feature_plot, feature_cursor = self._create_feature_plot()
            feature_values = QtWidgets.QLabel("Current values: n/a")
            feature_values.setWordWrap(True)

            layout.addWidget(video_plot, 5)
            layout.addWidget(pose_plot, 3)
            layout.addWidget(feature_values)
            layout.addWidget(feature_plot, 3)

            side.widgets.update(
                {
                    "body_mode": body_mode,
                    "video_label": video_label,
                    "pose_label": pose_label,
                    "feature_label": feature_label,
                    "suggestion_label": suggestion_label,
                    "feature_combos": feature_combos,
                    "video_plot": video_plot,
                    "video_image": video_image,
                    "video_text": video_text,
                    "pose_plot": pose_plot,
                    "pose_scatter": pose_scatter,
                    "pose_lines": [],
                    "pose_labels": [],
                    "feature_plot": feature_plot,
                    "feature_lines": [],
                    "feature_cursor": feature_cursor,
                    "feature_values": feature_values,
                }
            )
            return container

        def _create_video_plot(self) -> tuple[pg.PlotWidget, pg.ImageItem, pg.TextItem]:
            plot = pg.PlotWidget(background="#101010")
            plot.hideAxis("bottom")
            plot.hideAxis("left")
            plot.setMenuEnabled(False)
            plot.getViewBox().setMouseEnabled(x=False, y=False)
            plot.getViewBox().invertY(True)
            plot.setAspectLocked(True)
            image = pg.ImageItem()
            plot.addItem(image)
            text = pg.TextItem(color="#F5F5F5", anchor=(0, 0))
            plot.addItem(text)
            return plot, image, text

        def _create_pose_plot(self) -> tuple[pg.PlotWidget, pg.ScatterPlotItem]:
            plot = pg.PlotWidget(background="#111111")
            plot.showGrid(x=True, y=True, alpha=0.22)
            plot.setMenuEnabled(False)
            plot.setAspectLocked(True)
            plot.getViewBox().invertY(True)
            self._style_plot_axes(plot)
            plot.setLabel("bottom", "X", color="#F4F4F4", **{"font-size": "11pt"})
            plot.setLabel("left", "Y", color="#F4F4F4", **{"font-size": "11pt"})
            scatter = pg.ScatterPlotItem(size=16, pxMode=True)
            plot.addItem(scatter)
            return plot, scatter

        def _create_feature_plot(self) -> tuple[pg.PlotWidget, pg.InfiniteLine]:
            plot = pg.PlotWidget(background="#111111")
            plot.showGrid(x=True, y=True, alpha=0.24)
            plot.setMenuEnabled(False)
            self._style_plot_axes(plot)
            plot.setLabel("bottom", "Time (s)", color="#F4F4F4", **{"font-size": "11pt"})
            plot.setLabel("left", "Value", color="#F4F4F4", **{"font-size": "11pt"})
            cursor = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen("#EAEAEA", width=2))
            plot.addItem(cursor)
            return plot, cursor

        def _style_plot_axes(self, plot: pg.PlotWidget) -> None:
            tick_font = QtGui.QFont("Segoe UI", 10)
            for axis_name in ("bottom", "left"):
                axis = plot.getAxis(axis_name)
                axis.setTextPen(pg.mkPen("#F4F4F4"))
                axis.setTickPen(pg.mkPen("#A0A0A0"))
                axis.setPen(pg.mkPen("#A0A0A0"))
                axis.setStyle(tickFont=tick_font, tickTextOffset=10)

        def _apply_styles(self) -> None:
            font = QtGui.QFont("Segoe UI", 11)
            self.setFont(font)
            self.setStyleSheet(
                """
                QWidget {
                    background: #111111;
                    color: #F4F4F4;
                }
                QLabel {
                    color: #F4F4F4;
                    background: transparent;
                    font-size: 11pt;
                }
                QLabel#SideTitle {
                    font-size: 18pt;
                    font-weight: 700;
                }
                QPushButton {
                    min-height: 42px;
                    padding: 8px 14px;
                    font-size: 11pt;
                    font-weight: 600;
                    background: #1D1D1D;
                    color: #F4F4F4;
                    border: 1px solid #3B3B3B;
                    border-radius: 8px;
                }
                QComboBox {
                    min-height: 38px;
                    padding: 4px 8px;
                    font-size: 11pt;
                    background: #1D1D1D;
                    color: #F4F4F4;
                    border: 1px solid #3B3B3B;
                    border-radius: 6px;
                }
                QSlider::groove:horizontal {
                    height: 14px;
                    background: #404040;
                    border-radius: 7px;
                }
                QSlider::handle:horizontal {
                    background: #0A84FF;
                    width: 28px;
                    margin: -8px 0;
                    border-radius: 14px;
                }
                """
            )

        def _speed_multiplier(self) -> float:
            text = self.speed_combo.currentText().replace("x", "").strip()
            try:
                return max(0.05, float(text))
            except ValueError:
                return 1.0

        def _max_shared_time(self) -> float:
            durations = [side.duration_s() for side in (self.left, self.right) if side.pose is not None or side.video is not None]
            if not durations:
                return 0.0
            return min(durations) if len(durations) == 2 else durations[0]

        def _frame_step_seconds(self) -> float:
            fps_values = [side.fps for side in (self.left, self.right) if side.pose is not None or side.video is not None]
            if not fps_values:
                return 1.0 / FALLBACK_FPS
            return 1.0 / max(fps_values)

        def _set_time(self, time_s: float) -> None:
            max_time = self._max_shared_time()
            self.current_time_s = max(0.0, min(float(time_s), max_time))
            self._sync_slider_to_time()
            self._render_all()

        def _sync_slider_to_time(self) -> None:
            self._updating_slider = True
            self.time_slider.setValue(int(round(self.current_time_s * SLIDER_SCALE)))
            self._updating_slider = False

        def _toggle_play(self) -> None:
            if self.playing:
                self.playing = False
                self.timer.stop()
                self.play_button.setText("Play")
                return
            if self._max_shared_time() <= 0:
                return
            self.playing = True
            self.play_button.setText("Pause")
            self.timer.start()

        def _on_play_tick(self) -> None:
            next_time = self.current_time_s + self.timer.interval() / 1000.0 * self._speed_multiplier()
            max_time = self._max_shared_time()
            if next_time >= max_time:
                self._set_time(max_time)
                self.playing = False
                self.timer.stop()
                self.play_button.setText("Play")
                return
            self._set_time(next_time)

        def _step_frames(self, direction: int) -> None:
            self._set_time(self.current_time_s + direction * self._frame_step_seconds())

        def _on_slider_changed(self, value: int) -> None:
            if self._updating_slider:
                return
            self.current_time_s = float(value) / SLIDER_SCALE
            self._render_all()

        def _choose_video(self, side: SideState) -> None:
            start_dir = str(side.video_path.parent if side.video_path is not None else default_video_search_dir(side.name))
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"{side.name}: Load Video", start_dir, VIDEO_FILTER)
            if not filename:
                return
            self._load_video(side, Path(filename))

        def _choose_pose(self, side: SideState) -> None:
            start_dir = str(side.pose_path.parent if side.pose_path is not None else default_pose_search_dir(side.name))
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"{side.name}: Load Pose", start_dir, POSE_FILTER)
            if not filename:
                return
            self._load_pose(side, Path(filename))

        def _choose_features(self, side: SideState) -> None:
            if side.feature_path is not None:
                start_dir = side.feature_path.parent
            elif side.feature_suggestions:
                start_dir = side.feature_suggestions[0].parent
            else:
                start_dir = default_feature_search_dir(side.name)
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"{side.name}: Load Features", str(start_dir), FEATURE_FILTER)
            if not filename:
                if side.pose is not None and side.features is None:
                    self._compute_fallback_features(side)
                return
            self._load_features(side, Path(filename))

        def _load_video(self, side: SideState, path: Path) -> None:
            try:
                if side.video is not None:
                    side.video.close()
                side.video = VideoReader(path)
                side.video_path = path
                side.status_message = ""
                side.refresh_fps()
            except Exception as exc:
                self._show_error(f"{side.name} video load failed", exc)
                return
            self._auto_load_related_data_for_video(side)
            self._after_side_data_changed(side)

        def _auto_load_related_data_for_video(self, side: SideState) -> None:
            token_stem = side.recording_stem()

            pose_match = find_matching_file(
                token_stem,
                default_pose_search_dir(side.name),
                ("*.csv", "*.h5"),
            )

            messages: list[str] = []

            if pose_match is not None:
                try:
                    side.pose_path = pose_match
                    side.pose = build_pose_bundle(pose_match, fps=side.fps)
                    side.feature_suggestions = find_feature_suggestions(pose_match.stem, default_feature_search_dir(side.name))
                    messages.append(f"Auto-loaded pose: {pose_match.name}")
                except Exception as exc:
                    messages.append(f"Pose auto-load failed: {exc}")
                    side.pose = None
            else:
                side.pose = None
                messages.append("No matching pose found. Please load it manually.")

            if side.pose is not None:
                feature_match = None
                if side.feature_suggestions:
                    feature_match = side.feature_suggestions[0]
                else:
                    feature_match = find_matching_file(
                        token_stem,
                        default_feature_search_dir(side.name),
                        ("*.csv", "*.parquet"),
                    )
                if feature_match is not None:
                    try:
                        side.feature_path = feature_match
                        side.features = build_feature_bundle(
                            feature_match,
                            pose_bundle=side.pose,
                            recording_stem=side.recording_stem(),
                            fps=side.fps,
                        )
                        side.selected_features = pick_default_features(side.features.feature_cols, n=3)
                        messages.append(f"Auto-loaded features: {feature_match.name}")
                    except Exception as exc:
                        side.features = None
                        side.feature_path = None
                        messages.append(f"Feature auto-load failed: {exc}")
                else:
                    side.features = None
                    side.feature_path = None
                    messages.append("No matching feature table found. Please load it manually.")
            else:
                side.features = None
                side.feature_path = None

            side.status_message = " | ".join(messages)

        def _load_pose(self, side: SideState, path: Path) -> None:
            try:
                side.pose_path = path
                side.status_message = ""
                side.refresh_fps()
                side.pose = build_pose_bundle(path, fps=side.fps)
                side.feature_suggestions = find_feature_suggestions(path.stem, default_feature_search_dir(side.name))
                if side.feature_path is not None:
                    side.features = build_feature_bundle(
                        side.feature_path,
                        pose_bundle=side.pose,
                        recording_stem=side.recording_stem(),
                        fps=side.fps,
                    )
                    side.selected_features = pick_default_features(side.features.feature_cols, n=3)
                else:
                    self._compute_fallback_features(side)
            except Exception as exc:
                self._show_error(f"{side.name} pose load failed", exc)
                return
            self._after_side_data_changed(side)

        def _load_features(self, side: SideState, path: Path) -> None:
            if side.pose is None:
                self._show_error(f"{side.name} features load failed", ValueError("Load pose first so features can be aligned."))
                return
            try:
                side.feature_path = path
                side.status_message = ""
                side.features = build_feature_bundle(path, pose_bundle=side.pose, recording_stem=side.recording_stem(), fps=side.fps)
                side.selected_features = pick_default_features(side.features.feature_cols, n=3)
            except Exception as exc:
                self._show_error(f"{side.name} features load failed", exc)
                return
            self._after_side_data_changed(side)

        def _compute_fallback_features(self, side: SideState) -> None:
            if side.pose is None:
                return
            side.feature_path = None
            side.features = build_feature_bundle(None, pose_bundle=side.pose, recording_stem=side.recording_stem(), fps=side.fps)
            if not side.selected_features:
                side.selected_features = pick_default_features(side.features.feature_cols, n=3)

        def _after_side_data_changed(self, side: SideState) -> None:
            side.refresh_fps()
            if side.pose is not None:
                side.pose.fps = side.fps
            if side.features is not None:
                side.features.fps = side.fps
                if not side.selected_features:
                    side.selected_features = pick_default_features(side.features.feature_cols, n=3)
            self._populate_feature_combos(side)
            self._update_side_labels(side)
            self._update_shared_controls()
            self._set_time(min(self.current_time_s, self._max_shared_time()))

        def _populate_feature_combos(self, side: SideState) -> None:
            combos: list[QtWidgets.QComboBox] = side.widgets["feature_combos"]
            feature_cols = side.features.feature_cols if side.features is not None else []
            selected = side.selected_features or pick_default_features(feature_cols, n=3)
            selected = selected[:3]
            side.selected_features = selected
            for idx, combo in enumerate(combos):
                combo.blockSignals(True)
                combo.clear()
                combo.addItem("")
                combo.addItems(feature_cols)
                combo.setCurrentText(selected[idx] if idx < len(selected) else "")
                combo.blockSignals(False)

        def _update_side_labels(self, side: SideState) -> None:
            video_label: QtWidgets.QLabel = side.widgets["video_label"]
            pose_label: QtWidgets.QLabel = side.widgets["pose_label"]
            feature_label: QtWidgets.QLabel = side.widgets["feature_label"]
            suggestion_label: QtWidgets.QLabel = side.widgets["suggestion_label"]

            if side.video_path is not None and side.video is not None:
                video_label.setText(
                    f"Video: {side.video_path.name} | fps={side.fps:.3f} | frames={side.video.frame_count:,}"
                )
            else:
                video_label.setText("Video: not loaded")

            if side.pose_path is not None and side.pose is not None:
                pose_label.setText(
                    f"Pose: {side.pose_path.name} | bodyparts={len(side.pose.bodyparts)} | frames={side.pose.frame_count:,}"
                )
            else:
                pose_label.setText("Pose: not loaded")

            if side.features is not None:
                feature_label.setText(
                    f"Features: {side.features.source_label} | numeric={len(side.features.feature_cols)} | frames={side.features.frame_count:,}"
                )
            else:
                feature_label.setText("Features: not loaded")

            if side.status_message:
                suggestion_label.setText(f"Status: {side.status_message}")
            elif side.feature_suggestions:
                suggestion_label.setText(f"Feature suggestion: {side.feature_suggestions[0]}")
            else:
                suggestion_label.setText("Status: ready")

        def _update_shared_controls(self) -> None:
            max_time = self._max_shared_time()
            self.time_slider.setRange(0, int(round(max_time * SLIDER_SCALE)))
            self.time_slider.setEnabled(max_time > 0)
            enabled = max_time > 0
            self.play_button.setEnabled(enabled)
            self.step_back_button.setEnabled(enabled)
            self.step_forward_button.setEnabled(enabled)
            self.time_readout.setText(f"Time {self._format_time(self.current_time_s)} / {self._format_time(max_time)}")

        def _on_body_mode_changed(self, side: SideState, value: str) -> None:
            side.bodypart_mode = value
            self._render_side_pose(side)

        def _on_feature_selection_changed(self, side: SideState) -> None:
            combos: list[QtWidgets.QComboBox] = side.widgets["feature_combos"]
            selected: list[str] = []
            for combo in combos:
                text = combo.currentText().strip()
                if text and text not in selected:
                    selected.append(text)
            side.selected_features = selected
            self._render_side_features(side)

        def _render_all(self) -> None:
            self._update_shared_controls()
            self._render_side(self.left)
            self._render_side(self.right)

        def _render_side(self, side: SideState) -> None:
            self._render_side_video(side)
            self._render_side_pose(side)
            self._render_side_features(side)

        def _render_side_video(self, side: SideState) -> None:
            image_item: pg.ImageItem = side.widgets["video_image"]
            text_item: pg.TextItem = side.widgets["video_text"]
            plot: pg.PlotWidget = side.widgets["video_plot"]

            if side.video is None:
                image_item.clear()
                text_item.setText("No video")
                text_item.setPos(16, 16)
                return

            frame_idx = frame_from_time(self.current_time_s, side.fps, side.video.frame_count)
            frame = side.video.get_frame(frame_idx)
            if frame is None:
                image_item.clear()
                text_item.setText("Video read failed")
                text_item.setPos(16, 16)
                return

            if frame.ndim == 2:
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image_item.setImage(rgb, autoLevels=False)
            height, width = rgb.shape[:2]
            plot.setXRange(0, width, padding=0)
            plot.setYRange(0, height, padding=0)
            text_item.setText(f"{side.name} | frame {frame_idx:,} | {self._format_time(self.current_time_s)}")
            text_item.setPos(18, 18)

        def _clear_pose_items(self, side: SideState) -> None:
            pose_plot: pg.PlotWidget = side.widgets["pose_plot"]
            for item in side.widgets["pose_lines"]:
                pose_plot.removeItem(item)
            for item in side.widgets["pose_labels"]:
                pose_plot.removeItem(item)
            side.widgets["pose_lines"] = []
            side.widgets["pose_labels"] = []

        def _render_side_pose(self, side: SideState) -> None:
            pose_plot: pg.PlotWidget = side.widgets["pose_plot"]
            scatter: pg.ScatterPlotItem = side.widgets["pose_scatter"]
            self._clear_pose_items(side)

            if side.pose is None:
                scatter.setData([], [])
                return

            frame_idx = frame_from_time(self.current_time_s, side.fps, side.pose.frame_count)
            visible_bodyparts = choose_bodyparts_for_mode(side.pose.bodyparts, side.bodypart_mode)
            points: list[dict[str, Any]] = []
            coords_by_bp: dict[str, tuple[float, float]] = {}

            for bp in visible_bodyparts:
                arr = side.pose.xy.get(bp)
                if arr is None or frame_idx >= len(arr):
                    continue
                x, y = arr[frame_idx]
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                coords_by_bp[bp] = (float(x), float(y))
                points.append(
                    {
                        "pos": (float(x), float(y)),
                        "brush": pg.mkBrush(BODY_COLORS.get(bp, "#222222")),
                        "pen": pg.mkPen("#FFFFFF", width=1.2),
                        "size": 17,
                    }
                )

            scatter.setData(points)
            x0, x1, y0, y1 = side.pose.axis_bounds
            pose_plot.setXRange(x0, x1, padding=0)
            pose_plot.setYRange(y0, y1, padding=0)

            for bp_a, bp_b in SKELETON_EDGES:
                if bp_a not in coords_by_bp or bp_b not in coords_by_bp:
                    continue
                x_vals = [coords_by_bp[bp_a][0], coords_by_bp[bp_b][0]]
                y_vals = [coords_by_bp[bp_a][1], coords_by_bp[bp_b][1]]
                line = pg.PlotDataItem(x_vals, y_vals, pen=pg.mkPen("#CFCFCF", width=2))
                pose_plot.addItem(line)
                side.widgets["pose_lines"].append(line)

            for bp, (x_pos, y_pos) in coords_by_bp.items():
                label = pg.TextItem(text=bp, color="#F4F4F4", anchor=(0, 1))
                label.setPos(x_pos + 3.0, y_pos - 3.0)
                pose_plot.addItem(label)
                side.widgets["pose_labels"].append(label)

        def _render_side_features(self, side: SideState) -> None:
            plot: pg.PlotWidget = side.widgets["feature_plot"]
            value_label: QtWidgets.QLabel = side.widgets["feature_values"]
            cursor: pg.InfiniteLine = side.widgets["feature_cursor"]

            for line in side.widgets["feature_lines"]:
                plot.removeItem(line)
            side.widgets["feature_lines"] = []

            if side.features is None or not side.features.feature_cols:
                value_label.setText("Current values: n/a")
                cursor.setPos(self.current_time_s)
                return

            selected = side.selected_features or pick_default_features(side.features.feature_cols, n=3)
            selected = [name for name in selected if name in side.features.feature_cols][:3]
            side.selected_features = selected

            x_axis = side.features.time_axis()
            value_bits: list[str] = []
            for idx, name in enumerate(selected):
                series = pd.to_numeric(side.features.df[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
                line = plot.plot(
                    x_axis,
                    series.to_numpy(dtype=float),
                    pen=pg.mkPen(TRACE_COLORS[idx % len(TRACE_COLORS)], width=3),
                )
                side.widgets["feature_lines"].append(line)
                row_idx = side.features.row_index_for_time(self.current_time_s)
                current_val = pd.to_numeric(side.features.df.iloc[row_idx][name], errors="coerce")
                if pd.isna(current_val):
                    value_bits.append(f"<span style='color:{TRACE_COLORS[idx]}'>{name}: n/a</span>")
                else:
                    value_bits.append(f"<span style='color:{TRACE_COLORS[idx]}'>{name}: {float(current_val):+.3f}</span>")

            cursor.setPos(self.current_time_s)
            value_label.setText("Current values: " + " | ".join(value_bits) if value_bits else "Current values: n/a")
            if len(x_axis) > 0:
                plot.setXRange(float(x_axis.min()), float(x_axis.max()), padding=0.02)

        def _show_error(self, title: str, exc: Exception) -> None:
            QtWidgets.QMessageBox.critical(self, title, str(exc))

        @staticmethod
        def _format_time(time_s: float) -> str:
            total_ms = int(round(max(0.0, float(time_s)) * 1000.0))
            minutes, rem_ms = divmod(total_ms, 60000)
            seconds, millis = divmod(rem_ms, 1000)
            return f"{minutes:02d}:{seconds:02d}.{millis:03d}"

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - GUI teardown
            self.left.close()
            self.right.close()
            super().closeEvent(event)


def run() -> int:
    if not QT_AVAILABLE:
        message = "PyQt6 and pyqtgraph are required to run app_visual."
        if QT_IMPORT_ERROR is not None:
            message = f"{message}\nImport error: {QT_IMPORT_ERROR}"
        raise SystemExit(message)

    app = QtWidgets.QApplication(sys.argv)
    window = AppVisualWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    raise SystemExit(run())
