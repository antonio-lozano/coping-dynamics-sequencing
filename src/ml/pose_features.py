"""Pose feature extraction and dataset assembly for behavior modeling."""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import BEHAVIOR_MAPPING, FPS

# Body part mapping from DLC output to human-readable labels.
BODYPART_NAMES = {
    "nose": "Nose",
    "H1R": "Head Right 1",
    "H2R": "Head Right 2",
    "H1L": "Head Left 1",
    "H2L": "Head Left 2",
    "B1R": "Right Body 1",
    "B2R": "Right Body 2",
    "B3R": "Right Body 3",
    "B1L": "Left Body 1",
    "B2L": "Left Body 2",
    "B3L": "Left Body 3",
    "tail": "Tail",
    "S2": "Spine 2",
    "S1": "Spine 1",
}

DEFAULT_IMPORTANT_FEATURES = [
    "Summed Velocity",
    "Angular Velocity",
    "Nose - Tail Distance",
]


def normalize_recording_name(name: str) -> str:
    """Normalize recording name for robust matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_dlc_csv(csv_path: Path) -> pd.DataFrame:
    """Load a DLC CSV file and return x/y coordinates per bodypart per frame."""
    # DLC files commonly store scorer/bodypart/coord as first 3 header rows.
    for header_rows in ([1, 2], [0, 1]):
        try:
            df = pd.read_csv(csv_path, header=header_rows, index_col=0)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [f"{bp}_{coord}" for bp, coord in df.columns]
                keep_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
                if keep_cols:
                    return df[keep_cols].reset_index(drop=True)
        except Exception:
            continue

    # Fallback for already flattened CSVs.
    df = pd.read_csv(csv_path)
    keep_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    if not keep_cols:
        raise ValueError("No x/y coordinate columns found in CSV.")
    return df[keep_cols].reset_index(drop=True)


def compute_kinematic_features(
    pose_df: pd.DataFrame,
    fps: int = FPS,
    window: int = 15,
    bodypart_names: dict[str, str] | None = None,
    excluded_bodyparts: set[str] | None = None,
) -> pd.DataFrame:
    """Compute per-frame kinematic features from DLC x/y coordinates."""
    def _velocity_from_xy(xy: np.ndarray, fps_value: int, n: int) -> np.ndarray:
        vel = np.full(n, np.nan, dtype=float)
        if n <= 1:
            vel[:] = 0.0
            return vel
        prev_xy = xy[:-1]
        next_xy = xy[1:]
        valid = np.isfinite(prev_xy).all(axis=1) & np.isfinite(next_xy).all(axis=1)
        steps = np.full(n - 1, np.nan, dtype=float)
        if valid.any():
            delta = next_xy[valid] - prev_xy[valid]
            steps[valid] = np.sqrt(np.sum(delta ** 2, axis=1)) * fps_value
        vel[0] = 0.0
        vel[1:] = steps
        return vel

    names = bodypart_names or BODYPART_NAMES
    excluded = {str(bp).strip().lower() for bp in (excluded_bodyparts or set())}
    features: dict[str, np.ndarray] = {}
    n_frames = len(pose_df)

    bodyparts = sorted(set(c.rsplit("_", 1)[0] for c in pose_df.columns))
    coords = {}
    for bp in bodyparts:
        if str(bp).strip().lower() in excluded:
            continue
        x_col = f"{bp}_x"
        y_col = f"{bp}_y"
        if x_col in pose_df.columns and y_col in pose_df.columns:
            coords[bp] = pose_df[[x_col, y_col]].to_numpy()

    bp_list = list(coords.keys())

    # Pairwise distances.
    for i, bp1 in enumerate(bp_list):
        for bp2 in bp_list[i + 1 :]:
            dist = np.sqrt(np.sum((coords[bp1] - coords[bp2]) ** 2, axis=1))
            bp1_nice = names.get(bp1, bp1)
            bp2_nice = names.get(bp2, bp2)
            features[f"{bp1_nice} - {bp2_nice} Distance"] = dist

    # Bodypart velocities.
    vel_by_bp: dict[str, np.ndarray] = {}
    for bp in bp_list:
        xy = coords[bp]
        vel = _velocity_from_xy(xy, fps, n_frames)
        bp_nice = names.get(bp, bp)
        features[f"{bp_nice} Velocity"] = vel
        vel_by_bp[bp] = vel

    # Angular velocity from nose-tail axis.
    if "nose" in coords and "tail" in coords:
        nose = coords["nose"]
        tail = coords["tail"]
        angle = np.arctan2(nose[:, 1] - tail[:, 1], nose[:, 0] - tail[:, 0])
        ang_vel = np.zeros(n_frames, dtype=float)
        ang_diff = np.diff(angle)
        ang_diff = np.where(ang_diff > np.pi, ang_diff - 2 * np.pi, ang_diff)
        ang_diff = np.where(ang_diff < -np.pi, ang_diff + 2 * np.pi, ang_diff)
        ang_vel[1:] = np.abs(ang_diff) * fps
        features["Angular Velocity"] = ang_vel
        features["SD Angular Velocity"] = (
            pd.Series(ang_vel).rolling(window, center=True, min_periods=1).std().to_numpy()
        )

    # Summed velocity.
    if vel_by_bp:
        vel_stack = np.vstack([v for v in vel_by_bp.values()])
        all_vel = np.nansum(vel_stack, axis=0)
        valid_counts = np.sum(np.isfinite(vel_stack), axis=0)
        all_vel = np.where(valid_counts > 0, all_vel, np.nan)
    else:
        all_vel = np.full(n_frames, np.nan, dtype=float)
    features["Summed Velocity"] = all_vel

    # Center body coordinates and distances.
    body_bps = [bp for bp in bp_list if bp.startswith("B") or bp.startswith("S")]
    if body_bps:
        center_stack = np.stack([coords[bp] for bp in body_bps], axis=0)
        with np.errstate(invalid="ignore"):
            center = np.nanmean(center_stack, axis=0)
        features["Center Body X"] = center[:, 0]
        features["Center Body Y"] = center[:, 1]
        for bp in bp_list:
            dist_to_center = np.sqrt(np.sum((coords[bp] - center) ** 2, axis=1))
            bp_nice = names.get(bp, bp)
            features[f"{bp_nice} - Center Body Distance"] = dist_to_center

    # Rolling stats.
    for key in list(features.keys()):
        if "Distance" in key or "Velocity" in key:
            series = pd.Series(features[key])
            features[f"Mean {key}"] = series.rolling(window, center=True, min_periods=1).mean().to_numpy()
            features[f"SD {key}"] = series.rolling(window, center=True, min_periods=1).std().to_numpy()

    # Tail variability features.
    if "tail" in coords:
        tail_xy = coords["tail"]
        tail_vel = _velocity_from_xy(tail_xy, fps, n_frames)
        features["Mean Tail Velocity"] = (
            pd.Series(tail_vel).rolling(window, center=True, min_periods=1).mean().to_numpy()
        )
        features["Tail Variability"] = (
            pd.Series(tail_vel).rolling(window, center=True, min_periods=1).std().to_numpy()
        )
        for side in ("R", "L"):
            b3 = f"B3{side}"
            if b3 in coords:
                dist = np.sqrt(np.sum((coords[b3] - coords["tail"]) ** 2, axis=1))
                side_name = "Right" if side == "R" else "Left"
                features[f"{side_name} Body 3 - Tail Variability"] = (
                    pd.Series(dist).rolling(window, center=True, min_periods=1).std().to_numpy()
                )

    return pd.DataFrame(features)


def robust_normalize_features(
    feature_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    clip_value: float = 4.0,
    eps: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Robustly normalize features with per-column median/IQR scaling.

    Output values are clipped to [-clip_value, clip_value] to keep overlays stable.
    Returns both normalized dataframe and normalization stats.
    """
    cols = feature_cols or [c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])]
    out = feature_df.copy()
    stats: dict[str, dict[str, float]] = {}

    for col in cols:
        series = pd.to_numeric(feature_df[col], errors="coerce")
        values = series.to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            out[col] = 0.0
            stats[col] = {
                "median": 0.0,
                "q25": 0.0,
                "q75": 0.0,
                "iqr": 0.0,
                "scale_used": float(eps),
            }
            continue

        med = float(np.nanmedian(values))
        q25 = float(np.nanquantile(values, 0.25))
        q75 = float(np.nanquantile(values, 0.75))
        iqr = q75 - q25
        scale = iqr if iqr > eps else eps
        norm = (series - med) / scale
        norm = norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[col] = np.clip(norm.to_numpy(dtype=float), -clip_value, clip_value)
        stats[col] = {
            "median": med,
            "q25": q25,
            "q75": q75,
            "iqr": float(iqr),
            "scale_used": float(scale),
        }
    return out, stats


def _extract_behavior_labels(rec_data: dict, behavior_mapping: dict[int, str]) -> list[str] | None:
    if "syllable" in rec_data:
        labels = rec_data["syllable"]
    elif "syllables_reindexed" in rec_data:
        labels = rec_data["syllables_reindexed"]
    elif "syllables" in rec_data:
        labels = rec_data["syllables"]
    else:
        return None

    behavior_labels = []
    for syl in labels:
        if isinstance(syl, float) and np.isnan(syl):
            behavior_labels.append("Unassigned")
        else:
            behavior_labels.append(behavior_mapping.get(int(syl), "Unassigned"))
    return behavior_labels


def _match_recording_key(csv_stem: str, result_keys: Iterable[str]) -> str | None:
    fname_norm = normalize_recording_name(csv_stem)
    for key in result_keys:
        key_norm = normalize_recording_name(key)
        if key_norm == fname_norm or fname_norm in key_norm or key_norm in fname_norm:
            return key

    for key in result_keys:
        key_norm = normalize_recording_name(key)
        animal_match = re.search(r"animal\s*(\d+)", fname_norm)
        session_match = re.search(r"_(\d+)dlc", fname_norm)
        if animal_match:
            animal_id = animal_match.group(1)
            if f"animal{animal_id}" in key_norm:
                if session_match:
                    session_id = session_match.group(1)
                    if f"_{session_id}" in key_norm or f"_{session_id}." in str(key).lower():
                        return key
                else:
                    return key
    return None


def load_all_dlc_and_compute_features(
    dlc_dir: Path,
    results_pkl: Path,
    index_csv: Path,
    fps: int = FPS,
    label_col: str = "behavior_cluster",
    behavior_mapping: dict[int, str] | None = None,
    csv_glob: str = "*DLC*.csv",
    max_csv_files: int | None = None,
    verbose: bool = True,
    excluded_bodyparts: set[str] | None = None,
) -> pd.DataFrame | None:
    """Build labeled frame-level feature table by joining DLC and MoSeq labels."""
    mapping = behavior_mapping or BEHAVIOR_MAPPING

    if verbose:
        print(f"Loading MoSeq cluster results from {results_pkl}...")
    with open(results_pkl, "rb") as f:
        results_dict = pickle.load(f)

    index_df = pd.read_csv(index_csv)
    name_to_group = {
        normalize_recording_name(row["name"]): row["group"]
        for _, row in index_df.iterrows()
    }

    all_rows = []
    dlc_csvs = sorted(dlc_dir.glob(csv_glob))
    if max_csv_files is not None:
        dlc_csvs = dlc_csvs[: max(0, int(max_csv_files))]
    if verbose:
        print(f"Found {len(dlc_csvs)} DLC CSV files in {dlc_dir}.")

    for csv_path in dlc_csvs:
        matched_key = _match_recording_key(csv_path.stem, results_dict.keys())
        if matched_key is None:
            if verbose:
                print(f"  Skip {csv_path.name}: no matching recording key.")
            continue

        rec_data = results_dict[matched_key]
        behavior_labels = _extract_behavior_labels(rec_data, mapping)
        if behavior_labels is None:
            if verbose:
                print(f"  Skip {matched_key}: no syllable labels.")
            continue

        try:
            pose_df = load_dlc_csv(csv_path)
            feat_df = compute_kinematic_features(
                pose_df,
                fps=fps,
                excluded_bodyparts=excluded_bodyparts,
            )
        except Exception as exc:
            if verbose:
                print(f"  Skip {csv_path.name}: malformed/unreadable CSV ({exc}).")
            continue

        min_len = min(len(feat_df), len(behavior_labels))
        if min_len == 0:
            if verbose:
                print(f"  Skip {csv_path.name}: no usable frames after alignment.")
            continue

        feat_df = feat_df.iloc[:min_len].copy()
        feat_df[label_col] = behavior_labels[:min_len]
        feat_df["recording"] = matched_key

        feat_df["group"] = "Unknown"
        matched_norm = normalize_recording_name(matched_key)
        for name_key, group in name_to_group.items():
            if name_key in matched_norm:
                feat_df["group"] = group
                break

        all_rows.append(feat_df)
        if verbose:
            print(f"  Processed {csv_path.name} -> {matched_key} ({min_len} frames)")

    if not all_rows:
        return None

    combined = pd.concat(all_rows, ignore_index=True)
    if verbose:
        print(f"Total: {len(combined):,} frames with {combined.shape[1] - 3} features.")
    return combined
