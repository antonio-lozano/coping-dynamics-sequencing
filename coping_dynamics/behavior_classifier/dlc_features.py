"""Convert DeepLabCut tracks into Figure 7A-C classifier features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NOSE_CANDIDATES = ("nose", "snout", "Nose")
TAIL_CANDIDATES = ("tailbase", "tail_base", "TailBase", "body", "center")


def read_dlc_tracks(path: str | Path) -> pd.DataFrame:
    """Read a DLC filtered CSV/H5 file and return flat bodypart coordinate columns."""

    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        df = pd.read_hdf(path)
    else:
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)

    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels >= 3:
            df.columns = [f"{bp}_{coord}" for _, bp, coord, *rest in df.columns.to_flat_index()]
        elif df.columns.nlevels == 2:
            df.columns = [f"{bp}_{coord}" for bp, coord in df.columns.to_flat_index()]
    return df


def _bodyparts(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        if str(col).endswith("_x"):
            bp = str(col)[:-2]
            if f"{bp}_y" in df.columns:
                out.append(bp)
    if not out:
        raise ValueError("No paired DLC *_x/*_y bodypart columns found.")
    return out


def _first_present(candidates: tuple[str, ...], bodyparts: list[str]) -> str | None:
    lower_lookup = {bp.lower(): bp for bp in bodyparts}
    for name in candidates:
        if name.lower() in lower_lookup:
            return lower_lookup[name.lower()]
    return None


def _wrap_angle_delta(angle: np.ndarray) -> np.ndarray:
    delta = np.diff(angle, prepend=angle[0])
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _pca_heading(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    headings = np.zeros(x.shape[0], dtype=float)
    for i in range(x.shape[0]):
        pts = np.column_stack([x[i], y[i]])
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 2:
            headings[i] = headings[i - 1] if i else 0.0
            continue
        centered = pts - pts.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        headings[i] = float(np.arctan2(vh[0, 1], vh[0, 0]))
    return np.unwrap(headings)


def dlc_tracks_to_pose_table(
    dlc_path: str | Path,
    *,
    name: str | None = None,
    fps: float = 25.0,
    likelihood_threshold: float = 0.2,
) -> pd.DataFrame:
    """Build the base pose table expected by the behavior classifier."""

    flat = read_dlc_tracks(dlc_path)
    bps = _bodyparts(flat)
    x_cols = [f"{bp}_x" for bp in bps]
    y_cols = [f"{bp}_y" for bp in bps]

    x = flat[x_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = flat[y_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    for i, bp in enumerate(bps):
        lik_col = f"{bp}_likelihood"
        if lik_col in flat.columns:
            likelihood = pd.to_numeric(flat[lik_col], errors="coerce").to_numpy(dtype=float)
            bad = likelihood < likelihood_threshold
            x[bad, i] = np.nan
            y[bad, i] = np.nan

    centroid_x = np.nanmean(x, axis=1)
    centroid_y = np.nanmean(y, axis=1)
    centroid_x = pd.Series(centroid_x).interpolate(limit_direction="both").fillna(0.0).to_numpy()
    centroid_y = pd.Series(centroid_y).interpolate(limit_direction="both").fillna(0.0).to_numpy()

    nose = _first_present(NOSE_CANDIDATES, bps)
    tail = _first_present(TAIL_CANDIDATES, bps)
    body_length = np.zeros(len(flat), dtype=float)
    if nose and tail:
        body_dx = flat[f"{nose}_x"].to_numpy(dtype=float) - flat[f"{tail}_x"].to_numpy(dtype=float)
        body_dy = flat[f"{nose}_y"].to_numpy(dtype=float) - flat[f"{tail}_y"].to_numpy(dtype=float)
        body_length = np.sqrt(body_dx * body_dx + body_dy * body_dy)
        heading = np.arctan2(
            body_dy,
            body_dx,
        )
    elif nose:
        other = [bp for bp in bps if bp != nose]
        base_x = np.nanmean(flat[[f"{bp}_x" for bp in other]].to_numpy(dtype=float), axis=1)
        base_y = np.nanmean(flat[[f"{bp}_y" for bp in other]].to_numpy(dtype=float), axis=1)
        body_dx = flat[f"{nose}_x"].to_numpy(dtype=float) - base_x
        body_dy = flat[f"{nose}_y"].to_numpy(dtype=float) - base_y
        body_length = np.sqrt(body_dx * body_dx + body_dy * body_dy)
        heading = np.arctan2(
            body_dy,
            body_dx,
        )
    else:
        heading = _pca_heading(x, y)
        centered_x = x - np.nanmean(x, axis=1, keepdims=True)
        centered_y = y - np.nanmean(y, axis=1, keepdims=True)
        body_length = np.nanmax(np.sqrt(centered_x * centered_x + centered_y * centered_y), axis=1)

    heading = (
        pd.Series(np.unwrap(heading)).interpolate(limit_direction="both").fillna(0.0).to_numpy()
    )
    dx = np.diff(centroid_x, prepend=centroid_x[0])
    dy = np.diff(centroid_y, prepend=centroid_y[0])
    velocity_px_s = np.sqrt(dx * dx + dy * dy) * float(fps)
    angular_velocity = _wrap_angle_delta(heading) * float(fps)
    body_tilt = heading
    body_tilt_delta = _wrap_angle_delta(body_tilt) * float(fps)
    body_tilt_abs_delta = np.abs(body_tilt_delta)
    body_tilt_variability = (
        pd.Series(body_tilt_delta).rolling(window=15, min_periods=1).std().fillna(0.0).to_numpy()
    )
    global_turning_speed = np.abs(angular_velocity)

    source_name = name or Path(dlc_path).stem
    return pd.DataFrame(
        {
            "name": source_name,
            "frame_index": np.arange(len(flat), dtype=int),
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "heading": heading,
            "angular_velocity": angular_velocity,
            "velocity_px_s": velocity_px_s,
            "body_length": pd.Series(body_length).interpolate(limit_direction="both").fillna(0.0),
            "body_tilt": body_tilt,
            "body_tilt_delta": body_tilt_delta,
            "body_tilt_abs_delta": body_tilt_abs_delta,
            "body_tilt_variability": body_tilt_variability,
            "global_turning_speed": global_turning_speed,
        }
    )
