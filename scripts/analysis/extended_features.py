"""Extended feature engineering, pose augmentation, coordinate preprocessing,
and global normalization.

This module lives *outside* ``src/`` so the core training/inference library
remains untouched.  It is imported by ``pretrain_xgb_robust.py`` (training)
and ``predict_behavior_xgb.py`` (inference).

Key components
--------------
* **Coordinate preprocessing** – centroid centering + body-length
  normalization applied to raw DLC x/y coordinates *before* feature
  extraction.  Removes camera-position offsets and pixel-scale differences
  so features are comparable across datasets.
* **Pose augmentation** – mild rigid transforms (rotation, translation,
  isotropic scale jitter) applied to raw DLC x/y coordinates *before*
  feature extraction.  Keeps pairwise distances, velocities and angular
  features realistic while simulating camera/rig variability.
* **Extended features** – temporal context (lags, deltas, acceleration),
  postural shape (body area, elongation, limb symmetry, curvature) and
  frequency-domain (rolling FFT energy, dominant frequency).
* **Global normalization** – compute median/IQR from the full training
  distribution, save stats as a JSON artifact, and apply the *same* stats
  at inference so train/test scales match exactly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Coordinate preprocessing (centroid centering + body-length normalisation)
# ---------------------------------------------------------------------------

# Bodyparts that are NOT mapped in AYA → JEN and should be excluded from
# cross-dataset training.  These produce NaN/zero in AYA data.
UNMAPPED_BODYPARTS: set[str] = {"h2r", "h2l", "b1r", "b3r", "b1l", "b3l", "s2"}

# Default reference bodypart pair for body-length normalisation.
# Both nose and S1 (centroid) are available in JEN and AYA.
DEFAULT_REF_PAIR: tuple[str, str] = ("nose", "S1")


def preprocess_coordinates(
    pose_df: pd.DataFrame,
    ref_bodyparts: tuple[str, str] = DEFAULT_REF_PAIR,
    ref_distance: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Centre coordinates on per-frame centroid and normalise to body-length.

    Steps:
      1. **Centroid centering** — subtract the mean (x, y) across all
         bodyparts at each frame, eliminating camera-position offsets.
      2. **Body-length normalisation** — divide all coordinates by a
         reference body distance so features are expressed in body-length
         units, eliminating pixel-scale differences between rigs.

    Parameters
    ----------
    pose_df : DataFrame
        Raw DLC coordinates with ``bp_x`` / ``bp_y`` columns (output of
        ``load_dlc_csv``).
    ref_bodyparts : tuple[str, str]
        The two bodypart column prefixes (e.g. ``("nose", "S1")``) whose
        Euclidean distance defines "one body length".
    ref_distance : float or None
        If provided, use this value as the reference distance instead of
        computing the median from the data.  This is used at *inference* to
        apply the same scale as training (though in practice each recording
        gets its own median, so this parameter is mainly for documentation
        or future fixed-reference experiments).

    Returns
    -------
    out : DataFrame
        Same shape as *pose_df*, with centred & normalised coordinates.
    ref_dist_used : float
        The reference distance that was used for body-length normalisation
        (median Euclidean distance between the two reference bodyparts
        across the recording).
    """
    out = pose_df.copy()
    x_cols = sorted([c for c in out.columns if c.endswith("_x")])
    y_cols = sorted([c for c in out.columns if c.endswith("_y")])

    # --- 1. Centroid centering ----------------------------------------
    # Only average over bodyparts with actual data (non-NaN)
    cx = out[x_cols].mean(axis=1)
    cy = out[y_cols].mean(axis=1)
    for col in x_cols:
        out[col] = out[col] - cx
    for col in y_cols:
        out[col] = out[col] - cy

    # --- 2. Body-length normalisation ---------------------------------
    bp1, bp2 = ref_bodyparts
    x1c, y1c = f"{bp1}_x", f"{bp1}_y"
    x2c, y2c = f"{bp2}_x", f"{bp2}_y"

    if x1c not in out.columns or x2c not in out.columns:
        raise KeyError(
            f"Reference bodyparts {ref_bodyparts} not found in columns. "
            f"Available prefixes: {sorted(set(c.rsplit('_', 1)[0] for c in out.columns))}"
        )

    dx = out[x1c] - out[x2c]
    dy = out[y1c] - out[y2c]
    dist = np.sqrt(dx.astype(float) ** 2 + dy.astype(float) ** 2)

    if ref_distance is not None and ref_distance > 1e-6:
        ref_dist = float(ref_distance)
    else:
        ref_dist = float(np.nanmedian(dist))

    if ref_dist > 1e-6:
        for col in x_cols + y_cols:
            out[col] = out[col] / ref_dist
    else:
        ref_dist = 1.0  # degenerate case — no scaling

    return out, ref_dist


# ---------------------------------------------------------------------------
# 1. Pose augmentation
# ---------------------------------------------------------------------------

def _rotation_matrix(theta: float) -> np.ndarray:
    """2-D rotation matrix for angle *theta* (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def augment_pose(
    pose_df: pd.DataFrame,
    *,
    rotation_range_deg: float = 15.0,
    translation_range_frac: float = 0.05,
    scale_range: float = 0.10,
    per_frame: bool = False,
    seed: int | None = None,
) -> pd.DataFrame:
    """Apply a mild rigid transform + scale jitter to DLC x/y coordinates.

    Parameters
    ----------
    pose_df : DataFrame
        Columns like ``bp_x``, ``bp_y``  (output of ``load_dlc_csv``).
    rotation_range_deg : float
        Maximum rotation in either direction (degrees).
    translation_range_frac : float
        Maximum translation as fraction of the bounding-box span per axis.
    scale_range : float
        Maximum isotropic scale change (±).  0.10 → [0.9, 1.1].
    per_frame : bool
        If *True* sample a *different* transform per frame (heavy augmentation).
        Default *False* samples one transform per recording (mild, realistic).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    DataFrame with identical schema; coordinates transformed in-place copies.
    """
    rng = np.random.default_rng(seed)
    out = pose_df.copy()

    # Collect all (bp, axis) column pairs -------------------------
    x_cols = [c for c in out.columns if c.endswith("_x")]
    y_cols = [c.replace("_x", "_y") for c in x_cols if c.replace("_x", "_y") in out.columns]
    if not x_cols or not y_cols:
        return out

    # Stack into (N, P, 2).  P = number of bodyparts with x+y
    bp_count = len(x_cols)
    n_frames = len(out)
    xy_all = np.empty((n_frames, bp_count, 2), dtype=float)
    for i, (xc, yc) in enumerate(zip(x_cols, y_cols)):
        xy_all[:, i, 0] = out[xc].to_numpy(dtype=float)
        xy_all[:, i, 1] = out[yc].to_numpy(dtype=float)

    # Centre of mass per frame  (N, 2)
    with np.errstate(invalid="ignore"):
        com = np.nanmean(xy_all, axis=1)

    # Bounding-box span for translation scaling
    finite_mask = np.isfinite(xy_all)
    if finite_mask.any():
        x_vals = xy_all[:, :, 0][finite_mask[:, :, 0]]
        y_vals = xy_all[:, :, 1][finite_mask[:, :, 1]]
        span_x = float(np.ptp(x_vals)) if x_vals.size else 1.0
        span_y = float(np.ptp(y_vals)) if y_vals.size else 1.0
    else:
        span_x = span_y = 1.0

    # Sample transform(s) ----------------------------------------
    if per_frame:
        thetas = rng.uniform(-np.radians(rotation_range_deg),
                             np.radians(rotation_range_deg), size=n_frames)
        tx = rng.uniform(-translation_range_frac * span_x,
                         translation_range_frac * span_x, size=n_frames)
        ty = rng.uniform(-translation_range_frac * span_y,
                         translation_range_frac * span_y, size=n_frames)
        scales = rng.uniform(1.0 - scale_range, 1.0 + scale_range, size=n_frames)
    else:
        theta_val = rng.uniform(-np.radians(rotation_range_deg),
                                np.radians(rotation_range_deg))
        thetas = np.full(n_frames, theta_val)
        tx_val = rng.uniform(-translation_range_frac * span_x,
                             translation_range_frac * span_x)
        ty_val = rng.uniform(-translation_range_frac * span_y,
                             translation_range_frac * span_y)
        tx = np.full(n_frames, tx_val)
        ty = np.full(n_frames, ty_val)
        scale_val = rng.uniform(1.0 - scale_range, 1.0 + scale_range)
        scales = np.full(n_frames, scale_val)

    # Apply per-frame: centre → scale → rotate → un-centre + translate
    for t in range(n_frames):
        if not np.isfinite(com[t]).all():
            continue
        R = _rotation_matrix(thetas[t])
        s = scales[t]
        centred = xy_all[t] - com[t]            # (P, 2)
        transformed = (centred * s) @ R.T        # scale then rotate
        xy_all[t] = transformed + com[t] + np.array([tx[t], ty[t]])

    # Write back --------------------------------------------------
    for i, (xc, yc) in enumerate(zip(x_cols, y_cols)):
        out[xc] = xy_all[:, i, 0]
        out[yc] = xy_all[:, i, 1]

    return out


# ---------------------------------------------------------------------------
# 2. Extended feature computation
# ---------------------------------------------------------------------------

_LAG_OFFSETS = [1, 2, 5]
_LAG_FEATURES = ["Summed Velocity", "Angular Velocity"]
_FFT_FEATURES = ["Summed Velocity", "Angular Velocity"]

# Pairs of (Right, Left) bodyparts that share an index for symmetry
_LR_PAIRS = [
    ("Head Right 1", "Head Left 1"),
    ("Right Body 1", "Left Body 1"),
    ("Right Body 2", "Left Body 2"),
    ("Right Body 3", "Left Body 3"),
]


def _rolling_fft_energy(signal: np.ndarray, fps: int, window: int,
                        low_hz: float, high_hz: float) -> np.ndarray:
    """Energy in [low_hz, high_hz] band from rolling FFT windows (vectorised)."""
    n = len(signal)
    if n < 4:
        return np.full(n, np.nan, dtype=float)

    energy = np.full(n, np.nan, dtype=float)
    half = window // 2

    # Pad signal for uniform-length windows at edges
    padded = np.pad(signal, (half, half), mode="edge")
    # Build strided view → (n, window) without copying
    from numpy.lib.stride_tricks import as_strided
    strides = (padded.strides[0], padded.strides[0])
    windows = as_strided(padded, shape=(n, window), strides=strides)

    # Check all-finite per window in one shot
    finite_mask = np.isfinite(windows).all(axis=1)  # (n,)
    valid_windows = windows[finite_mask]             # (m, window)
    if valid_windows.shape[0] == 0:
        return energy

    # Batch FFT: subtract mean per row, then rfft all at once
    centred = valid_windows - valid_windows.mean(axis=1, keepdims=True)
    fft_mag = np.abs(np.fft.rfft(centred, n=window, axis=1))  # (m, window//2+1)

    freqs = np.fft.rfftfreq(window, d=1.0 / fps)
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    energy[finite_mask] = np.sum(fft_mag[:, band_mask] ** 2, axis=1)
    return energy


def _rolling_dominant_freq(signal: np.ndarray, fps: int, window: int) -> np.ndarray:
    """Dominant frequency from rolling FFT windows (vectorised)."""
    n = len(signal)
    if n < 4:
        return np.full(n, np.nan, dtype=float)

    dom = np.full(n, np.nan, dtype=float)
    half = window // 2

    padded = np.pad(signal, (half, half), mode="edge")
    from numpy.lib.stride_tricks import as_strided
    strides = (padded.strides[0], padded.strides[0])
    windows = as_strided(padded, shape=(n, window), strides=strides)

    finite_mask = np.isfinite(windows).all(axis=1)
    valid_windows = windows[finite_mask]
    if valid_windows.shape[0] == 0:
        return dom

    centred = valid_windows - valid_windows.mean(axis=1, keepdims=True)
    fft_mag = np.abs(np.fft.rfft(centred, n=window, axis=1))  # (m, window//2+1)
    freqs = np.fft.rfftfreq(window, d=1.0 / fps)

    if len(freqs) > 1:
        # Skip DC (index 0), find peak frequency
        dom[finite_mask] = freqs[1:][np.argmax(fft_mag[:, 1:], axis=1)]
    return dom


def compute_extended_features(
    pose_df: pd.DataFrame,
    base_features: pd.DataFrame,
    fps: int = 25,
    window: int = 25,
) -> pd.DataFrame:
    """Compute additional features on top of the base kinematic features.

    Parameters
    ----------
    pose_df : DataFrame
        Raw DLC x/y coordinates (output of ``load_dlc_csv``).
    base_features : DataFrame
        Output of ``compute_kinematic_features``.
    fps : int
        Frames per second.
    window : int
        Rolling window for FFT features (default: 1 second = 25 frames).

    Returns
    -------
    DataFrame of *only* the new extended features (same row count as inputs).
    Caller should ``pd.concat([base_features, extended], axis=1)``.
    """
    n = len(base_features)
    ext: dict[str, np.ndarray] = {}

    # --- A.  Temporal context: lags, deltas, acceleration ----------
    for feat_name in _LAG_FEATURES:
        if feat_name not in base_features.columns:
            continue
        arr = base_features[feat_name].to_numpy(dtype=float)
        for lag in _LAG_OFFSETS:
            lagged = np.full(n, np.nan, dtype=float)
            lagged[lag:] = arr[:-lag] if lag < n else np.nan
            ext[f"{feat_name} Lag{lag}"] = lagged

    # Deltas (first derivative) for every velocity / distance feature
    vel_dist_cols = [c for c in base_features.columns
                     if ("Velocity" in c or "Distance" in c)
                     and not c.startswith("Mean ") and not c.startswith("SD ")]
    for col in vel_dist_cols:
        arr = base_features[col].to_numpy(dtype=float)
        delta = np.full(n, 0.0, dtype=float)
        delta[1:] = np.diff(arr)
        ext[f"Delta {col}"] = delta

    # Acceleration (second derivative) for velocity features only
    vel_cols = [c for c in base_features.columns
                if "Velocity" in c
                and not c.startswith("Mean ") and not c.startswith("SD ")
                and not c.startswith("Delta ")]
    for col in vel_cols:
        arr = base_features[col].to_numpy(dtype=float)
        accel = np.full(n, 0.0, dtype=float)
        if n > 2:
            accel[2:] = arr[2:] - 2 * arr[1:-1] + arr[:-2]
        ext[f"Accel {col}"] = accel

    # --- B.  Postural shape ----------------------------------------
    # Collect bodypart coordinates
    bodyparts = sorted(set(c.rsplit("_", 1)[0] for c in pose_df.columns))
    coords: dict[str, np.ndarray] = {}
    for bp in bodyparts:
        xc, yc = f"{bp}_x", f"{bp}_y"
        if xc in pose_df.columns and yc in pose_df.columns:
            coords[bp] = pose_df[[xc, yc]].to_numpy(dtype=float)

    # Body area via Shoelace formula on all available bodyparts (vectorised)
    if len(coords) >= 3:
        bp_keys = list(coords.keys())
        pts = np.stack([coords[k] for k in bp_keys], axis=1)  # (N, P, 2)
        # All-finite mask per bodypart per frame
        valid_bp = np.isfinite(pts).all(axis=2)  # (N, P)
        n_valid = valid_bp.sum(axis=1)            # (N,)
        area = np.full(n, np.nan, dtype=float)
        # Only compute for frames with ≥3 valid bodyparts
        computable = n_valid >= 3
        if computable.any():
            # For frames where ALL bodyparts are valid (fast path, usually all)
            all_valid = n_valid == len(bp_keys)
            if all_valid.all():
                # Fully vectorised: sort by angle, then Shoelace
                centroids = pts.mean(axis=1, keepdims=True)  # (N, 1, 2)
                angles = np.arctan2(pts[:, :, 1] - centroids[:, :, 1],
                                    pts[:, :, 0] - centroids[:, :, 0])  # (N, P)
                order = np.argsort(angles, axis=1)  # (N, P)
                row_idx = np.arange(n)[:, None]
                sorted_pts = pts[row_idx, order]  # (N, P, 2)
                x_s = sorted_pts[:, :, 0]  # (N, P)
                y_s = sorted_pts[:, :, 1]
                y_roll = np.roll(y_s, 1, axis=1)
                x_roll = np.roll(x_s, 1, axis=1)
                area = 0.5 * np.abs(
                    np.sum(x_s * y_roll, axis=1) - np.sum(y_s * x_roll, axis=1)
                )
            else:
                # Mixed: vectorise all-valid frames, loop only partial ones
                if all_valid.any():
                    av_pts = pts[all_valid]
                    centroids = av_pts.mean(axis=1, keepdims=True)
                    angles = np.arctan2(av_pts[:, :, 1] - centroids[:, :, 1],
                                        av_pts[:, :, 0] - centroids[:, :, 0])
                    order = np.argsort(angles, axis=1)
                    row_idx = np.arange(all_valid.sum())[:, None]
                    sorted_pts = av_pts[row_idx, order]
                    x_s = sorted_pts[:, :, 0]
                    y_s = sorted_pts[:, :, 1]
                    y_roll = np.roll(y_s, 1, axis=1)
                    x_roll = np.roll(x_s, 1, axis=1)
                    area[all_valid] = 0.5 * np.abs(
                        np.sum(x_s * y_roll, axis=1) - np.sum(y_s * x_roll, axis=1)
                    )
                # Loop only the partial frames
                partial = computable & ~all_valid
                for t in np.where(partial)[0]:
                    xy = pts[t][valid_bp[t]]
                    centroid = xy.mean(axis=0)
                    ang = np.arctan2(xy[:, 1] - centroid[1], xy[:, 0] - centroid[0])
                    o = np.argsort(ang)
                    xy_s = xy[o]
                    area[t] = 0.5 * abs(
                        np.dot(xy_s[:, 0], np.roll(xy_s[:, 1], 1))
                        - np.dot(xy_s[:, 1], np.roll(xy_s[:, 0], 1))
                    )
        ext["Body Area"] = area

    # Elongation ratio: nose-tail distance / mean lateral width
    nose_tail_col = None
    for col in base_features.columns:
        if "Nose" in col and "Tail" in col and "Distance" in col and "Mean" not in col and "SD" not in col:
            nose_tail_col = col
            break
    lateral_col = None
    for col in base_features.columns:
        if "Right Body 2" in col and "Left Body 2" in col and "Distance" in col and "Mean" not in col and "SD" not in col:
            lateral_col = col
            break
    if nose_tail_col is not None and lateral_col is not None:
        nt = base_features[nose_tail_col].to_numpy(dtype=float)
        lat = base_features[lateral_col].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            elong = np.where(lat > 1e-6, nt / lat, np.nan)
        ext["Elongation Ratio"] = elong

    # Limb symmetry indices
    for right_name, left_name in _LR_PAIRS:
        # Find the right velocity and left velocity columns
        r_vel = f"{right_name} Velocity"
        l_vel = f"{left_name} Velocity"
        if r_vel in base_features.columns and l_vel in base_features.columns:
            r = base_features[r_vel].to_numpy(dtype=float)
            l = base_features[l_vel].to_numpy(dtype=float)
            denom = r + l
            with np.errstate(divide="ignore", invalid="ignore"):
                sym = np.where(denom > 1e-6, np.abs(r - l) / denom, 0.0)
            short_name = right_name.replace("Right ", "").replace(" ", "")
            ext[f"Symmetry {short_name}"] = sym

    # Body curvature: angle at S1 formed by nose→S1→tail
    if "nose" in coords and "S1" in coords and "tail" in coords:
        nose_xy = coords["nose"]
        s1_xy = coords["S1"]
        tail_xy = coords["tail"]
        v1 = nose_xy - s1_xy   # S1 → nose
        v2 = tail_xy - s1_xy   # S1 → tail
        dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        curvature = np.arctan2(np.abs(cross), dot)
        ext["Body Curvature"] = curvature

    # --- C.  Frequency domain (only if enough frames) ---------------
    if n >= window:
        for feat_name in _FFT_FEATURES:
            if feat_name not in base_features.columns:
                continue
            sig = base_features[feat_name].to_numpy(dtype=float)
            sig_clean = np.nan_to_num(sig, nan=0.0)

            ext[f"{feat_name} FFT Low Energy"] = _rolling_fft_energy(
                sig_clean, fps, window, low_hz=0.0, high_hz=3.0,
            )
            ext[f"{feat_name} FFT High Energy"] = _rolling_fft_energy(
                sig_clean, fps, window, low_hz=3.0, high_hz=fps / 2.0,
            )
            ext[f"{feat_name} Dominant Freq"] = _rolling_dominant_freq(
                sig_clean, fps, window,
            )

    # --- D.  Rolling stats for new continuous features ---------------
    roll_window = 15
    for key in list(ext.keys()):
        series = pd.Series(ext[key])
        ext[f"Mean {key}"] = series.rolling(roll_window, center=True, min_periods=1).mean().to_numpy()
        ext[f"SD {key}"] = series.rolling(roll_window, center=True, min_periods=1).std().to_numpy()

    return pd.DataFrame(ext, index=base_features.index)


# ---------------------------------------------------------------------------
# 3. Global normalization (fit on training data, apply at inference)
# ---------------------------------------------------------------------------

def fit_global_normalizer(
    feature_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    clip_value: float = 4.0,
    eps: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Compute *global* robust normalization stats from the full training set.

    Unlike ``robust_normalize_features`` (per-recording), this computes a
    single median/IQR per feature across *all* recordings and applies it.

    Returns
    -------
    normalized_df : DataFrame
        Same shape as input, with normalised numeric columns.
    stats : dict
        ``{feature_name: {median, q25, q75, iqr, scale_used}}``.
    """
    cols = feature_cols or [c for c in feature_df.columns
                            if pd.api.types.is_numeric_dtype(feature_df[c])]
    out = feature_df.copy()
    stats: dict[str, dict[str, float]] = {}

    for col in cols:
        series = pd.to_numeric(feature_df[col], errors="coerce")
        values = series.to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            out[col] = 0.0
            stats[col] = {"median": 0.0, "q25": 0.0, "q75": 0.0,
                          "iqr": 0.0, "scale_used": float(eps)}
            continue
        med = float(np.nanmedian(values))
        q25 = float(np.nanquantile(values, 0.25))
        q75 = float(np.nanquantile(values, 0.75))
        iqr = q75 - q25
        scale = iqr if iqr > eps else eps
        norm = (series - med) / scale
        norm = norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[col] = np.clip(norm.to_numpy(dtype=float), -clip_value, clip_value)
        stats[col] = {"median": med, "q25": q25, "q75": q75,
                      "iqr": float(iqr), "scale_used": float(scale)}
    return out, stats


def apply_global_normalizer(
    feature_df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    clip_value: float = 4.0,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Apply pre-computed normalization stats to a new feature DataFrame.

    Features present in *feature_df* but absent from *stats* are left as-is
    (they will be dropped later by ``align_features_for_inference``).
    Features in *stats* but absent from *feature_df* are ignored.
    """
    out = feature_df.copy()
    for col, st in stats.items():
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        med = st["median"]
        scale = st.get("scale_used", st.get("iqr", eps))
        if scale < eps:
            scale = eps
        norm = (series - med) / scale
        norm = norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[col] = np.clip(norm.to_numpy(dtype=float), -clip_value, clip_value)
    return out


def save_normalization_stats(stats: dict[str, dict[str, float]], path: Path) -> None:
    """Persist normalization stats as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_normalization_stats(path: Path) -> dict[str, dict[str, float]]:
    """Load normalization stats from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def has_extended_features(feature_names: list[str]) -> bool:
    """Check whether a model's feature list includes extended features."""
    _MARKERS = {"Lag1", "Lag2", "Lag5", "Delta ", "Accel ", "Body Area",
                "Elongation Ratio", "Symmetry ", "Body Curvature",
                "FFT Low Energy", "FFT High Energy", "Dominant Freq"}
    for name in feature_names:
        for marker in _MARKERS:
            if marker in name:
                return True
    return False
