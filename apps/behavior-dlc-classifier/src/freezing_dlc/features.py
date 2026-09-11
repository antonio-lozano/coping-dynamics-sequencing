from __future__ import annotations

from os import PathLike
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

FEATURE_SET_VERSION = "2026-08-stillness-v1"

HEAD_BODY_PARTS = ("nose", "H1R", "H2R", "H1L", "H2L")
HEAD_LEFT_PARTS = ("H1L", "H2L")
HEAD_RIGHT_PARTS = ("H1R", "H2R")

#: Windows, in frames, over which motion is summarised for the ``stillness`` set.
#: At 25 fps these span 0.2 s to 20 s. Freezing is immobility *sustained* for on
#: the order of a second, so no single frame's displacement can express it. The
#: bundled recordings have median bouts of 12, 26 and 66 frames, and different
#: laboratories draw the line at different durations, so the range is left wide
#: and the forest settles on whichever timescale matches the scoring in use.
STILLNESS_WINDOWS = (5, 15, 31, 61, 121, 251, 501)

#: Multiples of a recording's own motion floor that count as "still". Three levels
#: rather than one, because where tracking noise ends and real movement begins is
#: not the same in every recording.
STILLNESS_MULTIPLES = (1.5, 2.0, 4.0)

#: Percentile of frame-to-frame displacement taken as a recording's motion floor.
#: Low, because the floor is meant to describe the stillest moments. Measured at
#: the 25th percentile instead, cross-video agreement on the demo videos falls by
#: roughly a third, since on an animal that freezes rarely the 25th percentile
#: already sits well inside real movement.
MOTION_FLOOR_PERCENTILE = 5.0


def dlc_filtered_csv_to_flat_df(path: Union[str, PathLike[str]]) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    df.columns = [f"{bp}_{c}" for _, bp, c in df.columns]
    return df


def _columns_ending(df_flat: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Select the columns for one coordinate, matching the suffix rather than any part of the name.

    A substring test pulls in the wrong columns as soon as a body part is named
    something like ``paw_x1``, because ``paw_x1_y`` also contains ``_x``. That
    would silently pair x values with y values of a different body part.
    """
    return df_flat.loc[:, [c for c in df_flat.columns if str(c).endswith(suffix)]]


def freezing_feature_set(df_flat: pd.DataFrame) -> pd.DataFrame:
    """The ``compact`` feature set: eight numbers describing each frame on its own.

    Superseded by :func:`stillness_feature_set` and kept unchanged so that models
    trained against it still load. Do not adjust these definitions: a model file
    records only the *names* of its features, so altering what a name means here
    would silently change what an existing model is being asked to predict from.
    """
    x_df = _columns_ending(df_flat, "_x")
    y_df = _columns_ending(df_flat, "_y")
    if x_df.shape[1] == 0 or y_df.shape[1] == 0:
        raise ValueError("No *_x/*_y columns found.")

    x = x_df.to_numpy(dtype=float)
    y = y_df.to_numpy(dtype=float)

    dx = np.diff(x, axis=0, prepend=np.nan)
    dy = np.diff(y, axis=0, prepend=np.nan)
    vel = np.sqrt(dx**2 + dy**2)

    vel_sum = np.nansum(vel, axis=1)
    vel_count = np.sum(~np.isnan(vel), axis=1)
    velocity = np.divide(vel_sum, vel_count, out=np.zeros_like(vel_sum), where=vel_count > 0)

    velocity_std = np.nan_to_num(vel, nan=0.0).std(axis=1)
    acceleration = np.abs(np.diff(velocity, prepend=np.nan))
    x_spread = np.nanstd(x, axis=1)
    y_spread = np.nanstd(y, axis=1)

    vel_series = pd.Series(velocity)
    velocity_roll3 = vel_series.rolling(window=3, min_periods=1).mean().to_numpy()
    velocity_roll5 = vel_series.rolling(window=5, min_periods=1).mean().to_numpy()

    lik_df = _columns_ending(df_flat, "_likelihood")
    if lik_df.shape[1] > 0:
        likelihood_mean = lik_df.to_numpy(dtype=float).mean(axis=1)
    else:
        likelihood_mean = np.ones(len(df_flat), dtype=float)

    out = pd.DataFrame(
        {
            "velocity": velocity,
            "velocity_std": velocity_std,
            "acceleration": acceleration,
            "x_spread": x_spread,
            "y_spread": y_spread,
            "velocity_roll3": velocity_roll3,
            "velocity_roll5": velocity_roll5,
            "likelihood_mean": likelihood_mean,
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _velocity_and_spread(df_flat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Per frame, how far the tracked points moved and how much they disagreed.

    Body parts that were not tracked on a frame are left out of both figures
    rather than counted as motionless, which is what filling their gaps with
    zeros would do.
    """
    x = _columns_ending(df_flat, "_x").to_numpy(dtype=float)
    y = _columns_ending(df_flat, "_y").to_numpy(dtype=float)
    if x.size == 0 or y.size == 0:
        raise ValueError("No *_x/*_y columns found.")

    dx = np.diff(x, axis=0, prepend=np.nan)
    dy = np.diff(y, axis=0, prepend=np.nan)
    step = np.sqrt(dx**2 + dy**2)

    counts = np.sum(~np.isnan(step), axis=1)
    total = np.nansum(step, axis=1)
    mean = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)
    deviation = np.where(np.isnan(step), 0.0, step - mean[:, None])
    variance = np.divide(
        (deviation**2).sum(axis=1), counts, out=np.zeros_like(total), where=counts > 0
    )
    return mean, np.sqrt(variance)


def stillness_feature_set(df_flat: pd.DataFrame) -> pd.DataFrame:
    """Multi-scale description of how still the animal is, and for how long.

    The ``compact`` set describes single frames, with rolling means over three and
    five frames as its only memory. Freezing is not a property of a frame, so that
    set cannot represent it directly and the forest has to infer duration from
    instantaneous speed. This one measures stillness over windows from 0.2 s to
    20 s instead, and adds, for each window, the fraction of it the animal spent
    below its own motion floor.

    That floor is what makes the features carry across recordings. Frame-to-frame
    displacement is not comparable between videos: across the six bundled
    recordings its median spans two orders of magnitude, 0.012 to 1.4 pixels,
    while the animals themselves are all about 50 pixels long. The difference is
    tracking jitter, not behaviour, and a threshold learned on one recording is
    meaningless on another. Expressing stillness relative to each recording's own
    floor removes that offset. Leave-one-video-out agreement with the manual
    scores rises from 0.30 to 0.47 (Matthews correlation) against the ``compact``
    set, and improves on every one of the three labelled videos.

    The floor is estimated from the whole recording, so a recording must be
    featurised in one piece, which is how the pipeline processes it. It also
    assumes the recording contains some still frames. On an animal that never
    stops moving the floor overestimates stillness, which pushes the model
    towards over-calling freezing rather than under-calling it.
    """
    velocity, spread = _velocity_and_spread(df_flat)
    floor = float(np.percentile(velocity, MOTION_FLOOR_PERCENTILE)) + 1e-6

    velocity_series = pd.Series(velocity)
    spread_series = pd.Series(spread)
    out: Dict[str, np.ndarray] = {}

    for window in STILLNESS_WINDOWS:
        roll = velocity_series.rolling(window, min_periods=1, center=True)
        out[f"velocity_mean{window}"] = roll.mean().to_numpy()
        out[f"velocity_sd{window}"] = roll.std().fillna(0.0).to_numpy()
        out[f"velocity_min{window}"] = roll.min().to_numpy()
        out[f"velocity_max{window}"] = roll.max().to_numpy()
        out[f"spread_mean{window}"] = (
            spread_series.rolling(window, min_periods=1, center=True).mean().to_numpy()
        )

    for multiple in STILLNESS_MULTIPLES:
        still = pd.Series((velocity <= multiple * floor).astype(float))
        for window in STILLNESS_WINDOWS:
            out[f"still{multiple:g}_frac{window}"] = (
                still.rolling(window, min_periods=1, center=True).mean().to_numpy()
            )

    out["velocity_spread"] = spread

    # Taken from the compact set rather than recomputed, so the two modes cannot
    # end up with two meanings for one column name.
    base = freezing_feature_set(df_flat)
    for name in ("acceleration", "x_spread", "y_spread", "likelihood_mean"):
        out[name] = base[name].to_numpy()

    frame = pd.DataFrame(out)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _bodyparts_from_flat(df_flat: pd.DataFrame) -> List[str]:
    bps: List[str] = []
    for c in df_flat.columns:
        if c.endswith("_x"):
            bp = c[: -len("_x")]
            if f"{bp}_y" in df_flat.columns:
                bps.append(bp)
    if not bps:
        raise ValueError("No paired *_x/*_y bodypart columns found.")
    return bps


def _safe_angle_deg(dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(dy, dx))


def _angle_delta_deg(angle_deg: np.ndarray) -> np.ndarray:
    wrapped = (np.diff(angle_deg, prepend=angle_deg[0]) + 180.0) % 360.0 - 180.0
    wrapped[0] = 0.0
    return wrapped


def _mean_point(x: Dict[str, np.ndarray], y: Dict[str, np.ndarray], names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.vstack([x[name] for name in names])
    ys = np.vstack([y[name] for name in names])
    return np.nanmean(xs, axis=0), np.nanmean(ys, axis=0)


def _add_head_features(core: Dict[str, np.ndarray], x: Dict[str, np.ndarray], y: Dict[str, np.ndarray], bps: List[str]) -> None:
    head_parts = [bp for bp in HEAD_BODY_PARTS if bp in bps]
    left_parts = [bp for bp in HEAD_LEFT_PARTS if bp in bps]
    right_parts = [bp for bp in HEAD_RIGHT_PARTS if bp in bps]
    if len(head_parts) < 3 or "nose" not in head_parts:
        return

    head_cx, head_cy = _mean_point(x, y, head_parts)
    left_cx, left_cy = _mean_point(x, y, left_parts if left_parts else head_parts)
    right_cx, right_cy = _mean_point(x, y, right_parts if right_parts else head_parts)

    nose_x = x["nose"]
    nose_y = y["nose"]
    head_dx = np.diff(head_cx, prepend=np.nan)
    head_dy = np.diff(head_cy, prepend=np.nan)
    head_speed = np.sqrt(head_dx**2 + head_dy**2)

    nose_dx = np.diff(nose_x, prepend=np.nan)
    nose_dy = np.diff(nose_y, prepend=np.nan)
    nose_speed = np.sqrt(nose_dx**2 + nose_dy**2)

    snout_dx = nose_x - head_cx
    snout_dy = nose_y - head_cy
    snout_dist = np.sqrt(snout_dx**2 + snout_dy**2)
    head_heading = _safe_angle_deg(snout_dy, snout_dx)
    head_heading_delta = _angle_delta_deg(head_heading)

    head_width_dx = left_cx - right_cx
    head_width_dy = left_cy - right_cy
    head_width = np.sqrt(head_width_dx**2 + head_width_dy**2)
    head_width_angle = _safe_angle_deg(head_width_dy, head_width_dx)
    head_width_angle_delta = _angle_delta_deg(head_width_angle)

    head_relative_speed = np.sqrt((nose_dx - head_dx) ** 2 + (nose_dy - head_dy) ** 2)
    head_pose_shift = np.sqrt((np.diff(snout_dx, prepend=np.nan)) ** 2 + (np.diff(snout_dy, prepend=np.nan)) ** 2)

    nose_left_dist = np.sqrt((nose_x - left_cx) ** 2 + (nose_y - left_cy) ** 2)
    nose_right_dist = np.sqrt((nose_x - right_cx) ** 2 + (nose_y - right_cy) ** 2)
    head_asymmetry = nose_left_dist - nose_right_dist
    head_compactness = np.divide(
        snout_dist,
        head_width,
        out=np.zeros_like(snout_dist),
        where=head_width > 1e-6,
    )

    core["head_cx"] = head_cx
    core["head_cy"] = head_cy
    core["head_speed"] = head_speed
    core["nose_speed"] = nose_speed
    core["head_relative_speed"] = head_relative_speed
    core["head_pose_shift"] = head_pose_shift
    core["snout_dist"] = snout_dist
    core["head_heading"] = head_heading
    core["head_heading_delta"] = head_heading_delta
    core["head_width"] = head_width
    core["head_width_angle"] = head_width_angle
    core["head_width_angle_delta"] = head_width_angle_delta
    core["nose_left_dist"] = nose_left_dist
    core["nose_right_dist"] = nose_right_dist
    core["head_asymmetry"] = head_asymmetry
    core["head_compactness"] = head_compactness


def legacy_4k_feature_set(df_flat: pd.DataFrame, target_features: int = 4046) -> pd.DataFrame:
    """High-dimensional legacy-style feature set (~4k features).

    Designed to provide rich kinematic + temporal descriptors similar in scale
    to older workflows.
    """
    bps = _bodyparts_from_flat(df_flat)
    n = len(df_flat)

    x: Dict[str, np.ndarray] = {bp: df_flat[f"{bp}_x"].to_numpy(dtype=float) for bp in bps}
    y: Dict[str, np.ndarray] = {bp: df_flat[f"{bp}_y"].to_numpy(dtype=float) for bp in bps}

    core: Dict[str, np.ndarray] = {}

    # Base kinematics per bodypart.
    speed_cols: List[str] = []
    for bp in bps:
        dx = np.diff(x[bp], prepend=np.nan)
        dy = np.diff(y[bp], prepend=np.nan)
        speed = np.sqrt(dx**2 + dy**2)
        acc = np.abs(np.diff(np.nan_to_num(speed, nan=0.0), prepend=0.0))

        core[f"{bp}_x"] = x[bp]
        core[f"{bp}_y"] = y[bp]
        core[f"{bp}_dx"] = dx
        core[f"{bp}_dy"] = dy
        core[f"{bp}_speed"] = speed
        core[f"{bp}_acc"] = acc
        speed_cols.append(f"{bp}_speed")

    # Pairwise distances.
    for i, b1 in enumerate(bps):
        for b2 in bps[i + 1 :]:
            d = np.sqrt((x[b1] - x[b2]) ** 2 + (y[b1] - y[b2]) ** 2)
            core[f"dist_{b1}_{b2}"] = d

    # Segment angles across ordered bodyparts.
    for i in range(len(bps) - 1):
        b1, b2 = bps[i], bps[i + 1]
        ang = np.degrees(np.arctan2(y[b2] - y[b1], x[b2] - x[b1]))
        core[f"angle_{b1}_{b2}"] = ang

    # Head-centric descriptors help distinguish subtle head movement from true freezing.
    _add_head_features(core, x, y, bps)

    core_df = pd.DataFrame(core).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    parts: List[pd.DataFrame] = [core_df]

    windows = [2, 3, 5, 7, 10, 12, 15, 20, 30, 45]
    for w in windows:
        roll = core_df.rolling(window=w, min_periods=1)
        parts.append(roll.mean().add_suffix(f"_m{w}"))
        parts.append(roll.std(ddof=0).fillna(0.0).add_suffix(f"_sd{w}"))

    # Speed lags (7 lags) to enrich short-term temporal context.
    speed_df = pd.DataFrame(core_df[speed_cols]) if speed_cols else pd.DataFrame(index=core_df.index)
    for lag in range(1, 8):
        lag_df = pd.DataFrame(speed_df.shift(lag).fillna(0.0).add_suffix(f"_lag{lag}"))
        parts.append(lag_df)

    out = pd.DataFrame(pd.concat(parts, axis=1))
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Keep a deterministic width to preserve training/inference compatibility.
    if out.shape[1] < target_features:
        pad_count = target_features - out.shape[1]
        pad = pd.DataFrame(
            np.zeros((n, pad_count), dtype=float),
            columns=[f"zz_pad_{i:04d}" for i in range(pad_count)],
            index=out.index,
        )
        out = pd.concat([out, pad], axis=1)
    elif out.shape[1] > target_features:
        out = out.iloc[:, :target_features]

    return pd.DataFrame(out)


#: Every feature mode, in the one place both the command line and the trainer read.
FEATURE_MODES = ("stillness", "compact", "legacy4k")


def build_feature_set(df_flat: pd.DataFrame, mode: str = "stillness") -> pd.DataFrame:
    mode_norm = mode.lower().strip()
    if mode_norm in {"stillness", "default"}:
        return stillness_feature_set(df_flat)
    if mode_norm == "compact":
        return freezing_feature_set(df_flat)
    if mode_norm in {"legacy4k", "legacy", "4k", "highdim"}:
        return legacy_4k_feature_set(df_flat)
    raise ValueError(f"Unknown feature mode: {mode!r}. Choose one of {', '.join(FEATURE_MODES)}.")
