"""Feature engineering for per-frame behavior classification."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

BASE_FEATURES: tuple[str, ...] = (
    "centroid_x",
    "centroid_y",
    "heading",
    "angular_velocity",
    "velocity_px_s",
)
OPTIONAL_SHAP_FEATURES: tuple[str, ...] = (
    "body_length",
    "body_tilt",
    "body_tilt_delta",
    "body_tilt_abs_delta",
    "body_tilt_variability",
    "global_turning_speed",
)
ROLLING_FEATURES: tuple[str, ...] = ("velocity_px_s", "angular_velocity", "abs_ang")
OPTIONAL_ROLLING_FEATURES: tuple[str, ...] = (
    "body_tilt_delta",
    "body_tilt_abs_delta",
    "global_turning_speed",
)
LAG_FEATURES: tuple[str, ...] = ("velocity_px_s", "angular_velocity", "heading")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def add_behavior_labels(
    df: pd.DataFrame,
    syllable_to_behavior: dict[int, str],
    *,
    unassigned_label: str = "Unassigned",
    label_col: str = "behavior_label",
) -> pd.DataFrame:
    """Attach behavior labels from hand-curated syllable annotations."""

    require_columns(df, ["syllable"])
    out = df.copy()
    syllables = pd.to_numeric(out["syllable"], errors="coerce").astype("Int64")
    out[label_col] = syllables.map(syllable_to_behavior).fillna(unassigned_label)
    return out


def build_frame_features(
    df: pd.DataFrame,
    *,
    group_col: str = "name",
    frame_col: str = "frame_index",
    windows: Sequence[int] = (5, 15, 30),
    lags: Sequence[int] = (1, 5, 15),
) -> tuple[pd.DataFrame, list[str]]:
    """Create movement features matching the Figure 7A-C XGBoost checks."""

    require_columns(df, [group_col, frame_col, *BASE_FEATURES])
    out = df.copy()
    out = out.sort_values([group_col, frame_col]).reset_index(drop=True)

    for column in BASE_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["abs_ang"] = out["angular_velocity"].abs()
    feature_cols = list(BASE_FEATURES) + ["abs_ang"]
    for column in OPTIONAL_SHAP_FEATURES:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
            feature_cols.append(column)
    grouped = out.groupby(group_col, sort=False)

    for window in windows:
        rolling_cols = list(ROLLING_FEATURES) + [
            column for column in OPTIONAL_ROLLING_FEATURES if column in out.columns
        ]
        for column in rolling_cols:
            mean_col = f"{column}_rmean{window}"
            std_col = f"{column}_rstd{window}"
            out[mean_col] = grouped[column].transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
            out[std_col] = grouped[column].transform(
                lambda s: s.rolling(window, min_periods=1).std()
            )
            feature_cols.extend([mean_col, std_col])

    for lag in lags:
        for column in LAG_FEATURES:
            lag_col = f"{column}_lag{lag}"
            out[lag_col] = grouped[column].shift(lag)
            feature_cols.append(lag_col)

    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, feature_cols
