"""Numerical contract checks against the original 719-feature SHAP sample.

The archive is stronger evidence than comments or reconstructed notebooks: it
contains 2,638 feature rows from the archived SHAP analysis. These tests make
the preprocessing facts recoverable from those rows executable; they do not recover
the original train/test membership or every temporal input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freezing_dlc.behavior import (
    BEHAVIOR_BODYPARTS,
    BEHAVIOR_LABEL_ENCODER_PATH,
    BEHAVIOR_MODEL_PATH,
    TRAINING_BODY_LENGTH_PX,
    _compute_body_axis_rotation,
    build_behavior_feature_set,
    load_behavior_model,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = REPO_ROOT / "classifier" / "legacy_shap" / "shap_values.npz"


def _archive_frame() -> pd.DataFrame:
    if not ARCHIVE.is_file():
        pytest.skip("The manuscript training-feature archive is not bundled here.")
    archive = np.load(ARCHIVE, allow_pickle=True)
    return pd.DataFrame(archive["feature_values"], columns=archive["feature_names"])


def _wrapped_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left - right + np.pi) % (2.0 * np.pi) - np.pi


def test_model_schema_is_the_archived_719_feature_schema():
    frame = _archive_frame()
    if not BEHAVIOR_MODEL_PATH.is_file() or not BEHAVIOR_LABEL_ENCODER_PATH.is_file():
        pytest.skip("Install the authorized app model archive to run the model-schema check.")
    model = load_behavior_model(BEHAVIOR_MODEL_PATH, BEHAVIOR_LABEL_ENCODER_PATH)
    assert frame.shape == (2638, 719)
    assert list(frame.columns) == model.feature_names


def test_archived_coordinates_are_centered_but_not_rotated_or_scaled():
    frame = _archive_frame()
    x = frame[[f"{bp}_x" for bp in BEHAVIOR_BODYPARTS]].to_numpy(dtype=float)
    y = frame[[f"{bp}_y" for bp in BEHAVIOR_BODYPARTS]].to_numpy(dtype=float)

    assert np.max(np.abs(x.mean(axis=1))) < 6e-6
    assert np.max(np.abs(y.mean(axis=1))) < 7e-6

    orientation = np.arctan2(
        frame["nose_y"] - frame["tail_y"],
        frame["nose_x"] - frame["tail_x"],
    ).to_numpy()
    assert np.max(np.abs(_wrapped_delta(orientation, frame["orientation_angle"].to_numpy()))) < 1e-6
    # A tail-to-nose rotation would make this concentration approximately 1.
    assert abs(np.mean(np.exp(1j * orientation))) < 0.25

    body_length = np.hypot(
        frame["nose_x"] - frame["tail_x"],
        frame["nose_y"] - frame["tail_y"],
    )
    assert np.std(body_length) > 20.0
    assert float(np.median(body_length)) == pytest.approx(TRAINING_BODY_LENGTH_PX, abs=1.0)


def test_archived_geometry_and_pca_angle_match_the_feature_formulas():
    frame = _archive_frame()
    dist = np.hypot(
        frame["B1L_x"] - frame["B1R_x"],
        frame["B1L_y"] - frame["B1R_y"],
    )
    assert np.max(np.abs(dist - frame["dist_B1L_B1R"])) < 2e-5

    points = np.stack(
        [frame[[f"{bp}_x", f"{bp}_y"]].to_numpy(dtype=np.float32) for bp in BEHAVIOR_BODYPARTS],
        axis=1,
    )
    pca_angle = _compute_body_axis_rotation(points)
    assert np.max(np.abs(pca_angle - frame["global_body_angle"].to_numpy())) < 1e-6


def test_archived_lags_reconstruct_full_window_position_statistics():
    frame = _archive_frame()
    values = np.column_stack(
        [frame["B1L_x"], *[frame[f"B1L_x_t-{lag}"] for lag in range(1, 5)]]
    )
    # Sequence starts have fewer than five observations.  In a full window the
    # archived sum is exactly five times its mean, which identifies rows where
    # all four stored lags belong to the rolling calculation.
    complete = np.isclose(
        frame["B1L_x_roll_sum"],
        5.0 * frame["B1L_x_roll_mean"],
        atol=1e-4,
    )
    assert complete.mean() > 0.99
    np.testing.assert_allclose(
        values[complete].mean(axis=1),
        frame.loc[complete, "B1L_x_roll_mean"],
        atol=2e-5,
    )
    np.testing.assert_allclose(
        values[complete].sum(axis=1),
        frame.loc[complete, "B1L_x_roll_sum"],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        values[complete].std(axis=1, ddof=1),
        frame.loc[complete, "B1L_x_roll_std"],
        atol=2e-5,
    )


def test_default_builder_uses_training_frame_not_experimental_rotation():
    n = 12
    data: dict[str, np.ndarray] = {}
    angle = np.deg2rad(37.0)
    direction = np.array([np.cos(angle), np.sin(angle)])
    perpendicular = np.array([-direction[1], direction[0]])
    for index, bp in enumerate(BEHAVIOR_BODYPARTS):
        point = 300.0 + direction * (index * 4.0) + perpendicular * ((index % 3) * 2.0)
        data[f"{bp}_x"] = point[0] + np.arange(n)
        data[f"{bp}_y"] = np.full(n, point[1])
    data["tail_x"] = np.full(n, 300.0)
    data["tail_y"] = np.full(n, 300.0)
    data["nose_x"] = 300.0 + 100.0 * direction[0] + np.arange(n)
    data["nose_y"] = np.full(n, 300.0 + 100.0 * direction[1])

    features = build_behavior_feature_set(pd.DataFrame(data))
    assert features["orientation_angle"].iloc[0] == pytest.approx(angle, abs=1e-5)
    assert features["dist_nose_tail"].iloc[0] == pytest.approx(100.0, abs=1e-5)
    assert abs(features["centroid_x"].iloc[0]) < 1e-5
    assert abs(features["centroid_y"].iloc[0]) < 1e-5
