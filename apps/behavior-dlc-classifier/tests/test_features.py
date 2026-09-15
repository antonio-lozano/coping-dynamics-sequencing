import pandas as pd

import numpy as np

from freezing_dlc.features import (
    STILLNESS_MULTIPLES,
    STILLNESS_WINDOWS,
    build_feature_set,
    freezing_feature_set,
    legacy_4k_feature_set,
    stillness_feature_set,
)


import pytest


def _walk(n: int = 400, scale: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """A recording that alternates between holding still and moving.

    While the animal holds still the points still jitter, around a roughly fixed
    noise level rather than anywhere between zero and it, which is how tracking
    error actually behaves and what the motion floor is meant to estimate.
    """
    rng = np.random.default_rng(seed)
    moving = (np.arange(n) // 50) % 2 == 1
    step = np.where(moving, 4.0, 0.02) * (0.7 + 0.6 * rng.random(n))
    x = np.cumsum(step) * scale
    y = np.cumsum(step * 0.5) * scale
    return pd.DataFrame(
        {"nose_x": x, "nose_y": y, "tail_x": x + 50 * scale, "tail_y": y, "nose_likelihood": np.full(n, 0.9)}
    )


def test_stillness_features_survive_a_hundredfold_change_of_scale():
    """The property the feature set exists for.

    Frame-to-frame displacement is not comparable between recordings: across the
    bundled videos its median spans two orders of magnitude at identical animal
    size. Measuring stillness against each recording's own motion floor is what
    makes a threshold learned on one recording mean anything on another, so the
    stillness columns must not move when only the pixel scale does.
    """
    small = stillness_feature_set(_walk(scale=1.0))
    large = stillness_feature_set(_walk(scale=100.0))

    still_columns = [c for c in small.columns if c.startswith("still")]
    assert still_columns, "expected stillness fraction columns"
    for column in still_columns:
        assert np.allclose(small[column], large[column], atol=1e-9), column

    # The raw motion columns are meant to scale, which is why they cannot be the
    # only thing the classifier sees.
    assert large["velocity_mean5"].mean() > 50 * small["velocity_mean5"].mean()


def test_stillness_feature_set_is_finite_and_named_by_window():
    feat = stillness_feature_set(_walk())

    expected = 5 * len(STILLNESS_WINDOWS) + len(STILLNESS_MULTIPLES) * len(STILLNESS_WINDOWS) + 5
    assert feat.shape == (400, expected)
    assert np.isfinite(feat.to_numpy(dtype=float)).all()
    for window in STILLNESS_WINDOWS:
        assert f"velocity_mean{window}" in feat.columns


def test_stillness_tracks_the_animal_actually_holding_still():
    feat = stillness_feature_set(_walk())
    # A 15 frame window against 50 frame blocks, so only the frames either side
    # of a transition see both states.
    still_fraction = feat["still2_frac15"].to_numpy()
    moving = (np.arange(400) // 50) % 2 == 1

    assert still_fraction[~moving].mean() > 0.8
    assert still_fraction[moving].mean() < 0.2


def test_build_feature_set_names_the_modes_it_accepts():
    with pytest.raises(ValueError, match="stillness"):
        build_feature_set(_walk(), mode="nonsense")


def test_coordinate_columns_are_matched_by_suffix_not_substring():
    """A body part named ``paw_x1`` makes ``paw_x1_y`` contain ``_x``."""
    from freezing_dlc.features import _columns_ending

    df = pd.DataFrame(
        {
            "paw_x1_x": [0.0, 1.0],
            "paw_x1_y": [0.0, 1.0],
            "paw_x1_likelihood": [1.0, 1.0],
            "nose_x": [0.0, 2.0],
            "nose_y": [0.0, 2.0],
            "nose_likelihood": [1.0, 1.0],
        }
    )

    assert list(_columns_ending(df, "_x").columns) == ["paw_x1_x", "nose_x"]
    assert list(_columns_ending(df, "_y").columns) == ["paw_x1_y", "nose_y"]
    # The substring form also matches paw_x1_y and paw_x1_likelihood, giving four
    # columns where two are meant, and so pairing one body part's x with a y that
    # is not its own.
    assert len(df.filter(like="_x").columns) == 4


def test_feature_set_survives_a_body_part_named_like_a_coordinate():
    from freezing_dlc.features import freezing_feature_set

    df = pd.DataFrame(
        {
            "paw_x1_x": [0.0, 1.0, 2.0],
            "paw_x1_y": [0.0, 1.0, 2.0],
            "nose_x": [0.0, 2.0, 4.0],
            "nose_y": [0.0, 2.0, 4.0],
        }
    )

    feat = freezing_feature_set(df)

    assert len(feat) == 3
    assert feat["velocity"].tolist() == pytest.approx([0.0, (2**0.5 + 8**0.5) / 2, (2**0.5 + 8**0.5) / 2])


def test_freezing_feature_set_contains_expected_columns():
    df = pd.DataFrame({
        "nose_x": [0, 1, 2],
        "nose_y": [0, 0, 0],
        "tail_x": [0, 0, 0],
        "tail_y": [0, 1, 2],
        "nose_likelihood": [0.9, 0.8, 0.95],
        "tail_likelihood": [0.9, 0.85, 0.9],
    })
    feat = freezing_feature_set(df)
    for col in ["velocity", "velocity_std", "acceleration", "velocity_roll3", "likelihood_mean"]:
        assert col in feat.columns
    assert len(feat) == 3


def test_legacy_4k_feature_set_has_expected_width():
    bps = ["nose", "H1R", "H2R", "H1L", "H2L", "B1R", "B2R", "B3R", "B1L", "B2L", "B3L", "tail", "S2", "S1"]
    rows = 8
    data = {}
    for i, bp in enumerate(bps):
        data[f"{bp}_x"] = [float(j + i) for j in range(rows)]
        data[f"{bp}_y"] = [float(j - i) for j in range(rows)]
        data[f"{bp}_likelihood"] = [0.9 for _ in range(rows)]

    df = pd.DataFrame(data)
    feat = legacy_4k_feature_set(df)
    assert feat.shape == (rows, 4046)
    assert feat.isna().sum().sum() == 0
    for col in ["head_speed", "nose_speed", "head_relative_speed", "head_heading_delta", "head_compactness"]:
        assert col in feat.columns


def test_build_feature_set_legacy_alias():
    df = pd.DataFrame({"nose_x": [0.0, 1.0], "nose_y": [0.0, 1.0]})
    feat = build_feature_set(df, mode="legacy4k")
    assert feat.shape[1] == 4046
