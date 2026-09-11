"""Training row selection must not redefine physical temporal neighbours."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from coping_dynamics.behavior_classifier import pipeline


@pytest.fixture
def recordings(monkeypatch):
    # Feature preparation needs only encoding, not the optional XGBoost runtime.
    monkeypatch.setattr(
        pipeline, "_classifier_dependencies", lambda: {"LabelEncoder": LabelEncoder}
    )
    rows = []
    for name, offset in (("recording_a", 0), ("recording_b", 1000)):
        for frame in range(60):
            value = offset + frame
            rows.append(
                {
                    "name": name,
                    "frame_index": frame,
                    "syllable": 999 if frame % 3 == 1 else (0 if frame % 2 else 18),
                    "centroid_x": value,
                    "centroid_y": value / 2,
                    "heading": value / 10,
                    "angular_velocity": value / 100,
                    "velocity_px_s": value,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("include_unassigned,max_frames", [(True, 20), (False, None), (False, 20)])
def test_selected_rows_keep_recording_lags_and_windows(recordings, include_unassigned, max_frames):
    featured, _, y, encoder = pipeline._prepare_training_data(
        recordings, include_unassigned=include_unassigned, max_frames=max_frames
    )
    if max_frames is not None:
        assert len(featured) == max_frames
    if not include_unassigned:
        assert "Unassigned" not in set(featured.behavior_label)
    np.testing.assert_array_equal(encoder.inverse_transform(y), featured.behavior_label)
    for row in featured.itertuples():
        offset = 0 if row.name == "recording_a" else 1000
        expected_lag = offset + row.frame_index - 1 if row.frame_index else 0
        assert row.velocity_px_s_lag1 == expected_lag
        preceding = np.arange(max(0, row.frame_index - 4), row.frame_index + 1) + offset
        assert row.velocity_px_s_rmean5 == pytest.approx(preceding.mean())


def test_sampling_is_reproducible_and_input_is_not_mutated(recordings):
    original = recordings.copy(deep=True)
    first = pipeline._prepare_training_data(recordings, include_unassigned=True, max_frames=20)[0]
    second = pipeline._prepare_training_data(recordings, include_unassigned=True, max_frames=20)[0]
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(recordings, original)
