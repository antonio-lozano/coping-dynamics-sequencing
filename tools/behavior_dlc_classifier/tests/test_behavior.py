import numpy as np
import pandas as pd
import pytest

from freezing_dlc.behavior import (
    BEHAVIOR_BODYPARTS,
    TRAINING_BODY_LENGTH_PX,
    UNASSIGNED,
    build_behavior_feature_set,
    confidence_for_labels,
    egocentric_align,
    refine_behavior_labels,
    training_coordinate_frame,
)


def _synthetic_flat(n=100, speed=4.0, likelihood=1.0):
    """A perfectly tracked animal walking to the right at a constant speed."""
    data = {}
    for i, bp in enumerate(BEHAVIOR_BODYPARTS):
        data[f"{bp}_x"] = 100.0 + np.arange(n) * speed + i * 3.0
        data[f"{bp}_y"] = 100.0 + i * 3.0
        data[f"{bp}_likelihood"] = np.full(n, likelihood)
    # Put the nose ahead of the tail so the heading is well defined.
    data["nose_x"] = data["tail_x"] + 40.0
    return pd.DataFrame(data)


def test_training_coordinate_frame_centres_without_rotating_or_scaling():
    flat = _synthetic_flat()
    centred = training_coordinate_frame(flat)
    xs = np.column_stack([centred[f"{bp}_x"] for bp in BEHAVIOR_BODYPARTS])
    ys = np.column_stack([centred[f"{bp}_y"] for bp in BEHAVIOR_BODYPARTS])

    assert np.allclose(xs.mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(ys.mean(axis=1), 0.0, atol=1e-6)
    before_angle = np.arctan2(
        flat["nose_y"] - flat["tail_y"], flat["nose_x"] - flat["tail_x"]
    )
    after_angle = np.arctan2(
        centred["nose_y"] - centred["tail_y"],
        centred["nose_x"] - centred["tail_x"],
    )
    assert np.allclose(before_angle, after_angle, atol=1e-6)
    before_length = np.hypot(
        flat["nose_x"] - flat["tail_x"], flat["nose_y"] - flat["tail_y"]
    )
    after_length = np.hypot(
        centred["nose_x"] - centred["tail_x"],
        centred["nose_y"] - centred["tail_y"],
    )
    assert np.allclose(before_length, after_length, atol=1e-6)


def test_experimental_egocentric_align_centres_and_rotates():
    aligned = egocentric_align(_synthetic_flat())
    xs = np.column_stack([aligned[f"{bp}_x"] for bp in BEHAVIOR_BODYPARTS])
    ys = np.column_stack([aligned[f"{bp}_y"] for bp in BEHAVIOR_BODYPARTS])

    assert np.allclose(xs.mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(ys.mean(axis=1), 0.0, atol=1e-6)
    # The tail-to-nose axis ends up on +x, so the nose is ahead and level with the tail.
    assert np.allclose(aligned["nose_y"], aligned["tail_y"], atol=1e-6)
    assert (aligned["nose_x"] > aligned["tail_x"]).all()


def test_experimental_egocentric_align_preserves_shape():
    """Alignment may rescale the animal, but never distorts it."""
    flat = _synthetic_flat()
    aligned = egocentric_align(flat)
    ratios = []
    for a, b in (("nose", "tail"), ("B1L", "H2R"), ("S1", "H1L")):
        before = np.hypot(flat[f"{a}_x"] - flat[f"{b}_x"], flat[f"{a}_y"] - flat[f"{b}_y"])
        after = np.hypot(aligned[f"{a}_x"] - aligned[f"{b}_x"], aligned[f"{a}_y"] - aligned[f"{b}_y"])
        ratios.append(after / before)
    assert np.allclose(ratios, ratios[0][0], rtol=1e-6)


@pytest.mark.parametrize("camera_zoom", [0.25, 1.0, 4.0])
def test_experimental_alignment_removes_camera_scale(camera_zoom):
    """Two cameras at different distances must present the model with the same animal."""
    flat = _synthetic_flat()
    zoomed = flat.copy()
    for bp in BEHAVIOR_BODYPARTS:
        zoomed[f"{bp}_x"] *= camera_zoom
        zoomed[f"{bp}_y"] *= camera_zoom
    aligned = egocentric_align(zoomed)
    body_length = np.hypot(
        aligned["nose_x"] - aligned["tail_x"], aligned["nose_y"] - aligned["tail_y"]
    )
    assert np.allclose(np.median(body_length), TRAINING_BODY_LENGTH_PX, rtol=1e-6)


def test_foreign_keypoint_schema_is_rejected():
    alien = pd.DataFrame(
        {
            "snout_x": np.arange(20.0),
            "snout_y": np.arange(20.0),
            "tailbase_x": np.arange(20.0),
            "tailbase_y": np.arange(20.0),
        }
    )
    with pytest.raises(ValueError) as excinfo:
        build_behavior_feature_set(alien)
    message = str(excinfo.value)
    assert "nose" in message and "B1L" in message


def test_confidence_belongs_to_the_reported_label():
    prob_df = pd.DataFrame(
        {
            "prob_Locomotion": [0.10, 0.80],
            "prob_Sniffing": [0.90, 0.20],
        }
    )
    labels = np.array(["Locomotion", "Locomotion"], dtype=object)
    conf = confidence_for_labels(labels, prob_df)
    # Row 0's highest probability is Sniffing's 0.90, which must not be reported.
    assert conf == pytest.approx([10.0, 80.0])


def test_no_heuristic_rules_invent_labels():
    flat = _synthetic_flat()
    n = len(flat)
    raw = np.array(["Locomotion"] * n, dtype=object)
    prob_df = pd.DataFrame(
        {
            "prob_Locomotion": np.full(n, 0.6),
            "prob_Jump": np.full(n, 0.2),
            f"prob_{UNASSIGNED}": np.full(n, 0.2),
        }
    )
    final, source, metrics = refine_behavior_labels(raw, prob_df, flat)

    # A constant-speed animal used to be labelled Jump by the percentile rule.
    assert "Jump" not in set(final)
    assert set(final) == {"Locomotion"}
    assert set(source) <= {"xgboost", "freezing_model", "tracking_bad"}
    assert metrics["confidence"].to_numpy() == pytest.approx(np.full(n, 60.0))


def test_freezing_model_overrides_and_carries_its_own_confidence():
    flat = _synthetic_flat(n=10)
    raw = np.array(["Locomotion"] * 10, dtype=object)
    prob_df = pd.DataFrame({"prob_Locomotion": np.full(10, 0.7)})
    freezing_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    freezing_prob = np.full(10, 0.93)

    final, source, metrics = refine_behavior_labels(
        raw, prob_df, flat, freezing_pred=freezing_pred, freezing_prob=freezing_prob
    )
    assert list(final[:2]) == ["Freezing", "Freezing"]
    assert list(source[:2]) == ["freezing_model", "freezing_model"]
    assert metrics["confidence"].to_numpy()[:2] == pytest.approx([93.0, 93.0])
    assert metrics["confidence"].to_numpy()[2:] == pytest.approx(np.full(8, 70.0))


def test_bad_tracking_becomes_unassigned_without_a_confidence():
    flat = _synthetic_flat(n=10, likelihood=0.05)
    raw = np.array(["Locomotion"] * 10, dtype=object)
    prob_df = pd.DataFrame({"prob_Locomotion": np.full(10, 0.7)})

    final, source, metrics = refine_behavior_labels(raw, prob_df, flat)
    assert set(final) == {UNASSIGNED}
    assert set(source) == {"tracking_bad"}
    assert np.isnan(metrics["confidence"].to_numpy()).all()


def test_outputs_use_the_papers_unassigned_name():
    flat = _synthetic_flat(n=10, likelihood=0.05)
    raw = np.array(["Locomotion"] * 10, dtype=object)
    prob_df = pd.DataFrame({"prob_Locomotion": np.full(10, 0.7)})
    final, source, _ = refine_behavior_labels(raw, prob_df, flat)
    assert "Unspecified" not in set(final)
    assert UNASSIGNED in set(final)
