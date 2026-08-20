from pathlib import Path

import numpy as np
import pandas as pd

import pytest

from freezing_dlc.train import (
    _align_sequence_for_label_lag,
    _apply_label_lag_to_dataset,
    _best_threshold,
    _find_label_column,
    _index_by_key,
    _normalize_key,
    _require_both_classes,
    _sample_class_indices,
    _split_dataset,
    _threshold_metrics,
    build_training_dataset,
    train_model,
)


def _grouped_dataset(n_groups: int, frames_per_group: int = 40):
    groups = np.repeat([f"video_{i}" for i in range(n_groups)], frames_per_group)
    y = (np.arange(len(groups)) % 3 == 0).astype(np.uint8)
    return y, groups


def test_threshold_by_f1_calls_almost_everything_positive_but_mcc_does_not():
    """The defect that made the old model match the do-nothing classifier.

    Most frames are freezing here and the scores are only weakly informative, so
    lowering the cutoff buys recall faster than it costs precision and F1 keeps
    rising. Matthews correlation charges for the false positives, so it stops.
    """
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.85).astype(int)
    probs = np.clip(rng.normal(0.55, 0.25, size=4000) + 0.12 * y, 0.0, 1.0)
    grid = np.linspace(0.05, 0.95, 19)

    by_f1 = _best_threshold(y, probs, grid, criterion="f1")
    by_mcc = _best_threshold(y, probs, grid, criterion="mcc")

    assert by_f1.threshold < by_mcc.threshold
    # Chasing F1 lands within a whisker of calling every single frame freezing.
    assert by_f1.recall > 0.99
    assert by_mcc.mcc > by_f1.mcc


def test_threshold_metrics_agree_with_scikit_learn():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.3).astype(int)
    probs = rng.random(500)

    metrics = _threshold_metrics(y, probs, 0.5)

    sk = pytest.importorskip("sklearn.metrics")
    pred = (probs >= 0.5).astype(int)
    assert metrics.mcc == pytest.approx(sk.matthews_corrcoef(y, pred))
    assert metrics.balanced_accuracy == pytest.approx(sk.balanced_accuracy_score(y, pred))
    assert metrics.f1 == pytest.approx(sk.f1_score(y, pred))


def test_unknown_threshold_criterion_is_refused():
    metrics = _threshold_metrics(np.array([0, 1]), np.array([0.1, 0.9]), 0.5)
    with pytest.raises(ValueError, match="Unknown threshold criterion"):
        metrics.criterion_score("accuracy")


def test_find_label_column_prefers_pred_like_names():
    df = pd.DataFrame({"prob": [0.1, 0.9], "pred": [0, 1]})
    assert _find_label_column(df) == "pred"


def test_find_label_column_falls_back_to_first_numeric():
    df = pd.DataFrame({"name": ["a", "b"], "yhat": [1, 0]})
    assert _find_label_column(df) == "yhat"


def test_normalize_key_handles_animal_patterns():
    key = _normalize_key(Path("Animal_1_2DLC_resnet50_filtered.csv"))
    assert key == "animal_1_2"


def test_normalize_key_matches_trial_dlc_and_manual_labels():
    dlc_key = _normalize_key(Path("Trial    9_mouse3DLC_resnet50_Freezing_07-2020Sep29shuffle1_500000filtered.csv"))
    label_key = _normalize_key(Path("Trial_9_mouse3_labels.csv"))
    assert dlc_key == "trial9mouse3"
    assert label_key == "trial9mouse3"


def test_sample_class_indices_respects_limit_and_classes():
    y = pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype="uint8").to_numpy()
    keep = _sample_class_indices(y, max_frames=5, random_state=7)
    kept_y = y[keep]

    assert len(keep) == 5
    assert kept_y.sum() == 2
    assert int((kept_y == 0).sum()) == 3


def test_align_sequence_for_positive_label_lag():
    x_seq = np.arange(6, dtype=float).reshape(6, 1)
    y_seq = np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8)

    x_aligned, y_aligned = _align_sequence_for_label_lag(x_seq, y_seq, label_lag_frames=2)

    assert x_aligned[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert y_aligned.tolist() == [1, 1, 0, 0]


def test_align_sequence_for_negative_label_lag():
    x_seq = np.arange(6, dtype=float).reshape(6, 1)
    y_seq = np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8)

    x_aligned, y_aligned = _align_sequence_for_label_lag(x_seq, y_seq, label_lag_frames=-2)

    assert x_aligned[:, 0].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert y_aligned.tolist() == [0, 0, 1, 1]


def test_split_keeps_whole_videos_apart_and_holds_one_back():
    y, groups = _grouped_dataset(n_groups=5)

    split = _split_dataset(y, groups, split_mode="group", test_size=0.2, random_state=0)

    assert split.holdout_is_honest
    assert len(split.test_idx) > 0

    parts = [set(groups[idx].tolist()) for idx in (split.train_idx, split.val_idx, split.test_idx)]
    # No recording may appear in two partitions: frames within a video are near
    # duplicates, so any overlap would inflate the held-out score.
    assert parts[0] & parts[1] == set()
    assert parts[0] & parts[2] == set()
    assert parts[1] & parts[2] == set()
    assert set().union(*parts) == set(groups.tolist())

    covered = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
    assert sorted(covered.tolist()) == list(range(len(y)))


def test_split_admits_when_no_video_can_be_held_back():
    y, groups = _grouped_dataset(n_groups=2)

    split = _split_dataset(y, groups, split_mode="group", test_size=0.5, random_state=0)

    assert not split.holdout_is_honest
    assert len(split.test_idx) == 0
    assert set(groups[split.train_idx].tolist()) & set(groups[split.val_idx].tolist()) == set()


def test_random_split_is_never_reported_as_an_honest_holdout():
    y, groups = _grouped_dataset(n_groups=4)

    split = _split_dataset(y, groups, split_mode="random", test_size=0.25, random_state=0)

    assert split.mode == "random"
    assert not split.holdout_is_honest


def test_colliding_recording_keys_are_refused(tmp_path: Path):
    first = tmp_path / "Animal_1_2DLC_resnet50_filtered.csv"
    second = tmp_path / "Animal_1_2_labels.csv"
    first.write_text("a\n", encoding="utf-8")
    second.write_text("a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="share one recording key"):
        _index_by_key([first, second], "tracking")


def _write_dlc_csv(path: Path, n_rows: int, trailing_blank: bool = False) -> None:
    """A minimal DLC filtered CSV: three header rows, nose and tail tracked."""
    lines = [
        "scorer,DLC,DLC,DLC,DLC,DLC,DLC",
        "bodyparts,nose,nose,nose,tail,tail,tail",
        "coords,x,y,likelihood,x,y,likelihood",
    ]
    rng = np.random.default_rng(len(lines) + n_rows)
    x = np.cumsum(np.where((np.arange(n_rows) // 20) % 2 == 1, 4.0, 0.02) * (0.7 + 0.6 * rng.random(n_rows)))
    for i in range(n_rows):
        lines.append(f"{i},{x[i]:.3f},{x[i] * 0.5:.3f},0.9,{x[i] + 50:.3f},{x[i] * 0.5:.3f},0.9")
    path.write_text("\n".join(lines) + "\n" + ("\n" if trailing_blank else ""), encoding="utf-8")


def _write_labels_csv(path: Path, n_rows: int) -> None:
    labels = ((np.arange(n_rows) // 20) % 2 == 0).astype(int)
    path.write_text("pred\n" + "\n".join(str(v) for v in labels) + "\n", encoding="utf-8")


def test_trailing_blank_line_in_dlc_csv_does_not_break_the_dataset_build(tmp_path: Path):
    """The sampling plan sizes videos from the raw line count, which sees a
    trailing blank line that pandas skips; indexing had one row too many."""
    dlc_dir = tmp_path / "dlc"
    labels_dir = tmp_path / "labels"
    dlc_dir.mkdir()
    labels_dir.mkdir()
    _write_dlc_csv(dlc_dir / "Trial 9_mouse3DLC_resnet50filtered.csv", 40, trailing_blank=True)
    _write_labels_csv(labels_dir / "Trial_9_mouse3_labels.csv", 41)

    X, y, keys, feature_names = build_training_dataset(dlc_dir, labels_dir, show_progress=False)

    assert len(X) == 40
    assert len(y) == 40
    assert len(keys) == 40
    assert feature_names


def test_require_both_classes_names_the_split():
    _require_both_classes(np.array([0, 1], dtype=np.uint8), "fit split")
    with pytest.raises(ValueError, match="fit split contains only one class"):
        _require_both_classes(np.zeros(10, dtype=np.uint8), "fit split")


def test_label_lag_and_frame_cap_combine_without_misalignment(tmp_path: Path):
    """With both requested, the cap is applied after the lag, so a lagged label
    always belongs to the next *frame* rather than the next sampled row."""
    dlc_dir = tmp_path / "dlc"
    labels_dir = tmp_path / "labels"
    dlc_dir.mkdir()
    labels_dir.mkdir()
    for name in ("Trial 1_mouse1", "Trial 2_mouse2"):
        _write_dlc_csv(dlc_dir / f"{name}DLC_resnet50filtered.csv", 120)
        _write_labels_csv(labels_dir / f"{name.replace(' ', '_')}_labels.csv", 120)

    payload = train_model(
        dlc_dir=dlc_dir,
        labels_dir=labels_dir,
        out_model=tmp_path / "model.sav",
        feature_mode="compact",
        n_estimators=8,
        step_estimators=8,
        split_mode="group",
        max_train_frames=150,
        label_lag_frames=2,
        n_jobs=1,
        show_progress=False,
    )

    assert payload["alignment"]["label_lag_frames"] == 2
    # Two 120-frame videos lose 2 frames each to the lag, then the cap bites.
    assert payload["n_training_frames"] <= 150
    assert (tmp_path / "model.sav").exists()


def test_apply_label_lag_to_dataset_is_group_safe():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 1, 1, 0, 0, 1], dtype=np.uint8)
    groups = np.array(["video_a", "video_a", "video_a", "video_b", "video_b", "video_b"], dtype=str)

    X_out, y_out, groups_out, info = _apply_label_lag_to_dataset(X, y, groups, label_lag_frames=1)

    assert X_out.shape == (4, 2)
    assert y_out.tolist() == [1, 1, 0, 1]
    assert groups_out.tolist() == ["video_a", "video_a", "video_b", "video_b"]
    assert info["label_lag_frames"] == 1
    assert info["kept_frames"] == 4
    assert info["dropped_frames"] == 2
