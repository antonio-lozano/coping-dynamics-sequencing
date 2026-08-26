from pathlib import Path
import sys

import pandas as pd


BEHAVIORTRACK_SRC = Path(__file__).resolve().parents[1] / "BehaviorTrack" / "src"
if str(BEHAVIORTRACK_SRC) not in sys.path:
    sys.path.insert(0, str(BEHAVIORTRACK_SRC))

from utils.session_parser import Session, write_sessions  # noqa: E402
from utils.config_loader import save_session_context  # noqa: E402
from utils.validation import (  # noqa: E402
    BEHAVIORS,
    create_blinded_template,
    validate_predictions,
)
from utils.workspace import Workspace, load_workspace  # noqa: E402
import bh_track  # noqa: E402
from bh_track import TrackWindow, _archive_dlc_outputs, _restore_cached_outputs  # noqa: E402


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def _workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "project"
    results = tmp_path / "external_results"
    videos = tmp_path / "external_videos"
    videos.mkdir(parents=True)
    return Workspace(
        config_path=root / "config.yaml",
        config={"video": {"fps": 25, "videotype": ".mp4"}},
        root=root,
        tool_root=tmp_path,
        raw_videos=videos,
        results=results,
    )


def test_external_video_and_results_paths_persist_between_windows(tmp_path):
    tool = tmp_path / "behavior_dlc_classifier"
    config = tool / "BehaviorTrack" / "config" / "behaviortrack_config.yaml"
    engine_marker = tool / "src" / "freezing_dlc" / "__init__.py"
    engine_marker.parent.mkdir(parents=True)
    engine_marker.write_text("", encoding="utf-8")
    config.parent.mkdir(parents=True)
    config.write_text(
        "project:\n  workspace_root: ''\npaths:\n  raw_videos: Data/Raw_Videos\n  results: Results\n",
        encoding="utf-8",
    )
    videos = tmp_path / "experiment" / "Raw_Videos"
    results = tmp_path / "experiment" / "Results"
    save_session_context(
        config,
        {
            "raw_videos": str(videos),
            "results": str(results),
            "videotype": ".avi",
            "animal_id_pattern": "Mouse-[0-9]+",
            "filename_patterns": ["Hab", "Test"],
        },
    )

    first_window = load_workspace(config)
    second_window = load_workspace(config)
    assert first_window.raw_videos == videos.resolve()
    assert first_window.results == results.resolve()
    assert second_window.raw_videos == first_window.raw_videos
    assert second_window.results == first_window.results
    assert second_window.videotype == ".avi"
    assert second_window.section("onboarding")["animal_id_pattern"] == "Mouse-[0-9]+"
    assert second_window.section("onboarding")["filename_patterns"] == ["Hab", "Test"]


def test_animal_then_session_layout_and_blank_levels_are_omitted(tmp_path):
    workspace = _workspace(tmp_path)
    write_sessions(
        workspace.sessions_csv,
        [
            Session("mouse1.mp4", "mouse1", "Test", "mouse1"),
            Session("mouse2.mp4", "mouse2", "", "mouse2"),
        ],
    )
    assert workspace.dlc_analysis_dir("mouse1") == workspace.results / "DLC" / "mouse1" / "Test" / "Analysis"
    assert workspace.dlc_filtered_dir("mouse1") == workspace.results / "DLC" / "mouse1" / "Test" / "Filtered_CSV"
    assert workspace.dlc_tracked_videos_dir("mouse1") == workspace.results / "DLC" / "mouse1" / "Test" / "Tracked_Videos"
    assert workspace.feature_output_dir("mouse1") == workspace.results / "Features" / "mouse1" / "Test"
    assert workspace.behavior_predictions_dir("mouse1") == workspace.results / "Behaviors" / "mouse1" / "Test" / "Predictions"
    assert workspace.behavior_summaries_dir("mouse1") == workspace.results / "Behaviors" / "mouse1" / "Test" / "Summaries"
    assert workspace.dlc_filtered_dir("mouse2") == workspace.results / "DLC" / "mouse2" / "Filtered_CSV"
    assert workspace.feature_output_dir("mouse2") == workspace.results / "Features" / "mouse2"
    assert workspace.behavior_predictions_dir("mouse2") == workspace.results / "Behaviors" / "mouse2" / "Predictions"


def test_dlc_stage_artifacts_round_trip_through_external_results(tmp_path):
    workspace = _workspace(tmp_path)
    write_sessions(
        workspace.sessions_csv,
        [Session("mouse1.mp4", "C5121", "Test", "mouse1")],
    )
    prepared_dir = tmp_path / "working"
    prepared_dir.mkdir()
    prepared = prepared_dir / "mouse1.mp4"
    prepared.write_bytes(b"video")
    names = [
        "mouse1DLC_model_shuffle1_100000.h5",
        "mouse1DLC_model_shuffle1_100000.csv",
        "mouse1DLC_model_shuffle1_100000filtered.h5",
        "mouse1DLC_model_shuffle1_100000filtered.csv",
        "mouse1DLC_model_shuffle1_100000filtered_labeled.mp4",
    ]
    for name in names:
        (prepared_dir / name).write_bytes(b"stage")

    assert _archive_dlc_outputs(prepared, workspace) == 1
    assert (workspace.dlc_analysis_dir("mouse1") / names[0]).is_file()
    assert (workspace.dlc_filtered_dir("mouse1") / names[2]).is_file()
    assert (workspace.dlc_tracked_videos_dir("mouse1") / names[4]).is_file()

    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    resumed = resumed_dir / "mouse1.mp4"
    resumed.write_bytes(b"video")
    restored = _restore_cached_outputs(workspace.raw_videos / "mouse1.mp4", resumed, workspace)
    assert restored == len(names)
    assert all((resumed_dir / name).is_file() for name in names)


def test_tracking_publishes_each_recording_before_starting_the_next(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    videos = [workspace.raw_videos / "mouse1.mp4", workspace.raw_videos / "mouse2.mp4"]
    for video in videos:
        video.write_bytes(b"video")
    write_sessions(
        workspace.sessions_csv,
        [
            Session("mouse1.mp4", "mouse1", "Test", "mouse1"),
            Session("mouse2.mp4", "mouse2", "Test", "mouse2"),
        ],
    )
    config = tmp_path / "dlc_config.yaml"
    config.write_text("Task: test\n", encoding="utf-8")

    window = TrackWindow.__new__(TrackWindow)
    window.config_path = config
    window.workspace = workspace
    window.dlc_config_var = _Value(str(config))
    window.dlc_env_var = _Value("")
    window.clahe_var = _Value(False)
    window.labeled_var = _Value(False)
    window.force_var = _Value(False)
    window.dynamic_var = _Value(True)
    window._pending = lambda: (videos, videos)
    progress = []
    window.set_progress = lambda done, total: progress.append((done, total))

    calls = []

    def fake_run(_config, prepared, _videotype, _work_dir, _env, _labeled, _log, **kwargs):
        assert len(prepared) == 1
        video = prepared[0]
        if calls:
            first = workspace.dlc_filtered_dir("mouse1") / "mouse1DLC_testfiltered.csv"
            assert first.is_file(), "the first recording was not published before the second started"
        calls.append((video.name, kwargs["dynamic"]))
        (video.parent / f"{video.stem}DLC_testfiltered.csv").write_text("frame,x\n0,1\n", encoding="utf-8")

    monkeypatch.setattr(bh_track, "ensure_engine_importable", lambda _path: None)
    monkeypatch.setattr(
        bh_track,
        "_engine",
        lambda: (None, fake_run, lambda _env: None, lambda _env: "test DLC", None),
    )

    messages = []
    result = TrackWindow.work(window, messages.append)

    assert calls == [("mouse1.mp4", True), ("mouse2.mp4", True)]
    assert progress == [(0, 2), (1, 2), (2, 2)]
    assert result == "Processed 2 recording(s); 2 tracking file(s) collected."
    assert sum(message.startswith("[ready]") for message in messages) == 2
    assert (workspace.dlc_filtered_dir("mouse2") / "mouse2DLC_testfiltered.csv").is_file()


def test_validation_template_is_blind_and_independence_is_explicit(tmp_path):
    predictions = tmp_path / "predictions"
    labels = tmp_path / "labels"
    reports = tmp_path / "reports"
    predictions.mkdir()
    labels.mkdir()
    predicted = list(BEHAVIORS) * 2
    source = predictions / "mouse1_Test_behaviors.csv"
    pd.DataFrame({"frame": range(len(predicted)), "behavior": predicted}).to_csv(source, index=False)
    template = create_blinded_template(source, labels / "mouse1_Test_manual_labels.csv", 25.0)
    label_frame = pd.read_csv(template)
    assert list(label_frame.columns) == ["frame", "time_sec", "true_behavior"]
    label_frame["true_behavior"] = predicted
    label_frame.to_csv(template, index=False)

    result = validate_predictions(predictions, labels, reports)
    assert result.claim_status == "agreement audit only; independence not confirmed"
    metrics = pd.read_csv(result.metrics_csv)
    assert metrics["behavior"].tolist() == list(BEHAVIORS)
    assert metrics["f1"].eq(1.0).all()


def test_interval_annotations_are_supported_and_all_classes_stay_visible(tmp_path):
    predictions = tmp_path / "predictions"
    labels = tmp_path / "labels"
    predictions.mkdir()
    labels.mkdir()
    pd.DataFrame({"frame": range(4), "behavior": ["Jump", "Jump", "Turn", "Unassigned"]}).to_csv(
        predictions / "mouse1_behaviors.csv", index=False
    )
    pd.DataFrame(
        {
            "start_frame": [0, 2],
            "end_frame": [1, 3],
            "true_behavior": ["Jump", "Turn"],
        }
    ).to_csv(labels / "mouse1_labels.csv", index=False)
    result = validate_predictions(predictions, labels, tmp_path / "reports")
    metrics = pd.read_csv(result.metrics_csv)
    assert len(metrics) == 7
    assert int(metrics.loc[metrics.behavior == "Jump", "support_frames"].iloc[0]) == 2
    assert int(metrics.loc[metrics.behavior == "Turn", "support_frames"].iloc[0]) == 2
