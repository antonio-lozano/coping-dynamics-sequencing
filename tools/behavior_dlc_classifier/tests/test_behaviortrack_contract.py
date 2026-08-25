from pathlib import Path
import sys

import pandas as pd


BEHAVIORTRACK_SRC = Path(__file__).resolve().parents[1] / "BehaviorTrack" / "src"
if str(BEHAVIORTRACK_SRC) not in sys.path:
    sys.path.insert(0, str(BEHAVIORTRACK_SRC))

from utils.session_parser import Session, write_sessions  # noqa: E402
from utils.validation import (  # noqa: E402
    BEHAVIORS,
    create_blinded_template,
    validate_predictions,
)
from utils.workspace import Workspace  # noqa: E402


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


def test_session_first_layout_and_blank_session_omits_level(tmp_path):
    workspace = _workspace(tmp_path)
    write_sessions(
        workspace.sessions_csv,
        [
            Session("mouse1.mp4", "mouse1", "Test", "mouse1"),
            Session("mouse2.mp4", "mouse2", "", "mouse2"),
        ],
    )
    assert workspace.dlc_filtered_dir("mouse1") == workspace.results / "DLC" / "Test" / "Filtered_CSV"
    assert workspace.dlc_tracked_videos_dir("mouse1") == workspace.results / "DLC" / "Test" / "Tracked_Videos"
    assert workspace.feature_output_dir("mouse1") == workspace.results / "Features" / "Test"
    assert workspace.behavior_predictions_dir("mouse1") == workspace.results / "Behaviors" / "Test" / "Predictions"
    assert workspace.behavior_summaries_dir("mouse1") == workspace.results / "Behaviors" / "Test" / "Summaries"
    assert workspace.dlc_filtered_dir("mouse2") == workspace.results / "DLC" / "Filtered_CSV"
    assert workspace.feature_output_dir("mouse2") == workspace.results / "Features"
    assert workspace.behavior_predictions_dir("mouse2") == workspace.results / "Behaviors" / "Predictions"


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
