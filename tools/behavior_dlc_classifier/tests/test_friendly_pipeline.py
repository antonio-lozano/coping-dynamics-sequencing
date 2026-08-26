from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freezing_dlc.friendly_pipeline import (
    FriendlyPipelineSettings,
    _centroid_path_pixels,
    _run_dlc_in_env,
    _write_results_workbook,
    run_friendly_pipeline,
)


def test_dlc_resume_helpers_track_each_stage_independently(tmp_path):
    from freezing_dlc.friendly_pipeline import (
        _dlc_stage_lists,
        _has_dlc_analysis,
        _has_dlc_filtered,
        _has_dlc_filtered_video,
    )

    video = tmp_path / "session_mouse1.mp4"
    video.write_bytes(b"")
    assert not _has_dlc_analysis(video)
    assert not _has_dlc_filtered(video)
    assert not _has_dlc_filtered_video(video)

    (tmp_path / "session_mouse1DLC_model_shuffle1_100000.h5").write_bytes(b"")
    assert _has_dlc_analysis(video)
    assert not _has_dlc_filtered(video)

    (tmp_path / "session_mouse1DLC_model_shuffle1_100000filtered.h5").write_bytes(b"")
    assert not _has_dlc_filtered(video)
    (tmp_path / "session_mouse1DLC_model_shuffle1_100000filtered.csv").write_text("x")
    assert _has_dlc_filtered(video)

    (tmp_path / "session_mouse1DLC_model_shuffle1_100000filtered_labeled.mp4").write_bytes(b"")
    assert _has_dlc_filtered_video(video)

    # A filtered result is enough to create an overlay; losing an optional
    # unfiltered artifact must not trigger expensive inference again.
    (tmp_path / "session_mouse1DLC_model_shuffle1_100000.h5").unlink()
    analyze, filtering, labeled = _dlc_stage_lists([video], make_tracked_videos=True)
    assert analyze == []
    assert filtering == []
    assert labeled == []

    analyze, filtering, labeled = _dlc_stage_lists(
        [video], make_tracked_videos=True, force=True
    )
    assert analyze == [video]
    assert filtering == [video]
    assert labeled == [video]


def test_bundled_dlc_log_stream_is_always_decoded_as_utf8(tmp_path, monkeypatch):
    """Unicode tqdm output must not inherit Windows' cp1252 decoder."""
    captured = {}

    class FakeProcess:
        stdout = iter(["  1%|\u258f         | 1/100\n"])

        def wait(self):
            return 0

        def kill(self):
            raise AssertionError("successful child must not be killed")

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(
        "freezing_dlc.friendly_pipeline._dlc_python_command",
        lambda env: ["python"],
    )
    monkeypatch.setattr("freezing_dlc.friendly_pipeline.subprocess.Popen", fake_popen)

    messages = []
    _run_dlc_in_env(
        tmp_path / "config.yaml",
        [tmp_path / "video.mp4"],
        ".mp4",
        tmp_path / "work",
        "bundled",
        False,
        messages.append,
    )

    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
    assert captured["env"]["PYTHONUTF8"] == "1"
    assert "[dlc] 1%|\u258f" in messages[-1]

ROOT = Path(__file__).resolve().parents[1]
DEMO_FILTERED_DIR = ROOT / "data" / "raw" / "DLC_filtered"
COMPACT_MODEL = ROOT / "models" / "freezing_model.sav"


def test_centroid_path_length_of_a_straight_walk():
    flat = pd.DataFrame(
        {
            "nose_x": np.arange(5.0),
            "nose_y": np.zeros(5),
            "tail_x": np.arange(5.0) + 2.0,
            "tail_y": np.zeros(5),
        }
    )
    assert _centroid_path_pixels(flat) == pytest.approx(4.0)


def _fake_predictions() -> dict:
    return {
        "Trial 9_mouse3": {
            "pred": np.array([0, 1] * 400, dtype=int),
            "stem": "Trial 9_mouse3DLC_resnet50filtered",
            "locomotor_px": 123.0,
        }
    }


def _settings(tmp_path: Path, **overrides) -> FriendlyPipelineSettings:
    defaults = dict(project_dir=tmp_path, model_path=tmp_path, dlc_config_path=tmp_path)
    defaults.update(overrides)
    return FriendlyPipelineSettings(**defaults)


def test_workbook_sheet_names_carry_the_real_bin_length(tmp_path: Path):
    pytest.importorskip("openpyxl")
    out = tmp_path / "out.xlsx"

    _write_results_workbook(out, _fake_predictions(), _settings(tmp_path, bin_sec=60), lambda m: None)

    names = pd.ExcelFile(out).sheet_names
    assert "freezing_seconds_60stimebin" in names
    assert "freezing_percentage_60stimebin" in names


def test_workbook_sheet_names_stay_legacy_for_the_default_bin(tmp_path: Path):
    """The lab's historical workbooks use these exact names; the default run
    must keep producing them byte for byte."""
    pytest.importorskip("openpyxl")
    out = tmp_path / "out.xlsx"

    _write_results_workbook(out, _fake_predictions(), _settings(tmp_path), lambda m: None)

    names = pd.ExcelFile(out).sheet_names
    assert names == [
        "total_freezing_seconds",
        "freezing_seconds_30stimebin",
        "freezing_percentage_30stimebin",
        "locomotor_activity",
    ]


def test_pipeline_reuses_existing_tracking_when_dlc_is_skipped(tmp_path: Path):
    """With "Run DeepLabCut" unchecked, existing tracking beside the videos is
    the input. This used to look for it in a freshly created working folder,
    which is empty by construction, so the checkbox guaranteed a failure."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("openpyxl")
    pytest.importorskip("matplotlib")
    demo_csvs = sorted(DEMO_FILTERED_DIR.glob("*filtered.csv"))
    if not demo_csvs or not COMPACT_MODEL.exists():
        pytest.skip("bundled demo tracking or model not present")

    project = tmp_path / "proj"
    project.mkdir()

    # Real tracking schema, truncated for speed.
    src_csv = demo_csvs[0]
    lines = src_csv.read_text(encoding="utf-8").splitlines()[:303]
    (project / src_csv.name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # A tiny real video whose name matches the tracking file's stem.
    video_stem = src_csv.name.split("DLC")[0]
    writer = cv2.VideoWriter(
        str(project / f"{video_stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48)
    )
    assert writer.isOpened()
    rng = np.random.default_rng(0)
    for _ in range(20):
        writer.write(rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8))
    writer.release()

    settings = FriendlyPipelineSettings(
        project_dir=project,
        model_path=COMPACT_MODEL,
        dlc_config_path=project,
        run_clahe=False,
        run_dlc=False,
        make_tracked_videos=False,
        run_behavior_analysis=False,
    )
    package = run_friendly_pipeline(settings, log=lambda message: None)

    assert (package / "06_results" / "freezing_FINAL_results.xlsx").exists()
    assert list((package / "04_features" / "frame_predictions_per_animal").glob("*_pred.csv"))
    assert list((package / "03_DLC_analysis" / "filtered").glob("*filtered.csv"))
    assert (package / "07_ethogram" / "freezing_ethogram_vertical_chunks_FINAL.png").exists()
