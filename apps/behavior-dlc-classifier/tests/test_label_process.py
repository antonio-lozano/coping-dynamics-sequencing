import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from freezing_dlc import label_video


@pytest.mark.parametrize("saved", [True, False])
def test_label_process_returns_current_session_result(monkeypatch, tmp_path, saved):
    video = tmp_path / "recording with spaces.mp4"
    output = tmp_path / "labels.csv"
    output.write_text("old labels remain when canceled")
    def run(args, **kwargs):
        assert args[1:4] == ["-m", "freezing_dlc.cli", "label-video"]
        assert args[args.index("--video") + 1] == str(video)
        assert args[args.index("--existing-labels") + 1] == str(output)
        result = Path(args[args.index("--result-json") + 1])
        result.write_text(json.dumps({"saved": saved, "out_csv": str(output)}))
        return SimpleNamespace(returncode=0, stdout="labeler logs", stderr="")
    monkeypatch.setattr(label_video.subprocess, "run", run)
    result = label_video.annotate_video_in_subprocess(video, output, 25, output)
    assert result["saved"] is saved


def test_label_process_reports_child_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(label_video.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=1, stdout="", stderr="video decode failed"))
    with pytest.raises(RuntimeError, match="video decode failed"):
        label_video.annotate_video_in_subprocess(tmp_path / "v.mp4", tmp_path / "l.csv", 25)


def test_label_process_without_result_is_not_a_success(monkeypatch, tmp_path):
    monkeypatch.setattr(label_video.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout="", stderr=""))
    with pytest.raises(RuntimeError, match="without completion metadata"):
        label_video.annotate_video_in_subprocess(tmp_path / "v.mp4", tmp_path / "l.csv", 25)
