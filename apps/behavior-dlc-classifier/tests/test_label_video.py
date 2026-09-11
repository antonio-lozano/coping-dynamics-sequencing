import numpy as np
import pytest

from freezing_dlc.label_video import (
    _events_to_intervals,
    _events_to_labels,
    _frozen_frame_count,
    _labels_to_events,
    _normalize_events,
    _state_at_frame,
    _toggle_from_frame,
)


def test_normalize_events_drops_redundant_states():
    events = [(10, 1), (12, 1), (4, 0), (20, 0), (15, 1)]
    assert _normalize_events(events) == [(10, 1), (20, 0)]


def test_toggle_from_frame_creates_and_removes_transition():
    events = []
    events = _toggle_from_frame(events, 5)
    assert events == [(5, 1)]
    events = _toggle_from_frame(events, 5)
    assert events == []


def test_events_to_labels_matches_transitions():
    labels = _events_to_labels([(2, 1), (5, 0)], 8)
    assert np.array_equal(labels, np.array([0, 0, 1, 1, 1, 0, 0, 0], dtype=np.uint8))


def test_labels_round_trip_to_events():
    labels = np.array([0, 1, 1, 0, 0, 1], dtype=np.uint8)
    assert _events_to_labels(_labels_to_events(labels), len(labels)).tolist() == labels.tolist()


def test_small_video_keeps_controls_visible_without_changing_saved_frames(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from freezing_dlc.label_video import annotate_video_labels

    video = tmp_path / "small.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 200))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((200, 320, 3), 100, dtype=np.uint8))
    writer.release()
    original = video.read_bytes()
    rendered = []
    for name in ("namedWindow", "setMouseCallback", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *args: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(cv2, "imshow", lambda name, canvas: rendered.append(canvas.copy()))
    monkeypatch.setattr(cv2, "waitKeyEx", lambda *args: ord("s"))
    result = annotate_video_labels(video, tmp_path / "labels.csv", start_paused=True)

    assert result["saved"] and result["n_frames"] == 3
    assert rendered[0].shape[0] >= 520  # all four control/summary cards fit
    assert np.any(rendered[0][450:500, :300] > 150)  # summary text actually renders
    assert video.read_bytes() == original


def test_timeline_scrub_survives_paused_event_loop(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import json
    from freezing_dlc.label_video import annotate_video_labels

    video = tmp_path / "scrub.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 200))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((200, 320, 3), dtype=np.uint8))
    writer.release()
    callback = []
    canvas_size = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *args: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda name, fn: callback.append(fn))
    monkeypatch.setattr(cv2, "imshow", lambda name, canvas: canvas_size.append(canvas.shape))
    keys = iter((-1, ord("f"), ord("s")))

    def next_key(delay):
        key = next(keys)
        if key == -1:
            height, width = canvas_size[-1][:2]
            callback[0](cv2.EVENT_LBUTTONDOWN, width - 16, height - 17, 0, None)
        return key

    monkeypatch.setattr(cv2, "waitKeyEx", next_key)
    annotate_video_labels(video, tmp_path / "labels.csv", start_paused=True)
    saved = json.loads((tmp_path / "labels.json").read_text())
    assert saved["events"] == [{"frame": 9, "state": 1}]


def test_state_at_frame_uses_latest_transition():
    events = [(3, 1), (7, 0)]
    assert _state_at_frame(events, 2) == 0
    assert _state_at_frame(events, 3) == 1
    assert _state_at_frame(events, 6) == 1
    assert _state_at_frame(events, 7) == 0


def test_intervals_agree_with_full_label_expansion():
    """The render loop reads intervals straight off the toggle events now, so
    they must say exactly what expanding the events to per-frame labels says --
    including toggles past the end of the video and open final bouts."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(1, 30))
        events = [
            (int(rng.integers(0, 40)), int(rng.integers(0, 2))) for _ in range(int(rng.integers(0, 8)))
        ]

        labels = _events_to_labels(events, n)
        expected = []
        start = None
        for idx, value in enumerate(labels):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                expected.append((start, idx - 1))
                start = None
        if start is not None:
            expected.append((start, n - 1))

        got = [(d["start_frame"], d["end_frame"]) for d in _events_to_intervals(events, n, fps=25.0)]
        assert got == expected, (n, events)
        assert _frozen_frame_count(events, n) == int(labels.sum()), (n, events)
