import numpy as np

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
