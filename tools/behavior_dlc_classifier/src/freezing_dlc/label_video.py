from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .draw import blend_box as _blend_box
from .draw import mmss_tenths


Event = Tuple[int, int]

#: What ``cv2.waitKeyEx`` reports for the arrow keys on Windows, X11 and macOS.
#: ``waitKey & 0xFF`` folded these onto ``Q`` and ``S``, so pressing S with Caps
#: Lock on used to step a frame instead of saving.
_LEFT_KEYS = (2424832, 65361, 63234)
_RIGHT_KEYS = (2555904, 65363, 63235)


def _normalize_events(events: Sequence[Event]) -> List[Event]:
    by_frame = {}
    for frame_idx, state in events:
        frame = max(0, int(frame_idx))
        by_frame[frame] = 1 if int(state) else 0

    normalized: List[Event] = []
    prev_state = 0
    for frame in sorted(by_frame):
        state = by_frame[frame]
        if state != prev_state:
            normalized.append((frame, state))
            prev_state = state
    return normalized


def _state_at_frame(events: Sequence[Event], frame_idx: int) -> int:
    state = 0
    for event_frame, event_state in events:
        if event_frame > frame_idx:
            break
        state = int(event_state)
    return state


def _toggle_from_frame(events: Sequence[Event], frame_idx: int) -> List[Event]:
    frame = max(0, int(frame_idx))
    curr_state = _state_at_frame(events, frame)
    new_state = 0 if curr_state else 1
    prev_state = _state_at_frame(events, frame - 1) if frame > 0 else 0

    next_events = [(f, s) for f, s in events if int(f) != frame]
    if prev_state != new_state:
        next_events.append((frame, new_state))
    return _normalize_events(next_events)


def _labels_to_events(labels: Sequence[int]) -> List[Event]:
    events: List[Event] = []
    prev_state = 0
    for frame_idx, label in enumerate(labels):
        state = 1 if int(label) else 0
        if state != prev_state:
            events.append((frame_idx, state))
            prev_state = state
    return events


def _events_to_labels(events: Sequence[Event], n_frames: int) -> np.ndarray:
    labels = np.zeros(int(max(0, n_frames)), dtype=np.uint8)
    normalized = _normalize_events(events)
    if not normalized or len(labels) == 0:
        return labels

    state = 0
    start = 0
    for frame, next_state in normalized:
        frame = max(0, min(int(frame), len(labels)))
        if frame > start:
            labels[start:frame] = state
        state = int(next_state)
        start = frame
    if start < len(labels):
        labels[start:] = state
    return labels


def _events_to_intervals(events: Sequence[Event], n_frames: int, fps: float) -> List[dict]:
    """Freezing bouts as inclusive frame ranges, read straight off the toggles.

    The events *are* the transitions, so expanding them to a per-frame label
    array first (as this once did) only added an O(n_frames) scan to something
    the render loop calls on every frame.
    """
    n = int(max(0, n_frames))

    def bout(first: int, last: int) -> dict:
        return {
            "start_frame": int(first),
            "end_frame": int(last),
            "start_sec": float(first / fps) if fps > 0 else float(first),
            "end_sec": float(last / fps) if fps > 0 else float(last),
        }

    intervals: List[dict] = []
    start: Optional[int] = None
    for frame, state in _normalize_events(events):
        frame = min(int(frame), n)
        if state and start is None:
            if frame < n:
                start = frame
        elif not state and start is not None:
            intervals.append(bout(start, frame - 1))
            start = None
    if start is not None and n > 0:
        intervals.append(bout(start, n - 1))
    return intervals


def _frozen_frame_count(events: Sequence[Event], n_frames: int) -> int:
    total = 0
    for interval in _events_to_intervals(events, n_frames, fps=1.0):
        total += int(interval["end_frame"]) - int(interval["start_frame"]) + 1
    return total


def _load_existing_labels(path: Path) -> List[Event]:
    df = pd.read_csv(path)
    preferred = ["pred", "label", "freezing", "freeze", "y", "target"]
    lowered = {str(c).lower(): str(c) for c in df.columns}
    for name in preferred:
        if name in lowered:
            col = lowered[name]
            return _labels_to_events((df[col].fillna(0).to_numpy(dtype=float) >= 0.5).astype(np.uint8))
    raise ValueError(f"Could not find a label column in '{path}'.")


def _default_output_csv(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_labels.csv")


def annotate_video_labels(
    video_path: Path,
    out_csv: Optional[Path] = None,
    fps_override: Optional[float] = None,
    start_paused: bool = False,
    speed: float = 1.0,
    existing_labels: Optional[Path] = None,
) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for interactive video annotation.") from exc

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_csv = _default_output_csv(video_path) if out_csv is None else Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_csv.with_suffix(".json")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(fps_override) if fps_override is not None else float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0

    events = _load_existing_labels(existing_labels) if existing_labels else []
    history: List[List[Event]] = []
    state = {
        "frame_idx": 0,
        "events": _normalize_events(events),
        "playing": not bool(start_paused),
        "saved": False,
        "video_x_offset": 0,
    }

    window_name = f"freezing labeler: {video_path.name}"

    def push_history() -> None:
        history.append(list(state["events"]))

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and int(x) >= int(state["video_x_offset"]):
            timeline = state.get("timeline_rect")
            if timeline is not None:
                x1, y1, x2, y2 = timeline
                if int(y1) <= int(y) <= int(y2) and int(x2) > int(x1):
                    # Scrubbing must never edit labels; only F and a video click do.
                    rel = min(1.0, max(0.0, (float(x) - float(x1)) / float(x2 - x1)))
                    state["playing"] = False
                    state["frame_idx"] = int(round(rel * max(0, total_frames - 1)))
                    return
            push_history()
            state["events"] = _toggle_from_frame(state["events"], int(state["frame_idx"]))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print(f"[label] video={video_path}")
    print(f"[label] output_csv={out_csv}")
    print("[label] controls: press F at freezing start and press F again at freezing end")
    print("[label] controls: space play/pause | left/right or a/d step frame | u undo | r reset | s save+quit | q quit")
    print("[label] mouse: click the timeline to jump | click the video to toggle freezing")

    decoded_idx = -1  # index of the frame currently held in `frame`
    stream_pos = 0  # index cap.read() would decode next; sequential reads skip the seek
    frame = None
    try:
        while True:
            frame_idx = max(0, min(int(state["frame_idx"]), max(0, total_frames - 1)))
            if frame_idx != decoded_idx:
                if frame_idx != stream_pos:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, new_frame = cap.read()
                stream_pos = frame_idx + 1
                if not ok:
                    if decoded_idx < 0:
                        raise RuntimeError("Could not read video frame during annotation.")
                    # The container promised frames it cannot decode. Never
                    # throw the session's labels away over that: hold the last
                    # good frame and pause.
                    state["playing"] = False
                    state["frame_idx"] = decoded_idx
                    stream_pos = -1
                    continue
                if len(new_frame.shape) == 2:
                    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_GRAY2BGR)
                frame = new_frame
                decoded_idx = frame_idx

            current_state = _state_at_frame(state["events"], frame_idx)
            height, width = frame.shape[:2]
            panel_width = max(220, min(320, width // 2))
            state["video_x_offset"] = panel_width
            canvas = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
            canvas[:, :panel_width] = (18, 18, 24)
            canvas[:, panel_width:] = frame
            cv2.line(canvas, (panel_width - 1, 0), (panel_width - 1, height), (90, 90, 105), 1)

            status = "FREEZING" if current_state else "MOVING"
            mode = "PLAY" if state["playing"] else "PAUSE"
            accent = (52, 84, 235) if current_state else (60, 179, 113)
            text_main = (245, 245, 245)
            text_sub = (205, 205, 215)
            time_sec = frame_idx / fps if fps > 0 else float(frame_idx)
            event_count = len(state["events"])
            frozen_frames = _frozen_frame_count(state["events"], total_frames)
            freeze_seconds = frozen_frames / fps if fps > 0 else float(frozen_frames)

            panel_margin = 12
            inner_width = panel_width - panel_margin * 2
            card_x1 = panel_margin
            card_x2 = card_x1 + inner_width

            top_y1 = panel_margin
            top_y2 = top_y1 + 126
            _blend_box(canvas, (card_x1, top_y1), (card_x2, top_y2), (28, 28, 36), 0.9)
            cv2.rectangle(canvas, (card_x1, top_y1), (card_x2, top_y2), (90, 90, 105), 1)

            badge_x1 = card_x1 + 10
            badge_y1 = top_y1 + 10
            badge_x2 = min(card_x2 - 10, badge_x1 + 132)
            badge_y2 = badge_y1 + 28
            _blend_box(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), accent, 0.95)
            cv2.putText(canvas, status, (badge_x1 + 8, badge_y1 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, mode, (card_x2 - 64, badge_y1 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.46, text_sub, 1, cv2.LINE_AA)

            cv2.putText(canvas, "Current time", (card_x1 + 10, top_y1 + 56), cv2.FONT_HERSHEY_DUPLEX, 0.38, text_sub, 1, cv2.LINE_AA)
            cv2.putText(canvas, mmss_tenths(time_sec), (card_x1 + 10, top_y1 + 79), cv2.FONT_HERSHEY_DUPLEX, 0.62, text_main, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"frame {frame_idx + 1:,}/{total_frames:,}", (card_x1 + 10, top_y1 + 104), cv2.FONT_HERSHEY_DUPLEX, 0.38, text_sub, 1, cv2.LINE_AA)

            guide_y1 = top_y2 + 12
            guide_y2 = guide_y1 + 92
            _blend_box(canvas, (card_x1, guide_y1), (card_x2, guide_y2), (32, 34, 44), 0.92)
            cv2.rectangle(canvas, (card_x1, guide_y1), (card_x2, guide_y2), (116, 185, 164), 1)
            cv2.putText(canvas, "How to relabel", (card_x1 + 10, guide_y1 + 22), cv2.FONT_HERSHEY_DUPLEX, 0.42, text_main, 1, cv2.LINE_AA)
            guide_lines = [
                "1. Watch/pause at freeze start",
                "2. Press F",
                "3. Press F again at freeze end",
                "4. Press S when finished",
            ]
            for idx, line in enumerate(guide_lines):
                cv2.putText(canvas, line, (card_x1 + 10, guide_y1 + 44 + idx * 13), cv2.FONT_HERSHEY_DUPLEX, 0.32, text_sub, 1, cv2.LINE_AA)

            controls_y1 = guide_y2 + 12
            controls_y2 = controls_y1 + 128
            _blend_box(canvas, (card_x1, controls_y1), (card_x2, controls_y2), (28, 28, 36), 0.88)
            cv2.rectangle(canvas, (card_x1, controls_y1), (card_x2, controls_y2), (90, 90, 105), 1)
            cv2.putText(canvas, "Controls", (card_x1 + 10, controls_y1 + 22), cv2.FONT_HERSHEY_DUPLEX, 0.42, text_main, 1, cv2.LINE_AA)
            control_lines = [
                "f         start/end freezing",
                "space     play or pause",
                "arrows    step one frame",
                "u         undo last toggle",
                "r         reset labels",
                "s         save and quit",
                "mouse     bar jumps, video toggles",
            ]
            for idx, line in enumerate(control_lines):
                cv2.putText(canvas, line, (card_x1 + 10, controls_y1 + 46 + idx * 14), cv2.FONT_HERSHEY_DUPLEX, 0.34, text_sub, 1, cv2.LINE_AA)

            summary_y1 = controls_y2 + 12
            summary_y2 = min(height - 12, summary_y1 + 88)
            _blend_box(canvas, (card_x1, summary_y1), (card_x2, summary_y2), (28, 28, 36), 0.88)
            cv2.rectangle(canvas, (card_x1, summary_y1), (card_x2, summary_y2), (90, 90, 105), 1)
            cv2.putText(canvas, "Labeled so far", (card_x1 + 10, summary_y1 + 22), cv2.FONT_HERSHEY_DUPLEX, 0.42, text_main, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"toggles  {event_count}", (card_x1 + 10, summary_y1 + 46), cv2.FONT_HERSHEY_DUPLEX, 0.38, text_sub, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"freeze   {mmss_tenths(freeze_seconds)}", (card_x1 + 10, summary_y1 + 68), cv2.FONT_HERSHEY_DUPLEX, 0.38, text_sub, 1, cv2.LINE_AA)

            timeline_x1 = panel_width + 16
            timeline_x2 = panel_width + width - 16
            timeline_y1 = max(8, height - 24)
            timeline_y2 = height - 10
            state["timeline_rect"] = (timeline_x1, timeline_y1, timeline_x2, timeline_y2)
            _blend_box(canvas, (timeline_x1, timeline_y1), (timeline_x2, timeline_y2), (30, 30, 38), 0.88)
            cv2.rectangle(canvas, (timeline_x1, timeline_y1), (timeline_x2, timeline_y2), (110, 110, 126), 1)
            for interval in _events_to_intervals(state["events"], total_frames, fps):
                start_frame = int(interval["start_frame"])
                end_frame = int(interval["end_frame"])
                if total_frames > 1:
                    ix1 = timeline_x1 + int(round((start_frame / (total_frames - 1)) * (timeline_x2 - timeline_x1)))
                    ix2 = timeline_x1 + int(round((end_frame / (total_frames - 1)) * (timeline_x2 - timeline_x1)))
                else:
                    ix1, ix2 = timeline_x1, timeline_x2
                cv2.rectangle(canvas, (ix1, timeline_y1 + 2), (max(ix1 + 1, ix2), timeline_y2 - 2), (203, 98, 160), -1)
            if total_frames > 1:
                marker_x = timeline_x1 + int(round((frame_idx / (total_frames - 1)) * (timeline_x2 - timeline_x1)))
            else:
                marker_x = timeline_x1
            cv2.line(canvas, (marker_x, timeline_y1 - 4), (marker_x, timeline_y2 + 4), (255, 255, 255), 1)

            cv2.imshow(window_name, canvas)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            wait_ms = 30 if not state["playing"] else max(1, int(round(1000.0 / max(0.1, fps * max(0.1, speed)))))
            key = cv2.waitKeyEx(wait_ms)
            if 65 <= key <= 90:
                key += 32  # letters arrive uppercase when Caps Lock or Shift is on

            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                state["playing"] = not state["playing"]
            elif key == ord("f"):
                push_history()
                state["events"] = _toggle_from_frame(state["events"], frame_idx)
            elif key == ord("u"):
                if history:
                    state["events"] = history.pop()
            elif key == ord("r"):
                push_history()
                state["events"] = []
            elif key == ord("a") or key in _LEFT_KEYS:
                state["playing"] = False
                state["frame_idx"] = max(0, frame_idx - 1)
                continue
            elif key == ord("d") or key in _RIGHT_KEYS:
                state["playing"] = False
                state["frame_idx"] = min(max(0, total_frames - 1), frame_idx + 1)
                continue
            elif key in (ord("s"), 13):
                labels = _events_to_labels(state["events"], total_frames)
                out_df = pd.DataFrame(
                    {
                        "frame": np.arange(total_frames, dtype=int),
                        "time_sec": (np.arange(total_frames, dtype=float) / fps) if fps > 0 else np.arange(total_frames, dtype=float),
                        "pred": labels.astype(np.uint8),
                    }
                )
                out_df.to_csv(out_csv, index=False)
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "video": str(video_path),
                            "fps": fps,
                            "n_frames": total_frames,
                            "events": [{"frame": int(fm), "state": int(st)} for fm, st in state["events"]],
                            "intervals": _events_to_intervals(state["events"], total_frames, fps),
                        },
                        f,
                        indent=2,
                    )
                state["saved"] = True
                print(f"[label] saved {out_csv}")
                print(f"[label] saved {out_json}")
                break

            if state["playing"]:
                if frame_idx >= total_frames - 1:
                    state["playing"] = False
                else:
                    state["frame_idx"] = frame_idx + 1
            else:
                state["frame_idx"] = frame_idx
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return {
        "saved": bool(state["saved"]),
        "out_csv": str(out_csv),
        "out_json": str(out_json),
        "n_frames": int(total_frames),
        "fps": float(fps),
    }
