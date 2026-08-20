from pathlib import Path
from typing import Union

import numpy as np

from .draw import blend_box as _blend_box
from .draw import mmss_tenths


def annotate_video(video_in: Union[str, Path], video_out: Union[str, Path], probs: np.ndarray, pred: np.ndarray, fps: int) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_in}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_width = max(210, min(300, width // 3))
    out = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width + panel_width, height),
    )
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video for writing: {video_out}")

    try:
        _annotate_frames(cv2, cap, out, probs, pred, fps, width, height, panel_width)
    finally:
        cap.release()
        out.release()


def _annotate_frames(cv2, cap, out, probs, pred, fps, width, height, panel_width) -> None:
    n = min(len(probs), len(pred))
    i = 0
    cumulative_freezing_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        canvas = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
        canvas[:, panel_width:] = frame
        canvas[:, :panel_width] = (18, 18, 24)
        cv2.line(canvas, (panel_width - 1, 0), (panel_width - 1, height), (90, 90, 105), 1)
        if i < n:
            prob = float(max(0.0, min(1.0, float(probs[i]))))
            freezing_confidence = prob * 100.0
            moving_confidence = 100.0 - freezing_confidence
            freezing = bool(int(pred[i]))
            if freezing:
                cumulative_freezing_frames += 1
            status = "FREEZING" if freezing else "MOVING"
            freeze_color = (52, 84, 235)
            moving_color = (60, 179, 113)
            accent = freeze_color if freezing else moving_color
            text_main = (245, 245, 245)
            text_sub = (210, 210, 210)

            panel_margin = 12
            inner_width = panel_width - panel_margin * 2
            panel_height = 150
            panel_x1 = panel_margin
            panel_y1 = panel_margin
            panel_x2 = panel_x1 + inner_width
            panel_y2 = panel_y1 + panel_height

            _blend_box(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (28, 28, 36), 0.88)
            cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (90, 90, 105), 1)

            badge_x1 = panel_x1 + 10
            badge_y1 = panel_y1 + 10
            badge_x2 = min(panel_x2 - 10, badge_x1 + 132)
            badge_y2 = badge_y1 + 26
            _blend_box(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), accent, 0.95)
            cv2.putText(canvas, status, (badge_x1 + 8, badge_y1 + 18), cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

            bar_x1 = panel_x1 + 10
            bar_x2 = panel_x2 - 10

            confidence_rows = [
                ("Freezing", freezing_confidence, freeze_color, panel_y1 + 54),
                ("Moving", moving_confidence, moving_color, panel_y1 + 100),
            ]
            for label, confidence, color, row_y in confidence_rows:
                cv2.putText(canvas, label, (bar_x1, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.38, text_sub, 1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    f"{confidence:5.1f}%",
                    (bar_x2 - 72, row_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.42,
                    text_main,
                    1,
                    cv2.LINE_AA,
                )

                bar_y1 = row_y + 10
                bar_y2 = bar_y1 + 13
                cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (70, 70, 84), 1)
                _blend_box(canvas, (bar_x1 + 1, bar_y1 + 1), (bar_x2 - 1, bar_y2 - 1), (38, 38, 48), 0.85)
                fill_x2 = bar_x1 + int((bar_x2 - bar_x1) * (confidence / 100.0))
                if fill_x2 > bar_x1:
                    _blend_box(canvas, (bar_x1 + 1, bar_y1 + 1), (fill_x2, bar_y2 - 1), color, 0.95)
                for tick in range(1, 5):
                    tick_x = bar_x1 + int((bar_x2 - bar_x1) * tick / 5)
                    cv2.line(canvas, (tick_x, bar_y1 + 2), (tick_x, bar_y2 - 2), (95, 95, 108), 1)

            seconds = i / float(fps) if fps > 0 else 0.0
            footer = f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
            frame_label = f"frame {i + 1:,}"
            freeze_seconds = cumulative_freezing_frames / float(fps) if fps > 0 else float(cumulative_freezing_frames)
            freeze_label = f"freeze {mmss_tenths(freeze_seconds)}"

            cv2.putText(canvas, freeze_label, (panel_x1 + 10, height - 52), cv2.FONT_HERSHEY_DUPLEX, 0.36, text_main, 1, cv2.LINE_AA)
            cv2.putText(canvas, footer, (panel_x1 + 10, height - 30), cv2.FONT_HERSHEY_DUPLEX, 0.48, text_main, 1, cv2.LINE_AA)
            cv2.putText(canvas, frame_label, (panel_x1 + 10, height - 12), cv2.FONT_HERSHEY_DUPLEX, 0.34, text_sub, 1, cv2.LINE_AA)
        out.write(canvas)
        i += 1
