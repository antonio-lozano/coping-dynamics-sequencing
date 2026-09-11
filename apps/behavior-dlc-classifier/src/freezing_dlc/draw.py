"""Small OpenCV drawing helpers shared by every video renderer.

Three renderers (freezing annotator, behavior annotator, manual labeler) paint
the same translucent boxes and print time the same way. One copy each, so the
panels cannot drift apart.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def blend_box(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    """Paint a translucent solid box, touching only the pixels it covers.

    Blending just the box region instead of a full-frame copy is what keeps
    this affordable inside per-frame render loops.
    """
    import cv2

    height, width = frame.shape[:2]
    x1, y1 = max(0, int(top_left[0])), max(0, int(top_left[1]))
    x2, y2 = min(width, int(bottom_right[0])), min(height, int(bottom_right[1]))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    box = np.empty_like(roi)
    box[:] = color
    cv2.addWeighted(box, alpha, roi, 1.0 - alpha, 0, dst=roi)


def mmss_tenths(seconds: float) -> str:
    """A duration as MM:SS.t, the way every panel prints time."""
    whole = int(seconds)
    return f"{whole // 60:02d}:{whole % 60:02d}.{int((seconds - whole) * 10)}"
