import numpy as np
import pytest

from freezing_dlc.draw import blend_box, mmss_tenths


def test_blend_box_touches_only_the_box():
    pytest.importorskip("cv2")
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    blend_box(frame, (2, 3), (5, 6), (100, 200, 50), 0.5)

    assert frame[3, 2].tolist() == [50, 100, 25]
    assert frame[5, 4].tolist() == [50, 100, 25]
    # One pixel outside every edge stays black; the old implementation copied and
    # re-blended the whole canvas to draw one box.
    assert frame[2, 2].tolist() == [0, 0, 0]
    assert frame[6, 2].tolist() == [0, 0, 0]
    assert frame[3, 1].tolist() == [0, 0, 0]
    assert frame[3, 5].tolist() == [0, 0, 0]


def test_blend_box_clamps_out_of_bounds_corners():
    pytest.importorskip("cv2")
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    blend_box(frame, (-5, -5), (10, 10), (10, 20, 30), 1.0)

    assert (frame == np.array([10, 20, 30], dtype=np.uint8)).all()


def test_blend_box_ignores_an_empty_box():
    pytest.importorskip("cv2")
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    blend_box(frame, (3, 3), (3, 3), (255, 255, 255), 1.0)
    assert frame.sum() == 0


def test_mmss_tenths_formats_like_the_panels():
    assert mmss_tenths(0.0) == "00:00.0"
    assert mmss_tenths(65.49) == "01:05.4"
    assert mmss_tenths(600.0) == "10:00.0"
