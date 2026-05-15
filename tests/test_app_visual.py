from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from gui.app_visual import (
    build_feature_bundle,
    build_pose_bundle,
    feature_axis_seconds,
    flatten_tracking_xy_df,
    frame_from_time,
    right_video_dir_candidates,
    resolve_fps,
)

try:
    import tables  # noqa: F401

    HAVE_PYTABLES = True
except Exception:
    HAVE_PYTABLES = False


def _make_tracking_df(n_frames: int = 6) -> pd.DataFrame:
    index = pd.RangeIndex(n_frames)
    columns = pd.MultiIndex.from_tuples(
        [
            ("scorerA", "nose", "x"),
            ("scorerA", "nose", "y"),
            ("scorerA", "nose", "likelihood"),
            ("scorerA", "tail", "x"),
            ("scorerA", "tail", "y"),
            ("scorerA", "tail", "likelihood"),
            ("scorerA", "S1", "x"),
            ("scorerA", "S1", "y"),
            ("scorerA", "S1", "likelihood"),
            ("scorerA", "S2", "x"),
            ("scorerA", "S2", "y"),
            ("scorerA", "S2", "likelihood"),
        ],
        names=["scorer", "bodyparts", "coords"],
    )
    data = []
    for frame in range(n_frames):
        data.append(
            [
                frame + 0.0,
                frame + 1.0,
                0.99,
                frame + 5.0,
                frame + 2.0,
                0.98,
                frame + 2.0,
                frame + 2.5,
                0.97,
                frame + 3.0,
                frame + 2.8,
                0.96,
            ]
        )
    return pd.DataFrame(data, index=index, columns=columns)


class AppVisualHelperTests(unittest.TestCase):
    def test_frame_from_time_clamps_and_rounds(self) -> None:
        self.assertEqual(frame_from_time(0.0, fps=25.0, frame_count=10), 0)
        self.assertEqual(frame_from_time(0.021, fps=25.0, frame_count=10), 1)
        self.assertEqual(frame_from_time(10.0, fps=25.0, frame_count=10), 9)

    def test_resolve_fps_prefers_manifest_then_video_then_fallback(self) -> None:
        manifest_maps = [{"animal114dlcresnet50freezing072020sep29shuffle1100000": 29.97}]
        stem = "Animal 11_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000"

        self.assertAlmostEqual(resolve_fps(stem, manifest_maps, video_fps=60.0, fallback=25.0), 29.97)
        self.assertAlmostEqual(resolve_fps("missing_recording", [], video_fps=61.5, fallback=25.0), 61.5)
        self.assertAlmostEqual(resolve_fps("missing_recording", [], video_fps=None, fallback=25.0), 25.0)

    def test_right_video_candidates_prioritize_map_aya_raw_source(self) -> None:
        expected = Path(r"C:\aya\encoded")
        with mock.patch("gui.app_visual._resolve_default_input_dir", return_value=expected):
            candidates = right_video_dir_candidates()
        self.assertEqual(candidates[0], expected)

    def test_csv_pose_loading_extracts_xy_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "example_pose.csv"
            df = _make_tracking_df()
            df.to_csv(csv_path)

            pose = build_pose_bundle(csv_path, fps=25.0)

            self.assertEqual(pose.frame_count, len(df))
            self.assertIn("nose_x", pose.flat_xy_df.columns)
            self.assertIn("tail_y", pose.flat_xy_df.columns)
            self.assertEqual(set(["nose", "tail", "S1", "S2"]).issubset(set(pose.bodyparts)), True)

    @unittest.skipUnless(HAVE_PYTABLES, "PyTables is required to read/write DLC H5 pose files")
    def test_h5_pose_loading_extracts_xy_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "example_pose.h5"
            df = _make_tracking_df()
            df.to_hdf(h5_path, key="tracks")

            pose = build_pose_bundle(h5_path, fps=25.0)

            self.assertEqual(pose.frame_count, len(df))
            self.assertIn("S1_x", pose.flat_xy_df.columns)
            self.assertIn("S2_y", pose.flat_xy_df.columns)

    def test_flatten_tracking_xy_df_keeps_only_xy(self) -> None:
        flat = flatten_tracking_xy_df(_make_tracking_df())
        self.assertEqual(list(flat.columns), ["nose_x", "nose_y", "tail_x", "tail_y", "S1_x", "S1_y", "S2_x", "S2_y"])

    def test_feature_fallback_computes_frame_aligned_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "example_pose.csv"
            _make_tracking_df(n_frames=8).to_csv(csv_path)

            pose = build_pose_bundle(csv_path, fps=25.0)
            features = build_feature_bundle(None, pose_bundle=pose, recording_stem=csv_path.stem, fps=25.0)

            self.assertEqual(features.frame_count, pose.frame_count)
            self.assertIn("Summed Velocity", features.feature_cols)
            self.assertIn("Angular Velocity", features.feature_cols)
            self.assertIn("Nose - Tail Distance", features.feature_cols)
            self.assertTrue(np.allclose(feature_axis_seconds(features.df, 25.0)[:3], np.array([0.0, 0.04, 0.08])))


if __name__ == "__main__":
    unittest.main()
