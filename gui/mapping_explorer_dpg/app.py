"""DearPyGui Aya<->Jen mapping explorer."""
from __future__ import annotations

import difflib
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocessing.map_aya_to_jen import (  # noqa: E402
    DEFAULT_JEN_TO_AYA_MAP,
    JEN_BODY_ORDER,
    build_renamed_df,
    first_scorer,
    load_tracking_df,
)
from src.ml.pose_features import DEFAULT_IMPORTANT_FEATURES  # noqa: E402
from scripts.analysis.extended_features import (  # noqa: E402
    DEFAULT_REF_PAIR,
    preprocess_coordinates,
)

JEN_COLORS = {
    "nose": (230, 57, 70, 255),
    "H1R": (244, 162, 89, 255),
    "H2R": (194, 197, 204, 255),
    "H1L": (255, 159, 28, 255),
    "H2L": (194, 197, 204, 255),
    "B1R": (184, 188, 199, 255),
    "B2R": (42, 157, 143, 255),
    "B3R": (184, 188, 199, 255),
    "B1L": (184, 188, 199, 255),
    "B2L": (29, 120, 116, 255),
    "B3L": (184, 188, 199, 255),
    "tail": (58, 134, 255, 255),
    "S2": (184, 188, 199, 255),
    "S1": (141, 90, 151, 255),
}

AYA_COLORS = {
    "nose": (214, 40, 40, 255),
    "right_ear": (244, 162, 89, 255),
    "left_ear": (255, 191, 105, 255),
    "right_lateral": (42, 157, 143, 255),
    "left_lateral": (29, 120, 116, 255),
    "tail_base": (58, 134, 255, 255),
    "Centroid": (141, 90, 151, 255),
    "tail_end": (125, 133, 151, 255),
}

# Behavior class colors (matching manuscript figure palette)
BEHAVIOR_COLORS: dict[str, tuple[int, int, int, int]] = {
    "Freezing": (195, 123, 159, 255),
    "Sniffing": (76, 183, 165, 255),
    "Grooming": (126, 200, 227, 255),
    "Turn": (141, 198, 63, 255),
    "Locomotion": (248, 198, 80, 255),
    "Climbing": (244, 162, 89, 255),
    "Jump": (228, 87, 46, 255),
    "Unassigned": (160, 160, 160, 255),
}
BEHAVIOR_COLOR_DEFAULT = (80, 80, 80, 255)



def _normalize_recording_token(name: str) -> str:
    token = Path(name).stem.lower()
    token = token.replace("_matched_to_jen", "")
    token = re.sub(r"[^a-z0-9]+", "", token)
    return token


def _list_pose_files(directory: Path) -> list[str]:
    if not directory.exists() or not directory.is_dir():
        return []
    files = list(directory.glob("*.csv")) + list(directory.glob("*.h5"))
    dedup = {f.name: f for f in files}
    return sorted(dedup.keys(), key=str.lower)


def _discover_feature_runs(predictions_root: Path) -> list[str]:
    if not predictions_root.exists() or not predictions_root.is_dir():
        return []
    runs = set()
    for p in predictions_root.rglob("computed_features_raw.parquet"):
        try:
            runs.add(str(p.parent.relative_to(predictions_root)))
        except ValueError:
            runs.add(str(p.parent))
    for p in predictions_root.rglob("computed_features_raw.csv"):
        try:
            runs.add(str(p.parent.relative_to(predictions_root)))
        except ValueError:
            runs.add(str(p.parent))
    return sorted(runs, key=str.lower)


def _load_feature_table(run_dir: Path, mode: str) -> pd.DataFrame:
    if mode == "Model Aligned":
        parquet_path = run_dir / "computed_features_model_aligned.parquet"
        csv_path = run_dir / "computed_features_model_aligned.csv"
    else:
        parquet_path = run_dir / "computed_features_raw.parquet"
        csv_path = run_dir / "computed_features_raw.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No computed feature table found in {run_dir} for mode={mode}.")


def _load_predictions_table(run_dir: Path) -> pd.DataFrame:
    parquet_path = run_dir / "predictions_frame.parquet"
    csv_path = run_dir / "predictions_frame.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No predictions_frame table found in {run_dir}.")


def _match_recording_slice(df: pd.DataFrame, pose_stem: str) -> pd.DataFrame:
    if "recording" not in df.columns:
        return df.copy()
    token = _normalize_recording_token(pose_stem)
    rec = df["recording"].astype(str)
    rec_tok = rec.map(_normalize_recording_token)

    mask = rec_tok == token
    if not mask.any():
        mask = rec_tok.map(lambda t: token in t or t in token)
    if not mask.any():
        # Fallback: keep first recording block.
        first_rec = rec.iloc[0]
        mask = rec == first_rec
    return df.loc[mask].copy().reset_index(drop=True)


def _suggest_match(target_file: str, candidates: list[str]) -> str | None:
    if not candidates:
        return None
    target = _normalize_recording_token(target_file)
    best_name = candidates[0]
    best_score = -1.0
    for cand in candidates:
        cand_token = _normalize_recording_token(cand)
        if cand_token == target:
            return cand
        contains = target in cand_token or cand_token in target
        ratio = difflib.SequenceMatcher(a=target, b=cand_token).ratio()
        score = ratio + (0.2 if contains else 0.0)
        if score > best_score:
            best_score = score
            best_name = cand
    return best_name


def _load_fps_manifest(directory: Path) -> dict[str, float]:
    manifests = [directory / "fps_manifest.csv"]
    if directory.parent != directory:
        manifests.append(directory.parent / "fps_manifest.csv")
    out: dict[str, float] = {}
    for m in manifests:
        if not m.exists():
            continue
        try:
            df = pd.read_csv(m)
        except Exception:
            continue
        if "recording" not in df.columns or "fps" not in df.columns:
            continue
        for _, row in df.iterrows():
            fps = pd.to_numeric(row["fps"], errors="coerce")
            if pd.notna(fps) and float(fps) > 0:
                out[_normalize_recording_token(str(row["recording"]))] = float(fps)
    return out


def _resolve_fps(stem: str, manifest: dict[str, float], manual: float | None, default: float = 25.0) -> float:
    if manual is not None and manual > 0:
        return float(manual)
    token = _normalize_recording_token(stem)
    if token in manifest:
        return float(manifest[token])
    return float(default)


def _axis_bounds(xy_by_part: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    rows = []
    for arr in xy_by_part.values():
        if arr.size == 0:
            continue
        finite = np.isfinite(arr).all(axis=1)
        if finite.any():
            rows.append(arr[finite])
    if not rows:
        return (-1.0, 1.0, -1.0, 1.0)
    merged = np.vstack(rows)
    x_min = float(np.nanmin(merged[:, 0]))
    x_max = float(np.nanmax(merged[:, 0]))
    y_min = float(np.nanmin(merged[:, 1]))
    y_max = float(np.nanmax(merged[:, 1]))
    pad = max(x_max - x_min, y_max - y_min, 1.0) * 0.14
    return (x_min - pad, x_max + pad, y_min - pad, y_max + pad)


@dataclass
class PoseBundle:
    df: pd.DataFrame
    scorer: str
    bodyparts: list[str]
    xy: dict[str, np.ndarray]
    n_frames: int
    fps: float
    axis_bounds: tuple[float, float, float, float]
    source_label: str
    unit_label: str = "px"
    ref_dist_px: float | None = None
    ref_pair: tuple[str, str] | None = None

    def frame_to_time(self, frame: int) -> float:
        return float(frame) / max(self.fps, 1e-8)

    def time_to_frame(self, time_s: float) -> int:
        if self.n_frames <= 0:
            return 0
        frame = int(round(float(time_s) * max(self.fps, 1e-8)))
        return max(0, min(self.n_frames - 1, frame))


def _pose_bundle_from_df(
    df: pd.DataFrame,
    fps: float,
    source_label: str,
    *,
    unit_label: str = "px",
    ref_dist_px: float | None = None,
    ref_pair: tuple[str, str] | None = None,
) -> PoseBundle:
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected DLC MultiIndex columns (scorer/bodyparts/coords).")
    scorer = first_scorer(df)
    bps: list[str] = []
    for value in df.columns.get_level_values(1).unique().tolist():
        bp = str(value).strip()
        if bp.lower() in {"bodyparts", "unnamed: 0", ""}:
            continue
        if bp not in bps:
            bps.append(bp)

    n_frames = int(len(df))
    xy: dict[str, np.ndarray] = {}
    for bp in bps:
        x_col = (scorer, bp, "x")
        y_col = (scorer, bp, "y")
        arr = np.full((n_frames, 2), np.nan, dtype=float)
        if x_col in df.columns and y_col in df.columns:
            arr[:, 0] = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
            arr[:, 1] = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        xy[bp] = arr

    fps_value = float(fps) if float(fps) > 0 else 25.0
    return PoseBundle(
        df=df,
        scorer=scorer,
        bodyparts=bps,
        xy=xy,
        n_frames=n_frames,
        fps=fps_value,
        axis_bounds=_axis_bounds(xy),
        source_label=source_label,
        unit_label=unit_label,
        ref_dist_px=ref_dist_px,
        ref_pair=ref_pair,
    )


def _normalize_pose_df(
    df: pd.DataFrame,
    *,
    preferred_ref_pair: tuple[str, str] = DEFAULT_REF_PAIR,
) -> tuple[pd.DataFrame, float | None, tuple[str, str] | None]:
    """Normalize MultiIndex DLC dataframe using robust pipeline preprocessing."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df.copy(), None, None
    scorer = first_scorer(df)
    bodyparts = []
    for value in df.columns.get_level_values(1).unique().tolist():
        bp = str(value).strip()
        if bp.lower() in {"bodyparts", "unnamed: 0", ""}:
            continue
        bodyparts.append(bp)

    flat = pd.DataFrame(index=df.index)
    for bp in bodyparts:
        x_col = (scorer, bp, "x")
        y_col = (scorer, bp, "y")
        if x_col in df.columns and y_col in df.columns:
            flat[f"{bp}_x"] = pd.to_numeric(df[x_col], errors="coerce")
            flat[f"{bp}_y"] = pd.to_numeric(df[y_col], errors="coerce")
    if flat.empty:
        return df.copy(), None, None

    candidates: list[tuple[str, str]] = [preferred_ref_pair, ("nose", "Centroid"), ("Centroid", "tail_base"), ("nose", "tail")]
    used_pair: tuple[str, str] | None = None
    norm_flat = None
    ref_dist: float | None = None
    for pair in candidates:
        if f"{pair[0]}_x" not in flat.columns or f"{pair[1]}_x" not in flat.columns:
            continue
        try:
            norm_flat, ref_dist = preprocess_coordinates(flat, ref_bodyparts=pair)
            used_pair = pair
            break
        except Exception:
            continue
    if norm_flat is None:
        return df.copy(), None, None

    out = df.copy()
    for bp in bodyparts:
        x_name = f"{bp}_x"
        y_name = f"{bp}_y"
        x_col = (scorer, bp, "x")
        y_col = (scorer, bp, "y")
        if x_name in norm_flat.columns and y_name in norm_flat.columns and x_col in out.columns and y_col in out.columns:
            out[x_col] = norm_flat[x_name].to_numpy(dtype=float)
            out[y_col] = norm_flat[y_name].to_numpy(dtype=float)
    return out, (float(ref_dist) if ref_dist is not None else None), used_pair


class ExplorerApp:
    def __init__(self) -> None:
        self.jen_dir = str(REPO_ROOT / "data" / "to_predict" / "test")
        self.aya_dir = str(REPO_ROOT / "data" / "to_predict" / "matched_dlc_aya_to_jen")

        self.jen_files: list[str] = []
        self.aya_files: list[str] = []
        self.jen_file: str | None = None
        self.aya_file: str | None = None

        self.jen_bundle: PoseBundle | None = None
        self.jen_bundle_raw: PoseBundle | None = None
        self.jen_bundle_norm: PoseBundle | None = None
        self.aya_raw_bundle: PoseBundle | None = None
        self.aya_raw_bundle_raw: PoseBundle | None = None
        self.aya_raw_bundle_norm: PoseBundle | None = None
        self.aya_display_bundle: PoseBundle | None = None
        self.aya_display_bundle_raw: PoseBundle | None = None
        self.aya_display_bundle_norm: PoseBundle | None = None
        self.aya_mapping_details: dict[str, Any] = {}
        self.mapping_profile = dict(DEFAULT_JEN_TO_AYA_MAP)

        self.timeline_mode = "Frame"
        self.link_bars = False
        self.auto_pair = True
        self.aya_view_mode = "Raw Aya"
        self.pose_coord_mode = "Raw px"
        self.manual_fps_enabled = False
        self.manual_fps_jen: float | None = None
        self.manual_fps_aya: float | None = None

        self.left_frame = 0
        self.right_frame = 0
        self.left_time = 0.0
        self.right_time = 0.0
        self.left_zoom = 1.0
        self.right_zoom = 1.0
        self.show_hull = False
        self._suspend = False
        self._last_side = "left"
        self.sidebar_w = 430
        self.canvas_w = 580
        self.canvas_h = 380
        self._last_vp_w = 1700
        self._last_vp_h = 980

        self.playing = False
        self.play_mode = "none"  # one of: none, left, right, both
        self._last_render_time: float | None = None
        self._left_play_accum = 0.0
        self._right_play_accum = 0.0

        self.pred_root = str(REPO_ROOT / "results" / "predictions")
        self.feature_mode = "Raw"
        self.feature_runs: list[str] = []
        self.jen_feature_run: str | None = None
        self.aya_feature_run: str | None = None
        self.jen_features_df: pd.DataFrame | None = None
        self.aya_features_df: pd.DataFrame | None = None
        self.jen_feature_cols: list[str] = []
        self.aya_feature_cols: list[str] = []
        self.jen_features_selected: list[str] = []
        self.aya_features_selected: list[str] = []
        self.jen_predictions_df: pd.DataFrame | None = None
        self.aya_predictions_df: pd.DataFrame | None = None
        self.jen_pred_frame_map: dict[int, int] = {}
        self.aya_pred_frame_map: dict[int, int] = {}
        self.jen_pred_sorted_frames: np.ndarray = np.array([], dtype=int)
        self.aya_pred_sorted_frames: np.ndarray = np.array([], dtype=int)
        self.jen_pred_mix_text = "Prediction mix: n/a"
        self.aya_pred_mix_text = "Prediction mix: n/a"
        self._feature_plot_signatures: dict[str, tuple[Any, ...] | None] = {"jen": None, "aya": None}
        self.feature_stream_window_frames = 600.0
        self.feature_stream_window_seconds = 12.0

    def run(self) -> None:
        dpg.create_context()
        self._build_ui()
        dpg.create_viewport(
            title="Aya<->Jen Mapping Explorer (DearPyGui)",
            width=1700,
            height=980,
            min_width=1200,
            min_height=760,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        if hasattr(dpg, "set_viewport_clear_color"):
            dpg.set_viewport_clear_color((0, 0, 0, 255))
        self.refresh_file_lists()
        self.refresh_feature_runs()
        self.reload_pose_data()
        self._schedule_frame_tick()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def _schedule_frame_tick(self) -> None:
        if dpg.is_dearpygui_running():
            dpg.set_frame_callback(dpg.get_frame_count() + 1, self.on_render_frame)

    # ---- Responsive layout helpers --------------------------------
    def _calc_layout(self, vp_w: int, vp_h: int) -> None:
        """Recompute layout dimensions from viewport size."""
        self._last_vp_w = vp_w
        self._last_vp_h = vp_h
        content_w = max(800, vp_w - self.sidebar_w - 40)  # 40 = padding/scrollbar
        self.canvas_w = max(200, (content_w - 30) // 2)     # two side-by-side panels
        panel_w = self.canvas_w + 20                       # child_window margin
        content_h = max(600, vp_h)
        self.canvas_h = max(180, int(content_h * 0.38))
        self._panel_w = panel_w
        self._ethogram_w = self.canvas_w
        self._plot_w = self.canvas_w - 10

    def _apply_layout(self) -> None:
        """Push current dimensions into existing DPG items."""
        pw = self._panel_w
        cw, ch = self.canvas_w, self.canvas_h
        ew = self._ethogram_w
        plotw = self._plot_w

        for tag in ("left_panel_child", "right_panel_child"):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, width=pw, height=ch + 80)
        for tag in ("left_canvas", "right_canvas"):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, width=cw, height=ch)
        for tag in ("left_control_child", "right_control_child"):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, width=pw)
        for tag in ("jen_ethogram_child", "aya_ethogram_child"):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, width=pw)
        for tag in ("jen_ethogram_canvas", "aya_ethogram_canvas"):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, width=ew, height=50)
        for tag in ("jen_feature_child", "aya_feature_child"):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, width=pw)
        for tag in ("jen_feature_plot", "aya_feature_plot"):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, width=plotw)

    def _build_ui(self) -> None:
        self._calc_layout(1700, 980)
        pw = self._panel_w
        cw, ch = self.canvas_w, self.canvas_h
        ew = self._ethogram_w
        plotw = self._plot_w

        with dpg.window(tag="main_window", label="Aya<->Jen Mapping Explorer"):
            dpg.add_text("Keypoint explorer with computed-feature traces from robust pipeline outputs.")
            dpg.add_separator()

            with dpg.group(horizontal=True):
                with dpg.child_window(width=self.sidebar_w, autosize_y=True):
                    dpg.add_text("File Sources")
                    dpg.add_input_text(label="Jen pose directory", tag="jen_dir_input", default_value=self.jen_dir)
                    dpg.add_input_text(label="Aya pose directory", tag="aya_dir_input", default_value=self.aya_dir)
                    with dpg.group(horizontal=True):
                        dpg.add_checkbox(label="Auto-pair pose files", tag="auto_pair_checkbox", default_value=True, callback=self.on_auto_pair_toggle)
                        dpg.add_button(label="Refresh files", callback=self.on_refresh_files)

                    dpg.add_combo(label="Jen pose file", tag="jen_file_combo", items=[], callback=self.on_jen_file_changed, width=400)
                    dpg.add_combo(label="Aya pose file", tag="aya_file_combo", items=[], callback=self.on_aya_file_changed, width=400)

                    dpg.add_separator()
                    dpg.add_text("Feature Sources")
                    dpg.add_input_text(
                        label="Predictions root",
                        tag="pred_root_input",
                        default_value=self.pred_root,
                        width=400,
                    )
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Refresh runs", callback=self.on_refresh_feature_runs)
                        dpg.add_radio_button(
                            items=["Raw", "Model Aligned"],
                            tag="feature_mode_radio",
                            default_value="Raw",
                            horizontal=True,
                            callback=self.on_feature_mode_changed,
                        )
                    dpg.add_combo(
                        label="Jen feature run",
                        tag="jen_feature_run_combo",
                        items=[],
                        callback=self.on_jen_feature_run_changed,
                        width=400,
                    )
                    dpg.add_combo(
                        label="Aya feature run",
                        tag="aya_feature_run_combo",
                        items=[],
                        callback=self.on_aya_feature_run_changed,
                        width=400,
                    )
                    dpg.add_text("Feature status: waiting for run selection.", tag="feature_status_text")

                    dpg.add_separator()
                    dpg.add_text("Timeline & View")
                    dpg.add_radio_button(items=["Frame", "Time (s)"], tag="timeline_mode_radio", default_value="Frame", horizontal=True, callback=self.on_timeline_mode_changed)
                    dpg.add_checkbox(label="Link bars", tag="link_bars_checkbox", default_value=False, callback=self.on_link_changed)
                    dpg.add_radio_button(items=["Raw Aya", "Mapped to Jen"], tag="aya_view_mode_radio", default_value="Raw Aya", horizontal=True, callback=self.on_aya_view_mode_changed)
                    dpg.add_radio_button(
                        items=["Raw px", "Normalized pose"],
                        tag="pose_coord_mode_radio",
                        default_value="Raw px",
                        horizontal=True,
                        callback=self.on_pose_coord_mode_changed,
                    )
                    dpg.add_checkbox(
                        label="Show convex hull (yellow, alpha 0.5)",
                        tag="show_hull_checkbox",
                        default_value=False,
                        callback=self.on_hull_toggle,
                    )
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Reset Left View", callback=self.on_reset_left_view)
                        dpg.add_button(label="Reset Right View", callback=self.on_reset_right_view)
                        dpg.add_button(label="Reset Both Views", callback=self.on_reset_both_views)
                    dpg.add_checkbox(label="Manual FPS override", tag="manual_fps_checkbox", default_value=False, callback=self.on_manual_fps_toggle)
                    dpg.add_input_float(label="Jen FPS", tag="jen_fps_input", default_value=25.0, step=0.5, callback=self.on_manual_fps_changed, enabled=False)
                    dpg.add_input_float(label="Aya FPS", tag="aya_fps_input", default_value=25.0, step=0.5, callback=self.on_manual_fps_changed, enabled=False)
                    dpg.add_text("Resolved FPS: Jen=25.000, Aya=25.000", tag="fps_status_text")

                    dpg.add_separator()
                    dpg.add_text("Live Mapping Editor")
                    dpg.add_text("Select Aya source point for each JEN target.")
                    with dpg.child_window(tag="mapping_editor_child", width=410, height=360):
                        pass

                    dpg.add_separator()
                    dpg.add_text("Mapping Status (current Aya frame)")
                    with dpg.child_window(tag="mapping_status_child", width=410, height=190):
                        pass

                with dpg.group(tag="right_panel_group"):
                    dpg.add_text("Top: keypoints | Bottom: computed feature traces")
                    with dpg.group(horizontal=True):
                        with dpg.child_window(tag="left_panel_child", width=pw, height=ch + 80):
                            dpg.add_text("Jen", tag="left_panel_title")
                            dpg.add_drawlist(width=cw, height=ch, tag="left_canvas")
                            dpg.add_slider_float(
                                label="Left zoom",
                                tag="left_zoom_slider",
                                default_value=1.0,
                                min_value=1.0,
                                max_value=12.0,
                                callback=self.on_left_zoom_changed,
                                width=-1,
                            )
                            dpg.add_text("Scale: n/a", tag="left_scale_text", color=(190, 190, 190, 255))
                        with dpg.child_window(tag="right_panel_child", width=pw, height=ch + 80):
                            dpg.add_text("Aya", tag="right_panel_title")
                            dpg.add_drawlist(width=cw, height=ch, tag="right_canvas")
                            dpg.add_slider_float(
                                label="Right zoom",
                                tag="right_zoom_slider",
                                default_value=1.0,
                                min_value=1.0,
                                max_value=12.0,
                                callback=self.on_right_zoom_changed,
                                width=-1,
                            )
                            dpg.add_text("Scale: n/a", tag="right_scale_text", color=(190, 190, 190, 255))

                    with dpg.group(horizontal=True):
                        with dpg.child_window(tag="left_control_child", width=pw, height=150):
                            dpg.add_text("Jen frame / time")
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Play Left", tag="play_left_button", callback=self.on_play_left)
                                dpg.add_button(label="Play Right", tag="play_right_button", callback=self.on_play_right)
                                dpg.add_button(label="Play Both (Sync)", tag="play_both_button", callback=self.on_play_both)
                                dpg.add_button(label="Stop", tag="stop_play_button", callback=self.on_stop_play)
                                dpg.add_button(label="Step +1", callback=self.on_step_once)
                                dpg.add_slider_float(
                                    label="Speed",
                                    tag="play_speed",
                                    default_value=1.0,
                                    min_value=0.1,
                                    max_value=4.0,
                                    width=180,
                                )
                                dpg.add_checkbox(label="Loop", tag="play_loop", default_value=True)
                            dpg.add_slider_int(label="Jen frame", tag="left_frame_slider", min_value=0, max_value=1, callback=self.on_left_frame_slider)
                            dpg.add_slider_float(label="Jen time (s)", tag="left_time_slider", min_value=0.0, max_value=1.0, callback=self.on_left_time_slider, show=False)
                            dpg.add_text("Frame: 0 | Time: 0.000s", tag="left_time_readout")
                            dpg.add_text("Prediction: n/a", tag="left_pred_readout")
                            dpg.add_text("Prediction mix: n/a", tag="left_pred_mix_readout")
                        with dpg.child_window(tag="right_control_child", width=pw, height=150):
                            dpg.add_text("Aya frame / time")
                            dpg.add_slider_int(label="Aya frame", tag="right_frame_slider", min_value=0, max_value=1, callback=self.on_right_frame_slider)
                            dpg.add_slider_float(label="Aya time (s)", tag="right_time_slider", min_value=0.0, max_value=1.0, callback=self.on_right_time_slider, show=False)
                            dpg.add_text("Frame: 0 | Time: 0.000s", tag="right_time_readout")
                            dpg.add_text("Prediction: n/a", tag="right_pred_readout")
                            dpg.add_text("Prediction mix: n/a", tag="right_pred_mix_readout")

                    # --- Ethogram strips ---
                    with dpg.group(horizontal=True):
                        with dpg.child_window(tag="jen_ethogram_child", width=pw, height=80):
                            dpg.add_text("Jen ethogram (predicted behavior)", tag="jen_ethogram_label")
                            dpg.add_drawlist(width=ew, height=50, tag="jen_ethogram_canvas")
                        with dpg.child_window(tag="aya_ethogram_child", width=pw, height=80):
                            dpg.add_text("Aya ethogram (predicted behavior)", tag="aya_ethogram_label")
                            dpg.add_drawlist(width=ew, height=50, tag="aya_ethogram_canvas")

                    with dpg.group(horizontal=True):
                        with dpg.child_window(tag="jen_feature_child", width=pw, height=320):
                            dpg.add_text("Jen features")
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_combo(
                                    label="Add",
                                    tag="jen_feature_combo",
                                    items=[],
                                    callback=self.on_jen_feature_changed,
                                    width=-120,
                                )
                            with dpg.child_window(tag="jen_feature_checks", height=60, autosize_x=True):
                                pass  # checkboxes added dynamically
                            dpg.add_text("No Jen feature data loaded.", tag="jen_feature_info")
                            with dpg.plot(label="Jen Feature Plot", height=180, width=plotw, tag="jen_feature_plot"):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="jen_feature_x_axis")
                                dpg.add_plot_axis(dpg.mvYAxis, label="Value", tag="jen_feature_y_axis")
                        with dpg.child_window(tag="aya_feature_child", width=pw, height=320):
                            dpg.add_text("Aya features")
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_combo(
                                    label="Add",
                                    tag="aya_feature_combo",
                                    items=[],
                                    callback=self.on_aya_feature_changed,
                                    width=-120,
                                )
                            with dpg.child_window(tag="aya_feature_checks", height=60, autosize_x=True):
                                pass  # checkboxes added dynamically
                            dpg.add_text("No Aya feature data loaded.", tag="aya_feature_info")
                            with dpg.plot(label="Aya Feature Plot", height=180, width=plotw, tag="aya_feature_plot"):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="aya_feature_x_axis")
                                dpg.add_plot_axis(dpg.mvYAxis, label="Value", tag="aya_feature_y_axis")

    def refresh_file_lists(self) -> None:
        self.jen_dir = dpg.get_value("jen_dir_input")
        self.aya_dir = dpg.get_value("aya_dir_input")
        jen_path = Path(self.jen_dir)
        aya_path = Path(self.aya_dir)
        self.jen_files = _list_pose_files(jen_path)
        self.aya_files = _list_pose_files(aya_path)

        dpg.configure_item("jen_file_combo", items=self.jen_files)
        dpg.configure_item("aya_file_combo", items=self.aya_files)

        if self.jen_files and self.jen_file not in self.jen_files:
            self.jen_file = self.jen_files[0]
        if self.aya_files and self.aya_file not in self.aya_files:
            self.aya_file = self.aya_files[0]
        if self.auto_pair and self.jen_file and self.aya_files:
            suggestion = _suggest_match(self.jen_file, self.aya_files)
            if suggestion:
                self.aya_file = suggestion
        if self.jen_file:
            dpg.set_value("jen_file_combo", self.jen_file)
        if self.aya_file:
            dpg.set_value("aya_file_combo", self.aya_file)

    def refresh_feature_runs(self) -> None:
        self.pred_root = dpg.get_value("pred_root_input")
        root = Path(self.pred_root)
        self.feature_runs = _discover_feature_runs(root)
        dpg.configure_item("jen_feature_run_combo", items=self.feature_runs)
        dpg.configure_item("aya_feature_run_combo", items=self.feature_runs)

        if self.feature_runs:
            if self.jen_feature_run not in self.feature_runs:
                self.jen_feature_run = self._preferred_run(self.feature_runs)
            if self.aya_feature_run not in self.feature_runs:
                aya_pref = [r for r in self.feature_runs if "aya" in r.lower()]
                self.aya_feature_run = self._preferred_run(aya_pref or self.feature_runs)
        else:
            self.jen_feature_run = None
            self.aya_feature_run = None

        dpg.set_value("jen_feature_run_combo", self.jen_feature_run or "")
        dpg.set_value("aya_feature_run_combo", self.aya_feature_run or "")
        self._reload_features_for_selected_pose()

    @staticmethod
    def _preferred_run(runs: list[str]) -> str:
        if not runs:
            return ""

        def score(run: str) -> int:
            r = run.lower()
            s = 0
            if "robust_light_no_tail_v2" in r:
                s += 120
            if "aya_robust_v3" in r:
                s += 100
            if "aya_robust_v2" in r:
                s += 60
            if "robust" in r:
                s += 20
            if "check" in r:
                s -= 35
            if "v1" in r:
                s -= 20
            return s

        return sorted(runs, key=lambda r: (score(r), r.lower()), reverse=True)[0]

    def reload_pose_data(self) -> None:
        if not self.jen_file or not self.aya_file:
            return

        jen_path = Path(self.jen_dir) / self.jen_file
        aya_path = Path(self.aya_dir) / self.aya_file
        if not jen_path.exists() or not aya_path.exists():
            return

        jen_manifest = _load_fps_manifest(Path(self.jen_dir))
        aya_manifest = _load_fps_manifest(Path(self.aya_dir))
        self.manual_fps_jen = dpg.get_value("jen_fps_input") if self.manual_fps_enabled else None
        self.manual_fps_aya = dpg.get_value("aya_fps_input") if self.manual_fps_enabled else None
        jen_fps = _resolve_fps(jen_path.stem, jen_manifest, self.manual_fps_jen)
        aya_fps = _resolve_fps(aya_path.stem, aya_manifest, self.manual_fps_aya)
        dpg.set_value("fps_status_text", f"Resolved FPS: Jen={jen_fps:.3f}, Aya={aya_fps:.3f}")

        jen_df = load_tracking_df(jen_path)
        aya_df = load_tracking_df(aya_path)
        self.jen_bundle_raw = _pose_bundle_from_df(jen_df, fps=jen_fps, source_label=jen_path.stem, unit_label="px")
        jen_norm_df, jen_ref_dist_px, jen_ref_pair = _normalize_pose_df(jen_df, preferred_ref_pair=DEFAULT_REF_PAIR)
        self.jen_bundle_norm = _pose_bundle_from_df(
            jen_norm_df,
            fps=jen_fps,
            source_label=f"{jen_path.stem}_normalized",
            unit_label="BLU",
            ref_dist_px=jen_ref_dist_px,
            ref_pair=jen_ref_pair,
        )
        self.aya_raw_bundle_raw = _pose_bundle_from_df(aya_df, fps=aya_fps, source_label=aya_path.stem, unit_label="px")
        aya_norm_df, aya_ref_dist_px, aya_ref_pair = _normalize_pose_df(aya_df, preferred_ref_pair=DEFAULT_REF_PAIR)
        self.aya_raw_bundle_norm = _pose_bundle_from_df(
            aya_norm_df,
            fps=aya_fps,
            source_label=f"{aya_path.stem}_normalized",
            unit_label="BLU",
            ref_dist_px=aya_ref_dist_px,
            ref_pair=aya_ref_pair,
        )

        self._rebuild_mapping_editor_options()
        self._build_aya_display_bundle()
        self._apply_pose_coord_mode()
        self._reload_features_for_selected_pose()
        self._reset_timeline_bounds()
        self._render()

    def _load_side_features(self, run_name: str | None, pose_file: str | None) -> tuple[pd.DataFrame | None, str | None]:
        if not run_name or not pose_file:
            return None, "missing run or pose file"
        run_dir = Path(self.pred_root) / run_name
        if not run_dir.exists():
            return None, f"run folder missing: {run_name}"
        try:
            table = _load_feature_table(run_dir, self.feature_mode)
        except Exception as exc:  # pragma: no cover - runtime IO errors
            return None, str(exc)
        try:
            sliced = _match_recording_slice(table, Path(pose_file).stem)
        except Exception as exc:  # pragma: no cover - runtime matching errors
            return None, f"recording-match error: {exc}"
        return sliced, None

    def _load_side_predictions(self, run_name: str | None, pose_file: str | None) -> tuple[pd.DataFrame | None, str | None]:
        if not run_name or not pose_file:
            return None, "missing run or pose file"
        run_dir = Path(self.pred_root) / run_name
        if not run_dir.exists():
            return None, f"run folder missing: {run_name}"
        try:
            table = _load_predictions_table(run_dir)
        except Exception as exc:  # pragma: no cover - runtime IO errors
            return None, str(exc)
        try:
            sliced = _match_recording_slice(table, Path(pose_file).stem)
        except Exception as exc:  # pragma: no cover - runtime matching errors
            return None, f"recording-match error: {exc}"
        return sliced, None

    @staticmethod
    def _build_prediction_frame_index(df: pd.DataFrame | None) -> tuple[dict[int, int], np.ndarray]:
        if df is None or df.empty or "frame" not in df.columns:
            return {}, np.array([], dtype=int)
        frame_vals = pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(frame_vals)
        if not valid.any():
            return {}, np.array([], dtype=int)
        row_ids = np.flatnonzero(valid)
        frames_i = np.round(frame_vals[valid]).astype(int)
        lookup: dict[int, int] = {}
        for f, r in zip(frames_i, row_ids):
            if int(f) not in lookup:
                lookup[int(f)] = int(r)
        sorted_frames = np.array(sorted(lookup.keys()), dtype=int)
        return lookup, sorted_frames

    @staticmethod
    def _find_prediction_row(
        df: pd.DataFrame | None,
        frame_idx: int,
        time_s: float,
        frame_lookup: dict[int, int],
        sorted_frames: np.ndarray,
    ) -> pd.Series | None:
        if df is None or df.empty:
            return None
        if frame_lookup:
            if frame_idx in frame_lookup:
                return df.iloc[int(frame_lookup[frame_idx])]
            if sorted_frames.size > 0:
                pos = int(np.searchsorted(sorted_frames, frame_idx))
                if pos <= 0:
                    nearest = int(sorted_frames[0])
                elif pos >= sorted_frames.size:
                    nearest = int(sorted_frames[-1])
                else:
                    left = int(sorted_frames[pos - 1])
                    right = int(sorted_frames[pos])
                    nearest = left if abs(frame_idx - left) <= abs(frame_idx - right) else right
                return df.iloc[int(frame_lookup[nearest])]
        if "time_s" in df.columns:
            t_vals = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(t_vals)
            if valid.any():
                valid_idx = np.flatnonzero(valid)
                nearest_pos = int(np.argmin(np.abs(t_vals[valid] - float(time_s))))
                row_idx = int(valid_idx[nearest_pos])
                return df.iloc[row_idx]
        if "frame" in df.columns:
            f_vals = pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(f_vals)
            if valid.any():
                valid_idx = np.flatnonzero(valid)
                nearest_pos = int(np.argmin(np.abs(f_vals[valid] - float(frame_idx))))
                row_idx = int(valid_idx[nearest_pos])
                return df.iloc[row_idx]
        if len(df) == 1:
            return df.iloc[0]
        return None

    @staticmethod
    def _prediction_text_from_row(row: pd.Series | None) -> str:
        if row is None:
            return "Prediction: n/a"
        class_col = None
        for candidate in ("pred_class", "predicted_class", "behavior", "class"):
            if candidate in row.index:
                class_col = candidate
                break
        cls = str(row[class_col]) if class_col is not None and pd.notna(row[class_col]) else "n/a"

        conf_val: float | None = None
        for candidate in ("pred_confidence", "confidence", "prob_max", "max_probability"):
            if candidate in row.index:
                val = pd.to_numeric(row[candidate], errors="coerce")
                if pd.notna(val):
                    conf_val = float(val)
                    break
        if conf_val is None:
            prob_cols = [c for c in row.index if str(c).startswith("p_")]
            if prob_cols:
                prob_vals = pd.to_numeric(row[prob_cols], errors="coerce")
                if np.isfinite(prob_vals.to_numpy(dtype=float)).any():
                    conf_val = float(np.nanmax(prob_vals.to_numpy(dtype=float)))
        if conf_val is None:
            return f"Prediction: {cls}"
        return f"Prediction: {cls} ({conf_val:.3f})"

    @staticmethod
    def _prediction_mix_text(df: pd.DataFrame | None) -> str:
        if df is None or df.empty or "pred_class" not in df.columns:
            return "Prediction mix: n/a"
        vc = df["pred_class"].astype(str).value_counts(normalize=True)
        top = vc.head(3)
        parts = [f"{k} {v * 100:.1f}%" for k, v in top.items()]
        return "Prediction mix: " + " | ".join(parts)

    @staticmethod
    def _extract_feature_columns(df: pd.DataFrame | None) -> list[str]:
        if df is None or df.empty:
            return []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        meta_cols = {"frame", "time_s"}
        return [c for c in numeric_cols if c not in meta_cols]

    @staticmethod
    def _pick_default_features(feature_cols: list[str], n: int = 3) -> list[str]:
        if not feature_cols:
            return []
        chosen: list[str] = []
        for name in DEFAULT_IMPORTANT_FEATURES:
            if name in feature_cols and name not in chosen:
                chosen.append(name)
            if len(chosen) >= n:
                break
        if len(chosen) < n:
            for name in feature_cols:
                if name not in chosen:
                    chosen.append(name)
                if len(chosen) >= n:
                    break
        return chosen

    @staticmethod
    def _feature_set_for_plot(feature_cols: list[str], selected: list[str]) -> list[str]:
        """Return ordered list of features to plot. Uses *selected* if non-empty, else picks defaults."""
        if not feature_cols:
            return []
        if selected:
            return [s for s in selected if s in feature_cols]
        # fallback: pick up to 3 defaults
        chosen: list[str] = []
        for name in DEFAULT_IMPORTANT_FEATURES:
            if name in feature_cols and name not in chosen:
                chosen.append(name)
            if len(chosen) >= 3:
                break
        if len(chosen) < 3:
            for name in feature_cols:
                if name not in chosen:
                    chosen.append(name)
                if len(chosen) >= 3:
                    break
        return chosen

    def _reload_features_for_selected_pose(self) -> None:
        self.jen_features_df = None
        self.aya_features_df = None
        self.jen_predictions_df = None
        self.aya_predictions_df = None
        self.jen_pred_frame_map = {}
        self.aya_pred_frame_map = {}
        self.jen_pred_sorted_frames = np.array([], dtype=int)
        self.aya_pred_sorted_frames = np.array([], dtype=int)
        self.jen_pred_mix_text = "Prediction mix: n/a"
        self.aya_pred_mix_text = "Prediction mix: n/a"
        self.jen_feature_cols = []
        self.aya_feature_cols = []
        issues: list[str] = []

        jen_df, jen_err = self._load_side_features(self.jen_feature_run, self.jen_file)
        aya_df, aya_err = self._load_side_features(self.aya_feature_run, self.aya_file)
        jen_pred_df, jen_pred_err = self._load_side_predictions(self.jen_feature_run, self.jen_file)
        aya_pred_df, aya_pred_err = self._load_side_predictions(self.aya_feature_run, self.aya_file)
        if jen_err:
            issues.append(f"Jen: {jen_err}")
        if aya_err:
            issues.append(f"Aya: {aya_err}")
        if jen_pred_err:
            issues.append(f"Jen pred: {jen_pred_err}")
        if aya_pred_err:
            issues.append(f"Aya pred: {aya_pred_err}")

        if jen_df is not None:
            self.jen_features_df = jen_df
            self.jen_feature_cols = self._extract_feature_columns(jen_df)
        if aya_df is not None:
            self.aya_features_df = aya_df
            self.aya_feature_cols = self._extract_feature_columns(aya_df)
        if jen_pred_df is not None:
            self.jen_predictions_df = jen_pred_df
            self.jen_pred_frame_map, self.jen_pred_sorted_frames = self._build_prediction_frame_index(jen_pred_df)
            self.jen_pred_mix_text = self._prediction_mix_text(jen_pred_df)
        if aya_pred_df is not None:
            self.aya_predictions_df = aya_pred_df
            self.aya_pred_frame_map, self.aya_pred_sorted_frames = self._build_prediction_frame_index(aya_pred_df)
            self.aya_pred_mix_text = self._prediction_mix_text(aya_pred_df)

        if self.jen_feature_cols:
            # Keep only still-valid selections
            self.jen_features_selected = [s for s in self.jen_features_selected if s in self.jen_feature_cols]
            if not self.jen_features_selected:
                self.jen_features_selected = self._pick_default_features(self.jen_feature_cols)
        else:
            self.jen_features_selected = []
        if self.aya_feature_cols:
            self.aya_features_selected = [s for s in self.aya_features_selected if s in self.aya_feature_cols]
            if not self.aya_features_selected:
                self.aya_features_selected = self._pick_default_features(self.aya_feature_cols)
        else:
            self.aya_features_selected = []

        self._rebuild_feature_checkboxes("jen")
        self._rebuild_feature_checkboxes("aya")

        # Update combo items for quick-add
        dpg.configure_item("jen_feature_combo", items=self.jen_feature_cols)
        dpg.configure_item("aya_feature_combo", items=self.aya_feature_cols)

        if self.jen_features_df is not None:
            jen_show = self._feature_set_for_plot(self.jen_feature_cols, self.jen_features_selected)
            dpg.set_value(
                "jen_feature_info",
                f"{len(self.jen_feature_cols)} numeric | showing {len(jen_show)} traces ({self.feature_mode})",
            )
        else:
            dpg.set_value("jen_feature_info", "No Jen feature data loaded.")
        if self.aya_features_df is not None:
            aya_show = self._feature_set_for_plot(self.aya_feature_cols, self.aya_features_selected)
            dpg.set_value(
                "aya_feature_info",
                f"{len(self.aya_feature_cols)} numeric | showing {len(aya_show)} traces ({self.feature_mode})",
            )
        else:
            dpg.set_value("aya_feature_info", "No Aya feature data loaded.")

        if issues:
            dpg.set_value("feature_status_text", "Feature status: " + " | ".join(issues))
        elif self.jen_features_df is None and self.aya_features_df is None:
            dpg.set_value("feature_status_text", "Feature status: no feature tables loaded.")
        else:
            dpg.set_value("feature_status_text", f"Feature status: loaded mode={self.feature_mode}.")
        self._invalidate_feature_plot_cache()

    def _invalidate_feature_plot_cache(self) -> None:
        self._feature_plot_signatures["jen"] = None
        self._feature_plot_signatures["aya"] = None

    def _build_aya_display_bundle(self) -> None:
        if self.aya_raw_bundle_raw is None:
            self.aya_display_bundle = None
            return
        self.aya_mapping_details = {}
        if self.aya_view_mode == "Mapped to Jen":
            mapped_raw_df, details = build_renamed_df(
                source_df=self.aya_raw_bundle_raw.df,
                source_scorer=self.aya_raw_bundle_raw.scorer,
                target_scorer=self.jen_bundle_raw.scorer if self.jen_bundle_raw else self.aya_raw_bundle_raw.scorer,
                jen_to_aya_map=self.mapping_profile,
            )
            self.aya_mapping_details = details
            self.aya_display_bundle_raw = _pose_bundle_from_df(
                mapped_raw_df,
                fps=self.aya_raw_bundle_raw.fps,
                source_label=f"{self.aya_raw_bundle_raw.source_label}_mapped_view",
                unit_label="px",
            )
            mapped_norm_df, mapped_ref_px, mapped_ref_pair = _normalize_pose_df(
                mapped_raw_df, preferred_ref_pair=DEFAULT_REF_PAIR
            )
            self.aya_display_bundle_norm = _pose_bundle_from_df(
                mapped_norm_df,
                fps=self.aya_raw_bundle_raw.fps,
                source_label=f"{self.aya_raw_bundle_raw.source_label}_mapped_view_normalized",
                unit_label="BLU",
                ref_dist_px=mapped_ref_px,
                ref_pair=mapped_ref_pair,
            )
        else:
            self.aya_display_bundle_raw = self.aya_raw_bundle_raw
            self.aya_display_bundle_norm = self.aya_raw_bundle_norm
        self._apply_pose_coord_mode()

    def _apply_pose_coord_mode(self) -> None:
        if self.pose_coord_mode == "Normalized pose":
            self.jen_bundle = self.jen_bundle_norm or self.jen_bundle_raw
            self.aya_raw_bundle = self.aya_raw_bundle_norm or self.aya_raw_bundle_raw
            self.aya_display_bundle = self.aya_display_bundle_norm or self.aya_display_bundle_raw
        else:
            self.jen_bundle = self.jen_bundle_raw
            self.aya_raw_bundle = self.aya_raw_bundle_raw
            self.aya_display_bundle = self.aya_display_bundle_raw

    def _reset_timeline_bounds(self) -> None:
        if self.jen_bundle is None or self.aya_display_bundle is None:
            return
        self.left_frame = max(0, min(self.left_frame, self.jen_bundle.n_frames - 1))
        self.right_frame = max(0, min(self.right_frame, self.aya_display_bundle.n_frames - 1))
        self.left_time = self.jen_bundle.frame_to_time(self.left_frame)
        self.right_time = self.aya_display_bundle.frame_to_time(self.right_frame)

        dpg.configure_item("left_frame_slider", min_value=0, max_value=max(0, self.jen_bundle.n_frames - 1))
        dpg.configure_item("right_frame_slider", min_value=0, max_value=max(0, self.aya_display_bundle.n_frames - 1))
        dpg.configure_item(
            "left_time_slider",
            min_value=0.0,
            max_value=self.jen_bundle.frame_to_time(max(0, self.jen_bundle.n_frames - 1)),
        )
        dpg.configure_item(
            "right_time_slider",
            min_value=0.0,
            max_value=self.aya_display_bundle.frame_to_time(max(0, self.aya_display_bundle.n_frames - 1)),
        )
        self._apply_timeline_mode_visibility()
        self._set_slider_values()

    def _set_slider_values(self) -> None:
        self._suspend = True
        try:
            dpg.set_value("left_frame_slider", int(self.left_frame))
            dpg.set_value("right_frame_slider", int(self.right_frame))
            dpg.set_value("left_time_slider", float(self.left_time))
            dpg.set_value("right_time_slider", float(self.right_time))
        finally:
            self._suspend = False

    def _apply_timeline_mode_visibility(self) -> None:
        frame_mode = self.timeline_mode == "Frame"
        dpg.configure_item("left_frame_slider", show=frame_mode)
        dpg.configure_item("right_frame_slider", show=frame_mode)
        dpg.configure_item("left_time_slider", show=not frame_mode)
        dpg.configure_item("right_time_slider", show=not frame_mode)

    def _sync_linked(self) -> None:
        if not self.link_bars or self.jen_bundle is None or self.aya_display_bundle is None:
            return
        if self.timeline_mode == "Frame":
            if self._last_side == "left":
                left_max = max(1, self.jen_bundle.n_frames - 1)
                right_max = max(0, self.aya_display_bundle.n_frames - 1)
                ratio = self.left_frame / left_max
                self.right_frame = int(round(ratio * right_max))
                self.right_time = self.aya_display_bundle.frame_to_time(self.right_frame)
            else:
                right_max = max(1, self.aya_display_bundle.n_frames - 1)
                left_max = max(0, self.jen_bundle.n_frames - 1)
                ratio = self.right_frame / right_max
                self.left_frame = int(round(ratio * left_max))
                self.left_time = self.jen_bundle.frame_to_time(self.left_frame)
        else:
            if self._last_side == "left":
                max_t = self.aya_display_bundle.frame_to_time(max(0, self.aya_display_bundle.n_frames - 1))
                self.right_time = max(0.0, min(max_t, self.left_time))
                self.right_frame = self.aya_display_bundle.time_to_frame(self.right_time)
            else:
                max_t = self.jen_bundle.frame_to_time(max(0, self.jen_bundle.n_frames - 1))
                self.left_time = max(0.0, min(max_t, self.right_time))
                self.left_frame = self.jen_bundle.time_to_frame(self.left_time)

    def _render(self) -> None:
        if self.jen_bundle is None or self.aya_display_bundle is None or self.aya_raw_bundle is None:
            return
        left_scale = self._draw_bundle(
            self.jen_bundle,
            "left_canvas",
            self.left_frame,
            JEN_COLORS,
            self.left_zoom,
            self.jen_bundle.unit_label,
        )
        right_colors = JEN_COLORS if self.aya_view_mode == "Mapped to Jen" else AYA_COLORS
        right_scale = self._draw_bundle(
            self.aya_display_bundle,
            "right_canvas",
            self.right_frame,
            right_colors,
            self.right_zoom,
            self.aya_display_bundle.unit_label,
        )
        jen_pred_row = self._find_prediction_row(
            df=self.jen_predictions_df,
            frame_idx=self.left_frame,
            time_s=self.left_time,
            frame_lookup=self.jen_pred_frame_map,
            sorted_frames=self.jen_pred_sorted_frames,
        )
        aya_pred_row = self._find_prediction_row(
            df=self.aya_predictions_df,
            frame_idx=self.right_frame,
            time_s=self.right_time,
            frame_lookup=self.aya_pred_frame_map,
            sorted_frames=self.aya_pred_sorted_frames,
        )
        jen_pred_text = self._prediction_text_from_row(jen_pred_row)
        aya_pred_text = self._prediction_text_from_row(aya_pred_row)
        dpg.set_value(
            "left_panel_title",
            f"Jen | {self.jen_file or 'N/A'} | frame={self.left_frame} | t={self.left_time:.3f}s | {jen_pred_text.replace('Prediction: ', '')}",
        )
        right_label = "Aya Raw" if self.aya_view_mode == "Raw Aya" else "Aya Mapped->Jen"
        dpg.set_value(
            "right_panel_title",
            f"{right_label} | {self.aya_file or 'N/A'} | frame={self.right_frame} | t={self.right_time:.3f}s | {aya_pred_text.replace('Prediction: ', '')}",
        )
        dpg.set_value(
            "left_scale_text",
            (
                f"Scale: x[{left_scale['x_min']:.2f},{left_scale['x_max']:.2f}] {self.jen_bundle.unit_label}, "
                f"y[{left_scale['y_min']:.2f},{left_scale['y_max']:.2f}] {self.jen_bundle.unit_label} | "
                f"zoom={self.left_zoom:.2f} | {left_scale['sx']:.2f} screen-px/data-{self.jen_bundle.unit_label} (x), {left_scale['sy']:.2f} (y)"
                + (
                    f" | ref_dist_px={self.jen_bundle.ref_dist_px:.2f} ({self.jen_bundle.ref_pair[0]}-{self.jen_bundle.ref_pair[1]})"
                    if self.jen_bundle.ref_dist_px is not None and self.jen_bundle.ref_pair is not None
                    else ""
                )
            ),
        )
        dpg.set_value(
            "right_scale_text",
            (
                f"Scale: x[{right_scale['x_min']:.2f},{right_scale['x_max']:.2f}] {self.aya_display_bundle.unit_label}, "
                f"y[{right_scale['y_min']:.2f},{right_scale['y_max']:.2f}] {self.aya_display_bundle.unit_label} | "
                f"zoom={self.right_zoom:.2f} | {right_scale['sx']:.2f} screen-px/data-{self.aya_display_bundle.unit_label} (x), {right_scale['sy']:.2f} (y)"
                + (
                    f" | ref_dist_px={self.aya_display_bundle.ref_dist_px:.2f} ({self.aya_display_bundle.ref_pair[0]}-{self.aya_display_bundle.ref_pair[1]})"
                    if self.aya_display_bundle.ref_dist_px is not None and self.aya_display_bundle.ref_pair is not None
                    else ""
                )
            ),
        )
        dpg.set_value("left_time_readout", f"Frame: {self.left_frame} | Time: {self.left_time:.3f}s")
        dpg.set_value("right_time_readout", f"Frame: {self.right_frame} | Time: {self.right_time:.3f}s")
        dpg.set_value("left_pred_readout", jen_pred_text)
        dpg.set_value("right_pred_readout", aya_pred_text)
        dpg.set_value("left_pred_mix_readout", self.jen_pred_mix_text)
        dpg.set_value("right_pred_mix_readout", self.aya_pred_mix_text)
        self._render_mapping_status()
        self._render_ethograms()
        self._render_feature_plots()

    def _feature_x_cursor(
        self,
        df: pd.DataFrame | None,
        bundle: PoseBundle | None,
        frame_idx: int,
        time_s: float,
    ) -> float:
        if df is None or len(df) == 0:
            return float(time_s) if self.timeline_mode == "Time (s)" else float(frame_idx)
        if self.timeline_mode == "Time (s)":
            if "time_s" in df.columns:
                lo = float(pd.to_numeric(df["time_s"], errors="coerce").min())
                hi = float(pd.to_numeric(df["time_s"], errors="coerce").max())
                return float(np.clip(time_s, lo, hi))
            if "frame" in df.columns:
                fps = bundle.fps if bundle is not None else 25.0
                t = float(frame_idx) / max(fps, 1e-8)
                lo = float(pd.to_numeric(df["frame"], errors="coerce").min()) / max(fps, 1e-8)
                hi = float(pd.to_numeric(df["frame"], errors="coerce").max()) / max(fps, 1e-8)
                return float(np.clip(t, lo, hi))
            fps = bundle.fps if bundle is not None else 25.0
            idx_as_t = float(frame_idx) / max(fps, 1e-8)
            hi = max(0.0, (len(df) - 1) / max(fps, 1e-8))
            return float(np.clip(idx_as_t, 0.0, hi))

        if "frame" in df.columns:
            frame_vals = pd.to_numeric(df["frame"], errors="coerce")
            lo = float(frame_vals.min())
            hi = float(frame_vals.max())
            return float(np.clip(frame_idx, lo, hi))
        if bundle is not None and bundle.n_frames > 1 and len(df) > 1:
            ratio = np.clip(frame_idx / max(bundle.n_frames - 1, 1), 0.0, 1.0)
            idx = ratio * (len(df) - 1)
            return float(idx)
        return float(np.clip(frame_idx, 0, max(0, len(df) - 1)))

    def _series_x(self, df: pd.DataFrame, bundle: PoseBundle | None) -> np.ndarray:
        if self.timeline_mode == "Time (s)":
            if "time_s" in df.columns:
                return pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
            if "frame" in df.columns:
                fps = bundle.fps if bundle is not None else 25.0
                return pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float) / max(fps, 1e-8)
            fps = bundle.fps if bundle is not None else 25.0
            return np.arange(len(df), dtype=float) / max(fps, 1e-8)
        if "frame" in df.columns:
            return pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float)
        return np.arange(len(df), dtype=float)

    def _render_side_feature_plot(
        self,
        side_key: str,
        df: pd.DataFrame | None,
        feature_names: list[str],
        bundle: PoseBundle | None,
        frame_idx: int,
        time_s: float,
        x_axis_tag: str,
        y_axis_tag: str,
    ) -> None:
        if df is None or len(df) == 0 or not feature_names:
            dpg.delete_item(y_axis_tag, children_only=True)
            self._feature_plot_signatures[side_key] = None
            return

        x_full = self._series_x(df, bundle)
        cursor_x = self._feature_x_cursor(df, bundle, frame_idx, time_s)

        # --- Collect full valid series per feature ---
        full_items: list[tuple[str, np.ndarray, np.ndarray]] = []
        for name in feature_names:
            if name not in df.columns:
                continue
            y = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x_full) & np.isfinite(y)
            if not valid.any():
                continue
            full_items.append((name, x_full[valid], y[valid]))

        if not full_items:
            dpg.delete_item(y_axis_tag, children_only=True)
            self._feature_plot_signatures[side_key] = None
            return

        # x range from full data (needed for window bounds)
        all_x = np.concatenate([it[1] for it in full_items])
        x_min_full = float(np.nanmin(all_x))
        x_max_full = float(np.nanmax(all_x))
        if not np.isfinite(x_min_full) or not np.isfinite(x_max_full):
            return
        if x_max_full <= x_min_full:
            x_max_full = x_min_full + 1.0

        # --- Streaming: only show data up to cursor_x ---
        stream_items: list[tuple[str, np.ndarray, np.ndarray]] = []
        for name, xv, yv in full_items:
            mask = xv <= cursor_x
            if mask.any():
                stream_items.append((name, xv[mask], yv[mask]))
            else:
                stream_items.append((name, xv[:1], yv[:1]))  # at least one point

        # --- Compute visible x-window ---
        window_min, window_max = self._feature_stream_window_bounds(cursor_x, x_min_full, x_max_full)

        # --- Auto-scale Y to data visible in the window ---
        y_vals_in_window: list[np.ndarray] = []
        for _name, xv, yv in stream_items:
            in_win = (xv >= window_min) & (xv <= window_max)
            if in_win.any():
                y_vals_in_window.append(yv[in_win])
        if y_vals_in_window:
            y_concat = np.concatenate(y_vals_in_window)
            y_min = float(np.nanmin(y_concat))
            y_max = float(np.nanmax(y_concat))
        else:
            # fallback to all streamed data
            y_concat = np.concatenate([it[2] for it in stream_items])
            y_min = float(np.nanmin(y_concat))
            y_max = float(np.nanmax(y_concat))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return
        if y_max <= y_min:
            y_pad = max(abs(y_min) * 0.1, 1.0)
            y_min -= y_pad
            y_max += y_pad
        else:
            y_pad = (y_max - y_min) * 0.08
            y_min -= y_pad
            y_max += y_pad

        # --- Rebuild series when feature set changes ---
        signature: tuple[Any, ...] = (id(df), tuple(feature_names), self.timeline_mode)
        if self._feature_plot_signatures.get(side_key) != signature:
            dpg.delete_item(y_axis_tag, children_only=True)
            for idx, (name, xv, yv) in enumerate(stream_items):
                dpg.add_line_series(
                    xv.tolist(),
                    yv.tolist(),
                    label=name,
                    parent=y_axis_tag,
                    tag=f"{side_key}_feature_series_{idx}",
                )
            dpg.add_line_series(
                [cursor_x, cursor_x],
                [y_min, y_max],
                label="cursor",
                parent=y_axis_tag,
                tag=f"{side_key}_feature_cursor",
            )
            self._feature_plot_signatures[side_key] = signature
        else:
            # Update existing series data (streaming effect)
            for idx, (name, xv, yv) in enumerate(stream_items):
                tag = f"{side_key}_feature_series_{idx}"
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, [xv.tolist(), yv.tolist()])

        # Update cursor
        cursor_tag = f"{side_key}_feature_cursor"
        if dpg.does_item_exist(cursor_tag):
            dpg.set_value(cursor_tag, [[cursor_x, cursor_x], [y_min, y_max]])
        else:
            dpg.add_line_series(
                [cursor_x, cursor_x],
                [y_min, y_max],
                label="cursor",
                parent=y_axis_tag,
                tag=cursor_tag,
            )
        dpg.set_axis_limits(y_axis_tag, y_min, y_max)
        dpg.set_axis_limits(x_axis_tag, window_min, window_max)

    def _feature_stream_window_bounds(self, cursor_x: float, x_min: float, x_max: float) -> tuple[float, float]:
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            return (0.0, 1.0)
        if x_max <= x_min:
            return (x_min, x_min + 1.0)

        span = x_max - x_min
        if self.timeline_mode == "Time (s)":
            window = min(span, max(self.feature_stream_window_seconds, span * 0.22))
        else:
            window = min(span, max(self.feature_stream_window_frames, span * 0.22))
        if window >= span:
            return (x_min, x_max)

        start = cursor_x - window * 0.82
        end = cursor_x + window * 0.18
        if start < x_min:
            end += x_min - start
            start = x_min
        if end > x_max:
            start -= end - x_max
            end = x_max
        start = max(x_min, start)
        end = min(x_max, end)
        if end <= start:
            end = min(x_max, start + max(window, 1.0))
        return (float(start), float(end))

    def _render_ethogram(self, canvas_tag: str, pred_df: pd.DataFrame | None, bundle: PoseBundle | None, current_frame: int) -> None:
        """Draw a color-coded ethogram bar from predictions with a cursor."""
        dpg.delete_item(canvas_tag, children_only=True)
        w = float(self._ethogram_w)
        h = 50.0
        pad_l, pad_r = 6.0, 6.0
        bar_y0, bar_y1 = 4.0, h - 4.0
        draw_w = w - pad_l - pad_r

        # Background
        dpg.draw_rectangle(pmin=(0, 0), pmax=(w, h), color=(40, 40, 40, 255), fill=(20, 20, 20, 255), parent=canvas_tag)

        if pred_df is None or pred_df.empty or "pred_class" not in pred_df.columns or bundle is None:
            dpg.draw_text(pos=(pad_l + 4, 14), text="No predictions loaded.", color=(130, 130, 130, 255), size=14, parent=canvas_tag)
            return

        n_frames = bundle.n_frames
        if n_frames <= 0:
            return

        # Build frame -> class array
        classes = pred_df["pred_class"].astype(str).to_numpy()
        if "frame" in pred_df.columns:
            frames = pd.to_numeric(pred_df["frame"], errors="coerce").to_numpy(dtype=float)
        else:
            frames = np.arange(len(classes), dtype=float)

        # Bin into pixel columns
        n_bins = min(int(draw_w), n_frames)
        if n_bins <= 0:
            return
        bin_width = draw_w / n_bins
        frames_per_bin = max(1.0, n_frames / n_bins)

        # Pre-compute frame -> class lookup (nearest)
        valid = np.isfinite(frames)
        if not valid.any():
            return
        v_frames = frames[valid]
        v_classes = classes[valid]
        sort_idx = np.argsort(v_frames)
        v_frames = v_frames[sort_idx]
        v_classes = v_classes[sort_idx]

        for i in range(n_bins):
            bin_center_frame = (i + 0.5) * frames_per_bin
            idx = int(np.searchsorted(v_frames, bin_center_frame))
            idx = min(idx, len(v_classes) - 1)
            cls = v_classes[idx]
            color = BEHAVIOR_COLORS.get(cls, BEHAVIOR_COLOR_DEFAULT)
            x0 = pad_l + i * bin_width
            x1 = pad_l + (i + 1) * bin_width
            dpg.draw_rectangle(pmin=(x0, bar_y0), pmax=(x1, bar_y1), color=color, fill=color, parent=canvas_tag)

        # Draw cursor
        cursor_x = pad_l + (current_frame / max(n_frames - 1, 1)) * draw_w
        cursor_x = max(pad_l, min(pad_l + draw_w, cursor_x))
        dpg.draw_line((cursor_x, bar_y0 - 2), (cursor_x, bar_y1 + 2), color=(255, 255, 255, 255), thickness=2.0, parent=canvas_tag)

        # Draw legend at the right edge
        # Count unique classes present
        unique_classes = list(dict.fromkeys(v_classes))  # preserve order of appearance
        legend_x = pad_l + 2
        dpg.draw_text(pos=(legend_x, bar_y1 + 1), text=" | ".join(unique_classes[:6]), color=(200, 200, 200, 255), size=10, parent=canvas_tag)

    def _render_ethograms(self) -> None:
        """Render ethogram strips for both Jen and Aya."""
        self._render_ethogram(
            "jen_ethogram_canvas",
            self.jen_predictions_df,
            self.jen_bundle,
            self.left_frame,
        )
        self._render_ethogram(
            "aya_ethogram_canvas",
            self.aya_predictions_df,
            self.aya_display_bundle,
            self.right_frame,
        )

    def _render_feature_plots(self) -> None:
        dpg.configure_item("jen_feature_x_axis", label="Time (s)" if self.timeline_mode == "Time (s)" else "Frame")
        dpg.configure_item("aya_feature_x_axis", label="Time (s)" if self.timeline_mode == "Time (s)" else "Frame")
        jen_show = self._feature_set_for_plot(self.jen_feature_cols, self.jen_features_selected)
        self._render_side_feature_plot(
            side_key="jen",
            df=self.jen_features_df,
            feature_names=jen_show,
            bundle=self.jen_bundle,
            frame_idx=self.left_frame,
            time_s=self.left_time,
            x_axis_tag="jen_feature_x_axis",
            y_axis_tag="jen_feature_y_axis",
        )
        aya_show = self._feature_set_for_plot(self.aya_feature_cols, self.aya_features_selected)
        self._render_side_feature_plot(
            side_key="aya",
            df=self.aya_features_df,
            feature_names=aya_show,
            bundle=self.aya_display_bundle,
            frame_idx=self.right_frame,
            time_s=self.right_time,
            x_axis_tag="aya_feature_x_axis",
            y_axis_tag="aya_feature_y_axis",
        )

    def _draw_bundle(
        self,
        bundle: PoseBundle,
        canvas_tag: str,
        frame_idx: int,
        color_map: dict[str, tuple[int, int, int, int]],
        zoom_factor: float,
        unit_label: str,
    ) -> dict[str, float]:
        if bundle.n_frames <= 0:
            return {"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": 0.0, "sx": 0.0, "sy": 0.0}
        frame = max(0, min(frame_idx, bundle.n_frames - 1))
        dpg.delete_item(canvas_tag, children_only=True)

        base_x_min, base_x_max, base_y_min, base_y_max = bundle.axis_bounds
        base_w = max(1e-8, base_x_max - base_x_min)
        base_h = max(1e-8, base_y_max - base_y_min)
        zoom = max(1.0, float(zoom_factor))
        cx = (base_x_min + base_x_max) * 0.5
        cy = (base_y_min + base_y_max) * 0.5
        half_w = base_w / (2.0 * zoom)
        half_h = base_h / (2.0 * zoom)
        x_min, x_max = cx - half_w, cx + half_w
        y_min, y_max = cy - half_h, cy + half_h
        x_span = max(1e-8, x_max - x_min)
        y_span = max(1e-8, y_max - y_min)
        pad = 24.0
        width = float(self.canvas_w)
        height = float(self.canvas_h)

        def to_px(x: float, y: float) -> tuple[float, float]:
            px = pad + ((x - x_min) / x_span) * (width - 2 * pad)
            py = pad + ((y_max - y) / y_span) * (height - 2 * pad)
            return float(px), float(py)

        dpg.draw_rectangle(
            pmin=(1, 1),
            pmax=(width - 1, height - 1),
            color=(120, 130, 150, 255),
            fill=(6, 6, 8, 255),
            parent=canvas_tag,
        )

        missing = []
        valid_points_data: list[tuple[float, float]] = []
        for bp in bundle.bodyparts:
            arr = bundle.xy.get(bp)
            if arr is None or frame >= arr.shape[0]:
                missing.append(bp)
                continue
            x, y = arr[frame]
            if not (np.isfinite(x) and np.isfinite(y)):
                missing.append(bp)
                continue
            valid_points_data.append((float(x), float(y)))
            px, py = to_px(float(x), float(y))
            color = color_map.get(bp, (68, 68, 68, 255))
            dpg.draw_circle(
                center=(px, py),
                radius=5.5,
                color=(255, 255, 255, 255),
                fill=color,
                thickness=1.0,
                parent=canvas_tag,
            )
            dpg.draw_text(
                pos=(px + 8, py - 8),
                text=bp,
                color=color,
                size=16,
                parent=canvas_tag,
            )

        if missing:
            preview = ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
            dpg.draw_text(
                pos=(8, 8),
                text=f"Missing/NaN: {preview}",
                color=(170, 170, 170, 255),
                size=15,
                parent=canvas_tag,
            )

        if self.show_hull and len(valid_points_data) >= 3:
            hull_data = self._convex_hull(valid_points_data)
            if len(hull_data) >= 3:
                hull_px = [to_px(x, y) for x, y in hull_data]
                dpg.draw_polygon(
                    points=hull_px,
                    color=(255, 220, 30, 220),
                    fill=(255, 220, 30, 128),
                    thickness=1.3,
                    parent=canvas_tag,
                )

        # Draw a scale ruler in active data units.
        if unit_label == "BLU":
            candidate_units = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        else:
            candidate_units = [10, 20, 50, 100, 200, 500, 1000]
        max_scale_units = x_span * 0.35
        chosen_units = candidate_units[0]
        for units in candidate_units:
            if units <= max_scale_units:
                chosen_units = units
            else:
                break
        scale_px_len = (chosen_units / x_span) * (width - 2 * pad)
        ruler_x0 = pad + 12
        ruler_y = height - pad - 12
        ruler_x1 = ruler_x0 + scale_px_len
        dpg.draw_line((ruler_x0, ruler_y), (ruler_x1, ruler_y), color=(230, 230, 230, 255), thickness=2.0, parent=canvas_tag)
        dpg.draw_line((ruler_x0, ruler_y - 5), (ruler_x0, ruler_y + 5), color=(230, 230, 230, 255), thickness=1.0, parent=canvas_tag)
        dpg.draw_line((ruler_x1, ruler_y - 5), (ruler_x1, ruler_y + 5), color=(230, 230, 230, 255), thickness=1.0, parent=canvas_tag)
        dpg.draw_text(
            pos=(ruler_x0, ruler_y - 22),
            text=f"{chosen_units:g} {unit_label}",
            color=(220, 220, 220, 255),
            size=14,
            parent=canvas_tag,
        )

        sx = (width - 2 * pad) / x_span
        sy = (height - 2 * pad) / y_span
        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "sx": sx,
            "sy": sy,
        }

    @staticmethod
    def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        pts = sorted(set(points))
        if len(pts) <= 2:
            return pts

        def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: list[tuple[float, float]] = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper: list[tuple[float, float]] = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]

    def _advance_playback(self, dt: float) -> None:
        if self.jen_bundle is None or self.aya_display_bundle is None:
            return
        speed = max(0.05, float(dpg.get_value("play_speed")))
        loop = bool(dpg.get_value("play_loop"))
        mode = self.play_mode

        if mode == "both":
            self.left_time += dt * speed
            self.right_time += dt * speed
            left_max = self.jen_bundle.frame_to_time(max(0, self.jen_bundle.n_frames - 1))
            right_max = self.aya_display_bundle.frame_to_time(max(0, self.aya_display_bundle.n_frames - 1))
            finished = False

            if self.left_time > left_max:
                if loop and left_max > 0:
                    self.left_time = self.left_time % left_max
                else:
                    self.left_time = left_max
                    finished = True
            if self.right_time > right_max:
                if loop and right_max > 0:
                    self.right_time = self.right_time % right_max
                else:
                    self.right_time = right_max
                    finished = True

            self.left_frame = self.jen_bundle.time_to_frame(self.left_time)
            self.right_frame = self.aya_display_bundle.time_to_frame(self.right_time)
            if finished and not loop:
                self._set_play_mode("none")
            return

        if mode == "left" and self.timeline_mode == "Time (s)":
            self.left_time += dt * speed
            left_max = self.jen_bundle.frame_to_time(max(0, self.jen_bundle.n_frames - 1))
            if self.left_time > left_max:
                if loop and left_max > 0:
                    self.left_time = self.left_time % left_max
                else:
                    self.left_time = left_max
                    self._set_play_mode("none")
            self.left_frame = self.jen_bundle.time_to_frame(self.left_time)
            if self.link_bars:
                self._last_side = "left"
                self._sync_linked()
            return
        if mode == "right" and self.timeline_mode == "Time (s)":
            self.right_time += dt * speed
            right_max = self.aya_display_bundle.frame_to_time(max(0, self.aya_display_bundle.n_frames - 1))
            if self.right_time > right_max:
                if loop and right_max > 0:
                    self.right_time = self.right_time % right_max
                else:
                    self.right_time = right_max
                    self._set_play_mode("none")
            self.right_frame = self.aya_display_bundle.time_to_frame(self.right_time)
            if self.link_bars:
                self._last_side = "right"
                self._sync_linked()
            return

        if mode == "left":
            self._left_play_accum += dt * speed * self.jen_bundle.fps
        elif mode == "right":
            self._right_play_accum += dt * speed * self.aya_display_bundle.fps
        elif self.timeline_mode == "Time (s)":
            self.left_time += dt * speed
            self.right_time += dt * speed
            self.left_frame = self.jen_bundle.time_to_frame(self.left_time)
            self.right_frame = self.aya_display_bundle.time_to_frame(self.right_time)
            if self.link_bars:
                self._last_side = "left"
                self._sync_linked()
            return

        step_left = int(self._left_play_accum)
        step_right = int(self._right_play_accum)
        if step_left <= 0 and step_right <= 0:
            return
        self._left_play_accum -= step_left
        self._right_play_accum -= step_right

        self.left_frame += max(0, step_left)
        self.right_frame += max(0, step_right)
        left_max_idx = max(0, self.jen_bundle.n_frames - 1)
        right_max_idx = max(0, self.aya_display_bundle.n_frames - 1)
        finished = False

        if self.left_frame > left_max_idx:
            if loop and self.jen_bundle.n_frames > 0:
                self.left_frame = self.left_frame % self.jen_bundle.n_frames
            else:
                self.left_frame = left_max_idx
                finished = True
        if self.right_frame > right_max_idx:
            if loop and self.aya_display_bundle.n_frames > 0:
                self.right_frame = self.right_frame % self.aya_display_bundle.n_frames
            else:
                self.right_frame = right_max_idx
                finished = True

        self.left_time = self.jen_bundle.frame_to_time(self.left_frame)
        self.right_time = self.aya_display_bundle.frame_to_time(self.right_frame)
        if self.link_bars:
            self._last_side = "left"
            self._sync_linked()
        if finished and not loop:
            self._set_play_mode("none")

    def _render_mapping_status(self) -> None:
        for child in dpg.get_item_children("mapping_status_child", 1) or []:
            dpg.delete_item(child)
        if self.aya_raw_bundle is None:
            return
        frame = max(0, min(self.right_frame, self.aya_raw_bundle.n_frames - 1))
        available = set(self.aya_raw_bundle.bodyparts)
        color_map = {
            "mapped_finite": (46, 159, 99, 255),
            "mapped_nan": (240, 162, 2, 255),
            "missing_source_part": (209, 73, 91, 255),
            "unmapped": (141, 153, 174, 255),
        }
        label_map = {
            "mapped_finite": "mapped+finite",
            "mapped_nan": "mapped+NaN",
            "missing_source_part": "source missing",
            "unmapped": "unmapped",
        }
        with dpg.group(parent="mapping_status_child"):
            for jen_bp in JEN_BODY_ORDER:
                aya_bp = self.mapping_profile.get(jen_bp)
                if aya_bp is None:
                    status = "unmapped"
                    display = "NaN"
                elif aya_bp not in available:
                    status = "missing_source_part"
                    display = aya_bp
                else:
                    point = self.aya_raw_bundle.xy[aya_bp][frame]
                    status = "mapped_finite" if np.isfinite(point).all() else "mapped_nan"
                    display = aya_bp
                dpg.add_text(f"{jen_bp} <- {display} [{label_map[status]}]", color=color_map[status])

    def _rebuild_mapping_editor_options(self) -> None:
        for child in dpg.get_item_children("mapping_editor_child", 1) or []:
            dpg.delete_item(child)
        if self.aya_raw_bundle is None:
            return
        options = ["NaN"] + sorted(self.aya_raw_bundle.bodyparts)
        with dpg.group(parent="mapping_editor_child"):
            for jen_bp in JEN_BODY_ORDER:
                current = self.mapping_profile.get(jen_bp)
                value = current if current in options else "NaN"
                dpg.add_combo(
                    label=f"{jen_bp} <- Aya",
                    items=options,
                    default_value=value,
                    callback=self.on_mapping_combo_changed,
                    user_data=jen_bp,
                    width=350,
                )

    # Callbacks
    def on_refresh_feature_runs(self, sender: str, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self.refresh_feature_runs()
        self._render()

    def on_feature_mode_changed(self, sender: str, app_data: str, user_data: Any) -> None:
        del sender, user_data
        self.feature_mode = app_data
        self._reload_features_for_selected_pose()
        self._render()

    def on_jen_feature_run_changed(self, sender: str, app_data: str, user_data: Any) -> None:
        del sender, user_data
        self.jen_feature_run = app_data
        self._reload_features_for_selected_pose()
        self._render()

    def on_aya_feature_run_changed(self, sender: str, app_data: str, user_data: Any) -> None:
        del sender, user_data
        self.aya_feature_run = app_data
        self._reload_features_for_selected_pose()
        self._render()

    def _on_feature_checkbox(self, sender: str, checked: bool, user_data: Any) -> None:
        """Toggle a feature on/off for the appropriate side."""
        side, feat_name = user_data
        sel_list = self.jen_features_selected if side == "jen" else self.aya_features_selected
        if checked:
            if feat_name not in sel_list:
                sel_list.append(feat_name)
        else:
            if feat_name in sel_list:
                sel_list.remove(feat_name)
        # Invalidate plot cache so it redraws with new selection on next tick
        self._feature_plot_signatures[side] = None
        # Update info label
        cols = self.jen_feature_cols if side == "jen" else self.aya_feature_cols
        show = self._feature_set_for_plot(cols, sel_list)
        info_tag = f"{side}_feature_info"
        dpg.set_value(info_tag, f"{len(cols)} numeric | showing {len(show)} traces ({self.feature_mode})")

    def _rebuild_feature_checkboxes(self, side: str) -> None:
        """Rebuild the feature selection checkboxes for one side."""
        container_tag = f"{side}_feature_checks"
        if dpg.does_item_exist(container_tag):
            dpg.delete_item(container_tag, children_only=True)
        else:
            return
        cols = self.jen_feature_cols if side == "jen" else self.aya_feature_cols
        sel = self.jen_features_selected if side == "jen" else self.aya_features_selected
        for feat in cols:
            dpg.add_checkbox(
                label=feat,
                default_value=(feat in sel),
                callback=self._on_feature_checkbox,
                user_data=(side, feat),
                parent=container_tag,
            )

    def on_jen_feature_changed(self, sender: str, app_data: str, user_data: Any) -> None:
        """Legacy combo callback — kept for backward compat but no longer primary."""
        del sender, user_data
        if app_data and app_data in self.jen_feature_cols:
            if app_data not in self.jen_features_selected:
                self.jen_features_selected.append(app_data)
                self._feature_plot_signatures["jen"] = None
                self._rebuild_feature_checkboxes("jen")

    def on_aya_feature_changed(self, sender: str, app_data: str, user_data: Any) -> None:
        """Legacy combo callback — kept for backward compat but no longer primary."""
        del sender, user_data
        if app_data and app_data in self.aya_feature_cols:
            if app_data not in self.aya_features_selected:
                self.aya_features_selected.append(app_data)
                self._feature_plot_signatures["aya"] = None
                self._rebuild_feature_checkboxes("aya")

    def on_refresh_files(self, sender: str, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self.refresh_file_lists()
        self.refresh_feature_runs()
        self.reload_pose_data()

    def on_auto_pair_toggle(self, sender: str, app_data: bool, user_data: Any) -> None:
        del sender, user_data
        self.auto_pair = bool(app_data)

    def on_jen_file_changed(self, sender: str, app_data: str, user_data: Any) -> None:
        del sender, user_data
        self.jen_file = app_data
        if self.auto_pair and self.aya_files:
            suggestion = _suggest_match(self.jen_file, self.aya_files)
            if suggestion:
                self.aya_file = suggestion
                dpg.set_value("aya_file_combo", suggestion)
        self.reload_pose_data()

    def on_aya_file_changed(self, sender: str, app_data: str, user_data: Any) -> None:
        del sender, user_data
        self.aya_file = app_data
        self.reload_pose_data()

    def on_timeline_mode_changed(self, sender: str, app_data: str, user_data: Any) -> None:
        del sender, user_data
        self.timeline_mode = app_data
        self._apply_timeline_mode_visibility()
        self._render()

    def on_link_changed(self, sender: str, app_data: bool, user_data: Any) -> None:
        del sender, user_data
        self.link_bars = bool(app_data)

    def on_hull_toggle(self, sender: str, app_data: bool, user_data: Any) -> None:
        del sender, user_data
        self.show_hull = bool(app_data)
        self._render()

    def on_aya_view_mode_changed(self, sender: str, app_data: str, user_data: Any) -> None:
        del sender, user_data
        self.aya_view_mode = app_data
        self._build_aya_display_bundle()
        self._reset_timeline_bounds()
        self._render()

    def on_pose_coord_mode_changed(self, sender: str, app_data: str, user_data: Any) -> None:
        del sender, user_data
        self.pose_coord_mode = str(app_data)
        self._apply_pose_coord_mode()
        self._reset_timeline_bounds()
        self._render()

    def on_manual_fps_toggle(self, sender: str, app_data: bool, user_data: Any) -> None:
        del sender, user_data
        self.manual_fps_enabled = bool(app_data)
        dpg.configure_item("jen_fps_input", enabled=self.manual_fps_enabled)
        dpg.configure_item("aya_fps_input", enabled=self.manual_fps_enabled)
        self.reload_pose_data()

    def on_manual_fps_changed(self, sender: str, app_data: float, user_data: Any) -> None:
        del sender, app_data, user_data
        if self.manual_fps_enabled:
            self.reload_pose_data()

    def on_mapping_combo_changed(self, sender: str, app_data: str, user_data: str) -> None:
        del sender
        jen_bp = str(user_data)
        self.mapping_profile[jen_bp] = None if app_data == "NaN" else app_data
        if self.aya_view_mode == "Mapped to Jen":
            self._build_aya_display_bundle()
        self._render()

    def on_left_frame_slider(self, sender: str, app_data: int, user_data: Any) -> None:
        del sender, user_data
        if self._suspend or self.jen_bundle is None:
            return
        self._last_side = "left"
        self.left_frame = int(app_data)
        self.left_time = self.jen_bundle.frame_to_time(self.left_frame)
        self._sync_linked()
        self._set_slider_values()
        self._render()

    def on_right_frame_slider(self, sender: str, app_data: int, user_data: Any) -> None:
        del sender, user_data
        if self._suspend or self.aya_display_bundle is None:
            return
        self._last_side = "right"
        self.right_frame = int(app_data)
        self.right_time = self.aya_display_bundle.frame_to_time(self.right_frame)
        self._sync_linked()
        self._set_slider_values()
        self._render()

    def on_left_time_slider(self, sender: str, app_data: float, user_data: Any) -> None:
        del sender, user_data
        if self._suspend or self.jen_bundle is None:
            return
        self._last_side = "left"
        self.left_time = float(app_data)
        self.left_frame = self.jen_bundle.time_to_frame(self.left_time)
        self._sync_linked()
        self._set_slider_values()
        self._render()

    def on_right_time_slider(self, sender: str, app_data: float, user_data: Any) -> None:
        del sender, user_data
        if self._suspend or self.aya_display_bundle is None:
            return
        self._last_side = "right"
        self.right_time = float(app_data)
        self.right_frame = self.aya_display_bundle.time_to_frame(self.right_time)
        self._sync_linked()
        self._set_slider_values()
        self._render()

    def on_toggle_play(self, sender: str, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self.playing = not self.playing
        dpg.configure_item("play_button", label="Pause" if self.playing else "Play")
        self._last_render_time = time.perf_counter()

    def on_step_once(self, sender: str, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        if self.jen_bundle is None or self.aya_display_bundle is None:
            return
        mode = self.play_mode if self.play_mode != "none" else "both"
        if mode == "both":
            self.left_time += 1.0 / max(self.jen_bundle.fps, 1e-8)
            self.right_time += 1.0 / max(self.aya_display_bundle.fps, 1e-8)
            left_max = self.jen_bundle.frame_to_time(max(0, self.jen_bundle.n_frames - 1))
            right_max = self.aya_display_bundle.frame_to_time(max(0, self.aya_display_bundle.n_frames - 1))
            self.left_time = min(self.left_time, left_max)
            self.right_time = min(self.right_time, right_max)
            self.left_frame = self.jen_bundle.time_to_frame(self.left_time)
            self.right_frame = self.aya_display_bundle.time_to_frame(self.right_time)
        elif mode == "left":
            self.left_frame = min(self.left_frame + 1, max(0, self.jen_bundle.n_frames - 1))
            self.left_time = self.jen_bundle.frame_to_time(self.left_frame)
        elif mode == "right":
            self.right_frame = min(self.right_frame + 1, max(0, self.aya_display_bundle.n_frames - 1))
            self.right_time = self.aya_display_bundle.frame_to_time(self.right_frame)
        else:
            if self.timeline_mode == "Time (s)":
                self.left_time += 1.0 / max(self.jen_bundle.fps, 1e-8)
                self.right_time += 1.0 / max(self.aya_display_bundle.fps, 1e-8)
                self.left_frame = self.jen_bundle.time_to_frame(self.left_time)
                self.right_frame = self.aya_display_bundle.time_to_frame(self.right_time)
            else:
                self.left_frame = min(self.left_frame + 1, max(0, self.jen_bundle.n_frames - 1))
                self.right_frame = min(self.right_frame + 1, max(0, self.aya_display_bundle.n_frames - 1))
                self.left_time = self.jen_bundle.frame_to_time(self.left_frame)
                self.right_time = self.aya_display_bundle.frame_to_time(self.right_frame)
        if self.link_bars:
            self._last_side = "left"
            self._sync_linked()
        self._set_slider_values()
        self._render()

    def _set_play_mode(self, mode: str) -> None:
        if mode not in {"none", "left", "right", "both"}:
            mode = "none"
        if mode == "none":
            self.playing = False
            self.play_mode = "none"
        else:
            self.playing = True
            self.play_mode = mode
            self._last_render_time = time.perf_counter()
            self._left_play_accum = 0.0
            self._right_play_accum = 0.0
        self._update_play_buttons()

    def _update_play_buttons(self) -> None:
        dpg.configure_item(
            "play_left_button",
            label="Pause Left" if self.playing and self.play_mode == "left" else "Play Left",
        )
        dpg.configure_item(
            "play_right_button",
            label="Pause Right" if self.playing and self.play_mode == "right" else "Play Right",
        )
        dpg.configure_item(
            "play_both_button",
            label="Pause Both (Sync)" if self.playing and self.play_mode == "both" else "Play Both (Sync)",
        )

    def on_play_left(self, sender: str, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        if self.playing and self.play_mode == "left":
            self._set_play_mode("none")
        else:
            self._set_play_mode("left")

    def on_play_right(self, sender: str, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        if self.playing and self.play_mode == "right":
            self._set_play_mode("none")
        else:
            self._set_play_mode("right")

    def on_play_both(self, sender: str, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        if self.playing and self.play_mode == "both":
            self._set_play_mode("none")
        else:
            self._set_play_mode("both")

    def on_stop_play(self, sender: str, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self._set_play_mode("none")

    def on_left_zoom_changed(self, sender: str, app_data: float, user_data: Any) -> None:
        del sender, user_data
        self.left_zoom = max(1.0, float(app_data))
        self._render()

    def on_right_zoom_changed(self, sender: str, app_data: float, user_data: Any) -> None:
        del sender, user_data
        self.right_zoom = max(1.0, float(app_data))
        self._render()

    def on_reset_left_view(self, sender: str, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self.left_zoom = 1.0
        dpg.set_value("left_zoom_slider", 1.0)
        self._render()

    def on_reset_right_view(self, sender: str, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self.right_zoom = 1.0
        dpg.set_value("right_zoom_slider", 1.0)
        self._render()

    def on_reset_both_views(self, sender: str, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self.left_zoom = 1.0
        self.right_zoom = 1.0
        dpg.set_value("left_zoom_slider", 1.0)
        dpg.set_value("right_zoom_slider", 1.0)
        self._render()

    def on_render_frame(self, sender: str, app_data: Any) -> None:
        del sender, app_data
        # Check for viewport resize
        vp_w = dpg.get_viewport_client_width()
        vp_h = dpg.get_viewport_client_height()
        if vp_w != self._last_vp_w or vp_h != self._last_vp_h:
            self._calc_layout(vp_w, vp_h)
            self._apply_layout()
            self._render()
            self._render_ethograms()

        now = time.perf_counter()
        if self._last_render_time is None:
            self._last_render_time = now
        else:
            dt = max(0.0, now - self._last_render_time)
            self._last_render_time = now
            if self.playing and self.jen_bundle is not None and self.aya_display_bundle is not None:
                self._advance_playback(dt)
                self._set_slider_values()
                self._render()
        self._schedule_frame_tick()


def main() -> None:
    app = ExplorerApp()
    app.run()


if __name__ == "__main__":
    main()
