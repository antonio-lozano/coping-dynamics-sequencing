#!/usr/bin/env python
"""Create lightweight QC visualizations for predicted behavior outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import FPS, PREDICTIONS_DIR, TO_PREDICT_DIR  # noqa: E402
from src.ml.pose_features import (  # noqa: E402
    DEFAULT_IMPORTANT_FEATURES,
    compute_kinematic_features,
    load_dlc_csv,
    robust_normalize_features,
)


BEHAVIOR_PALETTE = {
    "Freezing": "#C37B9F",
    "Sniffing": "#4CB7A5",
    "Grooming": "#7EC8E3",
    "Turn": "#8DC63F",
    "Locomotion": "#F8C650",
    "Climbing": "#F4A259",
    "Jump": "#E4572E",
    "Unassigned": "#888888",
}

DEFAULT_SKELETON_EDGES = [
    ("nose", "H1R"),
    ("H1R", "H2R"),
    ("H2R", "B1R"),
    ("B1R", "B2R"),
    ("B2R", "B3R"),
    ("nose", "H1L"),
    ("H1L", "H2L"),
    ("H2L", "B1L"),
    ("B1L", "B2L"),
    ("B2L", "B3L"),
    ("B3R", "S2"),
    ("B3L", "S2"),
    ("S1", "S2"),
    ("S2", "tail"),
]

POINT_COLOR_HEX = [
    "#FF5A5F",
    "#2EC4B6",
    "#FF9F1C",
    "#E71D36",
    "#662E9B",
    "#00A6FB",
    "#A1C349",
    "#F15BB5",
    "#9B5DE5",
    "#00BBF9",
    "#00F5D4",
    "#FEE440",
    "#FB5607",
    "#3A86FF",
]

FEATURE_PANEL_COLORS = [(65, 158, 255), (80, 215, 140), (255, 205, 90)]  # BGR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing predictions_frame.* and predictions_summary.csv.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional custom path to predictions_summary.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for QC figures (default: <predictions-dir>/figures).",
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=None,
        help="Directory containing DLC CSVs used for prediction videos. "
        "Defaults to data/to_predict/<dataset_id> (repo) or configured TO_PREDICT_DIR/<dataset_id>.",
    )
    parser.add_argument(
        "--video-seconds",
        type=int,
        default=10,
        help="Duration of each behavior-overlay video in seconds (default: 10).",
    )
    parser.add_argument(
        "--video-full",
        action="store_true",
        help="Render full recording length (ignores --video-seconds).",
    )
    parser.add_argument(
        "--video-size",
        type=int,
        default=512,
        help="Output video width/height in pixels (default: 512).",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=float(FPS),
        help="FPS for output videos. Defaults to project FPS.",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Skeleton line thickness in pixels.",
    )
    parser.add_argument(
        "--feature-window",
        type=int,
        default=120,
        help="Rolling window (frames) drawn for live normalized-feature traces.",
    )
    parser.add_argument(
        "--feature-panel-scale",
        type=float,
        default=1.0,
        help="Scale factor for normalized-feature panel size (default: 1.0).",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=220,
        help="DPI used when creating QC matplotlib figure (default: 220).",
    )
    parser.add_argument(
        "--figure-save-dpi",
        type=int,
        default=300,
        help="DPI used when saving QC PNG (default: 300).",
    )
    return parser.parse_args()


def _load_frames(predictions_dir: Path) -> pd.DataFrame:
    parquet_path = predictions_dir / "predictions_frame.parquet"
    csv_path = predictions_dir / "predictions_frame.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing predictions frame table in {predictions_dir}.")


def _style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", colors="#4b4b4b", width=0.6, length=3, labelsize=8)


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def _resolve_pose_dir(predictions_dir: Path, explicit_pose_dir: Path | None) -> Path | None:
    if explicit_pose_dir is not None:
        return explicit_pose_dir
    dataset_id = predictions_dir.parent.name
    repo_candidate = repo_root / "data" / "to_predict" / dataset_id
    cfg_candidate = TO_PREDICT_DIR / dataset_id
    if repo_candidate.exists():
        return repo_candidate
    if cfg_candidate.exists():
        return cfg_candidate
    return None


def _build_transform(xy: np.ndarray, size: int, pad: int = 12):
    x_min = float(np.nanmin(xy[:, 0]))
    x_max = float(np.nanmax(xy[:, 0]))
    y_min = float(np.nanmin(xy[:, 1]))
    y_max = float(np.nanmax(xy[:, 1]))
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    scale = (size - 2 * pad) / max(x_range, y_range)
    x_extra = (size - 2 * pad - x_range * scale) / 2.0
    y_extra = (size - 2 * pad - y_range * scale) / 2.0

    def _map_point(x: float, y: float) -> tuple[int, int] | None:
        if np.isnan(x) or np.isnan(y):
            return None
        x_px = int(pad + x_extra + (x - x_min) * scale)
        y_px = int(pad + y_extra + (y - y_min) * scale)
        x_px = min(max(x_px, 0), size - 1)
        y_px = min(max(y_px, 0), size - 1)
        return (x_px, y_px)

    return _map_point


def _find_csv_for_recording(pose_dir: Path, recording: str) -> Path | None:
    exact = pose_dir / f"{recording}.csv"
    if exact.exists():
        return exact
    matches = sorted(pose_dir.glob(f"{recording}*.csv"))
    if matches:
        return matches[0]
    return None


def _shorten_feature_name(name: str, max_len: int = 22) -> str:
    text = str(name)
    return text if len(text) <= max_len else f"{text[: max_len - 1]}."


def _select_overlay_features(norm_feat_df: pd.DataFrame) -> list[str]:
    selected = []
    for col in DEFAULT_IMPORTANT_FEATURES:
        if col not in norm_feat_df.columns:
            continue
        s = pd.to_numeric(norm_feat_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        if float(s.abs().sum()) <= 1e-6:
            continue
        if float(s.std(ddof=0)) <= 1e-6:
            continue
        selected.append(col)
    if len(selected) >= 3:
        return selected[:3]

    fallback = []
    for col in norm_feat_df.columns:
        if not pd.api.types.is_numeric_dtype(norm_feat_df[col]):
            continue
        if col in selected:
            continue
        if float(norm_feat_df[col].abs().sum()) <= 1e-6:
            continue
        fallback.append(col)
    needed = 3 - len(selected)
    return selected + fallback[:needed]


def _draw_feature_panel(
    frame: np.ndarray,
    norm_feat_df: pd.DataFrame,
    feature_cols: list[str],
    frame_idx: int,
    window: int,
    panel_scale: float,
) -> None:
    if not feature_cols:
        return

    h, w = frame.shape[:2]
    scale = float(np.clip(panel_scale, 0.5, 1.5))
    panel_w = int(w * 0.48 * scale)
    panel_h = int(h * 0.36 * scale)
    margin = 8
    x0 = w - panel_w - margin
    y0 = h - panel_h - margin
    x1 = x0 + panel_w
    y1 = y0 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (22, 22, 22), thickness=-1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 180), thickness=1, lineType=cv2.LINE_AA)

    cv2.putText(
        frame,
        "Normalized features (z)",
        (x0 + 8, y0 + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (245, 245, 245),
        1,
        lineType=cv2.LINE_AA,
    )

    n = len(feature_cols)
    row_top = y0 + 24
    usable_h = panel_h - 28
    row_h = max(22, usable_h // max(1, n))

    start = max(0, frame_idx - window + 1)
    end = frame_idx + 1
    z_min, z_max = -3.0, 3.0

    for i, feat in enumerate(feature_cols):
        color = FEATURE_PANEL_COLORS[i % len(FEATURE_PANEL_COLORS)]
        values = pd.to_numeric(norm_feat_df[feat].iloc[start:end], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if values.size < 2:
            continue

        y_top = row_top + i * row_h
        y_bot = min(y_top + row_h - 4, y1 - 4)
        x_left = x0 + 8
        x_right = x1 - 8

        y_zero = int(y_bot - (0.0 - z_min) / (z_max - z_min) * (y_bot - y_top))
        cv2.line(frame, (x_left, y_zero), (x_right, y_zero), (110, 110, 110), 1, lineType=cv2.LINE_AA)

        xs = np.linspace(x_left, x_right, num=values.size)
        vals = np.clip(values, z_min, z_max)
        ys = y_bot - (vals - z_min) / (z_max - z_min) * (y_bot - y_top)
        pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

        current = float(vals[-1])
        label = f"{_shorten_feature_name(feat)}: {current:+.2f}"
        cv2.putText(
            frame,
            label,
            (x_left, y_top + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            lineType=cv2.LINE_AA,
        )


def _render_behavior_videos(
    frame_df: pd.DataFrame,
    out_dir: Path,
    pose_dir: Path | None,
    seconds: int,
    video_full: bool,
    fps: float,
    size: int,
    line_thickness: int,
    feature_window: int,
    feature_panel_scale: float,
) -> None:
    if pose_dir is None or not pose_dir.exists():
        print("Skipping videos: no pose directory found.")
        return

    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    for recording, rec_pred in frame_df.groupby("recording"):
        csv_path = _find_csv_for_recording(pose_dir, recording)
        if csv_path is None:
            print(f"Skip video for {recording}: source DLC CSV not found in {pose_dir}.")
            continue

        try:
            pose_df = load_dlc_csv(csv_path)
        except Exception as exc:
            print(f"Skip video for {recording}: failed to read DLC CSV ({exc}).")
            continue

        rec_pred = rec_pred.sort_values("frame").reset_index(drop=True)
        n_frames = min(len(rec_pred), len(pose_df))
        if not video_full:
            n_frames = min(n_frames, int(seconds * fps))
        if n_frames < 2:
            print(f"Skip video for {recording}: not enough aligned frames.")
            continue

        pose_df = pose_df.iloc[:n_frames]
        rec_pred = rec_pred.iloc[:n_frames]

        raw_feat_df = compute_kinematic_features(pose_df, fps=max(1, int(round(fps))))
        norm_feat_df, _ = robust_normalize_features(raw_feat_df)
        overlay_features = _select_overlay_features(norm_feat_df)
        if len(overlay_features) < 3:
            print(f"Note for {recording}: found {len(overlay_features)} usable normalized features for overlay.")

        bodyparts = sorted(set(c.rsplit("_", 1)[0] for c in pose_df.columns))
        if not bodyparts:
            print(f"Skip video for {recording}: no bodyparts found.")
            continue

        point_colors = {
            bp: _hex_to_bgr(POINT_COLOR_HEX[i % len(POINT_COLOR_HEX)])
            for i, bp in enumerate(bodyparts)
        }

        coords = {}
        valid_xy = []
        for bp in bodyparts:
            x_col = f"{bp}_x"
            y_col = f"{bp}_y"
            if x_col in pose_df.columns and y_col in pose_df.columns:
                xy = pose_df[[x_col, y_col]].to_numpy(dtype=float)
                coords[bp] = xy
                valid_xy.append(xy)
        if not coords:
            print(f"Skip video for {recording}: x/y coordinates unavailable.")
            continue

        all_xy = np.vstack(valid_xy)
        mapper = _build_transform(all_xy, size=size)

        out_path = videos_dir / f"{recording}_behavior_overlay.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (size, size),
        )

        if not writer.isOpened():
            print(f"Skip video for {recording}: could not open VideoWriter.")
            continue

        for i in range(n_frames):
            frame = np.zeros((size, size, 3), dtype=np.uint8)  # black background

            # White thick skeleton lines.
            for bp_a, bp_b in DEFAULT_SKELETON_EDGES:
                if bp_a not in coords or bp_b not in coords:
                    continue
                pt_a = mapper(coords[bp_a][i, 0], coords[bp_a][i, 1])
                pt_b = mapper(coords[bp_b][i, 0], coords[bp_b][i, 1])
                if pt_a is None or pt_b is None:
                    continue
                cv2.line(frame, pt_a, pt_b, (255, 255, 255), thickness=line_thickness, lineType=cv2.LINE_AA)

            # Colored bodypart points.
            for bp, xy in coords.items():
                pt = mapper(xy[i, 0], xy[i, 1])
                if pt is None:
                    continue
                cv2.circle(frame, pt, radius=max(3, line_thickness), color=point_colors[bp], thickness=-1)

            pred_class = str(rec_pred.loc[i, "pred_class"])
            pred_conf = float(rec_pred.loc[i, "pred_confidence"])
            cv2.putText(
                frame,
                f"{pred_class} ({pred_conf:.2f})",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Frame {int(rec_pred.loc[i, 'frame'])}",
                (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

            _draw_feature_panel(
                frame=frame,
                norm_feat_df=norm_feat_df,
                feature_cols=overlay_features,
                frame_idx=i,
                window=max(10, int(feature_window)),
                panel_scale=feature_panel_scale,
            )

            writer.write(frame)

        writer.release()
        print(f"Saved behavior video: {out_path}")


def main() -> int:
    args = parse_args()
    predictions_dir = args.predictions_dir
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    summary_file = args.summary_file or (predictions_dir / "predictions_summary.csv")
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    out_dir = args.out_dir or (predictions_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_file)
    frame_df = _load_frames(predictions_dir)
    if frame_df.empty:
        raise RuntimeError("Frame predictions are empty; nothing to visualize.")

    pct_cols = [c for c in summary_df.columns if c.startswith("%_") and c != "%_low_confidence(<0.5)"]
    classes = [c.replace("%_", "") for c in pct_cols]

    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=int(args.figure_dpi), facecolor="white")
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # A: class proportions (stacked bar by recording)
    if pct_cols:
        long = summary_df.melt(id_vars=["recording"], value_vars=pct_cols, var_name="class", value_name="percent")
        long["class"] = long["class"].str.replace("%_", "", regex=False)
        pivot = long.pivot(index="recording", columns="class", values="percent").fillna(0.0)
        pivot = pivot[sorted(pivot.columns)]
        x = np.arange(len(pivot))
        bottom = np.zeros(len(pivot))
        for cls in pivot.columns:
            vals = pivot[cls].to_numpy()
            ax_a.bar(
                x,
                vals,
                bottom=bottom,
                color=BEHAVIOR_PALETTE.get(cls, "#999999"),
                edgecolor="white",
                linewidth=0.4,
                label=cls,
            )
            bottom += vals
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=7)
        ax_a.set_ylabel("Percent of frames")
        ax_a.set_title("Class Proportions per Recording")
        ax_a.legend(frameon=False, fontsize=7, ncol=2)
    _style_axes(ax_a)

    # B: prediction timeline for up to 3 longest recordings
    frame_counts = frame_df.groupby("recording").size().sort_values(ascending=False)
    selected = frame_counts.head(3).index.tolist()
    if selected:
        class_to_idx = {cls: i for i, cls in enumerate(sorted(classes))}
        for i, rec in enumerate(selected):
            rec_df = frame_df[frame_df["recording"] == rec].copy()
            rec_df["class_idx"] = rec_df["pred_class"].map(class_to_idx).fillna(-1)
            ax_b.plot(rec_df["frame"], rec_df["class_idx"] + i * (len(class_to_idx) + 1), lw=1.0, label=rec)
        ax_b.set_title("Prediction Timeline (Top 3 by Frames)")
        ax_b.set_xlabel("Frame")
        ax_b.set_ylabel("Class index (offset by recording)")
        ax_b.legend(frameon=False, fontsize=7, loc="upper right")
    _style_axes(ax_b)

    # C: confidence distribution
    sns.histplot(
        data=frame_df,
        x="pred_confidence",
        bins=40,
        color="#4C72B0",
        alpha=0.8,
        ax=ax_c,
    )
    ax_c.axvline(0.5, color="#888888", linestyle="--", linewidth=1.0)
    ax_c.set_title("Prediction Confidence Distribution")
    ax_c.set_xlabel("Max class probability")
    _style_axes(ax_c)

    # D: rolling confidence traces
    window = 100
    for rec in selected:
        rec_df = frame_df[frame_df["recording"] == rec].sort_values("frame")
        rolling = rec_df["pred_confidence"].rolling(window=window, min_periods=1).mean()
        ax_d.plot(rec_df["frame"], rolling, linewidth=1.2, label=rec)
    ax_d.axhline(0.5, color="#888888", linestyle="--", linewidth=1.0)
    ax_d.set_title(f"Rolling Confidence (window={window} frames)")
    ax_d.set_xlabel("Frame")
    ax_d.set_ylabel("Rolling mean confidence")
    ax_d.legend(frameon=False, fontsize=7, loc="lower right")
    _style_axes(ax_d)

    plt.tight_layout()
    png_path = out_dir / "prediction_qc.png"
    pdf_path = out_dir / "prediction_qc.pdf"
    fig.savefig(png_path, dpi=int(args.figure_save_dpi), bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Saved QC figures:\n  - {png_path}\n  - {pdf_path}")

    pose_dir = _resolve_pose_dir(predictions_dir, args.pose_dir)
    _render_behavior_videos(
        frame_df=frame_df,
        out_dir=out_dir,
        pose_dir=pose_dir,
        seconds=args.video_seconds,
        video_full=bool(args.video_full),
        fps=args.video_fps,
        size=args.video_size,
        line_thickness=args.line_thickness,
        feature_window=args.feature_window,
        feature_panel_scale=float(args.feature_panel_scale),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
