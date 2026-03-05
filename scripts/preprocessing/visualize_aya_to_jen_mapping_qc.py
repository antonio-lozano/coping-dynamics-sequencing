#!/usr/bin/env python
"""Create side-by-side visual QC snapshots for AYA source vs mapped JEN keypoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


SOURCE_POINTS = [
    "nose",
    "right_ear",
    "left_ear",
    "right_lateral",
    "left_lateral",
    "tail_base",
    "Centroid",
    "tail_end",
]

TARGET_POINTS = [
    "nose",
    "H1R",
    "H2R",
    "H1L",
    "H2L",
    "B1R",
    "B2R",
    "B3R",
    "B1L",
    "B2L",
    "B3L",
    "tail",
    "S2",
    "S1",
]

SOURCE_COLORS = {
    "nose": "#FF5A5F",
    "right_ear": "#F4A259",
    "left_ear": "#F4A259",
    "right_lateral": "#2EC4B6",
    "left_lateral": "#2EC4B6",
    "tail_base": "#3A86FF",
    "Centroid": "#9B5DE5",
    "tail_end": "#A9A9A9",
}

TARGET_COLORS = {
    "nose": "#FF5A5F",
    "H1R": "#F4A259",
    "H2R": "#A9A9A9",
    "H1L": "#F4A259",
    "H2L": "#A9A9A9",
    "B1R": "#A9A9A9",
    "B2R": "#2EC4B6",
    "B3R": "#A9A9A9",
    "B1L": "#A9A9A9",
    "B2L": "#2EC4B6",
    "B3L": "#A9A9A9",
    "tail": "#3A86FF",
    "S2": "#A9A9A9",
    "S1": "#9B5DE5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(r"D:\NINDATA\Aya_behavior\encoded-20260131T123011Z-3-001\encoded"),
        help="Directory with raw AYA DLC files (.h5).",
    )
    parser.add_argument(
        "--mapped-dir",
        type=Path,
        default=Path("data/to_predict/matched_dlc_aya_to_jen"),
        help="Directory with mapped *_matched_to_jen.csv files.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("results/mapping_qc/aya_to_jen_mapping_qc.png"),
        help="Output image path for QC snapshot grid.",
    )
    parser.add_argument(
        "--n-files",
        type=int,
        default=3,
        help="Number of mapped files to visualize (default: 3).",
    )
    return parser.parse_args()


def _read_dlc(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".h5":
        return pd.read_hdf(path)
    return pd.read_csv(path, header=[0, 1, 2], index_col=0)


def _first_scorer(df: pd.DataFrame) -> str:
    scorers = [str(s) for s in df.columns.get_level_values(0).unique().tolist()]
    bad = {"scorer", "unnamed: 0", ""}
    keep = [s for s in scorers if s.strip().lower() not in bad]
    if not keep:
        raise ValueError("No valid scorer found.")
    return keep[0]


def _xy_for_point(df: pd.DataFrame, scorer: str, point: str) -> np.ndarray | None:
    key_x = (scorer, point, "x")
    key_y = (scorer, point, "y")
    if key_x not in df.columns or key_y not in df.columns:
        return None
    xy = df.loc[:, [key_x, key_y]].to_numpy(dtype=float)
    return xy


def _collect_points(df: pd.DataFrame, points: list[str]) -> dict[str, np.ndarray | None]:
    scorer = _first_scorer(df)
    out: dict[str, np.ndarray | None] = {}
    for p in points:
        out[p] = _xy_for_point(df, scorer, p)
    return out


def _draw_panel(
    ax: plt.Axes,
    points: dict[str, np.ndarray | None],
    frame_idx: int,
    title: str,
    colors: dict[str, str],
) -> None:
    ax.set_facecolor("#111111")
    valid_xy = []
    missing = []
    for name, xy in points.items():
        if xy is None or frame_idx >= len(xy):
            missing.append(name)
            continue
        x, y = xy[frame_idx]
        if not np.isfinite(x) or not np.isfinite(y):
            missing.append(name)
            continue
        valid_xy.append((name, x, y))

    if valid_xy:
        xs = np.array([v[1] for v in valid_xy], dtype=float)
        ys = np.array([v[2] for v in valid_xy], dtype=float)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        pad = max((x_max - x_min), (y_max - y_min), 1.0) * 0.2
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_max + pad, y_min - pad)  # invert y for image-like orientation

    for name, x, y in valid_xy:
        c = colors.get(name, "#FFFFFF")
        ax.scatter([x], [y], s=38, c=c, edgecolors="white", linewidths=0.5, zorder=3)
        ax.text(x + 2, y + 2, name, color=c, fontsize=7, weight="bold")

    if missing:
        msg = "NaN/missing: " + ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
        ax.text(
            0.02,
            0.02,
            msg,
            transform=ax.transAxes,
            fontsize=7,
            color="#DDDDDD",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#222222", edgecolor="#444444", alpha=0.8),
        )

    ax.set_title(title, color="white", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.8)


def _raw_from_mapped_name(raw_dir: Path, mapped_name: str) -> Path | None:
    stem = mapped_name.replace("_matched_to_jen.csv", "")
    cand = raw_dir / f"{stem}.h5"
    if cand.exists():
        return cand
    any_match = sorted(raw_dir.glob(f"{stem}*.h5"))
    if any_match:
        return any_match[0]
    return None


def main() -> int:
    args = parse_args()
    mapped_files = sorted(args.mapped_dir.glob("*_matched_to_jen.csv"))[: max(1, int(args.n_files))]
    if not mapped_files:
        raise FileNotFoundError(f"No mapped files found in {args.mapped_dir}")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(mapped_files)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), dpi=220, facecolor="#0B0B0B")
    if n_rows == 1:
        axes = np.array([axes])  # shape -> (1, 2)

    fractions = np.linspace(0.2, 0.8, num=n_rows)

    for i, mapped_path in enumerate(mapped_files):
        raw_path = _raw_from_mapped_name(args.raw_dir, mapped_path.name)
        if raw_path is None:
            raise FileNotFoundError(f"Raw AYA file not found for mapped file: {mapped_path.name}")

        raw_df = _read_dlc(raw_path)
        mapped_df = _read_dlc(mapped_path)
        raw_points = _collect_points(raw_df, SOURCE_POINTS)
        mapped_points = _collect_points(mapped_df, TARGET_POINTS)

        n_frames = min(len(raw_df), len(mapped_df))
        frame_idx = int(max(0, min(n_frames - 1, round((n_frames - 1) * float(fractions[i])))))

        _draw_panel(
            axes[i, 0],
            raw_points,
            frame_idx=frame_idx,
            title=f"AYA Source ({raw_path.name}) frame={frame_idx}",
            colors=SOURCE_COLORS,
        )
        _draw_panel(
            axes[i, 1],
            mapped_points,
            frame_idx=frame_idx,
            title=f"Mapped JEN ({mapped_path.name}) frame={frame_idx}",
            colors=TARGET_COLORS,
        )

    fig.suptitle(
        "AYA -> JEN Mapping QC (S2 intentionally unmapped, tail <- tail_base)",
        color="white",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out_file, dpi=300, bbox_inches="tight", facecolor="#0B0B0B")
    print(f"Saved mapping QC snapshots: {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
