"""
Figure 2 - manuscript validation, keypoint-moseq compatible regeneration.

This script keeps the final manuscript A-H layout, but mirrors the original
keypoint_moseq_project/SURF_results_analysis_spyder_CAPITULO.py processing
for the validation panels. In particular, it uses the original selected
syllables for the animal overlap panel, cumulative 95% usage cutoff for panel
D, and per-animal precision/recall aggregation before plotting panels E/F.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys
import warnings

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    BIN_SECONDS,
    FIGURE_DATA_DIR,
    FREEZING_DIR,
    FPS,
    INDEX_CSV,
    PALETTE,
    RESULTS_RAW_PKL,
    SOURCE_DATA_DIR,
)

LEGACY_FIGURES_DIR = repo_root / "figures"
FIGURE_OUTPUT_DIR = FIGURE_DATA_DIR
OVERLAP_SYLLABLES = {0, 28, 40}
TIMECOURSE_SYLLABLES = {0, 28}
EVENT_SPAN_STARTS_MIN = [3.5, 4.5, 5.5]
EVENT_SPAN_WIDTH_MIN = 0.5


# Prefer files dropped into data/source/; fall back to original-machine paths.
_REF_CSV_CANDIDATES = [
    SOURCE_DATA_DIR / "syllable_classification_metrics.csv",
    Path(r"H:\antonio\keypoint_moseq_project\code\syllable_recall_precision_f1_usage.csv"),
    Path(r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\syllable_recall_precision_f1_usage.csv"),
]
REFERENCE_METRICS_CSV: Path | None = next((p for p in _REF_CSV_CANDIDATES if p.exists()), None)

_PANEL_G_SVG_CANDIDATES = [
    SOURCE_DATA_DIR / "freezing_overlap_by_group.svg",
    Path(r"H:\antonio\keypoint_moseq_project\code\equipo_project\overlap_freezing_per_mouse_by_group_cleaned_sorted.svg"),
    Path(r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\overlap_freezing_per_mouse_by_group_cleaned_sorted.svg"),
]
PANEL_G_REFERENCE_SVG: Path | None = next((p for p in _PANEL_G_SVG_CANDIDATES if p.exists()), None)

# Prefer inputs extracted into data/source/; fall back to config (external) paths.
_FREEZING_DIR_DEFAULT = next(
    (p for p in [SOURCE_DATA_DIR / "freezing_predictions", FREEZING_DIR] if p.exists()),
    FREEZING_DIR,
)
_INDEX_CSV_DEFAULT = next(
    (p for p in [SOURCE_DATA_DIR / "animal_groups.csv", INDEX_CSV] if p.exists()),
    INDEX_CSV,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mode", choices=["auto", "moseq_df", "results_pkl"], default="auto")
    parser.add_argument("--moseq-df", type=Path, default=None, help="Path to legacy moseq_df.csv.")
    parser.add_argument("--results-pkl", type=Path, default=RESULTS_RAW_PKL)
    parser.add_argument("--freezing-dir", type=Path, default=_FREEZING_DIR_DEFAULT)
    parser.add_argument("--index-csv", type=Path, default=_INDEX_CSV_DEFAULT)
    parser.add_argument("--exclude-animals", type=str, default="Animal_48_6,48_6")
    parser.add_argument("--fail-on-fallback", action="store_true", default=False)
    parser.add_argument("--reference-metrics-csv", type=Path, default=REFERENCE_METRICS_CSV,
                        help="Pre-computed syllable_recall_precision_f1_usage.csv from original analysis.")
    parser.add_argument("--panel-g-reference-svg", type=Path, default=PANEL_G_REFERENCE_SVG,
                        help="Original overlap_freezing_per_mouse_by_group_cleaned_sorted.svg for panel G animal order/filter.")
    return parser.parse_args()


def _normalize(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def _parse_moseq_name(name: str) -> str:
    norm = _normalize(name)
    if "DLC" in norm:
        norm = norm.split("DLC")[0].rstrip("_")
    if "_resnet" in norm:
        norm = norm.split("_resnet")[0].rstrip("_")
    return norm


def _exclude_tokens(raw: str) -> set[str]:
    return {_normalize(tok).lower() for tok in raw.split(",") if tok.strip()}


def _is_excluded(name: str, excluded: set[str]) -> bool:
    return _normalize(name).lower() in excluded or _parse_moseq_name(name).lower() in excluded


def _style_axes(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(0.7)
    ax.tick_params(axis="both", colors="#4b4b4b", width=0.6, length=2.5, labelsize=6)


def _panel_tag(ax, tag: str, x: float = -0.17, y: float = 1.08) -> None:
    ax.text(x, y, tag, transform=ax.transAxes, fontsize=9, fontweight="bold", color="#4b4b4b", ha="center")


def _load_group_map(index_csv: Path) -> dict[str, str]:
    idx = pd.read_csv(index_csv)
    group_map: dict[str, str] = {}
    for _, row in idx.iterrows():
        raw = str(row["name"]).strip()
        norm = _normalize(raw)
        pref = _parse_moseq_name(raw)
        grp = str(row["group"])
        group_map[raw] = grp
        group_map[norm] = grp
        group_map[pref] = grp
        if raw.endswith("_freezing_predictions_only"):
            trimmed = raw.replace("_freezing_predictions_only", "")
            group_map[trimmed] = grp
            group_map[_normalize(trimmed)] = grp
    return group_map


def _load_freezing_frame_table(freezing_dir: Path, group_map: dict[str, str]) -> pd.DataFrame:
    records = []
    for csv_path in sorted(freezing_dir.glob("*_freezing_predictions_only.csv")):
        base = csv_path.name.replace("_freezing_predictions_only.csv", "")
        df = pd.read_csv(csv_path)
        if "Freezing_Jen_0-125_threshold" not in df.columns:
            continue
        norm_base = _normalize(base)
        prefix_base = _parse_moseq_name(norm_base)
        grp = group_map.get(base) or group_map.get(norm_base) or group_map.get(prefix_base) or "Unknown"
        frame = df["Unnamed: 0"] if "Unnamed: 0" in df.columns else np.arange(len(df))
        records.append(
            pd.DataFrame(
                {
                    "animal": base,
                    "group": grp,
                    "frame": frame,
                    "freezing": df["Freezing_Jen_0-125_threshold"].to_numpy(dtype=float),
                }
            )
        )
    if not records:
        raise FileNotFoundError(f"No SimBA freezing CSVs found in {freezing_dir}")
    return pd.concat(records, ignore_index=True)


def _freezing_summaries(
    freezing_df: pd.DataFrame,
    excluded: set[str],
    include_labels: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bin_size = int(FPS * BIN_SECONDS)
    df = freezing_df.copy()
    df["bin"] = (df["frame"] // bin_size).astype(int)
    agg = (
        df.groupby(["group", "bin", "animal"])["freezing"]
        .mean()
        .reset_index(name="freeze_frac")
    )
    agg = agg[~agg["animal"].map(lambda a: _is_excluded(str(a), excluded))].copy()
    if include_labels:
        agg = agg[agg["animal"].map(lambda a: _short_id(str(a)) in include_labels)].copy()
    agg = agg[agg["group"].isin(["Control", "ELS"])].copy()

    def split_animals(group_label: str) -> tuple[list[str], list[str]]:
        animals = sorted(agg.loc[agg["group"] == group_label, "animal"].unique().tolist())
        if len(animals) >= 41:
            return animals[:25], animals[25:41]
        if len(animals) >= 26:
            return animals[: len(animals) - 16], animals[len(animals) - 16 :]
        mid = max(1, len(animals) // 2)
        return animals[:mid], animals[mid:]

    ctrl_a, ctrl_b = split_animals("Control")
    els_a, els_b = split_animals("ELS")

    def summarize(animals: set[str] | list[str]) -> pd.DataFrame:
        data = agg[agg["animal"].isin(animals)]
        summary = (
            data.groupby(["group", "bin"])["freeze_frac"]
            .agg(["mean", "sem", "count"])
            .reset_index()
        )
        summary["time_min"] = (summary["bin"] + 1.0) * BIN_SECONDS / 60.0
        return summary

    summary_a = summarize(set(ctrl_b + els_b))
    summary_b = summarize(set(ctrl_a + els_a))
    summary_all = summarize(sorted(agg["animal"].unique().tolist()))
    return summary_a, summary_b, summary_all


def _load_freezing_vectors(freezing_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for csv_path in sorted(freezing_dir.glob("*_freezing_predictions_only.csv")):
        base = csv_path.name.replace("_freezing_predictions_only.csv", "")
        df = pd.read_csv(csv_path)
        if "Freezing_Jen_0-125_threshold" not in df.columns:
            continue
        vec = df["Freezing_Jen_0-125_threshold"].to_numpy(dtype=int)
        out[base] = vec
        out[_normalize(base)] = vec
        out[_parse_moseq_name(base)] = vec
    return out


def _resolve_moseq_df_path(explicit: Path | None, results_pkl: Path, index_csv: Path | None = None) -> Path | None:
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    rp = Path(results_pkl)
    if index_csv is not None:
        project_dir = Path(index_csv).parent
        candidates.extend(
            [
                project_dir / "2025_01_24-16_44_21" / "moseq_df.csv",
                project_dir.parent / "moseq_df.csv",
            ]
        )
    candidates.extend(
        [
            rp.parent / "moseq_df.csv",
            SOURCE_DATA_DIR / "moseq_syllables_per_frame.csv.gz",
            repo_root / "data" / "processed" / "moseq_df.csv",
            Path(r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\2025_01_24-16_44_21\moseq_df.csv"),
            Path(r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\equipo_project_data\moseq_df.csv"),
            Path(r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\moseq_df.csv"),
        ]
    )
    for path in candidates:
        if path is not None and path.exists():
            return path
    return None


def _choose_sequence_source(args: argparse.Namespace) -> tuple[str, Path, bool]:
    moseq_df_found = _resolve_moseq_df_path(args.moseq_df, args.results_pkl, args.index_csv)
    if args.source_mode == "moseq_df":
        if moseq_df_found is None:
            raise FileNotFoundError("source-mode=moseq_df but no moseq_df.csv found.")
        return "moseq_df", moseq_df_found, False
    if args.source_mode == "results_pkl":
        if not args.results_pkl.exists():
            raise FileNotFoundError(f"results_pkl not found: {args.results_pkl}")
        return "results_pkl", args.results_pkl, False
    if moseq_df_found is not None:
        return "moseq_df", moseq_df_found, False
    if not args.results_pkl.exists():
        raise FileNotFoundError("auto mode: no moseq_df source found and results_pkl does not exist.")
    if args.fail_on_fallback:
        raise RuntimeError(f"auto mode fallback denied: would use {args.results_pkl}")
    warnings.warn(f"Figure 2 fallback: moseq_df missing; using results_pkl={args.results_pkl}", RuntimeWarning)
    return "results_pkl", args.results_pkl, True


def _load_sequences_from_results_pkl(
    results_pkl: Path,
    group_map: dict[str, str],
    excluded: set[str],
    include_labels: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    with open(results_pkl, "rb") as f:
        results = pickle.load(f)
    out = {}
    for rec, data in results.items():
        if "syllable" not in data:
            continue
        rec_raw = str(rec).strip()
        if _is_excluded(rec_raw, excluded):
            continue
        if include_labels and _short_id(rec_raw) not in include_labels:
            continue
        grp = group_map.get(rec_raw) or group_map.get(_normalize(rec_raw)) or group_map.get(_parse_moseq_name(rec_raw)) or "Unknown"
        out[rec_raw] = {"seq": np.array(data["syllable"], dtype=int), "group": grp}
    return out


def _load_sequences_from_moseq_df(
    moseq_df_path: Path,
    group_map: dict[str, str],
    excluded: set[str],
    include_labels: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    df = pd.read_csv(moseq_df_path)
    required = {"name", "frame_index", "syllable"}
    if not required.issubset(df.columns):
        raise ValueError(f"{moseq_df_path} must include columns: {sorted(required)}")
    out = {}
    for rec, gdf in df.groupby("name", sort=False):
        rec_raw = str(rec).strip()
        if _is_excluded(rec_raw, excluded):
            continue
        if include_labels and _short_id(rec_raw) not in include_labels:
            continue
        grp = (
            (str(gdf["group"].iloc[0]) if "group" in gdf.columns else None)
            or group_map.get(rec_raw)
            or group_map.get(_normalize(rec_raw))
            or group_map.get(_parse_moseq_name(rec_raw))
            or "Unknown"
        )
        frames = pd.to_numeric(gdf["frame_index"], errors="coerce").fillna(-1).astype(int).to_numpy()
        sylls = pd.to_numeric(gdf["syllable"], errors="coerce").fillna(-1).astype(int).to_numpy()
        valid = frames >= 0
        if not valid.any():
            continue
        n = int(frames[valid].max()) + 1
        seq = np.full(n, -1, dtype=int)
        seq[frames[valid]] = sylls[valid]
        out[rec_raw] = {"seq": seq, "group": grp}
    return out


def _load_syllable_usage_from_moseq_df(
    moseq_df_path: Path,
    include_labels: set[str] | None = None,
) -> pd.DataFrame:
    """Original panel D logic: moseq_df['syllable'].value_counts().sort_values."""
    usecols = ["syllable", "name"] if include_labels else ["syllable"]
    df = pd.read_csv(moseq_df_path, usecols=usecols)
    if include_labels:
        df = df[df["name"].map(lambda n: _short_id(str(n)) in include_labels)].copy()
    counts = df["syllable"].value_counts().sort_values(ascending=False)
    usage = (counts / counts.sum() * 100.0).reset_index()
    usage.columns = ["syllable", "global_pct"]
    usage["syllable"] = usage["syllable"].astype(int)
    usage["cumulative_coverage"] = usage["global_pct"].cumsum()
    return usage


def _match_freeze(rec_name: str, freezing_records: dict[str, np.ndarray]) -> np.ndarray | None:
    for key in (str(rec_name).strip(), _normalize(rec_name), _parse_moseq_name(rec_name)):
        if key in freezing_records:
            return freezing_records[key]
    return None


def _compute_validation_tables(
    seq_by_recording: dict[str, dict[str, object]],
    freezing_records: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int, pd.DataFrame]:
    metrics_rows = []
    overlap_rows = []
    total_freeze_frames = 0
    matched_rec = 0
    missing_rec = 0

    for rec, payload in seq_by_recording.items():
        seq = np.array(payload["seq"], dtype=int)
        grp = str(payload["group"])
        freeze = _match_freeze(rec, freezing_records)
        if freeze is None:
            missing_rec += 1
            continue
        n = min(len(seq), len(freeze))
        if n <= 0:
            continue
        matched_rec += 1
        seq = seq[:n]
        freeze = np.array(freeze[:n], dtype=int)
        freeze_idx = set(np.flatnonzero(freeze == 1).tolist())
        total_freeze = len(freeze_idx)
        total_freeze_frames += total_freeze
        if total_freeze == 0:
            continue

        valid_seq = seq[seq >= 0]
        for syll in np.unique(valid_seq):
            s = int(syll)
            syll_idx = set(np.flatnonzero(seq == s).tolist())
            overlap = len(syll_idx.intersection(freeze_idx))
            total_syll = len(syll_idx)
            metrics_rows.append(
                {
                    "recording": rec,
                    "group": grp,
                    "syllable": s,
                    "precision": overlap / total_syll if total_syll else 0.0,
                    "recall": overlap / total_freeze if total_freeze else 0.0,
                    "frames": total_syll,
                    "tp": overlap,
                    "fp": total_syll - overlap,
                    "fn": total_freeze - overlap,
                }
            )

        selected_idx = set()
        for syll in OVERLAP_SYLLABLES:
            selected_idx.update(np.flatnonzero(seq == syll).tolist())
        overlap_rows.append(
            {
                "recording": rec,
                "group": grp,
                "overlap": len(selected_idx.intersection(freeze_idx)) / total_freeze * 100.0,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        raise RuntimeError("No matched recordings with freezing frames produced validation metrics.")

    # Per-syllable means across all animals (matching original script's groupby syllable .mean())
    agg = (
        metrics_df.groupby("syllable")[["precision", "recall", "frames", "tp", "fp", "fn"]]
        .agg({"precision": "mean", "recall": "mean", "frames": "sum", "tp": "sum", "fp": "sum", "fn": "sum"})
        .reset_index()
    )
    total_frames = agg["frames"].sum()
    agg["freezing_contrib"] = agg["tp"] / total_freeze_frames if total_freeze_frames else np.nan
    agg["frame_frac"] = agg["frames"] / total_frames if total_frames else np.nan

    # Per-syllable per-group means; fill missing group entries with 0 so both bars always show
    all_syllables = agg["syllable"].tolist()
    all_groups = ["Control", "ELS"]
    full_index = pd.MultiIndex.from_product([all_syllables, all_groups], names=["syllable", "group"])
    group_agg = (
        metrics_df.groupby(["syllable", "group"])[["precision", "recall", "frames", "tp", "fp", "fn"]]
        .agg({"precision": "mean", "recall": "mean", "frames": "sum", "tp": "sum", "fp": "sum", "fn": "sum"})
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # Global syllable frame counts across all recordings (for panel D x-axis ordering and %)
    all_sylls: list[int] = []
    for payload in seq_by_recording.values():
        s = np.array(payload["seq"], dtype=int)
        all_sylls.extend(s[s >= 0].tolist())
    global_counts = pd.Series(all_sylls).value_counts().sort_values(ascending=False)
    global_total = global_counts.sum()
    global_pct = (global_counts / global_total * 100.0).rename("global_pct")
    agg = agg.merge(global_pct.rename("global_pct"), left_on="syllable", right_index=True, how="left")
    agg["global_pct"] = agg["global_pct"].fillna(0.0)

    time_df = _compute_syllable_timecourse(seq_by_recording, TIMECOURSE_SYLLABLES)
    overlap_df = pd.DataFrame(overlap_rows)
    return metrics_df, agg, group_agg, overlap_df, matched_rec, missing_rec, time_df


def _compute_syllable_timecourse(
    seq_by_recording: dict[str, dict[str, object]],
    target_syll: set[int],
) -> pd.DataFrame:
    bin_size = int(FPS * BIN_SECONDS)
    bin_counts = []
    rows = []
    for rec, payload in seq_by_recording.items():
        grp = str(payload["group"])
        if grp not in ("Control", "ELS"):
            continue
        seq = np.array(payload["seq"], dtype=int)
        bins = len(seq) // bin_size
        if bins > 0:
            bin_counts.append(bins)
            rows.append((rec, grp, seq))
    if not bin_counts:
        raise RuntimeError("No recordings have enough frames for time-binning.")
    common_bins = min(bin_counts)
    out_rows = []
    for rec, grp, seq in rows:
        arr = seq[: common_bins * bin_size].reshape(common_bins, bin_size)
        for b, block in enumerate(arr):
            out_rows.append({"recording": rec, "group": grp, "bin": b, "pct": np.mean(np.isin(block, list(target_syll))) * 100.0})
    summary = (
        pd.DataFrame(out_rows)
        .groupby(["group", "bin"])["pct"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    summary["time_min"] = (summary["bin"] + 1.0) * BIN_SECONDS / 60.0
    return summary


def _plot_freezing_panel(ax, data: pd.DataFrame, title: str, tag: str) -> None:
    for start in EVENT_SPAN_STARTS_MIN:
        ax.axvspan(start, start + EVENT_SPAN_WIDTH_MIN, color="#F5F5F5", zorder=0)
    for group, gdata in data.groupby("group"):
        if group not in ("Control", "ELS"):
            continue
        color = PALETTE.get(group, "#4d4d4d")
        ax.plot(gdata["time_min"], gdata["mean"] * 100.0, label=group, color=color, linewidth=1.2, marker="o", markersize=2)
        ax.fill_between(
            gdata["time_min"],
            (gdata["mean"] - gdata["sem"]) * 100.0,
            (gdata["mean"] + gdata["sem"]) * 100.0,
            color=color,
            alpha=0.22,
            linewidth=0,
        )
    ax.set_title(title, fontsize=6.5, color="#4b4b4b", pad=4)
    ax.set_xlabel("Time (minutes)", fontsize=6)
    ax.set_ylabel("Freezing (% of time)", fontsize=6)
    ax.set_xlim(0, 7.5)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_ylim(0, 80)
    ax.set_yticks(np.arange(0, 81, 10))
    ax.legend(frameon=False, loc="lower right", fontsize=5, handlelength=1.3)
    ax.text(4.0, 74, "*", ha="center", va="center", color="#4b4b4b", fontsize=17)
    _style_axes(ax)
    _panel_tag(ax, tag)


def _short_id(name: str) -> str:
    norm = _normalize(str(name))
    match = re.search(r"Animal[_ ]?(\d+(?:[_-]\d+)?)", norm, flags=re.IGNORECASE)
    if match:
        return match.group(1).replace("_", ".").replace("-", ".")
    parts = norm.split("_")
    label = "_".join(parts[-2:]) if len(parts) >= 2 else norm
    return label.replace("_", ".").replace("-", ".")


def _animal_sort_number(name: str) -> int:
    label = _short_id(name)
    match = re.search(r"\d+", label)
    return int(match.group(0)) if match else 10**9


def _load_panel_g_reference_labels(svg_path: Path | None) -> list[str]:
    if svg_path is None or not svg_path.exists():
        return []
    labels: list[str] = []
    pattern = re.compile(r"<!--\s*(\d+\.\d+)\s*-->")
    for line in svg_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "<!-- Animal -->" in line:
            break
        match = pattern.search(line)
        if match:
            labels.append(match.group(1))
    return labels


def _top_metric_from_agg(agg: pd.DataFrame, metric: str, coverage_pct: float = 95.0) -> pd.DataFrame:
    """Mirror original plot_top_95: sort by mean_all and keep cumulative contribution."""
    series = (agg.set_index("syllable")[metric] * 100.0).sort_values(ascending=False)
    total = float(series.sum())
    selected: list[int] = []
    cumulative = 0.0
    for syll, value in series.items():
        selected.append(int(syll))
        if total > 0:
            cumulative += float(value) / total * 100.0
        if cumulative >= coverage_pct:
            break
    out = series.reindex(selected).reset_index()
    out.columns = ["syllable", f"{metric}_all"]
    return out


def _top_metric_from_original_metrics(metrics_df: pd.DataFrame, metric: str, coverage_pct: float = 95.0) -> pd.DataFrame:
    """Mirror original plot_top_95 aggregation over freezing_overlap_per_syllable."""
    rows = []
    for syll in sorted(metrics_df["syllable"].unique().tolist()):
        vals = metrics_df.loc[metrics_df["syllable"] == syll, metric]
        vals_els = metrics_df.loc[(metrics_df["syllable"] == syll) & (metrics_df["group"] == "ELS"), metric]
        vals_ctrl = metrics_df.loc[(metrics_df["syllable"] == syll) & (metrics_df["group"] == "Control"), metric]
        rows.append(
            {
                "syllable": int(syll),
                "mean_all": float(vals.mean()) if len(vals) else 0.0,
                "mean_els": float(vals_els.mean()) if len(vals_els) else 0.0,
                "mean_control": float(vals_ctrl.mean()) if len(vals_ctrl) else 0.0,
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_all", ascending=False).reset_index(drop=True)
    total = out["mean_all"].sum()
    if total > 0:
        out["cumulative_contribution"] = out["mean_all"].cumsum() / total * 100.0
        keep_n = int(np.argmax(out["cumulative_contribution"].to_numpy() >= coverage_pct)) + 1
        out = out.iloc[:keep_n].copy()
    return out


def _plot_metric_bars_ref(
    ax,
    show_df: pd.DataFrame,
    group_agg: pd.DataFrame,
    tag: str,
    metric_col: str = "precision_all",
    ylabel: str = "Mean Precision (%)",
    group_metric: str = "precision",
) -> None:
    """Plot grouped bars using reference overall values + per-group values from group_agg."""
    syllables = show_df["syllable"].astype(int).tolist()
    overall = show_df.set_index("syllable")[metric_col].reindex(syllables).to_numpy()
    ctrl_ser = group_agg[group_agg["group"] == "Control"].set_index("syllable")[group_metric]
    els_ser = group_agg[group_agg["group"] == "ELS"].set_index("syllable")[group_metric]
    # group_agg precision is 0-1; scale to %
    control = ctrl_ser.reindex(syllables).fillna(0).to_numpy() * 100.0
    els = els_ser.reindex(syllables).fillna(0).to_numpy() * 100.0
    x = np.arange(len(syllables))
    width = 0.25
    ax.bar(x - width, overall, width=width, color="#4B4B4B", edgecolor="#FFFFFF", linewidth=0.25, label="All Groups")
    ax.bar(x, control, width=width, color=PALETTE["Control"], edgecolor="#FFFFFF", linewidth=0.25, label="Control")
    ax.bar(x + width, els, width=width, color=PALETTE["ELS"], edgecolor="#FFFFFF", linewidth=0.25, label="ELS")
    ax.set_title(ylabel, fontsize=6.5, color="#4b4b4b", pad=8)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.set_xlabel("Syllable", fontsize=6, loc="right")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in syllables], rotation=90, fontsize=4.8)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    _style_axes(ax)
    _panel_tag(ax, tag, y=1.05)


def _plot_metric_bars_original(ax, show_df: pd.DataFrame, tag: str, ylabel: str, tag_x: float = -0.17) -> None:
    syllables = show_df["syllable"].astype(int).tolist()
    x = np.arange(len(syllables))
    width = 0.25
    ax.bar(x - width, show_df["mean_all"], width=width, color="#4B4B4B", edgecolor="#FFFFFF", linewidth=0.25, label="All Groups")
    ax.bar(x, show_df["mean_control"], width=width, color=PALETTE["Control"], edgecolor="#FFFFFF", linewidth=0.25, label="Control")
    ax.bar(x + width, show_df["mean_els"], width=width, color=PALETTE["ELS"], edgecolor="#FFFFFF", linewidth=0.25, label="ELS")
    ax.set_title(ylabel, fontsize=6.5, color="#4b4b4b", pad=4)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.set_xlabel("Syllable", fontsize=6, loc="right")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in syllables], rotation=90, fontsize=4.8)
    ax.set_xlim(-1.0, len(syllables) - 0.35)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    _style_axes(ax)
    _panel_tag(ax, tag, x=tag_x, y=1.05)


def _draw_section_label(fig, y0: float, y1: float, label: str) -> None:
    x0 = 0.006
    width = 0.026
    rect = Rectangle((x0, y0), width, y1 - y0, transform=fig.transFigure, facecolor="#E5E5E5", edgecolor="none", zorder=-1)
    fig.patches.append(rect)
    fig.text(x0 + width / 2, (y0 + y1) / 2, label, rotation=90, ha="center", va="center", fontsize=9, color="#777777")


def plot_figure(
    summaries: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    metrics_df: pd.DataFrame,
    agg: pd.DataFrame,
    group_agg: pd.DataFrame,
    overlap_df: pd.DataFrame,
    time_summary: pd.DataFrame,
    usage_df: pd.DataFrame | None = None,
    panel_g_reference_labels: list[str] | None = None,
    ref_metrics: pd.DataFrame | None = None,
) -> plt.Figure:
    sns.set_theme(style="ticks", context="paper", font_scale=0.85)
    fig = plt.figure(figsize=(8.27, 11.69), dpi=300, facecolor="white")
    gs = fig.add_gridspec(
        nrows=4,
        ncols=14,
        left=0.075,
        right=0.955,
        top=0.955,
        bottom=0.24,
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        hspace=0.56,
        wspace=0.28,
    )

    axA = fig.add_subplot(gs[0, 0:4])
    axB = fig.add_subplot(gs[0, 5:9])
    axC = fig.add_subplot(gs[0, 10:14])
    axD = fig.add_subplot(gs[1, 0:14])
    axE = fig.add_subplot(gs[2, 0:8])
    axF = fig.add_subplot(gs[2, 9:14])
    axG = fig.add_subplot(gs[3, 0:8])
    axH = fig.add_subplot(gs[3, 9:13])

    summary_a, summary_b, summary_all = summaries
    _plot_freezing_panel(axA, summary_a, "Sanguino-Gomez and Krugers, 2024", "A")
    _plot_freezing_panel(axB, summary_b, "Sanguino-Gomez et al., 2024", "B")
    _plot_freezing_panel(axC, summary_all, "Combined datasets", "C")

    # Panel D: original keypoint-moseq code uses value_counts().sort_values on moseq_df.
    if usage_df is not None:
        d_source = usage_df.copy().reset_index(drop=True)
    else:
        d_source = agg.sort_values("global_pct", ascending=False).reset_index(drop=True)
    axD.bar(np.arange(len(d_source)), d_source["global_pct"], color="#C77BA5", edgecolor="#FFFFFF", linewidth=0.25)
    cumulative = d_source["global_pct"].cumsum()
    cut_idx = int(np.argmax(cumulative.to_numpy() >= 95.0))
    axD.axvline(cut_idx + 0.5, color="#b0b0b0", linestyle="--", linewidth=0.9)
    axD.legend(
        handles=[plt.Line2D([0], [0], color="#b0b0b0", linestyle="--", linewidth=0.9, label="Threshold = 0.05")],
        frameon=True, fontsize=5, loc="lower right",
    )
    axD.set_ylabel("Total frames covered by syllable (%)", fontsize=6)
    axD.set_xlabel("Syllable", fontsize=6)
    axD.set_xticks(np.arange(len(d_source)))
    axD.set_xticklabels([str(s) for s in d_source["syllable"].astype(int)], rotation=90, fontsize=4.2)
    axD.set_xlim(-0.8, len(d_source) - 0.35)
    axD.set_ylim(0, 20.0)
    axD.set_yticks(np.arange(0, 20.1, 2.5))
    _style_axes(axD)
    _panel_tag(axD, "D", x=-0.045, y=1.04)

    # Panels E/F: original plot_top_95 over per-animal precision/recall rows.
    metrics_pct = metrics_df.copy()
    metrics_pct["precision"] = metrics_pct["precision"] * 100.0
    metrics_pct["recall"] = metrics_pct["recall"] * 100.0
    show_prec = _top_metric_from_original_metrics(metrics_pct, "precision", coverage_pct=95.0)
    show_rec = _top_metric_from_original_metrics(metrics_pct, "recall", coverage_pct=95.0)
    _plot_metric_bars_original(axE, show_prec, "E", "Mean Precision (%)", tag_x=-0.07)
    _plot_metric_bars_original(axF, show_rec, "F", "Mean Recall (%)")

    if not overlap_df.empty:
        plot_overlap_df = overlap_df.copy()
        plot_overlap_df["label"] = plot_overlap_df["recording"].map(_short_id)
        if panel_g_reference_labels:
            label_order = {label: i for i, label in enumerate(panel_g_reference_labels)}
            plot_overlap_df = plot_overlap_df[plot_overlap_df["label"].isin(label_order)].copy()
            plot_overlap_df["reference_order"] = plot_overlap_df["label"].map(label_order)
            plot_overlap_df = plot_overlap_df.sort_values("reference_order")
            overlap_plot = plot_overlap_df.reset_index(drop=True)
        else:
            overlap_records = plot_overlap_df.to_dict("records")
            for row in overlap_records:
                row["label_number"] = _animal_sort_number(str(row["recording"]))
            ctrl = sorted([row for row in overlap_records if row["group"] == "Control"], key=lambda row: row["label_number"])
            els = sorted([row for row in overlap_records if row["group"] == "ELS"], key=lambda row: row["label_number"])
            overlap_plot = pd.DataFrame(ctrl + els)
        colors = [PALETTE.get(g, "#cccccc") for g in overlap_plot["group"]]
        axG.bar(np.arange(len(overlap_plot)), overlap_plot["overlap"], color=colors, edgecolor="#FFFFFF", linewidth=0.2)
        axG.set_xticks(np.arange(len(overlap_plot)))
        g_tick_labels = overlap_plot["label"].tolist()
        axG.set_xticklabels(g_tick_labels, rotation=90)
        axG.set_xlim(-1.15, len(overlap_plot) - 0.35)
    axG.set_ylabel(
        "Overlap between unsupervised and\nsupervised freezing labels (% of frames)",
        fontsize=5.6,
        labelpad=1,
        multialignment="center",
    )
    axG.set_xlabel("Animal", fontsize=6, loc="right", labelpad=8)
    axG.set_ylim(0, 100)
    axG.legend(
        handles=[
            Rectangle((0, 0), 1, 1, color=PALETTE["Control"], label="Control"),
            Rectangle((0, 0), 1, 1, color=PALETTE["ELS"], label="ELS"),
        ],
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.025),
        ncol=2,
        borderaxespad=0.0,
        fontsize=5,
    )
    _style_axes(axG)
    axG.tick_params(axis="x", labelsize=4.2, length=1.8, pad=1)
    _panel_tag(axG, "G", x=-0.07, y=1.18)

    for span in EVENT_SPAN_STARTS_MIN:
        axH.axvspan(span, span + EVENT_SPAN_WIDTH_MIN, color="#F5F5F5", zorder=0)
    for group, data in time_summary.groupby("group"):
        if group not in ("Control", "ELS"):
            continue
        color = PALETTE.get(group, "#4d4d4d")
        axH.plot(data["time_min"], data["mean"], label=group, color=color, linewidth=1.2, marker="o", markersize=2)
        axH.fill_between(data["time_min"], data["mean"] - data["sem"], data["mean"] + data["sem"], color=color, alpha=0.22, linewidth=0)
    axH.set_title("Syllables 0, 28", fontsize=6.5, color="#4b4b4b", pad=4)
    axH.set_xlabel("Time (minutes)", fontsize=6)
    axH.set_ylabel("Freezing (% of time)", fontsize=6)
    axH.set_xlim(0, 7.5)
    axH.set_xticks([1, 2, 3, 4, 5, 6, 7])
    axH.set_ylim(0, 80)
    axH.legend(frameon=False, loc="lower right", fontsize=5, handlelength=1.3)
    axH.text(4.0, 74, "*", ha="center", va="center", color="#4b4b4b", fontsize=17)
    _style_axes(axH)
    _panel_tag(axH, "H", y=1.18)

    return fig


def _plot_metric_bars(ax, show_df: pd.DataFrame, group_agg: pd.DataFrame, metric: str, title: str, tag: str, ylim: float) -> None:
    syllables = show_df["syllable"].astype(int).tolist()
    overall = show_df.set_index("syllable")[metric].reindex(syllables).to_numpy()
    control = group_agg[group_agg["group"] == "Control"].set_index("syllable")[metric].reindex(syllables).to_numpy()
    els = group_agg[group_agg["group"] == "ELS"].set_index("syllable")[metric].reindex(syllables).to_numpy()
    x = np.arange(len(syllables))
    width = 0.25
    ax.bar(x - width, overall * 100.0, width=width, color="#4B4B4B", edgecolor="#FFFFFF", linewidth=0.25, label="All Groups")
    ax.bar(x, control * 100.0, width=width, color=PALETTE["Control"], edgecolor="#FFFFFF", linewidth=0.25, label="Control")
    ax.bar(x + width, els * 100.0, width=width, color=PALETTE["ELS"], edgecolor="#FFFFFF", linewidth=0.25, label="ELS")
    ax.set_title(title, fontsize=6.5, color="#4b4b4b", pad=4)
    ax.set_ylabel(title, fontsize=6)
    ax.set_xlabel("Syllable", fontsize=6, loc="right")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in syllables], rotation=90, fontsize=4.8)
    ax.set_ylim(0, ylim)
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    _style_axes(ax)
    _panel_tag(ax, tag, y=1.05)


def main() -> None:
    args = parse_args()
    if not args.freezing_dir.exists():
        raise FileNotFoundError(f"freezing_dir not found: {args.freezing_dir}")
    if not args.index_csv.exists():
        raise FileNotFoundError(f"index_csv not found: {args.index_csv}")

    excluded = _exclude_tokens(args.exclude_animals)
    group_map = _load_group_map(args.index_csv)
    panel_g_reference_labels = _load_panel_g_reference_labels(args.panel_g_reference_svg)
    include_labels = set(panel_g_reference_labels) if panel_g_reference_labels else None
    if panel_g_reference_labels:
        print(
            f"Using original manuscript cohort: {len(panel_g_reference_labels)} animals "
            f"from {args.panel_g_reference_svg}"
        )
    else:
        print("No manuscript cohort SVG found; using all available animals after explicit exclusions.")

    print("Loading supervised freezing traces...")
    freezing_df = _load_freezing_frame_table(args.freezing_dir, group_map)
    summaries = _freezing_summaries(freezing_df, excluded, include_labels=include_labels)

    print("Loading unsupervised syllable sequences and validation metrics...")
    source_used, source_path, fallback_used = _choose_sequence_source(args)
    print(f"Sequence source: {source_used} ({source_path})")
    if source_used == "moseq_df":
        seq_by_recording = _load_sequences_from_moseq_df(source_path, group_map, excluded, include_labels=include_labels)
        usage_df = _load_syllable_usage_from_moseq_df(source_path, include_labels=include_labels)
    else:
        seq_by_recording = _load_sequences_from_results_pkl(source_path, group_map, excluded, include_labels=include_labels)
        usage_df = None
    freezing_vectors = _load_freezing_vectors(args.freezing_dir)
    metrics_df, agg, group_agg, overlap_df, matched_rec, missing_rec, time_summary = _compute_validation_tables(
        seq_by_recording,
        freezing_vectors,
    )
    print(f"Matched recordings: {matched_rec}; missing freezing records: {missing_rec}")

    ref_metrics = None
    if args.reference_metrics_csv is not None and args.reference_metrics_csv.exists():
        ref_metrics = pd.read_csv(args.reference_metrics_csv)
        print(f"Reference metrics CSV loaded: {args.reference_metrics_csv}")
    else:
        print("No reference metrics CSV found; computing D/E from data.")
    print("Rendering final Figure 2 layout...")
    fig = plot_figure(
        summaries,
        metrics_df,
        agg,
        group_agg,
        overlap_df,
        time_summary,
        usage_df=usage_df,
        panel_g_reference_labels=panel_g_reference_labels,
        ref_metrics=ref_metrics,
    )
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIGURE_OUTPUT_DIR / "figure_2_validation.pdf"
    out_svg = FIGURE_OUTPUT_DIR / "figure_2_validation.svg"
    fig.savefig(out_pdf, facecolor="white")
    fig.savefig(out_svg, facecolor="white")
    LEGACY_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LEGACY_FIGURES_DIR / "figure2.pdf", facecolor="white")
    fig.savefig(LEGACY_FIGURES_DIR / "figure2.svg", facecolor="white")
    fig.savefig(LEGACY_FIGURES_DIR / "figure2.png", dpi=600, facecolor="white")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_svg}")
    print(f"Saved: {LEGACY_FIGURES_DIR / 'figure2.pdf'}")

    import os

    if not os.environ.get("BATCH_MODE"):
        plt.show()


if __name__ == "__main__":
    main()
