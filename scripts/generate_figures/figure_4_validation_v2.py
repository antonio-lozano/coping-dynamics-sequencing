"""
Figure 4 v2 - validation of MoSeq freezing syllables against SimBA freezing.

v2 focuses on legacy parity:
- prefers legacy moseq_df source in auto mode
- falls back to results_pkl with explicit warning/provenance
- keeps Figure 4 plotting layout while preserving the current script untouched
"""

#%%
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import (  # noqa: E402
    BIN_SECONDS,
    FREEZING_DIR,
    FPS,
    INDEX_CSV,
    MANUSCRIPT_FIGURES_DIR as RESULTS_DIR,
    PALETTE,
    RESULTS_RAW_PKL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mode", choices=["auto", "moseq_df", "results_pkl"], default="auto")
    parser.add_argument("--moseq-df", type=Path, default=None, help="Path to legacy moseq_df.csv.")
    parser.add_argument("--results-pkl", type=Path, default=RESULTS_RAW_PKL, help="Path to results pickle fallback.")
    parser.add_argument("--freezing-dir", type=Path, default=FREEZING_DIR)
    parser.add_argument("--index-csv", type=Path, default=INDEX_CSV)
    parser.add_argument("--exclude-animals", type=str, default="Animal_48_6,48_6")
    parser.add_argument("--warn-on-fallback", action="store_true", default=True)
    parser.add_argument("--fail-on-fallback", action="store_true", default=False)
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
    return group_map


def _load_freezing_records(freezing_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for csv_path in sorted(freezing_dir.glob("*_freezing_predictions_only.csv")):
        base = csv_path.name.replace("_freezing_predictions_only.csv", "")
        df = pd.read_csv(csv_path)
        if "Freezing_Jen_0-125_threshold" not in df.columns:
            continue
        vec = df["Freezing_Jen_0-125_threshold"].to_numpy(dtype=int)
        base_norm = _normalize(base)
        base_pref = _parse_moseq_name(base_norm)
        out[base] = vec
        out[base_norm] = vec
        out[base_pref] = vec
    return out


def _resolve_moseq_df_path(explicit: Path | None, results_pkl: Path) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    rp = Path(results_pkl)
    candidates.extend(
        [
            rp.parent / "moseq_df.csv",
            repo_root / "data" / "processed" / "moseq_df.csv",
            Path(r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\2025_01_24-16_44_21\moseq_df.csv"),
            Path(r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\equipo_project_data\moseq_df.csv"),
            Path(r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\moseq_df.csv"),
        ]
    )
    for p in candidates:
        if p is not None and p.exists():
            return p
    return None


def _load_sequences_from_results_pkl(
    results_pkl: Path,
    group_map: dict[str, str],
    excluded: set[str],
) -> dict[str, dict[str, object]]:
    with open(results_pkl, "rb") as f:
        results = pickle.load(f)
    out: dict[str, dict[str, object]] = {}
    for rec, data in results.items():
        if "syllable" not in data:
            continue
        rec_raw = str(rec).strip()
        if _is_excluded(rec_raw, excluded):
            continue
        rec_norm = _normalize(rec_raw)
        rec_pref = _parse_moseq_name(rec_norm)
        grp = group_map.get(rec_raw) or group_map.get(rec_norm) or group_map.get(rec_pref) or "Unknown"
        seq = np.array(data["syllable"], dtype=int)
        out[rec_raw] = {"seq": seq, "group": grp}
    return out


def _load_sequences_from_moseq_df(
    moseq_df_path: Path,
    group_map: dict[str, str],
    excluded: set[str],
) -> dict[str, dict[str, object]]:
    df = pd.read_csv(moseq_df_path)
    required = {"name", "frame_index", "syllable"}
    if not required.issubset(df.columns):
        raise ValueError(f"{moseq_df_path} must include columns: {sorted(required)}")
    out: dict[str, dict[str, object]] = {}
    for rec, gdf in df.groupby("name"):
        rec_raw = str(rec).strip()
        if _is_excluded(rec_raw, excluded):
            continue
        rec_norm = _normalize(rec_raw)
        rec_pref = _parse_moseq_name(rec_norm)
        grp = (
            (str(gdf["group"].iloc[0]) if "group" in gdf.columns else None)
            or group_map.get(rec_raw)
            or group_map.get(rec_norm)
            or group_map.get(rec_pref)
            or "Unknown"
        )
        frames = pd.to_numeric(gdf["frame_index"], errors="coerce").fillna(-1).astype(int).to_numpy()
        sylls = pd.to_numeric(gdf["syllable"], errors="coerce").fillna(-1).astype(int).to_numpy()
        valid = frames >= 0
        if not valid.any():
            continue
        frames = frames[valid]
        sylls = sylls[valid]
        n = int(frames.max()) + 1
        seq = np.full(n, -1, dtype=int)
        seq[frames] = sylls
        out[rec_raw] = {"seq": seq, "group": grp}
    return out


def _choose_source(args: argparse.Namespace) -> tuple[str, Path, bool]:
    moseq_df_found = _resolve_moseq_df_path(args.moseq_df, args.results_pkl)
    if args.source_mode == "moseq_df":
        if moseq_df_found is None:
            raise FileNotFoundError("source-mode=moseq_df but no moseq_df.csv found.")
        return ("moseq_df", moseq_df_found, False)
    if args.source_mode == "results_pkl":
        if not args.results_pkl.exists():
            raise FileNotFoundError(f"results_pkl not found: {args.results_pkl}")
        return ("results_pkl", args.results_pkl, False)

    # auto mode
    if moseq_df_found is not None:
        return ("moseq_df", moseq_df_found, False)
    if not args.results_pkl.exists():
        raise FileNotFoundError(
            "auto mode: no moseq_df source found and results_pkl does not exist."
        )
    if args.fail_on_fallback:
        raise RuntimeError(
            f"auto mode fallback denied: moseq_df missing; would fallback to {args.results_pkl}"
        )
    if args.warn_on_fallback:
        warnings.warn(
            f"Figure 4 v2 fallback: moseq_df source not found. Using results_pkl={args.results_pkl}",
            RuntimeWarning,
        )
    return ("results_pkl", args.results_pkl, True)


def _match_freeze(rec_name: str, freezing_records: dict[str, np.ndarray]) -> np.ndarray | None:
    rec_raw = str(rec_name).strip()
    rec_norm = _normalize(rec_raw)
    rec_pref = _parse_moseq_name(rec_norm)
    return (
        freezing_records.get(rec_raw)
        or freezing_records.get(rec_norm)
        or freezing_records.get(rec_pref)
    )


def main() -> None:
    args = parse_args()
    if not args.freezing_dir.exists():
        raise FileNotFoundError(f"freezing_dir not found: {args.freezing_dir}")
    if not args.index_csv.exists():
        raise FileNotFoundError(f"index_csv not found: {args.index_csv}")

    fps = FPS
    bin_seconds = BIN_SECONDS
    bin_size = int(fps * bin_seconds)
    excluded = _exclude_tokens(args.exclude_animals)

    print("Loading group assignments and freezing CSVs...")
    group_map = _load_group_map(args.index_csv)
    freezing_records = _load_freezing_records(args.freezing_dir)
    print(f"Freezing files loaded: {len(freezing_records)}")

    source_used, source_path, fallback_used = _choose_source(args)
    print(f"Source mode used: {source_used} ({source_path})")

    if source_used == "moseq_df":
        seq_by_recording = _load_sequences_from_moseq_df(source_path, group_map, excluded)
    else:
        seq_by_recording = _load_sequences_from_results_pkl(source_path, group_map, excluded)

    print(f"MoSeq recordings loaded: {len(seq_by_recording)}")
    if not seq_by_recording:
        raise RuntimeError("No MoSeq sequences available after filtering.")

    # --- Metrics (legacy-equivalent style)
    metrics_rows = []
    overlap_rows = []
    target_syll = {0, 28}
    total_freeze_frames = 0
    missing_rec = []
    matched_rec = 0

    for rec, payload in seq_by_recording.items():
        seq = np.array(payload["seq"], dtype=int)
        grp = str(payload["group"])
        freeze = _match_freeze(rec, freezing_records)
        if freeze is None:
            missing_rec.append(rec)
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
            precision = (overlap / total_syll) if total_syll > 0 else 0.0
            recall = (overlap / total_freeze) if total_freeze > 0 else 0.0
            metrics_rows.append(
                {
                    "recording": rec,
                    "group": grp,
                    "syllable": s,
                    "precision": precision,
                    "recall": recall,
                    "frames": total_syll,
                    "tp": overlap,
                    "fp": total_syll - overlap,
                    "fn": total_freeze - overlap,
                }
            )

        selected_idx = set()
        for s in target_syll:
            selected_idx.update(np.flatnonzero(seq == s).tolist())
        overlap_pct = (len(selected_idx.intersection(freeze_idx)) / total_freeze) * 100.0
        overlap_rows.append({"recording": rec, "group": grp, "overlap": overlap_pct})

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        raise RuntimeError("No matched recordings with freezing frames produced metrics.")
    print(f"Matched recordings: {matched_rec}; Missing freezing: {len(missing_rec)}")
    print("Group counts in metrics:")
    print(metrics_df["group"].value_counts())

    agg = (
        metrics_df.groupby("syllable")[["precision", "recall", "frames", "tp", "fp", "fn"]]
        .agg({"precision": "mean", "recall": "mean", "frames": "sum", "tp": "sum", "fp": "sum", "fn": "sum"})
        .reset_index()
    )
    total_frames = agg["frames"].sum()
    agg["freezing_contrib"] = agg["tp"] / total_freeze_frames if total_freeze_frames > 0 else np.nan
    agg["frame_frac"] = agg["frames"] / total_frames if total_frames > 0 else np.nan
    agg = agg.sort_values("precision", ascending=False)

    group_agg = (
        metrics_df.groupby(["syllable", "group"])[["precision", "recall", "frames", "tp", "fp", "fn"]]
        .agg({"precision": "mean", "recall": "mean", "frames": "sum", "tp": "sum", "fp": "sum", "fn": "sum"})
        .reset_index()
    )

    # --- Time-course of S0+S28
    bin_counts = []
    seq_rows = []
    for rec, payload in seq_by_recording.items():
        grp = str(payload["group"])
        if grp not in ("Control", "ELS"):
            continue
        seq = np.array(payload["seq"], dtype=int)
        bins = len(seq) // bin_size
        if bins <= 0:
            continue
        bin_counts.append(bins)
        seq_rows.append((rec, grp, seq))
    if not bin_counts:
        raise RuntimeError("No recordings have enough frames for time-binning.")
    common_bins = min(bin_counts)
    time_rows = []
    for rec, grp, seq in seq_rows:
        seq = seq[: common_bins * bin_size]
        arr = seq.reshape(common_bins, bin_size)
        for b, block in enumerate(arr):
            pct = np.mean(np.isin(block, list(target_syll))) * 100.0
            time_rows.append({"recording": rec, "group": grp, "bin": b, "pct": pct})
    time_df = pd.DataFrame(time_rows)
    time_summary = (
        time_df.groupby(["group", "bin"])["pct"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    time_summary["time_min"] = (time_summary["bin"] + 1.0) * (bin_seconds / 60.0)

    overlap_df = pd.DataFrame(overlap_rows)

    # --- Plot layout (same as Figure 4)
    print("Plotting Figure 4 v2 panels (A-E)...")
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    fig = plt.figure(figsize=(13, 12), dpi=300, facecolor="#FFFFFF")
    axA = fig.add_axes([0.07, 0.72, 0.88, 0.24])
    axB = fig.add_axes([0.07, 0.40, 0.42, 0.22])
    axC = fig.add_axes([0.58, 0.40, 0.36, 0.22])
    axD = fig.add_axes([0.07, 0.07, 0.54, 0.22])
    axE = fig.add_axes([0.70, 0.07, 0.24, 0.22])

    def _style_axes(ax):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#666666")
            ax.spines[spine].set_linewidth(1.0)
        ax.tick_params(axis="both", colors="#4b4b4b", width=0.8, length=3, labelsize=7)

    def _panel_tag(ax, tag):
        ax.text(-0.08, 1.04, tag, transform=ax.transAxes, fontsize=12, fontweight="bold", color="#4b4b4b")

    aggA = agg.sort_values("frames", ascending=False)
    axA.bar(
        np.arange(len(aggA)),
        aggA["frame_frac"] * 100,
        color="#ACDBCA",
        edgecolor="#FFFFFF",
        linewidth=0.6,
    )
    axA.set_title("Total frames covered by syllable (%)", fontsize=11, color="#4b4b4b")
    axA.set_ylabel("Total frames covered by syllable (%)")
    axA.set_xlabel("Syllables (sorted by count)")
    axA.set_xticks(np.arange(len(aggA)))
    axA.set_xticklabels([f"S{s}" for s in aggA["syllable"].astype(int)], rotation=90, fontsize=6)
    axA.set_ylim(0, 20)
    threshold = 0.05
    cut_idx = int(np.sum(aggA["frame_frac"] >= threshold))
    axA.axvline(cut_idx - 0.5, color="#cfcfcf", linestyle="--", linewidth=1)
    _style_axes(axA)
    _panel_tag(axA, "A")

    precision_order = [
        0, 28, 40, 61, 68, 51, 48, 53, 38, 44, 4, 13, 56, 79, 39, 36, 60, 76, 42, 20, 35, 50, 85, 1, 47, 26
    ]
    show_prec = agg[agg["syllable"].isin(precision_order)].set_index("syllable").reindex(precision_order).dropna().reset_index()

    def _group_vals(metric: str):
        overall = show_prec.set_index("syllable")[metric].reindex(show_prec["syllable"]).to_numpy()
        control = (
            group_agg[group_agg["group"] == "Control"]
            .set_index("syllable")[metric]
            .reindex(show_prec["syllable"])
            .to_numpy()
        )
        els = (
            group_agg[group_agg["group"] == "ELS"]
            .set_index("syllable")[metric]
            .reindex(show_prec["syllable"])
            .to_numpy()
        )
        return overall, control, els

    prec_all, prec_ctrl, prec_els = _group_vals("precision")
    x = np.arange(len(show_prec))
    w = 0.25
    axB.bar(x - w, np.array(prec_all) * 100, width=w, color="#4B4B4B", edgecolor="#FFFFFF", linewidth=0.4, label="All Groups")
    axB.bar(x, np.array(prec_els) * 100, width=w, color="#BE7896", edgecolor="#FFFFFF", linewidth=0.4, label="ELS")
    axB.bar(x + w, np.array(prec_ctrl) * 100, width=w, color="#F0BE50", edgecolor="#FFFFFF", linewidth=0.4, label="Control")
    axB.set_title("Mean Precision (%)", fontsize=11, color="#4b4b4b")
    axB.set_ylabel("Mean Precision (%)")
    axB.set_xlabel("Syllable")
    axB.set_xticks(x)
    axB.set_xticklabels([f"S{s}" for s in show_prec["syllable"].astype(int)], rotation=90, fontsize=7)
    axB.set_ylim(0, 100)
    axB.legend(frameon=False, fontsize=9, loc="upper right")
    _style_axes(axB)
    _panel_tag(axB, "B")

    recall_order = [0, 1, 28, 13, 4, 40]
    show_rec = agg[agg["syllable"].isin(recall_order)].set_index("syllable").reindex(recall_order).dropna().reset_index()
    rec_all, rec_ctrl, rec_els = _group_vals("recall")
    xr = np.arange(len(show_rec))
    axC.bar(xr - w, np.array(rec_all[: len(show_rec)]) * 100, width=w, color="#4B4B4B", edgecolor="#FFFFFF", linewidth=0.4, label="All Groups")
    axC.bar(xr, np.array(rec_els[: len(show_rec)]) * 100, width=w, color="#BE7896", edgecolor="#FFFFFF", linewidth=0.4, label="ELS")
    axC.bar(xr + w, np.array(rec_ctrl[: len(show_rec)]) * 100, width=w, color="#F0BE50", edgecolor="#FFFFFF", linewidth=0.4, label="Control")
    axC.set_title("Mean Recall (%)", fontsize=11, color="#4b4b4b")
    axC.set_ylabel("Mean Recall (%)")
    axC.set_xlabel("Syllable")
    axC.set_xticks(xr)
    axC.set_xticklabels([f"S{s}" for s in show_rec["syllable"].astype(int)], rotation=90, fontsize=7)
    axC.set_ylim(0, 80)
    axC.legend(frameon=False, fontsize=9, loc="upper right")
    _style_axes(axC)
    _panel_tag(axC, "C")

    if not overlap_df.empty:
        ctrl = overlap_df[overlap_df["group"] == "Control"].sort_values("recording")
        els = overlap_df[overlap_df["group"] == "ELS"].sort_values("recording")
        overlap_plot = pd.concat([ctrl, els], ignore_index=True)
        colors = [PALETTE.get(g, "#cccccc") for g in overlap_plot["group"]]
        axD.bar(np.arange(len(overlap_plot)), overlap_plot["overlap"], color=colors, edgecolor="#FFFFFF", linewidth=0.3)
        axD.set_xticks(np.arange(len(overlap_plot)))

        def _short_id(name: str) -> str:
            norm = _normalize(str(name))
            m = re.search(r"Animal[_ ]?(\d+(?:[_-]\d+)?)", norm, flags=re.IGNORECASE)
            if m:
                return m.group(1)
            parts = norm.split("_")
            return "_".join(parts[-2:]) if len(parts) >= 2 else norm

        axD.set_xticklabels([_short_id(n) for n in overlap_plot["recording"].tolist()], rotation=90, fontsize=6)
    axD.set_title("D. Overlap (%)", fontsize=11, color="#4b4b4b")
    axD.set_ylabel("Overlap between unsupervised and supervised\nfreezing labels (% of frames)")
    axD.set_ylim(0, 100)
    axD.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["Control"], label="Control"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["ELS"], label="ELS"),
    ], frameon=False, loc="upper right", fontsize=9)
    _style_axes(axD)
    _panel_tag(axD, "D")

    span_width_min = 0.5
    for span in [3.0, 4.5, 6.0]:
        axE.axvspan(span, span + span_width_min, color="#F2F2F2", zorder=0)
    for group, data in time_summary.groupby("group"):
        if group not in ("Control", "ELS"):
            continue
        axE.plot(data["time_min"], data["mean"], label=group, color=PALETTE.get(group, "#4d4d4d"), linewidth=2, marker="o", markersize=3)
        axE.fill_between(data["time_min"], data["mean"] - data["sem"], data["mean"] + data["sem"], color=PALETTE.get(group, "#4d4d4d"), alpha=0.2, linewidth=0)
    axE.set_title("Syllables 0, 28", fontsize=11, color="#4b4b4b")
    axE.set_xlabel("Time (minutes)")
    axE.set_ylabel("Freezing (% of time)")
    axE.set_xlim(0, 8)
    axE.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    axE.set_ylim(0, 80)
    axE.legend(frameon=False, loc="lower right", fontsize=9)
    _style_axes(axE)
    _panel_tag(axE, "E")

    out_png = RESULTS_DIR / "figure_4_validation_v2.png"
    out_pdf = RESULTS_DIR / "figure_4_validation_v2.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    groups_counts = (
        pd.Series([str(v["group"]) for v in seq_by_recording.values()])
        .value_counts()
        .to_dict()
    )
    provenance = {
        "source_mode_requested": args.source_mode,
        "source_mode_used": source_used,
        "source_path_used": str(source_path),
        "fallback_used": bool(fallback_used),
        "fallback_warning": bool(args.warn_on_fallback),
        "freezing_dir": str(args.freezing_dir),
        "index_csv": str(args.index_csv),
        "results_pkl": str(args.results_pkl),
        "moseq_df_arg": str(args.moseq_df) if args.moseq_df is not None else None,
        "excluded_animals": sorted(excluded),
        "n_seq_recordings_loaded": int(len(seq_by_recording)),
        "n_matched_recordings_metrics": int(matched_rec),
        "n_missing_freezing_records": int(len(missing_rec)),
        "recording_counts_by_group": {str(k): int(v) for k, v in groups_counts.items()},
        "fps": int(fps),
        "bin_seconds": int(bin_seconds),
        "notes": (
            "Precision/recall can change if source differs between legacy moseq_df and results_pkl. "
            "This provenance file records exactly which source was used."
        ),
    }
    prov_path = RESULTS_DIR / "figure_4_validation_v2_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Saved: {prov_path}")

    import os
    if not os.environ.get("BATCH_MODE"):
        plt.show()


if __name__ == "__main__":
    main()

