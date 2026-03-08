"""
Debug script 2: harmonized Figure 3 vs Figure 4 time-course comparison.

Purpose:
- Use the exact same matched recordings (Control/ELS only) for both signals.
- Compare 30s-bin freezing percentage from SimBA labels (Figure 3 source)
  against 30s-bin S0+S28 percentage from MoSeq syllables (Figure 4 panel E source).
- Export verbose CSVs for direct inspection.
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FREEZING_DIR, INDEX_CSV, RESULTS_RAW_PKL, FPS, BIN_SECONDS

OUT_DIR = REPO_ROOT / "debug" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED = {"Animal_48_6", "48_6"}
TARGET_SYLLABLES = {0, 28}


def _normalize(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def _token(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize(name).lower())


def _load_group_map(index_csv: Path) -> dict[str, str]:
    idx = pd.read_csv(index_csv)
    mapping: dict[str, str] = {}
    for _, row in idx.iterrows():
        raw = str(row["name"]).strip()
        norm = _normalize(raw)
        prefix = norm.split("DLC")[0].rstrip("_") if "DLC" in norm else norm
        grp = str(row["group"])
        mapping[raw] = grp
        mapping[norm] = grp
        mapping[prefix] = grp
        if raw.endswith("_freezing_predictions_only"):
            trimmed = raw.replace("_freezing_predictions_only", "")
            mapping[trimmed] = grp
            mapping[_normalize(trimmed)] = grp
    return mapping


def _load_freezing_tables(freezing_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for csv_path in sorted(freezing_dir.glob("*_freezing_predictions_only.csv")):
        df = pd.read_csv(csv_path)
        if "Freezing_Jen_0-125_threshold" not in df.columns:
            continue
        base = csv_path.name.replace("_freezing_predictions_only.csv", "")
        vec = df["Freezing_Jen_0-125_threshold"].to_numpy(dtype=int)
        base_norm = _normalize(base)
        base_prefix = base_norm.split("DLC")[0].rstrip("_")
        out[base] = vec
        out[base_norm] = vec
        out[base_prefix] = vec
    return out


def _match_freeze_for_recording(
    rec: str,
    freezing_records: dict[str, np.ndarray],
) -> tuple[str | None, np.ndarray | None]:
    rec_raw = str(rec).strip()
    rec_norm = _normalize(rec_raw)
    rec_prefix = rec_norm.split("DLC")[0].split("_resnet")[0].rstrip("_")
    if rec_raw in freezing_records:
        return rec_raw, freezing_records[rec_raw]
    if rec_norm in freezing_records:
        return rec_norm, freezing_records[rec_norm]
    if rec_prefix in freezing_records:
        return rec_prefix, freezing_records[rec_prefix]
    return None, None


def main() -> None:
    print("=== Harmonized Fig3 vs Fig4 Time-Course Debug ===")
    print(f"FREEZING_DIR: {FREEZING_DIR}")
    print(f"INDEX_CSV: {INDEX_CSV}")
    print(f"RESULTS_RAW_PKL: {RESULTS_RAW_PKL}")
    print(f"FPS={FPS}, BIN_SECONDS={BIN_SECONDS}")

    group_map = _load_group_map(INDEX_CSV)
    freezing_records = _load_freezing_tables(FREEZING_DIR)
    excluded_tokens = {_token(x) for x in EXCLUDED}

    with open(RESULTS_RAW_PKL, "rb") as f:
        results = pickle.load(f)

    bin_size = int(FPS * BIN_SECONDS)
    if bin_size <= 0:
        raise ValueError(f"Invalid bin_size computed from FPS={FPS}, BIN_SECONDS={BIN_SECONDS}")

    matched_rows: list[dict[str, Any]] = []
    for rec, data in results.items():
        if "syllable" not in data:
            continue
        rec_raw = str(rec).strip()
        if _token(rec_raw) in excluded_tokens:
            continue

        rec_norm = _normalize(rec_raw)
        rec_prefix = rec_norm.split("DLC")[0].split("_resnet")[0].rstrip("_")
        grp = group_map.get(rec_raw) or group_map.get(rec_norm) or group_map.get(rec_prefix) or "Unknown"
        if grp not in ("Control", "ELS"):
            continue

        match_key, freeze = _match_freeze_for_recording(rec_raw, freezing_records)
        if freeze is None:
            continue

        syll = np.array(data["syllable"])
        n = min(len(syll), len(freeze))
        if n < bin_size:
            continue
        matched_rows.append(
            {
                "recording": rec_raw,
                "group": grp,
                "matched_freeze_key": match_key,
                "n_frames_gt": int(len(syll)),
                "n_frames_pred": int(len(freeze)),
                "n_frames_used": int(n),
                "n_bins_possible": int(n // bin_size),
            }
        )

    matched_df = pd.DataFrame(matched_rows).sort_values(["group", "recording"]).reset_index(drop=True)
    if matched_df.empty:
        raise SystemExit("No harmonized recordings found (Control/ELS + matched freeze + enough frames).")

    common_bins = int(matched_df["n_bins_possible"].min())
    common_frames = common_bins * bin_size
    print(f"Matched recordings: {len(matched_df)}")
    print("Group counts:")
    print(matched_df["group"].value_counts().to_string())
    print(f"common_bins={common_bins}, bin_size={bin_size}, common_frames={common_frames}")

    long_rows: list[dict[str, Any]] = []
    sim_rows: list[dict[str, Any]] = []

    results_index = {str(k).strip(): v for k, v in results.items()}
    for row in matched_df.itertuples(index=False):
        rec = str(row.recording)
        data = results_index[rec]
        _, freeze = _match_freeze_for_recording(rec, freezing_records)
        if freeze is None:
            continue
        syll = np.array(data["syllable"])[:common_frames]
        freeze = np.array(freeze[:common_frames], dtype=int)

        syll_bins = syll.reshape(common_bins, bin_size)
        freeze_bins = freeze.reshape(common_bins, bin_size)

        freeze_pct = freeze_bins.mean(axis=1) * 100.0
        s028_pct = np.isin(syll_bins, list(TARGET_SYLLABLES)).mean(axis=1) * 100.0
        time_min = (np.arange(common_bins) + 1.0) * (BIN_SECONDS / 60.0)

        mae = float(np.mean(np.abs(s028_pct - freeze_pct)))
        corr = float(np.corrcoef(s028_pct, freeze_pct)[0, 1]) if common_bins > 1 else np.nan
        sim_rows.append(
            {
                "recording": rec,
                "group": row.group,
                "mae_pct_points": mae,
                "corr_s028_vs_freeze": corr,
                "mean_freeze_pct": float(np.mean(freeze_pct)),
                "mean_s028_pct": float(np.mean(s028_pct)),
            }
        )

        for b in range(common_bins):
            long_rows.append(
                {
                    "recording": rec,
                    "group": row.group,
                    "bin": int(b),
                    "time_min": float(time_min[b]),
                    "source": "freeze_supervised",
                    "pct": float(freeze_pct[b]),
                }
            )
            long_rows.append(
                {
                    "recording": rec,
                    "group": row.group,
                    "bin": int(b),
                    "time_min": float(time_min[b]),
                    "source": "s0_s28_moseq",
                    "pct": float(s028_pct[b]),
                }
            )

    long_df = pd.DataFrame(long_rows)
    sim_df = pd.DataFrame(sim_rows).sort_values(["group", "recording"]).reset_index(drop=True)

    group_summary = (
        long_df.groupby(["group", "source", "bin", "time_min"])["pct"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .sort_values(["source", "group", "bin"])
    )

    print("\nPer-recording similarity summary:")
    print(sim_df.groupby("group")[["mae_pct_points", "corr_s028_vs_freeze"]].mean().to_string())
    print("\nWorst MAE recordings (top 10):")
    print(sim_df.sort_values("mae_pct_points", ascending=False).head(10).to_string(index=False))

    matched_path = OUT_DIR / "harmonized_fig3_fig4_recordings.csv"
    long_path = OUT_DIR / "harmonized_fig3_fig4_timeseries_long.csv"
    summary_path = OUT_DIR / "harmonized_fig3_fig4_group_summary.csv"
    sim_path = OUT_DIR / "harmonized_fig3_fig4_recording_similarity.csv"

    matched_df.to_csv(matched_path, index=False)
    long_df.to_csv(long_path, index=False)
    group_summary.to_csv(summary_path, index=False)
    sim_df.to_csv(sim_path, index=False)

    print("\nSaved outputs:")
    print(f"- {matched_path}")
    print(f"- {long_path}")
    print(f"- {summary_path}")
    print(f"- {sim_path}")
    print("=== End Harmonized Debug ===")


if __name__ == "__main__":
    main()

