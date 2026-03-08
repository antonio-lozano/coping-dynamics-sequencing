"""
Verbose debug audit for subtle differences between:
- scripts/generate_figures/figure_3_freezing.py
- scripts/generate_figures/figure_4_validation.py
- SURF_results_analysis_spyder_CAPITULO.py

This script does not modify figure code. It only reports:
1) Which files/paths are being used.
2) Which animals/recordings are included per figure pipeline.
3) Frame-length alignment between MoSeq and freezing labels.
4) Key constants parsed from SURF script (paths, binning, selected syllables).
"""

from __future__ import annotations

import json
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

SURF_SCRIPT = Path(
    r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\SURF_results_analysis_spyder_CAPITULO.py"
)

# Mirror current exclusions in figure scripts.
FIG3_EXCLUDED = {"Animal_48_6", "48_6"}
FIG4_EXCLUDED = {"Animal_48_6", "48_6"}


def _normalize(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def _token(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize(name).lower())


def _parse_moseq_name(name: str) -> str:
    base = _normalize(name)
    if "DLC" in base:
        base = base.split("DLC")[0].rstrip("_")
    return base


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


def build_fig3_animals_df() -> pd.DataFrame:
    group_map = _load_group_map(INDEX_CSV)
    excluded_tokens = {_token(x) for x in FIG3_EXCLUDED}
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(FREEZING_DIR.glob("*_freezing_predictions_only.csv")):
        base = csv_path.name.replace("_freezing_predictions_only.csv", "")
        base_tok = _token(base)
        if base_tok in excluded_tokens:
            continue
        df = pd.read_csv(csv_path)
        if "Freezing_Jen_0-125_threshold" not in df.columns:
            continue
        norm_base = _normalize(base)
        prefix_base = norm_base.split("DLC")[0].rstrip("_") if "DLC" in norm_base else norm_base
        grp = group_map.get(base) or group_map.get(norm_base) or group_map.get(prefix_base) or "Unknown"
        rows.append(
            {
                "animal": base,
                "animal_prefix": _parse_moseq_name(base),
                "group": grp,
                "n_frames_freezing": int(len(df)),
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "animal"]).reset_index(drop=True)


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


def build_fig4_alignment_df() -> pd.DataFrame:
    group_map = _load_group_map(INDEX_CSV)
    excluded_tokens = {_token(x) for x in FIG4_EXCLUDED}
    freezing_records = _load_freezing_tables(FREEZING_DIR)
    with open(RESULTS_RAW_PKL, "rb") as f:
        results = pickle.load(f)

    rows: list[dict[str, Any]] = []
    for rec, data in results.items():
        if "syllable" not in data:
            continue
        rec_raw = str(rec).strip()
        if _token(rec_raw) in excluded_tokens:
            continue
        match_key, freeze = _match_freeze_for_recording(rec_raw, freezing_records)
        rec_norm = _normalize(rec_raw)
        rec_prefix = rec_norm.split("DLC")[0].split("_resnet")[0].rstrip("_")
        grp = group_map.get(rec_raw) or group_map.get(rec_norm) or group_map.get(rec_prefix) or "Unknown"
        gt_len = int(len(np.array(data["syllable"])))
        pred_len = int(len(freeze)) if freeze is not None else np.nan
        rows.append(
            {
                "recording": rec_raw,
                "recording_prefix": _parse_moseq_name(rec_raw),
                "group": grp,
                "matched_freezing_key": match_key,
                "has_freezing_match": freeze is not None,
                "ground_truth_frames": gt_len,
                "prediction_frames": pred_len,
                "delta_pred_minus_gt": (int(pred_len) - gt_len) if freeze is not None else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "recording"]).reset_index(drop=True)


def parse_surf_settings(surf_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"script_path": str(surf_path), "exists": surf_path.exists()}
    if not surf_path.exists():
        return out
    txt = surf_path.read_text(encoding="utf-8", errors="ignore")

    def _find_all(pattern: str) -> list[str]:
        return re.findall(pattern, txt, flags=re.MULTILINE)

    out["freezing_dir_assignments"] = _find_all(r"^\s*freezing_dir\s*=\s*(.+)$")
    out["project_dir_assignments"] = _find_all(r"^\s*project_dir\s*=\s*(.+)$")
    out["data_dir_assignments"] = _find_all(r"^\s*data_dir\s*=\s*(.+)$")
    out["model_name_assignments"] = _find_all(r"^\s*model_name\s*=\s*(.+)$")
    out["bin_size_frames_assignments"] = _find_all(r"^\s*bin_size_frames\s*=\s*(.+)$")
    out["fps_assignments"] = _find_all(r"^\s*fps\s*=\s*(.+)$")
    out["selected_syllables_assignments"] = _find_all(r"^\s*selected_syllables\s*=\s*(.+)$")
    out["use_freezing_ground_truth_assignments"] = _find_all(
        r"^\s*use_freezing_ground_truth\s*=\s*(.+)$"
    )
    return out


def main() -> None:
    print("=== Debug Audit: Figures vs SURF ===")
    print(f"FREEZING_DIR: {FREEZING_DIR}")
    print(f"INDEX_CSV: {INDEX_CSV}")
    print(f"RESULTS_RAW_PKL: {RESULTS_RAW_PKL}")
    print(f"FPS={FPS}, BIN_SECONDS(config)={BIN_SECONDS}")
    print(f"SURF script path: {SURF_SCRIPT} (exists={SURF_SCRIPT.exists()})")

    fig3_df = build_fig3_animals_df()
    fig4_df = build_fig4_alignment_df()
    surf_settings = parse_surf_settings(SURF_SCRIPT)

    print("\n--- Figure 3 input audit ---")
    print(f"Animals after fig3 exclusions: {len(fig3_df)}")
    if not fig3_df.empty:
        print("Group counts:")
        print(fig3_df["group"].value_counts(dropna=False).to_string())

    print("\n--- Figure 4 input audit ---")
    print(f"Recordings after fig4 exclusions: {len(fig4_df)}")
    if not fig4_df.empty:
        print("Group counts:")
        print(fig4_df["group"].value_counts(dropna=False).to_string())
        print(f"Matched freezing files: {int(fig4_df['has_freezing_match'].sum())}/{len(fig4_df)}")
        frame_mismatch = fig4_df[
            fig4_df["has_freezing_match"]
            & (fig4_df["ground_truth_frames"] != fig4_df["prediction_frames"])
        ]
        print(f"Frame-length mismatches: {len(frame_mismatch)}")

    # Compare animal/recording identity sets by parsed prefix.
    fig3_set = set(fig3_df["animal_prefix"].map(_token).tolist())
    fig4_set = set(fig4_df["recording_prefix"].map(_token).tolist())
    only_fig3 = sorted(fig3_set - fig4_set)
    only_fig4 = sorted(fig4_set - fig3_set)
    print("\n--- Identity set differences (prefix token) ---")
    print(f"Only in Figure 3 set: {len(only_fig3)}")
    print(f"Only in Figure 4 set: {len(only_fig4)}")
    if only_fig3:
        print("Sample only_fig3:", only_fig3[:20])
    if only_fig4:
        print("Sample only_fig4:", only_fig4[:20])

    print("\n--- SURF extracted settings ---")
    for key in (
        "freezing_dir_assignments",
        "project_dir_assignments",
        "data_dir_assignments",
        "model_name_assignments",
        "bin_size_frames_assignments",
        "fps_assignments",
        "selected_syllables_assignments",
        "use_freezing_ground_truth_assignments",
    ):
        vals = surf_settings.get(key, [])
        print(f"{key}: {vals[:5]}")

    # Save verbose outputs for offline diffing.
    fig3_path = OUT_DIR / "fig3_animals_audit.csv"
    fig4_path = OUT_DIR / "fig4_alignment_audit.csv"
    diff_path = OUT_DIR / "fig3_vs_fig4_identity_diff.csv"
    surf_json = OUT_DIR / "surf_settings_extracted.json"

    fig3_df.to_csv(fig3_path, index=False)
    fig4_df.to_csv(fig4_path, index=False)

    diff_df = pd.DataFrame(
        [{"source": "fig3_only", "token": t} for t in only_fig3]
        + [{"source": "fig4_only", "token": t} for t in only_fig4]
    )
    diff_df.to_csv(diff_path, index=False)
    surf_json.write_text(json.dumps(surf_settings, indent=2), encoding="utf-8")

    print("\nSaved outputs:")
    print(f"- {fig3_path}")
    print(f"- {fig4_path}")
    print(f"- {diff_path}")
    print(f"- {surf_json}")
    print("=== End Debug Audit ===")


if __name__ == "__main__":
    main()
