#!/usr/bin/env python
"""Precompute robust pipeline features into a reusable parquet feature store."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import DLC_DIR, INDEX_CSV, RESULTS_CLUSTERS_PKL  # noqa: E402
from src.ml.pose_features import (  # noqa: E402
    compute_kinematic_features,
    load_dlc_csv,
    normalize_recording_name,
    _extract_behavior_labels,
    _match_recording_key,
)
from src.config import BEHAVIOR_MAPPING  # noqa: E402
from scripts.analysis.extended_features import (  # noqa: E402
    DEFAULT_REF_PAIR,
    UNMAPPED_BODYPARTS,
    compute_extended_features,
    preprocess_coordinates,
)
from scripts.analysis.feature_store import (  # noqa: E402
    FeatureStoreMatchKey,
    ensure_store_dir,
    find_manifest_entry,
    load_manifest,
    stable_hash,
    upsert_manifest_entry,
    write_feature_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dlc-dir", type=Path, default=DLC_DIR)
    parser.add_argument("--results-clusters-pkl", type=Path, default=RESULTS_CLUSTERS_PKL)
    parser.add_argument("--index-csv", type=Path, default=INDEX_CSV)
    parser.add_argument("--glob", type=str, default="*DLC*.csv")
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--feature-store-dir", type=Path, default=repo_root / "results" / "model_training" / "feature_store")
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--feature-store-policy", choices=["prefer", "refresh", "off"], default="prefer")
    parser.add_argument("--exclude-bodyparts", type=str, default=None)
    parser.add_argument("--no-extended-features", action="store_true")
    parser.add_argument("--no-coord-preprocess", action="store_true")
    parser.add_argument("--ref-bodyparts", type=str, default="nose,S1")
    return parser.parse_args()


def _parse_bodyparts(raw: str | None, default: set[str] | None = None) -> set[str]:
    if raw is None:
        return set(default or set())
    if raw == "":
        return set()
    return {tok.strip().lower() for tok in raw.split(",") if tok.strip()}


def _resolve_group_map(index_csv: Path) -> dict[str, str]:
    idx = pd.read_csv(index_csv)
    out: dict[str, str] = {}
    for _, row in idx.iterrows():
        raw = str(row["name"]).strip()
        norm = normalize_recording_name(raw)
        out[raw] = str(row["group"])
        out[norm] = str(row["group"])
    return out


def _load_records(
    dlc_dir: Path,
    glob_pat: str,
    results_pkl: Path,
    index_csv: Path,
    max_recordings: int | None,
) -> list[dict]:
    with open(results_pkl, "rb") as f:
        results_dict = pickle.load(f)
    group_map = _resolve_group_map(index_csv)
    csvs = sorted(dlc_dir.glob(glob_pat))
    if max_recordings is not None:
        csvs = csvs[: max(0, int(max_recordings))]
    out = []
    for csv_path in csvs:
        matched_key = _match_recording_key(csv_path.stem, results_dict.keys())
        if matched_key is None:
            continue
        labels = _extract_behavior_labels(results_dict[matched_key], BEHAVIOR_MAPPING)
        if labels is None:
            continue
        try:
            pose_df = load_dlc_csv(csv_path)
        except Exception as exc:
            print(f"Skip {csv_path.name}: {exc}")
            continue
        matched_norm = normalize_recording_name(matched_key)
        group = group_map.get(matched_key) or group_map.get(matched_norm) or "Unknown"
        out.append(
            {
                "recording": str(matched_key),
                "source_path": str(csv_path.resolve()),
                "source_mtime": float(csv_path.stat().st_mtime),
                "pose_df": pose_df,
                "labels": labels,
                "group": group,
            }
        )
    return out


def _pose_qc_stats(pose_df: pd.DataFrame) -> tuple[float, float, float]:
    x_cols = [c for c in pose_df.columns if str(c).endswith("_x")]
    y_cols = [c for c in pose_df.columns if str(c).endswith("_y")]
    if not x_cols or not y_cols:
        return (0.0, 0.0, 0.0)
    xv = pose_df[x_cols].to_numpy(dtype=float)
    yv = pose_df[y_cols].to_numpy(dtype=float)
    finite = np.isfinite(xv) & np.isfinite(yv)
    finite_ratio = float(np.mean(finite)) if finite.size else 0.0
    x_range = float(np.nanmax(xv) - np.nanmin(xv)) if np.isfinite(xv).any() else 0.0
    y_range = float(np.nanmax(yv) - np.nanmin(yv)) if np.isfinite(yv).any() else 0.0
    return (finite_ratio, x_range, y_range)


def main() -> int:
    args = parse_args()
    if not args.dlc_dir.exists():
        raise FileNotFoundError(f"DLC dir not found: {args.dlc_dir}")
    if not args.results_clusters_pkl.exists():
        raise FileNotFoundError(f"Results pickle not found: {args.results_clusters_pkl}")
    if not args.index_csv.exists():
        raise FileNotFoundError(f"Index csv not found: {args.index_csv}")

    dataset_id = args.dataset_id or args.dlc_dir.name
    store_dir = ensure_store_dir(args.feature_store_dir, dataset_id)
    manifest_df = load_manifest(store_dir)
    print(f"Feature store dir: {store_dir}")
    print(f"Existing manifest rows: {len(manifest_df)}")

    excluded = _parse_bodyparts(args.exclude_bodyparts, default=UNMAPPED_BODYPARTS)
    use_extended = not args.no_extended_features
    coord_pre = not args.no_coord_preprocess
    ref_pair = tuple(args.ref_bodyparts.split(",")[:2]) if coord_pre else tuple(DEFAULT_REF_PAIR)

    preprocess_cfg = {"coord_preprocess": coord_pre, "ref_pair": list(ref_pair)}
    feature_cfg = {
        "excluded_bodyparts": sorted(excluded),
        "use_extended": bool(use_extended),
        "fps": float(args.fps),
    }
    preprocess_hash = stable_hash(preprocess_cfg)
    feature_hash = stable_hash(feature_cfg)

    records = _load_records(
        dlc_dir=args.dlc_dir,
        glob_pat=args.glob,
        results_pkl=args.results_clusters_pkl,
        index_csv=args.index_csv,
        max_recordings=args.max_recordings,
    )
    if not records:
        raise RuntimeError("No records loaded for precompute.")

    hits = 0
    writes = 0
    start = time.perf_counter()

    for rec in records:
        key = FeatureStoreMatchKey(
            recording=rec["recording"],
            variant="base",
            source_path=rec["source_path"],
            source_mtime=rec["source_mtime"],
            preprocess_config_hash=preprocess_hash,
            feature_config_hash=feature_hash,
        )
        entry = find_manifest_entry(manifest_df, key)
        if args.feature_store_policy == "prefer" and entry is not None:
            raw_rel = str(entry.get("table_path_raw") or "")
            if raw_rel and (store_dir / raw_rel).exists():
                hits += 1
                continue

        pose_df = rec["pose_df"]
        ref_dist_px = float("nan")
        if coord_pre:
            try:
                pose_df, ref_dist_px = preprocess_coordinates(pose_df, ref_bodyparts=(str(ref_pair[0]), str(ref_pair[1])))
            except Exception as exc:
                warnings.warn(
                    f"{rec['recording']}: preprocessing failed for ref pair {ref_pair}: {exc}. Using raw coordinates.",
                    RuntimeWarning,
                )
                ref_dist_px = float("nan")

        if np.isfinite(ref_dist_px):
            if ref_dist_px < 1e-3:
                warnings.warn(f"{rec['recording']}: ref_dist_px is near zero ({ref_dist_px:.6f}).", RuntimeWarning)
            if ref_dist_px > 2e4:
                warnings.warn(f"{rec['recording']}: ref_dist_px unusually large ({ref_dist_px:.2f}).", RuntimeWarning)

        raw_feat = compute_kinematic_features(
            pose_df,
            fps=int(round(args.fps)),
            excluded_bodyparts=excluded,
        )
        ext_feat = compute_extended_features(pose_df, raw_feat, fps=int(round(args.fps))) if use_extended else pd.DataFrame(index=raw_feat.index)

        raw_export = raw_feat.copy()
        raw_export.insert(0, "time_s", np.arange(len(raw_export), dtype=float) / max(float(args.fps), 1e-8))
        raw_export.insert(0, "frame", np.arange(len(raw_export), dtype=int))
        raw_export.insert(0, "recording", rec["recording"])
        raw_export.insert(0, "group", rec["group"])

        ext_rel = ""
        if use_extended and not ext_feat.empty:
            ext_export = ext_feat.copy()
            ext_export.insert(0, "time_s", np.arange(len(ext_export), dtype=float) / max(float(args.fps), 1e-8))
            ext_export.insert(0, "frame", np.arange(len(ext_export), dtype=int))
            ext_export.insert(0, "recording", rec["recording"])
            ext_export.insert(0, "group", rec["group"])
            ext_rel = write_feature_table(store_dir, ext_export, recording=rec["recording"], variant="base", kind="extended")

        raw_rel = write_feature_table(store_dir, raw_export, recording=rec["recording"], variant="base", kind="raw")
        finite_ratio, x_range_px, y_range_px = _pose_qc_stats(rec["pose_df"])
        entry = {
            "recording": rec["recording"],
            "variant": "base",
            "source_path": rec["source_path"],
            "source_mtime": float(rec["source_mtime"]),
            "n_frames": int(len(raw_feat)),
            "preprocess_config_hash": preprocess_hash,
            "feature_config_hash": feature_hash,
            "table_path_raw": raw_rel,
            "table_path_extended": ext_rel,
            "table_path_model_aligned": "",
            "ref_dist_px": float(ref_dist_px) if np.isfinite(ref_dist_px) else pd.NA,
            "finite_ratio": float(finite_ratio),
            "x_range_px": float(x_range_px),
            "y_range_px": float(y_range_px),
            "source_tag": "precompute_robust_features",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_df = upsert_manifest_entry(store_dir, entry)
        writes += 1

    elapsed = time.perf_counter() - start
    n_frames_total = int(sum(len(r["pose_df"]) for r in records))
    rate = n_frames_total / max(elapsed, 1e-9)
    print("=" * 72)
    print(f"Records: {len(records)} | cache_hits={hits} | writes={writes}")
    print(f"Elapsed: {elapsed:.1f}s | throughput={rate:,.1f} frames/s")
    print(f"Manifest: {store_dir / 'manifest.parquet'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

