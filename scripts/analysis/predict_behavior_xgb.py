#!/usr/bin/env python
"""Apply a saved XGBoost behavior model to new DLC data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import warnings

import pandas as pd
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import PREDICTIONS_DIR, TO_PREDICT_DIR  # noqa: E402
from src.ml.behavior_xgb import (  # noqa: E402
    align_features_for_inference,
    load_model_artifacts,
    predict_probabilities,
)
from src.ml.pose_features import (  # noqa: E402
    compute_kinematic_features,
    load_dlc_csv,
    robust_normalize_features,
)
from scripts.analysis.feature_store import (  # noqa: E402
    FeatureStoreMatchKey,
    ensure_store_dir,
    find_manifest_entry,
    load_manifest,
    stable_hash,
    upsert_manifest_entry,
    write_feature_table,
    read_feature_table,
)

# Extended features / global normalization (scripts-level; optional import)
try:
    from extended_features import (  # noqa: E402
        DEFAULT_REF_PAIR,
        apply_global_normalizer,
        compute_extended_features,
        has_extended_features,
        load_normalization_stats,
        preprocess_coordinates,
    )
    _HAS_EXTENDED = True
except ImportError:
    _HAS_EXTENDED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to trained model version directory.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing new DLC CSV files.")
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Dataset identifier used in output path. Defaults to input-dir folder name.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PREDICTIONS_DIR,
        help="Base predictions directory.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*DLC*.csv",
        help="Glob pattern for input CSVs.",
    )
    parser.add_argument(
        "--exclude-bodyparts",
        type=str,
        default=None,
        help="Optional comma-separated bodyparts to exclude. Defaults to metadata.excluded_bodyparts.",
    )
    parser.add_argument(
        "--save-computed-features",
        dest="save_computed_features",
        action="store_true",
        help=(
            "Save computed per-frame feature tables alongside predictions "
            "(raw computed + model-aligned). Default: on."
        ),
    )
    parser.add_argument(
        "--no-save-computed-features",
        dest="save_computed_features",
        action="store_false",
        help="Disable computed feature table exports.",
    )
    parser.add_argument(
        "--feature-store-dir",
        type=Path,
        default=repo_root / "results" / "model_training" / "feature_store",
        help="Directory for reusable per-recording feature parquet store.",
    )
    parser.add_argument(
        "--feature-store-policy",
        choices=["prefer", "refresh", "off"],
        default="prefer",
        help="prefer=load/write store, refresh=recompute and overwrite, off=disable store.",
    )
    parser.set_defaults(save_computed_features=True)
    return parser.parse_args()


def _parse_bodyparts(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    return {tok.strip().lower() for tok in raw.split(",") if tok.strip()}


def _make_summary(recording: str, pred_df: pd.DataFrame, class_names: list[str]) -> dict[str, float | str | int]:
    row: dict[str, float | str | int] = {
        "recording": recording,
        "n_frames": int(len(pred_df)),
        "mean_confidence": float(pred_df["pred_confidence"].mean()),
        "median_confidence": float(pred_df["pred_confidence"].median()),
        "%_low_confidence(<0.5)": float((pred_df["pred_confidence"] < 0.5).mean() * 100.0),
    }
    for cls in class_names:
        row[f"%_{cls}"] = float((pred_df["pred_class"] == cls).mean() * 100.0)
    return row


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
    print("=" * 72)
    print("Predicting behavior classes with saved XGBoost model")
    print("=" * 72)

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    model, metadata, load_source = load_model_artifacts(args.model_dir)
    class_names = list(metadata.get("class_names", []))
    feature_names = list(metadata.get("feature_names", []))
    if not class_names or not feature_names:
        raise ValueError("Metadata must include class_names and feature_names.")

    print(f"Loaded model from {args.model_dir} (source={load_source}).")
    print(f"Expected features: {len(feature_names)}; classes: {class_names}")
    model_fps = float(metadata.get("fps", 25.0) or 25.0)
    norm_cfg = metadata.get("feature_normalization", {})
    norm_enabled = bool(norm_cfg.get("applied", False))
    norm_method = str(norm_cfg.get("method", "none"))
    norm_clip = float(norm_cfg.get("clip_value", 4.0) or 4.0)
    if norm_enabled:
        print(f"Feature normalization enabled from metadata: method={norm_method}, clip={norm_clip}")

    # Load global normalization stats if the model was trained with global norm
    global_norm_stats = None
    if norm_enabled and norm_method == "global_robust_iqr":
        stats_file = norm_cfg.get("stats_file", "normalization_stats.json")
        stats_path = args.model_dir / stats_file
        if not _HAS_EXTENDED:
            raise ImportError(
                "Model requires global normalization but extended_features module "
                "is not importable.  Run from scripts/analysis/ directory."
            )
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Normalization stats file not found: {stats_path}. "
                f"Expected alongside model artifacts."
            )
        global_norm_stats = load_normalization_stats(stats_path)
        print(f"Loaded global normalization stats ({len(global_norm_stats)} features) from {stats_path}")

    # Detect whether extended features are needed
    use_extended = _HAS_EXTENDED and has_extended_features(feature_names)
    if use_extended:
        print("Extended features detected in model schema; will compute at inference.")

    meta_excluded = {str(x).strip().lower() for x in metadata.get("excluded_bodyparts", [])}
    arg_excluded = _parse_bodyparts(args.exclude_bodyparts)
    excluded_bodyparts = arg_excluded if args.exclude_bodyparts is not None else meta_excluded
    if excluded_bodyparts:
        print(f"Excluding bodyparts at inference: {sorted(excluded_bodyparts)}")

    # Detect coordinate preprocessing config from metadata
    coord_cfg = metadata.get("coordinate_preprocessing", {})
    coord_preprocess = bool(coord_cfg.get("enabled", False))
    coord_ref_pair = tuple(coord_cfg.get("ref_bodyparts", list(DEFAULT_REF_PAIR))) if _HAS_EXTENDED else None
    if coord_preprocess:
        if not _HAS_EXTENDED:
            raise ImportError(
                "Model requires coordinate preprocessing but extended_features module "
                "is not importable.  Run from scripts/analysis/ directory."
            )
        print(f"Coordinate preprocessing enabled: centroid centering + body-length norm (ref={coord_ref_pair})")

    input_csvs = sorted(args.input_dir.glob(args.glob))
    if not input_csvs:
        raise FileNotFoundError(
            f"No DLC CSV files found in {args.input_dir} matching pattern '{args.glob}'."
        )

    model_version = args.model_dir.name
    dataset_id = args.dataset_id or args.input_dir.name
    output_dir = args.out_dir / dataset_id / model_version
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset id: {dataset_id}")
    feature_store_dir = ensure_store_dir(args.feature_store_dir, dataset_id)
    manifest_df = load_manifest(feature_store_dir)
    if args.feature_store_policy != "off":
        print(f"Feature store: {feature_store_dir} (policy={args.feature_store_policy})")

    frame_rows = []
    summary_rows = []
    computed_raw_rows = []
    computed_model_rows = []
    failed = []
    store_hits_raw = 0
    store_hits_model = 0
    store_writes = 0
    t_store_io = 0.0
    t_store_compute = 0.0

    ref_pair = list(coord_ref_pair) if coord_ref_pair is not None else list(DEFAULT_REF_PAIR)
    preprocess_hash = stable_hash(
        {"coord_preprocess": bool(coord_preprocess), "ref_bodyparts": ref_pair}
    )
    raw_feature_hash = stable_hash(
        {
            "excluded_bodyparts": sorted(excluded_bodyparts),
            "use_extended": bool(use_extended),
            "fps": model_fps,
            "coord_preprocess": bool(coord_preprocess),
            "ref_bodyparts": ref_pair,
        }
    )
    model_feature_hash = stable_hash(
        {
            "raw_feature_hash": raw_feature_hash,
            "norm_enabled": bool(norm_enabled),
            "norm_method": str(norm_method),
            "norm_clip": float(norm_clip),
            "model_version": str(model_version),
            "feature_names": feature_names,
        }
    )

    for csv_path in input_csvs:
        try:
            recording = csv_path.stem
            source_path = str(csv_path.resolve())
            source_mtime = float(csv_path.stat().st_mtime)
            raw_feat_df: pd.DataFrame | None = None
            aligned: pd.DataFrame | None = None
            ref_dist_px: float = float("nan")
            finite_ratio = 0.0
            x_range_px = 0.0
            y_range_px = 0.0

            if args.feature_store_policy == "prefer":
                raw_key = FeatureStoreMatchKey(
                    recording=recording,
                    variant="inference_raw",
                    source_path=source_path,
                    source_mtime=source_mtime,
                    preprocess_config_hash=preprocess_hash,
                    feature_config_hash=raw_feature_hash,
                )
                raw_entry = find_manifest_entry(manifest_df, raw_key)
                if raw_entry is not None:
                    raw_rel = str(raw_entry.get("table_path_raw") or "")
                    if raw_rel and (feature_store_dir / raw_rel).exists():
                        t0 = time.perf_counter()
                        raw_tbl = read_feature_table(feature_store_dir, raw_rel)
                        raw_feat_df = raw_tbl.drop(columns=["group", "recording", "frame", "time_s"], errors="ignore")
                        t_store_io += time.perf_counter() - t0
                        store_hits_raw += 1
                        ref_val = pd.to_numeric(pd.Series([raw_entry.get("ref_dist_px")]), errors="coerce").iloc[0]
                        if pd.notna(ref_val):
                            ref_dist_px = float(ref_val)

                model_key = FeatureStoreMatchKey(
                    recording=recording,
                    variant="inference_model",
                    source_path=source_path,
                    source_mtime=source_mtime,
                    preprocess_config_hash=preprocess_hash,
                    feature_config_hash=model_feature_hash,
                )
                model_entry = find_manifest_entry(manifest_df, model_key)
                if model_entry is not None:
                    model_rel = str(model_entry.get("table_path_model_aligned") or "")
                    if model_rel and (feature_store_dir / model_rel).exists():
                        t0 = time.perf_counter()
                        model_tbl = read_feature_table(feature_store_dir, model_rel)
                        aligned = model_tbl.drop(columns=["group", "recording", "frame", "time_s"], errors="ignore")
                        t_store_io += time.perf_counter() - t0
                        store_hits_model += 1

            if raw_feat_df is None or args.feature_store_policy == "refresh":
                t0_compute = time.perf_counter()
                pose_df = load_dlc_csv(csv_path)
                if coord_preprocess:
                    pose_df, ref_dist_px = preprocess_coordinates(
                        pose_df, ref_bodyparts=coord_ref_pair,
                    )
                    if np.isfinite(ref_dist_px) and ref_dist_px < 1e-3:
                        warnings.warn(f"{recording}: ref_dist_px near zero ({ref_dist_px:.6f})", RuntimeWarning)
                finite_ratio, x_range_px, y_range_px = _pose_qc_stats(pose_df)
                raw_feat_df = compute_kinematic_features(
                    pose_df,
                    excluded_bodyparts=excluded_bodyparts,
                )
                if use_extended:
                    ext_df = compute_extended_features(pose_df, raw_feat_df)
                    raw_feat_df = pd.concat([raw_feat_df, ext_df], axis=1)
                t_store_compute += time.perf_counter() - t0_compute

                if args.feature_store_policy != "off":
                    raw_export = raw_feat_df.copy()
                    raw_export.insert(0, "time_s", np.arange(len(raw_export), dtype=float) / max(model_fps, 1e-8))
                    raw_export.insert(0, "frame", np.arange(len(raw_export), dtype=int))
                    raw_export.insert(0, "recording", recording)
                    raw_export.insert(0, "group", "")
                    raw_rel = write_feature_table(
                        feature_store_dir, raw_export, recording=recording, variant="inference_raw", kind="raw"
                    )
                    manifest_df = upsert_manifest_entry(
                        feature_store_dir,
                        {
                            "recording": recording,
                            "variant": "inference_raw",
                            "source_path": source_path,
                            "source_mtime": source_mtime,
                            "n_frames": int(len(raw_feat_df)),
                            "preprocess_config_hash": preprocess_hash,
                            "feature_config_hash": raw_feature_hash,
                            "table_path_raw": raw_rel,
                            "table_path_extended": "",
                            "table_path_model_aligned": "",
                            "ref_dist_px": float(ref_dist_px) if np.isfinite(ref_dist_px) else pd.NA,
                            "finite_ratio": float(finite_ratio),
                            "x_range_px": float(x_range_px),
                            "y_range_px": float(y_range_px),
                            "source_tag": "predict_behavior_xgb",
                            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    store_writes += 1

            if aligned is None or args.feature_store_policy == "refresh":
                model_feat_df = raw_feat_df.copy()
                if norm_enabled and norm_method == "global_robust_iqr" and global_norm_stats is not None:
                    model_feat_df = apply_global_normalizer(
                        model_feat_df, global_norm_stats, clip_value=norm_clip,
                    )
                elif norm_enabled and norm_method == "robust_iqr_per_recording":
                    model_feat_df, _ = robust_normalize_features(
                        model_feat_df,
                        feature_cols=list(model_feat_df.columns),
                        clip_value=norm_clip,
                    )
                aligned = align_features_for_inference(model_feat_df, feature_names=feature_names)
                if args.feature_store_policy != "off":
                    model_export = aligned.copy()
                    model_export.insert(0, "time_s", np.arange(len(model_export), dtype=float) / max(model_fps, 1e-8))
                    model_export.insert(0, "frame", np.arange(len(model_export), dtype=int))
                    model_export.insert(0, "recording", recording)
                    model_export.insert(0, "group", "")
                    model_rel = write_feature_table(
                        feature_store_dir, model_export, recording=recording, variant="inference_model", kind="model_aligned"
                    )
                    manifest_df = upsert_manifest_entry(
                        feature_store_dir,
                        {
                            "recording": recording,
                            "variant": "inference_model",
                            "source_path": source_path,
                            "source_mtime": source_mtime,
                            "n_frames": int(len(aligned)),
                            "preprocess_config_hash": preprocess_hash,
                            "feature_config_hash": model_feature_hash,
                            "table_path_raw": "",
                            "table_path_extended": "",
                            "table_path_model_aligned": model_rel,
                            "ref_dist_px": float(ref_dist_px) if np.isfinite(ref_dist_px) else pd.NA,
                            "finite_ratio": float(finite_ratio),
                            "x_range_px": float(x_range_px),
                            "y_range_px": float(y_range_px),
                            "source_tag": "predict_behavior_xgb",
                            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    store_writes += 1

            pred_df = predict_probabilities(model, aligned, class_names=class_names)
            pred_df.insert(0, "frame", range(len(pred_df)))
            pred_df.insert(0, "recording", recording)
            frame_rows.append(pred_df)
            summary_rows.append(_make_summary(recording, pred_df, class_names))

            if args.save_computed_features:
                raw_export = raw_feat_df.copy()
                raw_export.insert(0, "frame", range(len(raw_export)))
                raw_export.insert(0, "recording", recording)
                computed_raw_rows.append(raw_export)

                model_export = aligned.copy()
                model_export.insert(0, "frame", range(len(model_export)))
                model_export.insert(0, "recording", recording)
                computed_model_rows.append(model_export)

            print(f"  Processed {csv_path.name}: {len(pred_df)} frames")
        except Exception as exc:
            failed.append((csv_path.name, str(exc)))
            print(f"  Skip {csv_path.name}: {exc}")
            continue

    if not frame_rows:
        fail_msg = "\n".join([f"  - {name}: {msg}" for name, msg in failed])
        raise RuntimeError(f"No valid CSVs processed.\nFailures:\n{fail_msg}")

    frame_df = pd.concat(frame_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("recording").reset_index(drop=True)

    frame_parquet = output_dir / "predictions_frame.parquet"
    frame_csv = output_dir / "predictions_frame.csv"
    summary_csv = output_dir / "predictions_summary.csv"

    frame_df.to_parquet(frame_parquet, index=False)
    frame_df.to_csv(frame_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\nSaved prediction outputs:")
    print(f"  - {frame_parquet}")
    print(f"  - {frame_csv}")
    print(f"  - {summary_csv}")

    if args.save_computed_features and computed_raw_rows and computed_model_rows:
        raw_feat_df_all = pd.concat(computed_raw_rows, ignore_index=True)
        model_feat_df_all = pd.concat(computed_model_rows, ignore_index=True)

        raw_feat_parquet = output_dir / "computed_features_raw.parquet"
        raw_feat_csv = output_dir / "computed_features_raw.csv"
        model_feat_parquet = output_dir / "computed_features_model_aligned.parquet"
        model_feat_csv = output_dir / "computed_features_model_aligned.csv"

        raw_feat_df_all.to_parquet(raw_feat_parquet, index=False)
        raw_feat_df_all.to_csv(raw_feat_csv, index=False)
        model_feat_df_all.to_parquet(model_feat_parquet, index=False)
        model_feat_df_all.to_csv(model_feat_csv, index=False)

        print("  -", raw_feat_parquet)
        print("  -", raw_feat_csv)
        print("  -", model_feat_parquet)
        print("  -", model_feat_csv)

    if failed:
        print("\nCSV files skipped due to read/parse errors:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
    if args.feature_store_policy != "off":
        print(
            "Feature-store summary: "
            f"raw_hits={store_hits_raw}, model_hits={store_hits_model}, writes={store_writes}, "
            f"io={t_store_io:.1f}s, compute={t_store_compute:.1f}s, store={feature_store_dir}"
        )

    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
