#!/usr/bin/env python
"""Robust XGBoost pretraining with augmentation, extended features, and global normalization.

This script wraps the existing ``src/ml`` library to produce a model that
generalises better across datasets.  It adds three layers on top of the
standard ``train_behavior_xgb.py`` pipeline:

1. **Pose augmentation** – mild rigid transforms (rotation ±15°,
   translation ±5 %, scale jitter ±10 %) applied to raw DLC x/y
   coordinates *before* feature extraction.  Each recording is augmented
   ``--n-augmentations`` times → multiplies training data.

2. **Extended features** – temporal context (lags, deltas, acceleration),
   postural shape (body area, elongation, limb symmetry, curvature) and
   frequency-domain (rolling FFT energy / dominant frequency).

3. **Global normalization** – median/IQR computed over the *entire*
   training set (after augmentation) and saved as
   ``normalization_stats.json`` in the model directory so the same stats
   can be applied at inference.

All helpers live in ``scripts/analysis/extended_features.py``;
no files under ``src/`` are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import (  # noqa: E402
    DLC_DIR,
    FPS,
    INDEX_CSV,
    MODELS_DIR,
    RESULTS_CLUSTERS_PKL,
)
from src.ml.behavior_xgb import (  # noqa: E402
    build_feature_matrix,
    compute_cv_metrics,
    encode_labels,
    get_xgb_params,
    n_cv_folds,
    save_model_artifacts,
    stratified_subsample_by_label,
    train_xgb_classifier,
)
from src.ml.pose_features import (  # noqa: E402
    compute_kinematic_features,
    load_dlc_csv,
)

# Import the new helpers from the scripts-level module.
from extended_features import (  # noqa: E402
    DEFAULT_REF_PAIR,
    UNMAPPED_BODYPARTS,
    augment_pose,
    compute_extended_features,
    fit_global_normalizer,
    preprocess_coordinates,
    save_normalization_stats,
)
from feature_store import (  # noqa: E402
    FeatureStoreMatchKey,
    ensure_store_dir,
    find_manifest_entry,
    load_manifest,
    stable_hash,
    upsert_manifest_entry,
    write_feature_table,
    read_feature_table,
)


def _git_commit_hash(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dlc-dir", type=Path, default=DLC_DIR)
    parser.add_argument("--results-clusters-pkl", type=Path, default=RESULTS_CLUSTERS_PKL)
    parser.add_argument("--index-csv", type=Path, default=INDEX_CSV)
    parser.add_argument("--model-version", type=str,
                        default=datetime.now().strftime("%Y-%m-%d_robust_v1"))
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--max-samples-per-class", type=int, default=None)

    # Augmentation
    parser.add_argument("--n-augmentations", type=int, default=3,
                        help="Number of augmented copies per recording (0 = no augmentation).")
    parser.add_argument("--rotation-range", type=float, default=15.0,
                        help="Max rotation in degrees (±).")
    parser.add_argument("--translation-range", type=float, default=0.05,
                        help="Max translation as fraction of bounding-box span.")
    parser.add_argument("--scale-range", type=float, default=0.10,
                        help="Max isotropic scale jitter (±).")

    # Normalization
    parser.add_argument("--normalization-clip", type=float, default=4.0)

    # Extended features
    parser.add_argument("--no-extended-features", action="store_true",
                        help="Disable extended feature computation.")

    # XGBoost
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tree-method", type=str, default="hist")

    parser.add_argument("--exclude-bodyparts", type=str, default=None,
                        help="Comma-separated bodyparts to exclude.  "
                             "Default: unmapped bodyparts (H2R,H2L,B1R,B3R,B1L,B3L,S2).")
    # Coordinate preprocessing
    parser.add_argument("--no-coord-preprocess", action="store_true",
                        help="Disable centroid centering + body-length normalisation.")
    parser.add_argument("--ref-bodyparts", type=str, default="nose,S1",
                        help="Two bodyparts (comma-sep) for body-length reference.")
    # Feature caching
    parser.add_argument("--feature-cache", type=Path, default=None,
                        help="Path to cache precomputed features (.parquet).  "
                             "If the file exists and the config hash matches, "
                             "feature extraction is skipped entirely.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force recomputation even if cache exists.")
    parser.add_argument("--feature-store-dir", type=Path,
                        default=repo_root / "results" / "model_training" / "feature_store",
                        help="Directory for reusable per-recording feature parquet store.")
    parser.add_argument("--feature-store-policy", choices=["prefer", "refresh", "off"],
                        default="prefer",
                        help="prefer=load/write store, refresh=recompute and overwrite, off=disable store.")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny model/data settings for smoke tests.")
    return parser.parse_args()


def _parse_bodyparts(raw: str | None, default: set[str] | None = None) -> set[str]:
    if raw is None:
        return set(default) if default else set()
    if not raw:
        return set()
    return {tok.strip().lower() for tok in raw.split(",") if tok.strip()}


def _feature_cache_key(
    dlc_dir: Path,
    results_pkl: Path,
    excluded: set[str],
    n_aug: int,
    extended: bool,
    coord_preprocess: bool,
    ref_pair: tuple[str, ...] | None,
    aug_kwargs: dict,
    max_recordings: int | None,
) -> str:
    """Deterministic hash of everything that affects the feature table."""
    parts = [
        str(dlc_dir.resolve()),
        str(results_pkl.resolve()),
        ",".join(sorted(excluded)),
        str(n_aug),
        str(extended),
        str(coord_preprocess),
        str(ref_pair),
        str(sorted(aug_kwargs.items())),
        str(max_recordings),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _save_eval_artifacts(model_dir: Path, metrics: dict, class_names: list[str]) -> dict[str, Path]:
    saved: dict[str, Path] = {}
    cm_counts = metrics.get("confusion_matrix_oof_counts")
    cm_norm = metrics.get("confusion_matrix_oof_normalized")
    per_class_prf = metrics.get("per_class_prf_oof")
    if cm_counts is None or cm_norm is None or per_class_prf is None:
        return saved

    cm_counts_df = pd.DataFrame(cm_counts, index=class_names, columns=class_names)
    cm_norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    prf_df = pd.DataFrame(per_class_prf).T

    counts_csv = model_dir / "cv_confusion_matrix_oof_counts.csv"
    norm_csv = model_dir / "cv_confusion_matrix_oof_normalized.csv"
    report_csv = model_dir / "cv_classification_report_oof.csv"
    cm_counts_df.to_csv(counts_csv)
    cm_norm_df.to_csv(norm_csv)
    prf_df.to_csv(report_csv)
    saved["cv_confusion_matrix_oof_counts_csv"] = counts_csv
    saved["cv_confusion_matrix_oof_normalized_csv"] = norm_csv
    saved["cv_classification_report_oof_csv"] = report_csv

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=200, facecolor="white")
    sns.heatmap(cm_counts_df, annot=True, fmt="d", cmap="Blues", cbar=False,
                linewidths=0.3, linecolor="#EEEEEE", ax=axes[0])
    axes[0].set_title("OOF Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    sns.heatmap(cm_norm_df, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                linewidths=0.3, linecolor="#EEEEEE", vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("OOF Confusion Matrix (Row-normalized)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.tight_layout()

    cm_png = model_dir / "cv_confusion_matrix_oof.png"
    cm_pdf = model_dir / "cv_confusion_matrix_oof.pdf"
    fig.savefig(cm_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(cm_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved["cv_confusion_matrix_oof_png"] = cm_png
    saved["cv_confusion_matrix_oof_pdf"] = cm_pdf
    return saved


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _load_raw_pose_and_labels(
    dlc_dir: Path,
    results_pkl: Path,
    index_csv: Path,
    fps: int,
    csv_glob: str = "*DLC*.csv",
    max_csv_files: int | None = None,
    excluded_bodyparts: set[str] | None = None,
) -> list[dict]:
    """Load DLC CSVs and match them to MoSeq syllable labels.

    Returns a list of dicts, each with keys:
        * ``recording``  – matched key
        * ``pose_df``    – raw x/y DataFrame
        * ``labels``     – list[str] of behavior labels
        * ``group``      – group assignment
    """
    import pickle
    from src.config import BEHAVIOR_MAPPING
    from src.ml.pose_features import normalize_recording_name, _match_recording_key, _extract_behavior_labels

    mapping = BEHAVIOR_MAPPING
    with open(results_pkl, "rb") as f:
        results_dict = pickle.load(f)

    idx_df = pd.read_csv(index_csv)
    name_to_group = {
        normalize_recording_name(row["name"]): row["group"]
        for _, row in idx_df.iterrows()
    }

    dlc_csvs = sorted(dlc_dir.glob(csv_glob))
    if max_csv_files is not None:
        dlc_csvs = dlc_csvs[: max(0, int(max_csv_files))]

    records: list[dict] = []
    for csv_path in dlc_csvs:
        matched_key = _match_recording_key(csv_path.stem, results_dict.keys())
        if matched_key is None:
            print(f"  Skip {csv_path.name}: no matching recording key.")
            continue
        rec_data = results_dict[matched_key]
        behavior_labels = _extract_behavior_labels(rec_data, mapping)
        if behavior_labels is None:
            print(f"  Skip {matched_key}: no syllable labels.")
            continue
        try:
            pose_df = load_dlc_csv(csv_path)
        except Exception as exc:
            print(f"  Skip {csv_path.name}: {exc}")
            continue

        # Resolve group
        group = "Unknown"
        matched_norm = normalize_recording_name(matched_key)
        for name_key, g in name_to_group.items():
            if name_key in matched_norm:
                group = g
                break

        records.append({
            "recording": matched_key,
            "source_path": str(csv_path.resolve()),
            "source_mtime": float(csv_path.stat().st_mtime),
            "pose_df": pose_df,
            "labels": behavior_labels,
            "group": group,
        })
        print(f"  Loaded {csv_path.name} -> {matched_key} ({len(pose_df)} frames)")

    return records


def _build_feature_table(
    records: list[dict],
    fps: int,
    excluded_bodyparts: set[str] | None,
    extended: bool,
    n_augmentations: int,
    aug_kwargs: dict,
    coord_preprocess: bool = True,
    ref_bodyparts: tuple[str, str] = DEFAULT_REF_PAIR,
    precomputed_base_features: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame | None:
    """Build labelled frame-level feature table with optional augmentation and extended features."""
    all_rows: list[pd.DataFrame] = []

    for rec in records:
        pose_df = rec["pose_df"]
        labels = rec["labels"]
        recording = rec["recording"]
        group = rec["group"]

        # Coordinate preprocessing: centroid centering + body-length normalisation
        if coord_preprocess:
            pose_df, _ref = preprocess_coordinates(pose_df, ref_bodyparts=ref_bodyparts)

        # List of (suffix, pose_df) — original + augmented
        variants: list[tuple[str, pd.DataFrame]] = [("", pose_df)]
        for aug_idx in range(n_augmentations):
            aug = augment_pose(pose_df, seed=42 + aug_idx, **aug_kwargs)
            variants.append((f"_aug{aug_idx}", aug))

        for suffix, pdf in variants:
            try:
                if suffix == "" and precomputed_base_features and recording in precomputed_base_features:
                    feat_df = precomputed_base_features[recording].copy()
                else:
                    base_feat = compute_kinematic_features(
                        pdf, fps=fps, excluded_bodyparts=excluded_bodyparts,
                    )
                    if extended:
                        ext_feat = compute_extended_features(pdf, base_feat, fps=fps)
                        feat_df = pd.concat([base_feat, ext_feat], axis=1)
                    else:
                        feat_df = base_feat
            except Exception as exc:
                print(f"  Feature extraction failed for {recording}{suffix}: {exc}")
                continue

            min_len = min(len(feat_df), len(labels))
            if min_len == 0:
                continue
            feat_df = feat_df.iloc[:min_len].copy()
            feat_df["behavior_cluster"] = labels[:min_len]
            feat_df["recording"] = f"{recording}{suffix}"
            feat_df["group"] = group
            all_rows.append(feat_df)

    if not all_rows:
        return None
    return pd.concat(all_rows, ignore_index=True)


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


def _load_or_compute_base_features_from_store(
    *,
    records: list[dict],
    fps: int,
    excluded_bodyparts: set[str] | None,
    extended: bool,
    coord_preprocess: bool,
    ref_bodyparts: tuple[str, str],
    store_dir: Path,
    policy: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    if policy == "off":
        return {}, {"hits": 0.0, "writes": 0.0, "io_s": 0.0, "compute_s": 0.0}

    manifest_df = load_manifest(store_dir)
    preprocess_hash = stable_hash(
        {"coord_preprocess": bool(coord_preprocess), "ref_bodyparts": list(ref_bodyparts)}
    )
    feature_hash = stable_hash(
        {"excluded_bodyparts": sorted(excluded_bodyparts or set()), "extended": bool(extended), "fps": int(fps)}
    )

    hits = 0
    writes = 0
    io_s = 0.0
    compute_s = 0.0
    out: dict[str, pd.DataFrame] = {}

    for rec in records:
        key = FeatureStoreMatchKey(
            recording=str(rec["recording"]),
            variant="base",
            source_path=str(rec.get("source_path", "")),
            source_mtime=float(rec.get("source_mtime", 0.0)),
            preprocess_config_hash=preprocess_hash,
            feature_config_hash=feature_hash,
        )
        entry = None if policy == "refresh" else find_manifest_entry(manifest_df, key)
        if entry is not None:
            raw_rel = str(entry.get("table_path_raw") or "")
            ext_rel = str(entry.get("table_path_extended") or "")
            raw_exists = bool(raw_rel) and (store_dir / raw_rel).exists()
            ext_ok = (not extended) or (bool(ext_rel) and (store_dir / ext_rel).exists())
            if raw_exists and ext_ok:
                t0 = time.perf_counter()
                raw_tbl = read_feature_table(store_dir, raw_rel)
                raw_feat = raw_tbl.drop(
                    columns=["group", "recording", "frame", "time_s"], errors="ignore"
                )
                if extended:
                    ext_tbl = read_feature_table(store_dir, ext_rel)
                    ext_feat = ext_tbl.drop(
                        columns=["group", "recording", "frame", "time_s"], errors="ignore"
                    )
                    feat_df = pd.concat([raw_feat.reset_index(drop=True), ext_feat.reset_index(drop=True)], axis=1)
                else:
                    feat_df = raw_feat
                io_s += time.perf_counter() - t0
                out[str(rec["recording"])] = feat_df
                hits += 1
                continue

        t0 = time.perf_counter()
        pose_df = rec["pose_df"]
        ref_dist_px = float("nan")
        if coord_preprocess:
            try:
                pose_df, ref_dist_px = preprocess_coordinates(
                    pose_df, ref_bodyparts=(str(ref_bodyparts[0]), str(ref_bodyparts[1]))
                )
            except Exception as exc:
                warnings.warn(
                    f"{rec['recording']}: preprocessing failed for {ref_bodyparts}: {exc}. Using raw coordinates.",
                    RuntimeWarning,
                )
                ref_dist_px = float("nan")

        if np.isfinite(ref_dist_px):
            if ref_dist_px < 1e-3:
                warnings.warn(f"{rec['recording']}: ref_dist_px near zero ({ref_dist_px:.6f})", RuntimeWarning)
            if ref_dist_px > 2e4:
                warnings.warn(f"{rec['recording']}: ref_dist_px unusually large ({ref_dist_px:.2f})", RuntimeWarning)

        raw_feat = compute_kinematic_features(
            pose_df, fps=fps, excluded_bodyparts=excluded_bodyparts,
        )
        ext_feat = compute_extended_features(pose_df, raw_feat, fps=fps) if extended else pd.DataFrame(index=raw_feat.index)
        compute_s += time.perf_counter() - t0

        out[str(rec["recording"])] = (
            pd.concat([raw_feat.reset_index(drop=True), ext_feat.reset_index(drop=True)], axis=1)
            if extended else raw_feat.copy()
        )

        raw_export = raw_feat.copy()
        raw_export.insert(0, "time_s", np.arange(len(raw_export), dtype=float) / max(float(fps), 1e-8))
        raw_export.insert(0, "frame", np.arange(len(raw_export), dtype=int))
        raw_export.insert(0, "recording", str(rec["recording"]))
        raw_export.insert(0, "group", str(rec.get("group", "Unknown")))
        raw_rel = write_feature_table(
            store_dir, raw_export, recording=str(rec["recording"]), variant="base", kind="raw"
        )

        ext_rel = ""
        if extended and not ext_feat.empty:
            ext_export = ext_feat.copy()
            ext_export.insert(0, "time_s", np.arange(len(ext_export), dtype=float) / max(float(fps), 1e-8))
            ext_export.insert(0, "frame", np.arange(len(ext_export), dtype=int))
            ext_export.insert(0, "recording", str(rec["recording"]))
            ext_export.insert(0, "group", str(rec.get("group", "Unknown")))
            ext_rel = write_feature_table(
                store_dir, ext_export, recording=str(rec["recording"]), variant="base", kind="extended"
            )

        finite_ratio, x_range_px, y_range_px = _pose_qc_stats(rec["pose_df"])
        manifest_df = upsert_manifest_entry(
            store_dir,
            {
                "recording": str(rec["recording"]),
                "variant": "base",
                "source_path": str(rec.get("source_path", "")),
                "source_mtime": float(rec.get("source_mtime", 0.0)),
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
                "source_tag": "pretrain_xgb_robust",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        writes += 1

    return out, {"hits": float(hits), "writes": float(writes), "io_s": float(io_s), "compute_s": float(compute_s)}


def main() -> int:
    args = parse_args()
    print("=" * 72)
    print("Robust XGBoost pretraining (augmentation + extended features + global norm)")
    print("=" * 72)

    for path, label in [(args.dlc_dir, "DLC directory"),
                        (args.results_clusters_pkl, "Results pickle"),
                        (args.index_csv, "Index CSV")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    excluded_bodyparts = _parse_bodyparts(args.exclude_bodyparts, default=UNMAPPED_BODYPARTS)
    if excluded_bodyparts:
        print(f"Excluding bodyparts: {sorted(excluded_bodyparts)}")

    # Coordinate preprocessing config
    coord_preprocess = not args.no_coord_preprocess
    ref_pair = tuple(args.ref_bodyparts.split(",")[:2]) if coord_preprocess else None
    if coord_preprocess:
        print(f"Coordinate preprocessing: centroid centering + body-length norm (ref={ref_pair})")

    # --- 1. Feature computation or cache load ---------------------------
    n_aug = 0 if args.quick else args.n_augmentations
    aug_kwargs = {
        "rotation_range_deg": args.rotation_range,
        "translation_range_frac": args.translation_range,
        "scale_range": args.scale_range,
    }
    use_extended = not args.no_extended_features

    # Compute cache key + resolve cache path
    cache_key = _feature_cache_key(
        args.dlc_dir, args.results_clusters_pkl, excluded_bodyparts,
        n_aug, use_extended, coord_preprocess, ref_pair, aug_kwargs,
        args.max_recordings,
    )
    cache_path = args.feature_cache
    if cache_path is None:
        cache_path = repo_root / "results" / "model_training" / "feature_cache" / f"features_{cache_key}.parquet"

    cache_hit = cache_path.exists() and not args.no_cache and not args.quick
    feature_store_stats = {"hits": 0.0, "writes": 0.0, "io_s": 0.0, "compute_s": 0.0}
    feature_store_dir = ensure_store_dir(args.feature_store_dir, f"training_{args.dlc_dir.name}")
    if cache_hit:
        print(f"\n--- Loading cached features from {cache_path} ---")
        t0 = time.perf_counter()
        df = pd.read_parquet(cache_path)
        print(f"Loaded {len(df):,} frames in {time.perf_counter() - t0:.1f}s (cache key: {cache_key})")
    else:
        # --- Load raw pose + labels ---
        print("\nLoading DLC recordings and MoSeq labels...")
        records = _load_raw_pose_and_labels(
            dlc_dir=args.dlc_dir,
            results_pkl=args.results_clusters_pkl,
            index_csv=args.index_csv,
            fps=FPS,
            max_csv_files=args.max_recordings,
            excluded_bodyparts=excluded_bodyparts,
        )
        if not records:
            raise RuntimeError("No recordings loaded; cannot train model.")
        print(f"Loaded {len(records)} recordings.")

        # --- Load/store reusable per-recording base features ----------
        precomputed_base, feature_store_stats = _load_or_compute_base_features_from_store(
            records=records,
            fps=FPS,
            excluded_bodyparts=excluded_bodyparts,
            extended=use_extended,
            coord_preprocess=coord_preprocess,
            ref_bodyparts=ref_pair if ref_pair else DEFAULT_REF_PAIR,
            store_dir=feature_store_dir,
            policy=str(args.feature_store_policy),
        )
        if args.feature_store_policy != "off":
            print(
                "Feature-store stats: "
                f"hits={int(feature_store_stats['hits'])}, writes={int(feature_store_stats['writes'])}, "
                f"io={feature_store_stats['io_s']:.1f}s, compute={feature_store_stats['compute_s']:.1f}s, "
                f"store={feature_store_dir}"
            )

        # --- Build features (augmented + extended) ---
        print(f"\nBuilding features: augmentations={n_aug}, extended={use_extended}, coord_preprocess={coord_preprocess}")
        t0 = time.perf_counter()
        df = _build_feature_table(
            records, fps=FPS, excluded_bodyparts=excluded_bodyparts,
            extended=use_extended, n_augmentations=n_aug, aug_kwargs=aug_kwargs,
            coord_preprocess=coord_preprocess,
            ref_bodyparts=ref_pair if ref_pair else DEFAULT_REF_PAIR,
            precomputed_base_features=precomputed_base,
        )
        elapsed = time.perf_counter() - t0
        if df is None or df.empty:
            raise RuntimeError("No labelled features were generated; cannot train model.")
        print(f"Total frames (including augmentation): {len(df):,}  ({elapsed:.1f}s)")

        # Save cache for next run
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"Cached features to {cache_path} (key: {cache_key})")

    # --- 2. Subsampling -----------------------------------------------
    if args.quick:
        print("QUICK mode: stratified subsample (max 250 frames/class).")
        df = stratified_subsample_by_label(df, "behavior_cluster", 250, 42)
        if df.empty:
            raise RuntimeError("No samples after QUICK subsampling.")
        print(f"Using {len(df):,} frames for training.")
    elif args.max_samples_per_class is not None:
        print(f"Applying stratified cap: max {args.max_samples_per_class} frames/class.")
        df = stratified_subsample_by_label(
            df, "behavior_cluster", args.max_samples_per_class, 42,
        )
        if df.empty:
            raise RuntimeError("No samples after stratified subsampling.")
        print(f"Using {len(df):,} frames for training after class cap.")

    # --- 4. Global normalization ---------------------------------------
    excluded_meta = {"behavior_cluster", "recording", "group"}
    feature_cols = [c for c in df.columns if c not in excluded_meta]

    print(f"\nApplying global robust normalization (clip={args.normalization_clip})...")
    feat_only = df[feature_cols]
    norm_df, norm_stats = fit_global_normalizer(
        feat_only, feature_cols=feature_cols, clip_value=args.normalization_clip,
    )
    df.loc[:, feature_cols] = norm_df[feature_cols].to_numpy()
    print(f"Normalization stats computed for {len(norm_stats)} features.")

    # --- 5. Train XGBoost ---------------------------------------------
    x_df, x = build_feature_matrix(df, label_col="behavior_cluster")
    y, label_encoder, classes = encode_labels(df["behavior_cluster"])
    class_names = [str(c) for c in classes.tolist()]
    label_mapping = {cls: int(i) for i, cls in enumerate(class_names)}
    class_counts = (
        df["behavior_cluster"].value_counts().sort_index().rename_axis("class").to_dict()
    )

    params = get_xgb_params(quick=args.quick)
    if args.n_estimators is not None:
        params["n_estimators"] = int(args.n_estimators)
    if args.max_depth is not None:
        params["max_depth"] = int(args.max_depth)
    if args.learning_rate is not None:
        params["learning_rate"] = float(args.learning_rate)
    params["device"] = str(args.device)
    params["tree_method"] = str(args.tree_method)

    folds = int(args.cv_folds) if args.cv_folds is not None else n_cv_folds(quick=args.quick)
    print("\nTraining configuration:")
    print(f"  device={params['device']}, tree_method={params['tree_method']}")
    print(f"  n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, "
          f"learning_rate={params['learning_rate']}, cv_folds={folds}")

    print(f"Running {folds}-fold cross-validation...")
    metrics = compute_cv_metrics(x, y, classes, params, folds=folds, random_state=42)
    print(f"Mean CV accuracy: {metrics['overall_accuracy_mean']:.3f}")
    print(f"OOF accuracy:     {metrics['overall_accuracy_oof']:.3f}")
    print(f"OOF macro F1:     {metrics['macro_f1_oof']:.3f}")

    print("Training final model on all frames...")
    model = train_xgb_classifier(x=x, y=y, xgb_params=params)

    preprocess_hash = stable_hash(
        {"coord_preprocess": bool(coord_preprocess), "ref_bodyparts": list(ref_pair if ref_pair else DEFAULT_REF_PAIR)}
    )
    feature_hash = stable_hash(
        {"excluded_bodyparts": sorted(excluded_bodyparts or set()), "extended": bool(use_extended), "fps": int(FPS)}
    )
    manifest_now = load_manifest(feature_store_dir)
    ref_dist_summary: dict[str, float | int | None] = {
        "n_records_with_ref_dist": 0,
        "min_ref_dist_px": None,
        "median_ref_dist_px": None,
        "max_ref_dist_px": None,
        "mean_finite_ratio": None,
    }
    if not manifest_now.empty:
        used_original_recs = sorted(
            df["recording"].astype(str).apply(lambda r: r.split("_aug")[0]).unique().tolist()
        )
        m = manifest_now.copy()
        m = m[
            (m["variant"].astype(str) == "base")
            & (m["recording"].astype(str).isin(used_original_recs))
            & (m["preprocess_config_hash"].astype(str) == preprocess_hash)
            & (m["feature_config_hash"].astype(str) == feature_hash)
        ].copy()
        if not m.empty:
            ref_vals = pd.to_numeric(m["ref_dist_px"], errors="coerce").dropna().to_numpy(dtype=float)
            finite_vals = pd.to_numeric(m["finite_ratio"], errors="coerce").dropna().to_numpy(dtype=float)
            if ref_vals.size > 0:
                ref_dist_summary = {
                    "n_records_with_ref_dist": int(ref_vals.size),
                    "min_ref_dist_px": float(np.min(ref_vals)),
                    "median_ref_dist_px": float(np.median(ref_vals)),
                    "max_ref_dist_px": float(np.max(ref_vals)),
                    "mean_finite_ratio": float(np.mean(finite_vals)) if finite_vals.size > 0 else None,
                }

    # --- 6. Save artifacts --------------------------------------------
    model_dir = args.out_dir / args.model_version
    metadata = {
        "model_version": args.model_version,
        "feature_names": x_df.columns.tolist(),
        "class_names": class_names,
        "label_mapping": label_mapping,
        "fps": FPS,
        "feature_window": 15,
        "excluded_bodyparts": sorted(excluded_bodyparts),
        "extended_features": use_extended,
        "feature_normalization": {
            "applied": True,
            "method": "global_robust_iqr",
            "clip_value": float(args.normalization_clip),
            "stats_file": "normalization_stats.json",
            "notes": (
                "Global robust normalization (median/IQR) computed over "
                "all training frames (including augmented). Stats saved "
                "in normalization_stats.json and applied at inference."
            ),
        },
        "coordinate_preprocessing": {
            "enabled": coord_preprocess,
            "centroid_centering": coord_preprocess,
            "body_length_normalization": coord_preprocess,
            "ref_bodyparts": list(ref_pair) if ref_pair else list(DEFAULT_REF_PAIR),
            "ref_dist_px_summary": ref_dist_summary,
            "notes": (
                "Per-frame centroid subtracted from all coordinates, then "
                "all coordinates divided by median nose-to-S1 distance per "
                "recording.  Eliminates camera-position and pixel-scale "
                "differences between datasets."
            ),
        },
        "augmentation": {
            "enabled": n_aug > 0,
            "n_augmentations": n_aug,
            "rotation_range_deg": args.rotation_range,
            "translation_range_frac": args.translation_range,
            "scale_range": args.scale_range,
        },
        "feature_store": {
            "enabled": str(args.feature_store_policy) != "off",
            "policy": str(args.feature_store_policy),
            "store_dir": str(feature_store_dir),
            "cache_hits": int(feature_store_stats["hits"]),
            "writes": int(feature_store_stats["writes"]),
            "io_seconds": float(feature_store_stats["io_s"]),
            "compute_seconds": float(feature_store_stats["compute_s"]),
        },
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(repo_root),
        "training_sources": {
            "dlc_dir": str(args.dlc_dir.resolve()),
            "results_clusters_pkl": str(args.results_clusters_pkl.resolve()),
            "index_csv": str(args.index_csv.resolve()),
            "n_frames": int(len(df)),
            "n_features": int(len(x_df.columns)),
            "n_recordings_original": int(df["recording"].astype(str).apply(
                lambda r: r.split("_aug")[0]).nunique()),
            "n_recordings_total": int(df["recording"].nunique()),
            "recordings_used": sorted(df["recording"].astype(str).unique().tolist()),
            "class_counts_used": {str(k): int(v) for k, v in class_counts.items()},
            "subset_controls": {
                "quick": bool(args.quick),
                "max_recordings": (int(args.max_recordings)
                                   if args.max_recordings is not None else None),
                "max_samples_per_class": (int(args.max_samples_per_class)
                                          if args.max_samples_per_class is not None else None),
                "excluded_bodyparts": sorted(excluded_bodyparts),
            },
        },
        "xgb_params": params,
        "cv_folds": int(folds),
        "compute_backend": {
            "device": str(params.get("device", "cpu")),
            "tree_method": str(params.get("tree_method", "")),
            "xgboost_version": xgb.__version__,
        },
    }

    paths = save_model_artifacts(
        model=model, model_dir=model_dir, metadata=metadata,
        cv_metrics=metrics, save_pickle=True,
    )

    # Save normalization stats alongside the model
    norm_path = model_dir / "normalization_stats.json"
    save_normalization_stats(norm_stats, norm_path)
    paths["normalization_stats"] = norm_path

    eval_paths = _save_eval_artifacts(model_dir, metrics, class_names)
    paths.update(eval_paths)

    print("\nSaved artifacts:")
    for key, path in paths.items():
        print(f"  - {key}: {path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
