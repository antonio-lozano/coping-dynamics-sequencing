#!/usr/bin/env python
"""Train and persist an XGBoost behavior classifier from DLC + MoSeq labels."""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb

# Ensure repo root is importable when script is run directly.
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
    load_all_dlc_and_compute_features,
    robust_normalize_features,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dlc-dir", type=Path, default=DLC_DIR, help="Directory with DLC CSV files.")
    parser.add_argument(
        "--results-clusters-pkl",
        type=Path,
        default=RESULTS_CLUSTERS_PKL,
        help="Path to new_results_clusters.pkl containing behavior labels.",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=INDEX_CSV,
        help="Path to metadata/index.csv with group assignments.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d_baseline_v1"),
        help="Model version folder name.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MODELS_DIR,
        help="Base output directory for models.",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Limit number of DLC recordings loaded (deterministic: alphabetical order).",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Optional stratified cap per class after feature extraction.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Override XGBoost n_estimators.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Override XGBoost max_depth.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override XGBoost learning_rate.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Override number of cross-validation folds.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="XGBoost device backend (e.g., cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--tree-method",
        type=str,
        default="hist",
        help="XGBoost tree method (recommended for GPU: hist).",
    )
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="Apply robust per-recording normalization (median/IQR) to all numeric features.",
    )
    parser.add_argument(
        "--normalization-clip",
        type=float,
        default=4.0,
        help="Clip value for normalized features (default: 4.0).",
    )
    parser.add_argument(
        "--exclude-bodyparts",
        type=str,
        default="",
        help="Comma-separated bodyparts to exclude from feature computation (e.g., tail).",
    )
    parser.add_argument("--quick", action="store_true", help="Use tiny model/data settings for smoke tests.")
    return parser.parse_args()


def _parse_bodyparts(raw: str) -> set[str]:
    if not raw:
        return set()
    return {tok.strip().lower() for tok in raw.split(",") if tok.strip()}


def _save_eval_artifacts(model_dir: Path, metrics: dict, class_names: list[str]) -> dict[str, Path]:
    """Save confusion matrices and per-class report for easy inspection."""
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
    sns.heatmap(
        cm_counts_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.3,
        linecolor="#EEEEEE",
        ax=axes[0],
    )
    axes[0].set_title("OOF Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_norm_df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        linewidths=0.3,
        linecolor="#EEEEEE",
        vmin=0,
        vmax=1,
        ax=axes[1],
    )
    axes[1].set_title("OOF Confusion Matrix (Row-normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.tight_layout()

    cm_png = model_dir / "cv_confusion_matrix_oof.png"
    cm_pdf = model_dir / "cv_confusion_matrix_oof.pdf"
    fig.savefig(cm_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(cm_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved["cv_confusion_matrix_oof_png"] = cm_png
    saved["cv_confusion_matrix_oof_pdf"] = cm_pdf
    return saved


def main() -> int:
    args = parse_args()
    print("=" * 72)
    print("Training XGBoost behavior model")
    print("=" * 72)

    if not args.dlc_dir.exists():
        raise FileNotFoundError(f"DLC directory not found: {args.dlc_dir}")
    if not args.results_clusters_pkl.exists():
        raise FileNotFoundError(f"Results pickle not found: {args.results_clusters_pkl}")
    if not args.index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {args.index_csv}")

    excluded_bodyparts = _parse_bodyparts(args.exclude_bodyparts)
    if excluded_bodyparts:
        print(f"Excluding bodyparts from features: {sorted(excluded_bodyparts)}")

    df = load_all_dlc_and_compute_features(
        dlc_dir=args.dlc_dir,
        results_pkl=args.results_clusters_pkl,
        index_csv=args.index_csv,
        fps=FPS,
        label_col="behavior_cluster",
        max_csv_files=args.max_recordings,
        verbose=True,
        excluded_bodyparts=excluded_bodyparts,
    )
    if df is None or df.empty:
        raise RuntimeError("No labeled features were generated; cannot train model.")

    if args.quick:
        print("QUICK mode enabled: stratified subsample (max 250 frames/class).")
        df = stratified_subsample_by_label(
            df=df,
            label_col="behavior_cluster",
            max_samples_per_class=250,
            random_state=42,
        )
        if df.empty:
            raise RuntimeError("No samples remained after QUICK subsampling.")
        print(f"Using {len(df):,} frames for training.")
    elif args.max_samples_per_class is not None:
        print(f"Applying stratified cap: max {args.max_samples_per_class} frames/class.")
        df = stratified_subsample_by_label(
            df=df,
            label_col="behavior_cluster",
            max_samples_per_class=args.max_samples_per_class,
            random_state=42,
        )
        if df.empty:
            raise RuntimeError("No samples remained after stratified subsampling.")
        print(f"Using {len(df):,} frames for training after class cap.")

    if args.normalize_features:
        print(
            "Applying per-recording robust feature normalization "
            f"(median/IQR, clip={args.normalization_clip})..."
        )
        excluded = {"behavior_cluster", "recording", "group"}
        feature_cols = [c for c in df.columns if c not in excluded]
        normalized_parts = []
        for recording, rec_df in df.groupby("recording", sort=False):
            rec_feat_df = rec_df[feature_cols]
            rec_norm_df, _ = robust_normalize_features(
                rec_feat_df,
                feature_cols=feature_cols,
                clip_value=float(args.normalization_clip),
            )
            out_rec = rec_df.copy()
            out_rec.loc[:, feature_cols] = rec_norm_df[feature_cols].to_numpy()
            normalized_parts.append(out_rec)
        df = pd.concat(normalized_parts, ignore_index=True)

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
    print("Training configuration:")
    print(f"  device={params['device']}, tree_method={params['tree_method']}")
    print(
        f"  n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, "
        f"learning_rate={params['learning_rate']}, cv_folds={folds}"
    )
    print(f"Running {folds}-fold cross-validation...")
    metrics = compute_cv_metrics(x, y, classes, params, folds=folds, random_state=42)
    print(f"Mean CV accuracy: {metrics['overall_accuracy_mean']:.3f}")
    print(f"OOF accuracy: {metrics['overall_accuracy_oof']:.3f}")
    print(f"OOF macro F1: {metrics['macro_f1_oof']:.3f}")

    print("Training final model on all frames...")
    model = train_xgb_classifier(x=x, y=y, xgb_params=params)

    model_dir = args.out_dir / args.model_version
    metadata = {
        "model_version": args.model_version,
        "feature_names": x_df.columns.tolist(),
        "class_names": class_names,
        "label_mapping": label_mapping,
        "fps": FPS,
        "feature_window": 15,
        "excluded_bodyparts": sorted(excluded_bodyparts),
        "feature_normalization": {
            "applied": bool(args.normalize_features),
            "method": "robust_iqr_per_recording" if args.normalize_features else "none",
            "clip_value": float(args.normalization_clip) if args.normalize_features else None,
            "notes": (
                "Per-recording robust normalization (median/IQR) applied to all numeric features."
                if args.normalize_features
                else "Model trained on raw kinematic features without per-recording normalization."
            ),
        },
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(repo_root),
        "training_sources": {
            "dlc_dir": str(args.dlc_dir.resolve()),
            "results_clusters_pkl": str(args.results_clusters_pkl.resolve()),
            "index_csv": str(args.index_csv.resolve()),
            "n_frames": int(len(df)),
            "n_features": int(len(x_df.columns)),
            "n_recordings": int(df["recording"].nunique()),
            "recordings_used": sorted(df["recording"].astype(str).unique().tolist()),
            "class_counts_used": {str(k): int(v) for k, v in class_counts.items()},
            "subset_controls": {
                "quick": bool(args.quick),
                "max_recordings": int(args.max_recordings) if args.max_recordings is not None else None,
                "max_samples_per_class": (
                    int(args.max_samples_per_class) if args.max_samples_per_class is not None else None
                ),
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
        model=model,
        model_dir=model_dir,
        metadata=metadata,
        cv_metrics=metrics,
        save_pickle=True,
    )
    eval_paths = _save_eval_artifacts(model_dir=model_dir, metrics=metrics, class_names=class_names)
    paths.update(eval_paths)

    print("\nSaved artifacts:")
    for key, path in paths.items():
        print(f"  - {key}: {path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
