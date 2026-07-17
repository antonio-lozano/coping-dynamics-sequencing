"""Command-line interface for the Fig. 7 behavior classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import FIGURE_DATA_DIR, MOSEQ_DF, RESULTS_DIR


DEFAULT_MODEL = RESULTS_DIR / "behavior_classifier" / "fig7_behavior_xgb.joblib"
DEFAULT_METRICS = FIGURE_DATA_DIR / "fig7_behavior_classifier_cv_metrics.csv"
DEFAULT_CONFUSION = FIGURE_DATA_DIR / "fig7_behavior_classifier_confusion_matrix.csv"


def _model_kwargs(args: argparse.Namespace) -> dict[str, int | float]:
    return {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
    }


def train_command(args: argparse.Namespace) -> None:
    from .pipeline import cross_validate, load_moseq_table, save_bundle, train_behavior_classifier

    df = load_moseq_table(args.input)

    if not args.skip_cv:
        metrics, confusion = cross_validate(
            df,
            include_unassigned=not args.drop_unassigned,
            balanced_weights=not args.no_balanced_weights,
            n_splits=args.cv_splits,
            random_state=args.random_state,
            max_frames=args.max_frames,
            model_kwargs=_model_kwargs(args),
        )
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.confusion.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.metrics, index=False)
        confusion.to_csv(args.confusion, index=False)
        print(f"Wrote CV metrics: {args.metrics}")
        print(f"Wrote confusion matrix: {args.confusion}")

    bundle = train_behavior_classifier(
        df,
        include_unassigned=not args.drop_unassigned,
        balanced_weights=not args.no_balanced_weights,
        random_state=args.random_state,
        max_frames=args.max_frames,
        model_kwargs=_model_kwargs(args),
    )
    save_bundle(bundle, args.model)
    print(f"Wrote classifier: {args.model}")
    print(f"Classes: {', '.join(bundle.metadata['classes'])}")
    print(f"Training frames: {bundle.metadata['n_frames']}")


def predict_command(args: argparse.Namespace) -> None:
    from .pipeline import load_bundle, load_moseq_table, predict_behaviors

    bundle = load_bundle(args.model)
    df = load_moseq_table(args.input)
    if args.max_frames is not None and args.max_frames > 0:
        df = df.head(args.max_frames).copy()
    pred = predict_behaviors(df, bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.output, index=False)
    print(f"Wrote predictions: {args.output}")


def run_from_dlc_command(args: argparse.Namespace) -> None:
    from .raw_pipeline import predict_from_dlc_file

    features_path, pred_path = predict_from_dlc_file(
        dlc_file=args.dlc_file,
        model_path=args.model,
        output_dir=args.output_dir,
        fps=args.fps,
        name=args.name,
    )
    print(f"Wrote features: {features_path}")
    print(f"Wrote predictions: {pred_path}")


def run_from_raw_command(args: argparse.Namespace) -> None:
    from .raw_pipeline import run_from_raw_video

    summary = run_from_raw_video(
        video=args.video,
        dlc_config=args.dlc_config,
        model_path=args.model,
        output_root=args.output_root,
        fps=args.fps,
        videotype=args.videotype,
        reuse_existing=not args.force_dlc,
    )
    print(summary.to_string(index=False))


def train_from_dlc_command(args: argparse.Namespace) -> None:
    from .dlc_training import train_from_dlc

    outputs = train_from_dlc(
        dlc_dir=args.dlc_dir,
        labels_dir=args.labels_dir,
        out_model=args.model,
        metrics_path=args.metrics,
        confusion_path=args.confusion,
        training_table_path=args.training_table,
        fps=args.fps,
        max_frames=args.max_frames,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        skip_cv=args.skip_cv,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or apply the Fig. 7 behavior-cluster classifier."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train the behavior classifier.")
    train.add_argument("--input", type=Path, default=MOSEQ_DF)
    train.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    train.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    train.add_argument("--confusion", type=Path, default=DEFAULT_CONFUSION)
    train.add_argument("--cv-splits", type=int, default=3)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--max-frames", type=int, default=None)
    train.add_argument("--n-estimators", type=int, default=150)
    train.add_argument("--max-depth", type=int, default=6)
    train.add_argument("--learning-rate", type=float, default=0.2)
    train.add_argument("--drop-unassigned", action="store_true")
    train.add_argument("--no-balanced-weights", action="store_true")
    train.add_argument("--skip-cv", action="store_true")
    train.set_defaults(func=train_command)

    train_dlc = subparsers.add_parser(
        "train-from-dlc",
        help="Train from DLC filtered tracks plus frame-level syllable/behavior labels.",
    )
    train_dlc.add_argument("--dlc-dir", type=Path, required=True)
    train_dlc.add_argument("--labels-dir", type=Path, required=True)
    train_dlc.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    train_dlc.add_argument(
        "--metrics",
        type=Path,
        default=FIGURE_DATA_DIR / "fig7_behavior_classifier_dlc_cv_metrics.csv",
    )
    train_dlc.add_argument(
        "--confusion",
        type=Path,
        default=FIGURE_DATA_DIR / "fig7_behavior_classifier_dlc_confusion_matrix.csv",
    )
    train_dlc.add_argument(
        "--training-table",
        type=Path,
        default=FIGURE_DATA_DIR / "fig7_behavior_classifier_dlc_training_table.csv",
    )
    train_dlc.add_argument("--fps", type=float, default=25.0)
    train_dlc.add_argument("--max-frames", type=int, default=None)
    train_dlc.add_argument("--n-estimators", type=int, default=150)
    train_dlc.add_argument("--max-depth", type=int, default=6)
    train_dlc.add_argument("--learning-rate", type=float, default=0.2)
    train_dlc.add_argument("--random-state", type=int, default=42)
    train_dlc.add_argument("--skip-cv", action="store_true")
    train_dlc.set_defaults(func=train_from_dlc_command)

    predict = subparsers.add_parser("predict", help="Predict behaviors for a MoSeq table.")
    predict.add_argument("--input", type=Path, default=MOSEQ_DF)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict.add_argument("--max-frames", type=int, default=None)
    predict.add_argument(
        "--output",
        type=Path,
        default=FIGURE_DATA_DIR / "fig7_behavior_classifier_predictions.csv",
    )
    predict.set_defaults(func=predict_command)

    raw = subparsers.add_parser(
        "run-from-raw",
        help="Run DLC on a raw video, then predict Fig. 7 behaviors.",
    )
    raw.add_argument("--video", type=Path, required=True)
    raw.add_argument("--dlc-config", type=Path, required=True)
    raw.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    raw.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_DIR / "behavior_classifier" / "raw_runs",
    )
    raw.add_argument("--fps", type=float, default=25.0)
    raw.add_argument("--videotype", default=None)
    raw.add_argument("--force-dlc", action="store_true")
    raw.set_defaults(func=run_from_raw_command)

    dlc = subparsers.add_parser(
        "run-from-dlc",
        help="Predict Fig. 7 behaviors from an existing DLC filtered CSV/H5.",
    )
    dlc.add_argument("--dlc-file", type=Path, required=True)
    dlc.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    dlc.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "behavior_classifier" / "behavior_predictions",
    )
    dlc.add_argument("--fps", type=float, default=25.0)
    dlc.add_argument("--name", default=None)
    dlc.set_defaults(func=run_from_dlc_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
