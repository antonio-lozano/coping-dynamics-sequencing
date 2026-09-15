import argparse
import json
from pathlib import Path
from typing import List, Optional

from .config import PipelineConfig, default_model_path
from .dlc_stage import DEFAULT_DLC_CONFIG_PATH, run_deeplabcut_to_filtered
from .features import FEATURE_MODES
from .label_video import annotate_video_labels
from .pipeline import run_pipeline
from .train import (
    DEFAULT_THRESHOLD_CRITERION,
    THRESHOLD_CRITERIA,
    cross_validate_model,
    evaluate_model,
    train_model,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="freezing-dlc")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run freezing classifier")
    run.add_argument("--root", required=True)
    run.add_argument("--model", required=True)
    run.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Freezing decision threshold. Default: the value stored in the model file.",
    )
    run.add_argument("--fps", type=int, default=25)
    run.add_argument("--bin-sec", type=int, default=30)
    run.add_argument("--annotate", action="store_true")
    run.add_argument("--output-root")

    full = sub.add_parser(
        "run-from-raw",
        help="Run DeepLabCut from raw videos, then run freezing classifier",
    )
    full.add_argument("--root", required=True)
    full.add_argument("--model", required=True)
    full.add_argument("--dlc-config", required=True)
    full.add_argument("--videotype", default=".mp4")
    full.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Freezing decision threshold. Default: the value stored in the model file.",
    )
    full.add_argument("--fps", type=int, default=25)
    full.add_argument("--bin-sec", type=int, default=30)
    full.add_argument("--annotate", action="store_true")
    full.add_argument("--output-root")

    auto = sub.add_parser(
        "run-auto",
        help="Auto mode: uses project defaults so you only need to copy videos",
    )
    auto.add_argument("--project-root", default=".")
    auto.add_argument("--videotype", default=".mp4")
    auto.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Freezing decision threshold. Default: the value stored in the model file.",
    )
    auto.add_argument("--fps", type=int, default=25)
    auto.add_argument("--bin-sec", type=int, default=30)
    auto.add_argument("--annotate", action="store_true")

    train = sub.add_parser("train-model", help="Train a freezing model from DLC and label CSVs")
    train.add_argument("--dlc-dir", required=True)
    train.add_argument("--labels-dir", required=True)
    train.add_argument("--out-model", required=True)
    train.add_argument("--feature-mode", choices=list(FEATURE_MODES), default="stillness")
    train.add_argument("--n-estimators", type=int, default=600)
    train.add_argument("--step-estimators", type=int, default=100)
    train.add_argument(
        "--threshold-criterion",
        choices=list(THRESHOLD_CRITERIA),
        default=DEFAULT_THRESHOLD_CRITERION,
        help="What the decision threshold maximises. F1 climbs with the freezing rate, "
        "so choosing by it pushes the cutoff towards calling every frame freezing.",
    )
    train.add_argument("--stabilize", action="store_true")
    train.add_argument("--stability-patience", type=int, default=4)
    train.add_argument("--stability-min-delta", type=float, default=0.0005)
    train.add_argument("--min-estimators-for-stability", type=int, default=300)
    train.add_argument(
        "--min-samples-leaf",
        type=int,
        default=20,
        help="Smallest number of frames a tree leaf may cover. 1 lets the forest memorise "
        "individual frames, which on strongly correlated video makes the file large and "
        "the model worse on new animals. Default: 20.",
    )
    train.add_argument(
        "--no-refit-on-all",
        dest="refit_on_all",
        action="store_false",
        help="Save the model fitted on the training split only, instead of refitting it on "
        "every labelled frame once the held-out score has been measured.",
    )
    train.add_argument("--max-train-frames", type=int)
    train.add_argument("--split-mode", choices=["random", "group"], default="group")
    train.add_argument("--feature-cache-dir")
    train.add_argument("--force-rebuild-cache", action="store_true")
    train.add_argument("--extra-dlc-dir")
    train.add_argument("--extra-labels-dir")
    train.add_argument("--n-jobs", type=int, default=-1)
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--label-lag-frames", type=int, default=0)

    ev = sub.add_parser("eval-model", help="Evaluate a trained model on labeled DLC data")
    ev.add_argument("--model", required=True)
    ev.add_argument("--dlc-dir", required=True)
    ev.add_argument("--labels-dir", required=True)
    ev.add_argument("--threshold", type=float)
    ev.add_argument("--label-lag-frames", type=int)
    ev.add_argument("--max-eval-frames", type=int)
    ev.add_argument("--eval-chunk-size", type=int, default=100000)
    ev.add_argument("--feature-cache-dir")
    ev.add_argument("--force-rebuild-cache", action="store_true")

    cv = sub.add_parser(
        "cross-validate",
        help="Leave-one-video-out score: the number to quote when you have few videos",
    )
    cv.add_argument("--dlc-dir", required=True)
    cv.add_argument("--labels-dir", required=True)
    cv.add_argument("--feature-mode", default="stillness", choices=list(FEATURE_MODES))
    cv.add_argument("--n-estimators", type=int, default=200)
    cv.add_argument("--label-lag-frames", type=int, default=0)
    cv.add_argument(
        "--threshold",
        type=float,
        help="Fix the threshold instead of tuning it inside each fold. Needed with only two videos.",
    )
    cv.add_argument(
        "--threshold-criterion",
        choices=list(THRESHOLD_CRITERIA),
        default=DEFAULT_THRESHOLD_CRITERION,
        help="What the decision threshold maximises inside each fold.",
    )
    cv.add_argument(
        "--min-samples-leaf",
        type=int,
        default=20,
        help="Match whatever train-model used, so this scores the model you ship. Default: 20.",
    )
    cv.add_argument("--max-train-frames", type=int)
    cv.add_argument("--feature-cache-dir")
    cv.add_argument("--force-rebuild-cache", action="store_true")
    cv.add_argument("--n-jobs", type=int, default=-1)
    cv.add_argument("--random-state", type=int, default=42)

    label = sub.add_parser("label-video", help="Interactively annotate freezing on a video")
    label.add_argument("--video", required=True)
    label.add_argument("--out-csv")
    label.add_argument("--fps", type=float)
    label.add_argument("--speed", type=float, default=1.0)
    label.add_argument("--start-paused", action="store_true")
    label.add_argument("--existing-labels")
    label.add_argument("--result-json", help="Write annotation completion metadata for a GUI caller")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        cfg = PipelineConfig(
            root=Path(args.root),
            model_path=Path(args.model),
            output_root=Path(args.output_root) if args.output_root else None,
            threshold=args.threshold,
            fps=args.fps,
            bin_sec=args.bin_sec,
        )
        run_pipeline(cfg, annotate=args.annotate)
        return 0
    if args.cmd == "run-from-raw":
        cfg = PipelineConfig(
            root=Path(args.root),
            model_path=Path(args.model),
            output_root=Path(args.output_root) if args.output_root else None,
            threshold=args.threshold,
            fps=args.fps,
            bin_sec=args.bin_sec,
        )
        run_deeplabcut_to_filtered(
            config_path=Path(args.dlc_config),
            videos_dir=cfg.original_videos_dir,
            filtered_dir=cfg.filtered_dir,
            videotype=args.videotype,
        )
        run_pipeline(cfg, annotate=args.annotate)
        return 0
    if args.cmd == "run-auto":
        project_root = Path(args.project_root).resolve()
        data_root = project_root / "data" / "raw"
        model_path = default_model_path(project_root)
        dlc_config = DEFAULT_DLC_CONFIG_PATH
        output_root = project_root / "output"

        cfg = PipelineConfig(
            root=data_root,
            model_path=model_path,
            output_root=output_root,
            threshold=args.threshold,
            fps=args.fps,
            bin_sec=args.bin_sec,
        )
        run_deeplabcut_to_filtered(
            config_path=dlc_config,
            videos_dir=cfg.original_videos_dir,
            filtered_dir=cfg.filtered_dir,
            videotype=args.videotype,
        )
        run_pipeline(cfg, annotate=args.annotate)
        return 0
    if args.cmd == "train-model":
        payload = train_model(
            dlc_dir=Path(args.dlc_dir),
            labels_dir=Path(args.labels_dir),
            out_model=Path(args.out_model),
            feature_mode=args.feature_mode,
            n_estimators=args.n_estimators,
            step_estimators=args.step_estimators,
            threshold_criterion=args.threshold_criterion,
            min_samples_leaf=args.min_samples_leaf,
            refit_on_all=args.refit_on_all,
            stabilize=args.stabilize,
            stability_patience=args.stability_patience,
            stability_min_delta=args.stability_min_delta,
            min_estimators_for_stability=args.min_estimators_for_stability,
            max_train_frames=args.max_train_frames,
            split_mode=args.split_mode,
            feature_cache_dir=Path(args.feature_cache_dir) if args.feature_cache_dir else None,
            force_rebuild_cache=args.force_rebuild_cache,
            extra_dlc_dir=Path(args.extra_dlc_dir) if args.extra_dlc_dir else None,
            extra_labels_dir=Path(args.extra_labels_dir) if args.extra_labels_dir else None,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            test_size=args.test_size,
            label_lag_frames=args.label_lag_frames,
        )
        print(json.dumps({"saved_model": str(Path(args.out_model)), "metrics": payload.get("metrics", {})}, indent=2))
        return 0
    if args.cmd == "eval-model":
        metrics = evaluate_model(
            model_path=Path(args.model),
            dlc_dir=Path(args.dlc_dir),
            labels_dir=Path(args.labels_dir),
            threshold=args.threshold,
            label_lag_frames=args.label_lag_frames,
            max_eval_frames=args.max_eval_frames,
            eval_chunk_size=args.eval_chunk_size,
            feature_cache_dir=Path(args.feature_cache_dir) if args.feature_cache_dir else None,
            force_rebuild_cache=args.force_rebuild_cache,
        )
        print(json.dumps(metrics, indent=2))
        return 0
    if args.cmd == "cross-validate":
        summary = cross_validate_model(
            dlc_dir=Path(args.dlc_dir),
            labels_dir=Path(args.labels_dir),
            feature_mode=args.feature_mode,
            n_estimators=args.n_estimators,
            label_lag_frames=args.label_lag_frames,
            threshold=args.threshold,
            threshold_criterion=args.threshold_criterion,
            min_samples_leaf=args.min_samples_leaf,
            max_train_frames=args.max_train_frames,
            feature_cache_dir=Path(args.feature_cache_dir) if args.feature_cache_dir else None,
            force_rebuild_cache=args.force_rebuild_cache,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.cmd == "label-video":
        result = annotate_video_labels(
            video_path=Path(args.video),
            out_csv=Path(args.out_csv) if args.out_csv else None,
            fps_override=args.fps,
            start_paused=bool(args.start_paused),
            speed=float(args.speed),
            existing_labels=Path(args.existing_labels) if args.existing_labels else None,
        )
        if args.result_json:
            Path(args.result_json).write_text(json.dumps(result), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
