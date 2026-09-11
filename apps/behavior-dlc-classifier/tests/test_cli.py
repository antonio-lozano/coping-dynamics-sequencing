from freezing_dlc.cli import build_parser


def test_build_parser_supports_run_from_raw():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-from-raw",
            "--root",
            "X:/root",
            "--model",
            "X:/model.sav",
            "--dlc-config",
            "X:/config.yaml",
        ]
    )
    assert args.cmd == "run-from-raw"
    assert args.dlc_config.endswith("config.yaml")


def test_build_parser_supports_run_auto_defaults():
    parser = build_parser()
    args = parser.parse_args(["run-auto"])
    assert args.cmd == "run-auto"
    assert args.project_root == "."
    assert args.videotype == ".mp4"


def test_build_parser_supports_output_root_on_run():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--root",
            "X:/root",
            "--model",
            "X:/model.sav",
            "--output-root",
            "X:/output",
        ]
    )
    assert args.cmd == "run"
    assert args.output_root == "X:/output"


def test_build_parser_supports_train_model():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-model",
            "--dlc-dir",
            "X:/dlc",
            "--labels-dir",
            "X:/labels",
            "--out-model",
            "X:/models/freezing_model.sav",
        ]
    )
    assert args.cmd == "train-model"
    assert args.n_estimators == 600


def test_build_parser_supports_train_model_feature_mode():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-model",
            "--dlc-dir",
            "X:/dlc",
            "--labels-dir",
            "X:/labels",
            "--out-model",
            "X:/models/freezing_model.sav",
            "--feature-mode",
            "legacy4k",
        ]
    )
    assert args.feature_mode == "legacy4k"


def test_build_parser_supports_stabilize_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-model",
            "--dlc-dir",
            "X:/dlc",
            "--labels-dir",
            "X:/labels",
            "--out-model",
            "X:/models/freezing_model.sav",
            "--stabilize",
            "--step-estimators",
            "50",
            "--stability-patience",
            "3",
            "--stability-min-delta",
            "0.001",
            "--min-estimators-for-stability",
            "200",
        ]
    )
    assert args.stabilize is True
    assert args.step_estimators == 50
    assert args.stability_patience == 3
    assert args.stability_min_delta == 0.001
    assert args.min_estimators_for_stability == 200


def test_build_parser_supports_max_train_frames():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-model",
            "--dlc-dir",
            "X:/dlc",
            "--labels-dir",
            "X:/labels",
            "--out-model",
            "X:/models/freezing_model.sav",
            "--max-train-frames",
            "250000",
        ]
    )
    assert args.max_train_frames == 250000


def test_build_parser_supports_feature_cache_flags_train():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-model",
            "--dlc-dir",
            "X:/dlc",
            "--labels-dir",
            "X:/labels",
            "--out-model",
            "X:/models/freezing_model.sav",
            "--feature-cache-dir",
            "X:/cache",
            "--force-rebuild-cache",
        ]
    )
    assert args.feature_cache_dir == "X:/cache"
    assert args.force_rebuild_cache is True


def test_build_parser_supports_extra_dataset_train():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-model",
            "--dlc-dir",
            "X:/dlc",
            "--labels-dir",
            "X:/labels",
            "--out-model",
            "X:/models/freezing_model.sav",
            "--extra-dlc-dir",
            "Y:/dlc_extra",
            "--extra-labels-dir",
            "Y:/labels_extra",
        ]
    )
    assert args.extra_dlc_dir == "Y:/dlc_extra"
    assert args.extra_labels_dir == "Y:/labels_extra"


def test_build_parser_supports_n_jobs_train():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-model",
            "--dlc-dir",
            "X:/dlc",
            "--labels-dir",
            "X:/labels",
            "--out-model",
            "X:/models/freezing_model.sav",
            "--n-jobs",
            "3",
        ]
    )
    assert args.n_jobs == 3


def test_build_parser_supports_label_lag_train():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-model",
            "--dlc-dir",
            "X:/dlc",
            "--labels-dir",
            "X:/labels",
            "--out-model",
            "X:/models/freezing_model.sav",
            "--label-lag-frames",
            "2",
        ]
    )
    assert args.label_lag_frames == 2


def test_build_parser_supports_feature_cache_flags_eval():
    parser = build_parser()
    args = parser.parse_args(
        [
            "eval-model",
            "--model",
            "X:/models/freezing_model.sav",
            "--dlc-dir",
            "X:/dlc",
            "--labels-dir",
            "X:/labels",
            "--feature-cache-dir",
            "X:/cache",
        ]
    )
    assert args.feature_cache_dir == "X:/cache"


def test_build_parser_supports_eval_model():
    parser = build_parser()
    args = parser.parse_args(
        [
            "eval-model",
            "--model",
            "X:/models/freezing_model.sav",
            "--dlc-dir",
            "X:/dlc",
            "--labels-dir",
            "X:/labels",
            "--threshold",
            "0.88",
        ]
    )
    assert args.cmd == "eval-model"
    assert args.threshold == 0.88


def test_build_parser_supports_label_lag_eval():
    parser = build_parser()
    args = parser.parse_args(
        [
            "eval-model",
            "--model",
            "X:/models/freezing_model.sav",
            "--dlc-dir",
            "X:/dlc",
            "--labels-dir",
            "X:/labels",
            "--label-lag-frames",
            "3",
        ]
    )
    assert args.label_lag_frames == 3


def test_build_parser_supports_label_video():
    parser = build_parser()
    args = parser.parse_args(
        [
            "label-video",
            "--video",
            "X:/videos/sample.mp4",
            "--out-csv",
            "X:/labels/sample_labels.csv",
            "--speed",
            "1.25",
            "--start-paused",
        ]
    )
    assert args.cmd == "label-video"
    assert args.video == "X:/videos/sample.mp4"
    assert args.out_csv == "X:/labels/sample_labels.csv"
    assert args.speed == 1.25
    assert args.start_paused is True
