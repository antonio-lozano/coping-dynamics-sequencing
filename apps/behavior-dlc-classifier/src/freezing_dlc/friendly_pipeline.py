from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from glob import escape as glob_escape
from pathlib import Path
from shutil import copy2, which
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from .annotate import annotate_video
from .behavior import (
    BEHAVIOR_COLORS,
    BEHAVIOR_DISPLAY_ORDER,
    BEHAVIOR_LABEL_ENCODER_PATH,
    BEHAVIOR_MODEL_PATH,
    BEHAVIOR_ORDER,
    UNASSIGNED,
    annotate_behavior_video,
    build_behavior_feature_set,
    load_behavior_model,
    predict_behaviors,
    refine_behavior_labels,
    save_behavior_predictions,
)
from .dlc_stage import find_source_video, list_videos, prepare_dlc_config
from .features import build_feature_set, dlc_filtered_csv_to_flat_df
from .model import load_model, resolve_threshold
from .predict import predict_freezing, save_frame_predictions

LogFn = Callable[[str], None]

#: Where the bundled DeepLabCut environment lives when it was made with
#: "uv venv .venv-dlc". Used whenever the environment field is left blank.
DEFAULT_DLC_VENV = Path(__file__).resolve().parents[2] / ".venv-dlc"

DLC_DERIVED_VIDEO_MARKERS = (
    "dlc_resnet",
    "dlc_mobilenet",
    "_labeled",
)


@dataclass(frozen=True)
class FriendlyPipelineSettings:
    project_dir: Path
    model_path: Path
    dlc_config_path: Path
    videotype: str = ".mp4"
    threshold: Optional[float] = None
    fps: float = 25.0
    bin_sec: int = 30
    box_side_cm: float = 17.0
    box_side_pixels: float = 170.0
    run_clahe: bool = True
    run_dlc: bool = True
    #: Where DeepLabCut lives. A folder made by "uv venv", a path to a Python
    #: interpreter, or a bare conda environment name. Blank means the bundled
    #: .venv-dlc beside the tool.
    dlc_env: str = ""
    make_tracked_videos: bool = True
    run_behavior_analysis: bool = True
    behavior_model_path: Path = BEHAVIOR_MODEL_PATH
    behavior_label_encoder_path: Path = BEHAVIOR_LABEL_ENCODER_PATH
    ethogram_threshold: float = 0.70
    ethogram_gap_fill_sec: float = 1.0
    ethogram_min_bout_sec: float = 1.0


def run_friendly_pipeline(settings: FriendlyPipelineSettings, log: LogFn = print) -> Path:
    """Run the full no-code freezing analysis workflow and return the final package path."""
    # Validate before coercion: int(30.5) silently changes the requested analysis.
    _validate_settings(settings)
    settings = FriendlyPipelineSettings(
        project_dir=Path(settings.project_dir).resolve(),
        model_path=Path(settings.model_path).resolve(),
        dlc_config_path=Path(settings.dlc_config_path).resolve(),
        videotype=_normalize_ext(settings.videotype),
        threshold=None if settings.threshold is None else float(settings.threshold),
        fps=float(settings.fps),
        bin_sec=int(settings.bin_sec),
        box_side_cm=float(settings.box_side_cm),
        box_side_pixels=float(settings.box_side_pixels),
        run_clahe=bool(settings.run_clahe),
        run_dlc=bool(settings.run_dlc),
        dlc_env=str(settings.dlc_env).strip(),
        make_tracked_videos=bool(settings.make_tracked_videos),
        run_behavior_analysis=bool(settings.run_behavior_analysis),
        behavior_model_path=Path(settings.behavior_model_path).resolve(),
        behavior_label_encoder_path=Path(settings.behavior_label_encoder_path).resolve(),
        ethogram_threshold=float(settings.ethogram_threshold),
        ethogram_gap_fill_sec=float(settings.ethogram_gap_fill_sec),
        ethogram_min_bout_sec=float(settings.ethogram_min_bout_sec),
    )
    _validate_settings(settings)
    if settings.run_dlc:
        _check_deeplabcut_available(settings.dlc_env)

    package_dir = _next_analysis_package_dir(settings.project_dir)
    work_dir = package_dir / "_working"
    package = _create_package_dirs(package_dir)

    log(f"[setup] Final package: {package_dir}")
    raw_videos, source_videos = _prepare_original_videos(settings.project_dir, package["original"], settings.videotype, log)

    if settings.run_clahe:
        visible_clahe_videos = _preprocess_videos_clahe(raw_videos, package["clahe"], log)
        dlc_input_videos = _copy_videos_for_dlc(visible_clahe_videos, work_dir / "dlc_input_videos", log)
    else:
        dlc_input_videos = _copy_videos_for_dlc(raw_videos, work_dir / "dlc_input_videos", log)
        log("[clahe] Skipped; DLC will use original videos.")

    if settings.run_dlc:
        _run_dlc(
            settings.dlc_config_path,
            dlc_input_videos,
            settings.videotype,
            work_dir,
            settings.dlc_env,
            make_tracked_videos=settings.make_tracked_videos,
            log=log,
        )
    else:
        log("[dlc] Skipped; using existing DLC outputs next to selected videos.")

    # When DLC ran, its outputs sit beside the working copies it analysed; when
    # it was skipped, the only place existing outputs can be is beside the
    # user's own videos.
    collect_beside = dlc_input_videos if settings.run_dlc else source_videos
    _collect_dlc_outputs(collect_beside, package["tracked_unfiltered"], package["tracked_filtered"], package["dlc_unfiltered"], package["dlc_filtered"], log)
    predictions, effective_threshold = _run_freezing_predictions(
        package["dlc_filtered"],
        package["original"],
        package["annotations"],
        package["features"],
        package["predictions"],
        settings,
        log,
    )
    # Record the threshold that was actually used so the workbook, the ethogram and
    # the package summary all report a real number rather than "auto".
    settings = replace(settings, threshold=effective_threshold)
    _write_results_workbook(package["results"] / "freezing_FINAL_results.xlsx", predictions, settings, log)
    _write_ethogram(package["ethogram"] / "freezing_ethogram_vertical_chunks_FINAL.png", package["ethogram"] / "freezing_ethogram_vertical_chunks_FINAL.pdf", predictions, settings, log)
    if settings.run_behavior_analysis:
        behavior_predictions = _run_behavior_predictions(
            package["dlc_filtered"],
            package["original"],
            package["behavior_features"],
            package["behavior_predictions"],
            package["behavior_annotations"],
            settings,
            log,
            predictions,
        )
        _write_behavior_results(package["results"] / "behavior_FINAL_results.xlsx", behavior_predictions, settings, log)
        _write_behavior_ethograms(package["ethogram"], behavior_predictions, settings, log)
    _write_package_summary(package_dir, settings, log)

    log("[done] Analysis complete.")
    return package_dir


def _validate_settings(settings: FriendlyPipelineSettings) -> None:
    if not settings.project_dir.exists():
        raise FileNotFoundError(f"Project folder not found: {settings.project_dir}")
    if not settings.model_path.exists():
        raise FileNotFoundError(f"Freezing model not found: {settings.model_path}")
    if settings.run_dlc and not settings.dlc_config_path.exists():
        raise FileNotFoundError(f"DeepLabCut config not found: {settings.dlc_config_path}")
    if settings.run_behavior_analysis:
        if not settings.behavior_model_path.exists():
            raise FileNotFoundError(f"Behavior model not found: {settings.behavior_model_path}")
        if not settings.behavior_label_encoder_path.exists():
            raise FileNotFoundError(f"Behavior label encoder not found: {settings.behavior_label_encoder_path}")
    for name in ("fps", "bin_sec", "box_side_cm", "box_side_pixels", "ethogram_threshold", "ethogram_gap_fill_sec", "ethogram_min_bout_sec"):
        if not math.isfinite(float(getattr(settings, name))):
            raise ValueError(f"{name} must be a finite number.")
    if settings.threshold is not None and (not math.isfinite(float(settings.threshold)) or not 0 <= settings.threshold <= 1):
        raise ValueError("Freezing threshold must be between 0 and 1.")
    if not 0 <= settings.ethogram_threshold <= 1:
        raise ValueError("Ethogram threshold must be between 0 and 1.")
    if settings.ethogram_gap_fill_sec < 0 or settings.ethogram_min_bout_sec < 0:
        raise ValueError("Ethogram gap and bout durations must not be negative.")
    if not float(settings.bin_sec).is_integer():
        raise ValueError("Time bin must be a positive whole number of seconds.")
    if settings.fps <= 0:
        raise ValueError("FPS must be greater than 0.")
    if settings.bin_sec <= 0:
        raise ValueError("Time bin must be greater than 0 seconds.")
    if settings.box_side_cm <= 0 or settings.box_side_pixels <= 0:
        raise ValueError("Box side calibration must be positive.")


def _normalize_ext(videotype: str) -> str:
    ext = str(videotype).strip() or ".mp4"
    return ext if ext.startswith(".") else f".{ext}"


def _next_analysis_package_dir(project_dir: Path) -> Path:
    date_prefix = datetime.now().strftime("%Y%m%d")
    folder_name = _safe_folder_name(project_dir.name)
    base = project_dir / f"{date_prefix}_{folder_name}_analysis"
    if not base.exists():
        return base
    for idx in range(2, 1000):
        candidate = project_dir / f"{date_prefix}_{folder_name}_analysis_{idx}"
        if not candidate.exists():
            return candidate
    return project_dir / f"{date_prefix}_{folder_name}_analysis_{datetime.now().strftime('%H%M%S')}"


def _safe_folder_name(name: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_")
    return cleaned or "project"


def _create_package_dirs(package_dir: Path) -> dict[str, Path]:
    dirs = {
        "original": package_dir / "01_original_videos",
        "clahe": package_dir / "01_original_videos" / "CLAHE_preprocessed_videos",
        "tracked_unfiltered": package_dir / "02_tracked_videos" / "unfiltered",
        "tracked_filtered": package_dir / "02_tracked_videos" / "filtered",
        "dlc_unfiltered": package_dir / "03_DLC_analysis" / "unfiltered",
        "dlc_filtered": package_dir / "03_DLC_analysis" / "filtered",
        "features": package_dir / "04_features" / "frame_features_per_animal",
        "predictions": package_dir / "04_features" / "frame_predictions_per_animal",
        "behavior_features": package_dir / "04_features" / "behavior_features_per_animal",
        "behavior_predictions": package_dir / "04_features" / "behavior_predictions_per_animal",
        "annotations": package_dir / "05_annotated_videos" / "freezing",
        "behavior_annotations": package_dir / "05_annotated_videos" / "behaviors",
        "results": package_dir / "06_results",
        "ethogram": package_dir / "07_ethogram",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _prepare_original_videos(
    project_dir: Path, out_dir: Path, videotype: str, log: LogFn
) -> tuple[list[Path], list[Path]]:
    """Copy the raw recordings into the package; return (copies, their sources)."""
    source_dir = project_dir / "original_videos"
    if source_dir.exists():
        candidates = list_videos(source_dir, videotype)
    else:
        candidates = list_videos(project_dir, videotype)
    sources = [path for path in candidates if _looks_like_raw_video(path)]
    ignored = [path for path in candidates if path not in sources]
    if not sources:
        raise FileNotFoundError(
            f"No raw '{videotype}' videos found in {project_dir} or {source_dir}. "
            "DLC-labeled videos are ignored automatically."
        )

    identities: dict[tuple[str, int | None, int | None], Path] = {}
    for src in sources:
        identity = _parse_animal(src.stem)
        if identity in identities:
            raise ValueError(
                f"Videos '{identities[identity].name}' and '{src.name}' resolve to "
                f"the same recording identity ({identity[0]}). Analyze separate "
                "sessions in separate folders, or correct the trial/mouse identifiers "
                "if the filenames are wrong. No recordings have been combined."
            )
        identities[identity] = src

    copied = []
    for src in sources:
        dst = out_dir / src.name
        if src.resolve() != dst.resolve():
            copy2(src, dst)
        copied.append(dst)
    log(f"[videos] Copied {len(copied)} original videos.")
    if ignored:
        examples = ", ".join(path.name for path in ignored[:5])
        suffix = "..." if len(ignored) > 5 else ""
        log(f"[videos] Ignored {len(ignored)} DLC/labeled videos: {examples}{suffix}")
    return copied, sources


def _looks_like_raw_video(path: Path) -> bool:
    stem = path.stem.lower()
    return not any(marker in stem for marker in DLC_DERIVED_VIDEO_MARKERS)


def _check_deeplabcut_available(dlc_env: str) -> None:
    try:
        import deeplabcut  # noqa: F401  # type: ignore
        return
    except Exception as exc:  # pragma: no cover - depends on local DLC install
        direct_error = exc

    cmd = [*_dlc_python_command(dlc_env), "-c", "import deeplabcut"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "DeepLabCut could not be imported in this Python environment, nor in "
            f"the one at {_describe_dlc_env(dlc_env)}. Raw videos need it. "
            f"Create it with:\n\n{_DLC_SETUP_HINT}\n\n"
            f"Current environment error: {direct_error}. "
            f"DeepLabCut environment error: {(result.stderr or result.stdout).strip()}"
        ) from direct_error


def _preprocess_videos_clahe(videos: Iterable[Path], out_dir: Path, log: LogFn) -> list[Path]:
    import cv2

    videos = list(videos)
    out_dir.mkdir(parents=True, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = []
    for idx, video in enumerate(videos, start=1):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for CLAHE preprocessing: {video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dst = out_dir / video.name
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open CLAHE output for writing: {dst}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l_chan, a_chan, b_chan = cv2.split(lab)
                enhanced = cv2.merge((clahe.apply(l_chan), a_chan, b_chan))
                writer.write(cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR))
        finally:
            cap.release()
            writer.release()
        processed.append(dst)
        log(f"[clahe] {idx}/{len(videos)} {video.name}")
    return processed


def _copy_videos_for_dlc(videos: Iterable[Path], out_dir: Path, log: LogFn) -> list[Path]:
    videos = list(videos)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for video in videos:
        dst = out_dir / video.name
        if video.resolve() != dst.resolve():
            copy2(video, dst)
        copied.append(dst)
    log(f"[dlc] Prepared {len(copied)} clean video copies for DLC in hidden working folder.")
    return copied


def _run_dlc(
    config_path: Path,
    videos: list[Path],
    videotype: str,
    work_dir: Path,
    dlc_env: str,
    make_tracked_videos: bool,
    log: LogFn,
) -> None:
    config_path = prepare_dlc_config(config_path)

    try:
        import deeplabcut  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local DLC install
        log(f"[dlc] DeepLabCut is not in this Python environment ({exc}).")
        _run_dlc_in_env(config_path, videos, videotype, work_dir, dlc_env, make_tracked_videos, log)
        return

    video_strings = [str(v) for v in videos]
    cfg = str(config_path)
    log("[dlc] Running DeepLabCut analyze_videos...")
    deeplabcut.analyze_videos(cfg, video_strings, videotype=videotype, save_as_csv=True)

    if make_tracked_videos:
        log("[dlc] Creating unfiltered tracked videos...")
        deeplabcut.create_labeled_video(cfg, video_strings, videotype=videotype, filtered=False)

    log("[dlc] Filtering DLC predictions...")
    deeplabcut.filterpredictions(cfg, video_strings, videotype=videotype, save_as_csv=True)

    if make_tracked_videos:
        log("[dlc] Creating filtered tracked videos...")
        deeplabcut.create_labeled_video(cfg, video_strings, videotype=videotype, filtered=True)


def _run_dlc_in_env(
    config_path: Path,
    videos: list[Path],
    videotype: str,
    work_dir: Path,
    dlc_env: str,
    make_tracked_videos: bool,
    log: LogFn,
) -> None:
    log(f"[dlc] Running DeepLabCut in {_describe_dlc_env(dlc_env)}.")
    work_dir.mkdir(parents=True, exist_ok=True)
    job_path = work_dir / "dlc_job.json"
    script_path = work_dir / "run_dlc_job.py"
    job_path.write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "videos": [str(path) for path in videos],
                "videotype": videotype,
                "make_tracked_videos": make_tracked_videos,
            }
        ),
        encoding="utf-8",
    )
    script_path.write_text(_DLC_RUNNER_SCRIPT, encoding="utf-8")

    cmd = [*_dlc_python_command(dlc_env), str(script_path), str(job_path)]
    process = subprocess.Popen(
        cmd,
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if line:
            log(f"[dlc] {line}")
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"DeepLabCut failed in {_describe_dlc_env(dlc_env)} with exit code {returncode}.")


_DLC_SETUP_HINT = (
    "    uv venv .venv-dlc --python 3.10\n"
    "    uv pip install --python .venv-dlc -r requirements-dlc.txt\n"
    "On Apple Silicon, use requirements-dlc-macos.txt with the explicit override\n"
    "instead; see GUIDE.md, Tracking new videos, for platform-specific commands."
)


def _venv_python(root: Path) -> Optional[Path]:
    """The interpreter inside a virtual environment folder, if that is what it is."""
    candidate = root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    return candidate if candidate.is_file() else None


def _describe_dlc_env(dlc_env: str) -> str:
    """How to name the DeepLabCut environment in a message, without resolving it."""
    spec = dlc_env.strip()
    if not spec:
        return f"the bundled environment {DEFAULT_DLC_VENV}"
    if Path(spec).exists():
        return f"the environment at {spec}"
    return f"conda environment '{spec}'"


def _dlc_python_command(dlc_env: str) -> list[str]:
    """The command prefix that runs a Python script inside DeepLabCut's environment.

    Three spellings are accepted, so nobody has to convert what they already
    have: a folder made by "uv venv", a path straight to a Python interpreter,
    and a bare conda environment name. Blank means the bundled .venv-dlc.
    """
    spec = dlc_env.strip()
    root = Path(spec) if spec else DEFAULT_DLC_VENV

    interpreter = _venv_python(root)
    if interpreter is not None:
        return [str(interpreter)]
    if root.is_file():
        return [str(root)]

    if not spec:
        raise FileNotFoundError(
            f"The bundled DeepLabCut environment is not there yet ({DEFAULT_DLC_VENV}). "
            f"Create it with:\n\n{_DLC_SETUP_HINT}\n\n"
            "Or name an environment you already have in the DeepLabCut environment field."
        )
    # Not a path, so it is a conda environment name. "python" completes the
    # prefix, which lets both branches take the same script arguments.
    return [_conda_executable(), "run", "--no-capture-output", "-n", spec, "python"]


def _conda_search_paths() -> list[Path]:
    """Standard installation locations, in the order conda's own installers use."""
    home = Path.home()
    if sys.platform == "win32":
        roots = [
            home / "miniforge3",
            home / "mambaforge",
            home / "miniconda3",
            home / "anaconda3",
            Path("C:/ProgramData/miniconda3"),
            Path("C:/ProgramData/anaconda3"),
        ]
        return [root / sub / name for root in roots for sub in ("Scripts", "condabin") for name in ("conda.exe", "conda.bat")]
    roots = [
        home / "miniforge3",
        home / "mambaforge",
        home / "miniconda3",
        home / "anaconda3",
        Path("/opt/conda"),
        Path("/opt/miniconda3"),
        Path("/usr/local/miniconda3"),
    ]
    return [root / "bin" / "conda" for root in roots]


def _conda_executable() -> str:
    """Locate conda without assuming it is on PATH.

    Conda only reaches PATH in shells where its init script has run, which is
    routinely not the case for a desktop launcher or an editor-spawned terminal.
    """
    from_env = os.environ.get("CONDA_EXE")
    if from_env and Path(from_env).exists():
        return from_env

    for name in ("conda", "mamba", "micromamba"):
        found = which(name)
        if found:
            return found

    for candidate in _conda_search_paths():
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not find conda. Conda is only needed if the DeepLabCut environment "
        "field names a conda environment; the supported route needs no conda at all:\n\n"
        f"{_DLC_SETUP_HINT}\n\n"
        "Otherwise set CONDA_EXE, or put conda on PATH."
    )


_DLC_RUNNER_SCRIPT = r'''
import json
import sys

import deeplabcut


def main():
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        job = json.load(handle)
    cfg = job["config_path"]
    videos = job["videos"]
    videotype = job["videotype"]
    make_tracked_videos = bool(job["make_tracked_videos"])

    print("Running analyze_videos", flush=True)
    deeplabcut.analyze_videos(cfg, videos, videotype=videotype, save_as_csv=True)

    if make_tracked_videos:
        print("Creating unfiltered tracked videos", flush=True)
        deeplabcut.create_labeled_video(cfg, videos, videotype=videotype, filtered=False)

    print("Filtering predictions", flush=True)
    deeplabcut.filterpredictions(cfg, videos, videotype=videotype, save_as_csv=True)

    if make_tracked_videos:
        print("Creating filtered tracked videos", flush=True)
        deeplabcut.create_labeled_video(cfg, videos, videotype=videotype, filtered=True)


if __name__ == "__main__":
    main()
'''


def _collect_dlc_outputs(
    videos: list[Path],
    tracked_unfiltered_dir: Path,
    tracked_filtered_dir: Path,
    dlc_unfiltered_dir: Path,
    dlc_filtered_dir: Path,
    log: LogFn,
) -> None:
    counts = {"unfiltered_dlc": 0, "filtered_dlc": 0, "unfiltered_video": 0, "filtered_video": 0}
    for video in videos:
        folder = video.parent
        stem = glob_escape(video.stem)
        metadata = sorted(folder.glob(f"{stem}*includingmetadata.pickle"))
        for src in sorted(folder.glob(f"{stem}DLC*.csv")) + sorted(folder.glob(f"{stem}DLC*.h5")):
            if "filtered" in src.stem:
                copy2(src, dlc_filtered_dir / src.name)
                counts["filtered_dlc"] += 1
            else:
                copy2(src, dlc_unfiltered_dir / src.name)
                counts["unfiltered_dlc"] += 1
        for src in metadata:
            copy2(src, dlc_unfiltered_dir / src.name)
            copy2(src, dlc_filtered_dir / src.name)

        for src in sorted(folder.glob(f"{stem}*labeled.mp4")) + sorted(folder.glob(f"{stem}*labeled.avi")):
            if "filtered" in src.stem:
                copy2(src, tracked_filtered_dir / src.name)
                counts["filtered_video"] += 1
            else:
                copy2(src, tracked_unfiltered_dir / src.name)
                counts["unfiltered_video"] += 1

    filtered_csvs = list(dlc_filtered_dir.glob("*filtered.csv"))
    if not filtered_csvs:
        raise FileNotFoundError("No filtered DLC CSVs were found after DLC. Cannot continue.")
    log(
        "[collect] DLC files copied: "
        f"{counts['unfiltered_dlc']} unfiltered coordinate files, {counts['filtered_dlc']} filtered coordinate files."
    )
    log(
        "[collect] Tracked videos copied: "
        f"{counts['unfiltered_video']} unfiltered, {counts['filtered_video']} filtered."
    )


def _run_freezing_predictions(
    filtered_dir: Path,
    original_dir: Path,
    annotation_dir: Path,
    features_dir: Path,
    predictions_dir: Path,
    settings: FriendlyPipelineSettings,
    log: LogFn,
) -> tuple[dict[str, dict[str, object]], float]:
    model = load_model(settings.model_path)
    threshold = resolve_threshold(model, settings.threshold, 0.35)
    features_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    source = "requested" if settings.threshold is not None else "stored in the model"
    # Name the file, not "the default": two models ship and which one ran
    # changes the numbers, so it belongs in the log next to the threshold.
    log(f"[model] {settings.model_path.name}")
    log(
        f"[model] Threshold {threshold:.2f} ({source}); "
        f"feature mode '{model.feature_mode}' with {len(model.feature_names)} features."
    )

    outputs: dict[str, dict[str, object]] = {}
    csvs = sorted(filtered_dir.glob("*filtered.csv"))
    for idx, csv_path in enumerate(csvs, start=1):
        animal_key = _animal_key(csv_path.stem)
        log(f"[predict] {idx}/{len(csvs)} {csv_path.name}")
        flat = dlc_filtered_csv_to_flat_df(csv_path)
        features = build_feature_set(flat, mode=model.feature_mode)
        features.to_csv(features_dir / f"{csv_path.stem}_features.csv", index=False)
        probs, pred = predict_freezing(features, model, threshold)
        pred_csv = predictions_dir / f"{csv_path.stem}_pred.csv"
        save_frame_predictions(pred_csv, probs, pred)

        original = find_source_video(original_dir, csv_path.stem)
        annotated = annotation_dir / f"{csv_path.stem}_annotated.mp4"
        annotate_video(original, annotated, probs, pred, int(round(settings.fps)))
        outputs[animal_key] = {
            "csv_path": csv_path,
            "original_video": original,
            "prediction_csv": pred_csv,
            "annotated_video": annotated,
            "probs": probs,
            "pred": pred,
            "stem": csv_path.stem,
            # Measured here, where the tracking is already parsed, so the
            # workbook stage does not read every CSV a second time.
            "locomotor_px": _centroid_path_pixels(flat),
        }
    log(f"[features] Saved {len(csvs)} per-animal feature CSVs to {features_dir}")
    log(f"[predictions] Saved {len(csvs)} per-animal prediction CSVs to {predictions_dir}")
    return outputs, threshold


def _run_behavior_predictions(
    filtered_dir: Path,
    original_dir: Path,
    features_dir: Path,
    predictions_dir: Path,
    annotation_dir: Path,
    settings: FriendlyPipelineSettings,
    log: LogFn,
    freezing_predictions: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    behavior_model = load_behavior_model(settings.behavior_model_path, settings.behavior_label_encoder_path)
    features_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"[behavior] Loaded behavior model with {len(behavior_model.feature_names)} features "
        f"and classes: {', '.join(behavior_model.classes)}"
    )

    outputs: dict[str, dict[str, object]] = {}
    csvs = sorted(filtered_dir.glob("*filtered.csv"))
    for idx, csv_path in enumerate(csvs, start=1):
        animal_key = _animal_key(csv_path.stem)
        log(f"[behavior] {idx}/{len(csvs)} {csv_path.name}")
        flat = dlc_filtered_csv_to_flat_df(csv_path)
        features = build_behavior_feature_set(flat, behavior_model.feature_names, fps=settings.fps)
        feature_csv = features_dir / f"{csv_path.stem}_behavior_features.csv"
        features.to_csv(feature_csv, index=False)
        raw_labels, pred_codes, prob_df = predict_behaviors(features, behavior_model)
        freeze_payload = freezing_predictions.get(animal_key) if freezing_predictions else None
        freeze_pred = np.asarray(freeze_payload["pred"], dtype=int) if freeze_payload else None
        freeze_prob = np.asarray(freeze_payload["probs"], dtype=float) if freeze_payload else None
        original = find_source_video(original_dir, csv_path.stem)
        labels, label_source, metrics_df = refine_behavior_labels(
            raw_labels,
            prob_df,
            flat,
            freezing_pred=freeze_pred,
            freezing_prob=freeze_prob,
        )
        confidence = metrics_df["confidence"].to_numpy(dtype=float)
        pred_csv = predictions_dir / f"{csv_path.stem}_behavior_pred.csv"
        save_behavior_predictions(
            pred_csv,
            labels,
            pred_codes,
            prob_df,
            raw_labels=raw_labels,
            label_source=label_source,
            metrics_df=metrics_df,
            confidence=confidence,
        )

        annotated = annotation_dir / f"{csv_path.stem}_behavior_annotated.mp4"
        annotate_behavior_video(
            original,
            annotated,
            labels,
            prob_df,
            int(round(settings.fps)),
            metrics_df=metrics_df,
            confidence=confidence,
        )
        outputs[animal_key] = {
            "csv_path": csv_path,
            "original_video": original,
            "feature_csv": feature_csv,
            "prediction_csv": pred_csv,
            "annotated_video": annotated,
            "labels": labels,
            "raw_labels": raw_labels,
            "label_source": label_source,
            "metrics_df": metrics_df,
            "pred_codes": pred_codes,
            "prob_df": prob_df,
            "stem": csv_path.stem,
        }
    log(f"[behavior] Saved {len(csvs)} behavior feature/prediction files and annotated videos.")
    return outputs


def _write_results_workbook(
    out_xlsx: Path,
    predictions: dict[str, dict[str, object]],
    settings: FriendlyPipelineSettings,
    log: LogFn,
) -> None:
    total_rows = []
    bin_rows = []
    locomotor_rows = []
    cm_per_pixel = settings.box_side_cm / settings.box_side_pixels
    frames_per_bin = int(round(settings.fps * settings.bin_sec))
    if frames_per_bin <= 0:
        raise ValueError("Invalid fps/bin settings.")

    for animal_key, payload in predictions.items():
        pred = np.asarray(payload["pred"], dtype=int)
        stem = str(payload["stem"])
        animal, trial, mouse = _parse_animal(stem)
        duration_seconds = len(pred) / settings.fps if settings.fps else 0.0
        freezing_seconds = float(pred.sum() / settings.fps)
        moving_seconds = float(duration_seconds - freezing_seconds)
        total_rows.append(
            {
                "animal": animal,
                "trial": trial,
                "mouse": mouse,
                "frames": len(pred),
                "duration_seconds": duration_seconds,
                "freezing_seconds": freezing_seconds,
                "moving_seconds": moving_seconds,
                "freezing_percent": _pct(freezing_seconds, duration_seconds),
                "moving_percent": _pct(moving_seconds, duration_seconds),
            }
        )

        for bin_start in range(0, len(pred), frames_per_bin):
            bin_index = bin_start // frames_per_bin
            part = pred[bin_start : bin_start + frames_per_bin]
            bin_duration = len(part) / settings.fps
            bin_freezing = float(part.sum() / settings.fps)
            bin_rows.append(
                {
                    "animal": animal,
                    "trial": trial,
                    "mouse": mouse,
                    "bin_index": bin_index,
                    "freezing_seconds": bin_freezing,
                    "freezing_percent": _pct(bin_freezing, bin_duration),
                }
            )

        locomotor_px = float(payload["locomotor_px"])
        locomotor_rows.append(
            {
                "animal": animal,
                "trial": trial,
                "mouse": mouse,
                "total_locomotor_activity_pixels": locomotor_px,
                "total_locomotor_activity_cm": locomotor_px * cm_per_pixel,
                "cm_per_pixel": cm_per_pixel,
                "box_side_cm": settings.box_side_cm,
                "box_side_pixels_used": settings.box_side_pixels,
            }
        )

    totals = pd.DataFrame(total_rows).sort_values(["trial", "mouse"], na_position="last")
    bins = pd.DataFrame(bin_rows).sort_values(["trial", "mouse", "bin_index"], na_position="last")
    locomotor = pd.DataFrame(locomotor_rows).sort_values(["trial", "mouse"], na_position="last")

    seconds_wide = bins.pivot(index=["animal", "trial", "mouse"], columns="bin_index", values="freezing_seconds").reset_index()
    percent_wide = bins.pivot(index=["animal", "trial", "mouse"], columns="bin_index", values="freezing_percent").reset_index()
    seconds_wide.columns = _wide_bin_columns(seconds_wide.columns, "freezing_seconds")
    percent_wide.columns = _wide_bin_columns(percent_wide.columns, "freezing_percent")

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    # The sheet names carry the real bin length; with the default 30 s bin they
    # match the lab's historical workbooks exactly.
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        totals.to_excel(writer, sheet_name="total_freezing_seconds", index=False)
        seconds_wide.to_excel(writer, sheet_name=f"freezing_seconds_{settings.bin_sec}stimebin", index=False)
        percent_wide.to_excel(writer, sheet_name=f"freezing_percentage_{settings.bin_sec}stimebin", index=False)
        locomotor.to_excel(writer, sheet_name="locomotor_activity", index=False)
    log(f"[results] Wrote {out_xlsx.name}")


def _write_behavior_results(
    out_xlsx: Path,
    predictions: dict[str, dict[str, object]],
    settings: FriendlyPipelineSettings,
    log: LogFn,
) -> None:
    total_rows = []
    bin_rows = []
    frames_per_bin = int(round(settings.fps * settings.bin_sec))
    if frames_per_bin <= 0:
        raise ValueError("Invalid fps/bin settings.")

    for _, payload in predictions.items():
        labels = np.asarray(payload["labels"], dtype=str)
        stem = str(payload["stem"])
        animal, trial, mouse = _parse_animal(stem)
        duration_seconds = len(labels) / settings.fps if settings.fps else 0.0
        for behavior in BEHAVIOR_ORDER:
            seconds = float(np.sum(labels == behavior) / settings.fps)
            total_rows.append(
                {
                    "animal": animal,
                    "trial": trial,
                    "mouse": mouse,
                    "behavior": behavior,
                    "seconds": seconds,
                    "percent": _pct(seconds, duration_seconds),
                }
            )
        unassigned_seconds = float(np.sum(labels == UNASSIGNED) / settings.fps)
        if unassigned_seconds:
            total_rows.append(
                {
                    "animal": animal,
                    "trial": trial,
                    "mouse": mouse,
                    "behavior": UNASSIGNED,
                    "seconds": unassigned_seconds,
                    "percent": _pct(unassigned_seconds, duration_seconds),
                }
            )

        for bin_start in range(0, len(labels), frames_per_bin):
            bin_index = bin_start // frames_per_bin
            part = labels[bin_start : bin_start + frames_per_bin]
            bin_duration = len(part) / settings.fps
            for behavior in BEHAVIOR_ORDER:
                seconds = float(np.sum(part == behavior) / settings.fps)
                bin_rows.append(
                    {
                        "animal": animal,
                        "trial": trial,
                        "mouse": mouse,
                        "bin_index": bin_index,
                        "behavior": behavior,
                        "seconds": seconds,
                        "percent": _pct(seconds, bin_duration),
                    }
                )

    totals = pd.DataFrame(total_rows).sort_values(["trial", "mouse", "behavior"], na_position="last")
    bins = pd.DataFrame(bin_rows).sort_values(["trial", "mouse", "bin_index", "behavior"], na_position="last")
    seconds_wide = totals.pivot(index=["animal", "trial", "mouse"], columns="behavior", values="seconds").reset_index()
    percent_wide = totals.pivot(index=["animal", "trial", "mouse"], columns="behavior", values="percent").reset_index()

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx) as writer:
        totals.to_excel(writer, sheet_name="behavior_total_long", index=False)
        seconds_wide.to_excel(writer, sheet_name="behavior_total_seconds", index=False)
        percent_wide.to_excel(writer, sheet_name="behavior_total_percent", index=False)
        bins.to_excel(writer, sheet_name=f"behavior_{settings.bin_sec}s_long", index=False)
    log(f"[behavior] Wrote {out_xlsx.name}")


def _write_ethogram(
    out_png: Path,
    out_pdf: Path,
    predictions: dict[str, dict[str, object]],
    settings: FriendlyPipelineSettings,
    log: LogFn,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = []
    animals = []
    bin_sec = 0.5
    for payload in predictions.values():
        pred = np.asarray(payload["pred"], dtype=int)
        animal, trial, mouse = _parse_animal(str(payload["stem"]))
        animals.append((trial, mouse, animal))
        n_bins = int(np.ceil(len(pred) / (settings.fps * bin_sec)))
        raw = np.zeros(n_bins, dtype=int)
        for bin_idx in range(n_bins):
            start = int(round(bin_idx * bin_sec * settings.fps))
            end = int(round((bin_idx + 1) * bin_sec * settings.fps))
            part = pred[start:min(end, len(pred))]
            if len(part) and float(part.mean()) >= settings.ethogram_threshold:
                raw[bin_idx] = 1
        smooth = _remove_short_bouts(
            _fill_short_gaps(raw, int(round(settings.ethogram_gap_fill_sec / bin_sec))),
            int(round(settings.ethogram_min_bout_sec / bin_sec)),
        )
        for bin_idx, val in enumerate(smooth):
            if val:
                rows.append({"animal": animal, "trial": trial, "mouse": mouse, "minute": (bin_idx * bin_sec) / 60.0})

    event_df = pd.DataFrame(rows)
    animal_order = [a for _, _, a in sorted(animals, key=lambda x: (x[0] if x[0] is not None else 10**9, x[1] if x[1] is not None else 10**9, x[2]))]
    max_minutes = max(5.0, max((len(np.asarray(p["pred"])) / settings.fps / 60.0 for p in predictions.values()), default=5.0))

    fig, ax = plt.subplots(figsize=(12.5, max(7.0, len(animal_order) * 0.34)))
    pink = "#d57aa8"
    for row_idx, animal in enumerate(animal_order):
        y = len(animal_order) - 1 - row_idx
        if not event_df.empty:
            xs = event_df.loc[event_df["animal"] == animal, "minute"].to_numpy()
            if len(xs):
                ax.vlines(xs, y - 0.33, y + 0.33, color=pink, linewidth=0.65, alpha=0.98)
    ax.set_xlim(0, max_minutes)
    ax.set_ylim(-0.7, len(animal_order) - 0.3)
    ax.set_yticks(range(len(animal_order)))
    ax.set_yticklabels(list(reversed(animal_order)), fontsize=8, color="#4d4d4d")
    ax.set_xlabel("Time (minutes)", color="#4d4d4d")
    ax.set_ylabel("Animal", color="#4d4d4d")
    ax.set_title("Freezing ethogram", color="#4d4d4d", pad=14)
    _style_ethogram_axis(ax)
    ax.legend(handles=[Line2D([0], [0], color=pink, lw=1.8, label="Freezing")], loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=True, fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    log("[ethogram] Wrote final ethogram PNG/PDF.")


def _write_behavior_ethograms(
    ethogram_dir: Path,
    predictions: dict[str, dict[str, object]],
    settings: FriendlyPipelineSettings,
    log: LogFn,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D

    out_dir = ethogram_dir / "behavior_ethograms"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = ethogram_dir / "behavior_ethograms_ALL.pdf"
    handles = [Line2D([0], [0], color=BEHAVIOR_COLORS[b], lw=3, label=b) for b in BEHAVIOR_ORDER]
    bin_sec = 0.5
    frames_per_bin = max(1, int(round(settings.fps * bin_sec)))

    with PdfPages(pdf_path) as pdf:
        for _, payload in predictions.items():
            labels = np.asarray(payload["labels"], dtype=str)
            stem = str(payload["stem"])
            animal, trial, mouse = _parse_animal(stem)
            title = _animal_label(animal, trial, mouse)
            fig, ax = plt.subplots(figsize=(10.5, 3.0))
            for start in range(0, len(labels), frames_per_bin):
                part = labels[start : start + frames_per_bin]
                if len(part) == 0:
                    continue
                values, counts = np.unique(part, return_counts=True)
                behavior = str(values[int(np.argmax(counts))])
                if behavior not in BEHAVIOR_DISPLAY_ORDER:
                    continue
                y = BEHAVIOR_DISPLAY_ORDER.index(behavior)
                time_min = ((start + min(start + frames_per_bin, len(labels))) / 2.0) / settings.fps / 60.0
                ax.vlines(time_min, y - 0.38, y + 0.38, color=BEHAVIOR_COLORS[behavior], lw=0.55)

            max_minutes = max(1.0, len(labels) / settings.fps / 60.0)
            ax.set_xlim(0, max_minutes)
            ax.set_ylim(-0.8, len(BEHAVIOR_DISPLAY_ORDER) - 0.2)
            ax.set_yticks(range(len(BEHAVIOR_DISPLAY_ORDER)))
            ax.set_yticklabels(BEHAVIOR_DISPLAY_ORDER, fontsize=8, color="#4d4d4d")
            ax.set_xlabel("Time (minutes)", fontsize=8, color="#4d4d4d")
            ax.set_title(title, fontsize=10, color="#4d4d4d", pad=10)
            _style_ethogram_axis(ax)
            # Anchored by its top edge, so the legend grows downwards and cannot
            # land on top of the x axis label.
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=7, frameon=True, fontsize=7)
            fig.tight_layout()
            png_path = out_dir / f"{stem}_behavior_ethogram.png"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    log(f"[behavior] Wrote behavior ethograms to {out_dir} and {pdf_path.name}")


def _write_package_summary(package_dir: Path, settings: FriendlyPipelineSettings, log: LogFn) -> None:
    text = f"""Freezing analysis package

Open first:
- 06_results/freezing_FINAL_results.xlsx
- 06_results/behavior_FINAL_results.xlsx
- 07_ethogram/freezing_ethogram_vertical_chunks_FINAL.png
- 07_ethogram/behavior_ethograms_ALL.pdf
- 04_features/frame_predictions_per_animal for frame-by-frame freezing/moving probabilities
- 04_features/frame_features_per_animal for the classifier feature table per animal
- 04_features/behavior_predictions_per_animal for frame-by-frame seven-behavior predictions
- 04_features/behavior_features_per_animal for the seven-behavior feature table

Workflow:
- Original videos were copied into 01_original_videos.
- CLAHE preprocessing was {'enabled' if settings.run_clahe else 'disabled'} for DLC contrast enhancement.
- Generated CLAHE videos are in 01_original_videos/CLAHE_preprocessed_videos when preprocessing is enabled.
- 01_original_videos contains only source videos and CLAHE videos; DLC CSV/H5/labeled outputs are kept in later folders.
- DeepLabCut outputs are in 03_DLC_analysis, split into unfiltered and filtered.
- Tracked DLC videos are in 02_tracked_videos, split into unfiltered and filtered.
- Feature and frame prediction CSVs are in 04_features.
- Annotated videos are in 05_annotated_videos/freezing and 05_annotated_videos/behaviors.
- Freezing was called when classifier probability >= {settings.threshold:.2f}.
- Seven-behavior analysis was {'enabled' if settings.run_behavior_analysis else 'disabled'}.

Manual troubleshooting and refinement:
- Use the GUI button "Label videos and create refined model" if predictions need correction.
- Press "f" at the start of each freezing bout and press "f" again at the end; press "s" to save.
- The GUI asks whether to label another video.
- When finished, it trains a new .sav from the manual labels and matching filtered DLC CSVs.
- The new .sav is automatically selected in the GUI's freezing model field.

Results workbook sheets:
- total_freezing_seconds: total freezing/moving seconds and percentages per animal.
- freezing_seconds_30stimebin: freezing seconds in each {settings.bin_sec}-second time bin.
- freezing_percentage_30stimebin: freezing percentage in each {settings.bin_sec}-second time bin.
- locomotor_activity: centroid path length in pixels and cm.

Locomotor activity:
- Calculated from filtered DLC coordinates.
- Uses frame-to-frame displacement of the centroid of all tracked body parts.
- Conversion uses {settings.box_side_pixels:g} pixels = {settings.box_side_cm:g} cm.
"""
    (package_dir / "package_summary.txt").write_text(text, encoding="utf-8")
    log("[summary] Wrote package_summary.txt.")


def _animal_key(stem: str) -> str:
    return stem.split("DLC")[0].strip() or stem


def _parse_animal(stem: str) -> tuple[str, int | None, int | None]:
    import re

    match = re.search(r"Trial\s*(\d+)_mouse(\d+)", stem)
    if not match:
        return _animal_key(stem), None, None
    trial = int(match.group(1))
    mouse = int(match.group(2))
    return f"Trial {trial:02d} mouse{mouse}", trial, mouse


def _animal_label(animal: str, trial: int | None, mouse: int | None) -> str:
    parts = [animal]
    if trial is not None:
        parts.append(f"trial {trial}")
    if mouse is not None:
        parts.append(f"mouse {mouse}")
    return " | ".join(parts)


def _style_ethogram_axis(ax) -> None:
    """The shared look of every ethogram: no box, a soft grey baseline."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(axis="y", length=0)


def _pct(numerator: float, denominator: float) -> float:
    return float(100.0 * numerator / denominator) if denominator else 0.0


def _wide_bin_columns(columns: Iterable[object], metric: str) -> list[str]:
    out = []
    for col in columns:
        if col in {"animal", "trial", "mouse"}:
            out.append(str(col))
        else:
            out.append(f"bin_{int(col):02d}_{metric}")
    return out


def _centroid_path_pixels(flat: pd.DataFrame) -> float:
    x_cols = [c for c in flat.columns if c.endswith("_x")]
    y_cols = [c for c in flat.columns if c.endswith("_y")]
    bodyparts = sorted({c[:-2] for c in x_cols}.intersection({c[:-2] for c in y_cols}))
    if not bodyparts:
        return 0.0
    x = flat[[f"{bp}_x" for bp in bodyparts]].to_numpy(dtype=float)
    y = flat[[f"{bp}_y" for bp in bodyparts]].to_numpy(dtype=float)
    centroid_x = np.nanmean(x, axis=1)
    centroid_y = np.nanmean(y, axis=1)
    return float(np.nansum(np.sqrt(np.diff(centroid_x) ** 2 + np.diff(centroid_y) ** 2)))


def _fill_short_gaps(binary: np.ndarray, max_gap_bins: int) -> np.ndarray:
    arr = binary.copy()
    i = 0
    while i < len(arr):
        if arr[i] == 1:
            i += 1
            continue
        start = i
        while i < len(arr) and arr[i] == 0:
            i += 1
        end = i
        if start > 0 and end < len(arr) and arr[start - 1] == 1 and arr[end] == 1 and (end - start) <= max_gap_bins:
            arr[start:end] = 1
    return arr


def _remove_short_bouts(binary: np.ndarray, min_bout_bins: int) -> np.ndarray:
    arr = binary.copy()
    i = 0
    while i < len(arr):
        if arr[i] == 0:
            i += 1
            continue
        start = i
        while i < len(arr) and arr[i] == 1:
            i += 1
        if (i - start) < min_bout_bins:
            arr[start:i] = 0
    return arr
