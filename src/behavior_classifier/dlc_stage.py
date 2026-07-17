"""DeepLabCut execution helpers for raw-video behavior prediction."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2


def _video_extension(video: Path, videotype: str | None = None) -> str:
    if videotype:
        return videotype if videotype.startswith(".") else f".{videotype}"
    return video.suffix or ".mp4"


def _collect_filtered_outputs(videos: list[Path]) -> list[Path]:
    found: list[Path] = []
    for video in videos:
        found.extend(sorted(video.parent.glob(f"{video.stem}*filtered.csv")))
        found.extend(sorted(video.parent.glob(f"{video.stem}*filtered.h5")))
    return list(dict.fromkeys(found))


def run_deeplabcut_to_filtered(
    *,
    config_path: Path,
    videos: list[Path],
    filtered_dir: Path,
    videotype: str | None = None,
    reuse_existing: bool = True,
) -> list[Path]:
    """Run DLC analyze/filter for videos and copy filtered tracks to `filtered_dir`."""

    if not config_path.exists():
        raise FileNotFoundError(f"DeepLabCut config not found: {config_path}")
    missing = [video for video in videos if not video.exists()]
    if missing:
        raise FileNotFoundError(f"Video not found: {missing[0]}")

    found = _collect_filtered_outputs(videos) if reuse_existing else []
    if not found:
        try:
            import deeplabcut  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "DeepLabCut is not installed/runnable here and no existing filtered "
                "DLC outputs were found next to the video."
            ) from exc

        video_type = _video_extension(videos[0], videotype)
        video_paths = [str(video) for video in videos]
        cfg = str(config_path)
        deeplabcut.analyze_videos(cfg, video_paths, videotype=video_type, save_as_csv=True)
        deeplabcut.filterpredictions(cfg, video_paths, videotype=video_type, save_as_csv=True)
        found = _collect_filtered_outputs(videos)

    if not found:
        raise FileNotFoundError("No DLC '*filtered.csv' or '*filtered.h5' output was found.")

    filtered_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in found:
        dst = filtered_dir / src.name
        copy2(src, dst)
        copied.append(dst)
    return copied
