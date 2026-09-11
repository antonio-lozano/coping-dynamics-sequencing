from __future__ import annotations

from glob import escape as glob_escape
from pathlib import Path
from shutil import copy2

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: DeepLabCut project bundled with this tool. It is the network that produced the
#: demo tracking files, so a fresh checkout can analyse new videos without first
#: training or downloading anything.
DEFAULT_DLC_PROJECT_DIR = _REPO_ROOT / "models" / "dlc_project" / "Freezing_07-2020-Sanguino-Lozano-2020-09-29"
DEFAULT_DLC_CONFIG_PATH = DEFAULT_DLC_PROJECT_DIR / "config.yaml"


def list_videos(videos_dir: Path, videotype: str) -> list[Path]:
    """List videos of one type, matching the extension case-insensitively.

    Windows filesystems ignore case, so a folder of ``.MP4`` recordings is found
    there by a ``*.mp4`` pattern but not on Linux. Comparing suffixes directly
    keeps the same folder working on both.
    """
    ext = videotype if videotype.startswith(".") else f".{videotype}"
    ext = ext.lower()
    if not videos_dir.is_dir():
        return []
    return sorted(
        path for path in videos_dir.iterdir() if path.is_file() and path.suffix.lower() == ext
    )


def find_source_video(original_dir: Path, dlc_stem: str, videotype: str = ".mp4") -> Path:
    """Find the recording a tracking file came from.

    DeepLabCut appends its scorer name to the video stem, so
    ``Trial 9_mouse3DLC_resnet50_...filtered.csv`` belongs to ``Trial 9_mouse3.mp4``.

    Both the summary pipeline and the interface need this, and they previously
    carried a copy each. The copies drifted: one was fixed to compare extensions
    case-insensitively and to sort its matches, the other kept globbing ``*.mp4``
    and so failed on Linux for a folder of ``.MP4`` files. One function cannot
    drift from itself.
    """
    prefix = dlc_stem.split("DLC")[0]

    def best(paths: list[Path]) -> list[Path]:
        # An exact stem always outranks a longer name that merely starts the
        # same way, or "Trial 1"'s tracking would claim "Trial 10.mp4" whenever
        # Trial 1's own recording is missing.
        return [p for p in paths if p.stem == prefix] or [p for p in paths if p.name.startswith(prefix)]

    matches = best(list_videos(original_dir, videotype))
    if not matches and original_dir.is_dir():
        # The recording may simply not be the requested type. Accepting any
        # extension here beats failing when only the container differs.
        matches = best(sorted(path for path in original_dir.iterdir() if path.is_file()))
    if not matches:
        raise FileNotFoundError(f"No original video found for tracking file: {dlc_stem}")
    return matches[0]


def _collect_filtered_csvs(videos: list[Path]) -> list[Path]:
    collected: list[Path] = []
    for video in videos:
        # Escaped, or a legal bracket in a video name becomes glob syntax and
        # its tracking file is never found.
        collected.extend(sorted(video.parent.glob(f"{glob_escape(video.stem)}*filtered.csv")))
    # Deduplicate while preserving order
    return list(dict.fromkeys(collected))


#: Name of the patched config written next to a project's own config.yaml.
RESOLVED_CONFIG_NAME = "dlc_config_resolved.yaml"


def prepare_dlc_config(config_path: Path) -> Path:
    """Return a config whose ``project_path`` matches where the project actually lives.

    DeepLabCut project configs store an absolute ``project_path`` written on the
    machine that trained the network. The bundled project ships with a
    placeholder, and a user-supplied project may have been moved since it was
    created, so the stored path is rarely the right one.

    The patched copy goes *beside the original*, never into a working folder:
    DeepLabCut's own ``read_config`` overwrites ``project_path`` with the folder
    it read the config from, so a config kept anywhere else sends it looking for
    ``dlc-models`` in that other place and it reports the network as missing.
    Writing next to the original leaves the user's own config.yaml untouched
    while still giving DeepLabCut a file in the folder it needs.
    """
    config_path = Path(config_path).resolve()
    project_dir = config_path.parent

    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    patched: list[str] = []
    already_correct = False
    found = False
    for line in lines:
        if line.startswith("project_path:"):
            found = True
            current = line.split(":", 1)[1].strip()
            if current and Path(current) == project_dir:
                already_correct = True
            newline = "\n" if line.endswith("\n") else ""
            patched.append(f"project_path: {project_dir}{newline}")
        else:
            patched.append(line)

    # Pre-multianimal DLC projects declare bodyparts but omit this flag.
    # DLC 2.3's video renderer indexes it directly (analysis defaults to False).
    # Normalize only the runtime copy; never reinterpret explicit/multi-animal
    # metadata or rewrite the archived project configuration.
    keys = {line.split(":", 1)[0] for line in lines if ":" in line and not line[0].isspace()}
    legacy_single_animal = "bodyparts" in keys and not keys.intersection(
        {"multianimalproject", "multianimalbodyparts", "uniquebodyparts", "individuals"}
    )
    if legacy_single_animal:
        if patched and not patched[-1].endswith("\n"):
            patched[-1] += "\n"
        patched.append("multianimalproject: false\n")

    if (already_correct or not found) and not legacy_single_animal:
        return config_path

    patched_path = project_dir / RESOLVED_CONFIG_NAME
    try:
        patched_path.write_text("".join(patched), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"The DeepLabCut project folder is not writable ({project_dir}). "
            "DeepLabCut has to read its config from the folder holding dlc-models, "
            f"so {RESOLVED_CONFIG_NAME} could not be created. Either make the folder "
            "writable, or set project_path in config.yaml to that folder yourself."
        ) from exc
    return patched_path


def run_deeplabcut_to_filtered(
    config_path: Path,
    videos_dir: Path,
    filtered_dir: Path,
    videotype: str = ".mp4",
) -> list[Path]:
    if not videos_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {videos_dir}")

    videos = list_videos(videos_dir, videotype)
    if not videos:
        raise FileNotFoundError(
            f"No videos matching '*{videotype}' found in {videos_dir}"
        )

    found = _collect_filtered_csvs(videos)
    try:
        import deeplabcut  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        if not found:
            raise RuntimeError(
                "DeepLabCut is not installed or not runnable in this environment, "
                "and no existing '*filtered.csv' files were found next to the videos."
            ) from exc
        print("[dlc] DeepLabCut unavailable; reusing existing filtered CSVs next to the videos.")
    else:
        if not config_path.exists():
            if not found:
                raise FileNotFoundError(
                    f"DeepLabCut config not found: {config_path}, and no existing "
                    "'*filtered.csv' files were found next to the videos."
                )
            print("[dlc] No DeepLabCut config found; reusing existing filtered CSVs next to the videos.")
        else:
            cfg = str(prepare_dlc_config(config_path))
            videos_str = [str(v) for v in videos]
            deeplabcut.analyze_videos(cfg, videos_str, videotype=videotype, save_as_csv=True)
            deeplabcut.filterpredictions(cfg, videos_str, videotype=videotype, save_as_csv=True)
            found = _collect_filtered_csvs(videos)
            if not found:
                raise FileNotFoundError(
                    "DeepLabCut finished, but no '*filtered.csv' files were found next to videos."
                )

    uncovered = [video.name for video in videos if not any(c.stem.startswith(video.stem) for c in found)]
    if uncovered:
        print(
            f"[dlc] WARNING: no filtered CSVs for {len(uncovered)} video(s): "
            f"{', '.join(uncovered)}. They will be missing from the results."
        )

    filtered_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in found:
        dst = filtered_dir / src.name
        if src.resolve() != dst.resolve():
            copy2(src, dst)
        copied.append(dst)
    return copied
