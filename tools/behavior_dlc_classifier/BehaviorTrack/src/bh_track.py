#!/usr/bin/env python
"""Step 2: run DeepLabCut over the recordings.

This step deliberately calls the engine's own DeepLabCut stage rather than
driving ``deeplabcut`` directly. Launching DLC is the fiddliest part of the tool
— it usually lives in a *second* environment, its ``read_config`` rewrites
``project_path`` to wherever it read the config from, and a config kept anywhere
but beside ``dlc-models`` makes it report the network as missing. All of that is
already solved in ``freezing_dlc.friendly_pipeline``; a second copy here is
exactly the "hand-replica that silently drifts" the integration plan warns about,
so the private helpers are imported on purpose. The coupling is named in
BehaviorTrack/GUIDE.md.

Recordings that already have tracking beside them are skipped unless you ask for
a re-run, because tracking is the slow step.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from glob import escape as glob_escape
from pathlib import Path
from shutil import copy2
from tkinter import filedialog, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config_loader import ensure_engine_importable, resolve_config_path  # noqa: E402
from utils.runner import StepWindow, path_row  # noqa: E402
from utils.session_parser import find_tracking_for  # noqa: E402
from utils.workspace import load_workspace  # noqa: E402


def _copy_video_inputs(videos: list[Path], out_dir: Path, log) -> list[Path]:
    """Make persistent DLC inputs so raw recordings remain read-only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for video in videos:
        target = out_dir / video.name
        if video.resolve() != target.resolve():
            copy2(video, target)
        copied.append(target)
    log(f"[dlc] Prepared {len(copied)} working video copy/copies; source recordings were not modified.")
    return copied


def _restore_cached_outputs(source_video: Path, prepared_video: Path, workspace) -> int:
    """Put archived DLC artifacts beside a working video for stage-level resume."""
    stem = source_video.stem
    origins = (
        source_video.parent,
        workspace.dlc_analysis_dir(stem),
        workspace.dlc_filtered_dir(stem),
        workspace.dlc_tracked_videos_dir(stem),
    )
    patterns = (
        f"{glob_escape(stem)}DLC*.h5",
        f"{glob_escape(stem)}DLC*.csv",
        f"{glob_escape(stem)}DLC*includingmetadata.pickle",
        f"{glob_escape(stem)}*filtered_labeled.mp4",
        f"{glob_escape(stem)}*filtered_labeled.avi",
    )
    restored = 0
    seen: set[Path] = set()
    for origin in origins:
        if not origin.is_dir():
            continue
        for pattern in patterns:
            for source in origin.glob(pattern):
                resolved = source.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                target = prepared_video.parent / source.name
                if resolved != target.resolve():
                    copy2(source, target)
                    restored += 1
    return restored


def _archive_dlc_outputs(prepared_video: Path, workspace) -> int:
    """Archive every reusable DLC stage under Results/DLC/animal/session."""
    stem = prepared_video.stem
    escaped = glob_escape(stem)
    analysis_dir = workspace.dlc_analysis_dir(stem)
    filtered_dir = workspace.dlc_filtered_dir(stem)
    tracked_dir = workspace.dlc_tracked_videos_dir(stem)
    filtered_csvs = 0

    for source in sorted(prepared_video.parent.glob(f"{escaped}DLC*.h5")) + sorted(
        prepared_video.parent.glob(f"{escaped}DLC*.csv")
    ):
        destination_dir = filtered_dir if "filtered" in source.stem else analysis_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        copy2(source, destination_dir / source.name)
        if "filtered" in source.stem and source.suffix.lower() == ".csv":
            filtered_csvs += 1

    for source in prepared_video.parent.glob(f"{escaped}DLC*includingmetadata.pickle"):
        analysis_dir.mkdir(parents=True, exist_ok=True)
        copy2(source, analysis_dir / source.name)

    labeled = sorted(prepared_video.parent.glob(f"{escaped}*filtered_labeled.mp4"))
    labeled += sorted(prepared_video.parent.glob(f"{escaped}*filtered_labeled.avi"))
    for source in labeled:
        tracked_dir.mkdir(parents=True, exist_ok=True)
        copy2(source, tracked_dir / source.name)
    return filtered_csvs


def _engine():
    """Engine imports, deferred so ``--help`` works without DeepLabCut installed."""
    from freezing_dlc import dlc_stage
    from freezing_dlc.friendly_pipeline import (  # noqa: PLC2701 - see module docstring
        _check_deeplabcut_available,
        _describe_dlc_env,
        _preprocess_videos_clahe,
        _run_dlc,
    )

    return dlc_stage, _run_dlc, _check_deeplabcut_available, _describe_dlc_env, _preprocess_videos_clahe


class TrackWindow(StepWindow):
    step_title = "2. Run DeepLabCut tracking"
    step_subtitle = "Pose tracking with the bundled network"
    step_blurb = (
        "Tracks every recording that does not already have a filtered CSV beside it. DeepLabCut usually lives "
        "in its own environment - leave the field blank to use the .venv-dlc bundled with the tool, or name a "
        "conda environment or interpreter. This is the slow step; already-tracked recordings are skipped."
    )
    run_label = "Run tracking"

    def __init__(self, config_path: Path) -> None:
        super().__init__(config_path, geometry="1200x860")

    def initialize_state(self) -> None:
        self.workspace = load_workspace(self.config_path)
        models = self.workspace.models()
        tracking = self.workspace.section("tracking")
        self.videos_var = tk.StringVar(value=str(self.workspace.raw_videos))
        self.dlc_config_var = tk.StringVar(value=str(models.dlc_config))
        self.dlc_env_var = tk.StringVar(value=str(self.workspace.section("project").get("dlc_env", "") or ""))
        self.clahe_var = tk.BooleanVar(value=bool(tracking.get("run_clahe", True)))
        self.labeled_var = tk.BooleanVar(value=False)
        self.force_var = tk.BooleanVar(value=False)
        # Off by default on purpose: it is a change to what the network is
        # shown, not only to how fast it runs, so it is never applied to a
        # cohort unless it was asked for.
        self.dynamic_var = tk.BooleanVar(value=False)
        self.summary_var = tk.StringVar(value="")

    def after_build(self) -> None:
        self.refresh_summary()

    def results_dir(self) -> Path:
        return self.workspace.tracking_dir

    def body(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        path_row(parent, 0, "Video folder", self.videos_var, self.browse_videos)
        path_row(parent, 1, "DeepLabCut config", self.dlc_config_var, self.browse_config)

        row = ttk.Frame(parent, style="Panel.TFrame")
        row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text="DeepLabCut environment", style="MutedPanel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(row, textvariable=self.dlc_env_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(
            row,
            text="Blank = the bundled .venv-dlc beside the tool.",
            style="MutedPanel.TLabel",
        ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(4, 0))

        checks = ttk.Frame(parent, style="Panel.TFrame")
        checks.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(
            checks,
            text="CLAHE contrast equalisation before tracking (matches how the network was trained)",
            variable=self.clahe_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            checks, text="Also write DeepLabCut's labelled overlay videos", variable=self.labeled_var
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            checks, text="Re-track recordings that already have tracking", variable=self.force_var
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            checks,
            text=(
                "Dynamic cropping: much faster without a GPU, but the network sees a box "
                "around the animal rather than the whole arena, so poses may differ slightly "
                "from a full-frame run"
            ),
            variable=self.dynamic_var,
        ).grid(row=3, column=0, sticky="w", pady=(4, 0))

        ttk.Button(checks, text="Rescan", command=self.refresh_summary).grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Label(parent, textvariable=self.summary_var, style="Panel.TLabel", wraplength=1040, justify="left").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )

    # ------------------------------------------------------------------ ops #

    def browse_videos(self) -> None:
        chosen = filedialog.askdirectory(title="Folder holding the recordings", initialdir=self.videos_var.get() or ".")
        if chosen:
            self.videos_var.set(chosen)
            self.refresh_summary()

    def browse_config(self) -> None:
        chosen = filedialog.askopenfilename(
            title="DeepLabCut config.yaml",
            filetypes=[("DeepLabCut config", "config.yaml"), ("YAML", "*.yaml"), ("All files", "*.*")],
        )
        if chosen:
            self.dlc_config_var.set(chosen)

    def _pending(self) -> tuple[list[Path], list[Path]]:
        ensure_engine_importable(self.config_path)
        from freezing_dlc.dlc_stage import list_videos

        videos = list_videos(Path(self.videos_var.get()), self.workspace.videotype)
        videos = [v for v in videos if "DLC" not in v.stem and not v.stem.endswith("_labeled")]
        todo = [
            v
            for v in videos
            if self.force_var.get()
            or (
                (find_tracking_for(v) is None and self.workspace.tracking_for_stem(v.stem) is None)
                or (self.labeled_var.get() and not self.workspace.has_dlc_tracked_video(v.stem))
            )
        ]
        return videos, todo

    def refresh_summary(self) -> None:
        try:
            videos, todo = self._pending()
        except Exception as exc:
            self.summary_var.set(f"Could not scan: {exc}")
            return
        config = Path(self.dlc_config_var.get())
        network = config.parent.name if config.is_file() else "NOT FOUND"
        self.summary_var.set(
            f"{len(videos)} recordings, {len(todo)} to track.   Network: {network}"
            + ("" if config.is_file() else "   (the DeepLabCut config does not exist)")
        )

    def work(self, log) -> str:
        ensure_engine_importable(self.config_path)
        _dlc_stage, run_dlc, check_available, describe_env, clahe = _engine()

        videos, todo = self._pending()
        if not videos:
            raise RuntimeError(f"No '*{self.workspace.videotype}' recordings in {self.videos_var.get()}")
        if not todo:
            log("[dlc] Every recording already has tracking. Tick the re-track box to force a re-run.")
            return "Nothing to track."

        config_path = Path(self.dlc_config_var.get())
        if not config_path.is_file():
            raise FileNotFoundError(f"DeepLabCut config not found: {config_path}")

        dlc_env = self.dlc_env_var.get().strip()
        log(f"[dlc] Environment: {describe_env(dlc_env)}")
        check_available(dlc_env)

        self.workspace.ensure_dirs()
        work_dir = self.workspace.results / "_working"
        work_dir.mkdir(parents=True, exist_ok=True)

        inputs: list[Path]
        if self.clahe_var.get():
            log(f"[clahe] Equalising {len(todo)} recording(s)...")
            inputs = clahe(todo, work_dir / "clahe", log)
        else:
            log("[clahe] Skipped; DeepLabCut will read persistent copies of the original recordings.")
            inputs = _copy_video_inputs(todo, work_dir / "dlc_input", log)

        restored = sum(
            _restore_cached_outputs(source, prepared, self.workspace)
            for source, prepared in zip(todo, inputs)
        )
        if restored:
            log(f"[resume] Restored {restored} cached DLC artifact(s) for per-stage reuse.")

        log(f"[dlc] Tracking {len(inputs)} recording(s), one at a time. This is the slow step.")
        log(
            "[pipeline] Each completed recording is published immediately, so steps 3 and 4 "
            "can run while the next recording is tracked."
        )
        self.set_progress(0, len(inputs))

        # Deliberately keep the DeepLabCut call inside the per-recording loop.
        # DLC otherwise accepts the whole batch, but then filtering and
        # collection do not happen until every recording has finished. Publishing
        # each filtered CSV here gives the downstream steps a stable, complete
        # recording to consume while inference continues on the next one.
        filtered_csvs = 0
        completed = 0
        for index, video in enumerate(inputs, start=1):
            log("")
            log(f"--- [track {index}/{len(inputs)}] {video.name} ---")
            run_dlc(
                config_path,
                [video],
                self.workspace.videotype,
                work_dir,
                dlc_env,
                self.labeled_var.get(),
                log,
                force=self.force_var.get(),
                dynamic=self.dynamic_var.get(),
            )

            # Keep all stage artifacts, not only the final CSV. The H5 analysis
            # and filtered files let a later run perform only filtering or only
            # labeled-video generation without repeating inference.
            published = _archive_dlc_outputs(video, self.workspace)
            if not published:
                raise FileNotFoundError(
                    f"DeepLabCut finished {video.name}, but no '*filtered.csv' file was found."
                )
            filtered_csvs += published
            completed += 1
            self.set_progress(index, len(inputs))
            log(
                f"[ready] {video.stem}: {published} filtered CSV(s) published; "
                "steps 3 and 4 may use this recording now."
            )

        log(
            f"[collect] {filtered_csvs} filtered CSV(s), with reusable stage files -> "
            f"{self.workspace.tracking_dir}/<animal>/<session>"
        )
        return f"Processed {completed} recording(s); {filtered_csvs} tracking file(s) collected."

    def _on_finished(self, ok: bool) -> None:
        self.refresh_summary()


def self_test(config_path: Path) -> int:
    """Check the DLC wiring resolves without running DeepLabCut."""
    print("[self-test] bh_track")
    workspace = load_workspace(config_path)
    ensure_engine_importable(config_path)
    models = workspace.models()

    checks: list[tuple[str, bool]] = []
    print(f"  video folder   {workspace.raw_videos}")
    print(f"  dlc config     {models.dlc_config}")
    checks.append(("DeepLabCut config exists", models.dlc_config.is_file()))

    project_dir = models.dlc_config.parent
    iterations = sorted(p.name for p in (project_dir / "dlc-models").glob("iteration-*")) if (project_dir / "dlc-models").is_dir() else []
    print(f"  network        {project_dir.name}")
    print(f"  iterations     {', '.join(iterations) or 'none found'}")
    checks.append(("at least one iteration present", bool(iterations)))

    # Which iteration the config actually selects.
    selected = ""
    for line in models.dlc_config.read_text(encoding="utf-8").splitlines():
        if line.startswith("iteration:"):
            selected = line.split(":", 1)[1].strip()
    print(f"  config selects iteration-{selected or '?'}")
    checks.append(("config names an iteration that exists", f"iteration-{selected}" in iterations))

    snapshots = sorted((project_dir / "dlc-models" / f"iteration-{selected}").rglob("snapshot-*.index"))
    print(f"  snapshots      {', '.join(p.stem for p in snapshots) or 'none'}")
    checks.append(("selected iteration has a snapshot", bool(snapshots)))

    from freezing_dlc.friendly_pipeline import _describe_dlc_env, _dlc_python_command

    dlc_env = str(workspace.section("project").get("dlc_env", "") or "")
    print(f"  dlc env        {_describe_dlc_env(dlc_env)}")
    try:
        command = _dlc_python_command(dlc_env)
        print(f"  interpreter    {command[0]}")
        env_ready = True
    except FileNotFoundError as exc:
        print(f"  interpreter    not available - {str(exc).splitlines()[0]}")
        env_ready = False

    for name, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    failed = [name for name, ok in checks if not ok]
    if failed:
        print(f"[self-test] FAILED: {len(failed)} check(s)")
        return 1
    if not env_ready:
        print("[self-test] PASSED with warnings (DeepLabCut environment not created yet)")
        return 0
    print("[self-test] PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BehaviorTrack step 2: tracking")
    parser.add_argument("--config", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    config_path = resolve_config_path(args.config)
    if args.self_test:
        return self_test(config_path)
    TrackWindow(config_path).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
