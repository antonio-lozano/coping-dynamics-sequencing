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
BehaviorTrack/README.md.

Recordings that already have tracking beside them are skipped unless you ask for
a re-run, because tracking is the slow step.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from shutil import copy2
from tkinter import filedialog, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config_loader import ensure_engine_importable, resolve_config_path  # noqa: E402
from utils.runner import StepWindow, path_row  # noqa: E402
from utils.session_parser import find_tracking_for  # noqa: E402
from utils.workspace import load_workspace  # noqa: E402


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

        ttk.Button(checks, text="Rescan", command=self.refresh_summary).grid(row=3, column=0, sticky="w", pady=(10, 0))
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
        dlc_stage, run_dlc, check_available, describe_env, clahe = _engine()

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

        inputs = todo
        if self.clahe_var.get():
            log(f"[clahe] Equalising {len(todo)} recording(s)...")
            inputs = clahe(todo, work_dir / "clahe", log)
        else:
            log("[clahe] Skipped; DeepLabCut will read the original recordings.")

        log(f"[dlc] Tracking {len(inputs)} recording(s). This is the slow step.")
        self.set_progress(0, len(inputs))
        run_dlc(
            config_path,
            list(inputs),
            self.workspace.videotype,
            work_dir,
            dlc_env,
            self.labeled_var.get(),
            log,
        )
        self.set_progress(len(inputs), len(inputs))

        # `_run_dlc` already performed analysis and filtering (possibly in a
        # separate Python environment).  Calling run_deeplabcut_to_filtered
        # here would analyze every video a second time whenever DLC also happens
        # to be importable in this GUI environment.  Collect exactly what the
        # completed run wrote beside its input videos instead.
        found = dlc_stage._collect_filtered_csvs(inputs)  # noqa: SLF001 - engine's canonical filename matcher
        if not found:
            raise FileNotFoundError("DeepLabCut finished, but no '*filtered.csv' files were found.")
        copied: list[Path] = []
        for source in found:
            stem = source.stem.split("DLC")[0] or source.stem
            destination_dir = self.workspace.dlc_filtered_dir(stem)
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / source.name
            if source.resolve() != destination.resolve():
                copy2(source, destination)
            copied.append(destination)
            if self.labeled_var.get():
                tracked_dir = self.workspace.dlc_tracked_videos_dir(stem)
                tracked_dir.mkdir(parents=True, exist_ok=True)
                for labeled in sorted(source.parent.glob(f"{stem}*filtered_labeled.mp4")):
                    target = tracked_dir / labeled.name
                    if labeled.resolve() != target.resolve():
                        copy2(labeled, target)
        log(
            f"[collect] {len(copied)} filtered CSV(s) -> "
            f"{self.workspace.tracking_dir}/<optional session>/Filtered_CSV"
        )
        return f"Tracked {len(inputs)} recording(s); {len(copied)} tracking file(s) collected."

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
