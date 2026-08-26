#!/usr/bin/env python
"""Main BehaviorTrack GUI launcher.

The front door. Each numbered step opens in its own window, exactly as
BarnesTrack does, and the right-hand column reports what the workspace currently
holds so a step is never run against a folder that is not ready for it.

Where BarnesTrack computes maze metrics, step 3 here computes the feature sets
and classifies every frame into the seven behaviors.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config_loader import resolve_config_path  # noqa: E402
from utils.steps import STEPS, launch_step  # noqa: E402
from utils.ui_style import (  # noqa: E402
    apply_dark_theme,
    behavior_legend,
    open_in_file_explorer,
)
from utils.workspace import Workspace, load_workspace  # noqa: E402

REFRESH_MS = 2000



def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class BehaviorTrackApp:
    """Front door for the BehaviorTrack workflow."""

    def __init__(self, root: tk.Tk, config_path: Path) -> None:
        self.root = root
        self.config_path = config_path
        self.workspace: Workspace = load_workspace(config_path)
        self.started_at = time.time()
        self.status_vars: dict[str, tk.StringVar] = {}
        self.status_labels: dict[str, ttk.Label] = {}
        self.step_status: dict[str, tk.StringVar] = {}
        self.step_status_labels: dict[str, ttk.Label] = {}
        self.step_buttons: dict[str, ttk.Button] = {}
        self.workspace_var = tk.StringVar(value="")

        root.title("BehaviorTrack")
        root.geometry("1440x860")
        root.minsize(1180, 720)
        apply_dark_theme(root)
        self._build()
        self.refresh()
        self._schedule_refresh()

    # ---------------------------------------------------------------- build #

    def _build(self) -> None:
        shell = ttk.Frame(self.root, padding=18)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(3, weight=1)

        ttk.Label(shell, text="BehaviorTrack", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            shell,
            text="DeepLabCut behavior classification workflow",
            style="Accent.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 2))
        ttk.Label(
            shell,
            text=(
                "Start with only videos, then move through naming, DeepLabCut tracking, feature extraction "
                "and behavior classification, summaries, and independent manual validation. The archived "
                "feature/model contract is reproducible; target-rig accuracy is unknown until step 5 passes."
            ),
            style="Muted.TLabel",
            wraplength=1000,
            justify="left",
        ).grid(row=2, column=0, sticky="w", pady=(0, 14))

        body = ttk.Frame(shell)
        body.grid(row=3, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1, uniform="cols")
        body.columnconfigure(1, weight=1, uniform="cols")
        body.rowconfigure(0, weight=1)

        self._build_steps(body)
        self._build_status(body)

    def _build_steps(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Workflow windows", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            left,
            text="Run the steps in order. Each one opens in its own window.",
            style="MutedPanel.TLabel",
            wraplength=460,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 14))

        for index, (key, title, blurb, script) in enumerate(STEPS):
            self._step_button(left, index + 2, key, title, blurb, script)

        row = len(STEPS) + 2
        left.rowconfigure(row, weight=1)
        buttons = ttk.Frame(left, style="Panel.TFrame")
        buttons.grid(row=row + 1, column=0, sticky="sw", pady=(14, 0))
        ttk.Button(buttons, text="Open video folder", command=self.open_videos).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Open results folder", command=self.open_results).grid(row=0, column=1)

    def _step_button(self, parent: ttk.Frame, row: int, key: str, title: str, blurb: str, script: str) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        card.columnconfigure(0, weight=1)

        button = ttk.Button(
            card,
            text=title,
            style="Accent.TButton",
            command=lambda s=script: self.launch(s),
        )
        button.grid(row=0, column=0, sticky="ew")
        self.step_buttons[key] = button
        ttk.Label(card, text=blurb, style="MutedCard.TLabel", wraplength=440, justify="left").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        var = tk.StringVar(value="")
        label = ttk.Label(card, textvariable=var, style="MutedCard.TLabel")
        label.grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.step_status[key] = var
        self.step_status_labels[key] = label

    def _build_status(self, parent: ttk.Frame) -> None:
        right = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)

        ttk.Label(right, text="Workspace", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            right,
            textvariable=self.workspace_var,
            style="MutedPanel.TLabel",
            wraplength=460,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 12))

        card = ttk.Frame(right, style="Card.TFrame", padding=12)
        card.grid(row=2, column=0, sticky="ew")
        card.columnconfigure(1, weight=1)
        rows = [
            ("videos", "Recordings found"),
            ("tracked", "With DLC tracking"),
            ("classified", "Classified"),
            ("animals", "Animals named"),
            ("behavior_model", "Unified behavior model"),
            ("dlc_network", "DLC network"),
            ("models", "Models"),
        ]
        for index, (key, label_text) in enumerate(rows):
            ttk.Label(card, text=label_text, style="MutedCard.TLabel").grid(row=index, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="-")
            value_label = ttk.Label(card, textvariable=var, style="Card.TLabel")
            value_label.grid(row=index, column=1, sticky="e", pady=2)
            self.status_vars[key] = var
            self.status_labels[key] = value_label

        legend_card = ttk.Frame(right, style="Card.TFrame", padding=12)
        legend_card.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(legend_card, text="Behavior classes", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        behavior_legend(legend_card).grid(row=1, column=0, sticky="w", pady=(8, 0))

        notes = ttk.Frame(right, style="Panel.TFrame")
        notes.grid(row=4, column=0, sticky="nsew", pady=(12, 0))
        notes.columnconfigure(0, weight=1)
        self.notes_var = tk.StringVar(value="")
        ttk.Label(
            notes,
            textvariable=self.notes_var,
            style="WarnPanel.TLabel",
            wraplength=460,
            justify="left",
        ).grid(row=0, column=0, sticky="nw")

        self.uptime_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.uptime_var, style="MutedPanel.TLabel").grid(
            row=5, column=0, sticky="sw", pady=(10, 0)
        )

    # --------------------------------------------------------------- actions #

    def launch(self, script: str) -> None:
        try:
            launch_step(script, self.config_path)
        except FileNotFoundError as exc:
            messagebox.showerror("BehaviorTrack", f"Step not found:\n{exc}")
        except OSError as exc:
            messagebox.showerror("BehaviorTrack", f"Could not start {script}:\n{exc}")

    def open_videos(self) -> None:
        open_in_file_explorer(self.workspace.raw_videos)

    def open_results(self) -> None:
        open_in_file_explorer(self.workspace.results)

    # -------------------------------------------------------------- refresh #

    def _schedule_refresh(self) -> None:
        self.root.after(REFRESH_MS, self._tick)

    def _tick(self) -> None:
        self.refresh()
        self._schedule_refresh()

    def refresh(self) -> None:
        try:
            self.workspace = load_workspace(self.config_path)
            status = self.workspace.status()
        except Exception as exc:
            self.notes_var.set(f"Could not read the workspace: {exc}")
            return

        models = self.workspace.models() if not status["models_missing"] else None
        self.status_vars["videos"].set(str(status["videos"]))
        self.status_vars["tracked"].set(f"{status['tracked']} of {status['videos']}")
        self.status_vars["classified"].set(f"{status['classified']} of {status['videos']}")
        self.status_vars["animals"].set(str(status["animals"]))
        self.status_vars["behavior_model"].set(status["behavior_model"])
        self.status_vars["dlc_network"].set(
            models.dlc_config.parent.name if models else "unresolved"
        )
        missing = status["models_missing"]
        self.status_vars["models"].set("all present" if not missing else f"{len(missing)} missing")
        self.workspace_var.set(
            f"Videos: {self.workspace.raw_videos}\nResults: {self.workspace.results}"
        )

        self._tone("tracked", status["tracked"], status["videos"])
        self._tone("classified", status["classified"], status["videos"])
        self.status_labels["models"].configure(style="GoodCard.TLabel" if not missing else "BadCard.TLabel")

        self.step_status["setup"].set(
            f"sessions.csv saved - {status['videos']} recordings"
            if status["sessions_saved"]
            else "not yet saved"
        )
        self.step_status["track"].set(f"{status['untracked']} still to track" if status["untracked"] else "all tracked")
        self.step_status["classify"].set(
            f"{status['unclassified']} still to classify" if status["unclassified"] else "all classified"
        )
        self.step_status["results"].set(
            "workbook written" if self.workspace.summary_xlsx.is_file() else "no workbook yet"
        )
        validation_report = self.workspace.results / "Validation" / "Reports" / "validation_report.json"
        self.step_status["model"].set(
            "report written" if validation_report.is_file() else "target validation pending"
        )

        # A finished step goes green, so the workflow can be read at a glance
        # without parsing five status lines. "All tracked" with nothing to
        # track is not done, it is empty, so each count-based step also
        # requires that recordings exist.
        completed = {
            "setup": bool(status["sessions_saved"]),
            "track": bool(status["videos"]) and not status["untracked"],
            "classify": bool(status["videos"]) and not status["unclassified"],
            "results": self.workspace.summary_xlsx.is_file(),
            "model": validation_report.is_file(),
        }
        for key, button in self.step_buttons.items():
            button.configure(style="Good.TButton" if completed.get(key) else "Accent.TButton")

        notes: list[str] = []
        if not status["raw_videos_exists"]:
            notes.append(f"The video folder does not exist yet: {self.workspace.raw_videos}")
        elif status["videos"] == 0:
            notes.append("No recordings found. Put videos in the video folder, then run step 1.")
        if missing:
            notes.append("Missing: " + ", ".join(missing))
        if status["videos"]:
            notes.append(
                "The frozen 719-feature model has no bundled animal/session-held-out validation on this rig. "
                "Treat outputs as predictions until independent manual labels are evaluated in step 5."
            )
        self.notes_var.set("\n\n".join(notes))
        self.uptime_var.set(f"Open for {format_duration(time.time() - self.started_at)}")

    def _tone(self, key: str, done: int, total: int) -> None:
        if total and done == total:
            style = "GoodCard.TLabel"
        elif done:
            style = "WarnCard.TLabel"
        else:
            style = "MutedCard.TLabel"
        self.status_labels[key].configure(style=style)


# ------------------------------------------------------------------ self-test #

def self_test(config_path: Path) -> int:
    """Check the workspace resolves and the status computation runs, no window."""
    print("[self-test] bh_app")
    workspace = load_workspace(config_path)
    print(f"  config      {config_path}")
    print(f"  workspace   {workspace.root}")
    print(f"  tool root   {workspace.tool_root}")
    status = workspace.status()
    for key in ("videos", "tracked", "classified", "sessions_saved"):
        print(f"  {key:<12}{status[key]}")
    missing = status["models_missing"]
    print(f"  models      {'all present' if not missing else ', '.join(missing)}")
    for _, title, _, script in STEPS:
        path = Path(__file__).resolve().parent / script
        mark = "OK " if path.is_file() else "MISSING"
        print(f"  {mark} {script:<18}{title}")
    absent = [s for *_, s in STEPS if not (Path(__file__).resolve().parent / s).is_file()]
    if absent:
        print(f"[self-test] FAILED: {len(absent)} step script(s) missing")
        return 1
    if missing:
        print("[self-test] PASSED with warnings (models missing)")
        return 0
    print("[self-test] PASSED")
    return 0


def ui_smoke_test(config_path: Path) -> int:
    """Construct every real window and close it without running any analysis."""
    from bh_features import FeaturesWindow
    from bh_model import ModelWindow
    from bh_results import ResultsWindow
    from bh_setup import SetupWindow
    from bh_track import TrackWindow

    print("[ui-smoke-test] BehaviorTrack")
    root = tk.Tk()
    root.withdraw()
    BehaviorTrackApp(root, config_path)
    root.update_idletasks()
    root.destroy()
    print("  OK  BehaviorTrackApp")

    for window_type in (SetupWindow, TrackWindow, FeaturesWindow, ResultsWindow, ModelWindow):
        window = window_type(config_path)
        window.withdraw()
        window.update_idletasks()
        window.destroy()
        print(f"  OK  {window_type.__name__}")
    print("[ui-smoke-test] PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BehaviorTrack launcher")
    parser.add_argument("--config", default=None, help="Path to the BehaviorTrack config")
    parser.add_argument("--self-test", action="store_true", help="Check wiring and exit")
    parser.add_argument(
        "--ui-smoke-test",
        action="store_true",
        help="Construct every window once, close it, and exit",
    )
    args = parser.parse_args(argv)

    config_path = resolve_config_path(args.config)
    if args.self_test:
        return self_test(config_path)
    if args.ui_smoke_test:
        return ui_smoke_test(config_path)

    root = tk.Tk()
    BehaviorTrackApp(root, config_path)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
