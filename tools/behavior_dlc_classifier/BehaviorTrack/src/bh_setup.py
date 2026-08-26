#!/usr/bin/env python
"""Step 1: choose the recordings and name the sessions.

Nothing here touches a model. It reads a folder of videos, works out which
animal and session each filename refers to, shows the result so it can be
checked before anything expensive runs, and writes ``sessions.csv``.

The table is the point. Filename parsing is guesswork, and guesswork that is
never displayed becomes a silent mislabelling halfway through a cohort.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config_loader import (  # noqa: E402
    load_config,
    resolve_config_path,
    save_session_context,
)
from utils.runner import StepWindow, path_row  # noqa: E402
from utils.session_parser import (  # noqa: E402
    discover_sessions,
    summarise,
    write_sessions,
)
from utils.workspace import load_workspace  # noqa: E402


class SetupWindow(StepWindow):
    step_title = "1. Select videos and name sessions"
    step_subtitle = "Recordings, animal IDs and session types"
    step_blurb = (
        "Point at the folder holding your recordings. Every video is listed below with the animal ID and "
        "session type read from its filename, so you can correct the patterns before anything is analysed. "
        "Videos DeepLabCut itself produced are skipped."
    )
    run_label = "Save sessions.csv"

    def __init__(self, config_path: Path) -> None:
        super().__init__(config_path, geometry="1240x880")

    def initialize_state(self) -> None:
        self.workspace = load_workspace(self.config_path)
        onboarding = self.workspace.section("onboarding")
        self.videos_var = tk.StringVar(value=str(self.workspace.raw_videos))
        self.results_var = tk.StringVar(value=str(self.workspace.results))
        self.pattern_var = tk.StringVar(value=str(onboarding.get("animal_id_pattern", "") or ""))
        self.tokens_var = tk.StringVar(
            value=", ".join(str(t) for t in (onboarding.get("filename_patterns", []) or []))
        )
        self.videotype_var = tk.StringVar(value=self.workspace.videotype)
        self.summary_var = tk.StringVar(value="")

    def after_build(self) -> None:
        self.rescan()

    def results_dir(self) -> Path:
        return Path(self.results_var.get())

    def body(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        path_row(parent, 0, "Video folder", self.videos_var, self.browse_videos)
        path_row(parent, 1, "Results folder", self.results_var, self.browse_results)

        options = ttk.Frame(parent, style="Panel.TFrame")
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)

        ttk.Label(options, text="Animal ID pattern", style="MutedPanel.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(options, textvariable=self.pattern_var).grid(row=0, column=1, sticky="ew", padx=(8, 18), pady=4)
        ttk.Label(options, text="Session", style="MutedPanel.TLabel").grid(row=0, column=2, sticky="w", pady=4)
        ttk.Entry(options, textvariable=self.tokens_var).grid(row=0, column=3, sticky="ew", padx=(8, 18), pady=4)
        ttk.Label(options, text="Video type", style="MutedPanel.TLabel").grid(row=0, column=4, sticky="w", pady=4)
        ttk.Entry(options, textvariable=self.videotype_var, width=8).grid(row=0, column=5, sticky="w", padx=(8, 0), pady=4)

        ttk.Button(options, text="Rescan", command=self.rescan).grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Label(options, textvariable=self.summary_var, style="Panel.TLabel").grid(
            row=1, column=1, columnspan=5, sticky="w", pady=(10, 0), padx=(8, 0)
        )

        table = ttk.Frame(parent, style="Panel.TFrame")
        table.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
        table.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)
        columns = ("video", "animal_id", "session_type", "tracking")
        headings = ("Recording", "Animal", "Session", "Tracking")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", height=12)
        for column, heading in zip(columns, headings):
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=260 if column == "video" else 140, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)

    # ------------------------------------------------------------------ ops #

    def browse_videos(self) -> None:
        chosen = filedialog.askdirectory(title="Folder holding the recordings", initialdir=self.videos_var.get() or ".")
        if chosen:
            self.videos_var.set(chosen)
            self.rescan()

    def browse_results(self) -> None:
        chosen = filedialog.askdirectory(
            title="Folder where BehaviorTrack writes results",
            initialdir=self.results_var.get() or ".",
        )
        if chosen:
            self.results_var.set(chosen)

    def _tokens(self) -> list[str]:
        return [t.strip() for t in self.tokens_var.get().split(",") if t.strip()]

    def current_sessions(self) -> list:
        return discover_sessions(
            Path(self.videos_var.get()),
            self.videotype_var.get(),
            animal_id_pattern=self.pattern_var.get().strip(),
            filename_patterns=self._tokens(),
        )

    def rescan(self) -> None:
        sessions = self.current_sessions()
        self.tree.delete(*self.tree.get_children())
        for session in sessions:
            self.tree.insert(
                "",
                "end",
                values=(
                    session.video,
                    session.animal_id or "-",
                    session.session_type or "-",
                    "yes" if session.has_tracking else "no",
                ),
            )
        counts = summarise(sessions)
        if not counts["videos"]:
            self.summary_var.set("No recordings found in that folder.")
            return
        parts = [
            f"{counts['videos']} recordings",
            f"{counts['animals']} animals",
            f"{counts['tracked']} already tracked",
        ]
        if counts["unnamed"]:
            parts.append(f"{counts['unnamed']} without an animal ID")
        self.summary_var.set(" - ".join(parts))

    def work(self, log) -> str:
        sessions = self.current_sessions()
        if not sessions:
            raise RuntimeError(f"No '*{self.videotype_var.get()}' recordings found in {self.videos_var.get()}")

        results = Path(self.results_var.get()).expanduser().resolve()
        results.mkdir(parents=True, exist_ok=True)
        out = write_sessions(results / "sessions.csv", sessions)
        log(f"[write] {out}")

        counts = summarise(sessions)
        for key, value in counts.items():
            log(f"  {key:<10}{value}")
        if counts["unnamed"]:
            log(
                f"[warn] {counts['unnamed']} recording(s) produced no animal ID. "
                "They are still analysed; only the grouping column is blank."
            )

        context = {
            "raw_videos": str(Path(self.videos_var.get()).resolve()),
            "results": str(results),
            "videotype": self.videotype_var.get(),
            "animal_id_pattern": self.pattern_var.get().strip(),
            "filename_patterns": self._tokens(),
        }
        saved = save_session_context(self.config_path, context)
        log(f"[write] {saved}")
        return f"Saved {counts['videos']} recordings to sessions.csv"

    def _on_finished(self, ok: bool) -> None:
        if ok:
            self.rescan()


def self_test(config_path: Path) -> int:
    """Parse a synthetic folder of filenames without opening a window."""
    import tempfile

    print("[self-test] bh_setup")
    # The shipped config, not load_workspace: that layers the operator's saved
    # Step 1 choices on top, so whichever cohort was last onboarded would
    # decide whether this check passes. The fixtures below match what ships.
    onboarding = load_config(config_path).get("onboarding", {}) or {}
    pattern = str(onboarding.get("animal_id_pattern", "") or "")
    tokens = onboarding.get("filename_patterns", []) or []

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        names = [
            "Trial 1_Hab_Animal3.mp4",
            "Trial 2_Cond_Animal3.mp4",
            "Trial 3_Test_Animal4.mp4",
            "Trial 3_Test_Animal4DLC_resnet50_Freezing.mp4",  # DLC output, must be skipped
            "notes.txt",
        ]
        for name in names:
            (folder / name).write_bytes(b"")
        (folder / "Trial 1_Hab_Animal3DLC_resnet50_Freezingshuffle1_100000filtered.csv").write_text("x", encoding="utf-8")

        sessions = discover_sessions(folder, ".mp4", animal_id_pattern=pattern, filename_patterns=tokens)
        counts = summarise(sessions)
        for session in sessions:
            print(f"  {session.video:<46}{session.animal_id or '-':<8}{session.session_type or '-':<8}{'tracked' if session.has_tracking else ''}")
        checks = [
            ("3 recordings found (DLC output skipped)", counts["videos"] == 3),
            ("2 animals parsed", counts["animals"] == 2),
            ("1 already tracked", counts["tracked"] == 1),
            ("session types parsed", {s.session_type for s in sessions} == {"Hab", "Cond", "Test"}),
        ]
        failed = [name for name, ok in checks if not ok]
        for name, ok in checks:
            print(f"  {'OK  ' if ok else 'FAIL'} {name}")

        out = write_sessions(folder / "sessions.csv", sessions)
        print(f"  wrote {out.name} ({out.stat().st_size} bytes)")

    if failed:
        print(f"[self-test] FAILED: {len(failed)} check(s)")
        return 1
    print("[self-test] PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BehaviorTrack step 1: sessions")
    parser.add_argument("--config", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    config_path = resolve_config_path(args.config)
    if args.self_test:
        return self_test(config_path)
    SetupWindow(config_path).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
