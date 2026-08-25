"""A window that runs one long job and shows its log.

Four of the five steps have the same shape: a form at the top, a run button, and
a log that fills while a worker thread does the work. Written once here, because
the previous generation of this tool had the same pattern copied into three
modules and two of the copies had already drifted.

The worker never touches a widget. It appends to a queue, and the UI drains that
queue on a timer — Tkinter is not thread-safe, and a background thread calling
``insert`` directly is the classic way to get a GUI that freezes on some machines
and not others.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable

from .ui_style import COLORS, apply_dark_theme, open_in_file_explorer

POLL_MS = 100


class StepWindow(tk.Tk):
    """Base window for one workflow step."""

    #: Overridden by subclasses.
    step_title = "Step"
    step_subtitle = ""
    step_blurb = ""
    run_label = "Run"

    def __init__(self, config_path: Path, *, geometry: str = "1180x820") -> None:
        super().__init__()
        self.config_path = Path(config_path)
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()

        self.title(f"BehaviorTrack - {self.step_title}")
        self.geometry(geometry)
        self.minsize(980, 640)
        apply_dark_theme(self)

        # Tk variables need a live Tcl interpreter. Subclasses initialise
        # their state here, after Tk.__init__ but before body() builds widgets.
        self.initialize_state()
        self.status_var = tk.StringVar(value="Ready.")
        self._build_shell()
        self.after_build()
        self.after(POLL_MS, self._drain)

    # ----------------------------------------------------------------- build #

    def _build_shell(self) -> None:
        shell = ttk.Frame(self, padding=18)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(4, weight=1)

        ttk.Label(shell, text=self.step_title, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        if self.step_subtitle:
            ttk.Label(shell, text=self.step_subtitle, style="Accent.TLabel").grid(
                row=1, column=0, sticky="w", pady=(4, 2)
            )
        if self.step_blurb:
            ttk.Label(
                shell,
                text=self.step_blurb,
                style="Muted.TLabel",
                wraplength=1040,
                justify="left",
            ).grid(row=2, column=0, sticky="w", pady=(0, 14))

        form = ttk.Frame(shell, style="Panel.TFrame", padding=14)
        form.grid(row=3, column=0, sticky="ew")
        form.columnconfigure(0, weight=1)
        self.body(form)

        log_panel = ttk.Frame(shell, style="Panel.TFrame", padding=14)
        log_panel.grid(row=4, column=0, sticky="nsew", pady=(12, 0))
        log_panel.columnconfigure(0, weight=1)
        log_panel.rowconfigure(1, weight=1)
        ttk.Label(log_panel, text="Progress", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.log_widget = ScrolledText(
            log_panel,
            height=16,
            bg=COLORS["entry"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
            relief="flat",
            font=("Consolas", 9),
            wrap="word",
        )
        self.log_widget.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.log_widget.configure(state="disabled")

        footer = ttk.Frame(shell)
        footer.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        footer.columnconfigure(1, weight=1)
        self.run_button = ttk.Button(footer, text=self.run_label, style="Accent.TButton", command=self.start)
        self.run_button.grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(footer, mode="determinate", style="Horizontal.TProgressbar")
        self.progress.grid(row=0, column=1, sticky="ew", padx=12)
        ttk.Button(footer, text="Open results", command=self.open_results).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(footer, text="Close", command=self.destroy).grid(row=0, column=3)
        ttk.Label(shell, textvariable=self.status_var, style="Muted.TLabel").grid(
            row=6, column=0, sticky="w", pady=(8, 0)
        )

    # ------------------------------------------------------- subclass hooks #

    def initialize_state(self) -> None:
        """Create Tk variables and resolve initial state before widgets exist."""

    def after_build(self) -> None:
        """Populate tables or summaries after subclass widgets exist."""

    def body(self, parent: ttk.Frame) -> None:
        """Build the step's own form. Subclasses override."""

    def work(self, log: Callable[[str], None]) -> str:
        """Do the step's work on a worker thread. Subclasses override."""
        raise NotImplementedError

    def results_dir(self) -> Path:
        return self.config_path.parent

    # -------------------------------------------------------------- running #

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self.log_widget.configure(state="normal")
        self.log_widget.delete("1.0", "end")
        self.log_widget.configure(state="disabled")
        self._cancel.clear()
        self.run_button.configure(state="disabled")
        self.status_var.set("Working...")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)

        def target() -> None:
            try:
                message = self.work(self.log)
                self._queue.put(("done", message or "Finished."))
            except Exception as exc:  # surfaced in the log, not a silent stop
                self._queue.put(("log", traceback.format_exc().rstrip()))
                self._queue.put(("failed", str(exc)))

        self._worker = threading.Thread(target=target, daemon=True)
        self._worker.start()

    def log(self, message: str) -> None:
        """Thread-safe: append to the queue the UI drains."""
        self._queue.put(("log", str(message)))

    def set_progress(self, done: int, total: int) -> None:
        self._queue.put(("progress", (done, total)))

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self.log_widget.configure(state="normal")
                    self.log_widget.insert("end", f"{payload}\n")
                    self.log_widget.see("end")
                    self.log_widget.configure(state="disabled")
                elif kind == "progress":
                    done, total = payload  # type: ignore[misc]
                    self.progress.stop()
                    self.progress.configure(mode="determinate", maximum=max(1, total), value=done)
                elif kind in {"done", "failed"}:
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=0)
                    self.run_button.configure(state="normal")
                    self.status_var.set(str(payload))
                    self._on_finished(kind == "done")
        except queue.Empty:
            pass
        self.after(POLL_MS, self._drain)

    def _on_finished(self, ok: bool) -> None:
        """Hook for subclasses that refresh a table when the work completes."""

    def open_results(self) -> None:
        open_in_file_explorer(self.results_dir())


def path_row(
    parent: ttk.Frame,
    row: int,
    label: str,
    variable: tk.StringVar,
    on_browse: Callable[[], None],
    *,
    style_base: str = "Panel",
) -> None:
    """One 'label / entry / Browse' line, laid out the same everywhere."""
    ttk.Label(parent, text=label, style=f"Muted{style_base}.TLabel").grid(row=row, column=0, sticky="w", pady=4)
    entry = ttk.Entry(parent, textvariable=variable)
    entry.grid(row=row, column=1, sticky="ew", padx=8, pady=4)
    ttk.Button(parent, text="Browse", command=on_browse).grid(row=row, column=2, sticky="e", pady=4)
    parent.columnconfigure(1, weight=1)


def field_row(
    parent: ttk.Frame,
    row: int,
    column: int,
    label: str,
    variable: tk.StringVar,
    *,
    width: int = 10,
    style_base: str = "Panel",
) -> None:
    """One small labelled entry, for numeric settings."""
    ttk.Label(parent, text=label, style=f"Muted{style_base}.TLabel").grid(
        row=row, column=column, sticky="w", padx=(0, 6), pady=4
    )
    ttk.Entry(parent, textvariable=variable, width=width).grid(
        row=row, column=column + 1, sticky="w", padx=(0, 18), pady=4
    )
