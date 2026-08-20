from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Optional

from .behavior import BEHAVIOR_LABEL_ENCODER_PATH, BEHAVIOR_MODEL_PATH
from .config import default_model_path
from .dlc_stage import DEFAULT_DLC_CONFIG_PATH as BUNDLED_DLC_CONFIG_PATH
from .dlc_stage import list_videos
from .friendly_pipeline import FriendlyPipelineSettings, _looks_like_raw_video, run_friendly_pipeline
from .label_video import annotate_video_labels
from .train import _match_training_pairs, train_model


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = default_model_path()
DEFAULT_DLC_CONFIG_PATH = BUNDLED_DLC_CONFIG_PATH
DEFAULT_BEHAVIOR_MODEL_PATH = BEHAVIOR_MODEL_PATH
DEFAULT_BEHAVIOR_LABEL_ENCODER_PATH = BEHAVIOR_LABEL_ENCODER_PATH
COLORS = {
    "bg": "#101318",
    "panel": "#171c24",
    "panel_2": "#202633",
    "text": "#eef2f7",
    "muted": "#9aa6b5",
    "accent": "#6ee7b7",
    "accent_dark": "#0f766e",
    "field": "#0d1117",
    "border": "#2b3443",
    "error": "#fca5a5",
}
#: One spacing scale for the whole window, so panels, rows and fields line up
#: instead of each carrying its own hand-picked padding.
SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 18, "xl": 24}
#: Point sizes. Tk scales points by the screen's real DPI once the process is
#: DPI aware, so these stay correct on a 150% display.
TYPE = {"title": 21, "body": 11, "caption": 10, "mono": 10}


def _default_path(path: Path) -> str:
    if path.exists():
        return str(path.resolve())
    return ""


def _enable_dpi_awareness() -> None:
    """Ask Windows for real pixels before Tk measures anything.

    Without this the OS stretches the whole window on a scaled display, which
    is why the text looked soft on high-resolution laptops. Silently ignored
    everywhere else, and on Windows versions that predate the call.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
    except Exception:
        pass


def open_in_file_manager(path: Path) -> None:
    """Reveal a folder in the desktop file manager, on any supported platform."""
    target = str(path)
    if sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]  # Windows only
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    try:
        subprocess.Popen([opener, target])
    except FileNotFoundError as exc:
        raise OSError(f"No desktop file manager found (tried '{opener}').") from exc


class FreezingDlcApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DLC Behavior Pipeline")
        self.configure(bg=COLORS["bg"])

        self.log_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._busy_count = 0
        self.worker: threading.Thread | None = None
        self.manual_worker: threading.Thread | None = None
        self.refine_training_worker: threading.Thread | None = None
        self.last_output_dir: Path | None = None
        self.refine_videos: list[Path] = []
        self.refine_next_index = 0
        self.refine_labels_dir: Optional[Path] = None
        self.refine_dlc_dir: Optional[Path] = None

        self.project_dir = tk.StringVar()
        self.model_path = tk.StringVar(value=_default_path(DEFAULT_MODEL_PATH))
        self.dlc_config_path = tk.StringVar(value=_default_path(DEFAULT_DLC_CONFIG_PATH))
        self.behavior_model_path = tk.StringVar(value=_default_path(DEFAULT_BEHAVIOR_MODEL_PATH))
        self.behavior_label_encoder_path = tk.StringVar(value=_default_path(DEFAULT_BEHAVIOR_LABEL_ENCODER_PATH))
        self.videotype = tk.StringVar(value=".mp4")
        self.threshold = tk.StringVar(value="")
        self.dlc_env = tk.StringVar(value="")
        self.fps = tk.StringVar(value="25")
        self.bin_sec = tk.StringVar(value="30")
        self.box_side_cm = tk.StringVar(value="17")
        self.box_side_pixels = tk.StringVar(value="170")
        self.run_clahe = tk.BooleanVar(value=True)
        self.run_dlc = tk.BooleanVar(value=True)
        self.make_tracked_videos = tk.BooleanVar(value=True)
        self.run_behavior_analysis = tk.BooleanVar(value=True)

        self._setup_style()
        self._build_ui()
        self._apply_window_icon()
        self._size_to_content()
        self.after(100, self._drain_log_queue)

    def _apply_window_icon(self) -> None:
        """Paint a small ethogram mark, rather than leave Tk's default feather."""
        size = 32
        icon = tk.PhotoImage(width=size, height=size)
        icon.put(COLORS["panel"], to=(0, 0, size, size))
        for x in (4, 8, 9, 14, 18, 19, 20, 25):
            icon.put(COLORS["accent"], to=(x, 7, x + 2, 25))
        self._icon = icon  # a PhotoImage is dropped as soon as nothing holds it
        try:
            self.iconphoto(True, icon)
        except tk.TclError:
            pass

    def _size_to_content(self) -> None:
        """Open just tall enough to show the progress log, but never off-screen.

        A fixed default height hid the log on the machines this was written on,
        which is the one widget a user watches during a long run. Anything the
        screen cannot fit is reachable by scrolling instead of lost.
        """
        self.update_idletasks()
        width = min(max(1040, self._body.winfo_reqwidth() + 20), self.winfo_screenwidth() - 80)
        height = min(self._body.winfo_reqheight(), self.winfo_screenheight() - 120)
        self.geometry(f"{width}x{height}")
        # Width has a hard floor because nothing scrolls sideways; height does
        # not, because the body scrolls.
        self.minsize(width, min(560, height))

    def _setup_style(self) -> None:
        # Points only mean the right thing once Tk knows the screen's real DPI.
        self.tk.call("tk", "scaling", self.winfo_fpixels("1i") / 72.0)

        body = ("Segoe UI", TYPE["body"])
        caption = ("Segoe UI", TYPE["caption"])
        heading = ("Segoe UI", TYPE["caption"], "bold")

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=body, background=COLORS["bg"], foreground=COLORS["text"])
        style.configure("App.TFrame", background=COLORS["bg"])
        style.configure("Panel.TFrame", background=COLORS["panel"])

        # Captions are quieter and smaller than the values they name, so a row
        # reads value-first instead of as two competing pieces of text.
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=body)
        style.configure("Field.TLabel", background=COLORS["panel"], foreground=COLORS["muted"], font=caption)
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["muted"], font=caption)
        style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=("Segoe UI Semibold", TYPE["title"]))
        style.configure("Muted.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=body)

        style.configure(
            "TLabelframe",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            relief="solid",
            borderwidth=1,
        )
        style.configure("TLabelframe.Label", background=COLORS["bg"], foreground=COLORS["accent"], font=heading)

        # ttk does not inherit the root font into an entry's own text on
        # Windows, which is why the values used to render larger than their
        # captions. Setting it here is what puts them on one scale.
        style.configure(
            "TEntry",
            font=body,
            fieldbackground=COLORS["field"],
            foreground=COLORS["text"],
            insertcolor=COLORS["accent"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["border"],
            darkcolor=COLORS["border"],
            padding=(SPACE["sm"], SPACE["xs"]),
        )
        style.map(
            "TEntry",
            bordercolor=[("focus", COLORS["accent"])],
            lightcolor=[("focus", COLORS["accent"])],
            darkcolor=[("focus", COLORS["accent"])],
            fieldbackground=[("disabled", COLORS["panel"])],
        )

        style.configure(
            "TButton",
            font=body,
            background=COLORS["panel_2"],
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            focuscolor=COLORS["accent"],
            padding=(SPACE["md"], SPACE["sm"]),
        )
        style.map(
            "TButton",
            background=[("active", COLORS["accent_dark"]), ("disabled", COLORS["panel"])],
            foreground=[("active", COLORS["text"]), ("disabled", COLORS["border"])],
            bordercolor=[("active", COLORS["accent_dark"])],
        )
        style.configure(
            "Accent.TButton",
            font=("Segoe UI Semibold", TYPE["body"]),
            background=COLORS["accent_dark"],
            foreground=COLORS["text"],
            bordercolor=COLORS["accent"],
            focuscolor=COLORS["text"],
            padding=(SPACE["lg"], SPACE["sm"]),
        )
        style.map(
            "Accent.TButton",
            background=[("active", COLORS["accent"]), ("disabled", COLORS["panel_2"])],
            foreground=[("active", COLORS["field"]), ("disabled", COLORS["muted"])],
            bordercolor=[("disabled", COLORS["border"])],
        )

        # A ticked box fills solid with the accent, so "on" is legible at a
        # glance instead of resting on a small mark alone. Every colour here has
        # to be set through style.map: clam's own map beats a plain configure,
        # which is why the indicator ignored these settings when they were.
        style.configure(
            "Option.TCheckbutton",
            font=body,
            background=COLORS["panel"],
            foreground=COLORS["text"],
            focuscolor=COLORS["accent"],
            indicatorbackground=COLORS["field"],
            indicatorforeground=COLORS["accent"],
            indicatorsize=15,
            indicatormargin=(0, 0, SPACE["sm"], 0),
            upperbordercolor=COLORS["border"],
            lowerbordercolor=COLORS["border"],
            padding=(0, SPACE["xs"]),
        )
        style.map(
            "Option.TCheckbutton",
            background=[("active", COLORS["panel"])],
            foreground=[("active", COLORS["text"])],
            indicatorbackground=[("selected", COLORS["accent"]), ("!selected", COLORS["field"])],
            indicatorforeground=[("selected", COLORS["field"])],
            upperbordercolor=[("selected", COLORS["accent"]), ("active", COLORS["muted"])],
            lowerbordercolor=[("selected", COLORS["accent"]), ("active", COLORS["muted"])],
        )

        style.configure(
            "Vertical.TScrollbar",
            background=COLORS["panel_2"],
            troughcolor=COLORS["bg"],
            bordercolor=COLORS["bg"],
            arrowcolor=COLORS["muted"],
            lightcolor=COLORS["panel_2"],
            darkcolor=COLORS["panel_2"],
            arrowsize=13,
        )
        style.map(
            "Vertical.TScrollbar",
            background=[("active", COLORS["accent_dark"])],
            arrowcolor=[("active", COLORS["text"])],
        )

        style.configure(
            "Horizontal.TProgressbar",
            background=COLORS["accent_dark"],
            troughcolor=COLORS["field"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["accent_dark"],
            darkcolor=COLORS["accent_dark"],
            thickness=6,
        )

    def _scrollable_body(self) -> ttk.Frame:
        """Put the whole form in a scroller, so a short screen hides nothing.

        The form is taller than a 768-pixel laptop can display. Without this the
        controls at the bottom, Run included, simply fell off the window edge.
        The scrollbar appears only when the content really does not fit.
        """
        outer = ttk.Frame(self, style="App.TFrame")
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, bg=COLORS["bg"], highlightthickness=0, takefocus=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        bar.grid(row=0, column=1, sticky="ns")
        bar.grid_remove()
        canvas.configure(yscrollcommand=bar.set)

        body = ttk.Frame(canvas, style="App.TFrame")
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        applied: dict[str, int] = {}

        def resize(_event: Any = None) -> None:
            width, view = canvas.winfo_width(), canvas.winfo_height()
            # Stretch to the viewport when there is room, so the log still grows
            # with the window; fall back to the form's own height when there is not.
            height = max(body.winfo_reqheight(), view)
            if applied.get("w") == width and applied.get("h") == height:
                return  # itemconfigure re-fires <Configure>; this stops the loop
            applied.update(w=width, h=height)
            canvas.itemconfigure(window, width=width, height=height)
            canvas.configure(scrollregion=(0, 0, width, height))
            if height > view:
                bar.grid()
            else:
                bar.grid_remove()
                canvas.yview_moveto(0.0)

        canvas.bind("<Configure>", resize)
        body.bind("<Configure>", resize)
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.bind_all(sequence, lambda event, c=canvas: self._on_wheel(c, event))
        self._body = body
        return body

    def _on_wheel(self, canvas: tk.Canvas, event: Any) -> None:
        """Scroll the form, unless the pointer is over the log, which scrolls itself."""
        widget = self.winfo_containing(event.x_root, event.y_root)
        log = getattr(self, "log_text", None)
        while widget is not None:
            if widget is log:
                return
            widget = getattr(widget, "master", None)
        if getattr(event, "num", 0) in (4, 5):  # X11 reports the wheel as buttons
            steps = 1 if event.num == 5 else -1
        elif event.delta:
            # Windows counts in 120s; macOS trackpads report small single digits.
            steps = -int(event.delta / 120) or (-1 if event.delta > 0 else 1)
        else:
            return
        canvas.yview_scroll(steps, "units")

    def _build_ui(self) -> None:
        gap, edge = SPACE["sm"], SPACE["xl"]
        root = self._scrollable_body()

        header = ttk.Label(root, text="DLC Behavior Pipeline", style="Title.TLabel")
        header.pack(anchor="w", padx=edge, pady=(SPACE["xl"], 2))
        subtitle = ttk.Label(
            root,
            text="Choose a folder of videos. This tracks the animals, scores freezing and seven behaviors,\n"
            "and writes a results package beside the videos.",
            style="Muted.TLabel",
            justify="left",
        )
        subtitle.pack(anchor="w", padx=edge, pady=(0, SPACE["lg"]))

        form = ttk.LabelFrame(root, text="Videos and models")
        form.pack(fill="x", padx=edge, pady=(0, SPACE["md"]))
        # The one field a user has to fill, so the caret starts in it.
        self._path_row(form, 0, "Folder with your videos", self.project_dir, self._choose_project_dir).focus_set()
        self._path_row(form, 1, "Freezing model", self.model_path, self._choose_model)
        self._path_row(form, 2, "DeepLabCut config", self.dlc_config_path, self._choose_dlc_config)
        self._path_row(form, 3, "7-behavior model", self.behavior_model_path, self._choose_behavior_model)
        self._path_row(form, 4, "7-behavior class names", self.behavior_label_encoder_path, self._choose_behavior_labels)
        ttk.Label(
            form,
            text="Only the top row needs choosing. The three models below ship with the app and are already loaded.",
            style="Panel.TLabel",
        ).grid(row=5, column=0, columnspan=3, sticky="w", padx=SPACE["md"], pady=(SPACE["xs"], SPACE["md"]))

        options = ttk.LabelFrame(root, text="Analysis settings")
        options.pack(fill="x", padx=edge, pady=(0, SPACE["md"]))
        self._entry(options, 0, 0, "Video type", self.videotype, width=8)
        self._entry(options, 0, 2, "Freezing threshold", self.threshold, width=8, hint="blank = the model's own")
        self._entry(options, 0, 4, "Frames per second", self.fps, width=8)
        self._entry(options, 1, 0, "Time bin", self.bin_sec, width=8, hint="seconds")
        self._entry(options, 1, 2, "Arena side", self.box_side_cm, width=8, hint="cm")
        self._entry(options, 1, 4, "Arena side", self.box_side_pixels, width=8, hint="pixels")
        self._entry(options, 2, 0, "DeepLabCut environment", self.dlc_env, width=16, hint="blank = the bundled .venv-dlc")
        # minsize keeps the fields from being squeezed out of existence when the
        # window is narrower than the row would like to be.
        for column in (1, 3, 5):
            options.columnconfigure(column, weight=1, minsize=110)

        checks = ttk.Frame(options, style="Panel.TFrame")
        checks.grid(row=3, column=0, columnspan=6, sticky="w", padx=SPACE["md"], pady=(SPACE["md"], SPACE["md"]))
        box = {"style": "Option.TCheckbutton"}
        ttk.Checkbutton(checks, text="CLAHE contrast preprocessing", variable=self.run_clahe, **box).grid(row=0, column=0, sticky="w", padx=(0, SPACE["xl"]))
        ttk.Checkbutton(checks, text="Run DeepLabCut on raw videos", variable=self.run_dlc, **box).grid(row=0, column=1, sticky="w", padx=(0, SPACE["xl"]))
        ttk.Checkbutton(checks, text="Save tracked videos", variable=self.make_tracked_videos, **box).grid(row=1, column=0, sticky="w", padx=(0, SPACE["xl"]), pady=(SPACE["sm"], 0))
        ttk.Checkbutton(checks, text="Run 7-behavior analysis", variable=self.run_behavior_analysis, **box).grid(row=1, column=1, sticky="w", pady=(SPACE["sm"], 0))

        actions = ttk.Frame(root, style="App.TFrame")
        actions.pack(fill="x", padx=edge, pady=(SPACE["xs"], SPACE["lg"]))
        self.run_button = ttk.Button(actions, text="Run full analysis", command=self._start_run, style="Accent.TButton")
        self.run_button.pack(side="left")
        ttk.Button(actions, text="Open output folder", command=self._open_output_folder).pack(side="left", padx=gap)

        troubleshoot = ttk.LabelFrame(root, text="If the freezing results look wrong")
        troubleshoot.pack(fill="x", padx=edge, pady=(0, SPACE["md"]))
        ttk.Label(
            troubleshoot,
            text="Score a few videos by hand and the app will train a freezing model on them, then load it above.",
            style="Panel.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=SPACE["md"], pady=(SPACE["md"], 2))
        self.refine_button = ttk.Button(
            troubleshoot,
            text="Label videos and create refined model",
            command=self._start_refinement_wizard,
        )
        self.refine_button.grid(row=1, column=0, sticky="w", padx=SPACE["md"], pady=(SPACE["sm"], SPACE["md"]))
        ttk.Label(
            troubleshoot,
            text="In the labeler: F starts a freezing bout, F again ends it, S saves.",
            style="Panel.TLabel",
        ).grid(row=1, column=1, sticky="w", padx=SPACE["md"], pady=(SPACE["sm"], SPACE["md"]))
        troubleshoot.columnconfigure(1, weight=1)

        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.pack(fill="x", padx=edge, pady=(0, SPACE["md"]))

        log_frame = ttk.LabelFrame(root, text="Progress log")
        log_frame.pack(fill="both", expand=True, padx=edge, pady=(0, edge))
        # Asks for six lines and takes every spare pixel below that, so the
        # window opens compactly and the log is what grows when it is enlarged.
        self.log_text = ScrolledText(log_frame, wrap="word", height=6)
        self.log_text.pack(fill="both", expand=True, padx=SPACE["sm"], pady=SPACE["sm"])
        self.log_text.configure(
            bg=COLORS["field"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
            selectbackground=COLORS["accent_dark"],
            relief="flat",
            borderwidth=0,
            font=("Consolas", TYPE["mono"]),
            padx=SPACE["md"],
            pady=SPACE["sm"],
            spacing1=1,
            spacing3=2,
        )
        self.log_text.insert(
            "end",
            "Ready. Choose the folder holding your videos, then press Run full analysis.\n"
            "The freezing and 7-behavior models are already loaded.\n",
        )
        self.log_text.configure(state="disabled")

    def _path_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, command) -> ttk.Entry:
        pad = {"padx": SPACE["md"], "pady": SPACE["sm"]}
        ttk.Label(parent, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="w", **pad)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, **pad)
        parent.columnconfigure(1, weight=1, minsize=200)
        # A long path is only recognisable by its tail, so keep the end in view.
        self._show_path_end(entry)
        var.trace_add("write", lambda *_: self._show_path_end(entry))
        entry.bind("<Configure>", lambda _event: self._show_path_end(entry, unfocused_only=True))
        return entry

    def _show_path_end(self, entry: ttk.Entry, unfocused_only: bool = False) -> None:
        if unfocused_only and self.focus_get() is entry:
            return  # someone is typing in it; leave their cursor where it is
        entry.after_idle(lambda: entry.xview_moveto(1.0))

    def _entry(
        self,
        parent: ttk.Frame,
        row: int,
        col: int,
        label: str,
        var: tk.StringVar,
        width: int = 10,
        hint: str = "",
    ) -> None:
        pad = {"padx": SPACE["md"], "pady": SPACE["sm"]}
        text = f"{label}  ({hint})" if hint else label
        ttk.Label(parent, text=text, style="Field.TLabel").grid(row=row, column=col, sticky="w", **pad)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=col + 1, sticky="w", **pad)

    def _choose_project_dir(self) -> None:
        value = filedialog.askdirectory(title="Choose project folder containing videos")
        if value:
            self.project_dir.set(value)

    def _choose_model(self) -> None:
        value = filedialog.askopenfilename(title="Choose freezing model", filetypes=[("Model files", "*.sav *.joblib *.pkl"), ("All files", "*.*")])
        if value:
            self.model_path.set(value)

    def _choose_dlc_config(self) -> None:
        value = filedialog.askopenfilename(title="Choose DeepLabCut config.yaml", filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")])
        if value:
            self.dlc_config_path.set(value)

    def _choose_behavior_model(self) -> None:
        value = filedialog.askopenfilename(title="Choose 7-behavior model", filetypes=[("Model files", "*.pkl *.joblib *.sav"), ("All files", "*.*")])
        if value:
            self.behavior_model_path.set(value)

    def _choose_behavior_labels(self) -> None:
        value = filedialog.askopenfilename(title="Choose 7-behavior label encoder", filetypes=[("Pickle files", "*.pkl *.joblib *.sav"), ("All files", "*.*")])
        if value:
            self.behavior_label_encoder_path.set(value)

    def _start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Already running", "The analysis is already running.")
            return
        try:
            settings = self._settings_from_form()
        except Exception as exc:
            messagebox.showerror("Check settings", str(exc))
            return

        self.run_button.configure(state="disabled")
        self._progress_begin()
        self._append_log("\nStarting analysis...\n")
        self.worker = threading.Thread(target=self._run_worker, args=(settings,), daemon=True)
        self.worker.start()

    @staticmethod
    def _number(var: tk.StringVar, name: str) -> float:
        """A field's value as a number, or an error that names the field."""
        text = var.get().strip()
        try:
            return float(text)
        except ValueError:
            shown = f"'{text}'" if text else "blank"
            raise ValueError(f"{name} must be a number; it is {shown}.") from None

    def _settings_from_form(self) -> FriendlyPipelineSettings:
        if not self.project_dir.get().strip():
            raise ValueError("Choose a project folder with videos.")
        if not self.model_path.get().strip():
            raise ValueError("Choose a freezing model file.")
        if self.run_dlc.get() and not self.dlc_config_path.get().strip():
            raise ValueError("Choose a DeepLabCut config.yaml, or uncheck Run DeepLabCut.")
        if self.run_behavior_analysis.get() and not self.behavior_model_path.get().strip():
            raise ValueError("Choose a 7-behavior model, or uncheck Run 7-behavior analysis.")
        if self.run_behavior_analysis.get() and not self.behavior_label_encoder_path.get().strip():
            raise ValueError("Choose a 7-behavior label encoder, or uncheck Run 7-behavior analysis.")

        return FriendlyPipelineSettings(
            project_dir=Path(self.project_dir.get()),
            model_path=Path(self.model_path.get()),
            dlc_config_path=Path(self.dlc_config_path.get() or "."),
            videotype=self.videotype.get(),
            threshold=self._number(self.threshold, "Freezing threshold") if self.threshold.get().strip() else None,
            fps=self._number(self.fps, "Frames per second"),
            bin_sec=int(self._number(self.bin_sec, "Time bin")),
            box_side_cm=self._number(self.box_side_cm, "Arena side (cm)"),
            box_side_pixels=self._number(self.box_side_pixels, "Arena side (pixels)"),
            run_clahe=self.run_clahe.get(),
            run_dlc=self.run_dlc.get(),
            dlc_env=self.dlc_env.get(),
            make_tracked_videos=self.make_tracked_videos.get(),
            run_behavior_analysis=self.run_behavior_analysis.get(),
            behavior_model_path=Path(self.behavior_model_path.get() or "."),
            behavior_label_encoder_path=Path(self.behavior_label_encoder_path.get() or "."),
        )

    def _run_worker(self, settings: FriendlyPipelineSettings) -> None:
        try:
            output = run_friendly_pipeline(settings, log=lambda msg: self.log_queue.put(("log", msg)))
        except Exception as exc:
            self.log_queue.put(("error", str(exc)))
        else:
            self.last_output_dir = output
            self.log_queue.put(("done", str(output)))

    def _start_refinement_wizard(self) -> None:
        if self.manual_worker and self.manual_worker.is_alive():
            messagebox.showinfo("Refinement", "A manual scoring window is already open.")
            return
        if self.refine_training_worker and self.refine_training_worker.is_alive():
            messagebox.showinfo("Refinement", "A refined model is already being trained.")
            return
        folder_value = filedialog.askdirectory(title="Choose folder with videos to manually score")
        if not folder_value:
            return

        selected = Path(folder_value).resolve()
        video_dir = self._find_video_dir(selected)
        videos = self._raw_videos_in_folder(video_dir)
        if not videos:
            messagebox.showerror("No videos", f"No raw '{self.videotype.get()}' videos were found in:\n{video_dir}")
            return

        dlc_dir = self._find_filtered_dlc_dir(selected)
        if dlc_dir is None:
            messagebox.showerror(
                "No filtered DLC CSVs",
                "I could not find matching filtered DLC CSVs for refinement.\n\n"
                "Run the full analysis first, then start refinement from the original project folder or the final package folder.",
            )
            return

        self.refine_videos = videos
        self.refine_next_index = 0
        self.refine_labels_dir = selected / "manual_labels"
        self.refine_labels_dir.mkdir(parents=True, exist_ok=True)
        self.refine_dlc_dir = dlc_dir
        self.refine_button.configure(state="disabled")
        self._append_log(
            "\n[refine] Manual refinement started.\n"
            f"[refine] Videos: {len(videos)} from {video_dir}\n"
            f"[refine] Filtered DLC CSVs: {dlc_dir}\n"
            f"[refine] Manual labels will be saved in: {self.refine_labels_dir}\n"
        )
        self._start_next_refine_label()

    def _find_video_dir(self, folder: Path) -> Path:
        for child in ("01_original_videos", "original_videos"):
            candidate = folder / child
            if candidate.exists():
                return candidate
        return folder

    def _raw_videos_in_folder(self, folder: Path) -> list[Path]:
        ext = self.videotype.get().strip() or ".mp4"
        return [path for path in list_videos(folder, ext) if _looks_like_raw_video(path)]

    def _find_filtered_dlc_dir(self, folder: Path) -> Optional[Path]:
        def analysis_dirs(parent: Path) -> list[Path]:
            old_style = list(parent.glob("FINAL_freezing_package_*"))
            new_style = [path for path in parent.glob("*_analysis*") if path.is_dir()]
            return sorted(
                old_style + new_style,
                key=lambda path: path.stat().st_mtime if path.exists() else 0,
                reverse=True,
            )

        candidates: list[Path] = []
        for base in (folder, self._find_video_dir(folder)):
            candidates.extend(
                [
                    base / "03_DLC_analysis" / "filtered",
                    base / "DLC_filtered",
                    base,
                ]
            )
        if self.last_output_dir is not None:
            candidates.append(self.last_output_dir / "03_DLC_analysis" / "filtered")
        if self.project_dir.get().strip():
            project = Path(self.project_dir.get()).resolve()
            candidates.append(project / "03_DLC_analysis" / "filtered")
            candidates.extend(
                path / "03_DLC_analysis" / "filtered" for path in analysis_dirs(project)
            )
        candidates.extend(
            path / "03_DLC_analysis" / "filtered" for path in analysis_dirs(folder)
        )

        seen = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists() and list(candidate.glob("*filtered.csv")):
                return candidate
        return None

    def _start_next_refine_label(self) -> None:
        if self.refine_next_index >= len(self.refine_videos):
            self._start_refine_training()
            return
        video_path = self.refine_videos[self.refine_next_index]
        self.refine_next_index += 1
        assert self.refine_labels_dir is not None
        out_csv = self.refine_labels_dir / f"{video_path.stem}_labels.csv"
        existing = out_csv if out_csv.exists() else None
        try:
            fps = float(self.fps.get())
        except ValueError:
            fps = 25.0
        self._append_log(
            f"[refine] Scoring {video_path.name}\n"
            "In the video window: press F at freezing start, press F again at freezing end, then press S to save.\n"
        )
        self.manual_worker = threading.Thread(
            target=self._manual_score_worker,
            args=(video_path, out_csv, existing, fps),
            daemon=True,
        )
        self.manual_worker.start()

    def _manual_score_worker(self, video_path: Path, out_csv: Path, existing: Optional[Path], fps: float) -> None:
        try:
            result = annotate_video_labels(
                video_path=video_path,
                out_csv=out_csv,
                fps_override=fps,
                start_paused=True,
                speed=1.0,
                existing_labels=existing,
            )
        except Exception as exc:
            self.log_queue.put(("refine_model_error", f"Manual scoring failed: {exc}"))
            return
        self.log_queue.put(("refine_label_done", result))

    def _ask_continue_or_train(self, result: dict[str, Any]) -> None:
        if result.get("saved"):
            self._append_log(f"[refine] Saved labels: {result.get('out_csv')}\n")
        else:
            self._append_log("[refine] Labeler closed without saving this video.\n")
        remaining = len(self.refine_videos) - self.refine_next_index
        if remaining > 0:
            keep_going = messagebox.askyesno(
                "Label another video?",
                f"Label another video before training the refined model?\n\nRemaining videos in folder: {remaining}",
            )
            if keep_going:
                self._start_next_refine_label()
                return
        self._start_refine_training()

    def _start_refine_training(self) -> None:
        if self.refine_dlc_dir is None or self.refine_labels_dir is None:
            messagebox.showerror("Refinement", "Missing DLC or manual labels folder.")
            self.refine_button.configure(state="normal")
            return
        pairs = _match_training_pairs(self.refine_dlc_dir, self.refine_labels_dir)
        if not pairs:
            messagebox.showerror(
                "No matching labels",
                "No manual label CSVs matched the filtered DLC CSVs.\n\n"
                "Make sure the manual labels were made from the same videos that were analyzed with DLC.",
            )
            self.refine_button.configure(state="normal")
            return
        self._progress_begin()
        self._append_log(f"[refine] Training new model from {len(pairs)} manually labeled video(s)...\n")
        self.refine_training_worker = threading.Thread(target=self._refine_training_worker, daemon=True)
        self.refine_training_worker.start()

    def _refine_training_worker(self) -> None:
        assert self.refine_dlc_dir is not None
        assert self.refine_labels_dir is not None
        out_dir = REPO_ROOT / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_model = out_dir / f"manual_refined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sav"
        try:
            payload = train_model(
                dlc_dir=self.refine_dlc_dir,
                labels_dir=self.refine_labels_dir,
                out_model=out_model,
                feature_mode="legacy4k",
                n_estimators=600,
                step_estimators=100,
                stabilize=True,
                stability_patience=4,
                stability_min_delta=0.0005,
                min_estimators_for_stability=300,
                max_train_frames=400000,
                split_mode="group",
                n_jobs=4,
                test_size=0.2,
                label_lag_frames=0,
                show_progress=False,
            )
        except Exception as exc:
            self.log_queue.put(("refine_model_error", str(exc)))
            return
        self.log_queue.put(
            (
                "refine_model_done",
                {
                    "model_path": str(out_model),
                    "threshold": float(payload.get("threshold", 0.35)),
                    "metrics": payload.get("metrics", {}),
                },
            )
        )

    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, message = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(message + "\n")
                elif kind == "error":
                    self._progress_end()
                    self.run_button.configure(state="normal")
                    self._append_log(f"\nERROR: {message}\n")
                    messagebox.showerror("Analysis failed", message)
                elif kind == "refine_label_done":
                    self._ask_continue_or_train(message)
                elif kind == "refine_model_error":
                    self._progress_end()
                    self.refine_button.configure(state="normal")
                    self._append_log(f"\n[refine] ERROR: {message}\n")
                    messagebox.showerror("Refinement failed", str(message))
                elif kind == "refine_model_done":
                    self._progress_end()
                    self.refine_button.configure(state="normal")
                    model_path = str(message["model_path"])
                    threshold = float(message["threshold"])
                    metrics = message.get("metrics", {})
                    self.model_path.set(model_path)
                    self.threshold.set(f"{threshold:.2f}")
                    self._append_log(
                        "\n[refine] New model created and loaded into the GUI.\n"
                        f"[refine] Model: {model_path}\n"
                        f"[refine] Threshold: {threshold:.2f}\n"
                        f"[refine] Metrics: {metrics}\n"
                    )
                    messagebox.showinfo(
                        "Refined model ready",
                        f"New model created and selected in the GUI:\n{model_path}\n\n"
                        f"Threshold updated to {threshold:.2f}.",
                    )
                elif kind == "done":
                    self._progress_end()
                    self.run_button.configure(state="normal")
                    self._append_log(f"\nDone. Final package:\n{message}\n")
                    messagebox.showinfo("Analysis complete", f"Final package created:\n{message}")
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)

    def _progress_begin(self) -> None:
        """One task started; animate. An analysis and a refinement can overlap,
        so the bar only stops once the last of them ends."""
        self._busy_count += 1
        if self._busy_count == 1:
            self.progress.start(10)

    def _progress_end(self) -> None:
        self._busy_count = max(0, self._busy_count - 1)
        if self._busy_count == 0:
            self.progress.stop()

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _open_output_folder(self) -> None:
        path = self.last_output_dir or (Path(self.project_dir.get()) if self.project_dir.get() else None)
        if not path:
            messagebox.showinfo("No folder yet", "Run an analysis first or choose a project folder.")
            return
        try:
            open_in_file_manager(path)
        except OSError as exc:
            messagebox.showinfo("Could not open the folder", f"{path}\n\n{exc}")


def main() -> None:
    _enable_dpi_awareness()
    app = FreezingDlcApp()
    app.mainloop()


if __name__ == "__main__":
    main()
