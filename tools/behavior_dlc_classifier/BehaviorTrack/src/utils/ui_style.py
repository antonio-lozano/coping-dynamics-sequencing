"""Shared BehaviorTrack Tkinter styling and small UI helpers.

The palette is deliberately the same one BarnesTrack uses, so the two tools read
as siblings on a desk where both are open. What is added here is the behavior
palette: the seven classes carry fixed colours in the engine, and every window
that draws an ethogram, a legend or a class chip pulls them from one place, so a
class cannot be teal in the ethogram and orange in the legend.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

COLORS = {
    "bg": "#050607",
    "panel": "#101316",
    "panel2": "#171B20",
    "text": "#F3F1EA",
    "muted": "#A7A39A",
    "accent": "#C7A76C",
    "accent_dark": "#7F6735",
    "danger": "#5A1D1D",
    "entry": "#0B0E11",
    "line": "#2B3036",
    "good": "#74B88A",
    "warn": "#F0902E",
    "bad": "#D46A5D",
}

#: The seven behavior classes plus Unassigned, in the order the engine displays
#: them (top of an ethogram first). Colours mirror freezing_dlc.behavior so the
#: GUI and the annotated videos agree.
BEHAVIOR_DISPLAY_ORDER = [
    "Jump",
    "Climbing",
    "Locomotion",
    "Turn",
    "Grooming",
    "Sniffing",
    "Freezing",
]
UNASSIGNED = "Unassigned"
BEHAVIOR_COLORS = {
    "Freezing": "#d47aae",
    "Sniffing": "#6f9ead",
    "Grooming": "#a8d7e8",
    "Turn": "#b9dc43",
    "Locomotion": "#e2ba19",
    "Climbing": "#e88427",
    "Jump": "#e45d63",
    UNASSIGNED: "#b8b8b8",
}


def apply_dark_theme(root: tk.Misc) -> None:
    """Apply the shared BehaviorTrack dark theme."""
    root.configure(bg=COLORS["bg"])
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=COLORS["bg"], foreground=COLORS["text"], font=("Segoe UI", 10))
    style.configure("TFrame", background=COLORS["bg"])
    style.configure("Panel.TFrame", background=COLORS["panel"])
    style.configure("Card.TFrame", background=COLORS["panel2"])
    style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
    style.configure("Muted.TLabel", background=COLORS["bg"], foreground=COLORS["muted"])
    style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text"])
    style.configure("MutedPanel.TLabel", background=COLORS["panel"], foreground=COLORS["muted"])
    style.configure("Card.TLabel", background=COLORS["panel2"], foreground=COLORS["text"])
    style.configure("MutedCard.TLabel", background=COLORS["panel2"], foreground=COLORS["muted"])
    for name, key, base in (
        ("GoodCard", "good", "panel2"),
        ("WarnCard", "warn", "panel2"),
        ("BadCard", "bad", "panel2"),
        ("GoodPanel", "good", "panel"),
        ("WarnPanel", "warn", "panel"),
        ("BadPanel", "bad", "panel"),
    ):
        style.configure(
            f"{name}.TLabel",
            background=COLORS[base],
            foreground=COLORS[key],
            font=("Segoe UI", 10, "bold"),
        )
    style.configure(
        "Title.TLabel",
        background=COLORS["bg"],
        foreground=COLORS["text"],
        font=("Segoe UI", 22, "bold"),
    )
    style.configure(
        "Accent.TLabel",
        background=COLORS["bg"],
        foreground=COLORS["accent"],
        font=("Segoe UI", 10, "bold"),
    )
    style.configure(
        "CardTitle.TLabel",
        background=COLORS["panel2"],
        foreground=COLORS["accent"],
        font=("Segoe UI", 12, "bold"),
    )
    style.configure(
        "PanelTitle.TLabel",
        background=COLORS["panel"],
        foreground=COLORS["text"],
        font=("Segoe UI", 13, "bold"),
    )
    style.configure(
        "TButton",
        background=COLORS["panel2"],
        foreground=COLORS["text"],
        bordercolor=COLORS["line"],
        padding=(12, 8),
    )
    style.map(
        "TButton",
        background=[("active", COLORS["line"]), ("pressed", COLORS["accent_dark"])],
        foreground=[("disabled", COLORS["muted"])],
    )
    style.configure(
        "Accent.TButton",
        background=COLORS["accent"],
        foreground="#080808",
        bordercolor=COLORS["accent"],
        font=("Segoe UI", 10, "bold"),
    )
    style.map(
        "Accent.TButton",
        background=[("active", "#D9BC7D"), ("pressed", COLORS["accent_dark"])],
    )
    # A finished step. Same weight as Accent so a card does not change size
    # when it goes green, only colour.
    style.configure(
        "Good.TButton",
        background=COLORS["good"],
        foreground="#080808",
        bordercolor=COLORS["good"],
        font=("Segoe UI", 10, "bold"),
    )
    style.map(
        "Good.TButton",
        background=[("active", "#8FCBA2"), ("pressed", "#4E8C63")],
    )
    style.configure(
        "TRadiobutton",
        background=COLORS["bg"],
        foreground=COLORS["text"],
        indicatorcolor=COLORS["entry"],
    )
    style.map(
        "TRadiobutton",
        background=[("active", COLORS["bg"])],
        foreground=[("active", COLORS["text"])],
    )
    style.configure(
        "TCheckbutton",
        background=COLORS["panel"],
        foreground=COLORS["text"],
        indicatorcolor=COLORS["entry"],
    )
    style.map("TCheckbutton", background=[("active", COLORS["panel"])])
    style.configure(
        "Card.TCheckbutton",
        background=COLORS["panel2"],
        foreground=COLORS["text"],
        indicatorcolor=COLORS["entry"],
    )
    style.map("Card.TCheckbutton", background=[("active", COLORS["panel2"])])
    style.configure(
        "TEntry",
        fieldbackground=COLORS["entry"],
        foreground=COLORS["text"],
        insertcolor=COLORS["text"],
        bordercolor=COLORS["line"],
        lightcolor=COLORS["line"],
        darkcolor=COLORS["line"],
        padding=5,
    )
    style.configure(
        "TCombobox",
        fieldbackground=COLORS["entry"],
        foreground=COLORS["text"],
        background=COLORS["entry"],
        arrowcolor=COLORS["accent"],
        bordercolor=COLORS["line"],
        lightcolor=COLORS["line"],
        darkcolor=COLORS["line"],
        padding=5,
    )
    style.configure(
        "TLabelframe",
        background=COLORS["panel"],
        foreground=COLORS["accent"],
        bordercolor=COLORS["line"],
    )
    style.configure(
        "TLabelframe.Label",
        background=COLORS["panel"],
        foreground=COLORS["accent"],
        font=("Segoe UI", 10, "bold"),
    )
    style.configure(
        "Treeview",
        background=COLORS["entry"],
        fieldbackground=COLORS["entry"],
        foreground=COLORS["text"],
        bordercolor=COLORS["line"],
        rowheight=25,
    )
    style.configure(
        "Treeview.Heading",
        background=COLORS["panel2"],
        foreground=COLORS["accent"],
        relief="flat",
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "Treeview",
        background=[("selected", COLORS["accent_dark"])],
        foreground=[("selected", COLORS["text"])],
    )
    style.configure(
        "Horizontal.TProgressbar",
        troughcolor=COLORS["entry"],
        background=COLORS["accent"],
        bordercolor=COLORS["line"],
    )


def open_in_file_explorer(path: str | Path) -> None:
    """Open a local folder in the system file explorer."""
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(folder)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])


def behavior_legend(parent: tk.Misc, *, style_base: str = "Card") -> ttk.Frame:
    """A row of colour chips naming every behavior class.

    Built from BEHAVIOR_COLORS rather than a hand-written list, so a class added
    to the engine cannot go missing from the legend.
    """
    background = COLORS["panel2"] if style_base == "Card" else COLORS["panel"]
    frame = ttk.Frame(parent, style=f"{style_base}.TFrame")
    for column, name in enumerate([*BEHAVIOR_DISPLAY_ORDER, UNASSIGNED]):
        chip = tk.Canvas(frame, width=12, height=12, highlightthickness=0, bg=background)
        chip.create_rectangle(0, 0, 12, 12, fill=BEHAVIOR_COLORS[name], outline="")
        chip.grid(row=0, column=column * 2, padx=(0 if column == 0 else 10, 4), pady=2)
        ttk.Label(frame, text=name, style=f"Muted{style_base}.TLabel").grid(
            row=0, column=column * 2 + 1, sticky="w"
        )
    return frame
