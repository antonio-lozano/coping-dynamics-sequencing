# SPDX-License-Identifier: MIT
"""Click the four arena-floor corners on a frame and save them.

Automatic floor detection infers the arena from brightness, which keeps
failing on this footage: the floor carries a shading gradient, the right
wall's inner face is visible from the camera angle, and gloss breaks the
plateau assumption. The legacy pipeline never inferred the floor at all --
a person clicked four corners into an annotated mask. This is that step.

Click the four corners of the arena floor in any order. Drag any corner to
adjust it. The floor fills yellow and the wall band pink as you go, matching
the published overlay, so the fit can be judged directly rather than trusted.

    python scripts/annotate_arena_floor.py --image FRAME.png

On save it writes ``<image>_floor.json`` beside the frame and prints the
``--floor-corners`` string to paste into the video build.
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image, ImageTk

FLOOR_RGB = np.array([255, 255, 0])     # yellow, as in the published overlay
WALL_RGB = np.array([255, 192, 203])    # pink
MASK_ALPHA = 0.45
HANDLE = 7                              # px radius of a corner handle
GRAB = 18                               # px within which a click grabs a corner

CORNER_FILL = "#ff2d2d"
CORNER_EDGE = "#ffffff"
EDGE_LINE = "#ff2d2d"


def order_clockwise(points: np.ndarray) -> np.ndarray:
    """Order four corners consistently, starting top-left."""
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))   # smallest x+y is top-left
    return np.roll(ordered, -start, axis=0)


class FloorAnnotator:
    def __init__(self, root: tk.Tk, image_path: Path, initial: np.ndarray | None):
        self.root = root
        self.image_path = image_path
        self.original = Image.open(image_path).convert("RGB")
        self.width, self.height = self.original.size

        # Fit the frame on screen, leaving room for the instruction bar.
        margin = 190
        limit_w = root.winfo_screenwidth() - 80
        limit_h = root.winfo_screenheight() - margin
        self.scale = min(1.0, limit_w / self.width, limit_h / self.height)
        self.view = (int(self.width * self.scale), int(self.height * self.scale))
        self.base = np.asarray(
            self.original.resize(self.view, Image.Resampling.LANCZOS), dtype=float
        )

        self.corners: list[list[float]] = []
        if initial is not None:
            self.corners = [[float(x), float(y)] for x, y in initial]
        self.dragging: int | None = None
        self.saved = False

        root.title(f"Arena floor - {image_path.name}")
        root.configure(bg="#1e1e1e")

        header = tk.Label(
            root,
            text="Click the FOUR corners of the arena floor.  "
                 "Drag a corner to adjust.  Right-click removes the nearest.",
            font=("Segoe UI", 11), bg="#1e1e1e", fg="#f0f0f0", pady=8,
        )
        header.pack()

        self.canvas = tk.Canvas(
            root, width=self.view[0], height=self.view[1],
            highlightthickness=0, cursor="crosshair", bg="#101010",
        )
        self.canvas.pack(padx=12)

        self.status = tk.Label(
            root, text="", font=("Consolas", 10),
            bg="#1e1e1e", fg="#8fd6ff", pady=6,
        )
        self.status.pack()

        bar = tk.Frame(root, bg="#1e1e1e")
        bar.pack(pady=(2, 12))
        self.save_button = tk.Button(
            bar, text="Save floor  (Enter)", command=self.save,
            font=("Segoe UI", 10, "bold"), bg="#2f7d32", fg="white",
            activebackground="#3a9c3e", relief="flat", padx=18, pady=7,
        )
        self.save_button.pack(side="left", padx=6)
        tk.Button(
            bar, text="Reset  (R)", command=self.reset, font=("Segoe UI", 10),
            bg="#444444", fg="white", activebackground="#555555",
            relief="flat", padx=18, pady=7,
        ).pack(side="left", padx=6)
        tk.Button(
            bar, text="Cancel  (Esc)", command=self.cancel, font=("Segoe UI", 10),
            bg="#7d2f2f", fg="white", activebackground="#9c3a3a",
            relief="flat", padx=18, pady=7,
        ).pack(side="left", padx=6)

        # Closing with the window's X button would otherwise discard placed
        # corners without a word, which is the easiest way to lose the work.
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "dragging", None))
        self.canvas.bind("<Button-3>", self.on_right_click)
        root.bind("<Return>", lambda _e: self.save())
        root.bind("<Escape>", lambda _e: self.cancel())
        root.bind("r", lambda _e: self.reset())
        root.bind("R", lambda _e: self.reset())

        self.redraw()

    # ---------------------------------------------------------------- events

    def nearest(self, x: float, y: float) -> int | None:
        if not self.corners:
            return None
        distances = [np.hypot(cx - x, cy - y) for cx, cy in self.corners]
        index = int(np.argmin(distances))
        return index if distances[index] <= GRAB else None

    def on_click(self, event: tk.Event) -> None:
        index = self.nearest(event.x, event.y)
        if index is not None:
            self.dragging = index
            return
        if len(self.corners) >= 4:
            self.status.config(
                text="Four corners already placed - drag one, or press R to reset.")
            return
        self.corners.append([float(event.x), float(event.y)])
        self.redraw()

    def on_drag(self, event: tk.Event) -> None:
        if self.dragging is None:
            return
        self.corners[self.dragging] = [
            float(np.clip(event.x, 0, self.view[0] - 1)),
            float(np.clip(event.y, 0, self.view[1] - 1)),
        ]
        self.redraw()

    def on_right_click(self, event: tk.Event) -> None:
        index = self.nearest(event.x, event.y)
        if index is not None:
            self.corners.pop(index)
            self.redraw()

    # --------------------------------------------------------------- drawing

    def full_resolution(self) -> np.ndarray:
        points = np.array(self.corners, dtype=float) / self.scale
        return order_clockwise(points)

    def redraw(self) -> None:
        frame = self.base
        if len(self.corners) >= 3:
            quad = np.array(self.corners, dtype=float)
            quad = order_clockwise(quad) if len(quad) == 4 else quad
            yy, xx = np.mgrid[0:self.view[1], 0:self.view[0]]
            inside = MplPath(quad).contains_points(
                np.stack([xx.ravel(), yy.ravel()], axis=1)
            ).reshape(self.view[1], self.view[0])
            frame = self.base.copy()
            frame[inside] = frame[inside] * (1 - MASK_ALPHA) + FLOOR_RGB * MASK_ALPHA
            frame[~inside] = frame[~inside] * (1 - MASK_ALPHA) + WALL_RGB * MASK_ALPHA

        self.photo = ImageTk.PhotoImage(Image.fromarray(frame.astype(np.uint8)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        if len(self.corners) >= 2:
            ordered = (order_clockwise(np.array(self.corners, dtype=float))
                       if len(self.corners) == 4 else np.array(self.corners))
            closed = len(self.corners) == 4
            for i in range(len(ordered) - (0 if closed else 1)):
                x1, y1 = ordered[i]
                x2, y2 = ordered[(i + 1) % len(ordered)]
                self.canvas.create_line(x1, y1, x2, y2, fill=EDGE_LINE, width=2)

        for i, (x, y) in enumerate(self.corners, start=1):
            self.canvas.create_oval(x - HANDLE, y - HANDLE, x + HANDLE, y + HANDLE,
                                    fill=CORNER_FILL, outline=CORNER_EDGE, width=2)
            self.canvas.create_text(x + HANDLE + 9, y - HANDLE - 5, text=str(i),
                                    fill="#ffffff", font=("Segoe UI", 10, "bold"))

        placed = len(self.corners)
        if placed < 4:
            self.status.config(
                text=f"{placed} of 4 corners placed - click {4 - placed} more.")
            self.save_button.config(state="disabled", bg="#3a3a3a")
        else:
            corners = self.full_resolution()
            text = " ".join(f"{int(round(x))},{int(round(y))}" for x, y in corners)
            self.status.config(text=f"corners: {text}")
            self.save_button.config(state="normal", bg="#2f7d32")

    # ---------------------------------------------------------------- actions

    def reset(self) -> None:
        self.corners = []
        self.dragging = None
        self.redraw()

    def on_close(self) -> None:
        if len(self.corners) == 4:
            answer = messagebox.askyesnocancel(
                "Arena floor",
                "Save the four corners before closing?\n\n"
                "Yes  - save and close\n"
                "No   - discard and close",
            )
            if answer is None:
                return
            if answer:
                self.save()
                return
        self.cancel()

    def cancel(self) -> None:
        self.saved = False
        self.root.destroy()

    def save(self) -> None:
        if len(self.corners) != 4:
            messagebox.showwarning("Arena floor",
                                   "Place all four corners before saving.")
            return
        corners = self.full_resolution()
        argument = " ".join(f"{int(round(x))},{int(round(y))}" for x, y in corners)
        out = self.image_path.with_name(self.image_path.stem + "_floor.json")
        out.write_text(json.dumps({
            "image": self.image_path.name,
            "image_size": [self.width, self.height],
            "corners_xy": [[round(float(x), 1), round(float(y), 1)] for x, y in corners],
            "floor_corners_argument": argument,
        }, indent=2), encoding="utf-8")
        self.saved = True
        self.result = (out, argument)
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True,
                        help="frame to annotate")
    parser.add_argument("--corners", type=str,
                        help="pre-load corners as 'x,y x,y x,y x,y' to adjust")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Frame not found: {args.image}")

    initial = None
    if args.corners:
        numbers = [float(v) for v in args.corners.replace(",", " ").split()]
        if len(numbers) != 8:
            raise SystemExit("--corners needs 8 numbers (4 x,y pairs)")
        initial = np.array(numbers).reshape(4, 2)

    root = tk.Tk()
    if initial is not None:
        temporary = FloorAnnotator(root, args.image, None)
        temporary.corners = [[x * temporary.scale, y * temporary.scale]
                             for x, y in initial]
        app = temporary
        app.redraw()
    else:
        app = FloorAnnotator(root, args.image, None)
    root.mainloop()

    if getattr(app, "saved", False):
        out, argument = app.result
        print(f"wrote {out}")
        print()
        print("Pass this to the video build:")
        print(f'  --climb-floor-corners "{argument}"')
    else:
        print("cancelled; no corners written")


if __name__ == "__main__":
    main()
