#!/usr/bin/env python
"""Step 4: ethograms, per-class totals and the combined workbook.

Reads what step 3 wrote and turns it into things you can look at and hand to
someone else: one ethogram per recording, a per-recording composition table, and
a workbook with a sheet per view.

Every table carries the final QC-aware label beside the model's untouched argmax.
The only permitted difference is that unusable tracking becomes Unassigned.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from utils.config_loader import ensure_engine_importable, resolve_config_path  # noqa: E402
from utils.runner import StepWindow  # noqa: E402
from utils.ui_style import BEHAVIOR_COLORS, COLORS, UNASSIGNED, behavior_legend  # noqa: E402
from utils.workspace import load_workspace  # noqa: E402

BEHAVIOR_ROWS = ["Jump", "Climbing", "Locomotion", "Turn", "Grooming", "Sniffing", "Freezing", UNASSIGNED]


def read_behavior_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def stem_of(path: Path) -> str:
    return path.name[: -len("_behaviors.csv")]


def draw_ethogram_png(labels: pd.Series, out_png: Path, *, fps: float, title: str) -> Path:
    """One row per class, time along x, drawn with the shared palette."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    values = labels.astype(str).to_numpy()
    seconds = len(values) / float(fps)

    fig, ax = plt.subplots(figsize=(12, 3.4), dpi=150)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    rows = [name for name in BEHAVIOR_ROWS]
    for index, name in enumerate(rows):
        mask = values == name
        if not mask.any():
            continue
        # Contiguous runs, so one patch per bout rather than one per frame.
        changes = (mask[1:] != mask[:-1]).nonzero()[0] + 1
        edges = [0, *changes.tolist(), len(mask)]
        for start, end in zip(edges[:-1], edges[1:]):
            if mask[start]:
                ax.add_patch(
                    mpatches.Rectangle(
                        (start / fps, index - 0.4),
                        (end - start) / fps,
                        0.8,
                        facecolor=BEHAVIOR_COLORS.get(name, "#888888"),
                        edgecolor="none",
                    )
                )

    ax.set_xlim(0, max(seconds, 1e-6))
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, color=COLORS["text"], fontsize=8)
    ax.set_xlabel("Time (s)", color=COLORS["text"], fontsize=9)
    ax.set_title(title, color=COLORS["accent"], fontsize=10, loc="left")
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(COLORS["line"])
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png


def composition_table(behaviors_dir: Path, fps: float) -> pd.DataFrame:
    """One row per recording: QC-aware and untouched-model percentages."""
    rows = []
    for path in sorted(Path(behaviors_dir).rglob("*_behaviors.csv")):
        frame = read_behavior_csv(path)
        stem = stem_of(path)
        total = len(frame)
        row: dict[str, object] = {"recording": stem, "frames": total, "seconds": total / fps}
        final = frame["behavior"].astype(str)
        model_raw = frame["raw_behavior"].astype(str) if "raw_behavior" in frame.columns else final
        for name in BEHAVIOR_ROWS:
            row[name] = 100.0 * float((final == name).sum()) / total if total else 0.0
            row[f"{name}_model_raw"] = (
                100.0 * float((model_raw == name).sum()) / total if total else 0.0
            )
        changed = int((final != model_raw).sum())
        row["tracking_qc_changed_percent"] = 100.0 * changed / total if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def bouts_table(behaviors_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(behaviors_dir).rglob("*_behavior_bouts.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "recording", path.name[: -len("_behavior_bouts.csv")])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bins_table(behaviors_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(behaviors_dir).rglob("*_behavior_bins.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "recording", path.name[: -len("_behavior_bins.csv")])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_workbook(path: Path, composition: pd.DataFrame, bouts: pd.DataFrame, bins: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        composition.to_excel(writer, sheet_name="Composition", index=False)
        if not bins.empty:
            bins.to_excel(writer, sheet_name="TimeBins", index=False)
        if not bouts.empty:
            bouts.to_excel(writer, sheet_name="Bouts", index=False)
            summary = (
                bouts.groupby("behavior")
                .agg(
                    bouts=("frames", "size"),
                    total_seconds=("duration_sec", "sum"),
                    median_seconds=("duration_sec", "median"),
                    longest_seconds=("duration_sec", "max"),
                )
                .reset_index()
            )
            summary.to_excel(writer, sheet_name="BoutSummary", index=False)
    return path


class ResultsWindow(StepWindow):
    step_title = "4. Ethograms, totals and summaries"
    step_subtitle = "Look at what was classified, then export it"
    step_blurb = (
        "Draws one ethogram per recording, builds a per-recording composition table and writes the combined "
        "workbook. Final labels and untouched model argmax are both retained so tracking-QC exclusions are visible."
    )
    run_label = "Build ethograms and workbook"

    def __init__(self, config_path: Path) -> None:
        super().__init__(config_path, geometry="1320x920")

    def initialize_state(self) -> None:
        self.workspace = load_workspace(self.config_path)
        self.summary_var = tk.StringVar(value="")
        self.annotated_var = tk.BooleanVar(
            value=bool(self.workspace.section("outputs").get("make_annotated_videos", True))
        )

    def after_build(self) -> None:
        self.refresh_table()

    def results_dir(self) -> Path:
        return self.workspace.results

    def body(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(parent, textvariable=self.summary_var, style="Panel.TLabel", wraplength=1120, justify="left").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(parent, text="Refresh", command=self.refresh_table).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            parent,
            text="Also write annotated behavior videos (slower; useful for checking calls)",
            variable=self.annotated_var,
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))

        table = ttk.Frame(parent, style="Panel.TFrame")
        table.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        table.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)
        columns = ("recording", "seconds", "Freezing", "Freezing_model_raw", "Locomotion", "Turn", UNASSIGNED, "changed")
        headings = ("Recording", "Seconds", "Freeze %", "Freeze % model", "Locom %", "Turn %", "Unassigned %", "QC changed %")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", height=10)
        for column, heading in zip(columns, headings):
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=230 if column == "recording" else 110, anchor="w" if column == "recording" else "e")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)

        behavior_legend(parent, style_base="Panel").grid(row=4, column=0, sticky="w", pady=(10, 0))

    # ------------------------------------------------------------------ ops #

    def refresh_table(self) -> None:
        self.workspace = load_workspace(self.config_path)
        self.tree.delete(*self.tree.get_children())
        folder = self.workspace.behaviors_dir
        if not folder.is_dir() or not list(folder.rglob("*_behaviors.csv")):
            self.summary_var.set("Nothing classified yet. Run step 3 first.")
            return
        table = composition_table(folder, self.workspace.fps)
        for row in table.itertuples():
            self.tree.insert(
                "",
                "end",
                values=(
                    row.recording,
                    f"{row.seconds:.0f}",
                    f"{getattr(row, 'Freezing'):.1f}",
                    f"{getattr(row, 'Freezing_model_raw'):.1f}",
                    f"{getattr(row, 'Locomotion'):.1f}",
                    f"{getattr(row, 'Turn'):.1f}",
                    f"{getattr(row, UNASSIGNED):.1f}",
                    f"{row.tracking_qc_changed_percent:.1f}",
                ),
            )
        self.summary_var.set(
            f"{len(table)} recording(s) classified. QC-changed frames are tracking failures marked Unassigned."
        )

    def work(self, log) -> str:
        ensure_engine_importable(self.config_path)
        folder = self.workspace.behaviors_dir
        files = sorted(folder.rglob("*_behaviors.csv"))
        if not files:
            raise RuntimeError(f"No classified recordings in {folder}. Run step 3 first.")

        self.workspace.ensure_dirs()
        fps = self.workspace.fps
        made = 0
        annotated = 0
        annotate_requested = self.annotated_var.get()
        if annotate_requested:
            from freezing_dlc.behavior import annotate_behavior_video
            from freezing_dlc.dlc_stage import find_source_video

        write_ethograms = bool(self.workspace.section("outputs").get("write_ethograms", True))
        if write_ethograms or annotate_requested:
            for index, path in enumerate(files, start=1):
                frame = read_behavior_csv(path)
                stem = stem_of(path)
                if write_ethograms:
                    out = self.workspace.figure_output_dir(stem) / f"{stem}_ethogram.png"
                    draw_ethogram_png(frame["behavior"], out, fps=fps, title=stem)
                    log(f"[ethogram] {out.name}")
                    made += 1
                if annotate_requested:
                    try:
                        source = find_source_video(self.workspace.raw_videos, stem, self.workspace.videotype)
                    except FileNotFoundError as exc:
                        log(f"[warn] {stem}: annotated video skipped - {exc}")
                    else:
                        video_out = self.workspace.annotated_output_dir(stem) / f"{stem}_behaviors.mp4"
                        labels = frame["behavior"].astype(str).to_numpy()
                        confidence = frame["confidence"].to_numpy(dtype=float) if "confidence" in frame else None
                        annotate_behavior_video(
                            source,
                            video_out,
                            labels,
                            frame,
                            fps=max(1, int(round(fps))),
                            metrics_df=frame,
                            confidence=confidence,
                        )
                        log(f"[video] {video_out.name}")
                        annotated += 1
                self.set_progress(index, len(files))
        else:
            log("[media] Ethograms and annotated videos are both disabled.")

        composition = composition_table(folder, fps)
        bouts = bouts_table(folder)
        bins = bins_table(folder)
        out = write_workbook(self.workspace.summary_xlsx, composition, bouts, bins)
        log(f"[workbook] {out}")
        log(f"[workbook] Composition {len(composition)} rows, TimeBins {len(bins)}, Bouts {len(bouts)}")

        return (
            f"{made} ethogram(s), {annotated} annotated video(s), and a workbook "
            f"with {len(composition)} recording(s)."
        )

    def _on_finished(self, ok: bool) -> None:
        self.refresh_table()


def self_test(config_path: Path) -> int:
    """Draw an ethogram and build a workbook from synthetic classified output."""
    import tempfile

    import numpy as np

    print("[self-test] bh_results")
    workspace = load_workspace(config_path)
    fps = workspace.fps

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        frames = 1000
        labels = np.array(["Locomotion"] * 400 + ["Freezing"] * 300 + ["Turn"] * 200 + [UNASSIGNED] * 100)
        raw = labels.copy()
        raw[:20] = "Jump"  # a difference so the untouched-model column is exercised
        pd.DataFrame({"frame": np.arange(frames), "behavior": labels, "raw_behavior": raw}).to_csv(
            folder / "demo_behaviors.csv", index=False
        )
        pd.DataFrame(
            {"behavior": ["Locomotion"], "start_frame": [0], "end_frame": [399], "frames": [400], "start_sec": [0.0], "duration_sec": [16.0]}
        ).to_csv(folder / "demo_behavior_bouts.csv", index=False)
        pd.DataFrame({"bin": [0], "bin_start_sec": [0.0], "bin_seconds": [30.0], "Freezing": [30.0]}).to_csv(
            folder / "demo_behavior_bins.csv", index=False
        )

        table = composition_table(folder, fps)
        print(f"  composition   {len(table)} row(s), {len(table.columns)} columns")
        png = draw_ethogram_png(pd.Series(labels), folder / "demo_ethogram.png", fps=fps, title="demo")
        size = png.stat().st_size
        print(f"  ethogram      {png.name} ({size} bytes)")
        book = write_workbook(folder / "demo.xlsx", table, bouts_table(folder), bins_table(folder))
        # On Windows an open workbook handle prevents TemporaryDirectory from
        # deleting the file at the end of this test.  Keep the lifetime
        # explicit so the self-test behaves the same on every platform.
        with pd.ExcelFile(book) as workbook:
            sheets = workbook.sheet_names
        print(f"  workbook      {book.name} sheets={sheets}")

        checks = [
            ("one composition row", len(table) == 1),
            ("model column present and differs", abs(float(table.Jump_model_raw.iloc[0]) - 2.0) < 1e-6),
            ("QC-aware Jump is zero", float(table.Jump.iloc[0]) == 0.0),
            ("percentages sum to 100", abs(sum(float(table[c].iloc[0]) for c in BEHAVIOR_ROWS) - 100.0) < 1e-6),
            ("ethogram PNG is non-trivial", size > 5000),
            ("workbook has the expected sheets", {"Composition", "TimeBins", "Bouts", "BoutSummary"} <= set(sheets)),
        ]
        for name, ok in checks:
            print(f"  {'OK  ' if ok else 'FAIL'} {name}")

    failed = [name for name, ok in checks if not ok]
    if failed:
        print(f"[self-test] FAILED: {', '.join(failed)}")
        return 1
    print("[self-test] PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BehaviorTrack step 4: results")
    parser.add_argument("--config", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    config_path = resolve_config_path(args.config)
    if args.self_test:
        return self_test(config_path)
    ResultsWindow(config_path).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
