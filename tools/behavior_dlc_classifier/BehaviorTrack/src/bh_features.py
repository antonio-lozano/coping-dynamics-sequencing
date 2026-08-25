#!/usr/bin/env python
"""Step 3: compute the feature sets and classify every frame.

Where BarnesTrack turns tracking into maze metrics, this turns tracking into
behavior. One 719-feature set is built from the pose file and one XGBoost model
decides all seven behaviors, including Freezing. Coordinates are centred exactly
as in the archived training rows; they are not silently rotated or rescaled.

No percentile behavior rules, separate freezing override, or canonical label
smoothing are applied. Tracking failures become Unassigned, and the confidence
written out is the probability of the behavior actually named.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.classify import ClassifySettings, classify_one  # noqa: E402
from utils.config_loader import ensure_engine_importable, resolve_config_path  # noqa: E402
from utils.runner import StepWindow, field_row, path_row  # noqa: E402
from utils.ui_style import behavior_legend  # noqa: E402
from utils.workspace import load_workspace  # noqa: E402


def tracking_files(folder: Path) -> list[Path]:
    """Filtered DLC CSVs in a folder, newest naming first-come.

    Both the collected copies under Results/Tracking and the originals sitting
    beside the videos are accepted, so this step works whether or not step 2 ran
    in this session.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(folder.rglob("*filtered.csv"))


class FeaturesWindow(StepWindow):
    step_title = "3. Compute features and classify behaviors"
    step_subtitle = "One archived 719-feature model for all seven behaviors"
    step_blurb = (
        "Reads every filtered DeepLabCut CSV, reproduces the archived feature schema, and writes exact "
        "model calls, the complete 719-feature CSV, summaries, and a provenance/domain-audit file."
    )
    run_label = "Classify"

    def __init__(self, config_path: Path) -> None:
        super().__init__(config_path, geometry="1280x900")

    def initialize_state(self) -> None:
        self.workspace = load_workspace(self.config_path)
        behaviors = self.workspace.section("behaviors")
        features = self.workspace.section("features")
        models = self.workspace.models()

        default_dir = self.workspace.tracking_dir if tracking_files(self.workspace.tracking_dir) else self.workspace.raw_videos
        self.tracking_var = tk.StringVar(value=str(default_dir))
        self.behavior_model_var = tk.StringVar(value=str(models.behavior))
        self.fps_var = tk.StringVar(value=str(self.workspace.fps))
        self.bin_var = tk.StringVar(value=str(behaviors.get("bin_sec", 30)))
        self.window_var = tk.StringVar(value=str(features.get("behavior_window", 5)))
        self.skip_done_var = tk.BooleanVar(value=True)
        self.summary_var = tk.StringVar(value="")

    def after_build(self) -> None:
        self.refresh_summary()

    def results_dir(self) -> Path:
        return self.workspace.behaviors_dir

    def body(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        path_row(parent, 0, "Tracking folder", self.tracking_var, self.browse_tracking)
        path_row(parent, 1, "Unified behavior model", self.behavior_model_var, self.browse_behavior)

        settings = ttk.Frame(parent, style="Panel.TFrame")
        settings.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        field_row(settings, 0, 0, "Frames per second", self.fps_var, width=8)
        field_row(settings, 0, 2, "Summary bin (s)", self.bin_var, width=8)
        field_row(settings, 0, 4, "Feature window", self.window_var, width=8)
        ttk.Label(
            settings,
            text="Raw model calls: no behavior thresholds, separate freezing override, or smoothing.",
            style="MutedPanel.TLabel",
        ).grid(row=1, column=0, columnspan=6, sticky="w", pady=(4, 0))

        ttk.Checkbutton(
            parent, text="Skip recordings that are already classified", variable=self.skip_done_var
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Button(parent, text="Rescan", command=self.refresh_summary).grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Label(parent, textvariable=self.summary_var, style="Panel.TLabel", wraplength=1080, justify="left").grid(
            row=4, column=1, columnspan=2, sticky="w", pady=(8, 0), padx=(8, 0)
        )
        behavior_legend(parent, style_base="Panel").grid(row=5, column=0, columnspan=3, sticky="w", pady=(12, 0))

    # ------------------------------------------------------------------ ops #

    def browse_tracking(self) -> None:
        chosen = filedialog.askdirectory(title="Folder holding filtered DLC CSVs", initialdir=self.tracking_var.get() or ".")
        if chosen:
            self.tracking_var.set(chosen)
            self.refresh_summary()

    def browse_behavior(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Archived behavior model", filetypes=[("Pickle model", "*.pkl"), ("All files", "*.*")]
        )
        if chosen:
            self.behavior_model_var.set(chosen)

    def _pending(self) -> tuple[list[Path], list[Path]]:
        found = tracking_files(Path(self.tracking_var.get()))
        done = self.workspace.classified_stems()
        todo = [
            path
            for path in found
            if not self.skip_done_var.get() or (path.stem.split("DLC")[0] or path.stem) not in done
        ]
        return found, todo

    def refresh_summary(self) -> None:
        found, todo = self._pending()
        if not found:
            self.summary_var.set(
                "No '*filtered.csv' tracking files in that folder. Run step 2, or point at the folder holding them."
            )
            return
        self.summary_var.set(f"{len(found)} tracking file(s) found, {len(todo)} to classify.")

    def _settings(self) -> ClassifySettings:
        return ClassifySettings(
            behavior_model=Path(self.behavior_model_var.get()),
            behavior_label_encoder=self.workspace.models().behavior_label_encoder,
            fps=float(self.fps_var.get()),
            bin_sec=int(float(self.bin_var.get())),
            behavior_window=int(float(self.window_var.get())),
        )

    def work(self, log) -> str:
        ensure_engine_importable(self.config_path)
        found, todo = self._pending()
        if not found:
            raise RuntimeError(f"No '*filtered.csv' files in {self.tracking_var.get()}")
        if not todo:
            log("[classify] Everything in that folder is already classified.")
            return "Nothing to do."

        settings = self._settings()
        self.workspace.ensure_dirs()
        log(f"[setup] {len(todo)} recording(s) -> {self.workspace.results}")
        log(f"[setup] unified behavior model={settings.behavior_model.name}")
        log(f"[setup] fps={settings.fps}  feature_window={settings.behavior_window}")

        done = 0
        failures: list[str] = []
        for index, path in enumerate(todo, start=1):
            log("")
            log(f"--- [{index}/{len(todo)}] {path.name} ---")
            try:
                stem = path.stem.split("DLC")[0] or path.stem
                prediction_dir = self.workspace.behavior_predictions_dir(stem)
                result = classify_one(
                    path,
                    settings,
                    log,
                    out_dir=prediction_dir,
                    feature_dir=self.workspace.feature_output_dir(stem),
                    summary_dir=self.workspace.behavior_summaries_dir(stem),
                    provenance_dir=self.workspace.behavior_provenance_dir(stem),
                )
            except Exception as exc:
                failures.append(f"{path.name}: {exc}")
                log(f"[error] {exc}")
                continue
            top = result.totals.sort_values("percent", ascending=False).head(3)
            summary = ", ".join(f"{row.behavior} {row.percent:.1f}%" for row in top.itertuples())
            log(f"[result] {result.frames} frames - {summary}")
            for warning in result.warnings:
                log(f"[warn] {warning}")
            done += 1
            self.set_progress(index, len(todo))

        if failures:
            log("")
            log(f"[error] {len(failures)} recording(s) failed:")
            for line in failures:
                log(f"    {line}")
        return f"Classified {done} of {len(todo)} recording(s)" + (f"; {len(failures)} failed" if failures else ".")

    def _on_finished(self, ok: bool) -> None:
        self.workspace = load_workspace(self.config_path)
        self.refresh_summary()


def self_test(config_path: Path) -> int:
    """Classify a synthetic pose file end to end, with the real models."""
    import numpy as np
    import pandas as pd

    print("[self-test] bh_features")
    workspace = load_workspace(config_path)
    ensure_engine_importable(config_path)
    models = workspace.models()
    missing = models.missing()
    if missing:
        print(f"  cannot run: missing {', '.join(missing)}")
        return 1

    from freezing_dlc.behavior import BEHAVIOR_BODYPARTS

    # A synthetic recording: an animal drifting steadily, so tracking is valid
    # and every body part is present. The point is that the wiring runs, not that
    # the labels mean anything.
    frames = 400
    rng = np.random.default_rng(0)
    columns: dict[str, np.ndarray] = {}
    drift = np.linspace(0, 120, frames)
    for index, part in enumerate(BEHAVIOR_BODYPARTS):
        columns[f"{part}_x"] = 320 + drift + index * 8 + rng.normal(0, 0.4, frames)
        columns[f"{part}_y"] = 240 + index * 6 + rng.normal(0, 0.4, frames)
        columns[f"{part}_likelihood"] = np.full(frames, 0.99)
    df_flat = pd.DataFrame(columns)
    print(f"  synthetic pose  {frames} frames, {len(BEHAVIOR_BODYPARTS)} body parts")

    import tempfile

    from utils.classify import ClassifySettings as CS
    from utils.classify import bouts_from_labels, totals_from_labels
    from utils.model_audit import audit_model

    settings = CS(
        behavior_model=models.behavior,
        behavior_label_encoder=models.behavior_label_encoder,
        fps=25.0,
        bin_sec=30,
    )

    from freezing_dlc import behavior as behavior_mod

    model = behavior_mod.load_behavior_model(settings.behavior_model, settings.behavior_label_encoder)
    contract = audit_model(settings.behavior_model, settings.behavior_label_encoder, model)
    print(f"  behavior model  {len(model.feature_names)} features, {len(model.classes)} classes")
    print(f"  model contract  {'PASS' if contract.contract_ok else 'FAIL'}")
    feats = behavior_mod.build_behavior_feature_set(df_flat, model.feature_names, fps=25.0, window=5)
    print(f"  feature set     {feats.shape[0]} x {feats.shape[1]}")
    labels, codes, probs = behavior_mod.predict_behaviors(feats, model)
    final, source, metrics = behavior_mod.refine_behavior_labels(labels, probs, df_flat, None, None)
    classes = [*behavior_mod.BEHAVIOR_DISPLAY_ORDER, behavior_mod.UNASSIGNED]
    totals = totals_from_labels(final, 25.0, classes)

    checks = [
        ("archived model contract passes", contract.contract_ok),
        ("feature count matches the model", feats.shape[1] == len(model.feature_names)),
        ("one label per frame", len(final) == frames),
        ("labels are known classes", set(np.asarray(final, dtype=str)) <= set(classes)),
        ("totals cover 100%", abs(totals.percent.sum() - 100.0) < 1e-6),
        ("bouts reconstruct the frame count", int(bouts_from_labels(final, 25.0).frames.sum()) == frames),
    ]
    for name, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")

    top = totals.sort_values("percent", ascending=False).head(3)
    print("  top classes     " + ", ".join(f"{r.behavior} {r.percent:.1f}%" for r in top.itertuples()))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "synthetic_behaviors.csv"
        behavior_mod.save_behavior_predictions(out, np.asarray(final, dtype=object), codes, probs, metrics_df=metrics)
        rows = len(pd.read_csv(out))
        print(f"  wrote           {out.name} ({rows} rows)")
        checks.append(("written CSV has one row per frame", rows == frames))

    failed = [name for name, ok in checks if not ok]
    if failed:
        print(f"[self-test] FAILED: {', '.join(failed)}")
        return 1
    print("[self-test] PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BehaviorTrack step 3: features and classification")
    parser.add_argument("--config", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    config_path = resolve_config_path(args.config)
    if args.self_test:
        return self_test(config_path)
    FeaturesWindow(config_path).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
