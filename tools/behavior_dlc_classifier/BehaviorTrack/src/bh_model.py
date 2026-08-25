#!/usr/bin/env python
"""Step 5: independently validate all seven behavior classes on target data."""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config_loader import resolve_config_path  # noqa: E402
from utils.runner import StepWindow, path_row  # noqa: E402
from utils.validation import create_blinded_template, match_files, validate_predictions  # noqa: E402
from utils.workspace import load_workspace  # noqa: E402

OPERATIONS = ("Create blinded annotation templates", "Validate independent manual labels")


class ModelWindow(StepWindow):
    step_title = "5. Validate the seven behaviors"
    step_subtitle = "Independent target labels, per-class metrics, no frame-shuffled score"
    step_blurb = (
        "Creates annotation sheets without revealing model calls, then compares completed manual labels "
        "with the frozen classifier. Reports are grouped by recording and identify animal/session coverage."
    )
    run_label = "Run selected action"

    def __init__(self, config_path: Path) -> None:
        super().__init__(config_path, geometry="1280x820")

    def initialize_state(self) -> None:
        self.workspace = load_workspace(self.config_path)
        validation = self.workspace.results / "Validation"
        self.operation_var = tk.StringVar(value=OPERATIONS[0])
        self.predictions_var = tk.StringVar(value=str(self.workspace.behaviors_dir))
        self.labels_var = tk.StringVar(value=str(validation / "Manual_Labels"))
        self.report_var = tk.StringVar(value=str(validation / "Reports"))
        self.independent_var = tk.BooleanVar(value=False)
        self.note_var = tk.StringVar(value="")

    def after_build(self) -> None:
        self._operation_changed()

    def results_dir(self) -> Path:
        return Path(self.report_var.get())

    def body(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Action", style="MutedPanel.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        combo = ttk.Combobox(parent, textvariable=self.operation_var, values=OPERATIONS, state="readonly")
        combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=4)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._operation_changed())
        path_row(parent, 1, "Behavior predictions", self.predictions_var, self.browse_predictions)
        path_row(parent, 2, "Manual labels", self.labels_var, self.browse_labels)
        path_row(parent, 3, "Validation reports", self.report_var, self.browse_reports)
        ttk.Checkbutton(
            parent,
            text=(
                "I confirm these labels were made independently of the model calls and these recordings "
                "were not used to fit the selected classifier"
            ),
            variable=self.independent_var,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Label(
            parent, textvariable=self.note_var, style="WarnPanel.TLabel",
            wraplength=1080, justify="left",
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(12, 0))

    def _operation_changed(self) -> None:
        if self.operation_var.get() == OPERATIONS[0]:
            self.note_var.set(
                "Templates contain only frame, time, and an empty true_behavior column. Model predictions "
                "are hidden to reduce annotation bias. Interval labels with start_frame, end_frame, and "
                "true_behavior are also accepted."
            )
        else:
            self.note_var.set(
                "Validation never refits the model. It writes a confusion matrix, precision/recall/F1 for "
                "each behavior, and recording-level results. Missing classes remain visible."
            )

    def _browse_dir(self, title: str, variable: tk.StringVar) -> None:
        chosen = filedialog.askdirectory(title=title, initialdir=variable.get() or ".")
        if chosen:
            variable.set(chosen)

    def browse_predictions(self) -> None:
        self._browse_dir("Folder containing behavior predictions", self.predictions_var)

    def browse_labels(self) -> None:
        self._browse_dir("Folder containing independent manual labels", self.labels_var)

    def browse_reports(self) -> None:
        self._browse_dir("Folder for validation reports", self.report_var)

    def work(self, log: Callable[[str], None]) -> str:
        predictions = Path(self.predictions_var.get()).resolve()
        labels = Path(self.labels_var.get()).resolve()
        reports = Path(self.report_var.get()).resolve()
        prediction_files = sorted(predictions.rglob("*_behaviors.csv")) if predictions.is_dir() else []
        if not prediction_files:
            raise FileNotFoundError(f"No *_behaviors.csv predictions found in {predictions}")

        if self.operation_var.get() == OPERATIONS[0]:
            labels.mkdir(parents=True, exist_ok=True)
            made = 0
            for path in prediction_files:
                stem = path.name[: -len("_behaviors.csv")]
                out = labels / f"{stem}_manual_labels.csv"
                if out.exists():
                    log(f"[skip] {out.name} already exists")
                    continue
                create_blinded_template(path, out, self.workspace.fps)
                log(f"[write] {out.name}")
                made += 1
            return f"Created {made} blinded template(s); existing annotations were preserved."

        if not labels.is_dir():
            raise FileNotFoundError(f"Manual-label folder not found: {labels}")
        log(f"[setup] {len(match_files(predictions, labels))} matched recording(s)")
        result = validate_predictions(
            predictions,
            labels,
            reports,
            sessions=self.workspace.sessions(),
            independent_holdout_confirmed=self.independent_var.get(),
        )
        log(f"[report] {result.report_json}")
        log(f"[status] {result.claim_status}")
        for warning in result.warnings:
            log(f"[warn] {warning}")
        return f"Validated {result.labeled_frames} labeled frames from {result.matched_recordings} recording(s)."


def self_test(config_path: Path) -> int:
    """Exercise blinded templates and seven-class reporting without a GUI."""
    import tempfile

    import pandas as pd

    print("[self-test] bh_model")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        predictions, labels, reports = root / "predictions", root / "labels", root / "reports"
        predictions.mkdir()
        labels.mkdir()
        predicted = list(("Jump", "Climbing", "Locomotion", "Turn", "Grooming", "Sniffing", "Freezing")) * 2
        prediction_path = predictions / "mouse1_Test_behaviors.csv"
        pd.DataFrame({"frame": range(len(predicted)), "behavior": predicted}).to_csv(prediction_path, index=False)
        template = create_blinded_template(prediction_path, labels / "mouse1_Test_manual_labels.csv", 25.0)
        template_columns = pd.read_csv(template, nrows=0).columns
        completed = pd.read_csv(template)
        completed["true_behavior"] = predicted
        completed.to_csv(template, index=False)
        result = validate_predictions(predictions, labels, reports, independent_holdout_confirmed=False)
        checks = [
            ("template contains no prediction column", "behavior" not in template_columns),
            ("all seven classes reported", len(pd.read_csv(result.metrics_csv)) == 7),
            ("perfect synthetic agreement", pd.read_csv(result.metrics_csv)["f1"].eq(1.0).all()),
            ("independence is not silently claimed", result.claim_status.startswith("agreement audit")),
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
    parser = argparse.ArgumentParser(description="BehaviorTrack step 5: seven-behavior validation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    config_path = resolve_config_path(args.config)
    if args.self_test:
        return self_test(config_path)
    ModelWindow(config_path).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
