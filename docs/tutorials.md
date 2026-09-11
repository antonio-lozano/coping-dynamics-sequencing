# Tutorials

> Recommended isolated full run: `uv run python -m coping_dynamics.reproduction --output ../coping-reproduction` from the checkout root. Use a new destination. This preserves source artifacts and records comparisons and logs; nonzero status means failure, partial coverage or review required. The individual commands below write in place and belong in a disposable checkout.

These are checkout-based workflows using existing commands. They distinguish bundled downstream data from raw-video reprocessing. Successful execution must be recorded; these instructions are not a claim that every workflow is already verified.

## Choose your starting point

| Starting point | Next step | What you will get |
| --- | --- | --- |
| New to this repository | [First recording](first-recording.md) | A small, real-data saved-model inference run |
| Reviewing the paper | Steps 1–3 below | Integrity checks, isolated figures and reports |
| Classifying all bundled MoSeq frames | Step 4 | Predictions from the root model |
| Prefer a visual interface | [Companion GUI walkthrough](../apps/behavior-dlc-classifier/GUIDE.md#first-gui-run-without-installing-deeplabcut) | Reused-tracking app workflow |
| Bringing new videos | Step 5 and companion guide | Tracking prerequisites and model-transfer boundaries |

## 1. Verify the bundled study snapshot

From the repository root after [installation](installation.md):

```bash
uv run python scripts/check_reproducibility.py
uv run pytest -m "not slow" -q
```

Expected: zero exit status from both commands. The checker verifies tracked artifact integrity and layout; tests exercise their implemented contracts. Neither command reruns all manuscript analyses. Preserve the output if either fails, and do not regenerate the manifest to erase a mismatch.

## 2. Reproduce a data-derived figure

Use a disposable checkout/worktree because figure commands overwrite tracked outputs. Keep an untouched reference checkout for comparison. Figure 2 uses the bundled manual/automatic validation data and associated source tables; see [the data dictionary](data_dictionary.md) and [figure inventory](figure_structure.md).

```bash
uv run python scripts/generate_figures/figure_2_validation.py
```

Inspect `figures/figure2.pdf` and associated exports/source tables. Compare plotted values, labels and sample counts against their original sources; a readable PDF alone is not a scientific check. A standalone figure invocation may include variable export timestamps. For the runner's deterministic metadata configuration, use:

```bash
uv run python scripts/run_all_figures.py
```

That command regenerates all data-derived figures, not just Figure 2. Static schematics and archival exports remain separate. [Reproducibility](../REPRODUCIBILITY.md) explains reference comparison and byte-level limitations.

## 3. Rebuild reports and the study pipeline

In the disposable checkout:

```bash
uv run python scripts/build_raw_data_workbook.py
uv run python scripts/build_statistical_report.py
```

Expected output paths are `report/raw_data.xlsx` and `report/statistical_report.xlsx`; inspect workbook sheets and exported `statistics/` tables. Existing source tables must be present. To derive tables and run the complete implemented pipeline in order:

```bash
uv run python scripts/run_all.py --verbose
```

Save the original manifest before running: the runner refreshes it, so its final hash check is not a comparison against the old reference. The bundled-manuscript check concerns revised copies, not all claims in the original supplied manuscript. Do not run manuscript rewrite helpers.

## 4. Apply the root behavior classifier to bundled MoSeq data

This is inference with a saved model, not independent model validation or raw-video tracking:

```bash
uv run behavior-classifier predict \
  --input data/raw/moseq_syllables_per_frame.csv.gz \
  --model classifier/figure7_behavior_classifier.joblib \
  --output classifier/outputs/tutorial_predictions.csv
```

Check that the CSV contains recording/frame identifiers and predicted behavior labels; check probabilities where the model provides them. Record the model/input hashes and row counts, and compare against the exact input population. This command does not overwrite the model. The full table can be large; do not mistake a truncated-input smoke run for full inference verification.

The root model, the archived Supplementary Figure 4 SHAP model and the companion application's freezing model are distinct. Do not interchange their results or use one model's explanation arrays as evidence for another.

## 5. Classify existing DLC tracks or raw video

The separate [companion application guide](../apps/behavior-dlc-classifier/GUIDE.md) documents its environment and commands. Existing DLC tracks plus compatible classifier assets are prerequisites. Raw-video execution additionally requires a working matching DeepLabCut network/environment.

No bundled downstream reproduction command proves those upstream stages ran. Record them as **unverified** until genuine inputs produce validated tracking, predictions and summaries; record missing assets/environment as **blocked** with the exact prerequisite. A reused-tracking test does not validate raw-video pose estimation.

Training and cross-validation require explicit method/provenance review: root frame-level CV is not an independent-animal holdout. Do not retrain over archived models or replace reported metrics simply because a new split gives different values.
