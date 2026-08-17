# Cleanup TODO

Everything in the working tree is now tracked. That was done deliberately, to
stop the repository from depending on files that existed only on one machine:
several already-tracked scripts imported or required untracked ones, so a clean
clone was broken before this commit.

Tracking everything also swept in work that is unfinished, oversized, or only
ever meant to be run once. This note records what still needs a decision, so
the sweep does not quietly become the final state of the package.

Ordered roughly by how much it matters.

## 1. The legacy Figure 7 source PDF is gone

`data/raw/legacy_figures/figure7_classifier_original.pdf` disappeared from disk
during the session that produced this commit. It was **never committed on any
branch**, so it cannot be restored from git history.

Consequences:

- `scripts/check_reproducibility.py` still lists it as required, so the check
  currently fails on a clean clone. This is the only failing item.
- Figure 7 panel B no longer reads it. The generator falls back to
  `figure_source_data/figure7_classifier_global_shap.csv`, which was exported
  from that PDF and holds the same 20 x 8 values, so the figure still rebuilds
  correctly.

Decide one of:

- **Restore it** from the keypoint-MoSeq working tree or a backup, commit it,
  and drop the fallback branch in `figure7_global_shap_values()`; or
- **Retire it** — remove it from the required-file list and the manifest check
  in `check_reproducibility.py`, and state in `docs/figure_structure.md` that
  the exported CSV is now the archival source for those values.

The second option is honest only if the CSV really is accepted as the source of
record; the PDF is the primary artifact and the CSV is a derivative of it.

## 2. `classifier/legacy_shap/shap_values.npz` is 54 MB

Tracked as an ordinary git blob. It is under GitHub's 100 MB hard limit but
over the 50 MB warning threshold, and the repository history is already ~1.1 GB.

It earns its place: Supplementary Figure 4 regenerates from it alone, with no
`--source` and no external working tree. This was verified by rebuilding the
figure from the tracked archive — the source-data CSV came back byte-identical.

Still worth deciding whether it belongs in Git LFS. Moving it would be a
repository-wide workflow change affecting every clone, so it was not done
unilaterally.

Related, smaller point: the archive stores `feature_names` and `class_names` as
object arrays, so `load_archive()` must pass `allow_pickle=True`. That
partially undercuts the stated reason for moving off pickles
(`scripts/import_legacy_shap_values.py`). Saving those two arrays as plain
fixed-width strings would let the archive load with `allow_pickle=False`.

## 3. The two `.pkl` files beside it are still untracked

`classifier/legacy_shap/xgb_model.pkl` and `label_encoder.pkl` are excluded by
the global `*.pkl` rule in `.gitignore`. They are **not** needed to draw
Supplementary Figure 4 from the tracked `.npz`, so the figure is safe.

They are only needed to re-derive the archive from the legacy source. If that
path is meant to stay open on a clean clone, they need either an ignore
exception or a documented external location. Right now the re-derivation path
silently depends on one machine.

## 4. One-time migration utilities are now tracked as if they were pipeline steps

- `scripts/import_legacy_shap_summary.py`
- `scripts/import_legacy_shap_values.py`

Both are one-shot migrations that require `--source DIR` pointing at the legacy
keypoint-MoSeq tree, which is not part of this repository. Their own docstrings
say so, but nothing in the repository layout does.

Consider moving them under something like `scripts/migrations/`, or noting in
the README that they are historical and not part of a rebuild.

## 5. Manuscript-text helpers are unreviewed

- `scripts/prepare_results_differences.py`
- `scripts/prepare_revised_results.py`

These parse the manuscript `.docx` and emit revised Results text with
regenerated statistics. They carry hard-coded before/after strings such as
`beta = 0.621 -> 0.623`, tied to one manuscript revision.

They are not wired into `run_all_figures.py` or `check_reproducibility.py`.
Decide whether they are part of the deliverable or scratch work; if they stay,
they need a note about which manuscript revision they target, because the
hard-coded strings will silently stop matching after the next edit.

## 6. `shap` was missing from every environment file

`requirements.txt` advertised covering Supplementary Figure 4 but omitted
`shap`, which that generator imports at module load. `pyproject.toml` and
`environment.yml` omitted it too, so no documented install route could build
the figure. All three now declare `shap>=0.41`.

Two things to watch:

- Installing `shap` pulls `numba`/`llvmlite`, and resolvers will happily drag
  in numpy 2.x, which violates the `numpy<2.0.0` pin the rest of the package
  relies on. Verified working combination: numpy 1.26.4, shap 0.51.0,
  numba 0.67.0. If `uv sync` starts producing numpy 2.x, pin `numba` too.
- No lockfile is committed, so these resolutions are not reproducible across
  machines. Committing `uv.lock` would fix that.

## 7. Figures were not regenerated in this commit

The figure files here are the previously committed renders. Supplementary
Figure 4 was rebuilt during verification and then restored, because installing
`shap` also moved matplotlib 3.10.6 -> 3.11.0, and shipping one figure rendered
under a different matplotlib than its siblings would be misleading.

The rebuild differed only in the embedded matplotlib version string and
randomised clip-path IDs — no geometry or data changed. Before the next
release, rebuild **all** figures under one pinned matplotlib so the set is
internally consistent.

## 8. Excel lock file

`report/~$raw_data.xlsx` was present, meaning `raw_data.xlsx` was open in Excel
while these files were staged. `~$*.xlsx` is now in `.gitignore`.

Close the workbook before regenerating reports — Excel can hold a stale copy in
memory and write it back over a freshly generated one.
