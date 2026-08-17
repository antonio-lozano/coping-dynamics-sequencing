# Cleanup ledger

The 2026-08-17 "track everything" sweep tracked every working-tree file so a
clean clone would stop depending on files that existed only on one machine, and
recorded eight open items here. This revision of the note records how each was
closed and what genuinely remains. Ordered as in the original.

## Closed

1. **The legacy Figure 7 source PDF is retired.**
   `data/raw/legacy_figures/figure7_classifier_original.pdf` was lost before it
   was ever committed and exists on no machine we can reach.
   `figure_source_data/figure7_classifier_global_shap.csv`, written by the same
   extraction, is now the archival source of record (`docs/figure_structure.md`,
   `docs/data_dictionary.md`), and the checker no longer requires the PDF. The
   extraction code and the layout-check exception remain, so a recovered copy
   dropped back at the original path is re-extracted and compared, not rejected.

2. **`classifier/legacy_shap/shap_values.npz` stays a plain git blob.** 54 MB is
   under GitHub's hard limit, the figure rebuilds from it alone, and Git LFS
   would change every clone's workflow. Revisit only if GitHub starts refusing
   pushes or repository growth becomes a real problem.

3. **The two legacy pickles are tracked.** `xgb_model.pkl` (2.9 MB) and
   `label_encoder.pkl` (629 B) are committed under a scoped `.gitignore`
   exception. Before committing, the tracked `shap_values.npz` was re-derived
   from the same legacy tree and all four arrays came back identical.

4. **One-time utilities moved to `scripts/migrations/`.** Both legacy SHAP
   import utilities live there now, and README and `docs/figure_structure.md`
   state they are provenance, not rebuild steps.

5. **The manuscript-text helpers are annotated.** Their docstrings state they
   target the August 2026 manuscript revision and that the hard-coded
   before/after strings silently stop matching after the next edit.

6. **The environment is locked with `shap`.** `uv.lock` is committed and
   resolves shap 0.49.1 with numpy held at 1.26.4 — the feared numpy 2.x pull
   did not happen under the lock. `uv sync --locked` is the documented route.

7. **All figures are rebuilt under one environment.** Every render is
   regenerated under the locked matplotlib 3.11.0 on the reference machine, so
   the set embeds a single version. `run_all_figures.py` now includes
   Supplementary Figure 4.

8. **Excel lock files are ignored and skipped.** `~$*.xlsx` is in `.gitignore`
   and `update_manifest.py` skips such files. Still close workbooks before
   regenerating reports — Excel can write a stale copy back over a fresh one.

## Still open

- **Hunt a backup of the retired PDF.** If a copy of
  `figure7_classifier_original.pdf` turns up in mail, a backup, or another
  machine, place it at `data/raw/legacy_figures/` and rebuild Figure 7: the
  generator re-runs the original extraction so the archived CSV can be
  confirmed against it.

- **Full re-derivation of the SHAP archive still needs the legacy tree.**
  `shap_values.pkl` (60 MB) and `shap_sample.csv` (19 MB) remain outside the
  repository, and the ranks CSV that fed `import_legacy_shap_summary.py`
  (`features_defining_each_behavior_from_shap.csv`) no longer exists anywhere —
  its tracked output `legacy_feature_ranks.csv` is what preserves that
  information.

- **`shap_values.npz` needs `allow_pickle=True`** because `feature_names` and
  `class_names` are object arrays. Saving them as fixed-width strings would let
  the archive load with `allow_pickle=False` and finish the move off pickles.

- **`statistics/fig5_timecourse_mixedlm.csv` has 24 of 56 degenerate rows**
  (coefficients ~1e-19, standard errors ~5.8e7, p = 1.0) in intercept and
  main-effect terms; slopes and interactions are well-formed. A "Fix singular
  MixedLM" commit exists in history, so the singularity is known but not fully
  resolved. Confirm before quoting any intercept or main-effect term from that
  file.

- **The audit records one `major_mismatch`.** Figure 6W (Lempel–Ziv, vulnerable
  ELS vs Control): the manuscript reports p < 0.001, the regenerated model
  gives p = 0.042 (SE 4.42 vs 2.66). Recorded in
  `statistics/manuscript_consistency_audit.csv`; the manuscript text or the
  model choice needs to be reconciled by the authors.

- **Mint a Zenodo DOI at release.** Archiving a tagged release on Zenodo gives
  the repository a citable DOI and a snapshot independent of GitHub. Needs the
  maintainers' Zenodo account; once minted, add the DOI to the `identifiers`
  list in `CITATION.cff` and to the README citation section.

- **Move the 93 MB DeepLabCut training snapshot to a release asset.** The
  vendored tool tracks
  `tools/behavior_dlc_classifier/models/dlc_project/.../train/snapshot-500000.data-00000-of-00001`
  as a plain blob, a few MB under GitHub's hard 100 MB push limit — a
  retrained, slightly larger snapshot would be rejected outright. Decision
  2026-08-17: keep the blob for now; later attach it to a GitHub Release with
  a checksum-verified download step, the way the sibling repository already
  ships its 1 GB classifier model. Every commit stacked on top makes the
  eventual removal more history surgery, so prefer sooner over later.
