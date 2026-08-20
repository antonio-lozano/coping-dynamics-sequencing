# Workbook Match Audit

Generated on 2026-08-16.

## Outputs

- `report/raw_data.xlsx`: figure-aligned raw-data workbook for submission.
- `report/statistical_report.xlsx`: live recomputation containing the Combined
  full dataset plus separate Sanguino-Gomez & Krugers and Sanguino-Gomez et al.
  sections wherever source dataset identity exists.

Neither workbook builder copies a final `.xlsx` file. Missing manuscript-side raw
tables are tracked under `data/raw/manuscript_tables/raw_data/`. Historical
statistical cells whose original model inputs are not all available are stored
as typed, text-based records under
`data/raw/manuscript_tables/statistical_report/`.

The historical statistical cells remain a provenance record. The single final
statistical workbook contains only analyses recalculated by repository code.

## Raw-Data Workbook

`scripts/build_raw_data_workbook.py` regenerates `report/raw_data.xlsx` as a
34-sheet, figure-aligned workbook. It uses consistent `Animal`, `Group`, and
`Dataset` metadata headings; Title Case measure headings; verb-form behavior
labels; explicit Figure 4 panel ranges; resilience-stratified Figure 5 and 6
tables; Supplementary Figure 1 time courses plus total frequencies; and
animal-level Supplementary Figure 3 distance metrics. Figure 7A-N is split into
panel-accurate classifier, resilience-prediction, SHAP, predictor-catalog, and
time-course sheets. Supplementary Figure 4A-H has an explicit class-specific
classifier SHAP sheet carrying all 719 legacy parameters per behavior. Archive/helper sheets
that do not directly correspond to a figure are intentionally excluded.

For Figure 7, the raw workbook contains plotted source values and animal-level
contributions. The statistical workbook contains only detailed held-out/LOOCV
performance, aggregate Shapley estimates, and permutation results. Descriptive
source tables are not duplicated between the two workbooks.

The supplied historical raw-data workbook was previously validated as an exact
21-sheet reconstruction (193,249 union nonblank cells and 0 differing cells at
`--atol 1e-12`). The current submission-facing workbook intentionally differs
in organization and labels. Historical source tables remain tracked under
`data/raw/manuscript_tables/raw_data/` for provenance.

The historical report source-cell archive was validated against the supplied
workbook with:

- 18 / 18 exact sheets
- 20,843 populated cells
- 0 differing cells
- matching formulas, value types, coordinates, and sheet order

These exact cells are retained under
`data/raw/manuscript_tables/statistical_report/`; they are not presented as a
recalculated statistical output.

## Statistical Provenance

The historical archive includes legacy rounding, report-only values, and known
historical inconsistencies. Exact reconstruction does not imply that those
cells were recalculated from unavailable inputs. The source conversion is
performed by `scripts/import_legacy_statistical_report.py` and documented in
`data/raw/manuscript_tables/statistical_report/provenance.md`.

The calculated output is `report/statistical_report.xlsx`. It is generated
from tracked raw and processed tables and does not read the supplied final
workbook. It begins directly with the figure-aligned analyses.

## Dataset Coverage

| Sheet family | Combined full dataset | Sanguino-Gomez & Krugers | Sanguino-Gomez et al. | Method |
| --- | --- | --- | --- | --- |
| Fig.1A classifier validation | yes | unavailable | unavailable | Pearson regression |
| Fig.2A-C ground truth | yes | yes | yes | MixedLM |
| Fig.2D syllable usage | yes | yes | yes | descriptive |
| Fig.2E-F precision/recall | yes | yes | yes | descriptive |
| Fig.2G overlap sources | yes | yes | yes | descriptive |
| Fig.2H freezing syllables | yes | yes | yes | MixedLM |
| Fig.3 frequency/time course | yes | yes | yes | GEE / MixedLM |
| Fig.4 diversity/bouts/transitions | yes | yes | yes | MixedLM |
| Fig.5 distance/frequency/time course and posthoc | yes | yes | yes | MixedLM / GEE / Welch tests |
| Fig.6 resilience metrics | yes | yes | yes | MixedLM |
| Fig.7A-C classifier validation | raw workbook only | unavailable | unavailable | descriptive source values |
| Fig.7D-E resilience prediction | yes | cross-cohort where applicable | cross-cohort where applicable | ROC AUC / performance metrics |
| Fig.7F Shapley | yes | unavailable | unavailable | animal-level raw / aggregate report |
| Fig.7G-H predictor catalog | raw workbook only | cross-cohort values included | cross-cohort values included | descriptive source values |
| Fig.7I prediction onset | yes | combined | combined | plotted AUC / permutation test |
| Fig.7J-N time courses | raw workbook only | combined | combined | descriptive cross-validation summaries |
| Supplementary Fig.3 distances | yes | yes | yes | MixedLM |
| Supplementary Fig.4 class-specific SHAP | yes | unavailable | unavailable | descriptive classifier SHAP |

Source dataset identity for Figure 5 and Supplementary Figure 3 is recovered by a
validated animal-ID join to `data/processed/cluster_frequency_per_animal.csv`.
The classifier validation input has phase but no animal-level experiment field,
so it is correctly left combined.

Singular MixedLM fits are explicitly labeled `OLS fallback` when encountered.
No section is silently omitted. Figure 6 is recalculated from the bundled
250 ms syllable table used by its dedicated derivation script.
