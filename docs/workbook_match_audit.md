# Workbook Match Audit

Generated on 2026-08-15.

## Outputs

- `report/raw_data.xlsx`: historical raw-data workbook.
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

## Exact Baselines

`scripts/build_raw_data_workbook.py` regenerates `report/raw_data.xlsx` with:

- 21 / 21 exact sheets
- 193,249 union nonblank cells
- 0 differing cells
- matching sheet order

Raw floating-point cells are compared with `--atol 1e-12`; the strict
bit-level comparison finds only CSV/Excel round-trip differences around the
15th decimal place and no substantive value differences.

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
workbook. Its first two sheets state the reporting conventions and compare
headline regenerated values with the manuscript.

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
| Fig.5 distance/frequency/time course and posthoc | yes | yes | yes | OLS / GEE / MixedLM / Welch tests |
| Fig.6 resilience metrics | yes | yes | yes | MixedLM |
| Supplementary Fig.3 distances | yes | yes | yes | OLS |

Source dataset identity for Figure 5 and Supplementary Figure 3 is recovered by a
validated animal-ID join to `data/processed/cluster_frequency_per_animal.csv`.
The classifier validation input has phase but no animal-level experiment field,
so it is correctly left combined.

Singular MixedLM fits are explicitly labeled `OLS fallback` when encountered.
No section is silently omitted. Figure 6 is recalculated from the bundled
250 ms syllable table used by its dedicated derivation script.
