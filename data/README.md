# Data

All inputs needed to regenerate the manuscript figures are bundled here under
`data/source/` and tracked in git, so the figures reproduce with no external
data. The two large inputs (a per-frame syllable table and a results pickle) are
stored gzip-compressed to stay within GitHub's file-size limits; pandas and the
scripts read the compressed forms transparently.

## Figure source data (`data/source/`)

| File | Original source | Used by |
| --- | --- | --- |
| `syllable_usage_per_timebin_30s.csv` | `Coping_data2.zip::COping/Syllable_per_timebin_final(30s).csv` | Fig 3, Fig 5 |
| `syllable_usage_per_timebin_250ms.csv` | `Coping_data2.zip::COping/Syllable_per_timebin_final(250ms).csv` | Fig 4, Fig 6 |
| `bfl_scores.xlsx` | `Coping_data.zip::CSVs/BFL_scores.xlsx` | Fig 5 |
| `animal_groups.csv` | equipo_project `index.csv` (name → group) | Fig 2 |
| `syllable_classification_metrics.csv` | equipo_project | Fig 2 (panels D/E) |
| `freezing_overlap_by_group.svg` | equipo_project | Fig 2 (82-animal cohort) |
| `freezing_predictions/` (98 CSVs) | equipo_project `Freezing_predictions_light/` | Fig 2 |
| `moseq_syllables_per_frame.csv.gz` (~15 MB) | `Coping_data2.zip::COping/moseq_df_final.csv` | Fig 2 |
| `updated_results.pkl.gz` (~63 MB) | `Coping_data2.zip::COping/updated_results.pkl` | Fig 5, Supp 3 |

The optional `Behavioral_clusters.json` is not bundled — Fig 4 & 6 fall back to
the built-in `CLUSTER_MAP` in each script when it is absent.

The scripts resolve these files from `data/source/` first; if a file is missing
they fall back to the original `Coping_data.zip` / `Coping_data2.zip` archives.
Override locations with `COPING_DYNAMICS_SOURCE_DIR`, `COPING_DYNAMICS_DOWNLOADS`,
`COPING_DATA_ZIP`, `COPING_DATA2_ZIP`.

Regenerate every figure with:

```powershell
python scripts\run_all_figures.py
```

## Manuscript data workbooks

`Raw_data.xlsx` and `Statistical_report.xlsx` are compact manuscript-facing
exports (raw tabular values + statistical summaries: SimBA validation, freezing
time-course, cluster frequency/time-course, Figure 4 & 6 dynamics metrics,
Supplementary Figure 3 distance metrics, syllable/freezing overlap, and archived
statistical reports). Regenerate them with:

```powershell
python scripts\data_exports\build_freezing_data_workbooks.py
```

The exporter reads from the same source archives (`Coping_data.zip` /
`Coping_data2.zip`) and stores only curated tabular values in the repo.
