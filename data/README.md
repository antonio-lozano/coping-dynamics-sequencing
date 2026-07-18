# Data

The data directory is split into immutable inputs and generated analysis-ready
tables.

## Raw Inputs

`data/raw/` contains the tracked inputs needed to rebuild the data-derived
figures and reports.

| File | Original source | Used by |
| --- | --- | --- |
| `syllable_usage_per_timebin_30s.csv` | `Coping_data2.zip::COping/Syllable_per_timebin_final(30s).csv` | Fig 3, Fig 5, derived tables |
| `syllable_usage_per_timebin_250ms.csv` | `Coping_data2.zip::COping/Syllable_per_timebin_final(250ms).csv` | Fig 4, Fig 6 |
| `bfl_scores.xlsx` | `Coping_data.zip::CSVs/BFL_scores.xlsx` | Fig 5 |
| `animal_groups.csv` | project index table | Fig 2 |
| `syllable_classification_metrics.csv` | exported syllable metrics | Fig 2 |
| `freezing_overlap_by_group.csv` | exported overlap table | Fig 2 |
| `freezing_predictions/` | supervised freezing predictions | Fig 2 |
| `moseq_syllables_per_frame.csv.gz` | per-frame MoSeq syllable table | Fig 2, Fig 7 classifier |
| `updated_results.pkl.gz` | keypoint-MoSeq results pickle | Fig 5, Supp Fig 3 |

## Derived Data

`data/derived/` contains generated analysis-ready tables. Rebuild them from
`data/raw/` with:

```powershell
python scripts\derive_tables\cluster_tables.py
python scripts\derive_tables\tracking_exclusions.py
```

Key derived files:

- `cluster_frequency_per_animal.csv`
- `cluster_timecourse_per_animal.csv`
- `s0s28_timecourse_per_animal.csv`
- `supplementary_figure1_tracking_clusters.csv`
- `tracking_exclusions_per_animal.csv`

Figure 4 and Figure 6 use the built-in cluster maps in their generator scripts.

## Full Rebuild

Regenerate all derived tables, figures, workbooks, the manifest, and checks with:

```powershell
python scripts\run_all.py
```

## Report Workbooks

The manuscript workbooks live under `report/`:

- `report/raw_data.xlsx`
- `report/statistical_report.xlsx`

They are rebuilt by:

```powershell
python scripts\build_raw_data_workbook.py
python scripts\build_statistical_report.py
```
