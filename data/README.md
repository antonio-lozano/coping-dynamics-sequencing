# Data

All inputs needed to regenerate the data-derived manuscript figures are bundled
under `data/source/` and tracked in git. The two large inputs, a per-frame
syllable table and a results pickle, are stored gzip-compressed; pandas reads
them transparently.

## Figure Source Data

| File | Original source | Used by |
| --- | --- | --- |
| `syllable_usage_per_timebin_30s.csv` | `Coping_data2.zip::COping/Syllable_per_timebin_final(30s).csv` | Fig 3, Fig 5 |
| `syllable_usage_per_timebin_250ms.csv` | `Coping_data2.zip::COping/Syllable_per_timebin_final(250ms).csv` | Fig 4, Fig 6 |
| `bfl_scores.xlsx` | `Coping_data.zip::CSVs/BFL_scores.xlsx` | Fig 5 |
| `animal_groups.csv` | project index table | Fig 2 |
| `syllable_classification_metrics.csv` | exported syllable metrics | Fig 2 |
| `freezing_overlap_by_group.svg` | exported overlap panel | Fig 2 |
| `freezing_predictions/` | supervised freezing predictions | Fig 2 |
| `moseq_syllables_per_frame.csv.gz` | per-frame MoSeq syllable table | Fig 2 |
| `updated_results.pkl.gz` | keypoint-MoSeq results pickle | Fig 5, Supp 3 |
| `supplementary_figure1_tracking_clusters.csv` | exported manuscript source table | Supp 1 |

Figure 4 and Figure 6 use the built-in cluster maps in their generator scripts.

Regenerate the data-derived figures with:

```powershell
python scripts\run_all_figures.py
```

## Report Workbook

The final styled statistical workbook lives at
`report/STATISTICAL_REPORT_FINAL.xlsx` and is rebuilt by:

```powershell
python scripts\build_statistical_report.py
```
