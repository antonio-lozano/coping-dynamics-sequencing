# Report Workbooks

`STATISTICAL_REPORT_FINAL.xlsx` is the current manuscript statistical report.

Rebuild it from the tracked template with:

```powershell
python scripts\build_statistical_report.py
```

`STATISTICAL_REPORT_TEMPLATE.xlsx` is the curated workbook shape used by that
script. The script replaces the Figure 4 sheets with the current MixedLM
results and restyles the workbook consistently.
