"""Check Fig.4 sheet using read_only mode."""
import openpyxl
wb = openpyxl.load_workbook(
    "d:/coping-dynamics-sequencing/results/statistical_reports/Statistical_report.xlsx",
    read_only=True, data_only=True
)
ws = wb["Fig.4_Diversity_Dynamics"]
count = 0
for row in ws.iter_rows():
    vals = [c.value for c in row]
    if any(v is not None for v in vals):
        print(f"  row {row[0].row}: {vals[:5]}")
        count += 1
    if count > 80:
        break
wb.close()
print(f"\nTotal non-empty rows shown: {count}")
