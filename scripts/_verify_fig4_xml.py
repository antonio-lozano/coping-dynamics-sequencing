"""Read Fig.4 sheet XML directly to confirm data was written."""
import zipfile
import re
from pathlib import Path

path = "d:/coping-dynamics-sequencing/results/statistical_reports/Statistical_report.xlsx"
with zipfile.ZipFile(path) as z:
    sheets = [n for n in z.namelist() if 'worksheets/sheet' in n and n.endswith('.xml')]
    print("Worksheet files:", sheets)
    # Find Fig.4 sheet by checking workbook.xml
    wb_xml = z.read('xl/workbook.xml').decode('utf-8')
    sheet_matches = re.findall(r'name="([^"]+)"[^/]*/>', wb_xml)
    print("Sheet names in workbook.xml:", sheet_matches)

    # The last sheet is Fig.4 (6th sheet)
    # sheets list: sheet1..sheet6 (or in order)
    # Find which file is sheet 6
    rel_xml = z.read('xl/_rels/workbook.xml.rels').decode('utf-8')
    print("\nRels:", rel_xml[:1000])

    # Read the last sheet XML
    if len(sheets) >= 6:
        last_sheet = sheets[-1]
        content = z.read(last_sheet).decode('utf-8')
        # Extract sharedStrings
        ss = z.read('xl/sharedStrings.xml').decode('utf-8')
        strings = re.findall(r'<t[^>]*>([^<]*)</t>', ss)

        # Parse rows and cells from sheet
        rows = re.findall(r'<row r="(\d+)"[^>]*>(.*?)</row>', content, re.DOTALL)
        print(f"\n{last_sheet}: {len(rows)} rows with data")
        for row_num, row_content in rows[:30]:
            cells = re.findall(r'<c r="([^"]+)"[^>]*t="s"[^>]*><v>(\d+)</v></c>', row_content)
            cells_n = re.findall(r'<c r="([^"]+)"(?![^>]*t="s")[^>]*><v>([^<]+)</v></c>', row_content)
            vals = [(ref, strings[int(idx)]) for ref, idx in cells]
            vals += [(ref, val) for ref, val in cells_n]
            if vals:
                print(f"  Row {row_num}: {vals[:4]}")
