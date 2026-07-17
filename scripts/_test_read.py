"""Check test xlsx using non-streaming reader."""
import zipfile
import re

# Read the raw XML from the xlsx
path = "d:/coping-dynamics-sequencing/results/statistical_reports/_test.xlsx"
with zipfile.ZipFile(path) as z:
    print("Files in zip:", z.namelist())
    # Read sheet1
    for name in z.namelist():
        if 'sheet' in name.lower() and name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            print(f"\n{name} (first 2000 chars):")
            print(content[:2000])
