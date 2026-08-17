"""Import a legacy statistical workbook into transparent repository sources.

This is a one-time migration utility for values whose original analysis inputs
are no longer available.  The generated JSON/JSONL files are data sources, not
an Excel template: the report builder recreates the workbook without reading or
copying the supplied ``.xlsx`` file.

Run:
    python scripts/import_legacy_statistical_report.py path/to/report.xlsx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import openpyxl

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "data" / "raw" / "manuscript_tables" / "statistical_report"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, help="Legacy statistical report workbook")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.workbook.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.load_workbook(source, data_only=False)
    layout = {"sheets": []}

    cells_path = args.output_dir / "reported_cells.jsonl"
    with cells_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sheet_index, worksheet in enumerate(workbook.worksheets):
            layout["sheets"].append(
                {
                    "index": sheet_index,
                    "title": worksheet.title,
                    "show_grid_lines": worksheet.sheet_view.showGridLines,
                    "merged_ranges": [str(item) for item in worksheet.merged_cells.ranges],
                }
            )
            for cell in sorted(worksheet._cells.values(), key=lambda item: (item.row, item.column)):
                if cell.value is None:
                    continue
                record = {
                    "sheet": worksheet.title,
                    "coordinate": cell.coordinate,
                    "data_type": cell.data_type,
                    "value": cell.value,
                }
                handle.write(json.dumps(record, ensure_ascii=True, allow_nan=True) + "\n")

    layout_path = args.output_dir / "reported_layout.json"
    layout_path.write_text(json.dumps(layout, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Imported {source.name} into {args.output_dir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
