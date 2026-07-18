"""Build the manuscript raw-data workbook from tracked raw/derived files.

The workbook indexes every tracked data artifact and includes tabular sheets for
the manuscript-scale CSV/XLSX inputs and generated source-data tables. Large
compressed objects remain in `data/raw/` and are referenced in the inventory
rather than duplicated into Excel.

Run: python scripts/build_raw_data_workbook.py
"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "report" / "raw_data.xlsx"

RAW_DIR = REPO / "data" / "raw"
DERIVED_DIR = REPO / "data" / "derived"
SOURCE_DATA_DIR = REPO / "results" / "source_data"

ACCENT = "FF4D4D4D"
WHITE = Font(color="FFFFFFFF", bold=True, size=10)
BOLD = Font(bold=True)
FILL_ACCENT = PatternFill("solid", fgColor=ACCENT)
CENTER = Alignment(horizontal="center")

SHEET_INPUTS = [
    RAW_DIR / "animal_groups.csv",
    RAW_DIR / "freezing_predictions_index.csv",
    RAW_DIR / "freezing_overlap_by_group.csv",
    RAW_DIR / "syllable_classification_metrics.csv",
    RAW_DIR / "syllable_usage_per_timebin_30s.csv",
    RAW_DIR / "syllable_usage_per_timebin_250ms.csv",
    DERIVED_DIR / "cluster_frequency_per_animal.csv",
    DERIVED_DIR / "cluster_timecourse_per_animal.csv",
    DERIVED_DIR / "s0s28_timecourse_per_animal.csv",
    DERIVED_DIR / "supplementary_figure1_tracking_clusters.csv",
    DERIVED_DIR / "tracking_exclusions_per_animal.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path: Path) -> int | None:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
                return max(sum(1 for _ in handle) - 1, 0)
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return None
    return None


def sheet_name(path: Path) -> str:
    name = path.stem.replace("source_data_", "src_")
    replacements = {
        "syllable_usage_per_timebin_": "syllable_",
        "cluster_frequency_per_animal": "cluster_frequency",
        "cluster_timecourse_per_animal": "cluster_timecourse",
        "supplementary_figure1_tracking_clusters": "supp_fig1_tracking",
        "tracking_exclusions_per_animal": "tracking_exclusions",
        "syllable_classification_metrics": "precision_recall",
        "freezing_overlap_by_group": "freezing_overlap",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name[:31]


def style_widths(ws) -> None:
    for col_idx in range(1, ws.max_column + 1):
        letter = get_column_letter(col_idx)
        max_len = 10
        for cell in ws[letter]:
            if cell.value is not None:
                max_len = max(max_len, min(len(str(cell.value)), 55))
        ws.column_dimensions[letter].width = max_len + 2


def write_dataframe(ws, df: pd.DataFrame) -> None:
    for col_idx, column in enumerate(df.columns, 1):
        cell = ws.cell(1, col_idx, column)
        cell.fill = FILL_ACCENT
        cell.font = WHITE
        cell.alignment = CENTER
    for row_idx, row in enumerate(df.itertuples(index=False), 2):
        for col_idx, value in enumerate(row, 1):
            ws.cell(row_idx, col_idx, None if pd.isna(value) else value)
    ws.freeze_panes = "A2"
    style_widths(ws)


def add_readme(wb: openpyxl.Workbook) -> None:
    ws = wb.active
    ws.title = "README"
    rows = [
        ["Coping dynamics sequencing raw-data workbook"],
        ["Build script", "scripts/build_raw_data_workbook.py"],
        ["Output", OUT.relative_to(REPO).as_posix()],
        ["Raw inputs", "data/raw/"],
        ["Generated analysis-ready data", "data/derived/"],
        ["Figure source data", "results/source_data/"],
        ["Large full inputs", "Compressed files remain in data/raw and are indexed here."],
        ["Freezing prediction light table", "data/raw/freezing_predictions_light.csv.gz"],
    ]
    for row_idx, row in enumerate(rows, 1):
        for col_idx, value in enumerate(row, 1):
            cell = ws.cell(row_idx, col_idx, value)
            if row_idx == 1:
                cell.fill = FILL_ACCENT
                cell.font = WHITE
            elif col_idx == 1:
                cell.font = BOLD
    style_widths(ws)


def add_inventory(wb: openpyxl.Workbook) -> None:
    rows = []
    for base, category in [
        (RAW_DIR, "raw_input"),
        (DERIVED_DIR, "derived_data"),
        (SOURCE_DATA_DIR, "figure_source_data"),
    ]:
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rows.append(
                {
                    "path": path.relative_to(REPO).as_posix(),
                    "category": category,
                    "bytes": path.stat().st_size,
                    "rows_if_tabular": count_rows(path),
                    "sha256": sha256(path),
                }
            )
    ws = wb.create_sheet("data_inventory")
    write_dataframe(ws, pd.DataFrame(rows))


def add_freezing_prediction_index(wb: openpyxl.Workbook) -> None:
    rows = []
    pred_dir = RAW_DIR / "freezing_predictions"
    for path in sorted(pred_dir.glob("*.csv")):
        rows.append(
            {
                "file": path.name,
                "path": path.relative_to(REPO).as_posix(),
                "rows": count_rows(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    ws = wb.create_sheet("freezing_predictions")
    write_dataframe(ws, pd.DataFrame(rows))


def add_csv_sheets(wb: openpyxl.Workbook) -> None:
    used = set(wb.sheetnames)
    for path in SHEET_INPUTS + sorted(SOURCE_DATA_DIR.glob("*.csv")):
        name = sheet_name(path)
        base = name
        idx = 2
        while name in used:
            suffix = f"_{idx}"
            name = f"{base[:31 - len(suffix)]}{suffix}"
            idx += 1
        used.add(name)
        ws = wb.create_sheet(name)
        write_dataframe(ws, pd.read_csv(path))


def add_bfl_scores(wb: openpyxl.Workbook) -> None:
    path = RAW_DIR / "bfl_scores.xlsx"
    if not path.exists():
        return
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        name = f"bfl_{sheet}"[:31]
        ws = wb.create_sheet(name)
        write_dataframe(ws, pd.read_excel(xls, sheet_name=sheet))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    add_readme(wb)
    add_inventory(wb)
    add_freezing_prediction_index(wb)
    add_csv_sheets(wb)
    add_bfl_scores(wb)
    wb.save(OUT)
    print(f"Saved: {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
