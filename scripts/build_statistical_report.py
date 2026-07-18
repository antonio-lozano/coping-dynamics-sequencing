"""Build the manuscript statistical report workbook from tracked outputs.

The workbook is generated from scratch; no Excel template is required.

Run: python scripts/build_statistical_report.py
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import openpyxl
import pandas as pd
import statsmodels.formula.api as smf
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "report" / "statistical_report.xlsx"
STATISTICS_DIR = REPO / "results" / "statistics"

ACCENT = "FF4D4D4D"
ACCENT_SUB = "FF6E6E6E"
SIG = "FFF2C94C"
WHITE = Font(color="FFFFFFFF", bold=True, size=10)
WHITE_SM = Font(color="FFFFFFFF", bold=True, size=9)
BOLD = Font(bold=True)
CENTER = Alignment(horizontal="center")
FILL_ACCENT = PatternFill("solid", fgColor=ACCENT)
FILL_SUB = PatternFill("solid", fgColor=ACCENT_SUB)
FILL_SIG = PatternFill("solid", fgColor=SIG)

HDR = ["Parameter", "Coef.", "Std. Err.", "z", "P>|z|", "[0.025", "0.975]"]
PARAM_LABEL = {"Intercept": "Intercept", "Condition[T.ELS]": "Stress", "Experiment": "Experiment"}


def load_fig4_module():
    spec = importlib.util.spec_from_file_location(
        "fig4",
        REPO / "scripts" / "generate_figures" / "figure_4_diversity_dynamics.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Figure 4 generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit(df: pd.DataFrame, metric: str, combined: bool) -> tuple[list[list[object]], int]:
    d = df.dropna(subset=[metric]).copy()
    d["Condition"] = pd.Categorical(d["group"], categories=["Control", "ELS"])
    formula = f"{metric} ~ Condition + Experiment" if combined else f"{metric} ~ Condition"
    model = smf.mixedlm(formula, d, groups=d["Animal"]).fit(reml=False)
    ci = model.conf_int()
    rows = []
    for param in model.params.index:
        if param == "Group Var":
            continue
        rows.append(
            [
                PARAM_LABEL.get(param, param),
                float(model.params[param]),
                float(model.bse[param]),
                float(model.tvalues[param]),
                float(model.pvalues[param]),
                float(ci.loc[param, 0]),
                float(ci.loc[param, 1]),
            ]
        )
    return rows, int(model.nobs)


def style_widths(ws) -> None:
    for col_idx in range(1, ws.max_column + 1):
        letter = get_column_letter(col_idx)
        max_len = 10
        for cell in ws[letter]:
            if cell.value is not None:
                max_len = max(max_len, min(len(str(cell.value)), 55))
        ws.column_dimensions[letter].width = max_len + 2


def write_block(ws, top: int, left: int, title: str, table_source: pd.DataFrame, metric: str) -> None:
    c0 = left
    title_cell = ws.cell(top, c0, title)
    title_cell.fill = FILL_ACCENT
    title_cell.font = WHITE
    for j in range(1, 7):
        ws.cell(top, c0 + j).fill = FILL_ACCENT

    row = top + 1
    exp_specs = [
        ("Combined datasets", True, table_source),
        ("Experiment 1 (Sanguino-Gomez & Krugers, 2024)", False, table_source[table_source["Experiment"] == 1]),
        ("Experiment 2 (Sanguino-Gomez et al., 2024)", False, table_source[table_source["Experiment"] == 3]),
    ]
    for label, combined, src in exp_specs:
        section = ws.cell(row, c0, label)
        section.fill = FILL_SUB
        section.font = WHITE_SM
        for j in range(1, 7):
            ws.cell(row, c0 + j).fill = FILL_SUB
        row += 1

        for j, header in enumerate(HDR):
            cell = ws.cell(row, c0 + j, header)
            cell.fill = FILL_ACCENT
            cell.font = WHITE_SM
            cell.alignment = CENTER
        row += 1

        try:
            rows, n_obs = fit(src, metric, combined)
        except Exception as exc:
            ws.cell(row, c0, f"model failed: {exc}")
            row += 2
            continue

        for values in rows:
            is_effect = values[0] == "Stress"
            for j, value in enumerate(values):
                cell = ws.cell(row, c0 + j, round(value, 6) if isinstance(value, float) else value)
                if j == 4 and is_effect and isinstance(value, float) and value < 0.05:
                    cell.fill = FILL_SIG
                    cell.font = BOLD
            row += 1
        ws.cell(row, c0, "No. Observations")
        ws.cell(row, c0 + 1, n_obs)
        row += 2


def add_fig4_sheets(wb: openpyxl.Workbook) -> None:
    fig4 = load_fig4_module()
    _pred, _pred_seq, full_seq, meta = fig4.load_sequences()
    metrics, _usage = fig4.compute_frequency_metrics(full_seq, meta)
    bouts = fig4.bout_table(full_seq, meta)
    transitions = fig4.transition_metrics(full_seq, meta)
    mean_bouts = (
        bouts.groupby(["Animal", "group", "Experiment", "cluster"], as_index=False)["bout_duration"]
        .mean()
    )

    fig4_sheets = {
        "Fig4_frequency_metrics": [
            ("Simpson Index", metrics, "simpson"),
            ("Shannon entropy", metrics, "shannon"),
            ("Evenness", metrics, "evenness"),
            ("Cumulative usage index", metrics, "cui"),
        ],
        "Fig4_transition_metrics": [
            ("Lempel-Ziv complexity", transitions, "lz"),
            ("Recurrence rate", transitions, "recurrence"),
            ("Determinism", transitions, "determinism"),
            ("Markov entropy", transitions, "markov"),
        ],
        "Fig4_bout_duration": [
            (cluster, mean_bouts[mean_bouts["cluster"] == cluster].rename(columns={"bout_duration": cluster}), cluster)
            for cluster in ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]
        ],
    }

    for sheet_name, blocks in fig4_sheets.items():
        ws = wb.create_sheet(sheet_name)
        left = 1
        for title, source, metric in blocks:
            write_block(ws, 1, left, title, source, metric)
            left += 9
        style_widths(ws)


def safe_sheet_name(path: Path) -> str:
    name = path.stem
    replacements = {
        "stats_figure": "stats_fig",
        "MixedLM": "mixedlm",
        "manuscript": "ms",
        "corrected": "correct",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name[:31]


def write_dataframe_sheet(wb: openpyxl.Workbook, sheet_name: str, df: pd.DataFrame) -> None:
    ws = wb.create_sheet(sheet_name)
    for col_idx, column in enumerate(df.columns, 1):
        cell = ws.cell(1, col_idx, column)
        cell.fill = FILL_ACCENT
        cell.font = WHITE_SM
        cell.alignment = CENTER

    p_columns = {i + 1 for i, col in enumerate(df.columns) if col.lower() in {"p", "p_value", "p_bh_fdr", "bh_fdr"}}
    for row_idx, row in enumerate(df.itertuples(index=False), 2):
        for col_idx, value in enumerate(row, 1):
            if pd.isna(value):
                value = None
            cell = ws.cell(row_idx, col_idx, value)
            if col_idx in p_columns and isinstance(value, (int, float)) and value < 0.05:
                cell.fill = FILL_SIG
                cell.font = BOLD
    ws.freeze_panes = "A2"
    style_widths(ws)


def add_statistics_csv_sheets(wb: openpyxl.Workbook) -> None:
    used = set(wb.sheetnames)
    for csv_path in sorted(STATISTICS_DIR.glob("*.csv")):
        sheet_name = safe_sheet_name(csv_path)
        base = sheet_name
        idx = 2
        while sheet_name in used:
            suffix = f"_{idx}"
            sheet_name = f"{base[:31 - len(suffix)]}{suffix}"
            idx += 1
        used.add(sheet_name)
        write_dataframe_sheet(wb, sheet_name, pd.read_csv(csv_path))


def add_readme_sheet(wb: openpyxl.Workbook) -> None:
    ws = wb.active
    ws.title = "README"
    rows = [
        ["Coping dynamics sequencing statistical report"],
        ["Generated by", "scripts/build_statistical_report.py"],
        ["Output", OUT.relative_to(REPO).as_posix()],
        ["Source statistics directory", STATISTICS_DIR.relative_to(REPO).as_posix()],
        ["Figure 4 model", "Default MixedLM fits regenerated directly from data/raw."],
        ["Significance highlight", "Mustard fill marks p < 0.05 in p-value columns."],
        ["Manuscript text audit", "results/statistics/manuscript_results_text.md"],
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


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    add_readme_sheet(wb)
    add_fig4_sheets(wb)
    add_statistics_csv_sheets(wb)
    wb.save(OUT)
    print(f"Saved: {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
