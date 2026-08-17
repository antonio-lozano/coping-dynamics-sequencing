"""Build the manuscript statistical report workbook in final panel order.

The workbook is generated from tracked repository outputs and follows the
panel-centered structure of the manuscript statistical report supplied for
submission.

Run: python scripts/build_statistical_report.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from save_deterministic import save_workbook  # noqa: E402

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
REPORT_OUT = REPO / "report" / "statistical_report.xlsx"
STATISTICS_DIR = REPO / "statistics"
RAW_DIR = REPO / "data" / "raw"
PROCESSED_DIR = REPO / "data" / "processed"

TEXT = "FF4D4D4D"
TITLE_FILL = PatternFill("solid", fgColor="FFEDEDED")
HEADER_FILL = PatternFill("solid", fgColor="FFF5F5F5")
SIG_FILL = PatternFill("solid", fgColor="FFF2CEEF")
TREND_FILL = PatternFill("solid", fgColor="FFB8CCE4")
BODY_FONT = Font(color=TEXT, size=10)
TITLE_FONT = Font(color=TEXT, bold=True, size=10)
HEADER_FONT = Font(color=TEXT, bold=True, size=10)
SIG_FONT = Font(color=TEXT, bold=True, size=10)
CENTER = Alignment(horizontal="center")

HEADERS = ["Parameter", "Coef.", "Std. Err.", "z", "P>|z|", "P_BH_FDR", "[0.025", "0.975]"]
COMBINED_ANALYSIS = "Combined full dataset"
DATASET_LABELS = {
    1: "Sanguino-Gomez & Krugers",
    3: "Sanguino-Gomez et al.",
}


def analysis_label(experiment: int | None) -> str:
    return COMBINED_ANALYSIS if experiment is None else DATASET_LABELS[experiment]


def section_title(title: str, experiment: int | None) -> str:
    return f"{title} - {analysis_label(experiment)}"


PARAM_LABELS = {
    "Intercept": "Intercept",
    "Condition[T.ELS]": "Stress",
    "Experiment": "Dataset covariate",
    "C(group, Treatment('Control'))[T.ELS]": "Stress",
    "C(group_ext, Treatment('ELS'))[T.Control]": "Control /ELS",
    "C(group_ext, Treatment('ELS'))[T.ELS resilient]": "ELS/Resilient",
    "C(group_ext, Treatment('ELS'))[T.Control]:time_bin_numeric": "Control x time",
    "C(group_ext, Treatment('ELS'))[T.ELS resilient]:time_bin_numeric": "ELS resilient x time",
    "time_bin_numeric": "Time",
}


# Cluster block order used by the submitted report's Figure 3 and Figure 5
# sheets. It is not the order the derived CSVs happen to carry, and the sheets
# name the same clusters differently (Freeze/Freezing, Climb/Climbing, ...).
CLUSTER_ORDER = ["Freeze", "Jump", "Locomotion", "Climb", "Turn", "Sniff", "Groom"]
CLUSTER_ALIASES = {
    "Freezing": "Freeze",
    "Sniffing": "Sniff",
    "Grooming": "Groom",
    "Climbing": "Climb",
}


def order_clusters(labels: list) -> list:
    """Reorder behaviour-cluster blocks to the Figure 3 / Figure 5 order.

    Only those sheets use this order - Figure 4 and Figure 6 lay the same
    clusters out differently - so callers opt in rather than getting it by
    default. Labels that are not clusters keep the order they arrived in.
    """

    def rank(item):
        index, label = item
        name = CLUSTER_ALIASES.get(str(label).strip(), str(label).strip())
        return (
            (CLUSTER_ORDER.index(name), 0) if name in CLUSTER_ORDER else (len(CLUSTER_ORDER), index)
        )

    return [label for _, label in sorted(enumerate(labels), key=rank)]


def style_widths(ws: openpyxl.worksheet.worksheet.Worksheet) -> None:
    for col_idx in range(1, ws.max_column + 1):
        letter = get_column_letter(col_idx)
        max_len = 8
        for cell in ws[letter]:
            if cell.value is not None:
                max_len = max(max_len, min(len(str(cell.value)), 34))
        ws.column_dimensions[letter].width = max_len + 2


def style_populated_widths(ws: openpyxl.worksheet.worksheet.Worksheet) -> None:
    """Size populated columns without scanning all 16,384 Excel columns.

    The legacy Figure 5 time-course sheet intentionally preserves formulas in
    column XFD, so ``ws.max_column`` is not a useful iteration bound there.
    """
    populated_columns = sorted(
        {cell.column for cell in ws._cells.values() if cell.value is not None}
    )
    for col_idx in populated_columns:
        letter = get_column_letter(col_idx)
        values = [
            cell.value
            for cell in ws._cells.values()
            if cell.column == col_idx and cell.value is not None
        ]
        max_len = max([8, *(min(len(str(value)), 34) for value in values)])
        ws.column_dimensions[letter].width = max_len + 2


def format_p(value: object) -> object:
    if isinstance(value, (int, float)) and 0 < value < 1e-4:
        return float(value)
    return value


def numeric(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return value


def style_p_value(cell: openpyxl.cell.cell.Cell, value: object) -> None:
    """Highlight significant results and 0.05-0.10 trends consistently."""
    if not isinstance(value, (int, float)):
        return
    if value < 0.05:
        cell.fill = SIG_FILL
        cell.font = SIG_FONT
    elif value < 0.10:
        cell.fill = TREND_FILL
        cell.font = SIG_FONT


def apply_bh_fdr(
    blocks: list[dict],
    parameter_labels: list[str],
) -> None:
    """Apply BH-FDR across metric blocks, separately by analysis and contrast.

    Each block represents one behavior or metric and contains matching Combined,
    the two source datasets separately. This mirrors the manuscript's
    multiple-comparison families while keeping source-dataset analyses independent.
    """
    families: dict[tuple[str, str], list[tuple[pd.DataFrame, object, float]]] = {}
    normalized_labels = {label.strip() for label in parameter_labels}
    for block in blocks:
        for section in block.get("sections", [block]):
            params = section["params"]
            if "P_BH_FDR" not in params:
                params["P_BH_FDR"] = np.nan
            analysis = str(section["title"]).rsplit(" - ", 1)[-1]
            for row_index, row in params.iterrows():
                label = str(row.get("Parameter", "")).strip()
                p_value = pd.to_numeric(row.get("P>|z|"), errors="coerce")
                if label in normalized_labels and pd.notna(p_value):
                    families.setdefault((analysis, label), []).append(
                        (params, row_index, float(p_value))
                    )

    for references in families.values():
        adjusted = multipletests([item[2] for item in references], method="fdr_bh")[1]
        for (params, row_index, _), value in zip(references, adjusted):
            params.loc[row_index, "P_BH_FDR"] = float(value)


def insert_parameter_rows(params: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Insert derived contrasts before the blank/AIC/BIC footer."""
    if not rows:
        return params
    blank = params.index[params["Parameter"].astype(str).eq("")]
    split = int(blank[0]) if len(blank) else len(params)
    return pd.concat(
        [params.iloc[:split], pd.DataFrame(rows, columns=HEADERS), params.iloc[split:]],
        ignore_index=True,
    )


def scale_parameter_rows(
    params: pd.DataFrame,
    labels: dict[str, str],
    factor: float,
) -> pd.DataFrame:
    """Scale coefficients, SEs, and CIs while preserving z and p values."""
    out = params.copy()
    for old_label, new_label in labels.items():
        mask = out["Parameter"].astype(str).eq(old_label)
        out.loc[mask, "Parameter"] = new_label
        for column in ["Coef.", "Std. Err.", "[0.025", "0.975]"]:
            values = pd.to_numeric(out.loc[mask, column], errors="coerce")
            out.loc[mask, column] = values * factor
    return out


def new_sheet(wb: openpyxl.Workbook, sheet_name: str) -> openpyxl.worksheet.worksheet.Worksheet:
    ws = wb.create_sheet(sheet_name[:31])
    ws.sheet_view.showGridLines = False
    return ws


def write_title(
    ws: openpyxl.worksheet.worksheet.Worksheet, row: int, col: int, title: str, width: int
) -> None:
    ws.cell(row, col, title)
    ws.merge_cells(start_row=row, start_column=col, end_row=row, end_column=col + width - 1)
    for idx in range(col, col + width):
        cell = ws.cell(row, idx)
        cell.fill = TITLE_FILL
        cell.font = TITLE_FONT


def write_table(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    row: int,
    col: int,
    title: str,
    df: pd.DataFrame,
    *,
    header_gap: int = 2,
) -> int:
    width = max(len(df.columns), 1)
    write_title(ws, row, col, title, width)
    header_row = row + header_gap
    for offset, column in enumerate(df.columns):
        cell = ws.cell(header_row, col + offset, column)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER

    p_cols = {
        idx
        for idx, column in enumerate(df.columns)
        if str(column).lower()
        in {
            "p>|z|",
            "p",
            "p_value",
            "p_bh_fdr",
            "p_bonferroni",
            "p value",
            "p value corrected",
        }
    }
    for row_idx, values in enumerate(df.itertuples(index=False), header_row + 1):
        for offset, value in enumerate(values):
            value = numeric(value)
            cell = ws.cell(row_idx, col + offset, format_p(value))
            cell.font = BODY_FONT
            if offset in p_cols:
                style_p_value(cell, value)
    return header_row + len(df) + 2


def csv_model_rows(
    df: pd.DataFrame,
    *,
    parameter_col: str = "parameter",
    coef_col: str = "coef",
    se_col: str = "se",
    z_col: str = "z",
    p_col: str = "p_value",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        parameter = row.get(parameter_col, row.get("Parameter", row.get("contrast", "")))
        rows.append(
            {
                "Parameter": PARAM_LABELS.get(str(parameter), parameter),
                "Coef.": row.get(coef_col, row.get("Coef.", row.get("beta"))),
                "Std. Err.": row.get(se_col, row.get("Std. Err.", row.get("SE"))),
                "z": row.get(z_col, row.get("z")),
                "P>|z|": row.get(p_col, row.get("P>|z|", row.get("p"))),
                "[0.025": row.get(ci_low_col, row.get("[0.025", "")),
                "0.975]": row.get(ci_high_col, row.get("0.975]", "")),
            }
        )
    return pd.DataFrame(rows, columns=HEADERS)


def add_horizontal_blocks(
    wb: openpyxl.Workbook, sheet_name: str, blocks: list[tuple[str, pd.DataFrame]], *, gap: int = 3
) -> None:
    ws = new_sheet(wb, sheet_name)
    col = 1
    for title, df in blocks:
        write_table(ws, 1, col, title, df)
        col += len(df.columns) + gap
    style_widths(ws)


def add_vertical_tables(
    wb: openpyxl.Workbook,
    sheet_name: str,
    tables: list[tuple[str, pd.DataFrame]],
    *,
    header_gap: int = 1,
) -> None:
    """``header_gap=2`` leaves the blank spacer row under the title that most of
    the submitted single-table sheets use."""
    ws = new_sheet(wb, sheet_name)
    row = 1
    for title, df in tables:
        row = write_table(ws, row, 1, title, df, header_gap=header_gap)
    style_widths(ws)


# --- Composite metric-block serialization (matches the submitted workbook) ---------
BLOCK_PITCH = 12  # columns between the left edge of one metric block and the next
NOT_IN_REPO = "not in repository (manuscript-only)"


def model_param_table(res, label_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Serialise a fitted statsmodels result into the Parameter/Coef/.../CI table.

    The random-effect variance row is emitted under the label the manuscript
    workbook uses ("Dataset Var"); AIC/BIC follow as label/value rows.
    """
    labels = {**PARAM_LABELS, **(label_map or {})}
    ci = res.conf_int()
    rows = []
    for param in res.params.index:
        if str(param).endswith("Var") or param == "Group Var":
            rows.append(
                {
                    "Parameter": "Dataset Var",
                    "Coef.": res.params[param],
                    "Std. Err.": "",
                    "z": "",
                    "P>|z|": "",
                    "[0.025": "",
                    "0.975]": "",
                }
            )
            continue
        rows.append(
            {
                "Parameter": labels.get(str(param), param),
                "Coef.": res.params[param],
                "Std. Err.": res.bse.get(param, ""),
                "z": res.tvalues.get(param, ""),
                "P>|z|": res.pvalues.get(param, ""),
                "[0.025": ci.loc[param, 0],
                "0.975]": ci.loc[param, 1],
            }
        )
    rows.append(
        {
            "Parameter": "",
            "Coef.": "",
            "Std. Err.": "",
            "z": "",
            "P>|z|": "",
            "[0.025": "",
            "0.975]": "",
        }
    )
    for stat, attr in [("AIC", "aic"), ("BIC", "bic")]:
        value = getattr(res, attr, None)
        rows.append(
            {
                "Parameter": stat,
                "Coef.": value if value is not None else "",
                "Std. Err.": "",
                "z": "",
                "P>|z|": "",
                "[0.025": "",
                "0.975]": "",
            }
        )
    return pd.DataFrame(rows, columns=HEADERS)


def model_metadata(
    res, model_label: str, dep_var: str, data: pd.DataFrame, group_col: str | None
) -> list[tuple[str, object]]:
    """The Metric/Value metadata column that mirrors the statsmodels summary."""
    meta: list[tuple[str, object]] = [
        ("Model", model_label),
        ("Dependent Variable", dep_var),
        ("Method", "ML"),
        ("No. Observations", int(res.nobs) if hasattr(res, "nobs") else len(data)),
    ]
    if group_col is not None and group_col in data:
        sizes = data.groupby(group_col).size()
        meta += [
            ("No. Groups", int(sizes.shape[0])),
            ("Scale", getattr(res, "scale", "")),
            ("Log-Likelihood", getattr(res, "llf", "")),
            ("Min. group size", int(sizes.min())),
            ("Max. group size", int(sizes.max())),
            ("Mean group size", round(float(sizes.mean()), 1)),
            ("Converged", getattr(res, "converged", "")),
        ]
    else:
        meta += [("Scale", getattr(res, "scale", "")), ("Log-Likelihood", getattr(res, "llf", ""))]
    return meta


def descriptive_rows(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Stress / N / Mean / SD / SEM plus Cohen's d (Control vs ELS)."""
    rows = []
    stats_by_group = {}
    for grp in ["Control", "ELS"]:
        values = data.loc[data["group"] == grp, metric].dropna()
        n = int(values.shape[0])
        mean = float(values.mean()) if n else float("nan")
        sd = float(values.std(ddof=1)) if n > 1 else float("nan")
        sem = sd / np.sqrt(n) if n else float("nan")
        stats_by_group[grp] = (n, mean, sd)
        rows.append(
            {"Stress": grp, "N": n, "Mean": mean, "SD": sd, "SEM": sem, "": "", "Cohen's d": ""}
        )
    (n1, m1, s1), (n2, m2, s2) = stats_by_group["Control"], stats_by_group["ELS"]
    if n1 > 1 and n2 > 1:
        pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        rows[0]["Cohen's d"] = round((m2 - m1) / pooled, 3) if pooled else ""
    return pd.DataFrame(rows, columns=["Stress", "N", "Mean", "SD", "SEM", "", "Cohen's d"])


def write_metadata_column(
    ws, header_row: int, col: int, metadata: list[tuple[str, object]]
) -> None:
    ws.cell(header_row, col, "Metric").fill = HEADER_FILL
    ws.cell(header_row, col).font = HEADER_FONT
    ws.cell(header_row, col + 1, "Value").fill = HEADER_FILL
    ws.cell(header_row, col + 1).font = HEADER_FONT
    for offset, (label, value) in enumerate(metadata, header_row + 1):
        ws.cell(offset, col, label).font = BODY_FONT
        ws.cell(
            offset, col + 1, numeric(value) if isinstance(value, float) else value
        ).font = BODY_FONT


POSTHOC_COLS = ["Time (min)", "t stat", "P value", "P value corrected"]


def write_section(
    ws,
    top: int,
    left: int,
    *,
    title: str,
    params: pd.DataFrame,
    metadata: list[tuple[str, object]],
    header_gap: int,
    descriptive: pd.DataFrame | None = None,
    posthoc: pd.DataFrame | None = None,
    posthoc_flag: bool = False,
    meta_offset: int = 8,
) -> int:
    """Write one model section (params + optional posthoc + metadata + optional
    descriptive block). Returns the last row used so sections can be stacked."""
    header_row = top + header_gap
    write_table(ws, top, left, title, params, header_gap=header_gap)
    used = header_row + len(params)
    meta_col = left + meta_offset
    if posthoc is not None or posthoc_flag:
        head = ws.cell(header_row, left + 8, "Posthoc (Bonferroni-corrected)")
        head.fill = HEADER_FILL
        head.font = HEADER_FONT
        for offset, column in enumerate(POSTHOC_COLS):
            cell = ws.cell(header_row + 1, left + 8 + offset, column)
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
        if posthoc is not None:
            for row_idx, (_, values) in enumerate(posthoc.iterrows(), header_row + 2):
                for offset, value in enumerate(values):
                    cell = ws.cell(
                        row_idx,
                        left + 8 + offset,
                        numeric(value) if isinstance(value, float) else value,
                    )
                    cell.font = BODY_FONT
                    if offset in {2, 3}:
                        style_p_value(cell, value)
            used = max(used, header_row + 1 + len(posthoc))
        else:
            ws.cell(header_row + 2, left + 8, NOT_IN_REPO).font = BODY_FONT
            used = max(used, header_row + 2)
        meta_col = left + 13
    write_metadata_column(ws, header_row, meta_col, metadata)
    used = max(used, header_row + len(metadata))
    if descriptive is not None:
        desc_top = used + 3
        write_plain_table(ws, desc_top, left, descriptive)
        used = desc_top + len(descriptive)
    return used


def write_plain_table(ws, top: int, left: int, df: pd.DataFrame) -> None:
    """Write a header row + data rows with no merged title (used for the
    descriptive-statistics block below each model summary)."""
    for offset, column in enumerate(df.columns):
        # Spacer columns carry a blank name only to keep them distinct in the
        # frame; the sheet leaves those header cells empty.
        cell = ws.cell(top, left + offset, column if str(column).strip() else None)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER
    for row_idx, values in enumerate(df.itertuples(index=False), top + 1):
        for offset, value in enumerate(values):
            ws.cell(
                row_idx, left + offset, numeric(value) if isinstance(value, float) else value
            ).font = BODY_FONT


def add_metric_blocks_sheet(
    wb: openpyxl.Workbook,
    sheet_name: str,
    blocks: list[dict],
    *,
    header_gap: int,
    pitch: int = BLOCK_PITCH,
    first_col: int = 1,
    band_label: str | None = None,
    meta_offset: int = 8,
    columns: list[int] | None = None,
) -> None:
    """``first_col``/``band_label`` support the Fig.3A and Fig.5C layout, where
    column A carries a label for the row band and the blocks start in column B.
    ``columns`` gives explicit left edges for sheets whose pitch is not uniform."""
    ws = new_sheet(wb, sheet_name)
    if band_label is not None:
        cell = ws.cell(1, 1, band_label)
        cell.fill = TITLE_FILL
        cell.font = TITLE_FONT
    for index, block in enumerate(blocks):
        left = columns[index] if columns else first_col + index * pitch
        sections = block.get("sections", [block])
        top = 1
        for section in sections:
            bottom = write_section(
                ws,
                top,
                left,
                title=section["title"],
                params=section["params"],
                metadata=section["metadata"],
                header_gap=header_gap,
                descriptive=section.get("descriptive"),
                posthoc=section.get("posthoc"),
                posthoc_flag=section.get("posthoc_flag", False),
                meta_offset=meta_offset,
            )
            top = bottom + 3
    style_widths(ws)


TIMECOURSE_PARAM_LABELS = {
    "Intercept": "Intercept",
    "group[T.ELS]": "Stress",
    "time_bin_numeric": "Time",
    "group[T.ELS]:time_bin_numeric": "Stress x time",
    "Group Var": "Group Var",
    "experiment Var": "Dataset Var",
}


def timecourse_block(
    sub: pd.DataFrame, dep_var: str
) -> tuple[pd.DataFrame, list[tuple[str, object]]]:
    labels = {**PARAM_LABELS, **TIMECOURSE_PARAM_LABELS}
    first = sub.iloc[0]

    def pick(row, *names):
        for name in names:
            if name in row and pd.notna(row[name]):
                return row[name]
        return ""

    rows = []
    for _, row in sub.iterrows():
        param = str(row["parameter"])
        is_var = param.endswith("Var")
        rows.append(
            {
                "Parameter": labels.get(param, param),
                "Coef.": pick(row, "coef", "Coef."),
                "Std. Err.": "" if is_var else pick(row, "std_err", "se", "Std. Err."),
                "z": "" if is_var else pick(row, "z"),
                "P>|z|": "" if is_var else pick(row, "p_value", "p", "P>|z|"),
                "[0.025": "" if is_var else pick(row, "ci_low", "[0.025"),
                "0.975]": "" if is_var else pick(row, "ci_high", "0.975]"),
            }
        )
    rows.append({column: "" for column in HEADERS})
    for stat, col in [("AIC", "aic"), ("BIC", "bic")]:
        rows.append(
            {
                "Parameter": stat,
                "Coef.": pick(first, col),
                "Std. Err.": "",
                "z": "",
                "P>|z|": "",
                "[0.025": "",
                "0.975]": "",
            }
        )
    params = pd.DataFrame(rows, columns=HEADERS)

    def as_int(value):
        return int(value) if isinstance(value, (int, float)) and pd.notna(value) else ""

    metadata = [
        ("Model", "MixedLM"),
        ("Dependent Variable", dep_var),
        ("Method", "ML"),
        ("No. Observations", as_int(pick(first, "n_observations"))),
        ("No. Groups", as_int(pick(first, "n_animals"))),
        ("Log-Likelihood", pick(first, "log_likelihood")),
    ]
    return params, metadata


def timecourse_descriptive(
    tc: pd.DataFrame, cluster: str, value_col: str, time_col: str
) -> pd.DataFrame | None:
    sub = tc[tc["cluster"] == cluster]
    if sub.empty:
        return None
    rows = []
    group_values = {}
    for grp in ["Control", "ELS"]:
        grp_rows = sub[sub["group"] == grp]
        for time_value, block in grp_rows.groupby(time_col, sort=True):
            values = block[value_col].dropna()
            n = int(values.shape[0])
            mean = float(values.mean()) if n else float("nan")
            sd = float(values.std(ddof=1)) if n > 1 else float("nan")
            sem = sd / np.sqrt(n) if n else float("nan")
            rows.append(
                {
                    "Stress": grp,
                    "Time (min)": time_value / 60.0,
                    "N": n,
                    "Mean": mean,
                    "SD": sd,
                    "SEM": sem,
                    "": "",
                    "Cohen's d": "",
                }
            )
        group_values[grp] = grp_rows[value_col].dropna()
    frame = pd.DataFrame(
        rows, columns=["Stress", "Time (min)", "N", "Mean", "SD", "SEM", "", "Cohen's d"]
    )
    control, els = group_values["Control"], group_values["ELS"]
    if len(control) > 1 and len(els) > 1:
        pooled = np.sqrt(
            ((len(control) - 1) * control.std(ddof=1) ** 2 + (len(els) - 1) * els.std(ddof=1) ** 2)
            / (len(control) + len(els) - 2)
        )
        if pooled:
            frame.loc[len(frame)] = [
                "Global Cohen's d",
                round((els.mean() - control.mean()) / pooled, 3),
                "",
                "",
                "",
                "",
                "",
                "",
            ]
    return frame


def timecourse_posthoc(
    tc: pd.DataFrame, cluster: str, value_col: str, time_col: str, *, family_size: int | None = None
) -> pd.DataFrame | None:
    """Per-time-bin Control vs ELS contrast (Welch t-test) with Bonferroni
    correction - the manuscript's posthoc table. ``family_size`` is the number
    of comparisons the correction spans (all clusters x time bins when given)."""
    sub = tc[tc["cluster"] == cluster]
    if sub.empty:
        return None
    times = sorted(sub[time_col].unique())
    n_comparisons = family_size or len(times)
    rows = []
    for time_value in times:
        control = sub[(sub["group"] == "Control") & (sub[time_col] == time_value)][
            value_col
        ].dropna()
        els = sub[(sub["group"] == "ELS") & (sub[time_col] == time_value)][value_col].dropna()
        if len(control) > 1 and len(els) > 1:
            t_stat, p_value = stats.ttest_ind(control, els, equal_var=False)
            corrected = min(float(p_value) * n_comparisons, 1.0)
        else:
            t_stat, p_value, corrected = float("nan"), float("nan"), float("nan")
        rows.append(
            {
                "Time (min)": time_value / 60.0,
                "t stat": float(t_stat),
                "P value": float(p_value),
                "P value corrected": corrected,
            }
        )
    return pd.DataFrame(rows, columns=POSTHOC_COLS)


def add_timecourse_sheet(
    wb: openpyxl.Workbook,
    sheet_name: str,
    csv_path: Path,
    tc_source: Path | None,
    *,
    dep_var: str = "Percentage",
    value_col: str = "pct",
    time_col: str = "time_s",
    header_gap: int = 1,
    pitch: int = 18,
    order_blocks: bool = True,
    fdr_parameters: list[str] | None = None,
) -> None:
    df = pd.read_csv(csv_path)
    tc = pd.read_csv(tc_source) if tc_source is not None else None
    clusters = df["cluster"].drop_duplicates().tolist()
    if order_blocks:
        clusters = order_clusters(clusters)
    family_size = None
    if tc is not None:
        n_bins = tc[tc["cluster"] == clusters[0]][time_col].nunique()
        family_size = len(clusters) * n_bins
    blocks = []
    for cluster in clusters:
        if "analysis" in df.columns:
            analysis_specs = [
                ("Combined", section_title(cluster, None), None),
                ("Exp1", section_title(cluster, 1), 1),
                ("Exp3", section_title(cluster, 3), 3),
            ]
        else:
            analysis_specs = [(None, cluster, None)]

        sections = []
        for analysis, title, experiment in analysis_specs:
            if analysis is None:
                sub = df[df["cluster"] == cluster]
            else:
                sub = df[(df["cluster"] == cluster) & (df["analysis"] == analysis)]
            if sub.empty:
                continue

            params, metadata = timecourse_block(sub, dep_var)
            params = scale_parameter_rows(
                params,
                {
                    "Time": "Time (per minute)",
                    "Stress x time": "Stress x time (per minute)",
                },
                60.0,
            )
            metadata.extend(
                [
                    ("Time source unit", "minutes"),
                    ("Reported time effect", "change per minute"),
                ]
            )
            section = {"title": title, "params": params, "metadata": metadata}
            if tc is not None:
                tc_sub = tc if experiment is None else tc[tc["experiment"] == experiment]
                n_bins = tc_sub[tc_sub["cluster"] == cluster][time_col].nunique()
                section_family_size = len(clusters) * n_bins if n_bins else family_size
                section["descriptive"] = timecourse_descriptive(
                    tc_sub, cluster, value_col, time_col
                )
                section["posthoc"] = timecourse_posthoc(
                    tc_sub, cluster, value_col, time_col, family_size=section_family_size
                )
            else:
                # Resilient-group time-bin contrasts are not tracked in the repository.
                section["posthoc_flag"] = True
            sections.append(section)

        if sections:
            blocks.append({"sections": sections})
    if fdr_parameters:
        apply_bh_fdr(blocks, fdr_parameters)
    add_metric_blocks_sheet(wb, sheet_name, blocks, header_gap=header_gap, pitch=pitch)


def simba_validation_summary() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "simba_validation_manual_vs_automatic.csv")
    fit = stats.linregress(df["manual_percent"], df["automatic_percent"])
    return pd.DataFrame(
        [
            {
                "n": int(len(df)),
                "pearson_r": fit.rvalue,
                "r_squared": fit.rvalue**2,
                "p_display": "<0.001" if fit.pvalue < 0.001 else fit.pvalue,
                "slope": fit.slope,
                "intercept": fit.intercept,
                "slope_se": fit.stderr,
            }
        ]
    )


FPS = 25
BIN_SECONDS = 30

GROUND_TRUTH_BLOCKS = [
    (COMBINED_ANALYSIS, None),
    (DATASET_LABELS[1], 1),
    (DATASET_LABELS[3], 3),
]

GROUND_TRUTH_LABELS = {
    "Intercept": "Intercept",
    "condition[T.ELS]": "Stress",
    "time_bin_numeric": "Time",
    "condition[T.ELS]:time_bin_numeric": "Stress x time ",
    "experiment": "Dataset covariate (coded 1/3)",
    "Group Var": "Group Var",
    "experiment Var": "Dataset Var",
}


def freezing_time_bins() -> pd.DataFrame:
    """Per-animal freezing percentage per 30 s bin, the input to the Fig.2A-C model.

    Frames are binned at 25 fps (the acquisition rate; ``FPS`` in ``src.config``)
    and the bin index is 0-based, which is what reproduces the coefficients,
    log-likelihood and scale of the submitted report.
    """
    freezing = pd.read_csv(
        RAW_DIR / "freezing_predictions_light.csv.gz",
        usecols=["animal_id", "group", "frame", "freezing"],
    )
    freezing["animal_id"] = freezing["animal_id"].map(lambda value: f"{float(value):.1f}")
    freezing["time_bin_numeric"] = (freezing["frame"] // int(FPS * BIN_SECONDS)).astype(int)
    binned = freezing.groupby(["animal_id", "group", "time_bin_numeric"], as_index=False)[
        "freezing"
    ].mean()
    binned["freezing_pct"] = binned["freezing"] * 100.0
    binned["time_s"] = (binned["time_bin_numeric"] + 1) * BIN_SECONDS

    frequency = pd.read_csv(PROCESSED_DIR / "cluster_frequency_per_animal.csv")
    experiments = frequency[["animal_id", "experiment"]].drop_duplicates()
    experiments["animal_id"] = experiments["animal_id"].map(lambda value: f"{float(value):.1f}")
    binned = binned.merge(experiments, on="animal_id", how="inner")
    binned["condition"] = pd.Categorical(binned["group"], categories=["Control", "ELS"])
    return binned


def fit_ground_truth_model(data: pd.DataFrame, experiment: int | None):
    """Fit the freezing MixedLM for one panel of Fig.2A-C.

    The per-experiment panels fit ``freezing ~ stress * time`` with an animal
    random intercept. The combined panel additionally carries experiment both as
    a fixed covariate and as a variance component, which is what reproduces the
    submitted intercept, scale and log-likelihood.
    """
    formula = "freezing_pct ~ condition * time_bin_numeric"
    kwargs = dict(groups=data["animal_id"], re_formula="1")
    if experiment is None:
        formula += " + experiment"
        kwargs["vc_formula"] = {"experiment": "0 + C(experiment)"}
    model = smf.mixedlm(formula, data, **kwargs)
    try:
        return model.fit(reml=False)
    except Exception:
        return model.fit(reml=False, method="lbfgs")


def ground_truth_params(result) -> pd.DataFrame:
    """Serialize the fitted summary table, then append AIC/BIC below a blank row."""
    table = result.summary().tables[1]
    rows = []
    for name, row in table.iterrows():

        def value(column):
            text = str(row[column]).strip()
            if text in {"", "nan"}:
                return ""
            try:
                return float(text)
            except ValueError:
                return text

        rows.append(
            {
                "Parameter": GROUND_TRUTH_LABELS.get(str(name), str(name)),
                "Coef.": value("Coef."),
                "Std. Err.": value("Std.Err."),
                "z": value("z"),
                "P>|z|": value("P>|z|"),
                "[0.025": value("[0.025"),
                "0.975]": value("0.975]"),
            }
        )
    rows.append({column: "" for column in HEADERS})
    for label, value in [("AIC", result.aic), ("BIC", result.bic)]:
        rows.append(
            {
                "Parameter": label,
                "Coef.": float(value),
                "Std. Err.": "",
                "z": "",
                "P>|z|": "",
                "[0.025": "",
                "0.975]": "",
            }
        )
    return pd.DataFrame(rows, columns=HEADERS)


def ground_truth_metadata(result, data: pd.DataFrame) -> list[tuple[str, object]]:
    sizes = data.groupby("animal_id").size()
    return [
        ("Model", "MixedLM"),
        ("Dependent Variable", "Freezing_predictions"),
        ("Method", "ML"),
        ("No. Observations", int(len(data))),
        ("No. Groups", int(data["animal_id"].nunique())),
        ("Scale", round(float(result.scale), 4)),
        ("Log-Likelihood", round(float(result.llf), 4)),
        ("Min. group size", int(sizes.min())),
        ("Max. group size", int(sizes.max())),
        (
            "Mean group size",
            int(sizes.mean())
            if sizes.mean() == int(sizes.mean())
            else round(float(sizes.mean()), 1),
        ),
        ("Converged", "Yes" if result.converged else "No"),
    ]


def ground_truth_posthoc(data: pd.DataFrame, value_col: str = "freezing_pct") -> pd.DataFrame:
    """Per-bin Control vs ELS contrast, Bonferroni-corrected over the bins.

    Student's t (pooled variance), which is what the submitted report used - the
    groups are equal-sized here, so only the degrees of freedom differ from Welch.
    """
    times = sorted(data["time_s"].unique())
    rows = []
    for time_value in times:
        at_time = data[data["time_s"] == time_value]
        control = at_time[at_time["group"] == "Control"][value_col].dropna()
        els = at_time[at_time["group"] == "ELS"][value_col].dropna()
        if len(control) > 1 and len(els) > 1:
            t_stat, p_value = stats.ttest_ind(els, control, equal_var=True)
            corrected = min(float(p_value) * len(times), 1.0)
        else:
            t_stat, p_value, corrected = float("nan"), float("nan"), float("nan")
        rows.append(
            {
                "Time (min)": float(time_value / 60.0),
                "t stat": round(float(t_stat), 4),
                "P value": round(float(p_value), 4),
                "P value corrected": round(float(corrected), 4),
            }
        )
    return pd.DataFrame(rows, columns=POSTHOC_COLS)


def ground_truth_descriptive(data: pd.DataFrame, value_col: str = "freezing_pct") -> pd.DataFrame:
    """Control/ELS mean per bin, with the per-bin and global Control vs ELS Cohen's d.

    Laid out as the submitted sheet does: the descriptives in the first six
    columns and the effect sizes in their own Time/Cohen's d pair after a gap.
    """
    times = sorted(data["time_s"].unique())
    columns = ["Stress", "Time (min)", "N", "Mean", "SD", "SEM ", "", "Time (min) ", "Cohen's d"]
    rows = []
    for group in ["Control", "ELS"]:
        for time_value in times:
            values = data[(data["group"] == group) & (data["time_s"] == time_value)][
                value_col
            ].dropna()
            n = int(values.shape[0])
            sd = float(values.std(ddof=1)) if n > 1 else float("nan")
            rows.append(
                {
                    "Stress": group,
                    "Time (min)": float(time_value / 60.0),
                    "N": n,
                    "Mean": round(float(values.mean()), 5) if n else "",
                    "SD": round(sd, 5) if n > 1 else "",
                    "SEM ": round(sd / np.sqrt(n), 5) if n > 1 else "",
                    "": "",
                    "Time (min) ": "",
                    "Cohen's d": "",
                }
            )
    frame = pd.DataFrame(rows, columns=columns)

    def cohens_d(control: pd.Series, els: pd.Series) -> float | str:
        """Control minus ELS, pooled SD - the direction the submitted report uses."""
        if len(control) < 2 or len(els) < 2:
            return ""
        pooled = np.sqrt(
            ((len(control) - 1) * control.std(ddof=1) ** 2 + (len(els) - 1) * els.std(ddof=1) ** 2)
            / (len(control) + len(els) - 2)
        )
        return round(float((control.mean() - els.mean()) / pooled), 5) if pooled else ""

    for index, time_value in enumerate(times):
        at_time = data[data["time_s"] == time_value]
        frame.loc[index, "Time (min) "] = float(time_value / 60.0)
        frame.loc[index, "Cohen's d"] = cohens_d(
            at_time[at_time["group"] == "Control"][value_col].dropna(),
            at_time[at_time["group"] == "ELS"][value_col].dropna(),
        )
    # Session-level effect size, pooled over every animal x bin observation
    # rather than over per-animal means.
    global_d = cohens_d(
        data[data["group"] == "Control"][value_col].dropna(),
        data[data["group"] == "ELS"][value_col].dropna(),
    )
    blank = {column: "" for column in columns}
    frame.loc[len(frame)] = blank
    frame.loc[len(frame)] = {**blank, "Time (min) ": "Global Cohen's d"}
    frame.loc[len(frame)] = {**blank, "Time (min) ": global_d}
    # The effect-size pair repeats the "Time" header; the trailing space above
    # only keeps the two columns addressable while the frame is being built.
    frame.columns = [
        "Stress",
        "Time (min)",
        "N",
        "Mean",
        "SD",
        "SEM ",
        "",
        "Time (min)",
        "Cohen's d",
    ]
    return frame


def figure2_ground_truth_sections() -> list[dict]:
    data = freezing_time_bins()
    blocks = []
    for title, experiment in GROUND_TRUTH_BLOCKS:
        subset = data if experiment is None else data[data["experiment"] == experiment]
        subset = subset.copy()
        result = fit_ground_truth_model(subset, experiment)
        metadata = ground_truth_metadata(result, subset)
        metadata.extend(
            [
                ("Time source unit", "minutes"),
                ("Reported time effect", "change per minute"),
            ]
        )
        params = scale_parameter_rows(
            ground_truth_params(result),
            {
                "Time": "Time (per minute)",
                "Stress x time ": "Stress x time (per minute)",
            },
            60.0 / BIN_SECONDS,
        )
        blocks.append(
            {
                "title": title,
                "params": params,
                "metadata": metadata,
                "posthoc": ground_truth_posthoc(subset),
                "descriptive": ground_truth_descriptive(subset),
            }
        )
    return blocks


def figure2_ground_truth_stats() -> pd.DataFrame:
    df = pd.read_csv(REPO / "figure_source_data" / "figure2.csv")
    rows = []
    for _, row in df[df["effect"].isin(["time", "group:time"])].iterrows():
        parameter = "Time" if row["effect"] == "time" else "Stress x time"
        rows.append(
            {
                "experiment": row["experiment"],
                "Parameter": parameter,
                "Coef.": row["beta"],
                "Std. Err.": row["se"],
                "z": row["z"],
                "P>|z|": row["p_value"],
                "[0.025": "",
                "0.975]": "",
            }
        )
    return pd.DataFrame(rows)


def figure2_syllable_usage(experiment: int | None = None) -> pd.DataFrame:
    frames = pd.read_csv(RAW_DIR / "manuscript_tables" / "raw_data" / "Syllable_frames.csv")
    if experiment is not None:
        frames = frames[frames["experiment"] == experiment]
    total_frames = (
        frames[["animal_id", "total_video_frames"]].drop_duplicates()["total_video_frames"].sum()
    )
    n_videos = frames["animal_id"].nunique()
    counts = (
        frames.groupby("syllable", as_index=False)["frames_in_video"]
        .sum()
        .rename(columns={"syllable": "Syllable", "frames_in_video": "Total_frames_all_videos"})
    )
    videos = (
        frames.loc[frames["frames_in_video"] > 0]
        .groupby("syllable", as_index=False)["animal_id"]
        .nunique()
    )
    videos = videos.rename(columns={"syllable": "Syllable", "animal_id": "Videos_with_syllable"})
    out = counts.merge(videos, on="Syllable", how="left")
    out["Percentage_all_frames"] = out["Total_frames_all_videos"] / total_frames * 100
    out["Videos_with_syllable"] = (
        out["Videos_with_syllable"].fillna(0).astype(int).astype(str) + f"/{n_videos}"
    )
    # Most-used syllable first, as the submitted sheet lists them.
    out = out.sort_values("Total_frames_all_videos", ascending=False, kind="stable")
    return out[
        ["Syllable", "Total_frames_all_videos", "Percentage_all_frames", "Videos_with_syllable"]
    ].reset_index(drop=True)


def figure2_syllable_usage_summary(experiment: int | None = None) -> pd.DataFrame:
    all_frames = pd.read_csv(RAW_DIR / "manuscript_tables" / "raw_data" / "Syllable_frames.csv")
    global_label_space = int(all_frames["syllable"].max()) + 1
    frames = all_frames
    if experiment is not None:
        frames = frames[frames["experiment"] == experiment]
    counts = frames.groupby("syllable")["frames_in_video"].sum().sort_values(ascending=False)
    total = float(counts.sum())
    values = [
        ("Animals", int(frames["animal_id"].nunique())),
        ("Generated label space (global)", global_label_space),
        ("Label IDs represented (max + 1)", int(frames["syllable"].max()) + 1),
        ("Observed syllables", int((counts > 0).sum())),
        ("Syllables in manuscript cutoff", 38),
        ("Frames retained by top 38 (%)", float(counts.head(38).sum() / total * 100.0)),
        ("Total analyzed frames", int(total)),
    ]
    return pd.DataFrame(values, columns=["Metric", "Value"])


def figure2_overlap_summary() -> pd.DataFrame:
    """Summarize the animal-level 0+28+40 overlap values plotted in Figure 2G."""
    plotted = pd.read_csv(RAW_DIR / "freezing_overlap_by_group.csv")
    plotted["animal_id"] = (
        plotted["recording"]
        .str.extract(r"Animal[_ ](\d+_\d+)")[0]
        .str.replace("_", ".", regex=False)
    )
    experiments = pd.read_csv(PROCESSED_DIR / "cluster_frequency_per_animal.csv")[
        ["animal_id", "experiment"]
    ].drop_duplicates()
    experiments["animal_id"] = experiments["animal_id"].map(lambda value: f"{float(value):.1f}")
    plotted = plotted.merge(experiments, on="animal_id", how="left", validate="many_to_one")
    rows = []
    for analysis, subset in [
        (analysis_label(None), plotted),
        (analysis_label(1), plotted[plotted["experiment"] == 1]),
        (analysis_label(3), plotted[plotted["experiment"] == 3]),
    ]:
        control = subset.loc[subset["group"] == "Control", "overlap_pct"].dropna()
        els = subset.loc[subset["group"] == "ELS", "overlap_pct"].dropna()
        pooled = np.sqrt(
            ((len(control) - 1) * control.var(ddof=1) + (len(els) - 1) * els.var(ddof=1))
            / (len(control) + len(els) - 2)
        )
        effect = float((els.mean() - control.mean()) / pooled) if pooled else float("nan")
        for group, values in [("Control", control), ("ELS", els)]:
            rows.append(
                {
                    "Analysis": analysis,
                    "Group": group,
                    "N": int(values.count()),
                    "Mean": float(values.mean()),
                    "SD": float(values.std(ddof=1)),
                    "SEM": float(values.sem(ddof=1)),
                    "Cohen's d (ELS - Control)": effect if group == "Control" else "",
                }
            )
    return pd.DataFrame(rows)


def figure2_precision_recall_summary(experiment: int | None = None) -> pd.DataFrame:
    by_animal = pd.read_csv(RAW_DIR / "manuscript_tables" / "raw_data" / "Precision_recall.csv")
    if experiment is not None:
        by_animal = by_animal[by_animal["experiment"] == experiment]
    metrics = ["precision_percent", "recall_percent", "overlap_frames", "syllable_frames"]
    rows = []
    for syllable in sorted(by_animal["syllable"].dropna().unique()):
        at_syllable = by_animal[by_animal["syllable"] == syllable]
        effect_sizes = {}
        for metric in metrics:
            control = at_syllable.loc[at_syllable["group"] == "Control", metric].dropna()
            els = at_syllable.loc[at_syllable["group"] == "ELS", metric].dropna()
            if len(control) > 1 and len(els) > 1:
                pooled = np.sqrt(
                    ((len(control) - 1) * control.var(ddof=1) + (len(els) - 1) * els.var(ddof=1))
                    / (len(control) + len(els) - 2)
                )
                effect_sizes[metric] = (
                    float((els.mean() - control.mean()) / pooled) if pooled else float("nan")
                )
            else:
                effect_sizes[metric] = float("nan")

        for group in ["Control", "ELS"]:
            group_data = at_syllable[at_syllable["group"] == group]
            row = {"Syllable": int(syllable), "Group": group, "N": int(len(group_data))}
            for metric in metrics:
                values = group_data[metric].dropna()
                sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
                row[f"{metric}__Mean"] = float(values.mean()) if len(values) else float("nan")
                row[f"{metric}__SD"] = sd
                row[f"{metric}__SEM"] = (
                    sd / np.sqrt(len(values)) if len(values) > 1 else float("nan")
                )
                row[f"{metric}__Cohen's d"] = effect_sizes[metric] if group == "Control" else ""
            rows.append(row)
    return pd.DataFrame(rows)


PRECISION_RECALL_METRICS = [
    "precision_percent",
    "recall_percent",
    "overlap_frames",
    "syllable_frames",
]
PRECISION_RECALL_LABELS = {
    "precision_percent": "Precision_Percentage",
    "recall_percent": "Recall_Percentage",
    "overlap_frames": "Overlap_Frames",
    "syllable_frames": "Syllable_Frames",
}


def add_precision_recall_sheet(wb: openpyxl.Workbook) -> None:
    """Write Figure 2E-F with a two-level metric/statistic column header."""
    ws = new_sheet(wb, "Fig.2E-F_Precision_recall")
    row = 1
    sections = [
        (section_title("Precision/recall", None), figure2_precision_recall_summary()),
        (section_title("Precision/recall", 1), figure2_precision_recall_summary(1)),
        (section_title("Precision/recall", 3), figure2_precision_recall_summary(3)),
    ]
    subheaders = ["Mean", "SD", "SEM", "Cohen's d (ELS - Control)"]
    width = 3 + len(PRECISION_RECALL_METRICS) * len(subheaders)
    for title, frame in sections:
        write_title(ws, row, 1, title, width)
        top_header = row + 2
        lower_header = top_header + 1
        for col, label in enumerate(["Syllable", "Group", "N"], 1):
            ws.merge_cells(
                start_row=top_header, start_column=col, end_row=lower_header, end_column=col
            )
            cell = ws.cell(top_header, col, label)
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = CENTER
            ws.cell(lower_header, col).fill = HEADER_FILL

        col = 4
        for metric in PRECISION_RECALL_METRICS:
            ws.merge_cells(
                start_row=top_header, start_column=col, end_row=top_header, end_column=col + 3
            )
            for metric_col in range(col, col + 4):
                ws.cell(top_header, metric_col).fill = HEADER_FILL
            metric_cell = ws.cell(top_header, col, PRECISION_RECALL_LABELS[metric])
            metric_cell.font = HEADER_FONT
            metric_cell.alignment = CENTER
            for offset, label in enumerate(subheaders):
                cell = ws.cell(lower_header, col + offset, label)
                cell.fill = HEADER_FILL
                cell.font = HEADER_FONT
                cell.alignment = CENTER
            col += 4

        for row_idx, values in enumerate(frame.to_dict("records"), lower_header + 1):
            ws.cell(row_idx, 1, values["Syllable"]).font = BODY_FONT
            ws.cell(row_idx, 2, values["Group"]).font = BODY_FONT
            ws.cell(row_idx, 3, values["N"]).font = BODY_FONT
            col = 4
            for metric in PRECISION_RECALL_METRICS:
                for label in ["Mean", "SD", "SEM", "Cohen's d"]:
                    value = values.get(f"{metric}__{label}")
                    ws.cell(row_idx, col, numeric(value)).font = BODY_FONT
                    col += 1
        row = lower_header + len(frame) + 3
    style_widths(ws)


def syllable_timecourse(syllables: list[int]) -> pd.DataFrame:
    """Percentage of frames spent in ``syllables`` per animal per 30 s bin.

    Binned exactly as the Fig.2A-C ground truth is (25 fps, 0-based index), so
    the two sheets share a time axis.
    """
    frames = pd.read_csv(
        RAW_DIR / "moseq_syllables_per_frame.csv.gz",
        usecols=["name", "frame_index", "syllable", "group"],
    )
    frames["animal_id"] = (
        frames["name"].str.extract(r"Animal[_ ](\d+_\d+)")[0].str.replace("_", ".", regex=False)
    )
    frames["time_bin_numeric"] = (frames["frame_index"] // int(FPS * BIN_SECONDS)).astype(int)
    frames["hit"] = frames["syllable"].isin(syllables).astype(float)
    binned = frames.groupby(["animal_id", "group", "time_bin_numeric"], as_index=False)[
        "hit"
    ].mean()
    binned["pct"] = binned["hit"] * 100.0
    binned["time_s"] = (binned["time_bin_numeric"] + 1) * BIN_SECONDS

    frequency = pd.read_csv(PROCESSED_DIR / "cluster_frequency_per_animal.csv")
    experiments = frequency[["animal_id", "experiment"]].drop_duplicates()
    experiments["animal_id"] = experiments["animal_id"].map(lambda value: f"{float(value):.1f}")
    binned = binned.merge(experiments, on="animal_id", how="inner")
    binned["condition"] = pd.Categorical(binned["group"], categories=["Control", "ELS"])
    return binned


def figure2h_sections() -> list[dict]:
    """Fig.2H: the syllable-0 and syllable-0+28 timecourses, modelled like the
    other timecourse sheets (animal random intercept, experiment variance)."""
    blocks = []
    for title, syllables in [("Syllable 0", [0]), ("Syllables 0 and 28", [0, 28])]:
        data = syllable_timecourse(syllables)
        sections = []
        for section_name, experiment in [
            (section_title(title, None), None),
            (section_title(title, 1), 1),
            (section_title(title, 3), 3),
        ]:
            subset = data if experiment is None else data[data["experiment"] == experiment]
            formula = "pct ~ condition * time_bin_numeric"
            kwargs = {"groups": subset["animal_id"], "re_formula": "1"}
            if experiment is None:
                formula += " + experiment"
                kwargs["vc_formula"] = {"experiment": "0 + C(experiment)"}
            model = smf.mixedlm(formula, subset, **kwargs)
            try:
                result = model.fit(reml=False)
            except Exception:
                result = model.fit(reml=False, method="lbfgs")
            metadata = ground_truth_metadata(result, subset)
            metadata[1] = ("Dependent Variable", "Percentage")
            metadata.extend(
                [
                    ("Time source unit", "minutes"),
                    ("Reported time effect", "change per minute"),
                ]
            )
            params = ground_truth_params(result)
            params["Parameter"] = params["Parameter"].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            params = scale_parameter_rows(
                params,
                {
                    "Time": "Time (per minute)",
                    "Stress x time": "Stress x time (per minute)",
                },
                60.0 / BIN_SECONDS,
            )
            sections.append(
                {
                    "title": section_name,
                    "params": params,
                    "metadata": metadata,
                    "posthoc": ground_truth_posthoc(subset, "pct"),
                    "descriptive": ground_truth_descriptive(subset, "pct"),
                }
            )
        blocks.append({"sections": sections})
    return blocks


def figure2_freezing_syllables() -> pd.DataFrame:
    df = pd.read_csv(REPO / "figure_source_data" / "figure2.csv")
    return df[df["effect"].eq("overlap")].rename(
        columns={
            "metric": "Metric",
            "experiment": "Experiment",
            "beta": "Value",
            "se": "SE",
            "z": "z",
            "p_value": "p_value",
        }
    )[["Metric", "Experiment", "Value", "SE", "z", "p_value"]]


def add_figure2_sheets(wb: openpyxl.Workbook) -> None:
    add_metric_blocks_sheet(
        wb, "Fig.2A-C_Ground_truth ", figure2_ground_truth_sections(), header_gap=1, pitch=18
    )
    add_vertical_tables(
        wb,
        "Fig.2D_Syllable_usage",
        [
            (section_title("Syllable usage summary", None), figure2_syllable_usage_summary()),
            (section_title("Syllable usage summary", 1), figure2_syllable_usage_summary(1)),
            (section_title("Syllable usage summary", 3), figure2_syllable_usage_summary(3)),
            (section_title("Syllable usage", None), figure2_syllable_usage()),
            (section_title("Syllable usage", 1), figure2_syllable_usage(1)),
            (section_title("Syllable usage", 3), figure2_syllable_usage(3)),
        ],
        header_gap=2,
    )
    add_precision_recall_sheet(wb)
    add_vertical_tables(
        wb,
        "Fig.2G_Freezing_overlap",
        [("Freezing overlap (%)", figure2_overlap_summary())],
        header_gap=2,
    )
    add_metric_blocks_sheet(
        wb, "Fig.2H_Freezing_syllables", figure2h_sections(), header_gap=1, pitch=18
    )


def add_grouped_csv_blocks(
    wb: openpyxl.Workbook,
    sheet_name: str,
    path: Path,
    group_col: str,
    *,
    title_map: dict[str, str] | None = None,
    parameter_col: str = "parameter",
    coef_col: str = "coef",
    se_col: str = "se",
    p_col: str = "p_value",
    model_label: str = "MixedLM",
    posthoc_flag: bool = False,
    header_gap: int = 1,
    pitch: int = BLOCK_PITCH,
    order_blocks: bool = False,
    first_col: int = 1,
    band_label: str | None = None,
) -> None:
    df = pd.read_csv(path)
    blocks = []
    groups = dict(list(df.groupby(group_col, sort=False)))
    labels = order_clusters(list(groups)) if order_blocks else list(groups)
    for label in labels:
        sub = groups[label]
        title = title_map.get(str(label), str(label)) if title_map else str(label)
        params = csv_model_rows(
            sub, parameter_col=parameter_col, coef_col=coef_col, se_col=se_col, p_col=p_col
        )
        blocks.append(
            {
                "title": title,
                "params": params,
                "metadata": [("Model", model_label)],
                "posthoc_flag": posthoc_flag,
            }
        )
    add_metric_blocks_sheet(
        wb,
        sheet_name,
        blocks,
        header_gap=header_gap,
        pitch=pitch,
        first_col=first_col,
        band_label=band_label,
    )


def fit_fig3a_gee(data: pd.DataFrame, *, with_experiment: bool):
    formula = "frequency_seconds ~ C(group, Treatment('Control'))"
    if with_experiment:
        formula += " + experiment"
    return smf.gee(
        formula,
        data=data,
        groups=data["animal_id"],
        family=sm.families.NegativeBinomial(alpha=1.0),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()


def gee_param_table(result) -> pd.DataFrame:
    ci = result.conf_int()
    rows = []
    labels = {
        "Intercept": "Intercept",
        "C(group, Treatment('Control'))[T.ELS]": "Stress",
        "experiment": "Dataset covariate (coded 1/3)",
    }
    for param in result.params.index:
        rows.append(
            {
                "Parameter": labels.get(str(param), param),
                "Coef.": result.params[param],
                "Std. Err.": result.bse[param],
                "z": result.tvalues[param],
                "P>|z|": result.pvalues[param],
                "[0.025": ci.loc[param, 0],
                "0.975]": ci.loc[param, 1],
            }
        )
    return pd.DataFrame(rows, columns=HEADERS)


def add_figure3a_frequency_sheet(wb: openpyxl.Workbook) -> None:
    source = pd.read_csv(PROCESSED_DIR / "cluster_frequency_per_animal.csv")
    source["group"] = pd.Categorical(source["group"], categories=["Control", "ELS"])
    labels = order_clusters(source["cluster"].drop_duplicates().tolist())
    blocks = []
    for cluster in labels:
        cluster_data = (
            source[source["cluster"] == cluster].dropna(subset=["frequency_seconds"]).copy()
        )
        sections = []
        for title, subset, with_experiment in [
            (section_title(cluster, None), cluster_data, True),
            (section_title(cluster, 1), cluster_data[cluster_data["experiment"] == 1], False),
            (section_title(cluster, 3), cluster_data[cluster_data["experiment"] == 3], False),
        ]:
            if subset.empty or subset["frequency_seconds"].std() == 0:
                continue
            result = fit_fig3a_gee(subset, with_experiment=with_experiment)
            sections.append(
                {
                    "title": title,
                    "params": gee_param_table(result),
                    "metadata": [("Model", "GEE Negative Binomial")],
                    "descriptive": descriptive_rows(subset, "frequency_seconds"),
                }
            )
        if sections:
            blocks.append({"sections": sections})
    apply_bh_fdr(blocks, ["Stress"])
    add_metric_blocks_sheet(
        wb,
        "Fig.3A_Clusters_frequency",
        blocks,
        header_gap=1,
        pitch=13,
        first_col=2,
        band_label=COMBINED_ANALYSIS,
    )


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


def load_fig6_module():
    spec = importlib.util.spec_from_file_location(
        "fig6",
        REPO / "scripts" / "derive_tables" / "fig6_resilience_stats.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Figure 6 generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_condition_model(
    df: pd.DataFrame, metric: str, *, with_experiment: bool
) -> tuple[object, str, pd.DataFrame]:
    d = df.dropna(subset=[metric]).copy()
    d["Condition"] = pd.Categorical(d["group"], categories=["Control", "ELS"])
    formula = f"{metric} ~ Condition + Experiment" if with_experiment else f"{metric} ~ Condition"
    try:
        res = smf.mixedlm(formula, d, groups=d["Animal"]).fit(reml=False)
        return res, "MixedLM", d
    except Exception:
        return smf.ols(formula, d).fit(), "OLS fallback", d


def metric_section(
    title: str, source: pd.DataFrame, metric: str, dep_var: str, *, with_experiment: bool
) -> dict:
    res, label, data = fit_condition_model(source, metric, with_experiment=with_experiment)
    return {
        "title": title,
        "params": model_param_table(res),
        "metadata": model_metadata(res, label, dep_var, data, "Animal"),
        "descriptive": descriptive_rows(data, metric),
    }


def fig4_block(title: str, source: pd.DataFrame, metric: str, dep_var: str) -> dict:
    """A metric column with the full dataset and both source datasets."""
    return {
        "sections": [
            metric_section(
                section_title(title, None), source, metric, dep_var, with_experiment=True
            ),
            metric_section(
                section_title(title, 1),
                source[source["Experiment"] == 1],
                metric,
                dep_var,
                with_experiment=False,
            ),
            metric_section(
                section_title(title, 3),
                source[source["Experiment"] == 3],
                metric,
                dep_var,
                with_experiment=False,
            ),
        ]
    }


def add_fig4_sheets(wb: openpyxl.Workbook) -> None:
    fig4 = load_fig4_module()
    _pred, _pred_seq, full_seq, meta = fig4.load_sequences()
    metrics, _usage = fig4.compute_frequency_metrics(full_seq, meta)
    bouts = fig4.bout_table(full_seq, meta)
    transitions = fig4.transition_metrics(full_seq, meta)
    mean_bouts = bouts.groupby(["Animal", "group", "Experiment", "cluster"], as_index=False)[
        "bout_duration"
    ].mean()

    add_metric_blocks_sheet(
        wb,
        "Fig.4E-H_frequency_metrics",
        [
            fig4_block("Simpson Index", metrics, "simpson", "Percentage"),
            fig4_block("Shannon entropy", metrics, "shannon", "Percentage"),
            fig4_block("Evenness", metrics, "evenness", "Percentage"),
            fig4_block("Cumulative usage index", metrics, "cui", "Percentage"),
        ],
        header_gap=2,
    )

    overall_bouts = bouts.groupby(["Animal", "group", "Experiment"], as_index=False)[
        "bout_duration"
    ].mean()
    bout_blocks = [fig4_block("Overall", overall_bouts, "bout_duration", "MeanBoutDuration")]
    for cluster in ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]:
        sub = mean_bouts[mean_bouts["cluster"] == cluster].rename(
            columns={"bout_duration": cluster}
        )
        bout_blocks.append(fig4_block(cluster, sub, cluster, "MeanBoutDuration"))
    # The manuscript's Figure 4 bout family contains the seven behaviors; the
    # separate overall-duration summary is descriptive and is not in that FDR family.
    apply_bh_fdr(bout_blocks[1:], ["Stress"])
    add_metric_blocks_sheet(wb, "Fig.4J-Q_bout_duration", bout_blocks, header_gap=2)

    add_metric_blocks_sheet(
        wb,
        "Fig.4T-W_transition_metrics",
        [
            fig4_block("Lempel-Ziv complexity", transitions, "lz", "LZ complexity"),
            fig4_block("Determinism", transitions, "determinism", "Determinism"),
            fig4_block("Recurrence rate", transitions, "recurrence", "Recurrence"),
            fig4_block("Markov entropy", transitions, "markov", "Markov entropy"),
        ],
        header_gap=2,
    )


def attach_experiment(df: pd.DataFrame, animal_col: str) -> pd.DataFrame:
    """Attach experiment identity using the canonical per-animal cluster table."""
    lookup = pd.read_csv(PROCESSED_DIR / "cluster_frequency_per_animal.csv")[
        ["animal_id", "experiment"]
    ].drop_duplicates()
    lookup["animal_key"] = lookup["animal_id"].map(lambda value: f"{float(value):.1f}")
    out = df.copy()
    out["animal_key"] = out[animal_col].map(lambda value: f"{float(value):.1f}")
    out = out.merge(
        lookup[["animal_key", "experiment"]], on="animal_key", how="left", validate="many_to_one"
    )
    if out["experiment"].isna().any():
        missing = out.loc[out["experiment"].isna(), animal_col].drop_duplicates().tolist()
        raise ValueError(f"Missing experiment mapping for animals: {missing}")
    return out


def score_block(
    title: str, df: pd.DataFrame, value_col: str, dep_var: str, *, with_experiment: bool
) -> dict:
    """One per-animal distance-score MixedLM block using the documented model."""
    formula = f"{value_col} ~ C(group, Treatment('Control'))"
    if with_experiment:
        formula += " + experiment"
    model = smf.mixedlm(formula, df, groups=df["animal_key"])
    try:
        res = model.fit(reml=False, method="lbfgs")
    except Exception:
        res = model.fit(reml=False)
    params = model_param_table(
        res,
        {
            "C(group, Treatment('Control'))[T.ELS]": "Stress",
            "experiment": "Dataset covariate (coded 1/3)",
        },
    )
    metadata = model_metadata(res, "MixedLM", dep_var, df, "animal_key")
    metadata.append(("Random intercept", "Animal"))
    return {
        "title": title,
        "params": params,
        "metadata": metadata,
        "descriptive": descriptive_rows(df, value_col),
    }


def score_metric_block(title: str, df: pd.DataFrame, value_col: str, dep_var: str) -> dict:
    return {
        "sections": [
            score_block(section_title(title, None), df, value_col, dep_var, with_experiment=True),
            score_block(
                section_title(title, 1),
                df[df["experiment"] == 1],
                value_col,
                dep_var,
                with_experiment=False,
            ),
            score_block(
                section_title(title, 3),
                df[df["experiment"] == 3],
                value_col,
                dep_var,
                with_experiment=False,
            ),
        ]
    }


def add_figure5_sheets(wb: openpyxl.Workbook) -> None:
    scores = attach_experiment(pd.read_csv(PROCESSED_DIR / "figure5_dynamics_scores.csv"), "animal")
    add_metric_blocks_sheet(
        wb,
        "Fig.5A-B_Dynamics",
        [score_metric_block("Euclidean Distance", scores, "dynamics_score", "Euclidean")],
        header_gap=1,
    )
    add_figure5_frequency_sheet(
        wb,
    )
    add_figure5_timecourse_sheet(
        wb,
    )


RESILIENCE_GROUPS = ["Control", "ELS", "ELS resilient"]
# Row label -> contrast over [Intercept, T.ELS, T.ELS resilient].
# Labels state the subtraction direction explicitly; this avoids the ambiguous
# slash notation in the historical workbook.
RESILIENCE_CONTRASTS = [
    ("ELS vulnerable - Control", [0.0, 1.0, 0.0]),
    ("ELS resilient - Control", [0.0, 0.0, 1.0]),
    ("ELS resilient - ELS vulnerable", [0.0, -1.0, 1.0]),
]
COHEN_PAIRS = [
    ("ELS vulnerable - Control", "Control", "ELS"),
    ("ELS resilient - Control", "Control", "ELS resilient"),
    ("ELS resilient - ELS vulnerable", "ELS", "ELS resilient"),
]
# The Figure 6 sheets are laid out on a pitch of 13, except that a blank column
# was inserted before the fourth and eighth blocks, so the left edges are given
# explicitly rather than derived from a uniform pitch.
FIG6_COLUMNS_4 = [1, 14, 27, 41]
FIG6_COLUMNS_8 = [1, 14, 27, 41, 54, 67, 80, 94]
# Bout-duration blocks are titled with the short cluster name.
FIG6_BOUT_TITLES = {
    "Freezing": "Freeze",
    "Sniffing": "Sniff",
    "Grooming": "Groom",
    "Climbing": "Climb",
}


def resilience_groups() -> pd.DataFrame:
    """Per-animal Control / ELS / ELS-resilient assignment."""
    scores = pd.read_csv(PROCESSED_DIR / "figure5_dynamics_scores.csv")
    scores["Animal"] = scores["animal"].map(lambda value: f"{float(value):.1f}")
    return scores[["Animal", "group_ext"]]


def resilience_section(
    title: str,
    source: pd.DataFrame,
    metric: str,
    dep_var: str,
    *,
    with_experiment: bool,
) -> dict:
    """One Figure 6 metric block: three-group model, its three pairwise
    contrasts, and the descriptive block with a Cohen's d per pair."""
    data = source.dropna(subset=[metric]).copy()
    data["group_ext"] = pd.Categorical(data["group_ext"], categories=RESILIENCE_GROUPS)

    def fit(reference: str):
        formula = f"{metric} ~ C(group_ext, Treatment('{reference}'))"
        if with_experiment:
            formula += " + Experiment"
        try:
            return smf.mixedlm(formula, data, groups=data["Animal"]).fit(reml=False), "MixedLM"
        except Exception:
            return smf.ols(formula, data).fit(), "OLS fallback"

    control_result, control_label = fit("Control")
    els_result, els_label = fit("ELS")
    result = control_result
    model_label = control_label if control_label == els_label else f"{control_label}; {els_label}"

    rows = [
        {
            "Parameter": "Intercept",
            "Coef.": float(result.params["Intercept"]),
            "Std. Err.": float(result.bse["Intercept"]),
            "z": float(result.tvalues["Intercept"]),
            "P>|z|": float(result.pvalues["Intercept"]),
            "[0.025": float(result.conf_int().loc["Intercept", 0]),
            "0.975]": float(result.conf_int().loc["Intercept", 1]),
        }
    ]
    contrast_specs = [
        (
            "ELS vulnerable - Control",
            control_result,
            "C(group_ext, Treatment('Control'))[T.ELS]",
        ),
        (
            "ELS resilient - Control",
            control_result,
            "C(group_ext, Treatment('Control'))[T.ELS resilient]",
        ),
        (
            "ELS resilient - ELS vulnerable",
            els_result,
            "C(group_ext, Treatment('ELS'))[T.ELS resilient]",
        ),
    ]
    for label, fitted, parameter in contrast_specs:
        low, high = fitted.conf_int().loc[parameter]
        rows.append(
            {
                "Parameter": label,
                "Coef.": float(fitted.params[parameter]),
                "Std. Err.": float(fitted.bse[parameter]),
                "z": float(fitted.tvalues[parameter]),
                "P>|z|": float(fitted.pvalues[parameter]),
                "P_BH_FDR": np.nan,
                "[0.025": float(low),
                "0.975]": float(high),
            }
        )
    for name in ["Experiment"] if with_experiment else []:
        rows.append(
            {
                "Parameter": "Dataset covariate (coded 1/3)",
                "Coef.": float(result.params[name]),
                "Std. Err.": float(result.bse[name]),
                "z": float(result.tvalues[name]),
                "P>|z|": float(result.pvalues[name]),
                "[0.025": float(result.conf_int().loc[name, 0]),
                "0.975]": float(result.conf_int().loc[name, 1]),
            }
        )
    variance = [p for p in result.params.index if str(p).endswith("Var")]
    for name in variance:
        rows.append(
            {
                "Parameter": "Dataset Var",
                "Coef.": float(result.params[name]),
                "Std. Err.": "",
                "z": "",
                "P>|z|": "",
                "[0.025": "",
                "0.975]": "",
            }
        )
    rows.append({column: "" for column in HEADERS})
    for label, value in [("AIC", result.aic), ("BIC", result.bic)]:
        rows.append(
            {
                "Parameter": label,
                "Coef.": float(value),
                "Std. Err.": "",
                "z": "",
                "P>|z|": "",
                "[0.025": "",
                "0.975]": "",
            }
        )

    sizes = data.groupby("Animal").size()
    metadata = [
        ("Model", model_label),
        ("Dependent Variable", dep_var),
        ("Method", "ML"),
        ("No. Observations", int(len(data))),
        ("No. Groups", int(data["Animal"].nunique())),
        ("Scale", round(float(result.scale), 4)),
        ("Log-Likelihood", round(float(result.llf), 4)),
        ("Min. group size", int(sizes.min())),
        ("Max. group size", int(sizes.max())),
        (
            "Mean group size",
            int(sizes.mean())
            if sizes.mean() == int(sizes.mean())
            else round(float(sizes.mean()), 1),
        ),
        ("Converged", bool(getattr(result, "converged", True))),
        ("Data level", "one row per animal"),
        ("Contrast fitting", "Control- and ELS-reference parameterizations"),
    ]
    return {
        "title": title,
        "params": pd.DataFrame(rows, columns=HEADERS),
        "metadata": metadata,
        "descriptive": resilience_descriptive(data, metric),
    }


def resilience_block(title: str, source: pd.DataFrame, metric: str, dep_var: str) -> dict:
    return {
        "sections": [
            resilience_section(
                section_title(title, None), source, metric, dep_var, with_experiment=True
            ),
            resilience_section(
                section_title(title, 1),
                source[source["Experiment"] == 1],
                metric,
                dep_var,
                with_experiment=False,
            ),
            resilience_section(
                section_title(title, 3),
                source[source["Experiment"] == 3],
                metric,
                dep_var,
                with_experiment=False,
            ),
        ]
    }


def resilience_descriptive(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-group N/Mean/SD/SEM, with a Cohen's d for each of the three pairs
    listed alongside in its own label/value pair."""
    columns = ["Stress", "N", "Mean", "SD", "SEM ", "", "Cohen's d", " "]
    summary = {}
    rows = []
    for group in RESILIENCE_GROUPS:
        values = data.loc[data["group_ext"] == group, metric].dropna()
        n = int(values.shape[0])
        sd = float(values.std(ddof=1)) if n > 1 else float("nan")
        summary[group] = values
        rows.append(
            {
                "Stress": group.replace(" ", "_"),
                "N": n,
                "Mean": round(float(values.mean()), 5) if n else "",
                "SD": round(sd, 5) if n > 1 else "",
                "SEM ": round(sd / np.sqrt(n), 5) if n > 1 else "",
                "": "",
                "Cohen's d": "",
                " ": "",
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    for index, (label, first, second) in enumerate(COHEN_PAIRS):
        a, b = summary[first], summary[second]
        frame.loc[index, "Cohen's d"] = label
        if len(a) > 1 and len(b) > 1:
            pooled = np.sqrt(
                ((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
                / (len(a) + len(b) - 2)
            )
            frame.loc[index, " "] = (
                round(float((b.mean() - a.mean()) / pooled), 5) if pooled else ""
            )
    return frame


def linear_contrast_row(result, label: str, weights: list[float]) -> dict:
    vector = np.zeros(len(result.params))
    vector[: len(weights)] = weights
    effect = float(vector @ np.asarray(result.params))
    variance = float(vector @ np.asarray(result.cov_params()) @ vector)
    se = float(np.sqrt(max(variance, 0.0)))
    z_value = effect / se if se else float("nan")
    p_value = float(2 * stats.norm.sf(abs(z_value))) if se else float("nan")
    return {
        "Parameter": label,
        "Coef.": effect,
        "Std. Err.": se,
        "z": z_value,
        "P>|z|": p_value,
        "P_BH_FDR": np.nan,
        "[0.025": effect - 1.959963984540054 * se,
        "0.975]": effect + 1.959963984540054 * se,
    }


def named_contrast_row(
    result,
    label: str,
    weights: dict[str, float],
    *,
    scale: float = 1.0,
) -> dict:
    """Return a reference-independent contrast from named model parameters."""
    names = list(result.params.index)
    vector = np.array([weights.get(name, 0.0) for name in names], dtype=float)
    missing = sorted(set(weights).difference(names))
    if missing:
        raise KeyError(f"Model is missing parameters required for {label}: {missing}")
    effect = float(vector @ np.asarray(result.params)) * scale
    covariance = np.asarray(result.cov_params())
    se = float(np.sqrt(max(float(vector @ covariance @ vector), 0.0))) * abs(scale)
    z_value = effect / se if se else float("nan")
    p_value = float(2 * stats.norm.sf(abs(z_value))) if se else float("nan")
    return {
        "Parameter": label,
        "Coef.": effect,
        "Std. Err.": se,
        "z": z_value,
        "P>|z|": p_value,
        "P_BH_FDR": np.nan,
        "[0.025": effect - 1.959963984540054 * se,
        "0.975]": effect + 1.959963984540054 * se,
    }


def figure5_frequency_section(title: str, data: pd.DataFrame, *, with_experiment: bool) -> dict:
    subset = data.copy()
    subset["group_ext"] = pd.Categorical(subset["group_ext"], categories=RESILIENCE_GROUPS)
    formula = "frequency_seconds ~ group_ext"
    if with_experiment:
        formula += " + experiment"
    result = smf.gee(
        formula,
        data=subset,
        groups=subset["animal_key"],
        family=sm.families.NegativeBinomial(alpha=1.0),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    rows = [linear_contrast_row(result, "Intercept", [1.0, 0.0, 0.0])]
    rows.extend(
        linear_contrast_row(result, label, weights) for label, weights in RESILIENCE_CONTRASTS
    )
    if with_experiment:
        experiment_weights = [0.0] * len(result.params)
        experiment_weights[-1] = 1.0
        rows.append(
            linear_contrast_row(result, "Dataset covariate (coded 1/3)", experiment_weights)
        )
    params = pd.DataFrame(rows, columns=HEADERS)
    metadata = [
        ("Model", "GEE Negative Binomial"),
        ("Dependent Variable", "Frequency"),
        ("No. Observations", int(result.nobs)),
        ("No. Groups", int(subset["animal_key"].nunique())),
    ]
    descriptive = resilience_descriptive(
        subset.rename(columns={"animal_key": "Animal"}), "frequency_seconds"
    )
    return {"title": title, "params": params, "metadata": metadata, "descriptive": descriptive}


def add_figure5_frequency_sheet(wb: openpyxl.Workbook) -> None:
    frequency = pd.read_csv(PROCESSED_DIR / "cluster_frequency_per_animal.csv")
    frequency["animal_key"] = frequency["animal_id"].map(lambda value: f"{float(value):.1f}")
    groups = resilience_groups().rename(columns={"Animal": "animal_key"})
    frequency = frequency.merge(groups, on="animal_key", how="inner", validate="many_to_one")
    blocks = []
    for cluster in order_clusters(frequency["cluster"].drop_duplicates().tolist()):
        cluster_data = frequency[frequency["cluster"] == cluster]
        blocks.append(
            {
                "sections": [
                    figure5_frequency_section(
                        section_title(cluster, None), cluster_data, with_experiment=True
                    ),
                    figure5_frequency_section(
                        section_title(cluster, 1),
                        cluster_data[cluster_data["experiment"] == 1],
                        with_experiment=False,
                    ),
                    figure5_frequency_section(
                        section_title(cluster, 3),
                        cluster_data[cluster_data["experiment"] == 3],
                        with_experiment=False,
                    ),
                ]
            }
        )
    apply_bh_fdr(blocks, ["ELS resilient - ELS vulnerable"])
    add_metric_blocks_sheet(
        wb,
        "Fig.5C_Cluster_frequency",
        blocks,
        header_gap=1,
        pitch=13,
        first_col=1,
    )


def resilience_timecourse_descriptive(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in RESILIENCE_GROUPS:
        group_data = data[data["group_ext"] == group]
        for time_value, block in group_data.groupby("time_s", sort=True):
            values = block["pct"].dropna()
            n = len(values)
            sd = float(values.std(ddof=1)) if n > 1 else float("nan")
            rows.append(
                {
                    "Stress": group.replace(" ", "_"),
                    "Time (min)": time_value / 60.0,
                    "N": n,
                    "Mean": float(values.mean()) if n else "",
                    "SD": sd if n > 1 else "",
                    "SEM": sd / np.sqrt(n) if n > 1 else "",
                }
            )
    return pd.DataFrame(rows, columns=["Stress", "Time (min)", "N", "Mean", "SD", "SEM"])


def figure5_timecourse_posthoc(data: pd.DataFrame) -> pd.DataFrame:
    """Welch pairwise tests at each time bin for all three resilience groups.

    Bonferroni and BH-FDR are applied within each dataset/behavior family over
    the three pairwise contrasts at all time bins.
    """
    pairs = [
        ("ELS vulnerable - Control", "Control", "ELS"),
        ("ELS resilient - Control", "Control", "ELS resilient"),
        ("ELS resilient - ELS vulnerable", "ELS", "ELS resilient"),
    ]
    rows = []
    analyses = [
        (analysis_label(None), data),
        (analysis_label(1), data[data["experiment"] == 1]),
        (analysis_label(3), data[data["experiment"] == 3]),
    ]
    for analysis, analysis_data in analyses:
        for cluster in order_clusters(analysis_data["cluster"].drop_duplicates().tolist()):
            cluster_data = analysis_data[analysis_data["cluster"] == cluster]
            for time_s, at_time in cluster_data.groupby("time_s", sort=True):
                for contrast, first_group, second_group in pairs:
                    first = at_time.loc[at_time["group_ext"] == first_group, "pct"].dropna()
                    second = at_time.loc[at_time["group_ext"] == second_group, "pct"].dropna()
                    if len(first) > 1 and len(second) > 1:
                        t_value, p_value = stats.ttest_ind(second, first, equal_var=False)
                    else:
                        t_value, p_value = float("nan"), float("nan")
                    rows.append(
                        {
                            "Analysis": analysis,
                            "Behavior": cluster,
                            "Contrast": contrast,
                            "Time (min)": float(time_s / 60.0),
                            "N_first": int(len(first)),
                            "N_second": int(len(second)),
                            "Mean_difference": float(second.mean() - first.mean())
                            if len(first) and len(second)
                            else float("nan"),
                            "t": float(t_value),
                            "p_value": float(p_value),
                        }
                    )

    frame = pd.DataFrame(rows)
    frame["P_Bonferroni"] = np.nan
    frame["P_BH_FDR"] = np.nan
    for _, indices in frame.groupby(["Analysis", "Behavior"], sort=False).groups.items():
        valid = frame.loc[indices, "p_value"].dropna()
        if valid.empty:
            continue
        frame.loc[valid.index, "P_Bonferroni"] = multipletests(valid, method="bonferroni")[1]
        frame.loc[valid.index, "P_BH_FDR"] = multipletests(valid, method="fdr_bh")[1]
    return frame


def figure5_timecourse_section(title: str, data: pd.DataFrame, *, with_experiment: bool) -> dict:
    subset = data.copy()
    subset["group_ext"] = pd.Categorical(subset["group_ext"], categories=RESILIENCE_GROUPS)
    formula = "pct ~ C(group_ext, Treatment('ELS')) * time_bin"
    if with_experiment:
        formula += " + C(experiment)"
    try:
        result = smf.mixedlm(formula, subset, groups=subset["animal_id"]).fit(reml=False)
        model_label = "MixedLM"
        metadata = model_metadata(result, model_label, "Percentage", subset, "animal_id")
    except Exception:
        result = smf.ols(formula, subset).fit()
        model_label = "OLS fallback"
        metadata = model_metadata(result, model_label, "Percentage", subset, None)
    labels = {
        "C(group_ext, Treatment('ELS'))[T.Control]": "Control - ELS vulnerable",
        "C(group_ext, Treatment('ELS'))[T.ELS resilient]": "ELS resilient - ELS vulnerable",
        "time_bin": "Time",
        "C(group_ext, Treatment('ELS'))[T.Control]:time_bin": "Control - ELS vulnerable x time",
        "C(group_ext, Treatment('ELS'))[T.ELS resilient]:time_bin": "ELS resilient - ELS vulnerable x time",
        "C(experiment)[T.3]": f"{DATASET_LABELS[3]} - {DATASET_LABELS[1]}",
    }
    params = model_param_table(result, labels)
    params = scale_parameter_rows(
        params,
        {
            "Time": "Time (per minute)",
            "Control - ELS vulnerable x time": "Control - ELS vulnerable x time (per minute)",
            "ELS resilient - ELS vulnerable x time": "ELS resilient - ELS vulnerable x time (per minute)",
        },
        60.0,
    )
    params = insert_parameter_rows(
        params,
        [
            named_contrast_row(
                result,
                "ELS resilient - Control",
                {
                    "C(group_ext, Treatment('ELS'))[T.Control]": -1.0,
                    "C(group_ext, Treatment('ELS'))[T.ELS resilient]": 1.0,
                },
            ),
            named_contrast_row(
                result,
                "ELS resilient - Control x time (per minute)",
                {
                    "C(group_ext, Treatment('ELS'))[T.Control]:time_bin": -1.0,
                    "C(group_ext, Treatment('ELS'))[T.ELS resilient]:time_bin": 1.0,
                },
                scale=60.0,
            ),
        ],
    )
    metadata.extend(
        [
            ("Time source unit", "minutes"),
            ("Reported time effect", "change per minute"),
        ]
    )
    return {
        "title": title,
        "params": params,
        "metadata": metadata,
        "descriptive": resilience_timecourse_descriptive(subset),
    }


def add_figure5_timecourse_sheet(wb: openpyxl.Workbook) -> None:
    timecourse = pd.read_csv(PROCESSED_DIR / "cluster_timecourse_per_animal.csv")
    timecourse["animal_key"] = timecourse["animal_id"].map(lambda value: f"{float(value):.1f}")
    groups = resilience_groups().rename(columns={"Animal": "animal_key"})
    timecourse = timecourse.merge(groups, on="animal_key", how="inner", validate="many_to_one")
    blocks = []
    for cluster in order_clusters(timecourse["cluster"].drop_duplicates().tolist()):
        cluster_data = timecourse[timecourse["cluster"] == cluster]
        blocks.append(
            {
                "sections": [
                    figure5_timecourse_section(
                        section_title(cluster, None), cluster_data, with_experiment=True
                    ),
                    figure5_timecourse_section(
                        section_title(cluster, 1),
                        cluster_data[cluster_data["experiment"] == 1],
                        with_experiment=False,
                    ),
                    figure5_timecourse_section(
                        section_title(cluster, 3),
                        cluster_data[cluster_data["experiment"] == 3],
                        with_experiment=False,
                    ),
                ]
            }
        )
    apply_bh_fdr(blocks, ["ELS resilient - ELS vulnerable x time (per minute)"])
    add_metric_blocks_sheet(wb, "Fig.5D-J_Cluster_timecourse", blocks, header_gap=1, pitch=18)
    add_vertical_tables(
        wb,
        "Fig.5D-J_Time_posthoc",
        [("Figure 5 pairwise posthoc tests by time bin", figure5_timecourse_posthoc(timecourse))],
        header_gap=2,
    )


def add_figure6_sheets(wb: openpyxl.Workbook) -> None:
    fig6 = load_fig6_module()
    source = fig6.load()

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Animal"] = out["Animal"].map(lambda value: f"{float(value):.1f}")
        out["group_ext"] = out["New_condition"].replace({"ELS_resilient": "ELS resilient"})
        return out.drop(columns=["New_condition"])

    diversity = normalize(fig6.diversity_table(source)).rename(
        columns={"Entropy": "shannon", "Evenness": "evenness", "Simpson": "simpson", "CUI": "cui"}
    )
    bouts = normalize(fig6.bout_table(source)).rename(
        columns={"Cluster": "cluster", "MeanBoutDuration": "bout_duration"}
    )
    transition = normalize(fig6.transition_table(source)).rename(
        columns={
            "LZ_complexity": "lz",
            "Recurrence": "recurrence",
            "Determinism": "determinism",
            "MarkovEntropyIdx": "markov",
        }
    )
    diversity_blocks = [
        resilience_block("Simpson Index", diversity, "simpson", "Simpson"),
        resilience_block("Shannon Entropy Index", diversity, "shannon", "Entropy"),
        resilience_block("Evenness Index", diversity, "evenness", "Evenness"),
        resilience_block("Cumulative Usage Index", diversity, "cui", "CUI"),
    ]
    apply_bh_fdr(diversity_blocks, [label for label, _ in RESILIENCE_CONTRASTS])
    add_metric_blocks_sheet(
        wb,
        "Fig.6G-K_Frequency_metrics",
        diversity_blocks,
        header_gap=1,
        meta_offset=9,
        columns=FIG6_COLUMNS_4,
    )

    bout_blocks = []
    overall = bouts.groupby(["Animal", "Experiment", "group_ext"], as_index=False, observed=True)[
        "bout_duration"
    ].mean()
    bout_blocks.append(resilience_block("Overall", overall, "bout_duration", "MeanBoutDuration"))
    for cluster in ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]:
        sub = bouts[bouts["cluster"] == cluster].rename(columns={"bout_duration": cluster})
        bout_blocks.append(
            resilience_block(
                FIG6_BOUT_TITLES.get(cluster, cluster), sub, cluster, "MeanBoutDuration"
            )
        )
    apply_bh_fdr(bout_blocks[1:], [label for label, _ in RESILIENCE_CONTRASTS])
    add_metric_blocks_sheet(
        wb,
        "Fig.6L-S_Bout_duration",
        bout_blocks,
        header_gap=1,
        meta_offset=9,
        columns=FIG6_COLUMNS_8,
    )

    transition_blocks = [
        resilience_block("Lempel-Ziv Complexity", transition, "lz", "LZ complexity"),
        resilience_block("Recurrence Rate", transition, "recurrence", "Recurrence"),
        resilience_block("Determinism", transition, "determinism", "Determinism"),
        resilience_block("Markov Entropy Index", transition, "markov", "Markov entropy"),
    ]
    apply_bh_fdr(transition_blocks, [label for label, _ in RESILIENCE_CONTRASTS])
    add_metric_blocks_sheet(
        wb,
        "Fig.6W-Z_Transition_metrics",
        transition_blocks,
        header_gap=1,
        meta_offset=9,
        columns=FIG6_COLUMNS_4,
    )


def resilience_overlap_summary() -> pd.DataFrame:
    """Animals classified as resilient by both Euclidean and each control metric."""
    euclidean = pd.read_csv(PROCESSED_DIR / "figure5_dynamics_scores.csv")
    euclidean["animal_key"] = euclidean["animal"].map(lambda value: f"{float(value):.1f}")
    euclidean_mask = euclidean["resilient_by_zero"].astype(str).str.lower().eq("true")
    euclidean_animals = set(
        euclidean.loc[(euclidean["group"] == "ELS") & euclidean_mask, "animal_key"]
    )

    supplementary = pd.read_csv(PROCESSED_DIR / "supplementary_figure3_distance_scores.csv")
    supplementary["animal_key"] = supplementary["animal"].map(lambda value: f"{float(value):.1f}")
    supplementary_mask = supplementary["resilient_by_zero"].astype(str).str.lower().eq("true")
    supplementary = supplementary.loc[(supplementary["group"] == "ELS") & supplementary_mask]

    rows = []
    denominator = len(euclidean_animals)
    for metric, metric_rows in supplementary.groupby("metric", sort=False):
        metric_animals = set(metric_rows["animal_key"])
        overlap = sorted(euclidean_animals.intersection(metric_animals), key=float)
        rows.append(
            {
                "Metric": metric,
                "Overlapping_Animals": ", ".join(overlap),
                "Overlap_Count": len(overlap),
                "Overlap_Percentage_of_Euclidean": round(100.0 * len(overlap) / denominator, 1)
                if denominator
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def add_supplementary_sheets(wb: openpyxl.Workbook) -> None:
    scores = attach_experiment(
        pd.read_csv(PROCESSED_DIR / "supplementary_figure3_distance_scores.csv"), "animal"
    )
    blocks = [
        score_metric_block(f"{metric}_score", sub, "dynamics_score", metric)
        for metric, sub in scores.groupby("metric", sort=False)
    ]
    add_metric_blocks_sheet(wb, "Suppl.Fig.3A-K", blocks, header_gap=1, pitch=11)

    add_vertical_tables(
        wb,
        "Resilience_overlap_methods",
        [("Animals overlapping with the Euclidean resilient set", resilience_overlap_summary())],
        header_gap=2,
    )


def add_figure7_sheets(wb: openpyxl.Workbook) -> None:
    """Add Figure 7 inferential/performance summaries without raw-data copies."""
    cross_cohort = pd.read_csv(STATISTICS_DIR / "figure7_cross_cohort_auc.csv")
    dataset_names = {
        "Exp1": "Sanguino-Gomez and Krugers, 2024",
        "Exp3": "Sanguino-Gomez et al., 2024",
    }
    cross_cohort["train_experiment"] = cross_cohort["train_experiment"].replace(dataset_names)
    cross_cohort["test_experiment"] = cross_cohort["test_experiment"].replace(dataset_names)
    cross_cohort = cross_cohort.rename(
        columns={"train_experiment": "train_dataset", "test_experiment": "test_dataset"}
    )
    de_tests = pd.read_csv(STATISTICS_DIR / "figure7_panel_de_auc_tests.csv")
    de_tests["test_experiment"] = de_tests["test_experiment"].replace(dataset_names)
    de_tests = de_tests.rename(columns={"test_experiment": "test_dataset"})
    add_vertical_tables(
        wb,
        "Fig.7D-E_Prediction",
        [
            ("Held-out-cohort ROC AUC", cross_cohort),
            (
                "Full-session leave-one-out ROC AUC",
                pd.read_csv(STATISTICS_DIR / "figure7_full_session_auc.csv"),
            ),
            (
                "Behaviour dynamics vs Freeze dynamics only: paired bootstrap test on "
                "the AUC difference (same held-out animals scored by both models; "
                "two-sided, 10000 resamples)",
                de_tests,
            ),
        ],
        header_gap=1,
    )
    add_vertical_tables(
        wb,
        "Fig.7F_SHAP",
        [
            (
                "Exact linear-model Shapley contributions",
                pd.read_csv(REPO / "figure_source_data" / "figure7_shapley_contributions.csv"),
            )
        ],
        header_gap=1,
    )
    add_vertical_tables(
        wb,
        "Fig.7I_Prediction_onset",
        [
            (
                "Within-cohort permutation reference across opening-session horizons",
                pd.read_csv(STATISTICS_DIR / "figure7_prediction_permutation.csv"),
            )
        ],
        header_gap=1,
    )


def validate_report_workbook(wb: openpyxl.Workbook) -> None:
    """Fail the build if manuscript-facing coverage disappears."""
    required_sheets = {
        "Fig.1A_ SimBA_validation",
        "Fig.2A-C_Ground_truth ",
        "Fig.2D_Syllable_usage",
        "Fig.2E-F_Precision_recall",
        "Fig.2G_Freezing_overlap",
        "Fig.2H_Freezing_syllables",
        "Fig.3A_Clusters_frequency",
        "Fig.3B-H_Clusters_over_time",
        "Fig.4E-H_frequency_metrics",
        "Fig.4J-Q_bout_duration",
        "Fig.4T-W_transition_metrics",
        "Fig.5A-B_Dynamics",
        "Fig.5C_Cluster_frequency",
        "Fig.5D-J_Cluster_timecourse",
        "Fig.5D-J_Time_posthoc",
        "Fig.6G-K_Frequency_metrics",
        "Fig.6L-S_Bout_duration",
        "Fig.6W-Z_Transition_metrics",
        "Fig.7D-E_Prediction",
        "Fig.7F_SHAP",
        "Fig.7I_Prediction_onset",
        "Suppl.Fig.3A-K",
    }
    missing = sorted(required_sheets.difference(wb.sheetnames))
    if missing:
        raise AssertionError(f"Statistical report is missing sheets: {missing}")

    raw_only_figure7 = {
        "Fig.7A_Classifier_accuracy",
        "Fig.7B_Classifier_SHAP",
        "Fig.7C_Confusion_matrix",
        "Fig.7G-H_Predictors",
        "Fig.7J_Family_timecourse",
        "Fig.7K-N_Metric_timecourse",
    }
    duplicated = sorted(raw_only_figure7.intersection(wb.sheetnames))
    if duplicated:
        raise AssertionError(f"Statistical report duplicates raw Figure 7 sheets: {duplicated}")

    legacy_percent_labels = []
    for sheet in wb.worksheets:
        for cell in sheet._cells.values():
            if not isinstance(cell.value, str):
                continue
            words = cell.value.replace(" ", "_").lower().split("_")
            if "percent" in words:
                legacy_percent_labels.append(f"{sheet.title}!{cell.coordinate}={cell.value}")
    if legacy_percent_labels:
        raise AssertionError(
            "Statistical report contains legacy Percent labels: " + ", ".join(legacy_percent_labels)
        )

    fdr_sheets = {
        "Fig.3A_Clusters_frequency",
        "Fig.3B-H_Clusters_over_time",
        "Fig.4J-Q_bout_duration",
        "Fig.5A-B_Dynamics",
        "Fig.5C_Cluster_frequency",
        "Fig.5D-J_Cluster_timecourse",
        "Fig.5D-J_Time_posthoc",
        "Fig.6G-K_Frequency_metrics",
        "Fig.6L-S_Bout_duration",
        "Fig.6W-Z_Transition_metrics",
        "Suppl.Fig.3A-K",
    }
    for sheet_name in sorted(fdr_sheets):
        values = [cell.value for cell in wb[sheet_name]._cells.values()]
        if "P_BH_FDR" not in values:
            raise AssertionError(f"{sheet_name} has no BH-FDR column")

    fig5_values = [cell.value for cell in wb["Fig.5D-J_Cluster_timecourse"]._cells.values()]
    direct_label = "ELS resilient - ELS vulnerable x time (per minute)"
    if fig5_values.count(direct_label) != len(CLUSTER_ORDER) * 3:
        raise AssertionError(
            "Figure 5 time-course report does not contain every full/source-dataset contrast"
        )


def build_statistical_report(output: Path = REPORT_OUT) -> None:
    """Build the single canonical full and source-dataset statistical report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    add_vertical_tables(
        wb,
        "Fig.1A_ SimBA_validation",
        [("SimBA validation correlation", simba_validation_summary())],
        header_gap=2,
    )
    add_figure2_sheets(wb)
    add_figure3a_frequency_sheet(wb)
    add_timecourse_sheet(
        wb,
        "Fig.3B-H_Clusters_over_time",
        STATISTICS_DIR / "stats_figure3_overtime.csv",
        PROCESSED_DIR / "cluster_timecourse_per_animal.csv",
        value_col="pct",
        time_col="time_s",
        header_gap=1,
        fdr_parameters=["Stress x time (per minute)"],
    )
    add_fig4_sheets(wb)
    add_figure5_sheets(wb)
    add_figure6_sheets(wb)
    add_figure7_sheets(wb)
    add_supplementary_sheets(wb)
    validate_report_workbook(wb)
    save_workbook(wb, output)
    print(f"Saved full and source-dataset statistical report: {output.relative_to(REPO)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_OUT,
        help="Output workbook path (default: report/statistical_report.xlsx).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else REPO / args.output
    build_statistical_report(output)


if __name__ == "__main__":
    main()
