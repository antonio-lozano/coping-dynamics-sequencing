"""Build the manuscript statistical report workbook in final panel order.

The workbook is generated from tracked repository outputs and follows the
panel-centered structure of the manuscript statistical report supplied for
submission.

Run: python scripts/build_statistical_report.py
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import statsmodels.formula.api as smf
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy import stats


warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "report" / "statistical_report.xlsx"
STATISTICS_DIR = REPO / "statistics"
RAW_DIR = REPO / "data" / "raw"
PROCESSED_DIR = REPO / "data" / "processed"

TEXT = "FF4D4D4D"
TITLE_FILL = PatternFill("solid", fgColor="FFEDEDED")
HEADER_FILL = PatternFill("solid", fgColor="FFF5F5F5")
SIG_FILL = PatternFill("solid", fgColor="FFF2CEEF")
BODY_FONT = Font(color=TEXT, size=10)
TITLE_FONT = Font(color=TEXT, bold=True, size=10)
HEADER_FONT = Font(color=TEXT, bold=True, size=10)
SIG_FONT = Font(color=TEXT, bold=True, size=10)
CENTER = Alignment(horizontal="center")

HEADERS = ["Parameter", "Coef.", "Std. Err.", "z", "P>|z|", "[0.025", "0.975]"]
PARAM_LABELS = {
    "Intercept": "Intercept",
    "Condition[T.ELS]": "Stress",
    "Experiment": "Experiment",
    "C(group, Treatment('Control'))[T.ELS]": "Stress",
    "C(group_ext, Treatment('ELS'))[T.Control]": "Control /ELS",
    "C(group_ext, Treatment('ELS'))[T.ELS resilient]": "ELS/Resilient",
    "C(group_ext, Treatment('ELS'))[T.Control]:time_bin_numeric": "Control x time",
    "C(group_ext, Treatment('ELS'))[T.ELS resilient]:time_bin_numeric": "ELS resilient x time",
    "time_bin_numeric": "Time",
}


def style_widths(ws: openpyxl.worksheet.worksheet.Worksheet) -> None:
    for col_idx in range(1, ws.max_column + 1):
        letter = get_column_letter(col_idx)
        max_len = 8
        for cell in ws[letter]:
            if cell.value is not None:
                max_len = max(max_len, min(len(str(cell.value)), 34))
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


def new_sheet(wb: openpyxl.Workbook, sheet_name: str) -> openpyxl.worksheet.worksheet.Worksheet:
    ws = wb.create_sheet(sheet_name[:31])
    ws.sheet_view.showGridLines = False
    return ws


def write_title(ws: openpyxl.worksheet.worksheet.Worksheet, row: int, col: int, title: str, width: int) -> None:
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

    p_cols = {idx for idx, column in enumerate(df.columns) if str(column).lower() in {"p>|z|", "p", "p_value", "p_bh_fdr"}}
    for row_idx, values in enumerate(df.itertuples(index=False), header_row + 1):
        for offset, value in enumerate(values):
            value = numeric(value)
            cell = ws.cell(row_idx, col + offset, format_p(value))
            cell.font = BODY_FONT
            if offset in p_cols and isinstance(value, (int, float)) and value < 0.05:
                cell.fill = SIG_FILL
                cell.font = SIG_FONT
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


def add_horizontal_blocks(wb: openpyxl.Workbook, sheet_name: str, blocks: list[tuple[str, pd.DataFrame]], *, gap: int = 3) -> None:
    ws = new_sheet(wb, sheet_name)
    col = 1
    for title, df in blocks:
        write_table(ws, 1, col, title, df)
        col += len(df.columns) + gap
    style_widths(ws)


def add_vertical_tables(wb: openpyxl.Workbook, sheet_name: str, tables: list[tuple[str, pd.DataFrame]]) -> None:
    ws = new_sheet(wb, sheet_name)
    row = 1
    for title, df in tables:
        row = write_table(ws, row, 1, title, df, header_gap=1)
    style_widths(ws)


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


def figure2_syllable_usage() -> pd.DataFrame:
    frames = pd.read_csv(RAW_DIR / "moseq_syllables_per_frame.csv.gz", usecols=["name", "syllable"])
    total_frames = len(frames)
    n_videos = frames["name"].nunique()
    counts = frames.groupby("syllable", as_index=False).size().rename(columns={"syllable": "Syllable", "size": "Total_frames_all_videos"})
    videos = frames.drop_duplicates(["name", "syllable"]).groupby("syllable", as_index=False).size()
    videos = videos.rename(columns={"syllable": "Syllable", "size": "Videos_with_syllable"})
    out = counts.merge(videos, on="Syllable", how="left")
    out["Percent_all_frames"] = out["Total_frames_all_videos"] / total_frames * 100
    out["Videos_with_syllable"] = out["Videos_with_syllable"].astype(str) + f"/{n_videos}"
    return out[["Syllable", "Total_frames_all_videos", "Percent_all_frames", "Videos_with_syllable"]].sort_values("Syllable")


def figure2_precision_recall_summary() -> pd.DataFrame:
    frames = pd.read_csv(RAW_DIR / "moseq_syllables_per_frame.csv.gz", usecols=["name", "frame_index", "syllable", "group"])
    frames["animal_id"] = frames["name"].str.extract(r"Animal[_ ](\d+_\d+)")[0].str.replace("_", ".", regex=False)
    freezing = pd.read_csv(RAW_DIR / "freezing_predictions_light.csv.gz", usecols=["animal_id", "frame", "freezing"])
    freezing["animal_id"] = freezing["animal_id"].map(lambda x: f"{float(x):.1f}")
    merged = frames.merge(
        freezing.rename(columns={"frame": "frame_index"}),
        on=["animal_id", "frame_index"],
        how="left",
    )
    merged["freezing"] = merged["freezing"].fillna(0).astype(int)
    by_animal = (
        merged.groupby(["animal_id", "group", "syllable"], as_index=False)
        .agg(syllable_frames=("syllable", "size"), overlap_frames=("freezing", "sum"))
    )
    freeze_total = merged.groupby("animal_id", as_index=False)["freezing"].sum().rename(columns={"freezing": "freezing_frames"})
    by_animal = by_animal.merge(freeze_total, on="animal_id", how="left")
    by_animal["precision_percent"] = by_animal["overlap_frames"] / by_animal["syllable_frames"] * 100
    by_animal["recall_percent"] = by_animal["overlap_frames"] / by_animal["freezing_frames"] * 100
    long = by_animal.melt(
        id_vars=["syllable", "group"],
        value_vars=["precision_percent", "recall_percent", "overlap_frames"],
        var_name="Metric",
        value_name="value",
    )
    out = (
        long.groupby(["syllable", "group", "Metric"], as_index=False)["value"]
        .agg(N="count", Mean="mean", SD="std")
        .rename(columns={"syllable": "Syllable", "group": "Group"})
    )
    out["SEM"] = out["SD"] / np.sqrt(out["N"])
    return out[["Syllable", "Group", "Metric", "N", "Mean", "SD", "SEM"]]


def figure2_freezing_syllables() -> pd.DataFrame:
    df = pd.read_csv(REPO / "figure_source_data" / "figure2.csv")
    return df[df["effect"].eq("overlap")].rename(
        columns={"metric": "Metric", "experiment": "Experiment", "beta": "Value", "se": "SE", "z": "z", "p_value": "p_value"}
    )[["Metric", "Experiment", "Value", "SE", "z", "p_value"]]


def add_figure2_sheets(wb: openpyxl.Workbook) -> None:
    add_vertical_tables(
        wb,
        "Fig.2A-C_Ground_truth ",
        [(experiment, rows[HEADERS]) for experiment, rows in figure2_ground_truth_stats().groupby("experiment", sort=False)],
    )
    add_vertical_tables(wb, "Fig.2D_Syllable_usage", [("Syllable usage tested across all 82 videos", figure2_syllable_usage())])
    add_vertical_tables(wb, "Fig.2E-F_Precision_recall", [("Precision/recall summary by syllable and group", figure2_precision_recall_summary())])
    add_vertical_tables(wb, "Fig.2H_Freezing_syllables", [("Freezing syllable overlap", figure2_freezing_syllables())])


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
    gap: int = 3,
) -> None:
    df = pd.read_csv(path)
    blocks = []
    for label, sub in df.groupby(group_col, sort=False):
        title = title_map.get(str(label), str(label)) if title_map else str(label)
        blocks.append((title, csv_model_rows(sub, parameter_col=parameter_col, coef_col=coef_col, se_col=se_col, p_col=p_col)))
    add_horizontal_blocks(wb, sheet_name, blocks, gap=gap)


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


def fig4_fit(df: pd.DataFrame, metric: str, combined: bool) -> pd.DataFrame:
    d = df.dropna(subset=[metric]).copy()
    d["Condition"] = pd.Categorical(d["group"], categories=["Control", "ELS"])
    formula = f"{metric} ~ Condition + Experiment" if combined else f"{metric} ~ Condition"
    try:
        model = smf.mixedlm(formula, d, groups=d["Animal"]).fit(reml=False)
        model_label = "MixedLM"
    except Exception:
        model = smf.ols(formula, d).fit()
        model_label = "OLS fallback"
    ci = model.conf_int()
    rows = []
    for param in model.params.index:
        if param == "Group Var":
            continue
        rows.append(
            {
                "Parameter": PARAM_LABELS.get(param, param),
                "Coef.": model.params[param],
                "Std. Err.": model.bse[param],
                "z": model.tvalues[param],
                "P>|z|": model.pvalues[param],
                "[0.025": ci.loc[param, 0],
                "0.975]": ci.loc[param, 1],
            }
        )
    rows.append(
        {
            "Parameter": "Model",
            "Coef.": model_label,
            "Std. Err.": "",
            "z": "",
            "P>|z|": "",
            "[0.025": "",
            "0.975]": "",
        }
    )
    return pd.DataFrame(rows, columns=HEADERS)


def add_fig4_sheet(wb: openpyxl.Workbook, sheet_name: str, blocks: list[tuple[str, pd.DataFrame, str]]) -> None:
    ws = new_sheet(wb, sheet_name)
    col = 1
    for title, source, metric in blocks:
        row = 1
        write_title(ws, row, col, title, len(HEADERS))
        row += 2
        for label, combined, data in [
            ("Combined datasets", True, source),
            ("Experiment 1", False, source[source["Experiment"] == 1]),
            ("Experiment 2", False, source[source["Experiment"] == 3]),
        ]:
            fitted = fig4_fit(data, metric, combined)
            row = write_table(ws, row, col, label, fitted, header_gap=1)
        col += len(HEADERS) + 3
    style_widths(ws)


def add_fig4_sheets(wb: openpyxl.Workbook) -> None:
    fig4 = load_fig4_module()
    _pred, _pred_seq, full_seq, meta = fig4.load_sequences()
    metrics, _usage = fig4.compute_frequency_metrics(full_seq, meta)
    bouts = fig4.bout_table(full_seq, meta)
    transitions = fig4.transition_metrics(full_seq, meta)
    mean_bouts = bouts.groupby(["Animal", "group", "Experiment", "cluster"], as_index=False)["bout_duration"].mean()

    add_fig4_sheet(
        wb,
        "Fig.4E-H_frequency_metrics",
        [
            ("Simpson Index", metrics, "simpson"),
            ("Shannon entropy", metrics, "shannon"),
            ("Evenness", metrics, "evenness"),
            ("Cumulative usage index", metrics, "cui"),
        ],
    )
    add_fig4_sheet(
        wb,
        "Fig.4J-Q_bout_duration",
        [
            (cluster, mean_bouts[mean_bouts["cluster"] == cluster].rename(columns={"bout_duration": cluster}), cluster)
            for cluster in ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]
        ],
    )
    add_fig4_sheet(
        wb,
        "Fig.4T-W_transition_metrics",
        [
            ("Lempel-Ziv complexity", transitions, "lz"),
            ("Recurrence rate", transitions, "recurrence"),
            ("Determinism", transitions, "determinism"),
            ("Markov entropy", transitions, "markov"),
        ],
    )


def fit_group_score(df: pd.DataFrame, value_col: str, group_col: str = "group") -> pd.DataFrame:
    formula = f"{value_col} ~ C({group_col}, Treatment('Control'))"
    model = smf.ols(formula, df).fit()
    ci = model.conf_int()
    rows = []
    for param in model.params.index:
        rows.append(
            {
                "Parameter": PARAM_LABELS.get(param, str(param).replace("C(group, Treatment('Control'))[T.ELS]", "Stress")),
                "Coef.": model.params[param],
                "Std. Err.": model.bse[param],
                "z": model.tvalues[param],
                "P>|z|": model.pvalues[param],
                "[0.025": ci.loc[param, 0],
                "0.975]": ci.loc[param, 1],
            }
        )
    return pd.DataFrame(rows, columns=HEADERS)


def add_figure5_sheets(wb: openpyxl.Workbook) -> None:
    scores = pd.read_csv(PROCESSED_DIR / "figure5_dynamics_scores.csv")
    add_vertical_tables(wb, "Fig.5B_Euclidean distance", [("Euclidean distance", fit_group_score(scores, "dynamics_score"))])
    add_grouped_csv_blocks(
        wb,
        "Fig.5C_Clusters_frequency",
        STATISTICS_DIR / "fig5c_frequency_contrasts.csv",
        "cluster",
        parameter_col="Parameter",
        coef_col="Coef.",
        se_col="Std. Err.",
        p_col="P>|z|",
    )
    add_grouped_csv_blocks(
        wb,
        "Fig.5D-J_Clusters_over_time",
        STATISTICS_DIR / "fig5_timecourse_mixedlm.csv",
        "cluster",
        parameter_col="parameter",
        coef_col="coef",
        se_col="se",
        p_col="p",
    )


def add_figure6_sheets(wb: openpyxl.Workbook) -> None:
    add_grouped_csv_blocks(
        wb,
        "Fig.6G-J_frequency_metrics",
        STATISTICS_DIR / "fig6_diversity_resilience_stats.csv",
        "metric",
        parameter_col="contrast",
        coef_col="beta",
        se_col="SE",
        p_col="p",
    )
    add_grouped_csv_blocks(
        wb,
        "Fig.6L-S_bout_duration",
        STATISTICS_DIR / "fig6_bout_resilience_stats.csv",
        "cluster",
        parameter_col="contrast",
        coef_col="beta",
        se_col="SE",
        p_col="p",
    )
    add_grouped_csv_blocks(
        wb,
        "Fig.6W-Z_transition_metrics",
        STATISTICS_DIR / "fig6_transition_resilience_stats.csv",
        "metric",
        parameter_col="contrast",
        coef_col="beta",
        se_col="SE",
        p_col="p",
    )


def add_supplementary_sheets(wb: openpyxl.Workbook) -> None:
    scores = pd.read_csv(PROCESSED_DIR / "supplementary_figure3_distance_scores.csv")
    tables = []
    for metric, sub in scores.groupby("metric", sort=False):
        tables.append((f"{metric}_score", fit_group_score(sub, "dynamics_score")))
    add_horizontal_blocks(wb, "Sup-Fig.3_distance_scores", tables)

    overlap = pd.concat(
        [
            pd.read_csv(PROCESSED_DIR / "figure5_resilience_threshold_audit.csv").assign(metric_source="Euclidean"),
            pd.read_csv(PROCESSED_DIR / "supplementary_figure3_threshold_audit.csv").assign(metric_source="Supplementary"),
        ],
        ignore_index=True,
    )
    add_vertical_tables(wb, "Resilience_overlap_methods", [("Resilience overlap methods", overlap)])


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    add_vertical_tables(wb, "Fig.1A_ SimBA_validation", [("SimBA validation correlation", simba_validation_summary())])
    add_figure2_sheets(wb)
    add_grouped_csv_blocks(
        wb,
        "Fig.3A_Clusters_frequency",
        STATISTICS_DIR / "stats_figure3A_GEE.csv",
        "cluster",
        parameter_col="parameter",
        coef_col="beta",
        se_col="se",
    )
    add_grouped_csv_blocks(
        wb,
        "Fig.3B-H_Clusters_over_time",
        STATISTICS_DIR / "stats_figure3_overtime.csv",
        "cluster",
        parameter_col="parameter",
        coef_col="coef",
        se_col="se",
        p_col="p",
    )
    add_fig4_sheets(wb)
    add_figure5_sheets(wb)
    add_figure6_sheets(wb)
    add_supplementary_sheets(wb)
    wb.save(OUT)
    print(f"Saved: {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
