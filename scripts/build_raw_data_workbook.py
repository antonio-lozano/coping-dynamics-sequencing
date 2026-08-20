"""Build the manuscript raw-data workbook in final panel order.

The workbook is generated from tracked repository inputs and uses the same
panel-centered structure as the manuscript raw-data workbook supplied for
submission.

Run: python scripts/build_raw_data_workbook.py
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from save_deterministic import save_workbook  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "report" / "raw_data.xlsx"

RAW_DIR = REPO / "data" / "raw"
PROCESSED_DIR = REPO / "data" / "processed"
STATS_DIR = REPO / "statistics"
FIGURE_SOURCE_DIR = REPO / "figure_source_data"
MANUSCRIPT_RAW_TABLE_DIR = RAW_DIR / "manuscript_tables" / "raw_data"

FPS = 25
BIN_SECONDS = 30
SESSION_SECONDS = 450
PROJECT_LABELS = {
    1: "Sanguino-Gomez and Krugers, 2024",
    3: "Sanguino-Gomez et al., 2024",
}

ELS_RESILIENT = {
    "2.4",
    "15.4",
    "26.4",
    "43.3",
    "43.5",
    "49.6",
    "123.3",
    "134.2",
    "148.4",
    "150.3",
    "150.5",
    "159.4",
}

BEHAVIOR_SYLLABLES = {
    "Freeze": [0, 28],
    "Sniff": [18, 20],
    "Groom": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climb": [111],
    "Jump": [23, 29, 30, 34],
}

BEHAVIOR_LABELS = {
    "Freezing": "Freeze",
    "Sniffing": "Sniff",
    "Grooming": "Groom",
    "Climbing": "Climb",
}

COLUMN_LABELS = {
    "animal_id": "Animal",
    "animal_index": "Animal",
    "animal": "Animal",
    "animal_label": "Animal",
    "group": "Group",
    "group_ext": "Group",
    "project": "Dataset",
    "Citation": "Dataset",
    "recording": "Recording",
    "manual_percent": "Manual Percent",
    "automatic_percent": "Automatic Percent",
    "cluster": "Behavior",
    "figure_panel_current_export": "Figure Panel",
    "simpson_index": "Simpson Index",
    "shannon_entropy_index": "Shannon Entropy Index",
    "evenness_index": "Evenness Index",
    "cumulative_usage_index": "Cumulative Usage Index",
    "lempel_ziv_complexity": "Lempel-Ziv Complexity",
    "recurrence_rate": "Recurrence Rate",
    "determinism": "Determinism",
    "markov_entropy": "Markov Entropy",
    "bout_duration_seconds": "Bout Duration (s)",
    "frequency_seconds": "Frequency (s)",
    "total_frequency_seconds": "Total Frequency (s)",
    "from_cluster": "From Behavior",
    "to_cluster": "To Behavior",
    "mean_transition_count_per_animal": "Mean Transitions per Animal",
    "time_min": "Time (min)",
    "percent": "Percent of Time",
    "mds1": "MDS Dimension 1",
    "mds2": "MDS Dimension 2",
    "dynamics_score": "Dynamics Score",
    "resilient_by_zero": "ELS Resilient (Score < 0)",
    "loocv_accuracy": "LOOCV Accuracy",
}

TEXT = "FF4D4D4D"
TITLE_FILL = PatternFill("solid", fgColor="FFEDEDED")
HEADER_FILL = PatternFill("solid", fgColor="FFF5F5F5")
BODY_FONT = Font(color=TEXT, size=10)
TITLE_FONT = Font(color=TEXT, bold=True, size=10)
HEADER_FONT = Font(color=TEXT, bold=True, size=10)
CENTER = Alignment(horizontal="center")


def format_animal(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return f"{number:.1f}"


def sort_key(value: object) -> tuple[int, float, str]:
    text = format_animal(value)
    try:
        return (0, float(text), text)
    except ValueError:
        return (1, 0.0, text)


GROUP_RANK = {"Control": 0, "ELS": 1, "ELS vulnerable": 1, "ELS resilient": 2}


def _animal_str(value: object) -> str:
    return format_animal(value)


def _animal_num(value: object) -> float:
    try:
        return float(format_animal(value))
    except ValueError:
        return float("inf")


def ordered(df: pd.DataFrame, spec: list[tuple[str, str]]) -> pd.DataFrame:
    """Reorder rows to match the manuscript workbook.

    Each sheet in the submitted workbook carries its own row order. ``spec`` is a
    list of (column, kind) pairs applied as a stable multi-key sort. ``kind`` is
    one of: ``astr`` (animal id as string, e.g. "11.4" before "2.4"), ``anum``
    (animal id numeric), ``group`` (Control before ELS), ``num`` (numeric),
    ``str`` (string).
    """
    work = df.copy()
    key_cols = []
    for i, (column, kind) in enumerate(spec):
        key = f"__sort_{i}"
        if kind == "astr":
            work[key] = work[column].map(_animal_str)
        elif kind == "anum":
            work[key] = work[column].map(_animal_num)
        elif kind == "group":
            work[key] = work[column].map(lambda value: GROUP_RANK.get(str(value), 9))
        elif kind == "num":
            work[key] = pd.to_numeric(work[column], errors="coerce")
        else:
            work[key] = work[column].astype(str)
        key_cols.append(key)
    work = work.sort_values(key_cols, kind="stable")
    return work.drop(columns=key_cols).reset_index(drop=True)


def numeric_animal(df: pd.DataFrame, column: str = "animal_id") -> pd.DataFrame:
    """Write the animal id as a number rather than text.

    Row order is still decided by ``ordered`` (some sheets sort the id as a
    string); this only changes the cell type, which the cluster sheets of the
    manuscript workbook store as numeric.
    """
    out = df.copy()
    out[column] = pd.to_numeric(out[column].map(format_animal), errors="coerce")
    return out


def manuscript_raw_table(sheet_name: str) -> pd.DataFrame:
    """Read a manuscript source table for sheets whose frame-level source is absent."""
    text_cols = {
        "animal_id",
        "Animal",
        "name",
        "animal",
        "group",
        "project",
        "selected_syllables",
        "tested_in_all_videos",
        "figure_panel_current_export",
        "from_cluster",
        "to_cluster",
    }
    path = MANUSCRIPT_RAW_TABLE_DIR / f"{sheet_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manuscript raw source table: {path.relative_to(REPO)}")
    header = pd.read_csv(path, nrows=0)
    dtype = {column: "string" for column in header.columns if column in text_cols}
    return pd.read_csv(path, dtype=dtype)


def animal_from_name(name: str) -> str:
    match = re.search(r"Animal[_ ](\d+)_(\d+)", str(name))
    if not match:
        return str(name)
    return f"{int(match.group(1))}.{int(match.group(2))}"


def experiment_map() -> pd.DataFrame:
    usage = pd.read_csv(RAW_DIR / "syllable_usage_per_timebin_30s.csv", usecols=["Animal", "Condition", "Experiment"])
    out = usage.drop_duplicates().rename(columns={"Animal": "animal_id", "Condition": "group"})
    out["animal_id"] = out["animal_id"].map(format_animal)
    out["project"] = out["Experiment"].map(PROJECT_LABELS)
    return out


def dataset_label(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        key = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    return PROJECT_LABELS.get(key, str(value))


def resilience_group(animal: object, group: object) -> str:
    group_text = str(group)
    if group_text == "Control":
        return "Control"
    if group_text in {"ELS", "ELS vulnerable", "ELS resilient"}:
        return "ELS resilient" if format_animal(animal) in ELS_RESILIENT else "ELS vulnerable"
    return group_text


def display_column_name(column: object) -> str:
    name = str(column)
    match = re.fullmatch(r"t_(\d{3})s", name)
    if match:
        minutes = int(match.group(1)) / 60.0
        return f"{minutes:g}_min"
    words = COLUMN_LABELS.get(name, name.replace("_", " ").strip().title())
    replacements = {
        "Mds": "MDS",
        "Id": "ID",
        "Sem": "SEM",
        "Shap": "SHAP",
        "Lz": "LZ",
        "Cui": "CUI",
    }
    for old, new in replacements.items():
        words = words.replace(old, new)
    words = re.sub(r"\bPercent\b", "Percentage", words, flags=re.IGNORECASE)
    return re.sub(r"\s+", "_", words.strip())


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply submission-facing metadata, headings, and behavior labels."""
    out = df.copy()
    path_pattern = re.compile(r"(?:[A-Za-z]:[\\/]|(?:^|\s)[\\/](?:Users|home|mnt|data)[\\/])", re.IGNORECASE)
    drop_columns = []
    for column in out.columns:
        if str(column).strip().lower() == "recording":
            drop_columns.append(column)
            continue
        if pd.api.types.is_object_dtype(out[column]) or pd.api.types.is_string_dtype(out[column]):
            values = out[column].dropna().astype(str)
            if values.map(lambda value: bool(path_pattern.search(value))).any():
                drop_columns.append(column)
    out = out.drop(columns=drop_columns)
    out = out.drop(
        columns=[column for column in ["figure_panel_current_export", "Figure Panel"] if column in out.columns]
    )
    experiment_col = next((col for col in ["Experiment", "experiment"] if col in out.columns), None)
    has_dataset = any(col in out.columns for col in ["Dataset", "Citation", "project"])
    if experiment_col is not None and not has_dataset:
        out["Dataset"] = out[experiment_col].map(dataset_label)

    out = out.drop(columns=[col for col in ["Experiment", "experiment"] if col in out.columns])

    # Every figure uses the short behavior names (Freeze, Sniff, Groom, Climb)
    # shown in the published panels; the legacy classifier's verb forms are
    # normalised to them on import.
    for column in ["cluster", "from_cluster", "to_cluster"]:
        if column in out.columns:
            out[column] = out[column].replace(BEHAVIOR_LABELS)

    out = out.rename(columns={column: display_column_name(column) for column in out.columns})

    metadata = [column for column in ["Animal", "Group", "Dataset"] if column in out.columns]
    remaining = [column for column in out.columns if column not in metadata]
    out = out[metadata + remaining]
    if "Animal" in out.columns:
        out["__animal_sort"] = pd.to_numeric(out["Animal"], errors="coerce")
        out = out.sort_values("__animal_sort", kind="stable", na_position="last").drop(columns="__animal_sort")
    return out.reset_index(drop=True)


def name_map() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "moseq_syllables_per_frame.csv.gz", usecols=["name"])
    names = pd.DataFrame({"animal": sorted(df["name"].unique())})
    names["animal_id"] = names["animal"].map(animal_from_name)
    return names


def style_widths(ws: openpyxl.worksheet.worksheet.Worksheet) -> None:
    for col_idx in range(1, ws.max_column + 1):
        letter = get_column_letter(col_idx)
        max_len = 8
        for cell in ws[letter]:
            if cell.value is not None:
                max_len = max(max_len, min(len(str(cell.value)), 42))
        ws.column_dimensions[letter].width = max_len + 2


def write_titled_dataframe(wb: openpyxl.Workbook, sheet_name: str, title: str, df: pd.DataFrame) -> None:
    df = standardize_dataframe(df)
    ws = wb.create_sheet(sheet_name[:31])
    ncols = max(len(df.columns), 1)
    ws.cell(1, 1, title)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    for col in range(1, ncols + 1):
        cell = ws.cell(1, col)
        cell.fill = TITLE_FILL
        cell.font = TITLE_FONT

    for col_idx, column in enumerate(df.columns, 1):
        cell = ws.cell(3, col_idx, column)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER

    for row_idx, row in enumerate(df.itertuples(index=False), 4):
        for col_idx, value in enumerate(row, 1):
            if pd.isna(value):
                value = None
            cell = ws.cell(row_idx, col_idx, value)
            cell.font = BODY_FONT

    ws.sheet_view.showGridLines = False
    style_widths(ws)


def pivot_time_table(df: pd.DataFrame, index_cols: list[str], time_col: str, value_col: str) -> pd.DataFrame:
    wide = df.pivot_table(index=index_cols, columns=time_col, values=value_col, aggfunc="first").reset_index()
    time_cols = sorted([col for col in wide.columns if isinstance(col, (int, float, np.integer, np.floating))])
    rename = {col: f"t_{int(col):03d}s" for col in time_cols}
    wide = wide.rename(columns=rename)
    return wide[index_cols + [rename[col] for col in time_cols]]


def freezing_ground_truth() -> pd.DataFrame:
    freezing = pd.read_csv(RAW_DIR / "freezing_predictions_light.csv.gz")
    freezing["animal_id"] = freezing["animal_id"].map(format_animal)
    freezing["time_s"] = ((freezing["frame"] // (FPS * BIN_SECONDS)) + 1) * BIN_SECONDS
    grouped = (
        freezing.groupby(["animal_id", "group", "time_s"], as_index=False)["freezing"]
        .mean()
        .rename(columns={"freezing": "freezing_percent"})
    )
    grouped["freezing_percent"] *= 100
    grouped = grouped.merge(experiment_map()[["animal_id", "Experiment", "project"]], on="animal_id", how="inner")
    wide = pivot_time_table(grouped, ["animal_id", "group", "project", "Experiment"], "time_s", "freezing_percent")
    wide = wide.rename(columns={"animal_id": "Animal", "group": "Group", "project": "Citation"})
    return ordered(wide, [("Animal", "astr")])


CLUSTER_ORDER = ["Climb", "Freeze", "Groom", "Jump", "Locomotion", "Sniff", "Turn"]


def cluster_frequency() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "cluster_frequency_per_animal.csv")
    df["animal_id"] = df["animal_id"].map(format_animal)
    # The manuscript workbook keeps a complete animal x cluster grid, including
    # clusters an animal never performed (frequency 0). The derived table drops
    # those zero rows, so reindex to the full grid before writing.
    meta = df[["animal_id", "group", "experiment"]].drop_duplicates()
    grid = meta.merge(pd.DataFrame({"cluster": CLUSTER_ORDER}), how="cross")
    df = grid.merge(df[["animal_id", "cluster", "frequency_seconds"]], on=["animal_id", "cluster"], how="left")
    df["frequency_seconds"] = df["frequency_seconds"].fillna(0.0)
    df["project"] = df["experiment"].map(PROJECT_LABELS)
    df = df[["animal_id", "group", "project", "experiment", "cluster", "frequency_seconds"]]
    return ordered(df, [("experiment", "num"), ("group", "group"), ("animal_id", "anum"), ("cluster", "str")])


def cluster_timecourse() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "cluster_timecourse_per_animal.csv")
    df["animal_id"] = df["animal_id"].map(format_animal)
    df["project"] = df["experiment"].map(PROJECT_LABELS)
    wide = pivot_time_table(df, ["animal_id", "group", "project", "experiment", "cluster"], "time_s", "pct")
    return wide.sort_values(["experiment", "animal_id", "cluster"], key=lambda s: s.map(sort_key) if s.name == "animal_id" else s)


def fig4_frequency_metrics() -> pd.DataFrame:
    df = pd.read_csv(STATS_DIR / "fig4_diversity_per_animal.csv")
    df["Animal"] = df["Animal"].map(format_animal)
    df = df.rename(
        columns={
            "Animal": "animal_id",
            "Experiment": "experiment",
            "simpson": "simpson_index",
            "shannon": "shannon_entropy_index",
            "evenness": "evenness_index",
            "cui": "cumulative_usage_index",
        }
    )
    df = df[["animal_id", "group", "experiment", "simpson_index", "shannon_entropy_index", "evenness_index", "cumulative_usage_index"]]
    return ordered(df, [("animal_id", "astr")])


def fig4_usage_profile() -> pd.DataFrame:
    if (MANUSCRIPT_RAW_TABLE_DIR / "Fig4_usage_profile.csv").exists():
        return manuscript_raw_table("Fig4_usage_profile")

    df = pd.read_csv(PROCESSED_DIR / "cluster_frequency_per_animal.csv")
    df["animal_id"] = df["animal_id"].map(format_animal)
    label_map = {"Climb": "Climbing", "Freeze": "Freezing", "Groom": "Grooming", "Sniff": "Sniffing"}
    df["cluster"] = df["cluster"].replace(label_map)
    wide = df.pivot_table(
        index=["animal_id", "group", "experiment"],
        columns="cluster",
        values="frequency_seconds",
        aggfunc="sum",
        fill_value=0,
    )
    for cluster in ["Climbing", "Freezing", "Grooming", "Jump", "Locomotion", "Sniffing", "Turn"]:
        if cluster not in wide.columns:
            wide[cluster] = 0.0
    wide = wide / SESSION_SECONDS
    wide["Unmapped_or_excluded"] = (1.0 - wide.sum(axis=1)).clip(lower=0)
    wide = wide.reset_index()
    cols = [
        "animal_id",
        "group",
        "experiment",
        "Unmapped_or_excluded",
        "Climbing",
        "Freezing",
        "Grooming",
        "Jump",
        "Locomotion",
        "Sniffing",
        "Turn",
    ]
    return ordered(wide[cols], [("animal_id", "astr")])


def fig4_bout_duration() -> pd.DataFrame:
    overall = pd.read_csv(STATS_DIR / "fig4_bout_overall_per_animal.csv").rename(
        columns={"Animal": "animal_id", "Experiment": "experiment", "bout_mean": "bout_duration_seconds"}
    )
    overall["figure_panel_current_export"] = "J"
    overall["cluster"] = "Overall"
    clusters = pd.read_csv(STATS_DIR / "fig4_bout_cluster_per_animal.csv").rename(
        columns={"Animal": "animal_id", "Experiment": "experiment", "bout_duration": "bout_duration_seconds"}
    )
    panel_map = {
        "Freezing": "K",
        "Sniffing": "L",
        "Grooming": "M",
        "Turn": "N",
        "Locomotion": "O",
        "Climbing": "P",
        "Jump": "Q",
    }
    clusters["figure_panel_current_export"] = clusters["cluster"].map(panel_map)
    out = pd.concat([overall, clusters], ignore_index=True)
    out["animal_id"] = out["animal_id"].map(format_animal)
    out = out[["figure_panel_current_export", "animal_id", "group", "experiment", "cluster", "bout_duration_seconds"]]
    return ordered(out, [("figure_panel_current_export", "str"), ("experiment", "num"), ("group", "group"), ("animal_id", "astr")])


def fig4_transition_metrics() -> pd.DataFrame:
    df = pd.read_csv(STATS_DIR / "fig4_transition_per_animal.csv")
    df["Animal"] = df["Animal"].map(format_animal)
    df = df.rename(
        columns={
            "Animal": "animal_id",
            "Experiment": "experiment",
            "lz": "lempel_ziv_complexity",
            "recurrence": "recurrence_rate",
            "determinism": "determinism",
            "markov": "markov_entropy",
        }
    )
    df = df[["animal_id", "group", "experiment", "lempel_ziv_complexity", "recurrence_rate", "determinism", "markov_entropy"]]
    return ordered(df, [("animal_id", "astr")])


def load_fig4_module():
    spec = importlib.util.spec_from_file_location("fig4", REPO / "scripts" / "generate_figures" / "figure_4_diversity_dynamics.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Figure 4 script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fig4_transition_chords() -> pd.DataFrame:
    if (MANUSCRIPT_RAW_TABLE_DIR / "Fig4_transition_chords.csv").exists():
        return manuscript_raw_table("Fig4_transition_chords")

    fig4 = load_fig4_module()
    _pred, _pred_sequences, full_sequences, meta = fig4.load_sequences()
    rows = []
    for panel, group in [("R", "Control"), ("S", "ELS")]:
        mats = []
        for animal in meta.loc[meta["group"] == group, "Animal"].astype(str):
            seq = full_sequences.get(animal)
            if seq:
                mats.append(fig4.transition_matrix_flow(seq))
        mat = np.mean(mats, axis=0) if mats else np.zeros((len(fig4.DISPLAY_ORDER), len(fig4.DISPLAY_ORDER)))
        for i, source in enumerate(fig4.DISPLAY_ORDER):
            for j, target in enumerate(fig4.DISPLAY_ORDER):
                rows.append(
                    {
                        "figure_panel_current_export": panel,
                        "group": group,
                        "from_cluster": source,
                        "to_cluster": target,
                        "mean_transition_count_per_animal": mat[i, j],
                    }
                )
    return pd.DataFrame(rows)


def load_fig5_module():
    spec = importlib.util.spec_from_file_location(
        "fig5", REPO / "scripts" / "generate_figures" / "figure_5_resilience_dynamics.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Figure 5 script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def figure5_dynamics() -> pd.DataFrame:
    scores = pd.read_csv(PROCESSED_DIR / "figure5_dynamics_scores.csv")
    scores["animal"] = scores["animal"].map(format_animal)
    fig5 = load_fig5_module()
    scores["loocv_accuracy"] = fig5.loocv_logistic(
        scores[["mds1", "mds2"]].to_numpy(), scores["group"].to_numpy()
    )
    scores = scores.merge(
        experiment_map()[["animal_id", "Experiment"]],
        left_on="animal",
        right_on="animal_id",
        how="left",
    )
    scores["group"] = scores.apply(lambda row: resilience_group(row["animal"], row["group"]), axis=1)
    scores = scores[
        [
            "animal",
            "group",
            "Experiment",
            "mds1",
            "mds2",
            "dynamics_score",
            "resilient_by_zero",
            "loocv_accuracy",
        ]
    ]
    return ordered(scores, [("Experiment", "num"), ("group", "group"), ("animal", "anum")])


def figure5_cluster_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW_DIR / "syllable_usage_per_timebin_30s.csv")
    raw = raw.rename(columns={"Condition": "group", "Time Bin": "time_bin"})
    raw["Animal"] = raw["Animal"].map(format_animal)
    base_cols = ["Animal", "group", "Experiment", "time_bin"]
    base = raw[base_cols].drop_duplicates()
    frames = []
    for behavior, syllables in BEHAVIOR_SYLLABLES.items():
        values = (
            raw[raw["Syllable"].isin(syllables)]
            .groupby(base_cols, as_index=False)["Percentage"]
            .sum()
        )
        values = base.merge(values, on=base_cols, how="left")
        values["Percentage"] = values["Percentage"].fillna(0.0)
        values["cluster"] = behavior
        frames.append(values)

    timecourse_long = pd.concat(frames, ignore_index=True)
    timecourse_long["group"] = timecourse_long.apply(
        lambda row: resilience_group(row["Animal"], row["group"]), axis=1
    )
    timecourse_long["time_seconds"] = pd.to_numeric(timecourse_long["time_bin"]) + BIN_SECONDS
    timecourse_long = timecourse_long.rename(columns={"Animal": "animal_id", "Percentage": "percent"})[
        ["animal_id", "group", "Experiment", "cluster", "time_seconds", "percent"]
    ]

    frequency = timecourse_long.copy()
    frequency["frequency_seconds"] = frequency["percent"] / 100.0 * BIN_SECONDS
    frequency = (
        frequency.groupby(["animal_id", "group", "Experiment", "cluster"], as_index=False)["frequency_seconds"]
        .sum()
    )
    frequency = ordered(
        frequency,
        [("Experiment", "num"), ("group", "group"), ("animal_id", "anum"), ("cluster", "str")],
    )

    timecourse = pivot_time_table(
        timecourse_long,
        ["animal_id", "group", "Experiment", "cluster"],
        "time_seconds",
        "percent",
    )
    timecourse = ordered(
        timecourse,
        [("Experiment", "num"), ("group", "group"), ("animal_id", "anum"), ("cluster", "str")],
    )
    return frequency, timecourse


def load_fig6_module():
    spec = importlib.util.spec_from_file_location(
        "fig6", REPO / "scripts" / "generate_figures" / "figure_6_resilience_diversity.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Figure 6 script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def figure6_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fig6 = load_fig6_module()
    _pred, _pred_sequences, full_sequences, meta = fig6.load_sequences()
    metrics, _usage = fig6.compute_frequency_metrics(full_sequences, meta)
    bouts = fig6.bout_table(full_sequences, meta)
    transitions = fig6.transition_metrics(full_sequences, meta)

    def relabel_groups(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if "group_ext" in out.columns:
            out["group_ext"] = out["group_ext"].replace({"ELS": "ELS vulnerable"})
        return out

    metrics = relabel_groups(metrics)
    bouts = relabel_groups(bouts)
    transitions = relabel_groups(transitions)

    metric_labels = metrics.rename(
        columns={
            "Animal": "animal_id",
            "group_ext": "group",
            "simpson": "simpson_index",
            "shannon": "shannon_entropy_index",
            "evenness": "evenness_index",
            "cui": "cumulative_usage_index",
        }
    )
    frequency_metrics = metric_labels[
        [
            "animal_id",
            "group",
            "Experiment",
            "simpson_index",
            "shannon_entropy_index",
            "evenness_index",
            "cumulative_usage_index",
        ]
    ]

    overall = (
        bouts.groupby(["Animal", "group_ext", "Experiment"], as_index=False)["bout_duration"]
        .mean()
        .assign(cluster="Overall", figure_panel_current_export="L")
    )
    cluster_means = (
        bouts[bouts["cluster"].isin(fig6.DISPLAY_ORDER)]
        .groupby(["Animal", "group_ext", "Experiment", "cluster"], as_index=False)["bout_duration"]
        .mean()
    )
    panel_map = {
        "Freezing": "M",
        "Sniffing": "N",
        "Grooming": "O",
        "Turn": "P",
        "Locomotion": "Q",
        "Climbing": "R",
        "Jump": "S",
    }
    cluster_means["figure_panel_current_export"] = cluster_means["cluster"].map(panel_map)
    bout_points = pd.concat([overall, cluster_means], ignore_index=True).rename(
        columns={
            "Animal": "animal_id",
            "group_ext": "group",
            "bout_duration": "bout_duration_seconds",
        }
    )
    bout_points = bout_points[
        ["figure_panel_current_export", "animal_id", "group", "Experiment", "cluster", "bout_duration_seconds"]
    ]

    chord_rows = []
    group_map = meta.set_index("Animal")["group_ext"].to_dict()
    for panel, source_group, output_group in [
        ("T", "Control", "Control"),
        ("U", "ELS", "ELS vulnerable"),
        ("V", "ELS resilient", "ELS resilient"),
    ]:
        matrices = [
            fig6.transition_matrix_flow(sequence)
            for animal, sequence in full_sequences.items()
            if group_map.get(str(animal)) == source_group
        ]
        matrix = np.mean(matrices, axis=0) if matrices else np.zeros((len(fig6.DISPLAY_ORDER), len(fig6.DISPLAY_ORDER)))
        for i, source in enumerate(fig6.DISPLAY_ORDER):
            for j, target in enumerate(fig6.DISPLAY_ORDER):
                chord_rows.append(
                    {
                        "figure_panel_current_export": panel,
                        "group": output_group,
                        "from_cluster": source,
                        "to_cluster": target,
                        "mean_transition_count_per_animal": matrix[i, j],
                    }
                )
    chords = pd.DataFrame(chord_rows)

    transition_points = transitions.rename(
        columns={
            "Animal": "animal_id",
            "group_ext": "group",
            "lz": "lempel_ziv_complexity",
            "recurrence": "recurrence_rate",
            "markov": "markov_entropy",
        }
    )

    return frequency_metrics, bout_points, chords, transition_points


def supplementary_tracking_time() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "supplementary_figure1_tracking_clusters.csv")
    wide = pivot_time_table(df, ["animal_id", "group", "project", "experiment", "cluster"], "seconds", "percentage")
    frequency = (
        df.assign(total_frequency_seconds=df["percentage"] / 100.0 * BIN_SECONDS)
        .groupby(["animal_id", "group", "project", "experiment", "cluster"], as_index=False)["total_frequency_seconds"]
        .sum()
    )
    wide = wide.merge(frequency, on=["animal_id", "group", "project", "experiment", "cluster"], how="left")
    wide["animal_id"] = wide["animal_id"].map(format_animal)
    return wide.sort_values(["experiment", "animal_id", "cluster"], key=lambda s: s.map(sort_key) if s.name == "animal_id" else s)


def supplementary_tracking_frequency() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "supplementary_figure1_tracking_clusters.csv")
    # `seconds` is the time-bin timestamp (30, 60, 90 ...), not a per-bin
    # duration. Total time in a cluster is the sum of each bin's occupancy
    # (percentage of the 30 s bin), matching the manuscript workbook.
    df["cluster_seconds"] = df["percentage"] / 100.0 * BIN_SECONDS
    out = (
        df.groupby(["animal_id", "group", "project", "experiment", "cluster"], as_index=False)["cluster_seconds"]
        .sum()
        .rename(columns={"cluster_seconds": "frequency_seconds"})
    )
    out["animal_id"] = out["animal_id"].map(format_animal)
    return ordered(out, [("experiment", "num"), ("group", "group"), ("animal_id", "anum"), ("cluster", "str")])


def supplementary_figure3_scores() -> pd.DataFrame:
    scores = pd.read_csv(PROCESSED_DIR / "supplementary_figure3_distance_scores.csv")
    scores["animal"] = scores["animal"].map(format_animal)
    scores = scores.merge(
        experiment_map()[["animal_id", "Experiment"]],
        left_on="animal",
        right_on="animal_id",
        how="left",
    )
    scores = scores[
        [
            "animal",
            "group",
            "Experiment",
            "metric",
            "mds1",
            "mds2",
            "dynamics_score",
            "resilient_by_zero",
            "loocv_accuracy",
        ]
    ]
    return ordered(scores, [("metric", "str"), ("Experiment", "num"), ("group", "group"), ("animal", "anum")])


def syllable_timebin_30s() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "syllable_usage_per_timebin_30s.csv")
    df = df.rename(columns={"Condition": "group", "Time Bin": "time_bin_seconds"})[
        ["Animal", "group", "Experiment", "time_bin_seconds", "Syllable", "Percentage"]
    ]
    return ordered(df, [("Animal", "astr"), ("time_bin_seconds", "num"), ("Syllable", "num")])


def behavioral_flexibility_scores() -> pd.DataFrame:
    return pd.read_excel(RAW_DIR / "behavioral_flexibility_scores.xlsx")


def syllable_frames() -> pd.DataFrame:
    if (MANUSCRIPT_RAW_TABLE_DIR / "Syllable_frames.csv").exists():
        return manuscript_raw_table("Syllable_frames")

    frames = pd.read_csv(RAW_DIR / "moseq_syllables_per_frame.csv.gz", usecols=["name", "syllable", "group"])
    frames["animal_id"] = frames["name"].map(animal_from_name)
    frames = frames.merge(experiment_map()[["animal_id", "Experiment", "project"]], on="animal_id", how="left")
    counts = (
        frames.groupby(["animal_id", "name", "group", "project", "Experiment", "syllable"], as_index=False)
        .size()
        .rename(columns={"size": "frames_in_video", "Experiment": "experiment"})
    )
    totals = frames.groupby(["animal_id", "name"], as_index=False).size().rename(columns={"size": "total_video_frames"})
    all_syll = frames.groupby("syllable", as_index=False).size().rename(columns={"size": "total_frames_all_videos"})
    counts = counts.merge(totals, on=["animal_id", "name"], how="left").merge(all_syll, on="syllable", how="left")
    counts["percent_video_frames"] = counts["frames_in_video"] / counts["total_video_frames"] * 100
    counts["percent_all_frames"] = counts["total_frames_all_videos"] / frames.shape[0] * 100
    counts["tested_in_all_videos"] = "Yes, all 82 manuscript-cohort videos"
    cols = [
        "animal_id",
        "name",
        "group",
        "project",
        "experiment",
        "syllable",
        "frames_in_video",
        "total_video_frames",
        "total_frames_all_videos",
        "percent_video_frames",
        "percent_all_frames",
        "tested_in_all_videos",
    ]
    return counts[cols].sort_values(["syllable", "experiment", "animal_id"], key=lambda s: s.map(sort_key) if s.name == "animal_id" else s)


def precision_recall_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    precision_path = MANUSCRIPT_RAW_TABLE_DIR / "Precision_recall.csv"
    overlap_path = MANUSCRIPT_RAW_TABLE_DIR / "Overlap_0_28.csv"
    if precision_path.exists() and overlap_path.exists():
        return manuscript_raw_table("Precision_recall"), manuscript_raw_table("Overlap_0_28")

    frames = pd.read_csv(RAW_DIR / "moseq_syllables_per_frame.csv.gz", usecols=["name", "frame_index", "syllable", "group"])
    frames["animal_id"] = frames["name"].map(animal_from_name)
    frames = frames.merge(experiment_map()[["animal_id", "Experiment", "project"]], on="animal_id", how="left")
    freezing = pd.read_csv(RAW_DIR / "freezing_predictions_light.csv.gz", usecols=["animal_id", "frame", "freezing"])
    freezing["animal_id"] = freezing["animal_id"].map(format_animal)
    merged = frames.merge(
        freezing.rename(columns={"frame": "frame_index"}),
        on=["animal_id", "frame_index"],
        how="left",
    )
    merged["freezing"] = merged["freezing"].fillna(0).astype(int)

    syllable_counts = (
        merged.groupby(["animal_id", "name", "group", "project", "Experiment", "syllable"], as_index=False)
        .agg(syllable_frames=("syllable", "size"), overlap_frames=("freezing", "sum"))
        .rename(columns={"name": "animal", "Experiment": "experiment"})
    )
    freezing_counts = merged.groupby("animal_id", as_index=False)["freezing"].sum().rename(columns={"freezing": "freezing_frames"})
    syllable_counts = syllable_counts.merge(freezing_counts, on="animal_id", how="left")
    syllable_counts["precision_percent"] = np.where(
        syllable_counts["syllable_frames"] > 0,
        syllable_counts["overlap_frames"] / syllable_counts["syllable_frames"] * 100,
        np.nan,
    )
    syllable_counts["recall_percent"] = np.where(
        syllable_counts["freezing_frames"] > 0,
        syllable_counts["overlap_frames"] / syllable_counts["freezing_frames"] * 100,
        np.nan,
    )
    precision = syllable_counts[
        [
            "animal_id",
            "animal",
            "group",
            "project",
            "experiment",
            "syllable",
            "syllable_frames",
            "freezing_frames",
            "overlap_frames",
            "precision_percent",
            "recall_percent",
        ]
    ]
    precision = ordered(precision, [("experiment", "num"), ("group", "group"), ("animal_id", "astr"), ("syllable", "num")])

    selected = merged[merged["syllable"].isin([0, 28])].copy()
    overlap = (
        selected.groupby(["animal_id", "name", "group", "project", "Experiment"], as_index=False)
        .agg(syllable_0_28_frames=("syllable", "size"), overlap_frames=("freezing", "sum"))
        .rename(columns={"name": "animal", "Experiment": "experiment"})
    )
    overlap = overlap.merge(freezing_counts, on="animal_id", how="left")
    overlap["selected_syllables"] = "0 + 28"
    overlap["percent_freezing_covered_by_syllable_0_28"] = overlap["overlap_frames"] / overlap["freezing_frames"] * 100
    overlap["precision_syllable_0_28_vs_freezing"] = overlap["overlap_frames"] / overlap["syllable_0_28_frames"] * 100
    overlap = overlap[
        [
            "animal_id",
            "animal",
            "group",
            "project",
            "experiment",
            "selected_syllables",
            "freezing_frames",
            "syllable_0_28_frames",
            "overlap_frames",
            "percent_freezing_covered_by_syllable_0_28",
            "precision_syllable_0_28_vs_freezing",
        ]
    ]
    overlap = ordered(overlap, [("experiment", "num"), ("group", "group"), ("animal_id", "astr")])
    return precision, overlap


def syllable_0_28_timecourse() -> pd.DataFrame:
    combined_path = MANUSCRIPT_RAW_TABLE_DIR / "Syll_0_28_combined.csv"
    if combined_path.exists():
        return manuscript_raw_table("Syll_0_28_combined")

    tc = pd.read_csv(PROCESSED_DIR / "cluster_timecourse_per_animal.csv")
    tc = tc[tc["cluster"] == "Freeze"].copy()
    tc["animal_id"] = tc["animal_id"].map(format_animal)
    tc = tc.merge(name_map(), on="animal_id", how="left")
    tc["project"] = tc["experiment"].map(PROJECT_LABELS)
    wide = pivot_time_table(tc, ["animal_id", "animal", "group", "project", "experiment"], "time_s", "pct")
    return ordered(wide, [("animal_id", "astr")])


def figure2_syllable_usage_raw() -> pd.DataFrame:
    """Per-animal syllable-label usage underlying Figure 2D."""
    frames = syllable_frames().drop(columns=["name", "animal"], errors="ignore")
    columns = [
        "animal_id",
        "group",
        "project",
        "experiment",
        "syllable",
        "frames_in_video",
        "total_video_frames",
        "total_frames_all_videos",
        "percent_video_frames",
        "percent_all_frames",
    ]
    return ordered(frames[columns], [("experiment", "num"), ("animal_id", "anum"), ("syllable", "num")])


def figure2_precision_recall_raw() -> pd.DataFrame:
    """Per-animal/per-syllable precision and recall underlying Figure 2E-F."""
    precision, _ = precision_recall_tables()
    precision = precision.drop(columns=["animal", "name"], errors="ignore")
    columns = [
        "animal_id",
        "group",
        "project",
        "experiment",
        "syllable",
        "precision_percent",
        "recall_percent",
        "overlap_frames",
        "syllable_frames",
        "freezing_frames",
    ]
    return ordered(precision[columns], [("experiment", "num"), ("animal_id", "anum"), ("syllable", "num")])


def figure2_freezing_overlap_raw() -> pd.DataFrame:
    """Animal-level 0+28+40 overlap values plotted in Figure 2G."""
    overlap = pd.read_csv(RAW_DIR / "freezing_overlap_by_group.csv")
    overlap["animal_id"] = overlap["recording"].map(animal_from_name)
    overlap = overlap.merge(
        experiment_map()[["animal_id", "Experiment", "project"]],
        on="animal_id",
        how="left",
        validate="one_to_one",
    )
    overlap["selected_syllables"] = "0 + 28 + 40"
    overlap = overlap.rename(columns={"Experiment": "experiment", "overlap_pct": "freezing_overlap_percent"})
    columns = [
        "animal_id",
        "group",
        "project",
        "experiment",
        "selected_syllables",
        "freezing_overlap_percent",
    ]
    return ordered(overlap[columns], [("experiment", "num"), ("animal_id", "anum")])


def figure2_freezing_syllables_raw() -> pd.DataFrame:
    """Per-animal 0+28 syllable time course underlying Figure 2H."""
    timecourse = syllable_0_28_timecourse().drop(columns=["animal", "name"], errors="ignore")
    return ordered(timecourse, [("experiment", "num"), ("animal_id", "anum")])


def figure7_source(name: str) -> pd.DataFrame:
    """Load a tracked Figure 7/Supplementary Figure 4 source table."""
    path = FIGURE_SOURCE_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(REPO)}. Run scripts/run_all_figures.py first."
        )
    return pd.read_csv(path)


def figure7_classifier_shap_raw() -> pd.DataFrame:
    """All 20 features plotted in Figure 7B, across all eight classes."""
    data = figure7_source("figure7_classifier_global_shap.csv")
    behavior_columns = [
        "Freeze",
        "Sniff",
        "Groom",
        "Turn",
        "Locomotion",
        "Climb",
        "Jump",
        "Unassigned",
    ]
    missing = [column for column in ["feature", *behavior_columns] if column not in data.columns]
    if missing or len(data) != 20:
        raise AssertionError(
            f"Figure 7B SHAP source must contain 20 plotted features and eight classes; "
            f"missing={missing}, rows={len(data)}"
        )
    return data[["feature", *behavior_columns]]


def supplementary_figure4_shap_raw() -> pd.DataFrame:
    """All 719 legacy parameters per class; the published panels plot the top 10."""
    data = figure7_source("supplementary_figure4_shap_summary.csv")
    required = [
        "behavior",
        "rank",
        "parameter",
        "parameter_label",
        "mean_absolute_shap",
        "mean_shap",
        "sd_shap",
        "sem_shap",
        "minimum_shap",
        "maximum_shap",
        "n_frames",
        "plotted_top_10",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing or len(data) != 719 * 8:
        raise AssertionError(
            "Supplementary Figure 4 SHAP source must contain all 719 legacy parameters "
            f"for eight classes; missing={missing}, rows={len(data)}"
        )
    return data[required]


def figure7_prediction_auc_raw() -> pd.DataFrame:
    """Plotted Figure 7D-E AUC values without report-only diagnostics."""
    data = figure7_source("figure7_full_session_auc.csv").copy()
    dataset_names = {
        "Exp1": PROJECT_LABELS[1],
        "Exp3": PROJECT_LABELS[3],
    }
    data["validation_method"] = data["panel"].map(
        {
            "B held-out cohort": "Held-out cohort",
            "C full-session LOOCV": "Full-session leave-one-out cross-validation",
        }
    )
    data["train_dataset"] = data["train_experiment"].replace(dataset_names)
    data["test_dataset"] = data["test_experiment"].replace(dataset_names)
    full_session = data["validation_method"].eq("Full-session leave-one-out cross-validation")
    data.loc[full_session, ["train_dataset", "test_dataset"]] = "Both datasets"
    return data[
        [
            "validation_method",
            "feature_set",
            "train_dataset",
            "test_dataset",
            "n_features",
            "roc_auc",
            "ci_low",
            "ci_high",
        ]
    ]


def figure7_shap_per_animal_raw() -> pd.DataFrame:
    """Animal-level Figure 7F contributions without the helper label field."""
    data = figure7_source("figure7_shapley_per_animal.csv").copy()
    data = data.rename(columns={"animal_label": "animal", "profile": "group"})
    data["group"] = data["group"].replace(
        {"resilient": "ELS resilient", "vulnerable": "ELS vulnerable"}
    )
    return data


def figure7_prediction_onset_raw() -> pd.DataFrame:
    """Values drawn in Figure 7I; permutation statistics stay in the report."""
    data = figure7_source("figure7_prediction_onset.csv")
    return data[["feature_set", "horizon_min", "n_features", "roc_auc"]]


def figure7_predictor_catalog_raw() -> pd.DataFrame:
    """Figure 7G-H values in long form with explicit dataset directions."""
    data = figure7_source("figure7_predictor_catalog.csv")
    identity = data[["predictor", "kind", "n_features"]]
    dataset_1 = PROJECT_LABELS[1]
    dataset_3 = PROJECT_LABELS[3]
    tables = []
    for method, train, test, auc, low, high in [
        (
            "Held-out cohort",
            dataset_1,
            dataset_3,
            "auc_kru_to_gom",
            "ci_kru_lo",
            "ci_kru_hi",
        ),
        (
            "Held-out cohort",
            dataset_3,
            dataset_1,
            "auc_gom_to_kru",
            "ci_gom_lo",
            "ci_gom_hi",
        ),
        (
            "Repeated stratified cross-validation",
            "Both datasets",
            "Both datasets",
            "cv_auc",
            "cv_ci_low",
            "cv_ci_high",
        ),
    ]:
        table = identity.copy()
        table["validation_method"] = method
        table["train_dataset"] = train
        table["test_dataset"] = test
        table["roc_auc"] = data[auc]
        table["ci_low"] = data[low]
        table["ci_high"] = data[high]
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def figure7_family_timecourse_raw() -> pd.DataFrame:
    """Figure 7J values in long form with explicit dataset directions."""
    data = figure7_source("figure7_predictor_timecourse.csv")
    identity = data[["horizon_min", "predictor", "n_features"]]
    dataset_1 = PROJECT_LABELS[1]
    dataset_3 = PROJECT_LABELS[3]
    tables = []
    for method, train, test, auc, low, high in [
        (
            "Repeated stratified cross-validation",
            "Both datasets",
            "Both datasets",
            "cv_auc",
            "cv_lo",
            "cv_hi",
        ),
        (
            "Held-out cohort",
            dataset_1,
            dataset_3,
            "auc_kru_to_gom",
            None,
            None,
        ),
        (
            "Held-out cohort",
            dataset_3,
            dataset_1,
            "auc_gom_to_kru",
            None,
            None,
        ),
    ]:
        table = identity.copy()
        table["validation_method"] = method
        table["train_dataset"] = train
        table["test_dataset"] = test
        table["roc_auc"] = data[auc]
        table["ci_low"] = data[low] if low else float("nan")
        table["ci_high"] = data[high] if high else float("nan")
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def figure7_metric_timecourse_raw() -> pd.DataFrame:
    """Figure 7K-N curves without the redundant panel helper column."""
    data = figure7_source("figure7_individual_timecourse_auc.csv")
    return data.drop(columns=["panel"], errors="ignore")


def validate_figure_structure(wb: openpyxl.Workbook) -> None:
    """Keep the canonical Figure 7 and supplementary panel map intact."""
    required_in_order = [
        "Fig.7A_Classifier_accuracy",
        "Fig.7B_Classifier_SHAP",
        "Fig.7C_Confusion_matrix",
        "Fig.7D-E_Prediction_AUC",
        "Fig.7F_SHAP_contributions",
        "Fig.7G-H_Predictor_catalog",
        "Fig.7I_Prediction_onset",
        "Fig.7J_Family_timecourse",
        "Fig.7K-N_Metric_timecourse",
        "Suppl.Fig.1A-D",
        "Suppl.Fig.3A-K",
        "Suppl.Fig.4A-H_SHAP",
    ]
    missing = [name for name in required_in_order if name not in wb.sheetnames]
    if missing:
        raise AssertionError(f"Raw-data workbook is missing canonical figure sheets: {missing}")
    positions = [wb.sheetnames.index(name) for name in required_in_order]
    if positions != sorted(positions):
        raise AssertionError("Raw-data workbook figure sheets are not in canonical panel order")

    for name in [sheet for sheet in required_in_order if sheet.startswith("Fig.7")]:
        worksheet = wb[name]
        headers = [cell.value for cell in worksheet[3] if cell.value is not None]
        forbidden = {"Label", "Animal_Label", "Panel"}.intersection(headers)
        if forbidden:
            raise AssertionError(f"{name} contains helper columns: {sorted(forbidden)}")
        values = {cell.value for row in worksheet.iter_rows() for cell in row}
        stale_datasets = {"Exp1", "Exp3"}.intersection(values)
        if stale_datasets:
            raise AssertionError(f"{name} contains abbreviated dataset names: {sorted(stale_datasets)}")

    shap_sheet = wb["Fig.7B_Classifier_SHAP"]
    if shap_sheet.max_row - 3 != 20 or shap_sheet.max_column != 9:
        raise AssertionError("Figure 7B must contain all 20 plotted features across eight classes")

    supplementary_shap = wb["Suppl.Fig.4A-H_SHAP"]
    if supplementary_shap.max_row - 3 != 719 * 8:
        raise AssertionError(
            "Supplementary Figure 4 must contain all 719 legacy parameters for all eight classes"
        )


def build_raw_data_workbook(output: Path = OUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ground = freezing_ground_truth()
    figure5_frequency, figure5_timecourse = figure5_cluster_tables()
    figure6_frequency, figure6_bouts, figure6_chords, figure6_transitions = figure6_tables()

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    simba = pd.read_csv(RAW_DIR / "simba_validation_manual_vs_automatic.csv").drop(columns=["phase"])
    write_titled_dataframe(wb, "Fig.1A_SimBA_validation", "Figure 1A: SimBA automatic versus manual validation", simba)
    write_titled_dataframe(
        wb,
        "Fig.2A_Ground_truth",
        "Figure 2A: freezing per time bin, Sanguino-Gomez and Krugers, 2024",
        ground[ground["Experiment"] == 1].drop(columns=["Experiment"]).reset_index(drop=True),
    )
    write_titled_dataframe(
        wb,
        "Fig.2B_Ground_truth",
        "Figure 2B: freezing per time bin, Sanguino-Gomez et al., 2024",
        ground[ground["Experiment"] == 3].drop(columns=["Experiment"]).reset_index(drop=True),
    )
    write_titled_dataframe(wb, "Fig.2C_Ground_truth", "Figure 2C: freezing per time bin, combined datasets", ground.reset_index(drop=True))
    write_titled_dataframe(wb, "Fig.2D_Syllable_usage", "Figure 2D: per-animal usage of each syllable label", figure2_syllable_usage_raw())
    write_titled_dataframe(wb, "Fig.2E-F_Precision_recall", "Figure 2E-F: per-animal precision and recall for each syllable label", figure2_precision_recall_raw())
    write_titled_dataframe(wb, "Fig.2G_Freezing_overlap", "Figure 2G: animal-level freezing overlap for syllables 0, 28, and 40", figure2_freezing_overlap_raw())
    write_titled_dataframe(wb, "Fig.2H_Freezing_syllables", "Figure 2H: per-animal freezing syllables 0 and 28 by time in minutes", figure2_freezing_syllables_raw())
    write_titled_dataframe(wb, "Fig.3A_Cluster_frequency", "Figure 3A: behavior frequency in seconds per animal", numeric_animal(cluster_frequency()).reset_index(drop=True))
    write_titled_dataframe(wb, "Fig.3B-H_Cluster_timecourse", "Figure 3B-H: behavior percentage per 30-second time bin", numeric_animal(cluster_timecourse()).reset_index(drop=True))
    write_titled_dataframe(wb, "Fig.4E-H_Frequency_metrics", "Figure 4E-H: diversity and cumulative usage metrics per animal", fig4_frequency_metrics())
    write_titled_dataframe(wb, "Fig.4I_Usage_profile", "Figure 4I: behavior usage proportions per animal", fig4_usage_profile())
    write_titled_dataframe(wb, "Fig.4J-Q_Bout_duration", "Figure 4J-Q: mean bout duration per animal", fig4_bout_duration())
    write_titled_dataframe(wb, "Fig.4R-S_Transition_chords", "Figure 4R-S: mean transitions per animal used in chord plots", fig4_transition_chords())
    write_titled_dataframe(wb, "Fig.4T-W_Transition_metrics", "Figure 4T-W: sequence metrics per animal", fig4_transition_metrics())
    write_titled_dataframe(wb, "Fig.5A-B_Dynamics", "Figure 5A-B: MDS coordinates and behavioral dynamics score per animal", figure5_dynamics())
    write_titled_dataframe(wb, "Fig.5C_Cluster_frequency", "Figure 5C: behavior frequency in seconds per animal and resilience group", figure5_frequency)
    write_titled_dataframe(wb, "Fig.5D-J_Cluster_timecourse", "Figure 5D-J: behavior percentage per 30-second time bin and resilience group", figure5_timecourse)
    write_titled_dataframe(wb, "Fig.6G-K_Frequency_metrics", "Figure 6G-K: diversity metrics per animal", figure6_frequency)
    write_titled_dataframe(wb, "Fig.6L-S_Bout_duration", "Figure 6L-S: mean bout duration per animal and resilience group", figure6_bouts)
    write_titled_dataframe(wb, "Fig.6T-V_Transition_chords", "Figure 6T-V: mean transitions per animal used in resilience chord plots", figure6_chords)
    write_titled_dataframe(wb, "Fig.6W-Z_Transition_metrics", "Figure 6W-Z: sequence metrics per animal and resilience group", figure6_transitions)
    write_titled_dataframe(wb, "Fig.7A_Classifier_accuracy", "Figure 7A: behavior-classifier accuracy and chance reference", figure7_source("figure7_classifier_accuracy.csv"))
    write_titled_dataframe(wb, "Fig.7B_Classifier_SHAP", "Figure 7B: all 20 plotted SHAP features across eight behavior classes", figure7_classifier_shap_raw())
    write_titled_dataframe(wb, "Fig.7C_Confusion_matrix", "Figure 7C: normalized behavior-classifier confusion matrix", figure7_source("figure7_classifier_confusion_matrix.csv").rename(columns={"true_class": "actual_behavior"}))
    write_titled_dataframe(wb, "Fig.7D-E_Prediction_AUC", "Figure 7D-E: plotted held-out-cohort and full-session AUC values", figure7_prediction_auc_raw())
    write_titled_dataframe(wb, "Fig.7F_SHAP_contributions", "Figure 7F: animal-level Shapley contributions to full-session resilience prediction", figure7_shap_per_animal_raw())
    write_titled_dataframe(wb, "Fig.7G-H_Predictor_catalog", "Figure 7G-H: held-out-cohort and cross-validation performance for all predictors", figure7_predictor_catalog_raw())
    write_titled_dataframe(wb, "Fig.7I_Prediction_onset", "Figure 7I: plotted resilience-prediction onset values", figure7_prediction_onset_raw())
    write_titled_dataframe(wb, "Fig.7J_Family_timecourse", "Figure 7J: predictor-family performance across opening-session horizons", figure7_family_timecourse_raw())
    write_titled_dataframe(wb, "Fig.7K-N_Metric_timecourse", "Figure 7K-N: individual metric performance across opening-session horizons", figure7_metric_timecourse_raw())
    write_titled_dataframe(wb, "Suppl.Fig.1A-D", "Supplementary Figure 1A-D: omitted behavior time courses and total frequencies", numeric_animal(supplementary_tracking_time()))
    write_titled_dataframe(wb, "Suppl.Fig.3A-K", "Supplementary Figure 3A-K: distance-metric control scores per animal", supplementary_figure3_scores())
    write_titled_dataframe(wb, "Suppl.Fig.4A-H_SHAP", "Supplementary Figure 4A-H: all 719 legacy classifier parameters per behavior; top 10 plotted per class", supplementary_figure4_shap_raw())
    validate_figure_structure(wb)
    save_workbook(wb, output)
    print(f"Saved: {output.relative_to(REPO)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT,
        help="Output workbook path (default: report/raw_data.xlsx).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else REPO / args.output
    build_raw_data_workbook(output)


if __name__ == "__main__":
    main()
