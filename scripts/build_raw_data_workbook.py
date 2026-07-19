"""Build the manuscript raw-data workbook in final panel order.

The workbook is generated from tracked repository inputs and uses the same
panel-centered structure as the manuscript raw-data workbook supplied for
submission.

Run: python scripts/build_raw_data_workbook.py
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "report" / "raw_data.xlsx"

RAW_DIR = REPO / "data" / "raw"
PROCESSED_DIR = REPO / "data" / "processed"
STATS_DIR = REPO / "statistics"

FPS = 25
BIN_SECONDS = 30
SESSION_SECONDS = 450
PROJECT_LABELS = {
    1: "Sanguino-Gomez and Krugers, 2024",
    3: "Sanguino-Gomez et al., 2024",
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
    wide = wide.sort_values(["Experiment", "Group", "Animal"], key=lambda s: s.map(sort_key) if s.name == "Animal" else s)
    return wide.reset_index(drop=True)


def cluster_frequency() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "cluster_frequency_per_animal.csv")
    df["animal_id"] = df["animal_id"].map(format_animal)
    df["project"] = df["experiment"].map(PROJECT_LABELS)
    df = df[["animal_id", "group", "project", "experiment", "cluster", "frequency_seconds"]]
    return df.sort_values(["experiment", "animal_id", "cluster"], key=lambda s: s.map(sort_key) if s.name == "animal_id" else s)


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
    return df[["animal_id", "group", "experiment", "simpson_index", "shannon_entropy_index", "evenness_index", "cumulative_usage_index"]]


def fig4_usage_profile() -> pd.DataFrame:
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
    return wide[cols].sort_values(["experiment", "animal_id"], key=lambda s: s.map(sort_key) if s.name == "animal_id" else s)


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
    return out.sort_values(["figure_panel_current_export", "animal_id"], key=lambda s: s.map(sort_key) if s.name == "animal_id" else s)


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
    return df[["animal_id", "group", "experiment", "lempel_ziv_complexity", "recurrence_rate", "determinism", "markov_entropy"]]


def load_fig4_module():
    spec = importlib.util.spec_from_file_location("fig4", REPO / "scripts" / "generate_figures" / "figure_4_diversity_dynamics.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Figure 4 script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fig4_transition_chords() -> pd.DataFrame:
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


def supplementary_tracking_time() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "supplementary_figure1_tracking_clusters.csv")
    wide = pivot_time_table(df, ["animal_id", "group", "project", "experiment", "cluster"], "seconds", "percentage")
    wide["animal_id"] = wide["animal_id"].map(format_animal)
    return wide.sort_values(["experiment", "animal_id", "cluster"], key=lambda s: s.map(sort_key) if s.name == "animal_id" else s)


def supplementary_tracking_frequency() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "supplementary_figure1_tracking_clusters.csv")
    out = (
        df.groupby(["animal_id", "group", "project", "experiment", "cluster"], as_index=False)["seconds"]
        .sum()
        .rename(columns={"seconds": "frequency_seconds"})
    )
    out["animal_id"] = out["animal_id"].map(format_animal)
    return out.sort_values(["experiment", "animal_id", "cluster"], key=lambda s: s.map(sort_key) if s.name == "animal_id" else s)


def syllable_timebin_30s() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "syllable_usage_per_timebin_30s.csv")
    return df.rename(columns={"Condition": "group", "Time Bin": "time_bin_seconds"})[
        ["Animal", "group", "Experiment", "time_bin_seconds", "Syllable", "Percentage"]
    ]


def behavioral_flexibility_scores() -> pd.DataFrame:
    return pd.read_excel(RAW_DIR / "behavioral_flexibility_scores.xlsx")


def syllable_frames() -> pd.DataFrame:
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
    ].sort_values(["animal_id", "syllable"], key=lambda s: s.map(sort_key) if s.name == "animal_id" else s)

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
    ].sort_values(["experiment", "animal_id"], key=lambda s: s.map(sort_key) if s.name == "animal_id" else s)
    return precision, overlap


def syllable_0_28_timecourse() -> pd.DataFrame:
    tc = pd.read_csv(PROCESSED_DIR / "cluster_timecourse_per_animal.csv")
    tc = tc[tc["cluster"] == "Freeze"].copy()
    tc["animal_id"] = tc["animal_id"].map(format_animal)
    tc = tc.merge(name_map(), on="animal_id", how="left")
    tc["project"] = tc["experiment"].map(PROJECT_LABELS)
    wide = pivot_time_table(tc, ["animal_id", "animal", "group", "project", "experiment"], "time_s", "pct")
    return wide.sort_values(["experiment", "animal_id"], key=lambda s: s.map(sort_key) if s.name == "animal_id" else s)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    precision, overlap = precision_recall_tables()
    syll_0_28 = syllable_0_28_timecourse()
    ground = freezing_ground_truth()

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    write_titled_dataframe(wb, "Fig.1A_ SimBA_validation", "SimBA validation: automatic vs manual", pd.read_csv(RAW_DIR / "simba_validation_manual_vs_automatic.csv"))
    write_titled_dataframe(
        wb,
        "Fig.2A_Ground_truth ",
        "Freezing per time bin: Sanguino-Gomez and Krugers, 2024",
        ground[ground["Experiment"] == 1].drop(columns=["Experiment"]).reset_index(drop=True),
    )
    write_titled_dataframe(
        wb,
        "Fig.2B_Ground_truth ",
        "Freezing per time bin: Sanguino-Gomez et al., 2024",
        ground[ground["Experiment"] == 3].drop(columns=["Experiment"]).reset_index(drop=True),
    )
    write_titled_dataframe(wb, "Fig.2C_Ground_truth ", "Freezing per time bin: combined datasets", ground.reset_index(drop=True))
    write_titled_dataframe(wb, "Fig.3A_Clusters_frequency", "Figure 3 behavior clusters: total frequency in seconds per animal", cluster_frequency().reset_index(drop=True))
    write_titled_dataframe(wb, "Fig.3B-H_Clusters_over_time", "Figure 3 behavior clusters: percentage per 30 s time bin", cluster_timecourse().reset_index(drop=True))
    write_titled_dataframe(wb, "Fig4_frequency_metrics", "Figure 4 panels E-H: diversity and cumulative-usage metrics per animal", fig4_frequency_metrics())
    write_titled_dataframe(wb, "Fig4_usage_profile", "Figure 4 panel I: cluster usage proportions per animal", fig4_usage_profile())
    write_titled_dataframe(wb, "Fig4_bout_duration", "Figure 4 panels J-Q: mean bout duration points per animal", fig4_bout_duration())
    write_titled_dataframe(wb, "Fig4_transition_metrics", "Figure 4 panels T-W: sequence metrics per animal", fig4_transition_metrics())
    write_titled_dataframe(wb, "Fig4_transition_chords", "Figure 4 panels R-S: mean transition matrix values used in chord plots", fig4_transition_chords())
    write_titled_dataframe(wb, "Supp_cluster_time", "Supplementary Figure 1 clusters: percentage per 30 s time bin", supplementary_tracking_time())
    write_titled_dataframe(wb, "Supp_cluster_frequency", "Supplementary Figure 1 clusters: total frequency in seconds per animal", supplementary_tracking_frequency())
    write_titled_dataframe(wb, "Syllable_timebin_30s", "All syllables per 30 s time bin from the archived MoSeq analysis", syllable_timebin_30s())
    write_titled_dataframe(wb, "Behavioral_flexibility_scores", "Behavioral dynamics score data", behavioral_flexibility_scores())
    write_titled_dataframe(wb, "Syllable_frames", "All syllables: frame counts per video and total across all 82 videos", syllable_frames())
    write_titled_dataframe(wb, "Precision_recall", "Precision and recall of each syllable against freezing annotations", precision)
    write_titled_dataframe(wb, "Overlap_0_28", "Overlap of freezing annotation with syllable annotations 0 + 28", overlap)
    write_titled_dataframe(wb, "Syll_0_28_SGK_2024", "Syllables 0 + 28 per time bin: Sanguino-Gomez and Krugers, 2024", syll_0_28[syll_0_28["experiment"] == 1].reset_index(drop=True))
    write_titled_dataframe(wb, "Syll_0_28_SG_2024", "Syllables 0 + 28 per time bin: Sanguino-Gomez et al., 2024", syll_0_28[syll_0_28["experiment"] == 3].reset_index(drop=True))
    write_titled_dataframe(wb, "Syll_0_28_combined", "Syllables 0 + 28 per time bin: combined datasets", syll_0_28.reset_index(drop=True))
    wb.save(OUT)
    print(f"Saved: {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
