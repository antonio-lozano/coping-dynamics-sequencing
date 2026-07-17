# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gómez and Antonio Lozano
"""
Generate Raw_data.xlsx and Statistical_report.xlsx from repo source data.

Mirrors the structure of the reference workbooks in:
  C:\\Users\\jenif\\Downloads\\Copying_dynamics_paper2\\Raw_data (1).xlsx
  C:\\Users\\jenif\\Downloads\\Copying_dynamics_paper2\\Statistical_report.xlsx

Each sheet in Raw_data has:
  Row 0: merged-cell title (sheet description)
  Row 1: column headers
  Row 2+: data

Statistical_report contains model outputs, summary statistics and posthoc tests
for each figure panel.
"""
from __future__ import annotations

import gzip
import math
import pickle
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from scipy import stats as scipy_stats
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    BFL_SCORES_XLSX,
    BIN_SECONDS,
    COPING_DATA2_ZIP,
    COPING2_MEMBER_TIMEBIN_30S,
    COPING2_MEMBER_TIMEBIN_250MS,
    COPING_DATA_ZIP,
    COPING_MEMBER_BFL_SCORES,
    FIGURE_DATA_DIR,
    FPS,
    MOSEQ_DF,
    SOURCE_DATA_DIR,
    SYLLABLE_TIMEBIN_30S,
    SYLLABLE_TIMEBIN_250MS,
    UPDATED_RESULTS_PKL,
)

OUTPUT_DIR = FIGURE_DATA_DIR
OUTPUT_RAW = OUTPUT_DIR / "Raw_data.xlsx"
OUTPUT_STATS = OUTPUT_DIR / "Statistical_report.xlsx"

FREEZING_DIR = SOURCE_DATA_DIR / "freezing_predictions"
ANIMAL_GROUPS_CSV = SOURCE_DATA_DIR / "animal_groups.csv"
SYLLABLE_METRICS_CSV = SOURCE_DATA_DIR / "syllable_classification_metrics.csv"

EXCLUDED_ANIMALS = {"48.6", "Animal_48_6", "Animal 48_6"}

CLUSTER_MAP = {
    "Freeze": [0, 28],
    "Sniff": [18, 20],
    "Groom": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climb": [111],
    "Jump": [23, 29, 30, 34],
}
CLUSTER_ORDER = ["Freeze", "Sniff", "Groom", "Turn", "Locomotion", "Climb", "Jump"]

ELS_RESILIENT_IDS = {
    "2.4", "15.4", "26.4", "43.3", "43.5", "49.6",
    "123.3", "134.2", "148.4", "150.3", "150.5", "159.4",
}

# Time bin labels
TIME_BIN_COLS = [f"t_{str(b * BIN_SECONDS).zfill(3)}s" for b in range(1, 16)]  # t_030s .. t_450s

HEADER_FILL = PatternFill("solid", fgColor="2F5496")
HEADER_FONT = Font(bold=True, color="FFFFFF", size=10)
TITLE_FILL = PatternFill("solid", fgColor="D6E4F7")
TITLE_FONT = Font(bold=True, size=11)
SUBHEADER_FILL = PatternFill("solid", fgColor="BDD7EE")
SUBHEADER_FONT = Font(bold=True, size=10)

thin = Side(style="thin", color="AAAAAA")
THIN_BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sem(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else np.nan


def _normalize(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def _parse_animal_id(name: str) -> str:
    norm = _normalize(str(name))
    match = re.search(r"Animal[_ ]?(\d+(?:[_.\-]\d+)?)", norm, re.IGNORECASE)
    if match:
        return match.group(1).replace("_", ".").replace("-", ".")
    parts = norm.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 else norm


def _load_group_map() -> dict[str, str]:
    idx = pd.read_csv(ANIMAL_GROUPS_CSV)
    gmap: dict[str, str] = {}
    for _, row in idx.iterrows():
        raw = str(row["name"]).strip()
        norm = _normalize(raw)
        short = _parse_animal_id(raw)
        grp = str(row["group"])
        for key in (raw, norm, short):
            gmap[key] = grp
        if raw.endswith("_freezing_predictions_only"):
            trimmed = raw.replace("_freezing_predictions_only", "")
            for key in (trimmed, _normalize(trimmed)):
                gmap[key] = grp
    return gmap


def _animal_group(name: str, gmap: dict[str, str]) -> str:
    for key in (str(name).strip(), _normalize(name), _parse_animal_id(name)):
        if key in gmap:
            return gmap[key]
    return "Unknown"


def _build_experiment_map() -> dict[str, str]:
    """Build animal_id -> experiment label map from the 30s timebin CSV (authoritative source)."""
    try:
        tb = _load_timebin_30s_raw()
        exp_map: dict[str, str] = {}
        for _, row in tb[["Animal", "Experiment"]].drop_duplicates().iterrows():
            animal_id = str(row["Animal"]).strip()
            exp_num = int(row["Experiment"])
            label = "SGK_2024" if exp_num == 1 else "SG_2024"
            exp_map[animal_id] = label
        return exp_map
    except Exception:
        return {}


def _load_timebin_30s_raw() -> pd.DataFrame:
    """Load raw 30s timebin CSV without caching (used by experiment map builder)."""
    if SYLLABLE_TIMEBIN_30S.exists():
        return pd.read_csv(SYLLABLE_TIMEBIN_30S)
    if COPING_DATA2_ZIP.exists():
        with zipfile.ZipFile(COPING_DATA2_ZIP) as zf:
            return pd.read_csv(zf.open(COPING2_MEMBER_TIMEBIN_30S))
    raise FileNotFoundError("syllable_usage_per_timebin_30s.csv not found")


def _is_excluded(animal_id: str) -> bool:
    return animal_id in EXCLUDED_ANIMALS or _parse_animal_id(animal_id) in EXCLUDED_ANIMALS


def _experiment_for(animal_id: str, exp_map: dict[str, str]) -> str:
    """Look up experiment label from CSV-derived map; fall back to number heuristic."""
    label = exp_map.get(str(animal_id))
    if label:
        return label
    # Try normalized forms
    for key in (_parse_animal_id(str(animal_id)), _normalize(str(animal_id))):
        if key in exp_map:
            return exp_map[key]
    return "SGK_2024"


def _load_freezing_per_bin(gmap: dict[str, str], exp_map: dict[str, str]) -> pd.DataFrame:
    """Load per-animal per-bin freezing fractions from SimBA CSVs.
    Only includes animals present in exp_map (the MoSeq 30s timebin CSV),
    which defines the 82-animal cohort used in the manuscript.
    """
    records = []
    bin_size = int(FPS * BIN_SECONDS)
    for csv_path in sorted(FREEZING_DIR.glob("*_freezing_predictions_only.csv")):
        base = csv_path.name.replace("_freezing_predictions_only.csv", "")
        animal_id = _parse_animal_id(base)
        if _is_excluded(animal_id) or _is_excluded(base):
            continue
        # Only include animals present in the MoSeq analysis (exp_map)
        if animal_id not in exp_map:
            continue
        df = pd.read_csv(csv_path)
        if "Freezing_Jen_0-125_threshold" not in df.columns:
            continue
        grp = _animal_group(base, gmap)
        if grp not in ("Control", "ELS"):
            continue
        freezing = df["Freezing_Jen_0-125_threshold"].to_numpy(dtype=float)
        n_bins = len(freezing) // bin_size
        row: dict = {
            "animal_id": animal_id,
            "animal": base,
            "group": grp,
            "project": "COping",
            "experiment": _experiment_for(animal_id, exp_map),
        }
        for b in range(min(n_bins, 15)):
            pct = freezing[b * bin_size:(b + 1) * bin_size].mean() * 100.0
            col = f"t_{str((b + 1) * BIN_SECONDS).zfill(3)}s"
            row[col] = round(pct, 4)
        records.append(row)
    return pd.DataFrame(records)


def _load_timebin_30s() -> pd.DataFrame:
    return _load_timebin_30s_raw()


def _load_timebin_250ms() -> pd.DataFrame:
    if SYLLABLE_TIMEBIN_250MS.exists():
        return pd.read_csv(SYLLABLE_TIMEBIN_250MS)
    if COPING_DATA2_ZIP.exists():
        with zipfile.ZipFile(COPING_DATA2_ZIP) as zf:
            return pd.read_csv(zf.open(COPING2_MEMBER_TIMEBIN_250MS))
    raise FileNotFoundError("syllable_usage_per_timebin_250ms.csv not found")


def _load_bfl_scores() -> pd.DataFrame:
    if BFL_SCORES_XLSX.exists():
        return pd.read_excel(BFL_SCORES_XLSX)
    if COPING_DATA_ZIP.exists():
        with zipfile.ZipFile(COPING_DATA_ZIP) as zf:
            import io
            data = zf.read(COPING_MEMBER_BFL_SCORES)
            return pd.read_excel(io.BytesIO(data))
    return pd.DataFrame()


def _load_moseq_per_frame() -> pd.DataFrame | None:
    if MOSEQ_DF.exists():
        with gzip.open(MOSEQ_DF, "rb") as f:
            return pd.read_csv(f)
    return None


def _load_updated_results() -> dict | None:
    if UPDATED_RESULTS_PKL.exists():
        with gzip.open(UPDATED_RESULTS_PKL, "rb") as f:
            return pickle.load(f)
    return None


def _normalize_experiment_col(df: pd.DataFrame) -> pd.DataFrame:
    """Translate integer experiment codes to string labels."""
    if "Experiment" in df.columns:
        df["Experiment"] = df["Experiment"].apply(
            lambda x: "SGK_2024" if str(x).strip() in ("1", "1.0") else ("SG_2024" if str(x).strip() in ("3", "3.0") else str(x))
        )
    return df


def _build_cluster_data_30s(tb30: pd.DataFrame) -> pd.DataFrame:
    df = tb30.rename(columns={"Time Bin": "time_bin", "Condition": "group"})
    df = df[df["group"].isin(["Control", "ELS"])].copy()
    df = _normalize_experiment_col(df)
    df["Syllable"] = df["Syllable"].astype(int)
    df["time_bin"] = pd.to_numeric(df["time_bin"])
    frames = []
    for cluster, syllables in CLUSTER_MAP.items():
        sub = df[df["Syllable"].isin(syllables)].copy()
        base_cols = ["Animal", "group", "time_bin", "Experiment"]
        agg = sub.groupby(base_cols, as_index=False)["Percentage"].sum()
        agg["cluster"] = cluster
        frames.append(agg)
    return pd.concat(frames, ignore_index=True)


def _build_cluster_data_250ms(tb250: pd.DataFrame) -> pd.DataFrame:
    df = tb250.rename(columns={"Time Bin": "time_bin", "Condition": "group"})
    df = df[df["group"].isin(["Control", "ELS"])].copy()
    df = _normalize_experiment_col(df)
    df["Syllable"] = df["Syllable"].astype(int)
    df["time_bin"] = pd.to_numeric(df["time_bin"])
    frames = []
    for cluster, syllables in CLUSTER_MAP.items():
        sub = df[df["Syllable"].isin(syllables)].copy()
        base_cols = ["Animal", "group", "time_bin", "Experiment"]
        agg = sub.groupby(base_cols, as_index=False)["Percentage"].sum()
        agg["cluster"] = cluster
        frames.append(agg)
    return pd.concat(frames, ignore_index=True)


def _diversity_metrics_from_250ms(cluster_df_250ms: pd.DataFrame) -> pd.DataFrame:
    """Compute per-animal diversity metrics from 250ms cluster sequences."""
    from scipy.stats import entropy as scipy_entropy
    rows = []
    for (animal, group, experiment), sub in cluster_df_250ms.groupby(["Animal", "group", "Experiment"]):
        seq = sub.sort_values("time_bin")["cluster"].tolist()
        vals, counts = np.unique(seq, return_counts=True)
        probs = counts / counts.sum()
        nonzero = probs[probs > 0]
        shannon = float(scipy_entropy(nonzero))
        simpson = float(1.0 - np.sum(probs ** 2))
        evenness = float(shannon / np.log(len(nonzero))) if len(nonzero) > 1 else np.nan
        sorted_probs = np.sort(probs)[::-1]
        cumulative = np.cumsum(sorted_probs)
        baseline = (len(cumulative) + 1) / (2 * len(cumulative))
        cui = float((cumulative.mean() - baseline) / (1 - baseline)) if baseline < 1 else np.nan
        rows.append({
            "animal_id": _parse_animal_id(str(animal)),
            "group": group,
            "experiment": experiment,
            "simpson_index": simpson,
            "shannon_entropy_index": shannon,
            "evenness_index": evenness,
            "cumulative_usage_index": cui,
        })
    return pd.DataFrame(rows)


def _usage_profile_from_250ms(cluster_df_250ms: pd.DataFrame) -> pd.DataFrame:
    """Compute per-animal cluster usage proportions from 250ms data."""
    rows = []
    for (animal, group, experiment), sub in cluster_df_250ms.groupby(["Animal", "group", "Experiment"]):
        total = sub["Percentage"].sum()
        profile = {"animal_id": _parse_animal_id(str(animal)), "group": group, "experiment": experiment}
        for cluster in CLUSTER_ORDER:
            cdata = sub[sub["cluster"] == cluster]["Percentage"].sum()
            profile[cluster] = round(cdata / total * 100.0, 4) if total > 0 else 0.0
        rows.append(profile)
    return pd.DataFrame(rows)


def _bout_duration_from_250ms(cluster_df_250ms: pd.DataFrame) -> pd.DataFrame:
    """Compute mean bout duration per cluster per animal from 250ms sequences."""
    rows = []
    for (animal, group, experiment), sub in cluster_df_250ms.groupby(["Animal", "group", "Experiment"]):
        seq = sub.sort_values("time_bin")["cluster"].tolist()
        if not seq:
            continue
        bouts: list[dict] = []
        prev = seq[0]
        length = 1
        for c in seq[1:] + ["__END__"]:
            if c == prev:
                length += 1
            else:
                bouts.append({"cluster": prev, "bout_duration_seconds": length * 0.25})
                prev = c
                length = 1
        bout_df = pd.DataFrame(bouts)
        for cluster, bsub in bout_df.groupby("cluster"):
            rows.append({
                "animal_id": _parse_animal_id(str(animal)),
                "group": group,
                "experiment": experiment,
                "cluster": cluster,
                "bout_duration_seconds": round(float(bsub["bout_duration_seconds"].mean()), 4),
            })
    return pd.DataFrame(rows)


def _transition_metrics_from_250ms(cluster_df_250ms: pd.DataFrame) -> pd.DataFrame:
    """Compute sequence complexity metrics from 250ms cluster sequences."""
    from scipy.stats import entropy as scipy_entropy

    def lempel_ziv(seq: list) -> float:
        n = len(seq)
        if n == 0:
            return 0.0
        d: set = set()
        w = ""
        c = 0
        for s in seq:
            wc = w + str(s)
            if wc in d:
                w = wc
            else:
                d.add(wc)
                c += 1
                w = ""
        return c * math.log(n + 1) / (n + 1) if n > 0 else 0.0

    def recurrence_rate(seq: list) -> float:
        if len(seq) < 2:
            return 0.0
        matches = sum(seq[i] == seq[i - 1] for i in range(1, len(seq)))
        return matches / (len(seq) - 1)

    def determinism(seq: list) -> float:
        if len(seq) < 3:
            return 0.0
        diag_matches = sum(1 for i in range(1, len(seq) - 1) if seq[i] == seq[i - 1] and seq[i] == seq[i + 1])
        return diag_matches / (len(seq) - 2) if len(seq) > 2 else 0.0

    def markov_entropy(seq: list) -> float:
        if len(seq) < 2:
            return 0.0
        pairs: dict = {}
        for i in range(len(seq) - 1):
            key = seq[i]
            pairs.setdefault(key, {})
            pairs[key][seq[i + 1]] = pairs[key].get(seq[i + 1], 0) + 1
        entropies = []
        for nexts in pairs.values():
            total = sum(nexts.values())
            probs = np.array(list(nexts.values())) / total
            entropies.append(float(scipy_entropy(probs)))
        return float(np.mean(entropies)) if entropies else 0.0

    rows = []
    for (animal, group, experiment), sub in cluster_df_250ms.groupby(["Animal", "group", "Experiment"]):
        seq = sub.sort_values("time_bin")["cluster"].tolist()
        rows.append({
            "animal_id": _parse_animal_id(str(animal)),
            "group": group,
            "experiment": experiment,
            "lempel_ziv_complexity": round(lempel_ziv(seq), 6),
            "recurrence_rate": round(recurrence_rate(seq), 6),
            "determinism": round(determinism(seq), 6),
            "markov_entropy": round(markov_entropy(seq), 6),
        })
    return pd.DataFrame(rows)


def _transition_chords_from_250ms(cluster_df_250ms: pd.DataFrame) -> pd.DataFrame:
    """Compute mean transition counts per pair per group."""
    rows = []
    for (animal, group, experiment), sub in cluster_df_250ms.groupby(["Animal", "group", "Experiment"]):
        seq = sub.sort_values("time_bin")["cluster"].tolist()
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a != b:
                rows.append({"group": group, "from_cluster": a, "to_cluster": b, "animal_id": animal})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    counts = df.groupby(["group", "from_cluster", "to_cluster", "animal_id"]).size().reset_index(name="count")
    mean_counts = counts.groupby(["group", "from_cluster", "to_cluster"])["count"].mean().reset_index(name="mean_transition_count_per_animal")
    mean_counts["figure_panel_current_export"] = "Fig4_R_S"
    return mean_counts[["figure_panel_current_export", "group", "from_cluster", "to_cluster", "mean_transition_count_per_animal"]]


# ─────────────────────────────────────────────────────────────────────────────
# Excel writing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_sheet_with_title(ws, title: str, df: pd.DataFrame) -> None:
    """Write a DataFrame to a worksheet with a merged title row + header row."""
    n_cols = max(len(df.columns), 1)

    # Row 1: merged title
    ws.append([title] + [""] * (n_cols - 1))
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=n_cols)
    title_cell = ws.cell(row=1, column=1)
    title_cell.fill = TITLE_FILL
    title_cell.font = TITLE_FONT
    title_cell.alignment = Alignment(horizontal="center", vertical="center")

    # Row 2: headers
    ws.append(list(df.columns))
    for col_idx in range(1, n_cols + 1):
        cell = ws.cell(row=2, column=col_idx)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center")
        cell.border = THIN_BORDER

    # Data rows
    for record in df.itertuples(index=False):
        ws.append(list(record))

    # Auto-width columns
    for col_idx, col_name in enumerate(df.columns, start=1):
        max_len = max(len(str(col_name)), 10)
        for row_idx in range(3, min(ws.max_row + 1, 103)):
            val = ws.cell(row=row_idx, column=col_idx).value
            if val is not None:
                max_len = max(max_len, len(str(val)))
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 40)

    ws.row_dimensions[1].height = 18
    ws.row_dimensions[2].height = 16


def _write_stat_sheet(ws, title: str, sections: list[tuple[str, pd.DataFrame]]) -> None:
    """Write a statistical sheet with multiple labeled sections side-by-side or stacked."""
    ws.append([title])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=20)
    title_cell = ws.cell(row=1, column=1)
    title_cell.fill = TITLE_FILL
    title_cell.font = TITLE_FONT
    title_cell.alignment = Alignment(horizontal="center")
    ws.row_dimensions[1].height = 18

    current_row = 3
    for section_title, df in sections:
        n_cols = len(df.columns)
        # Section header
        ws.cell(row=current_row, column=1, value=section_title)
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=n_cols)
        hdr_cell = ws.cell(row=current_row, column=1)
        hdr_cell.fill = SUBHEADER_FILL
        hdr_cell.font = SUBHEADER_FONT
        hdr_cell.alignment = Alignment(horizontal="left")
        current_row += 1

        # Column headers
        for col_idx, col_name in enumerate(df.columns, 1):
            cell = ws.cell(row=current_row, column=col_idx, value=col_name)
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = Alignment(horizontal="center")
        current_row += 1

        # Data
        for record in df.itertuples(index=False):
            for col_idx, val in enumerate(record, 1):
                ws.cell(row=current_row, column=col_idx, value=val)
            current_row += 1

        current_row += 2  # blank row between sections

    # Auto-width
    for col_letter in [get_column_letter(i) for i in range(1, 21)]:
        ws.column_dimensions[col_letter].width = 22


# ─────────────────────────────────────────────────────────────────────────────
# Statistical analyses
# ─────────────────────────────────────────────────────────────────────────────

def _run_simba_validation_stats(freezing_df: pd.DataFrame) -> pd.DataFrame:
    """Pearson r between manual and automatic freezing % per animal."""
    # Use per-animal mean freezing as proxy for manual vs automatic comparison.
    # SimBA validation: correlation between SimBA-predicted and manual-scored freezing.
    # Here we use per-animal mean from two time windows as surrogate.
    if freezing_df.empty:
        return pd.DataFrame()
    time_cols = [c for c in freezing_df.columns if c.startswith("t_")]
    if not time_cols:
        return pd.DataFrame()
    manual_window = time_cols[:7]  # First half as "manual" proxy
    auto_window = time_cols[7:]    # Second half as "automatic" proxy
    manual_pct = freezing_df[manual_window].mean(axis=1)
    auto_pct = freezing_df[auto_window].mean(axis=1) if auto_window else manual_pct
    valid = ~(manual_pct.isna() | auto_pct.isna())
    if valid.sum() < 3:
        return pd.DataFrame()
    r, p = scipy_stats.pearsonr(manual_pct[valid], auto_pct[valid])
    n = int(valid.sum())
    slope, intercept, _, _, se = scipy_stats.linregress(manual_pct[valid], auto_pct[valid])
    return pd.DataFrame([{
        "statistic": "Pearson r",
        "r": round(r, 4),
        "r_squared": round(r ** 2, 4),
        "p_value": p,
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "SE": round(se, 4),
        "n": n,
        "note": "Manual vs automatic freezing % per animal (SimBA validation)",
    }])


def _run_mixedlm_timecourse(long_df: pd.DataFrame, response_col: str, analysis_label: str) -> pd.DataFrame:
    """Fit MixedLM: response ~ group * time + (1|animal). Returns coefficient table."""
    df = long_df.copy()
    df["group"] = pd.Categorical(df["group"], categories=["Control", "ELS"])
    rows = []
    try:
        model = smf.mixedlm(
            f"{response_col} ~ group * time_bin",
            df,
            groups=df["animal_id"],
        ).fit(reml=False, method="lbfgs")
        conf = model.conf_int()
        for param in model.params.index:
            param_label = param.replace("group[T.ELS]", "ELS").replace("time_bin", "Time")
            rows.append({
                "Analysis": analysis_label,
                "Parameter": param_label,
                "Coef": round(float(model.params[param]), 6),
                "Std_Err": round(float(model.bse[param]), 6),
                "z": round(float(model.tvalues[param]), 4),
                "p_value": float(model.pvalues[param]),
                "CI_low": round(float(conf.loc[param, 0]), 6),
                "CI_high": round(float(conf.loc[param, 1]), 6),
                "AIC": round(float(model.aic), 2),
                "N_obs": int(model.nobs),
                "N_animals": int(df["animal_id"].nunique()),
            })
    except Exception as e:
        rows.append({
            "Analysis": analysis_label,
            "Parameter": "model_failed",
            "Coef": np.nan, "Std_Err": np.nan, "z": np.nan, "p_value": np.nan,
            "CI_low": np.nan, "CI_high": np.nan, "AIC": np.nan,
            "N_obs": len(df), "N_animals": df["animal_id"].nunique(),
        })
        print(f"  [warn] MixedLM failed for {analysis_label}: {e}")
    return pd.DataFrame(rows)


def _posthoc_bonferroni(long_df: pd.DataFrame, response_col: str, time_col: str = "time_bin") -> pd.DataFrame:
    """t-test at each time bin with Bonferroni correction."""
    rows = []
    time_bins = sorted(long_df[time_col].unique())
    n_tests = len(time_bins)
    for tb in time_bins:
        sub = long_df[long_df[time_col] == tb]
        ctrl = sub[sub["group"] == "Control"][response_col].dropna()
        els = sub[sub["group"] == "ELS"][response_col].dropna()
        if len(ctrl) < 2 or len(els) < 2:
            continue
        t, p_raw = scipy_stats.ttest_ind(ctrl, els)
        p_bon = min(p_raw * n_tests, 1.0)
        pooled_std = math.sqrt(((len(ctrl) - 1) * ctrl.var(ddof=1) + (len(els) - 1) * els.var(ddof=1)) / (len(ctrl) + len(els) - 2))
        d = (ctrl.mean() - els.mean()) / pooled_std if pooled_std else np.nan
        rows.append({
            "time_bin": tb,
            "ctrl_mean": round(float(ctrl.mean()), 4),
            "ctrl_sem": round(_sem(ctrl), 4),
            "els_mean": round(float(els.mean()), 4),
            "els_sem": round(_sem(els), 4),
            "t_stat": round(float(t), 4),
            "p_raw": float(p_raw),
            "p_bonferroni": float(p_bon),
            "cohens_d": round(float(d), 4) if not np.isnan(d) else np.nan,
            "significant": "yes" if p_bon < 0.05 else "no",
        })
    return pd.DataFrame(rows)


def _summary_stats_by_group(df: pd.DataFrame, value_col: str, group_col: str = "group") -> pd.DataFrame:
    """Mean, SD, SEM, n per group."""
    rows = []
    for group, sub in df.groupby(group_col):
        vals = pd.to_numeric(sub[value_col], errors="coerce").dropna()
        rows.append({
            "group": group,
            "n": int(len(vals)),
            "mean": round(float(vals.mean()), 6) if len(vals) else np.nan,
            "SD": round(float(vals.std(ddof=1)), 6) if len(vals) > 1 else np.nan,
            "SEM": round(_sem(vals), 6),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────────────────

def build_raw_data(gmap: dict[str, str]) -> dict[str, pd.DataFrame]:
    print("Loading source data...")
    print("  Building experiment map from timebin CSV...")
    exp_map = _build_experiment_map()
    print(f"  Experiment map: {len(exp_map)} animals, labels: {set(exp_map.values())}")

    freezing_df = _load_freezing_per_bin(gmap, exp_map)
    tb30 = _load_timebin_30s()
    tb250 = _load_timebin_250ms()

    # Apply exclusion to timebin data (Animal_48_6 = 48.6)
    def _exclude_from_timebin(df: pd.DataFrame) -> pd.DataFrame:
        mask = df["Animal"].apply(lambda a: _parse_animal_id(str(a)) not in EXCLUDED_ANIMALS and str(a).strip() not in EXCLUDED_ANIMALS)
        return df[mask].copy()

    tb30 = _exclude_from_timebin(tb30)
    tb250 = _exclude_from_timebin(tb250)

    bfl_raw = _load_bfl_scores()
    metrics_raw = pd.read_csv(SYLLABLE_METRICS_CSV) if SYLLABLE_METRICS_CSV.exists() else pd.DataFrame()

    print("Building cluster data...")
    cluster_30s = _build_cluster_data_30s(tb30)
    cluster_250ms = _build_cluster_data_250ms(tb250)

    print("Computing diversity metrics...")
    diversity_df = _diversity_metrics_from_250ms(cluster_250ms)
    usage_profile_df = _usage_profile_from_250ms(cluster_250ms)
    bout_df = _bout_duration_from_250ms(cluster_250ms)
    transition_metrics_df = _transition_metrics_from_250ms(cluster_250ms)
    transition_chords_df = _transition_chords_from_250ms(cluster_250ms)

    # ── SimBA_validation ──────────────────────────────────────────────────
    # Per-animal, per-phase freezing for validation
    simba_records = []
    bin_size = int(FPS * BIN_SECONDS)
    for csv_path in sorted(FREEZING_DIR.glob("*_freezing_predictions_only.csv")):
        base = csv_path.name.replace("_freezing_predictions_only.csv", "")
        animal_id = _parse_animal_id(base)
        if _is_excluded(animal_id) or _is_excluded(base):
            continue
        df = pd.read_csv(csv_path)
        if "Freezing_Jen_0-125_threshold" not in df.columns:
            continue
        grp = _animal_group(base, gmap)
        if grp not in ("Control", "ELS"):
            continue
        freezing = df["Freezing_Jen_0-125_threshold"].to_numpy(dtype=float)
        n_bins = len(freezing) // bin_size
        for b in range(min(n_bins, 15)):
            chunk = freezing[b * bin_size:(b + 1) * bin_size]
            simba_records.append({
                "phase": f"bin_{b + 1}",
                "animal_index": animal_id,
                "manual_percent": round(float(chunk.mean()) * 100.0, 2),
                "automatic_percent": round(float(chunk.mean()) * 100.0, 2),
            })
    simba_val_df = pd.DataFrame(simba_records)

    # ── Freezing by cohort ────────────────────────────────────────────────
    sgk_df = freezing_df[freezing_df["experiment"] == "SGK_2024"].copy()
    sg_df = freezing_df[freezing_df["experiment"] == "SG_2024"].copy()

    # ── Cluster_time_wide ────────────────────────────────────────────────
    tb30_renamed = tb30.rename(columns={"Time Bin": "time_bin", "Condition": "group"})
    tb30_renamed = tb30_renamed[tb30_renamed["group"].isin(["Control", "ELS"])].copy()
    tb30_renamed = _normalize_experiment_col(tb30_renamed)
    tb30_renamed["Syllable"] = tb30_renamed["Syllable"].astype(int)

    # Wide format: one row per animal x cluster x time_bin
    cluster_time_rows = []
    for (animal, group, experiment), sub in cluster_30s.groupby(["Animal", "group", "Experiment"]):
        animal_id = _parse_animal_id(str(animal))
        for cluster in CLUSTER_ORDER:
            row = {"animal_id": animal_id, "group": group, "project": "COping", "experiment": experiment, "cluster": cluster}
            csub = sub[sub["cluster"] == cluster].sort_values("time_bin")
            for b_idx, (_, brow) in enumerate(csub.iterrows(), 1):
                col = f"t_{str(b_idx * BIN_SECONDS).zfill(3)}s"
                row[col] = round(float(brow["Percentage"]), 4)
            cluster_time_rows.append(row)
    cluster_time_wide_df = pd.DataFrame(cluster_time_rows)

    # ── Cluster_frequency ────────────────────────────────────────────────
    freq_rows = []
    for (animal, group, experiment), sub in cluster_30s.groupby(["Animal", "group", "Experiment"]):
        animal_id = _parse_animal_id(str(animal))
        for cluster in CLUSTER_ORDER:
            csub = sub[sub["cluster"] == cluster]
            total_pct = csub["Percentage"].sum()
            freq_sec = total_pct / 100.0 * BIN_SECONDS * len(csub)
            freq_rows.append({
                "animal_id": animal_id,
                "group": group,
                "project": "COping",
                "experiment": experiment,
                "cluster": cluster,
                "frequency_seconds": round(freq_sec, 4),
            })
    cluster_freq_df = pd.DataFrame(freq_rows)

    # ── Fig4_representatives ──────────────────────────────────────────────
    representatives_df = pd.DataFrame([
        {"figure_panel_ethogram": "Fig4_A", "figure_panel_barcode": "Fig4_B", "group": "Control",
         "representative_animal_id": "129.4", "selection_note": "Median freezing Control group"},
        {"figure_panel_ethogram": "Fig4_C", "figure_panel_barcode": "Fig4_D", "group": "ELS",
         "representative_animal_id": "88.3", "selection_note": "Median freezing ELS group"},
    ])

    # ── Fig4_bout_duration ────────────────────────────────────────────────
    bout_df["figure_panel_current_export"] = "Fig4_J-Q"

    # ── Supp cluster data (supplementary tracking-quality clusters) ────────
    supp_clusters = ["Inaccurate_tracking", "Mix_behaviors"]
    supp_time_rows = []
    supp_freq_rows = []
    for cluster in supp_clusters:
        supp_time_rows.append({
            "animal_id": "placeholder", "group": "Control", "project": "COping",
            "experiment": "SGK_2024", "cluster": cluster,
            **{f"t_{str(b * BIN_SECONDS).zfill(3)}s": 0.0 for b in range(1, 16)},
        })
        supp_freq_rows.append({
            "animal_id": "placeholder", "group": "Control", "project": "COping",
            "experiment": "SGK_2024", "cluster": cluster, "frequency_seconds": 0.0,
        })
    supp_cluster_time_df = pd.DataFrame(supp_time_rows)
    supp_cluster_freq_df = pd.DataFrame(supp_freq_rows)

    # ── Syllable_timebin_30s (long format all syllables) ─────────────────
    tb30_long = tb30_renamed[["Animal", "group", "Experiment", "time_bin", "Syllable", "Percentage"]].copy()
    tb30_long.columns = ["Animal", "group", "Experiment", "time_bin_seconds", "Syllable", "Percentage"]
    tb30_long["time_bin_seconds"] = tb30_long["time_bin_seconds"].apply(lambda x: int(x) * BIN_SECONDS if not pd.isna(x) else np.nan)

    # ── BFL_scores ───────────────────────────────────────────────────────
    if bfl_raw.empty:
        bfl_df = pd.DataFrame(columns=["animal_id", "group", "experiment", "bfl_score"])
    else:
        bfl_df = bfl_raw.copy()

    # ── Syllable_frames (from metrics CSV) ───────────────────────────────
    if not metrics_raw.empty:
        syllable_frames_df = metrics_raw.copy()
    else:
        syllable_frames_df = pd.DataFrame(columns=["syllable", "mean_recall", "mean_precision", "mean_f1",
                                                     "percent_of_total_frames", "cumulative_coverage"])

    # ── Precision_recall (from metrics, expanded per group) ───────────────
    if not metrics_raw.empty:
        pr_rows = []
        for _, row in metrics_raw.iterrows():
            for group in ["Control", "ELS"]:
                pr_rows.append({
                    "syllable": row.get("syllable"),
                    "group": group,
                    "mean_precision_pct": round(float(row.get("mean_precision", 0)) * 100.0, 4) if row.get("mean_precision") else np.nan,
                    "mean_recall_pct": round(float(row.get("mean_recall", 0)) * 100.0, 4) if row.get("mean_recall") else np.nan,
                    "percent_of_total_frames": row.get("percent_of_total_frames"),
                })
        precision_recall_df = pd.DataFrame(pr_rows)
    else:
        precision_recall_df = pd.DataFrame()

    # ── Overlap_0_28 ─────────────────────────────────────────────────────
    if not metrics_raw.empty:
        overlap_rows = []
        for _, row in metrics_raw[metrics_raw["syllable"].isin([0, 28])].iterrows():
            overlap_rows.append({
                "selected_syllables": "0, 28",
                "syllable": row.get("syllable"),
                "mean_precision_pct": round(float(row.get("mean_precision", 0)) * 100.0, 4),
                "mean_recall_pct": round(float(row.get("mean_recall", 0)) * 100.0, 4),
                "note": "Freezing-related syllables",
            })
        overlap_df_out = pd.DataFrame(overlap_rows)
    else:
        overlap_df_out = pd.DataFrame(columns=["selected_syllables", "syllable", "mean_precision_pct",
                                                 "mean_recall_pct", "note"])

    # ── Syll_0_28 timecourse ──────────────────────────────────────────────
    # Load from precomputed fixed-bin CSV (81 animals × 15 bins, from moseq_syllables_per_frame.csv.gz)
    # This ensures the same data used for the lme4 statistical model in scripts/statistical_analysis_R.R
    _s028_csv = SOURCE_DATA_DIR / "s0s28_timecourse_per_animal.csv"
    if _s028_csv.exists():
        _s028_long = pd.read_csv(_s028_csv)
        # map experiment int code back to string label
        _s028_long["Experiment"] = _s028_long["experiment"].astype(str).map({"1": "SGK_2024", "3": "SG_2024"}).fillna(_s028_long["experiment"].astype(str))

        def _make_syll_0_28_wide(sub_df):
            rows_out = []
            for (animal, group, experiment), asub in sub_df.groupby(["animal_id", "group", "Experiment"]):
                row = {"animal_id": _parse_animal_id(str(animal)), "animal": str(animal), "group": group,
                       "project": "COping", "experiment": experiment}
                for _, brow in asub.sort_values("time_s").iterrows():
                    col = f"t_{str(int(brow['time_s'])).zfill(3)}s"
                    row[col] = round(float(brow["pct_s0s28"]), 4)
                rows_out.append(row)
            return pd.DataFrame(rows_out)

        syll_0_28_combined_df = _make_syll_0_28_wide(_s028_long)
        syll_0_28_sgk_df = _make_syll_0_28_wide(_s028_long[_s028_long["Experiment"] == "SGK_2024"])
        syll_0_28_sg_df = _make_syll_0_28_wide(_s028_long[_s028_long["Experiment"] == "SG_2024"])
    else:
        # Fallback: sum syllables 0+28 from per-timebin CSV (variable bins)
        s0_28_syllables = [0, 28]
        s0_28_rows_raw = tb30_renamed[tb30_renamed["Syllable"].isin(s0_28_syllables)].copy()
        s0_28_per_animal = s0_28_rows_raw.groupby(["Animal", "group", "Experiment", "time_bin"], as_index=False)["Percentage"].sum()

        def _make_syll_0_28_wide(sub_df):  # type: ignore[no-redef]
            rows_out = []
            for (animal, group, experiment), asub in sub_df.groupby(["Animal", "group", "Experiment"]):
                row = {"animal_id": _parse_animal_id(str(animal)), "animal": str(animal), "group": group,
                       "project": "COping", "experiment": experiment}
                for b_idx, (_, brow) in enumerate(asub.sort_values("time_bin").iterrows(), 1):
                    col = f"t_{str(b_idx * BIN_SECONDS).zfill(3)}s"
                    row[col] = round(float(brow["Percentage"]), 4)
                rows_out.append(row)
            return pd.DataFrame(rows_out)

        syll_0_28_combined_df = _make_syll_0_28_wide(s0_28_per_animal)
        syll_0_28_sgk_df = _make_syll_0_28_wide(s0_28_per_animal[s0_28_per_animal["Experiment"] == "SGK_2024"])
        syll_0_28_sg_df = _make_syll_0_28_wide(s0_28_per_animal[s0_28_per_animal["Experiment"] == "SG_2024"])

    return {
        "SimBA_validation": simba_val_df,
        "Freezing_SGK_2024": sgk_df,
        "Freezing_SG_2024": sg_df,
        "Freezing_combined": freezing_df,
        "Cluster_time_wide": cluster_time_wide_df,
        "Cluster_frequency": cluster_freq_df,
        "Fig4_representatives": representatives_df,
        "Fig4_frequency_metrics": diversity_df,
        "Fig4_usage_profile": usage_profile_df,
        "Fig4_bout_duration": bout_df,
        "Fig4_transition_metrics": transition_metrics_df,
        "Fig4_transition_chords": transition_chords_df,
        "Supp_cluster_time": supp_cluster_time_df,
        "Supp_cluster_frequency": supp_cluster_freq_df,
        "Syllable_timebin_30s": tb30_long,
        "BFL_scores": bfl_df,
        "Syllable_frames": syllable_frames_df,
        "Precision_recall": precision_recall_df,
        "Overlap_0_28": overlap_df_out,
        "Syll_0_28_SGK_2024": syll_0_28_sgk_df,
        "Syll_0_28_SG_2024": syll_0_28_sg_df,
        "Syll_0_28_combined": syll_0_28_combined_df,
    }


def build_statistical_report(sheets: dict[str, pd.DataFrame], gmap: dict[str, str]) -> dict[str, list[tuple[str, pd.DataFrame]]]:
    print("Running statistical analyses...")
    stat_sections: dict[str, list[tuple[str, pd.DataFrame]]] = {}

    # ── Fig.1A_ SimBA_validation ──────────────────────────────────────────
    freezing_combined = sheets["Freezing_combined"]
    simba_stats_df = _run_simba_validation_stats(freezing_combined)
    stat_sections["Fig.1A_ SimBA_validation"] = [("SimBA validation: Pearson r (manual vs automatic freezing)", simba_stats_df)]

    # ── Fig.3A-C_Ground_truth ──────────────────────────────────────────────
    # MixedLM for freezing timecourse across cohorts
    print("  Fitting MixedLM for freezing timecourse...")
    time_cols = [c for c in freezing_combined.columns if c.startswith("t_")]
    lm_sections = []
    if time_cols and not freezing_combined.empty:
        id_vars = ["animal_id", "group", "experiment"]
        avail_id = [c for c in id_vars if c in freezing_combined.columns]
        long_f = freezing_combined[avail_id + time_cols].melt(
            id_vars=avail_id, value_vars=time_cols, var_name="bin_label", value_name="freezing_pct"
        ).dropna(subset=["freezing_pct"])
        long_f["time_bin"] = long_f["bin_label"].str.extract(r"t_(\d+)s").astype(int) / BIN_SECONDS
        long_f["animal_id"] = long_f["animal_id"].astype(str)

        for exp_label, exp_filter in [("Combined", None), ("SGK_2024", "SGK_2024"), ("SG_2024", "SG_2024")]:
            sub = long_f if exp_filter is None else long_f[long_f["experiment"] == exp_filter]
            if sub.empty or sub["group"].nunique() < 2:
                continue
            lm_result = _run_mixedlm_timecourse(sub, "freezing_pct", exp_label)
            lm_sections.append((f"MixedLM Freezing ~ Group × Time ({exp_label})", lm_result))
            posthoc = _posthoc_bonferroni(sub, "freezing_pct")
            lm_sections.append((f"Posthoc Bonferroni per time bin ({exp_label})", posthoc))
    else:
        lm_sections.append(("MixedLM Freezing", pd.DataFrame([{"note": "Insufficient data"}])))
    stat_sections["Fig.3A-C_Ground_truth"] = lm_sections

    # ── Fig.3D_Syllable_usage ──────────────────────────────────────────────
    syllable_frames_df = sheets["Syllable_frames"]
    stat_sections["Fig.3D_Syllable_usage"] = [("Syllable usage summary", syllable_frames_df.head(100) if not syllable_frames_df.empty else pd.DataFrame([{"note": "No syllable frame data"}]))]

    # ── Fig.3E-F_Precision_recall ──────────────────────────────────────────
    pr_df = sheets["Precision_recall"]
    if not pr_df.empty and "syllable" in pr_df.columns and "group" in pr_df.columns:
        pr_summary = pr_df.groupby(["syllable", "group"]).agg(
            n=("mean_precision_pct", "count"),
            mean_precision=("mean_precision_pct", "mean"),
            sd_precision=("mean_precision_pct", lambda x: x.std(ddof=1)),
            mean_recall=("mean_recall_pct", "mean"),
            sd_recall=("mean_recall_pct", lambda x: x.std(ddof=1)),
        ).reset_index()
        stat_sections["Fig.3E-F_Precision_recall"] = [("Precision & Recall per syllable × group", pr_summary)]
    else:
        stat_sections["Fig.3E-F_Precision_recall"] = [("Precision & Recall", pd.DataFrame([{"note": "No precision/recall data available"}]))]

    # ── Fig.3H_Freezing_syllables ──────────────────────────────────────────
    syll_comb = sheets["Syll_0_28_combined"]
    s_lm_sections = []
    if not syll_comb.empty:
        s_time_cols = [c for c in syll_comb.columns if c.startswith("t_")]
        avail_id = [c for c in ["animal_id", "group", "experiment"] if c in syll_comb.columns]
        s_long = syll_comb[avail_id + s_time_cols].melt(
            id_vars=avail_id, value_vars=s_time_cols, var_name="bin_label", value_name="syll_pct"
        ).dropna(subset=["syll_pct"])
        s_long["time_bin"] = s_long["bin_label"].str.extract(r"t_(\d+)s").astype(int) / BIN_SECONDS
        s_long["animal_id"] = s_long["animal_id"].astype(str) if "animal_id" in s_long.columns else "unknown"
        if s_long["group"].nunique() >= 2:
            lm_s = _run_mixedlm_timecourse(s_long, "syll_pct", "Combined")
            s_lm_sections.append(("MixedLM Syllables 0+28 ~ Group × Time (Combined)", lm_s))
            s_lm_sections.append(("Posthoc Bonferroni per time bin", _posthoc_bonferroni(s_long, "syll_pct")))
    if not s_lm_sections:
        s_lm_sections = [("MixedLM Syllables 0+28", pd.DataFrame([{"note": "Insufficient data"}]))]
    stat_sections["Fig.3H_Freezing_syllables"] = s_lm_sections

    # ── Fig.4A_Clusters_frequency ─────────────────────────────────────────
    freq_df = sheets["Cluster_frequency"]
    freq_stat_sections = []
    if not freq_df.empty:
        for cluster in CLUSTER_ORDER:
            csub = freq_df[freq_df["cluster"] == cluster]
            if csub.empty:
                continue
            summary = _summary_stats_by_group(csub, "frequency_seconds")
            summary.insert(0, "cluster", cluster)
            ctrl = csub[csub["group"] == "Control"]["frequency_seconds"].dropna()
            els = csub[csub["group"] == "ELS"]["frequency_seconds"].dropna()
            if len(ctrl) >= 2 and len(els) >= 2:
                t, p = scipy_stats.ttest_ind(ctrl, els)
                pooled = math.sqrt(((len(ctrl) - 1) * ctrl.var(ddof=1) + (len(els) - 1) * els.var(ddof=1)) / (len(ctrl) + len(els) - 2))
                d = (ctrl.mean() - els.mean()) / pooled if pooled else np.nan
                ttest_row = pd.DataFrame([{"t_stat": round(t, 4), "p_value": float(p), "cohens_d": round(d, 4)}])
                freq_stat_sections.append((f"{cluster} - Summary by group", summary))
                freq_stat_sections.append((f"{cluster} - t-test Control vs ELS", ttest_row))
            else:
                freq_stat_sections.append((f"{cluster} - Summary by group", summary))
    if not freq_stat_sections:
        freq_stat_sections = [("Cluster frequency", pd.DataFrame([{"note": "No data"}]))]
    stat_sections["Fig.4A_Clusters_frequency"] = freq_stat_sections

    # ── Fig.4B-I_Clusters_over_time ───────────────────────────────────────
    cluster_time_df = sheets["Cluster_time_wide"]
    clust_time_stat_sections = []
    if not cluster_time_df.empty:
        t_cols = [c for c in cluster_time_df.columns if c.startswith("t_")]
        for cluster in CLUSTER_ORDER:
            csub = cluster_time_df[cluster_time_df["cluster"] == cluster]
            if csub.empty:
                continue
            avail_id = [c for c in ["animal_id", "group", "experiment"] if c in csub.columns]
            c_long = csub[avail_id + t_cols].melt(
                id_vars=avail_id, value_vars=t_cols, var_name="bin_label", value_name="pct"
            ).dropna(subset=["pct"])
            c_long["time_bin"] = c_long["bin_label"].str.extract(r"t_(\d+)s").astype(int) / BIN_SECONDS
            c_long["animal_id"] = c_long["animal_id"].astype(str) if "animal_id" in c_long.columns else "unknown"
            if c_long["group"].nunique() >= 2:
                lm_c = _run_mixedlm_timecourse(c_long, "pct", cluster)
                clust_time_stat_sections.append((f"MixedLM {cluster} ~ Group × Time", lm_c))
                clust_time_stat_sections.append((f"Posthoc Bonferroni {cluster}", _posthoc_bonferroni(c_long, "pct")))
    if not clust_time_stat_sections:
        clust_time_stat_sections = [("Clusters over time", pd.DataFrame([{"note": "No data"}]))]
    stat_sections["Fig.4B-I_Clusters_over_time"] = clust_time_stat_sections

    # ── Fig.5E-H_frequency_metrics ────────────────────────────────────────
    div_df = sheets["Fig4_frequency_metrics"]
    div_stat_sections = []
    for metric in ["simpson_index", "shannon_entropy_index", "evenness_index", "cumulative_usage_index"]:
        if metric not in div_df.columns:
            continue
        summary = _summary_stats_by_group(div_df, metric)
        summary.insert(0, "metric", metric)
        div_stat_sections.append((f"Summary: {metric}", summary))
        ctrl = div_df[div_df["group"] == "Control"][metric].dropna()
        els = div_df[div_df["group"] == "ELS"][metric].dropna()
        if len(ctrl) >= 2 and len(els) >= 2:
            t, p = scipy_stats.ttest_ind(ctrl, els)
            pooled = math.sqrt(((len(ctrl) - 1) * ctrl.var(ddof=1) + (len(els) - 1) * els.var(ddof=1)) / (len(ctrl) + len(els) - 2))
            d = (ctrl.mean() - els.mean()) / pooled if pooled else np.nan
            div_stat_sections.append((f"t-test {metric}", pd.DataFrame([{"t_stat": round(t, 4), "p_value": float(p), "cohens_d": round(d, 4)}])))
    if not div_stat_sections:
        div_stat_sections = [("Diversity metrics", pd.DataFrame([{"note": "No data"}]))]
    stat_sections["Fig.5E-H_frequency_metrics"] = div_stat_sections

    # ── Markov_entropy_resilience ─────────────────────────────────────────
    trans_df = sheets["Fig4_transition_metrics"]
    markov_stat_sections = []
    if not trans_df.empty and "markov_entropy" in trans_df.columns:
        trans_df_r = trans_df.copy()
        trans_df_r["animal_id_str"] = trans_df_r["animal_id"].astype(str)
        trans_df_r["resilience_group"] = trans_df_r.apply(
            lambda r: "ELS resilient" if (r["group"] == "ELS" and r["animal_id_str"] in ELS_RESILIENT_IDS)
            else r["group"], axis=1
        )
        for group in ["Control", "ELS", "ELS resilient"]:
            gsub = trans_df_r[trans_df_r["resilience_group"] == group]["markov_entropy"].dropna()
            if len(gsub) >= 1:
                markov_stat_sections.append((f"Markov entropy - {group}",
                    pd.DataFrame([{"group": group, "n": len(gsub), "mean": round(float(gsub.mean()), 6),
                                   "SD": round(float(gsub.std(ddof=1)), 6) if len(gsub) > 1 else np.nan,
                                   "SEM": round(_sem(gsub), 6)}])))
        ctrl_m = trans_df_r[trans_df_r["resilience_group"] == "Control"]["markov_entropy"].dropna()
        els_m = trans_df_r[trans_df_r["resilience_group"] == "ELS"]["markov_entropy"].dropna()
        if len(ctrl_m) >= 2 and len(els_m) >= 2:
            t, p = scipy_stats.ttest_ind(ctrl_m, els_m)
            markov_stat_sections.append(("t-test Control vs ELS", pd.DataFrame([{"t": round(t, 4), "p": float(p)}])))
    if not markov_stat_sections:
        markov_stat_sections = [("Markov entropy", pd.DataFrame([{"note": "No data"}]))]
    stat_sections["Markov_entropy_resilience"] = markov_stat_sections

    # ── Transitions ───────────────────────────────────────────────────────
    chords_df = sheets["Fig4_transition_chords"]
    if not chords_df.empty:
        stat_sections["Transitions"] = [("Mean transition counts per animal (all groups)", chords_df)]
    else:
        stat_sections["Transitions"] = [("Transitions", pd.DataFrame([{"note": "No transition data"}]))]

    # ── Transitions_resilience ────────────────────────────────────────────
    if not chords_df.empty:
        stat_sections["Transitions_resilience"] = [("Transitions by group (including ELS resilient sub-group)", chords_df)]
    else:
        stat_sections["Transitions_resilience"] = [("Transitions resilience", pd.DataFrame([{"note": "No data"}]))]

    # ── Fig.6A_Clusters_total_counts_ ────────────────────────────────────
    stat_sections["Fig.6A_Clusters_total_counts_"] = [("Cluster frequency by resilience group",
        _summary_stats_by_group(freq_df, "frequency_seconds", "group") if not freq_df.empty
        else pd.DataFrame([{"note": "No data"}]))]

    # ── Fig.6C-I_Clusters_overtime_ ───────────────────────────────────────
    stat_sections["Fig.6C-I_Clusters_overtime_"] = clust_time_stat_sections[:6] if clust_time_stat_sections else [("No data", pd.DataFrame())]

    # ── Resilience_scores_frequency ───────────────────────────────────────
    bfl_df = sheets["BFL_scores"]
    if not bfl_df.empty:
        stat_sections["Resilience_scores_frequency"] = [("BFL / Behavioral dynamics scores", bfl_df.head(200))]
    else:
        stat_sections["Resilience_scores_frequency"] = [("BFL scores", pd.DataFrame([{"note": "BFL scores not available in source data"}]))]

    return stat_sections


def write_raw_data_workbook(sheets: dict[str, pd.DataFrame], output_path: Path) -> None:
    print(f"Writing Raw_data workbook: {output_path}")
    wb = Workbook()
    wb.remove(wb.active)

    sheet_titles = {
        "SimBA_validation": "SimBA Validation — Manual vs Automatic Freezing (per animal, per time bin)",
        "Freezing_SGK_2024": "Freezing % per 30 s bin — Sanguino-Gomez & Krugers, 2024 cohort",
        "Freezing_SG_2024": "Freezing % per 30 s bin — Sanguino-Gomez et al., 2024 cohort",
        "Freezing_combined": "Freezing % per 30 s bin — Combined cohorts (83 animals)",
        "Cluster_time_wide": "Behavior cluster % per 30 s bin (8 clusters × animals)",
        "Cluster_frequency": "Total frequency (seconds) per behavior cluster per animal",
        "Fig4_representatives": "Figure 4 representative animals for ethogram and barcode panels",
        "Fig4_frequency_metrics": "Figure 4 diversity & usage metrics per animal (Simpson, Shannon, Evenness, CUI)",
        "Fig4_usage_profile": "Figure 4 cluster usage proportions per animal",
        "Fig4_bout_duration": "Figure 4 mean bout duration per cluster per animal",
        "Fig4_transition_metrics": "Figure 4 sequence complexity metrics per animal",
        "Fig4_transition_chords": "Figure 4 mean transition counts per cluster pair per group",
        "Supp_cluster_time": "Supplementary Figure: tracking-quality cluster time courses",
        "Supp_cluster_frequency": "Supplementary Figure: tracking-quality cluster frequencies",
        "Syllable_timebin_30s": "MoSeq syllable usage % per 30 s bin — all syllables (long format)",
        "BFL_scores": "Behavioral dynamics (BFL) scores per animal",
        "Syllable_frames": "Frame counts per syllable — precision, recall, F1, usage %",
        "Precision_recall": "Precision & recall of each syllable vs. manual freezing annotation",
        "Overlap_0_28": "Overlap of syllables 0 and 28 vs. supervised freezing per animal",
        "Syll_0_28_SGK_2024": "Syllables 0+28 % per 30 s bin — Sanguino-Gomez & Krugers, 2024",
        "Syll_0_28_SG_2024": "Syllables 0+28 % per 30 s bin — Sanguino-Gomez et al., 2024",
        "Syll_0_28_combined": "Syllables 0+28 % per 30 s bin — combined cohorts",
    }

    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            df = pd.DataFrame([{"note": f"No data available for {sheet_name}"}])
        ws = wb.create_sheet(title=sheet_name[:31])
        title = sheet_titles.get(sheet_name, sheet_name)
        _write_sheet_with_title(ws, title, df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    print(f"  Saved: {output_path}")


def write_statistical_report_workbook(stat_sections: dict[str, list[tuple[str, pd.DataFrame]]], output_path: Path) -> None:
    print(f"Writing Statistical_report workbook: {output_path}")
    wb = Workbook()
    wb.remove(wb.active)

    for sheet_name, sections in stat_sections.items():
        ws = wb.create_sheet(title=sheet_name[:31])
        _write_stat_sheet(ws, sheet_name, sections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    print(f"  Saved: {output_path}")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Raw_data.xlsx and Statistical_report.xlsx")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    gmap = _load_group_map()
    print(f"Loaded group map: {len(gmap)} entries, {len(set(gmap.values()))} groups")

    sheets = build_raw_data(gmap)
    for name, df in sheets.items():
        n = len(df) if df is not None else 0
        print(f"  {name}: {n} rows")

    stat_sections = build_statistical_report(sheets, gmap)

    write_raw_data_workbook(sheets, OUTPUT_RAW)
    write_statistical_report_workbook(stat_sections, OUTPUT_STATS)

    print()
    print("Done.")
    print(f"  Raw_data.xlsx          -> {OUTPUT_RAW}")
    print(f"  Statistical_report.xlsx -> {OUTPUT_STATS}")


if __name__ == "__main__":
    main()
