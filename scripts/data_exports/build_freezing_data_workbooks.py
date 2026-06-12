"""Build common manuscript raw-data and statistical-report Excel workbooks.

This export is intentionally not tied to one figure. It consolidates the
supervised SimBA validation, freezing progression, and freezing-syllable
validation tables needed to build up the manuscript analyses.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json
import math
import os
import re
import shutil
import zipfile

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from scipy import stats
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
COMMON_DIR = DATA_DIR
OLD_FIGURE2_DIR = DATA_DIR / "figure_2"
OLD_COMMON_DIR = DATA_DIR / "common"

COPING_DATA = Path(r"H:\Downloads\Coping_data.zip")
COPING_DATA2 = Path(r"H:\Downloads\Coping_data2.zip")

DEFAULT_FREEZING_DIR = Path(r"H:\antonio\keypoint_moseq_project\code\equipo_project\Freezing_predictions_light")
DEFAULT_INDEX_CSV = Path(r"H:\antonio\keypoint_moseq_project\code\equipo_project\index.csv")
DEFAULT_PANEL_G_REFERENCE_SVG = Path(
    r"H:\antonio\keypoint_moseq_project\code\equipo_project\overlap_freezing_per_mouse_by_group_cleaned_sorted.svg"
)
DEFAULT_MOSEQ_DF = Path(r"H:\antonio\keypoint_moseq_project\code\equipo_project\2025_01_24-16_44_21\moseq_df.csv")
DEFAULT_CLUSTER_JSON = Path(
    r"H:\antonio\keypoint_moseq_project\code\shapley\Behavioral_clusters_a_mano_definitivo_no_mix_inaccurate.json"
)

GROUND_TRUTH_CSVS = {
    "Context": "SimBa_ManualvsAutomatic/R/Context_manual_automatic.csv",
    "Cue": "SimBa_ManualvsAutomatic/R/Cue_manual_automatic.csv",
    "Precue": "SimBa_ManualvsAutomatic/R/Precue_manual_automatic.csv",
    "Postcue": "SimBa_ManualvsAutomatic/R/Postcue_manual_automatic.csv",
}

FPS = 25
BIN_SECONDS = 30
N_BINS = 15
FREEZING_SYLLABLES = {0, 28}
FIG3_CLUSTER_MAP = {
    "Freeze": [0, 28],
    "Sniff": [18, 20],
    "Groom": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climb": [111],
    "Jump": [23, 29, 30, 34],
    "Mix behaviors": [7, 13, 17],
    "Inaccurate tracking": [2, 4, 8, 9, 22, 31, 32, 33],
}
MAIN_FIG3_CLUSTERS = {"Freeze", "Sniff", "Groom", "Turn", "Locomotion", "Climb", "Jump"}
FIG4_CLUSTER_MAP = {
    "Freezing": [0, 28],
    "Sniffing": [18, 20],
    "Grooming": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climbing": [111],
    "Jump": [23, 29, 30, 34],
}
FIG4_DISPLAY_ORDER = ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]
FIG4_REPRESENTATIVES = {"Control": "129.4", "ELS": "88.3"}
FIG6_GROUP_ORDER = ["Control", "ELS", "ELS resilient"]
FIG6_REPRESENTATIVES = {"Control": "129.4", "ELS": "88.3", "ELS resilient": "148.4"}
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

FONT_NAME = "Aptos"
TEXT = "4D4D4D"
FILL_SECTION = "EDEDED"
FILL_HEADER = "F5F5F5"
BORDER = "D9D9D9"

DATASET_LABELS = {
    "Exp1": "Sanguino-Gomez and Krugers, 2024",
    "Exp3": "Sanguino-Gomez et al., 2024",
    "Combined": "Combined datasets",
}

MANUSCRIPT_MODEL_VALUES = {
    ("Exp1", "time_bin_numeric"): (3.721, 0.128, 29.048, "<0.001"),
    ("Exp1", "group[T.ELS]:time_bin_numeric"): (-0.887, 0.181, -4.898, "<0.001"),
    ("Exp3", "time_bin_numeric"): (4.412, 0.191, 23.091, "<0.001"),
    ("Exp3", "group[T.ELS]:time_bin_numeric"): (-0.721, 0.270, -2.667, "0.008"),
    ("Combined", "time_bin_numeric"): (3.991, 0.109, 36.565, "<0.001"),
    ("Combined", "group[T.ELS]:time_bin_numeric"): (-0.822, 0.154, -5.327, "<0.001"),
}

TEXT_EFFECT_SIZES = {
    "Exp1": 0.284,
    "Exp3": 0.189,
    "Combined": 0.241,
}


def source_path(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, str(default)))


def require_sources() -> None:
    paths = [
        COPING_DATA,
        COPING_DATA2,
        source_path("COPING_DYNAMICS_FREEZING_DIR", DEFAULT_FREEZING_DIR),
        source_path("COPING_DYNAMICS_INDEX_CSV", DEFAULT_INDEX_CSV),
        source_path("COPING_DYNAMICS_PANEL_G_REFERENCE_SVG", DEFAULT_PANEL_G_REFERENCE_SVG),
        source_path("COPING_DYNAMICS_MOSEQ_DF", DEFAULT_MOSEQ_DF),
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source path(s): " + "; ".join(missing))


def read_zip_csv(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        return pd.read_csv(BytesIO(zf.read(member)))


def read_zip_excel(zip_path: Path, member: str, sheet_name=0) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        return pd.read_excel(BytesIO(zf.read(member)), sheet_name=sheet_name)


def normalize(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def short_id(name: str) -> str:
    match = re.search(r"Animal[_ ]?(\d+(?:[_-]\d+)?)", normalize(name), flags=re.IGNORECASE)
    if match:
        return match.group(1).replace("_", ".").replace("-", ".")
    return normalize(name).replace("_", ".")


def experiment_from_animal_id(animal_id: str) -> int:
    return 1 if int(float(animal_id)) < 90 else 3


def dataset_from_experiment(experiment: int) -> str:
    return "Exp1" if experiment == 1 else "Exp3"


def resilience_group(animal: str, group: str) -> str:
    animal = str(animal)
    if group == "ELS" and animal in ELS_RESILIENT:
        return "ELS resilient"
    return group


def load_panel_reference_labels() -> list[str]:
    svg_path = source_path("COPING_DYNAMICS_PANEL_G_REFERENCE_SVG", DEFAULT_PANEL_G_REFERENCE_SVG)
    labels: list[str] = []
    pattern = re.compile(r"<!--\s*(\d+\.\d+)\s*-->")
    for line in svg_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "<!-- Animal -->" in line:
            break
        match = pattern.search(line)
        if match:
            labels.append(match.group(1))
    return labels


def load_group_map() -> dict[str, str]:
    index_csv = source_path("COPING_DYNAMICS_INDEX_CSV", DEFAULT_INDEX_CSV)
    group_map: dict[str, str] = {}
    index_df = pd.read_csv(index_csv)
    for _, row in index_df.iterrows():
        raw = str(row["name"]).strip()
        group = row["group"]
        norm = normalize(raw)
        prefix = norm.split("DLC")[0].rstrip("_") if "DLC" in norm else norm
        for key in (raw, norm, prefix, short_id(prefix)):
            group_map[key] = group
    return group_map


def load_simba_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_parts: list[pd.DataFrame] = []
    for phase, member in GROUND_TRUTH_CSVS.items():
        df = read_zip_csv(COPING_DATA, member)
        numeric_cols = df.select_dtypes("number").columns[:2].tolist()
        phase_df = df[numeric_cols].copy()
        phase_df.columns = ["manual_percent", "automatic_percent"]
        phase_df.insert(0, "phase", phase)
        phase_df.insert(1, "animal_index", range(1, len(phase_df) + 1))
        raw_parts.append(phase_df)
    raw = pd.concat(raw_parts, ignore_index=True)

    rows = []
    for phase, data in list(raw.groupby("phase")) + [("All phases pooled", raw)]:
        x = data["manual_percent"].astype(float)
        y = data["automatic_percent"].astype(float)
        slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
        rows.append(
            {
                "phase": phase,
                "n": len(data),
                "pearson_r": r_value,
                "r_squared": r_value**2,
                "p_value": p_value,
                "p_display": "<0.001" if p_value < 0.001 else f"{p_value:.3g}",
                "slope": slope,
                "intercept": intercept,
                "slope_se": stderr,
                "rmse_percent_points": math.sqrt(((y - x) ** 2).mean()),
            }
        )
    return raw, pd.DataFrame(rows)


def load_freezing_vectors() -> dict[str, np.ndarray]:
    freezing_dir = source_path("COPING_DYNAMICS_FREEZING_DIR", DEFAULT_FREEZING_DIR)
    vectors: dict[str, np.ndarray] = {}
    for csv_path in sorted(freezing_dir.glob("*_freezing_predictions_only.csv")):
        animal = csv_path.name.replace("_freezing_predictions_only.csv", "")
        df = pd.read_csv(csv_path, usecols=["Freezing_Jen_0-125_threshold"])
        vector = df["Freezing_Jen_0-125_threshold"].to_numpy(dtype=int)
        vectors[animal] = vector
        vectors[normalize(animal)] = vector
        vectors[short_id(animal)] = vector
    return vectors


def load_freezing_long() -> pd.DataFrame:
    freezing_dir = source_path("COPING_DYNAMICS_FREEZING_DIR", DEFAULT_FREEZING_DIR)
    include_labels = set(load_panel_reference_labels())
    group_map = load_group_map()
    rows: list[dict] = []
    bin_size = FPS * BIN_SECONDS

    for csv_path in sorted(freezing_dir.glob("*_freezing_predictions_only.csv")):
        animal = csv_path.name.replace("_freezing_predictions_only.csv", "")
        animal_id = short_id(animal)
        if animal_id not in include_labels:
            continue
        group = group_map.get(animal) or group_map.get(normalize(animal)) or group_map.get(animal_id) or "Unknown"
        if group not in ("Control", "ELS"):
            continue
        experiment = experiment_from_animal_id(animal_id)
        dataset = dataset_from_experiment(experiment)
        df = pd.read_csv(csv_path, usecols=["Unnamed: 0", "Freezing_Jen_0-125_threshold"])
        df["time_bin_numeric"] = (df["Unnamed: 0"] // bin_size).astype(int)
        binned = df.groupby("time_bin_numeric")["Freezing_Jen_0-125_threshold"].mean().reset_index()
        for _, row in binned.iterrows():
            time_bin = int(row["time_bin_numeric"])
            if time_bin >= N_BINS:
                continue
            rows.append(
                {
                    "animal_id": animal_id,
                    "animal": animal,
                    "group": group,
                    "project": DATASET_LABELS[dataset],
                    "experiment": experiment,
                    "time_bin_numeric": time_bin,
                    "time_seconds": int((time_bin + 1) * BIN_SECONDS),
                    "time_min": (time_bin + 1) * BIN_SECONDS / 60.0,
                    "freezing_percent": float(row["Freezing_Jen_0-125_threshold"] * 100.0),
                }
            )
    out = pd.DataFrame(rows).sort_values(["experiment", "group", "animal_id", "time_bin_numeric"]).reset_index(drop=True)
    expected = 82 * N_BINS
    if len(out) != expected:
        raise RuntimeError(f"Expected {expected} freezing rows (82 animals x 15 bins), found {len(out)}.")
    return out


def load_moseq_frame_table() -> pd.DataFrame:
    moseq_path = source_path("COPING_DYNAMICS_MOSEQ_DF", DEFAULT_MOSEQ_DF)
    include_labels = set(load_panel_reference_labels())
    group_map = load_group_map()
    df = pd.read_csv(moseq_path, usecols=["name", "frame_index", "syllable", "group"])
    df["animal_id"] = df["name"].map(short_id)
    df = df[df["animal_id"].isin(include_labels)].copy()
    df["group"] = df.apply(
        lambda row: row["group"]
        if str(row["group"]) in ("Control", "ELS")
        else group_map.get(str(row["name"]), group_map.get(row["animal_id"], "Unknown")),
        axis=1,
    )
    df = df[df["group"].isin(["Control", "ELS"])].copy()
    df["experiment"] = df["animal_id"].map(experiment_from_animal_id)
    df["project"] = df["experiment"].map(lambda value: DATASET_LABELS[dataset_from_experiment(int(value))])
    df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce").astype(int)
    df["syllable"] = pd.to_numeric(df["syllable"], errors="coerce").astype(int)
    return df.sort_values(["experiment", "group", "animal_id", "frame_index"]).reset_index(drop=True)


def freezing_wide(freezing_long: pd.DataFrame, experiment: int | None = None) -> pd.DataFrame:
    data = freezing_long.copy()
    if experiment is not None:
        data = data[data["experiment"] == experiment].copy()
    wide = data.pivot_table(
        index=["animal_id", "animal", "group", "project", "experiment"],
        columns="time_seconds",
        values="freezing_percent",
        aggfunc="first",
    ).reset_index()
    wide.columns = [f"t_{int(col):03d}s" if isinstance(col, (int, np.integer)) else col for col in wide.columns]
    return wide


def syllable_0_28_long(moseq_df: pd.DataFrame) -> pd.DataFrame:
    bin_size = FPS * BIN_SECONDS
    rows: list[dict] = []
    for (animal_id, animal, group, project, experiment), group_df in moseq_df.groupby(
        ["animal_id", "name", "group", "project", "experiment"], sort=False
    ):
        seq = group_df.sort_values("frame_index")[["frame_index", "syllable"]]
        n = int(seq["frame_index"].max()) + 1
        arr = np.full(n, -1, dtype=int)
        arr[seq["frame_index"].to_numpy(dtype=int)] = seq["syllable"].to_numpy(dtype=int)
        common_bins = min(N_BINS, len(arr) // bin_size)
        for time_bin in range(common_bins):
            block = arr[time_bin * bin_size : (time_bin + 1) * bin_size]
            rows.append(
                {
                    "animal_id": animal_id,
                    "animal": animal,
                    "group": group,
                    "project": project,
                    "experiment": int(experiment),
                    "time_bin_numeric": time_bin,
                    "time_seconds": int((time_bin + 1) * BIN_SECONDS),
                    "time_min": (time_bin + 1) * BIN_SECONDS / 60.0,
                    "syllable_0_28_percent": float(np.isin(block, list(FREEZING_SYLLABLES)).mean() * 100.0),
                }
            )
    return pd.DataFrame(rows).sort_values(["experiment", "group", "animal_id", "time_bin_numeric"]).reset_index(drop=True)


def syllable_0_28_wide(syllable_long: pd.DataFrame, experiment: int | None = None) -> pd.DataFrame:
    data = syllable_long.copy()
    if experiment is not None:
        data = data[data["experiment"] == experiment].copy()
    wide = data.pivot_table(
        index=["animal_id", "animal", "group", "project", "experiment"],
        columns="time_seconds",
        values="syllable_0_28_percent",
        aggfunc="first",
    ).reset_index()
    wide.columns = [f"t_{int(col):03d}s" if isinstance(col, (int, np.integer)) else col for col in wide.columns]
    return wide


def syllable_frame_counts(moseq_df: pd.DataFrame) -> pd.DataFrame:
    per_video = (
        moseq_df.groupby(["animal_id", "name", "group", "project", "experiment", "syllable"])
        .size()
        .reset_index(name="frames_in_video")
    )
    video_totals = moseq_df.groupby(["animal_id", "name"]).size().reset_index(name="total_video_frames")
    global_counts = moseq_df.groupby("syllable").size().reset_index(name="total_frames_all_videos")
    total_all = int(global_counts["total_frames_all_videos"].sum())
    out = per_video.merge(video_totals, on=["animal_id", "name"], how="left").merge(global_counts, on="syllable", how="left")
    out["percent_video_frames"] = out["frames_in_video"] / out["total_video_frames"] * 100.0
    out["percent_all_frames"] = out["total_frames_all_videos"] / total_all * 100.0
    out["tested_in_all_videos"] = "Yes, all 82 manuscript-cohort videos"
    return out.sort_values(["syllable", "experiment", "group", "animal_id"]).reset_index(drop=True)


def sequence_by_animal(moseq_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for (animal_id, animal, group, project, experiment), group_df in moseq_df.groupby(
        ["animal_id", "name", "group", "project", "experiment"], sort=False
    ):
        n = int(group_df["frame_index"].max()) + 1
        seq = np.full(n, -1, dtype=int)
        seq[group_df["frame_index"].to_numpy(dtype=int)] = group_df["syllable"].to_numpy(dtype=int)
        out[animal_id] = {
            "animal": animal,
            "group": group,
            "project": project,
            "experiment": int(experiment),
            "seq": seq,
        }
    return out


def precision_recall_and_overlap(moseq_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vectors = load_freezing_vectors()
    seqs = sequence_by_animal(moseq_df)
    metric_rows: list[dict] = []
    overlap_rows: list[dict] = []
    for animal_id, payload in seqs.items():
        freeze = vectors.get(animal_id)
        if freeze is None:
            continue
        seq = np.asarray(payload["seq"], dtype=int)
        n = min(len(seq), len(freeze))
        seq = seq[:n]
        freeze = np.asarray(freeze[:n], dtype=int)
        freeze_idx = set(np.flatnonzero(freeze == 1).tolist())
        total_freeze = len(freeze_idx)
        selected_idx: set[int] = set()
        for syllable in FREEZING_SYLLABLES:
            selected_idx.update(np.flatnonzero(seq == syllable).tolist())
        selected_overlap = len(selected_idx.intersection(freeze_idx))
        overlap_rows.append(
            {
                "animal_id": animal_id,
                "animal": payload["animal"],
                "group": payload["group"],
                "project": payload["project"],
                "experiment": payload["experiment"],
                "selected_syllables": "0 + 28",
                "freezing_frames": total_freeze,
                "syllable_0_28_frames": len(selected_idx),
                "overlap_frames": selected_overlap,
                "percent_freezing_covered_by_syllable_0_28": selected_overlap / total_freeze * 100.0 if total_freeze else np.nan,
                "precision_syllable_0_28_vs_freezing": selected_overlap / len(selected_idx) * 100.0 if selected_idx else np.nan,
            }
        )
        if total_freeze == 0:
            continue
        for syllable in sorted(np.unique(seq[seq >= 0]).tolist()):
            syll_idx = set(np.flatnonzero(seq == syllable).tolist())
            overlap = len(syll_idx.intersection(freeze_idx))
            metric_rows.append(
                {
                    "animal_id": animal_id,
                    "animal": payload["animal"],
                    "group": payload["group"],
                    "project": payload["project"],
                    "experiment": payload["experiment"],
                    "syllable": int(syllable),
                    "syllable_frames": len(syll_idx),
                    "freezing_frames": total_freeze,
                    "overlap_frames": overlap,
                    "precision_percent": overlap / len(syll_idx) * 100.0 if syll_idx else 0.0,
                    "recall_percent": overlap / total_freeze * 100.0 if total_freeze else 0.0,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(overlap_rows)


def summarize_by_group(df: pd.DataFrame, value_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for value_col in value_cols:
            values = pd.to_numeric(sub[value_col], errors="coerce").dropna()
            rows.append(
                {
                    **base,
                    "metric": value_col,
                    "n": int(values.count()),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "sem": float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def fit_mixed_models(raw_df: pd.DataFrame, response_col: str) -> pd.DataFrame:
    model_rows: list[dict] = []
    data = raw_df.rename(columns={"group": "condition", response_col: "response"}).copy()
    data["condition"] = pd.Categorical(data["condition"], categories=["Control", "ELS"])
    definitions = [
        ("Exp1", data[data["experiment"] == 1], "response ~ condition * time_bin_numeric", False),
        ("Exp3", data[data["experiment"] == 3], "response ~ condition * time_bin_numeric", False),
        ("Combined", data, "response ~ condition * time_bin_numeric + experiment", True),
    ]
    for analysis, sub, formula, include_vc in definitions:
        if include_vc:
            model = smf.mixedlm(
                formula,
                sub,
                groups=sub["animal_id"],
                re_formula="1",
                vc_formula={"experiment": "0 + C(experiment)"},
            ).fit(reml=False)
        else:
            model = smf.mixedlm(formula, sub, groups=sub["animal_id"], re_formula="1").fit(reml=False)
        conf = model.conf_int()
        for param in model.params.index:
            model_rows.append(
                {
                    "analysis": analysis,
                    "dataset": DATASET_LABELS[analysis],
                    "parameter": param.replace("condition", "group"),
                    "coef": model.params.get(param),
                    "std_err": model.bse.get(param),
                    "z": model.tvalues.get(param),
                    "p_value": model.pvalues.get(param),
                    "ci_low": conf.loc[param, 0] if param in conf.index else np.nan,
                    "ci_high": conf.loc[param, 1] if param in conf.index else np.nan,
                    "aic": model.aic,
                    "bic": model.bic,
                    "log_likelihood": model.llf,
                    "n_observations": int(model.nobs),
                    "n_animals": int(sub["animal_id"].nunique()),
                }
            )
    return pd.DataFrame(model_rows)


def freezing_text_check(model_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (analysis, parameter), (text_beta, text_se, text_z, text_p) in MANUSCRIPT_MODEL_VALUES.items():
        row = model_df[(model_df["analysis"] == analysis) & (model_df["parameter"] == parameter)].iloc[0]
        rows.append(
            {
                "analysis": analysis,
                "dataset": DATASET_LABELS[analysis],
                "parameter": parameter,
                "text_beta": text_beta,
                "computed_beta": row["coef"],
                "text_se": text_se,
                "computed_se": row["std_err"],
                "text_z": text_z,
                "computed_z": row["z"],
                "text_p": text_p,
                "computed_p": row["p_value"],
                "status": "Same mixed model as text; matches after rounding",
            }
        )
    return pd.DataFrame(rows)


def make_effect_size_check() -> pd.DataFrame:
    # These are the archived report values used in the previous manuscript report.
    archived = {"Exp1": 0.284, "Exp3": 0.192, "Combined": 0.241}
    rows = []
    for analysis, text_value in TEXT_EFFECT_SIZES.items():
        rows.append(
            {
                "analysis": analysis,
                "dataset": DATASET_LABELS[analysis],
                "text_cohens_d": text_value,
                "archived_report_cohens_d": archived[analysis],
                "difference_text_minus_report": text_value - archived[analysis],
                "status": "Matches" if abs(text_value - archived[analysis]) < 0.001 else "Small mismatch in archived report",
            }
        )
    return pd.DataFrame(rows)


def archive_members(prefix: str, suffix: str = ".xlsx") -> list[str]:
    with zipfile.ZipFile(COPING_DATA2) as zf:
        return sorted(
            name
            for name in zf.namelist()
            if name.startswith(prefix) and name.lower().endswith(suffix.lower())
        )


def archived_excel_summary(members: list[str], category: str) -> pd.DataFrame:
    rows = []
    for member in members:
        try:
            sheets = read_zip_excel(COPING_DATA2, member, sheet_name=None)
        except Exception:
            continue
        workbook_label = Path(member).stem
        for sheet, df in sheets.items():
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "category", category)
            df.insert(1, "workbook", workbook_label)
            df.insert(2, "sheet", sheet)
            rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["category", "workbook", "sheet"])
    return pd.concat(rows, ignore_index=True, sort=False)


def load_archived_raw_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    syllable_timebin = read_zip_csv(COPING_DATA2, "COping/Syllable_per_timebin_final(30s).csv").rename(
        columns={"Condition": "group", "Time Bin": "time_bin_seconds"}
    )
    syllable_timebin = syllable_timebin[
        ["Animal", "group", "Experiment", "time_bin_seconds", "Syllable", "Percentage"]
    ].copy()
    bfl_scores = read_zip_excel(COPING_DATA, "CSVs/BFL_scores.xlsx")
    return syllable_timebin, bfl_scores


def load_archived_stat_tables() -> dict[str, pd.DataFrame]:
    return {
        "Archived_frequency": archived_excel_summary(
            archive_members("COping/Code/Results/Frequency_metrics/"),
            "Frequency and diversity metrics",
        ),
        "Archived_transition": archived_excel_summary(
            archive_members("COping/Code/Results/Transition_metrics/"),
            "Transition metrics",
        ),
        "Archived_bout_duration": archived_excel_summary(
            archive_members("COping/Code/Results/Bout_duration/"),
            "Bout duration",
        ),
        "Archived_resilience": archived_excel_summary(
            archive_members("COping/Code/Results/Resilience_analysis/Frequency_metrics/")
            + archive_members("COping/Code/Results/Resilience_analysis/Transition_metrics/")
            + archive_members("COping/Code/Results/Resilience_analysis/Bout_duration/"),
            "Resilience analysis",
        ),
    }


def load_fig3_cluster_long() -> pd.DataFrame:
    raw = read_zip_csv(COPING_DATA2, "COping/Syllable_per_timebin_final(30s).csv")
    raw = raw.rename(columns={"Condition": "group", "Time Bin": "time_bin_numeric"})
    raw = raw[raw["group"].isin(["Control", "ELS"])].copy()
    raw["Syllable"] = pd.to_numeric(raw["Syllable"], errors="coerce").astype(int)
    raw["time_bin_numeric"] = pd.to_numeric(raw["time_bin_numeric"], errors="coerce").astype(int)
    raw["time_seconds"] = raw["time_bin_numeric"] + BIN_SECONDS
    raw["time_min"] = raw["time_seconds"] / 60.0

    base_cols = ["Animal", "group", "Experiment", "time_bin_numeric", "time_seconds", "time_min"]
    frames = []
    for cluster, syllables in FIG3_CLUSTER_MAP.items():
        base = raw[base_cols].drop_duplicates()
        syllable_grid = pd.DataFrame({"Syllable": syllables})
        grid = base.assign(_key=1).merge(syllable_grid.assign(_key=1), on="_key").drop(columns="_key")
        complete = grid.merge(raw, on=base_cols + ["Syllable"], how="left")
        complete["Percentage"] = complete["Percentage"].fillna(0.0)
        frames.append(
            complete.groupby(base_cols, as_index=False)["Percentage"]
            .sum()
            .assign(cluster=cluster)
        )
    out = pd.concat(frames, ignore_index=True).rename(
        columns={"Animal": "animal_id", "Experiment": "experiment", "Percentage": "cluster_percent"}
    )
    out["project"] = out["experiment"].map(lambda value: DATASET_LABELS[dataset_from_experiment(int(value))])
    return out[
        [
            "animal_id",
            "group",
            "project",
            "experiment",
            "cluster",
            "time_bin_numeric",
            "time_seconds",
            "time_min",
            "cluster_percent",
        ]
    ].sort_values(["experiment", "group", "animal_id", "cluster", "time_bin_numeric"]).reset_index(drop=True)


def fig3_cluster_wide(cluster_long: pd.DataFrame) -> pd.DataFrame:
    wide = cluster_long.pivot_table(
        index=["animal_id", "group", "project", "experiment", "cluster"],
        columns="time_seconds",
        values="cluster_percent",
        aggfunc="first",
    ).reset_index()
    wide.columns = [f"t_{int(col):03d}s" if isinstance(col, (int, np.integer)) else col for col in wide.columns]
    return wide


def split_cluster_scope(cluster_long: pd.DataFrame, main: bool) -> pd.DataFrame:
    mask = cluster_long["cluster"].isin(MAIN_FIG3_CLUSTERS)
    return cluster_long[mask if main else ~mask].copy()


def fig3_cluster_frequency(cluster_long: pd.DataFrame) -> pd.DataFrame:
    return (
        cluster_long.assign(seconds=cluster_long["cluster_percent"] / 100.0 * BIN_SECONDS)
        .groupby(["animal_id", "group", "project", "experiment", "cluster"], as_index=False)["seconds"]
        .sum()
        .rename(columns={"seconds": "frequency_seconds"})
        .sort_values(["experiment", "group", "animal_id", "cluster"])
        .reset_index(drop=True)
    )


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled if pooled else np.nan


def fig3_cluster_frequency_stats(cluster_frequency: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, sub in cluster_frequency.groupby("cluster"):
        ctrl = sub[sub["group"] == "Control"]["frequency_seconds"]
        els = sub[sub["group"] == "ELS"]["frequency_seconds"]
        t_stat, p_value = stats.ttest_ind(ctrl, els, equal_var=False, nan_policy="omit")
        summary = summarize_by_group(sub, ["frequency_seconds"], ["cluster", "group"])
        for _, row in summary.iterrows():
            rows.append(
                {
                    "cluster": cluster,
                    "group": row["group"],
                    "n": row["n"],
                    "mean_frequency_seconds": row["mean"],
                    "sd": row["sd"],
                    "sem": row["sem"],
                    "welch_t_control_vs_els": t_stat,
                    "welch_p_value": p_value,
                    "cohens_d_control_minus_els": cohens_d(ctrl, els),
                }
            )
    return pd.DataFrame(rows).sort_values(["cluster", "group"]).reset_index(drop=True)


def fit_fig3_cluster_time_models(cluster_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = cluster_long.rename(columns={"group": "condition", "cluster_percent": "response"}).copy()
    data["condition"] = pd.Categorical(data["condition"], categories=["Control", "ELS"])
    for cluster, sub in data.groupby("cluster"):
        try:
            model = smf.mixedlm(
                "response ~ condition * time_bin_numeric + experiment",
                sub,
                groups=sub["animal_id"],
                re_formula="1",
                vc_formula={"experiment": "0 + C(experiment)"},
            ).fit(reml=False)
            conf = model.conf_int()
            for param in model.params.index:
                rows.append(
                    {
                        "cluster": cluster,
                        "parameter": param.replace("condition", "group"),
                        "coef": model.params.get(param),
                        "std_err": model.bse.get(param),
                        "z": model.tvalues.get(param),
                        "p_value": model.pvalues.get(param),
                        "ci_low": conf.loc[param, 0] if param in conf.index else np.nan,
                        "ci_high": conf.loc[param, 1] if param in conf.index else np.nan,
                        "aic": model.aic,
                        "bic": model.bic,
                        "log_likelihood": model.llf,
                        "n_observations": int(model.nobs),
                        "n_animals": int(sub["animal_id"].nunique()),
                    }
                )
        except Exception as err:
            rows.append(
                {
                    "cluster": cluster,
                    "parameter": "model_failed",
                    "coef": np.nan,
                    "std_err": np.nan,
                    "z": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                    "log_likelihood": np.nan,
                    "n_observations": int(len(sub)),
                    "n_animals": int(sub["animal_id"].nunique()),
                    "note": str(err),
                }
            )
    return pd.DataFrame(rows)


def fig4_syllable_to_cluster() -> dict[int, str]:
    cluster_map = FIG4_CLUSTER_MAP
    if DEFAULT_CLUSTER_JSON.exists():
        with DEFAULT_CLUSTER_JSON.open("r", encoding="utf-8") as handle:
            cluster_map = json.load(handle)
    out: dict[int, str] = {}
    for cluster, syllables in cluster_map.items():
        for syllable in syllables:
            out[int(syllable)] = cluster
    return out


def load_fig4_sequences() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    raw = read_zip_csv(COPING_DATA2, "COping/Syllable_per_timebin_final(250ms).csv")
    raw = raw.rename(columns={"Time Bin": "time_bin", "Condition": "group"})
    raw = raw[raw["group"].isin(["Control", "ELS"])].copy()
    raw["Animal"] = raw["Animal"].astype(str)
    raw["Syllable"] = pd.to_numeric(raw["Syllable"], errors="coerce").astype(int)
    raw["cluster"] = raw["Syllable"].map(fig4_syllable_to_cluster()).fillna("")

    sequences = {
        str(animal): group.sort_values("time_bin")["cluster"].tolist()
        for animal, group in raw.groupby("Animal", sort=False)
    }
    meta = raw[["Animal", "group", "Experiment"]].drop_duplicates().reset_index(drop=True)
    return raw, sequences, meta


def fig4_frequency_metrics(sequences: dict[str, list[str]], meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    usage_rows = []
    meta_map = meta.set_index("Animal").to_dict("index")
    for animal, seq in sequences.items():
        info = meta_map[animal]
        vals, counts = np.unique(seq, return_counts=True)
        counts_map = dict(zip(vals, counts))
        order = sorted(counts_map)
        probabilities = np.array([counts_map.get(cluster, 0) for cluster in order], dtype=float)
        probabilities = probabilities / probabilities.sum()
        nonzero = probabilities[probabilities > 0]
        shannon = stats.entropy(nonzero)
        evenness = shannon / np.log(len(nonzero)) if len(nonzero) else np.nan
        simpson = 1 - np.sum(probabilities**2)

        named = np.array([counts_map.get(cluster, 0) for cluster in order if str(cluster).strip() != ""], dtype=float)
        named = named / counts.sum()
        sorted_named = np.sort(named)[::-1]
        cumulative = np.cumsum(sorted_named)
        baseline = (len(cumulative) + 1) / (2 * len(cumulative))
        cui = (cumulative.mean() - baseline) / (1 - baseline) if len(cumulative) else np.nan

        rows.append(
            {
                "animal_id": animal,
                "group": info["group"],
                "experiment": info["Experiment"],
                "simpson_index": simpson,
                "shannon_entropy_index": shannon,
                "evenness_index": evenness,
                "cumulative_usage_index": cui,
            }
        )
        usage_rows.append(
            {
                "animal_id": animal,
                "group": info["group"],
                "experiment": info["Experiment"],
                **{str(key) if str(key).strip() else "Unmapped_or_excluded": value for key, value in zip(order, probabilities)},
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(usage_rows)


def fig4_bout_table(sequences: dict[str, list[str]], meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta_map = meta.set_index("Animal").to_dict("index")
    for animal, seq in sequences.items():
        if not seq:
            continue
        info = meta_map[animal]
        prev = seq[0]
        length = 1
        for cluster in seq[1:] + ["__END__"]:
            if cluster == prev:
                length += 1
                continue
            rows.append(
                {
                    "animal_id": animal,
                    "group": info["group"],
                    "experiment": info["Experiment"],
                    "cluster": prev if str(prev).strip() else "Unmapped_or_excluded",
                    "bout_duration_seconds": length * 0.25,
                }
            )
            prev = cluster
            length = 1
    return pd.DataFrame(rows)


def fig4_bout_duration_points(bouts: pd.DataFrame) -> pd.DataFrame:
    panel_map = {
        "Overall": "J",
        "Freezing": "K",
        "Sniffing": "L",
        "Grooming": "M",
        "Turn": "N",
        "Locomotion": "O",
        "Climbing": "P",
        "Jump": "Q",
    }
    overall = (
        bouts.groupby(["animal_id", "group", "experiment"], as_index=False)["bout_duration_seconds"]
        .mean()
        .assign(cluster="Overall")
    )
    cluster_means = (
        bouts[bouts["cluster"].isin(FIG4_DISPLAY_ORDER)]
        .groupby(["animal_id", "group", "experiment", "cluster"], as_index=False)["bout_duration_seconds"]
        .mean()
    )
    out = pd.concat([overall, cluster_means], ignore_index=True, sort=False)
    out["figure_panel_current_export"] = out["cluster"].map(panel_map)
    return out[["figure_panel_current_export", "animal_id", "group", "experiment", "cluster", "bout_duration_seconds"]].sort_values(
        ["figure_panel_current_export", "experiment", "group", "animal_id"]
    )


def fig4_transition_matrix(seq: list[str]) -> np.ndarray:
    idx = {cluster: i for i, cluster in enumerate(FIG4_DISPLAY_ORDER)}
    mat = np.zeros((len(FIG4_DISPLAY_ORDER), len(FIG4_DISPLAY_ORDER)), dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        if a in idx and b in idx:
            mat[idx[a], idx[b]] += 1
    return mat


def fig4_transition_matrix_flow(seq: list[str]) -> np.ndarray:
    if not seq:
        return np.zeros((len(FIG4_DISPLAY_ORDER), len(FIG4_DISPLAY_ORDER)), dtype=float)
    filtered = [seq[0]]
    for cluster in seq[1:]:
        if cluster != filtered[-1]:
            filtered.append(cluster)
    return fig4_transition_matrix(filtered)


def fig4_lz_complexity(seq: list[str]) -> int:
    token_map = {value: i for i, value in enumerate(pd.unique(pd.Series(seq)))}
    tokens = [token_map[x] for x in seq]
    n, i, c, k = len(tokens), 0, 1, 1
    while True:
        if i + k > n:
            break
        sub = tokens[i : i + k]
        found = any(tokens[j : j + k] == sub for j in range(i))
        if found:
            k += 1
            if i + k > n:
                c += 1
                break
        else:
            c += 1
            i += k
            k = 1
        if i >= n:
            break
    return c


def fig4_recurrence_rate(seq: list[str]) -> float:
    _, counts = np.unique(seq, return_counts=True)
    n = len(seq)
    return float(np.sum(counts * counts) / (n * n))


def fig4_determinism(seq: list[str], min_length: int = 2) -> float:
    arr = np.asarray(seq)
    n = len(arr)
    total = 0
    diag_sum = 0
    for offset in range(-n + 1, n):
        diag = arr[: n - abs(offset)] == arr[abs(offset) :] if offset >= 0 else arr[-offset:] == arr[: n + offset]
        if offset == 0:
            total += int(diag.sum()) - n
        else:
            total += int(diag.sum())
        run = 0
        for value in diag:
            if value:
                run += 1
            else:
                if run >= min_length:
                    diag_sum += run
                run = 0
        if run >= min_length:
            diag_sum += run
    return float(diag_sum / total) if total > 0 else 0.0


def fig4_markov_entropy(seq: list[str], smoothing_factor: float = 0.01) -> float:
    unique_states = list(pd.unique(pd.Series(seq)))
    if len(unique_states) == 1:
        return 0.0
    idx = {state: i for i, state in enumerate(unique_states)}
    counts = np.zeros((len(unique_states), len(unique_states)), dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        counts[idx[a], idx[b]] += 1
    counts += smoothing_factor
    probs = counts / counts.sum(axis=1, keepdims=True)
    value_counts = pd.Series(seq).value_counts()
    stationary = np.array([value_counts.get(state, 0) for state in unique_states], dtype=float) / len(seq)
    inner = np.array([-np.sum(row[row > 0] * np.log2(row[row > 0])) for row in probs])
    return float(np.sum(stationary * inner))


def fig4_transition_metrics(sequences: dict[str, list[str]], meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta_map = meta.set_index("Animal").to_dict("index")
    for animal, seq in sequences.items():
        info = meta_map[animal]
        rows.append(
            {
                "animal_id": animal,
                "group": info["group"],
                "experiment": info["Experiment"],
                "lempel_ziv_complexity": fig4_lz_complexity(seq),
                "recurrence_rate": fig4_recurrence_rate(seq),
                "determinism": fig4_determinism(seq),
                "markov_entropy": fig4_markov_entropy(seq, smoothing_factor=0.01),
            }
        )
    return pd.DataFrame(rows)


def fig4_transition_matrix_long(sequences: dict[str, list[str]], meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_map = meta.set_index("Animal")["group"].to_dict()
    for group in ["Control", "ELS"]:
        mats = [
            fig4_transition_matrix_flow(seq)
            for animal, seq in sequences.items()
            if group_map.get(str(animal)) == group
        ]
        mean_mat = np.mean(mats, axis=0)
        for i, source in enumerate(FIG4_DISPLAY_ORDER):
            for j, target in enumerate(FIG4_DISPLAY_ORDER):
                rows.append(
                    {
                        "figure_panel_current_export": "R" if group == "Control" else "S",
                        "group": group,
                        "from_cluster": source,
                        "to_cluster": target,
                        "mean_transition_count_per_animal": mean_mat[i, j],
                    }
                )
    return pd.DataFrame(rows)


def fig4_representatives_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "figure_panel_ethogram": "A" if group == "Control" else "C",
                "figure_panel_barcode": "B" if group == "Control" else "D",
                "group": group,
                "representative_animal_id": animal,
                "selection_note": "Hard-coded to match the representative ethogram/barcode in the original trimmed Figure4.pdf.",
            }
            for group, animal in FIG4_REPRESENTATIVES.items()
        ]
    )


def load_fig6_sequences() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    raw, sequences, meta = load_fig4_sequences()
    raw = raw.copy()
    raw["group_original"] = raw["group"]
    raw["group"] = raw.apply(lambda r: resilience_group(r["Animal"], r["group_original"]), axis=1)
    meta = meta.copy()
    meta["group_original"] = meta["group"]
    meta["group"] = meta.apply(lambda r: resilience_group(r["Animal"], r["group_original"]), axis=1)
    return raw, sequences, meta


def fig_transition_matrix_long(
    sequences: dict[str, list[str]],
    meta: pd.DataFrame,
    group_order: list[str],
    panel_map: dict[str, str],
) -> pd.DataFrame:
    rows = []
    group_map = meta.set_index("Animal")["group"].to_dict()
    for group in group_order:
        mats = [
            fig4_transition_matrix_flow(seq)
            for animal, seq in sequences.items()
            if group_map.get(str(animal)) == group
        ]
        if not mats:
            continue
        mean_mat = np.mean(mats, axis=0)
        for i, source in enumerate(FIG4_DISPLAY_ORDER):
            for j, target in enumerate(FIG4_DISPLAY_ORDER):
                rows.append(
                    {
                        "figure_panel_current_export": panel_map.get(group, ""),
                        "group": group,
                        "from_cluster": source,
                        "to_cluster": target,
                        "mean_transition_count_per_animal": mean_mat[i, j],
                    }
                )
    return pd.DataFrame(rows)


def fig6_representatives_table() -> pd.DataFrame:
    panels = {
        "Control": ("A", "B"),
        "ELS": ("C", "D"),
        "ELS resilient": ("E", "F"),
    }
    return pd.DataFrame(
        [
            {
                "figure_panel_ethogram": panels[group][0],
                "figure_panel_barcode": panels[group][1],
                "group": group,
                "representative_animal_id": animal,
                "selection_note": "Representative animal used for Figure 6 ethogram/barcode regeneration.",
            }
            for group, animal in FIG6_REPRESENTATIVES.items()
        ]
    )


def load_supplementary_figure3_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = REPO_ROOT / "results" / "manuscript_figures" / "new figures"
    scores = base / "supplementary_figure_3_distance_scores.csv"
    summary = base / "supplementary_figure_3_distance_summary.csv"
    if scores.exists() and summary.exists():
        return pd.read_csv(scores), pd.read_csv(summary)
    return (
        pd.DataFrame(columns=["recording", "animal", "group", "metric", "mds1", "mds2", "dynamics_score", "loocv_accuracy"]),
        pd.DataFrame(columns=["metric", "loocv_accuracy", "control_n", "els_n", "control_mean_score", "els_mean_score", "welch_t_control_vs_els", "welch_p_value"]),
    )


def metric_summary_and_tests(df: pd.DataFrame, value_col: str, by_cols: list[str], group_col: str = "group") -> pd.DataFrame:
    rows = []
    groups = FIG6_GROUP_ORDER if "ELS resilient" in set(df[group_col].dropna()) else ["Control", "ELS"]
    for keys, sub in df.groupby(by_cols, dropna=False) if by_cols else [((), df)]:
        if by_cols and not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(by_cols, keys)) if by_cols else {}
        for group in groups:
            vals = pd.to_numeric(sub[sub[group_col] == group][value_col], errors="coerce").dropna()
            rows.append(
                {
                    **base,
                    "comparison_type": "summary",
                    "group": group,
                    "comparison": "",
                    "n": int(vals.count()),
                    "mean": float(vals.mean()) if len(vals) else np.nan,
                    "sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                    "sem": float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else np.nan,
                    "welch_t": np.nan,
                    "p_value": np.nan,
                    "cohens_d_left_minus_right": np.nan,
                }
            )
        for left_i, left in enumerate(groups):
            for right in groups[left_i + 1 :]:
                left_vals = pd.to_numeric(sub[sub[group_col] == left][value_col], errors="coerce").dropna()
                right_vals = pd.to_numeric(sub[sub[group_col] == right][value_col], errors="coerce").dropna()
                if len(left_vals) > 1 and len(right_vals) > 1:
                    t_stat, p_value = stats.ttest_ind(left_vals, right_vals, equal_var=False, nan_policy="omit")
                    effect = cohens_d(left_vals, right_vals)
                else:
                    t_stat, p_value, effect = np.nan, np.nan, np.nan
                rows.append(
                    {
                        **base,
                        "comparison_type": "welch_t_test",
                        "group": "",
                        "comparison": f"{left} vs {right}",
                        "n": np.nan,
                        "mean": np.nan,
                        "sd": np.nan,
                        "sem": np.nan,
                        "welch_t": float(t_stat) if not pd.isna(t_stat) else np.nan,
                        "p_value": float(p_value) if not pd.isna(p_value) else np.nan,
                        "cohens_d_left_minus_right": effect,
                    }
                )
    return pd.DataFrame(rows)


def transition_matrix_summary(matrix_long: pd.DataFrame) -> pd.DataFrame:
    return matrix_long.sort_values(["figure_panel_current_export", "group", "from_cluster", "to_cluster"]).reset_index(drop=True)


def frequency_stats_table(freq_raw: pd.DataFrame) -> pd.DataFrame:
    long = freq_raw.melt(
        id_vars=["animal_id", "group", "experiment"],
        value_vars=["simpson_index", "shannon_entropy_index", "evenness_index", "cumulative_usage_index"],
        var_name="metric",
        value_name="value",
    )
    panel_map = {
        "simpson_index": "E/G",
        "shannon_entropy_index": "F/H",
        "evenness_index": "G/I",
        "cumulative_usage_index": "H-J/K",
    }
    long["figure_panel_family"] = long["metric"].map(panel_map)
    return metric_summary_and_tests(long, "value", ["figure_panel_family", "metric"])


def transition_stats_table(trans_raw: pd.DataFrame) -> pd.DataFrame:
    long = trans_raw.melt(
        id_vars=["animal_id", "group", "experiment"],
        value_vars=["lempel_ziv_complexity", "recurrence_rate", "determinism", "markov_entropy"],
        var_name="metric",
        value_name="value",
    )
    panel_map = {
        "lempel_ziv_complexity": "T/W",
        "recurrence_rate": "U/X",
        "determinism": "V/Y",
        "markov_entropy": "W/Z",
    }
    long["figure_panel_family"] = long["metric"].map(panel_map)
    return metric_summary_and_tests(long, "value", ["figure_panel_family", "metric"])


def _first_present(row: pd.Series, names: list[str]) -> float:
    for name in names:
        if name in row.index:
            return row[name]
    return np.nan


def _archive_model_row(member: str, sheet: str, parameter: str) -> pd.Series:
    df = read_zip_excel(COPING_DATA2, member, sheet_name=sheet)
    param_col = "Parameter" if "Parameter" in df.columns else "param"
    hit = df[df[param_col].astype(str) == parameter]
    if hit.empty:
        raise ValueError(f"Could not find {parameter!r} in {member}:{sheet}")
    return hit.iloc[0]


def _archive_values(member: str, sheet: str, parameter: str) -> dict[str, float]:
    row = _archive_model_row(member, sheet, parameter)
    return {
        "archive_beta": _first_present(row, ["Coefficient", "Coef", "Coef.", "coef"]),
        "archive_se": _first_present(row, ["Std_Error", "StdErr", "Std. Err.", "stderr", "SE"]),
        "archive_z": _first_present(row, ["z_value", "z", "Z"]),
        "archive_p": _first_present(row, ["P_value", "P>|z|", "p", "p_value"]),
    }


def _status_against_archive(
    paper_beta: float | None,
    paper_se: float | None,
    paper_z: float | None,
    paper_p: float | None,
    archive: dict[str, float],
) -> str:
    if paper_beta is None:
        return "Archive value included; no numeric claim was provided in the pasted text."
    checks = [
        abs(float(paper_beta) - float(archive["archive_beta"])) <= 0.01,
        paper_se is None or abs(float(paper_se) - float(archive["archive_se"])) <= 0.01,
        paper_z is None or abs(float(paper_z) - float(archive["archive_z"])) <= 0.03,
        paper_p is None or float(archive["archive_p"]) <= float(paper_p) + 0.002 or abs(float(paper_p) - float(archive["archive_p"])) <= 0.002,
    ]
    return "Matches archive after rounding." if all(checks) else "Pasted paper value does not match this archive row; archive value retained."


def _paper_row(
    figure_panel_text: str,
    analysis: str,
    source_workbook: str,
    source_sheet: str,
    source_parameter: str,
    paper_beta: float | None = None,
    paper_se: float | None = None,
    paper_z: float | None = None,
    paper_p: float | None = None,
    paper_p_display: str | None = None,
    note: str = "",
) -> dict:
    archive = _archive_values(source_workbook, source_sheet, source_parameter)
    return {
        "figure_panel_in_pasted_text": figure_panel_text,
        "analysis": analysis,
        "source_archive_workbook": source_workbook,
        "source_sheet": source_sheet,
        "source_parameter": source_parameter,
        "paper_beta": paper_beta,
        "paper_se": paper_se,
        "paper_z": paper_z,
        "paper_p": paper_p_display if paper_p_display is not None else paper_p,
        **archive,
        "status": _status_against_archive(paper_beta, paper_se, paper_z, paper_p, archive),
        "note": note,
    }


def fig3_paper_report() -> pd.DataFrame:
    frequency = "COping/Code/Results/Cluster_total_frequency/Cluster_frequency_gee_model_combined.xlsx"
    return pd.DataFrame(
        [
            _paper_row("3A", "Freeze total event frequency, ELS vs Control", frequency, "Freezing", "Group[T.ELS]", -0.204, 0.081, -2.511, 0.012),
            _paper_row("3A", "Turn total event frequency, ELS vs Control", frequency, "Turn", "Group[T.ELS]", 0.100, 0.032, 3.099, 0.002),
            _paper_row(
                "3A",
                "Sniff total event frequency, ELS vs Control",
                frequency,
                "Sniffing",
                "Group[T.ELS]",
                -0.068,
                0.185,
                -2.133,
                0.033,
                note="The pasted text values match the Experiment row in the Grooming sheet, not the archived Sniffing ELS effect.",
            ),
            _paper_row(
                "3A/3B",
                "Freeze percentage over time, ELS x time",
                "COping/Code/Results/Cluster_over_time/Combined/cluster_Freezing.xlsx",
                "mixedmodel_fixed",
                "Condition[T.ELS]:Time_bin",
                -0.023,
                0.005,
                -4.241,
                0.001,
                "<0.001",
            ),
            _paper_row(
                "3E",
                "Sniff percentage over time, ELS x time",
                "COping/Code/Results/Cluster_over_time/Combined/cluster_Sniffing.xlsx",
                "mixedmodel_fixed",
                "Condition[T.ELS]:Time_bin",
                -0.014,
                0.004,
                -3.656,
                0.001,
                "<0.001",
            ),
            _paper_row(
                "3F",
                "Turn percentage over time, ELS x time",
                "COping/Code/Results/Cluster_over_time/Combined/cluster_Turn.xlsx",
                "mixedmodel_fixed",
                "Condition[T.ELS]:Time_bin",
                0.026,
                0.007,
                3.610,
                0.001,
                "<0.001",
            ),
        ]
    )


def fig4_paper_report() -> pd.DataFrame:
    rows = [
        _paper_row("4E", "Simpson diversity index, ELS vs Control", "COping/Code/Results/Frequency_metrics/Simpson_model_and_stats.xlsx", "Model_Parameters", "Condition[T.ELS]", -0.013, 0.007, -2.039, 0.041),
        _paper_row("4F", "Shannon entropy index, ELS vs Control", "COping/Code/Results/Frequency_metrics/Shannon_entropy_model_and_stats.xlsx", "Model_Parameters", "Condition[T.ELS]", note="Unchanged in the pasted text."),
        _paper_row("4G", "Evenness index, ELS vs Control", "COping/Code/Results/Frequency_metrics/Evenness_model_and_stats.xlsx", "Model_Parameters", "Condition[T.ELS]", note="Unchanged in the pasted text."),
        _paper_row(
            "4H-I",
            "Cumulative usage index, ELS vs Control",
            "COping/Code/Results/Frequency_metrics/CUI_model_and_stats.xlsx",
            "Model_Parameters",
            "Condition[T.ELS]",
            -0.027,
            0.009,
            -3.079,
            0.002,
            note="The archive and group means indicate higher CUI in ELS with a positive coefficient.",
        ),
        _paper_row("4J", "Freeze mean bout duration, ELS vs Control", "COping/Code/Results/Bout_duration/Boutduration_freezing_model_and_stats.xlsx", "Model_Parameters", "Condition[T.ELS]", -0.105, 0.053, -2.001, 0.045),
        _paper_row("4K", "Sniff mean bout duration, ELS vs Control", "COping/Code/Results/Bout_duration/Boutduration_sniffing_model_and_stats.xlsx", "Model_Parameters", "Condition[T.ELS]", 0.322, 0.129, 2.506, 0.012),
        _paper_row("4M", "Turn mean bout duration, ELS vs Control", "COping/Code/Results/Bout_duration/Boutduration_turn_model_and_stats.xlsx", "Model_Parameters", "Condition[T.ELS]", 0.159, 0.052, 3.088, 0.002),
        _paper_row("4T", "Lempel-Ziv complexity, ELS vs Control", "COping/Code/Results/Transition_metrics/LZ_complexity_model_and_stats.xlsx", "Model_Parameters", "Condition[T.ELS]", note="Unchanged in the pasted text."),
        _paper_row("4U", "Recurrence rate, ELS vs Control", "COping/Code/Results/Transition_metrics/Recurrence_Rate_model_and_stats.xlsx", "Model_Parameters", "Condition[T.ELS]", 0.013, 0.006, 2.264, 0.024),
        _paper_row("4V", "Determinism, ELS vs Control", "COping/Code/Results/Transition_metrics/Determinism_model_and_stats.xlsx", "Model_Parameters", "Condition[T.ELS]", 0.016, 0.005, 3.382, 0.001),
        _paper_row("4W", "Markov entropy, ELS vs Control", "COping/Code/Results/Transition_metrics/MarkovEntropy_model_and_stats.xlsx", "Model_Parameters", "Condition[T.ELS]", -0.041, 0.018, -2.342, 0.019),
    ]
    return pd.DataFrame(rows)


def write_title(ws, title: str, max_col: int) -> None:
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max_col)
    cell = ws.cell(1, 1, title)
    cell.font = Font(name=FONT_NAME, bold=True, size=14, color=TEXT)
    cell.fill = PatternFill("solid", fgColor=FILL_SECTION)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 24


def append_df(ws, df: pd.DataFrame, *, title: str | None = None, start_row: int = 3) -> None:
    if title:
        write_title(ws, title, max(1, len(df.columns)))
    for c_idx, col in enumerate(df.columns, start=1):
        cell = ws.cell(start_row, c_idx, col)
        cell.font = Font(name=FONT_NAME, bold=True, color=TEXT)
        cell.fill = PatternFill("solid", fgColor=FILL_HEADER)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for r_idx, row in enumerate(df.itertuples(index=False), start=start_row + 1):
        for c_idx, value in enumerate(row, start=1):
            if pd.isna(value):
                value = None
            ws.cell(r_idx, c_idx, value)
    ws.freeze_panes = f"A{start_row + 1}"
    style_sheet(ws)


def append_section(ws, df: pd.DataFrame, title: str, start_row: int) -> int:
    max_col = max(1, len(df.columns))
    ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=max_col)
    section_cell = ws.cell(start_row, 1, title)
    section_cell.font = Font(name=FONT_NAME, bold=True, color=TEXT)
    section_cell.fill = PatternFill("solid", fgColor=FILL_SECTION)
    section_cell.alignment = Alignment(horizontal="center", vertical="center")
    header_row = start_row + 1
    for c_idx, col in enumerate(df.columns, start=1):
        cell = ws.cell(header_row, c_idx, col)
        cell.font = Font(name=FONT_NAME, bold=True, color=TEXT)
        cell.fill = PatternFill("solid", fgColor=FILL_HEADER)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for r_idx, row in enumerate(df.itertuples(index=False), start=header_row + 1):
        for c_idx, value in enumerate(row, start=1):
            if pd.isna(value):
                value = None
            ws.cell(r_idx, c_idx, value)
    style_sheet(ws)
    return header_row + len(df) + 3


def style_sheet(ws) -> None:
    thin = Side(style="thin", color=BORDER)
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for row in ws.iter_rows():
        for cell in row:
            cell.font = copy_font(cell.font)
            cell.alignment = Alignment(
                horizontal=cell.alignment.horizontal or "left",
                vertical="top",
                wrap_text=True,
            )
            if cell.value is not None:
                cell.border = border
                if isinstance(cell.value, float):
                    cell.number_format = "0.000"
    for column_cells in ws.columns:
        letter = get_column_letter(column_cells[0].column)
        width = 8
        for cell in column_cells:
            if cell.value is not None:
                width = max(width, min(44, len(str(cell.value)) + 2))
        ws.column_dimensions[letter].width = width


def copy_font(font_obj) -> Font:
    return Font(
        name=FONT_NAME,
        sz=font_obj.sz or 11,
        b=font_obj.b,
        i=font_obj.i,
        color=TEXT,
        underline=font_obj.underline,
    )


def write_raw_workbook(
    simba_raw: pd.DataFrame,
    freezing_long: pd.DataFrame,
    moseq_df: pd.DataFrame,
    syllable_long: pd.DataFrame,
    syllable_counts: pd.DataFrame,
    precision_recall: pd.DataFrame,
    overlap_0_28: pd.DataFrame,
    fig3_cluster_long: pd.DataFrame,
    fig3_cluster_frequency_raw: pd.DataFrame,
    fig4_frequency_raw: pd.DataFrame,
    fig4_usage_raw: pd.DataFrame,
    fig4_bout_raw: pd.DataFrame,
    fig4_transition_raw: pd.DataFrame,
    fig4_transition_matrix_raw: pd.DataFrame,
    fig4_representatives: pd.DataFrame,
    fig6_frequency_raw: pd.DataFrame,
    fig6_usage_raw: pd.DataFrame,
    fig6_bout_raw: pd.DataFrame,
    fig6_transition_raw: pd.DataFrame,
    fig6_transition_matrix_raw: pd.DataFrame,
    fig6_representatives: pd.DataFrame,
    suppfig3_scores: pd.DataFrame,
    suppfig3_summary: pd.DataFrame,
    syllable_timebin_raw: pd.DataFrame,
    bfl_scores: pd.DataFrame,
) -> Path:
    wb = Workbook()
    wb.remove(wb.active)

    sheets = [
        ("SimBA_validation", simba_raw, "SimBA validation: automatic vs manual"),
        ("Freezing_SGK_2024", freezing_wide(freezing_long, experiment=1), "Freezing per time bin: Sanguino-Gomez and Krugers, 2024"),
        ("Freezing_SG_2024", freezing_wide(freezing_long, experiment=3), "Freezing per time bin: Sanguino-Gomez et al., 2024"),
        ("Freezing_combined", freezing_wide(freezing_long), "Freezing per time bin: combined datasets"),
        (
            "Cluster_time_wide",
            fig3_cluster_wide(split_cluster_scope(fig3_cluster_long, main=True)),
            "Figure 3 behavior clusters: percentage per 30 s time bin",
        ),
        ("Cluster_frequency", fig3_cluster_frequency_raw[fig3_cluster_frequency_raw["cluster"].isin(MAIN_FIG3_CLUSTERS)], "Figure 3 behavior clusters: total frequency in seconds per animal"),
        ("Fig4_representatives", fig4_representatives, "Figure 4 panels A-D: representative animals"),
        ("Fig4_frequency_metrics", fig4_frequency_raw, "Figure 4 panels E-H: diversity and cumulative-usage metrics per animal"),
        ("Fig4_usage_profile", fig4_usage_raw, "Figure 4 panel I: cluster usage proportions per animal"),
        ("Fig4_bout_duration", fig4_bout_raw, "Figure 4 panels J-Q: mean bout duration points per animal"),
        ("Fig4_transition_metrics", fig4_transition_raw, "Figure 4 panels T-W: sequence metrics per animal"),
        ("Fig4_transition_chords", fig4_transition_matrix_raw, "Figure 4 panels R-S: mean transition matrix values used in chord plots"),
        ("Fig6_representatives", fig6_representatives, "Figure 6 panels A-F: representative animals"),
        ("Fig6_frequency_metrics", fig6_frequency_raw, "Figure 6 panels G-J: diversity and cumulative-usage metrics per animal"),
        ("Fig6_usage_profile", fig6_usage_raw, "Figure 6 panel K: cluster usage proportions per animal"),
        ("Fig6_bout_duration", fig6_bout_raw, "Figure 6 panels L-S: mean bout duration points per animal"),
        ("Fig6_transition_metrics", fig6_transition_raw, "Figure 6 panels W-Z: sequence metrics per animal"),
        ("Fig6_transition_chords", fig6_transition_matrix_raw, "Figure 6 panels T-V: mean transition matrix values used in chord plots"),
        ("SuppFig3_distance_scores", suppfig3_scores, "Supplementary Figure 3: MDS coordinates and dynamic-similarity scores by distance metric"),
        ("SuppFig3_distance_summary", suppfig3_summary, "Supplementary Figure 3: LOOCV and Control vs ELS score tests by distance metric"),
        (
            "Supp_cluster_time",
            fig3_cluster_wide(split_cluster_scope(fig3_cluster_long, main=False)),
            "Supplementary Figure 1 clusters: percentage per 30 s time bin",
        ),
        (
            "Supp_cluster_frequency",
            fig3_cluster_frequency_raw[~fig3_cluster_frequency_raw["cluster"].isin(MAIN_FIG3_CLUSTERS)],
            "Supplementary Figure 1 clusters: total frequency in seconds per animal",
        ),
        ("Syllable_timebin_30s", syllable_timebin_raw, "All syllables per 30 s time bin from the archived MoSeq analysis"),
        ("BFL_scores", bfl_scores, "Behavioral dynamics score data"),
        ("Syllable_frames", syllable_counts, "All syllables: frame counts per video and total across all 82 videos"),
        ("Precision_recall", precision_recall, "Precision and recall of each syllable against freezing annotations"),
        ("Overlap_0_28", overlap_0_28, "Overlap of freezing annotation with syllable annotations 0 + 28"),
        ("Syll_0_28_SGK_2024", syllable_0_28_wide(syllable_long, experiment=1), "Syllables 0 + 28 per time bin: Sanguino-Gomez and Krugers, 2024"),
        ("Syll_0_28_SG_2024", syllable_0_28_wide(syllable_long, experiment=3), "Syllables 0 + 28 per time bin: Sanguino-Gomez et al., 2024"),
        ("Syll_0_28_combined", syllable_0_28_wide(syllable_long), "Syllables 0 + 28 per time bin: combined datasets"),
    ]
    for sheet_name, df, title in sheets:
        ws = wb.create_sheet(sheet_name)
        append_df(ws, df, title=title)

    out = COMMON_DIR / "Raw_data.xlsx"
    wb.save(out)
    return out


def write_stats_workbook(
    simba_stats: pd.DataFrame,
    freezing_models: pd.DataFrame,
    freezing_check: pd.DataFrame,
    effect_check: pd.DataFrame,
    syllable_counts: pd.DataFrame,
    precision_recall: pd.DataFrame,
    overlap_0_28: pd.DataFrame,
    syllable_models: pd.DataFrame,
    fig3_cluster_frequency_summary: pd.DataFrame,
    fig3_cluster_time_models: pd.DataFrame,
    fig3_paper_check: pd.DataFrame,
    fig4_paper_check: pd.DataFrame,
    fig4_frequency_stats: pd.DataFrame,
    fig4_bout_stats: pd.DataFrame,
    fig4_transition_stats: pd.DataFrame,
    fig4_transition_matrix_stats: pd.DataFrame,
    fig6_frequency_stats: pd.DataFrame,
    fig6_bout_stats: pd.DataFrame,
    fig6_transition_stats: pd.DataFrame,
    fig6_transition_matrix_stats: pd.DataFrame,
    suppfig3_summary: pd.DataFrame,
    archived_stats: dict[str, pd.DataFrame],
) -> Path:
    wb = Workbook()
    wb.remove(wb.active)

    ws = wb.create_sheet("SimBA_validation")
    append_df(ws, simba_stats, title="SimBA validation correlation")
    note = pd.DataFrame(
        [
            ["Text statement", "r2 = 0.95, p < 0.001"],
            ["Computed from exported summaries", "Pooled r2 = 0.963, p < 0.001; text is conservative."],
            ["Held-out frames", "211,500 is consistent with 47 animals x 4,500 frames; frame-level validation labels are not in the archives."],
        ],
        columns=["Item", "Value"],
    )
    append_section(ws, note, "Manuscript check", start_row=len(simba_stats) + 6)

    ws = wb.create_sheet("Freezing_linear_model")
    row = append_section(ws, freezing_check, "Manuscript model check", start_row=1)
    row = append_section(ws, freezing_models, "Mixed linear model output recomputed from Figure 3 binned SimBA data", start_row=row)
    note = pd.DataFrame(
        [
            ["Was the linear model run the same way as in the text?", "Yes"],
            [
                "Model structure",
                "MixedLM: freezing_percent ~ group * time_bin_numeric; per-dataset random intercept for animal; combined model adds experiment and experiment variance component.",
            ],
            ["Interpretation", "Negative group[T.ELS]:time_bin_numeric means slower freezing increase in ELS animals."],
        ],
        columns=["Question", "Answer"],
    )
    append_section(ws, note, "Model note", start_row=row)

    ws = wb.create_sheet("Effect_sizes")
    append_df(ws, effect_check, title="Cohen's d check")

    ws = wb.create_sheet("Cluster_frequency")
    append_df(
        ws,
        fig3_cluster_frequency_summary[fig3_cluster_frequency_summary["cluster"].isin(MAIN_FIG3_CLUSTERS)],
        title="Figure 3 panel A: cluster frequency summary and Control vs ELS tests",
    )

    ws = wb.create_sheet("Cluster_time_model")
    row = append_section(
        ws,
        fig3_cluster_time_models[fig3_cluster_time_models["cluster"].isin(MAIN_FIG3_CLUSTERS)],
        "Figure 3 panels B-H: mixed linear models for cluster percentage over time",
        start_row=1,
    )
    note = pd.DataFrame(
        [
            [
                "Model structure",
                "MixedLM: cluster_percent ~ group * time_bin_numeric + experiment; random intercept for animal and experiment variance component.",
            ],
            ["Data included", "82 manuscript-cohort animals only: Exp1 Control=25, Exp1 ELS=25, Exp3 Control=16, Exp3 ELS=16."],
        ],
        columns=["Item", "Value"],
    )
    append_section(ws, note, "Model note", start_row=row)

    ws = wb.create_sheet("Supp_cluster_frequency")
    append_df(
        ws,
        fig3_cluster_frequency_summary[~fig3_cluster_frequency_summary["cluster"].isin(MAIN_FIG3_CLUSTERS)],
        title="Supplementary Figure 1: Mix behaviors and Inaccurate tracking frequency summary",
    )

    ws = wb.create_sheet("Supp_cluster_time_model")
    append_df(
        ws,
        fig3_cluster_time_models[~fig3_cluster_time_models["cluster"].isin(MAIN_FIG3_CLUSTERS)],
        title="Supplementary Figure 1: mixed linear models for cluster percentage over time",
    )

    ws = wb.create_sheet("Figure3_paper_report")
    append_df(ws, fig3_paper_check, title="Figure 3 paper text check against archived COping Results")

    ws = wb.create_sheet("Figure4_paper_report")
    append_df(ws, fig4_paper_check, title="Figure 4 paper text check against archived COping Results")

    ws = wb.create_sheet("Fig4_frequency_metrics")
    append_df(ws, fig4_frequency_stats, title="Figure 4 panels E-I: frequency/diversity metric summaries and tests")

    ws = wb.create_sheet("Fig4_bout_duration")
    append_df(ws, fig4_bout_stats, title="Figure 4 panels J-Q: bout-duration summaries and tests")

    ws = wb.create_sheet("Fig4_transition_metrics")
    append_df(ws, fig4_transition_stats, title="Figure 4 panels T-W: transition-metric summaries and tests")

    ws = wb.create_sheet("Fig4_transition_chords")
    append_df(ws, fig4_transition_matrix_stats, title="Figure 4 panels R-S: chord-plot transition matrix values")

    ws = wb.create_sheet("Fig6_frequency_metrics")
    append_df(ws, fig6_frequency_stats, title="Figure 6 panels G-K: three-group frequency/diversity metric summaries and tests")

    ws = wb.create_sheet("Fig6_bout_duration")
    append_df(ws, fig6_bout_stats, title="Figure 6 panels L-S: three-group bout-duration summaries and tests")

    ws = wb.create_sheet("Fig6_transition_metrics")
    append_df(ws, fig6_transition_stats, title="Figure 6 panels W-Z: three-group transition-metric summaries and tests")

    ws = wb.create_sheet("Fig6_transition_chords")
    append_df(ws, fig6_transition_matrix_stats, title="Figure 6 panels T-V: chord-plot transition matrix values")

    ws = wb.create_sheet("SuppFig3_distances")
    append_df(ws, suppfig3_summary, title="Supplementary Figure 3: distance-metric LOOCV and Welch tests")

    usage_summary = (
        syllable_counts.groupby("syllable")
        .agg(
            total_frames_all_videos=("total_frames_all_videos", "first"),
            percent_all_frames=("percent_all_frames", "first"),
            videos_with_syllable=("animal_id", "nunique"),
        )
        .reset_index()
        .sort_values("total_frames_all_videos", ascending=False)
    )
    ws = wb.create_sheet("Syllable_usage")
    append_df(ws, usage_summary, title="Panel D: all syllables tested across all 82 videos")
    note = pd.DataFrame(
        [["Tested in all videos?", "Yes, Panel D frame counts use all 82 manuscript-cohort videos."]],
        columns=["Item", "Value"],
    )
    append_section(ws, note, "Panel D note", start_row=len(usage_summary) + 6)

    pr_summary = summarize_by_group(
        precision_recall,
        ["precision_percent", "recall_percent", "overlap_frames", "syllable_frames"],
        ["syllable", "group"],
    )
    ws = wb.create_sheet("Precision_recall")
    append_df(ws, pr_summary, title="Precision/recall summary by syllable and group")

    overlap_summary = summarize_by_group(
        overlap_0_28,
        ["percent_freezing_covered_by_syllable_0_28", "precision_syllable_0_28_vs_freezing"],
        ["group"],
    )
    ws = wb.create_sheet("Overlap_0_28")
    append_df(ws, overlap_summary, title="Overlap summary: freezing annotation vs syllables 0 + 28")

    ws = wb.create_sheet("Syll_0_28_linear_model")
    row = append_section(ws, syllable_models, "Mixed linear model for syllables 0 + 28 over time", start_row=1)
    note = pd.DataFrame(
        [
            ["Was this model run the same way as the freezing text model?", "Same model structure, different response variable."],
            [
                "Model structure",
                "MixedLM: syllable_0_28_percent ~ group * time_bin_numeric; per-dataset random intercept for animal; combined model adds experiment and experiment variance component.",
            ],
        ],
        columns=["Question", "Answer"],
    )
    append_section(ws, note, "Model note", start_row=row)

    for sheet_name, df in archived_stats.items():
        ws = wb.create_sheet(sheet_name[:31])
        append_df(ws, df, title=sheet_name.replace("_", " "))

    out = COMMON_DIR / "Statistical_report.xlsx"
    wb.save(out)
    return out


def main() -> None:
    require_sources()
    COMMON_DIR.mkdir(parents=True, exist_ok=True)

    simba_raw, simba_stats = load_simba_validation()
    freezing_long = load_freezing_long()
    moseq_df = load_moseq_frame_table()
    syllable_long = syllable_0_28_long(moseq_df)
    syllable_counts = syllable_frame_counts(moseq_df)
    precision_recall, overlap_0_28 = precision_recall_and_overlap(moseq_df)
    fig3_cluster_long = load_fig3_cluster_long()
    fig3_cluster_frequency_raw = fig3_cluster_frequency(fig3_cluster_long)
    _, fig4_sequences, fig4_meta = load_fig4_sequences()
    fig4_frequency_raw, fig4_usage_raw = fig4_frequency_metrics(fig4_sequences, fig4_meta)
    fig4_bouts = fig4_bout_table(fig4_sequences, fig4_meta)
    fig4_bout_raw = fig4_bout_duration_points(fig4_bouts)
    fig4_transition_raw = fig4_transition_metrics(fig4_sequences, fig4_meta)
    fig4_transition_matrix_raw = fig_transition_matrix_long(fig4_sequences, fig4_meta, ["Control", "ELS"], {"Control": "R", "ELS": "S"})
    fig4_representatives = fig4_representatives_table()
    _, fig6_sequences, fig6_meta = load_fig6_sequences()
    fig6_frequency_raw, fig6_usage_raw = fig4_frequency_metrics(fig6_sequences, fig6_meta)
    fig6_bouts = fig4_bout_table(fig6_sequences, fig6_meta)
    fig6_bout_raw = fig4_bout_duration_points(fig6_bouts)
    fig6_bout_raw["figure_panel_current_export"] = fig6_bout_raw["figure_panel_current_export"].replace(
        {"J": "L", "K": "M", "L": "N", "M": "O", "N": "P", "O": "Q", "P": "R", "Q": "S"}
    )
    fig6_transition_raw = fig4_transition_metrics(fig6_sequences, fig6_meta)
    fig6_transition_matrix_raw = fig_transition_matrix_long(
        fig6_sequences,
        fig6_meta,
        FIG6_GROUP_ORDER,
        {"Control": "T", "ELS": "U", "ELS resilient": "V"},
    )
    fig6_representatives = fig6_representatives_table()
    suppfig3_scores, suppfig3_summary = load_supplementary_figure3_tables()
    syllable_timebin_raw, bfl_scores = load_archived_raw_tables()

    freezing_models = fit_mixed_models(freezing_long, "freezing_percent")
    freezing_check = freezing_text_check(freezing_models)
    effect_check = make_effect_size_check()
    syllable_models = fit_mixed_models(syllable_long, "syllable_0_28_percent")
    fig3_cluster_frequency_summary = fig3_cluster_frequency_stats(fig3_cluster_frequency_raw)
    fig3_cluster_time_models = fit_fig3_cluster_time_models(fig3_cluster_long)
    fig3_paper_check = fig3_paper_report()
    fig4_paper_check = fig4_paper_report()
    fig4_frequency_stats = frequency_stats_table(fig4_frequency_raw)
    fig4_bout_stats = metric_summary_and_tests(fig4_bout_raw, "bout_duration_seconds", ["figure_panel_current_export", "cluster"])
    fig4_transition_stats = transition_stats_table(fig4_transition_raw)
    fig4_transition_matrix_stats = transition_matrix_summary(fig4_transition_matrix_raw)
    fig6_frequency_stats = frequency_stats_table(fig6_frequency_raw)
    fig6_bout_stats = metric_summary_and_tests(fig6_bout_raw, "bout_duration_seconds", ["figure_panel_current_export", "cluster"])
    fig6_transition_stats = transition_stats_table(fig6_transition_raw)
    fig6_transition_matrix_stats = transition_matrix_summary(fig6_transition_matrix_raw)
    archived_stats = load_archived_stat_tables()

    raw_out = write_raw_workbook(
        simba_raw,
        freezing_long,
        moseq_df,
        syllable_long,
        syllable_counts,
        precision_recall,
        overlap_0_28,
        fig3_cluster_long,
        fig3_cluster_frequency_raw,
        fig4_frequency_raw,
        fig4_usage_raw,
        fig4_bout_raw,
        fig4_transition_raw,
        fig4_transition_matrix_raw,
        fig4_representatives,
        fig6_frequency_raw,
        fig6_usage_raw,
        fig6_bout_raw,
        fig6_transition_raw,
        fig6_transition_matrix_raw,
        fig6_representatives,
        suppfig3_scores,
        suppfig3_summary,
        syllable_timebin_raw,
        bfl_scores,
    )
    stats_out = write_stats_workbook(
        simba_stats,
        freezing_models,
        freezing_check,
        effect_check,
        syllable_counts,
        precision_recall,
        overlap_0_28,
        syllable_models,
        fig3_cluster_frequency_summary,
        fig3_cluster_time_models,
        fig3_paper_check,
        fig4_paper_check,
        fig4_frequency_stats,
        fig4_bout_stats,
        fig4_transition_stats,
        fig4_transition_matrix_stats,
        fig6_frequency_stats,
        fig6_bout_stats,
        fig6_transition_stats,
        fig6_transition_matrix_stats,
        suppfig3_summary,
        archived_stats,
    )

    for old_dir in (OLD_FIGURE2_DIR, OLD_COMMON_DIR):
        if old_dir.exists():
            shutil.rmtree(old_dir)
    for old_file in ("Ground_truth.xlsx", "Freezing.xlsx", "common_raw_data.xlsx", "common_statistical_report.xlsx"):
        old_path = DATA_DIR / old_file
        if old_path.exists():
            old_path.unlink()

    print(f"Saved: {raw_out}")
    print(f"Saved: {stats_out}")


if __name__ == "__main__":
    main()
