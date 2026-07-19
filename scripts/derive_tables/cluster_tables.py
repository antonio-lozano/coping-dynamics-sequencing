"""Derive analysis-ready cluster tables from the raw 30 s syllable table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "syllable_usage_per_timebin_30s.csv"
PROCESSED = REPO / "data" / "processed"

OUT_FREQ = PROCESSED / "cluster_frequency_per_animal.csv"
OUT_TIME = PROCESSED / "cluster_timecourse_per_animal.csv"
OUT_S0S28 = PROCESSED / "s0s28_timecourse_per_animal.csv"

CLUSTER_MAP = {
    "Freeze": [0, 28],
    "Sniff": [18, 20],
    "Groom": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climb": [111],
    "Jump": [23, 29, 30, 34],
}
BIN_SECONDS = 30


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df.columns = df.columns.str.strip()
    df["Syllable"] = df["Syllable"].astype(int)
    df["time_bin"] = pd.to_numeric(df["Time Bin"])
    df["Percentage"] = pd.to_numeric(df["Percentage"])
    df["animal_id"] = df["Animal"].astype(str).str.strip()
    df["group"] = df["Condition"].astype(str).str.strip()
    df["experiment"] = pd.to_numeric(df["Experiment"])
    return df


def complete_index(df: pd.DataFrame, clusters: list[str]) -> pd.DataFrame:
    animals = df[["animal_id", "group", "experiment"]].drop_duplicates()
    time_s = pd.DataFrame({"time_s": [(i + 1) * BIN_SECONDS for i in range(15)]})
    cluster_df = pd.DataFrame({"cluster": clusters})
    index = (
        animals.assign(_key=1)
        .merge(cluster_df.assign(_key=1), on="_key")
        .merge(time_s.assign(_key=1), on="_key")
        .drop(columns="_key")
    )
    index["time_bin"] = index["time_s"] - BIN_SECONDS
    return index


def derive_cluster_frequency(df: pd.DataFrame) -> pd.DataFrame:
    syllable_to_cluster = {s: c for c, syllables in CLUSTER_MAP.items() for s in syllables}
    mapped = df.copy()
    mapped["cluster"] = mapped["Syllable"].map(syllable_to_cluster)
    mapped = mapped[mapped["cluster"].notna()].copy()
    mapped["seconds"] = mapped["Percentage"] / 100.0 * BIN_SECONDS
    return (
        mapped.groupby(["animal_id", "group", "experiment", "cluster"], as_index=False)["seconds"]
        .sum()
        .rename(columns={"seconds": "frequency_seconds"})
    )


def derive_cluster_timecourse(df: pd.DataFrame) -> pd.DataFrame:
    syllable_to_cluster = {s: c for c, syllables in CLUSTER_MAP.items() for s in syllables}
    mapped = df.copy()
    mapped["cluster"] = mapped["Syllable"].map(syllable_to_cluster)
    mapped = mapped[mapped["cluster"].notna()].copy()
    collapsed = (
        mapped.groupby(["animal_id", "group", "experiment", "cluster", "time_bin"], as_index=False)[
            "Percentage"
        ]
        .sum()
        .rename(columns={"Percentage": "pct"})
    )
    collapsed["time_s"] = collapsed["time_bin"] + BIN_SECONDS
    index = complete_index(mapped, list(CLUSTER_MAP.keys()))
    return index.merge(
        collapsed[["animal_id", "cluster", "time_s", "pct"]],
        on=["animal_id", "cluster", "time_s"],
        how="left",
    ).fillna({"pct": 0.0})


def derive_s0s28_timecourse(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["Syllable"].isin([0, 28])].copy()
    collapsed = (
        subset.groupby(["animal_id", "group", "experiment", "time_bin"], as_index=False)["Percentage"]
        .sum()
        .rename(columns={"Percentage": "pct_s0s28"})
    )
    collapsed["time_s"] = collapsed["time_bin"] + BIN_SECONDS
    animals = df[["animal_id", "group", "experiment"]].drop_duplicates()
    time_s = pd.DataFrame({"time_s": [(i + 1) * BIN_SECONDS for i in range(15)]})
    index = animals.assign(_key=1).merge(time_s.assign(_key=1), on="_key").drop(columns="_key")
    index["time_bin"] = index["time_s"] - BIN_SECONDS
    out = index.merge(
        collapsed[["animal_id", "time_s", "pct_s0s28"]],
        on=["animal_id", "time_s"],
        how="left",
    ).fillna({"pct_s0s28": 0.0})
    out["bin"] = (out["time_s"] // BIN_SECONDS - 1).astype(int)
    return out[["animal_id", "group", "experiment", "bin", "time_s", "pct_s0s28"]]


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    raw = load_raw()

    freq = derive_cluster_frequency(raw)
    freq.to_csv(OUT_FREQ, index=False)

    timecourse = derive_cluster_timecourse(raw)
    timecourse.to_csv(OUT_TIME, index=False)

    s0s28 = derive_s0s28_timecourse(raw)
    s0s28.to_csv(OUT_S0S28, index=False)

    print(f"Saved {OUT_FREQ.relative_to(REPO)} ({len(freq)} rows)")
    print(f"Saved {OUT_TIME.relative_to(REPO)} ({len(timecourse)} rows)")
    print(f"Saved {OUT_S0S28.relative_to(REPO)} ({len(s0s28)} rows)")


if __name__ == "__main__":
    main()
