"""Derive tracking-exclusion tables from the raw 30 s syllable table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "syllable_usage_per_timebin_30s.csv"
PROCESSED = REPO / "data" / "processed"

OUT_ANIMAL = PROCESSED / "tracking_exclusions_per_animal.csv"
OUT_TIME = PROCESSED / "supplementary_figure1_tracking_clusters.csv"

INACCURATE_TRACKING = {2, 4, 8, 9, 22, 31, 32, 33}
MIX_BEHAVIORS = {7, 13, 17}
BIN_SECONDS = 30
PROJECT = {
    1: "Sanguino-Gomez and Krugers, 2024",
    3: "Sanguino-Gomez et al., 2024",
}


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df.columns = df.columns.str.strip()
    df["Syllable"] = df["Syllable"].astype(int)
    df["Percentage"] = pd.to_numeric(df["Percentage"])
    df["time_start_s"] = pd.to_numeric(df["Time Bin"])
    df["animal_id"] = df["Animal"].astype(str).str.strip()
    df["group"] = df["Condition"].astype(str).str.strip()
    df["experiment"] = pd.to_numeric(df["Experiment"]).astype(int)
    return df


def subset_seconds(df: pd.DataFrame, syllables: set[int], column: str) -> pd.DataFrame:
    sub = df[df["Syllable"].isin(syllables)].copy()
    sub[column] = sub["Percentage"] * BIN_SECONDS / 100.0
    return (
        sub.groupby(["animal_id", "group", "experiment"], as_index=False)[column]
        .sum()
    )


def derive_per_animal(df: pd.DataFrame) -> pd.DataFrame:
    inaccurate = subset_seconds(df, INACCURATE_TRACKING, "inaccurate_tracking_seconds")
    mixed = subset_seconds(df, MIX_BEHAVIORS, "mix_behaviors_seconds")
    out = inaccurate.merge(mixed, on=["animal_id", "group", "experiment"], how="outer").fillna(0.0)
    out["total_excluded_seconds"] = out["inaccurate_tracking_seconds"] + out["mix_behaviors_seconds"]
    return out[
        [
            "animal_id",
            "group",
            "experiment",
            "inaccurate_tracking_seconds",
            "mix_behaviors_seconds",
            "total_excluded_seconds",
        ]
    ]


def derive_supplementary_time(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("Inaccurate tracking", INACCURATE_TRACKING),
        ("Mix behaviors", MIX_BEHAVIORS),
    ]
    rows = []
    base = df[["animal_id", "group", "experiment", "time_start_s"]].drop_duplicates()
    for cluster, syllables in specs:
        sub = df[df["Syllable"].isin(syllables)].copy()
        collapsed = (
            sub.groupby(["animal_id", "group", "experiment", "time_start_s"], as_index=False)[
                "Percentage"
            ]
            .sum()
        )
        merged = base.merge(
            collapsed,
            on=["animal_id", "group", "experiment", "time_start_s"],
            how="left",
        ).fillna({"Percentage": 0.0})
        merged["cluster"] = cluster
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True)
    out["seconds"] = out["time_start_s"] + BIN_SECONDS
    out["time_min"] = out["seconds"] / 60.0
    out["time_bin"] = out["seconds"].map(lambda value: f"t_{int(value):03d}s")
    out["project"] = out["experiment"].map(PROJECT)
    out = out.rename(columns={"Percentage": "percentage"})
    return out[
        [
            "animal_id",
            "group",
            "project",
            "experiment",
            "cluster",
            "time_bin",
            "percentage",
            "seconds",
            "time_min",
        ]
    ].sort_values(["animal_id", "cluster", "seconds"])


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    per_animal = derive_per_animal(raw)
    supp_time = derive_supplementary_time(raw)

    per_animal.to_csv(OUT_ANIMAL, index=False)
    supp_time.to_csv(OUT_TIME, index=False)

    print(f"Saved {OUT_ANIMAL.relative_to(REPO)} ({len(per_animal)} rows)")
    print(f"Saved {OUT_TIME.relative_to(REPO)} ({len(supp_time)} rows)")


if __name__ == "__main__":
    main()
