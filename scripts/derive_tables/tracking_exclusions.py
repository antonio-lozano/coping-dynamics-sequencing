"""Compute per-animal tracking exclusion frequency (seconds) from syllable usage CSV."""
import pandas as pd
from pathlib import Path

REPO = Path(__file__).parent.parent
SRC  = REPO / "dataset" / "source" / "syllable_usage_per_timebin_30s.csv"
OUT  = REPO / "dataset" / "source" / "tracking_exclusions_per_animal.csv"

INACCURATE_TRACKING = {2, 4, 8, 9, 22, 31, 32, 33}
MIX_BEHAVIORS       = {7, 13, 17}
BIN_SECONDS         = 30

df = pd.read_csv(SRC)
df.columns = df.columns.str.strip()
df["Syllable"] = df["Syllable"].astype(int)

# Separate the two exclusion types
inac = df[df["Syllable"].isin(INACCURATE_TRACKING)].copy()
mix  = df[df["Syllable"].isin(MIX_BEHAVIORS)].copy()

def sum_seconds(sub):
    """Sum pct × bin_seconds / 100 per animal across all bins and syllables."""
    sub["seconds"] = sub["Percentage"] * BIN_SECONDS / 100.0
    return (
        sub.groupby(["Animal", "Condition", "Experiment"])["seconds"]
        .sum()
        .reset_index()
    )

inac_s = sum_seconds(inac).rename(columns={"seconds": "inaccurate_tracking_seconds"})
mix_s  = sum_seconds(mix).rename(columns={"seconds": "mix_behaviors_seconds"})

merged = inac_s.merge(mix_s, on=["Animal", "Condition", "Experiment"], how="outer").fillna(0)
merged["total_excluded_seconds"] = merged["inaccurate_tracking_seconds"] + merged["mix_behaviors_seconds"]
merged["group"] = merged["Condition"]
merged["animal_id"] = merged["Animal"].astype(str)
merged["experiment"] = merged["Experiment"].astype(str)

out = merged[["animal_id", "group", "experiment",
              "inaccurate_tracking_seconds", "mix_behaviors_seconds",
              "total_excluded_seconds"]]
out.to_csv(OUT, index=False)
print(f"Saved {len(out)} rows to {OUT}")

# Summary by group
for col in ["inaccurate_tracking_seconds", "mix_behaviors_seconds", "total_excluded_seconds"]:
    print(f"\n{col}:")
    print(out.groupby("group")[col].agg(["mean", "std", "count"]).round(2))
