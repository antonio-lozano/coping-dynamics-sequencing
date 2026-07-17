"""Compute correct per-animal cluster frequency and timecourse from raw syllable CSV."""
import pandas as pd
from pathlib import Path

REPO = Path(__file__).parent.parent
SRC  = REPO / "data" / "source" / "syllable_usage_per_timebin_30s.csv"
AG   = REPO / "data" / "source" / "animal_groups.csv"

OUT_FREQ = REPO / "data" / "source" / "cluster_frequency_per_animal.csv"
OUT_TIME = REPO / "data" / "source" / "cluster_timecourse_per_animal.csv"

CLUSTER_MAP = {
    "Freeze":     [0, 28],
    "Sniff":      [18, 20],
    "Groom":      [24],
    "Turn":       [1, 3, 5, 6, 10, 15, 26, 27],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climb":      [111],
    "Jump":       [23, 29, 30, 34],
}
BIN_SECONDS = 30
EXCLUDED_ANIMALS = {"48.6"}

# Build syllable → cluster map
syll_to_cluster = {}
for cl, sylls in CLUSTER_MAP.items():
    for s in sylls:
        syll_to_cluster[s] = cl

df = pd.read_csv(SRC)
df.columns = df.columns.str.strip()
df["Syllable"] = df["Syllable"].astype(int)
df["time_bin"] = pd.to_numeric(df["Time Bin"])
df["Percentage"] = pd.to_numeric(df["Percentage"])

# Parse animal_id: "11.4" → "11.4"
df["animal_id"] = df["Animal"].astype(str).str.strip()
df["group"] = df["Condition"].str.strip()
df["experiment"] = df["Experiment"].astype(str).str.strip()

# Filter excluded
df = df[~df["animal_id"].isin(EXCLUDED_ANIMALS)]

# Assign cluster
df["cluster"] = df["Syllable"].map(syll_to_cluster)
df = df[df["cluster"].notna()].copy()

# ── Cluster FREQUENCY (total seconds per animal per cluster) ──────────────────
# Correct formula: sum(pct/100 * 30) across all bins and syllables in cluster
freq = (
    df.groupby(["animal_id", "group", "experiment", "cluster"], as_index=False)
    .apply(lambda g: pd.Series({"frequency_seconds": (g["Percentage"] / 100.0 * BIN_SECONDS).sum()}), include_groups=False)
    .reset_index(drop=True)
)
freq.to_csv(OUT_FREQ, index=False)
print(f"Cluster frequency: {len(freq)} rows → {OUT_FREQ.name}")

# Verify against source_data_figure3.csv values
print("\nFrequency means by group (verification):")
print(freq.groupby(["cluster", "group"])["frequency_seconds"].mean().round(2).unstack())

# ── Cluster TIMECOURSE (pct per animal per timebin per cluster) ───────────────
# Group syllables into clusters per animal-timebin, correct bin label from actual time_bin
clust_time = (
    df.groupby(["animal_id", "group", "experiment", "cluster", "time_bin"], as_index=False)
    ["Percentage"].sum()
    .rename(columns={"Percentage": "pct"})
)

# time_s: actual seconds from time_bin value
# time_bin values in the CSV: 0.0, 1.0, ..., 14.0 (0-indexed) OR 1.0, 2.0,...?
print("\ntime_bin range:", df["time_bin"].min(), "to", df["time_bin"].max())
print("Unique time_bins:", sorted(df["time_bin"].unique())[:5], "...")

# time_bin values are already in seconds (0, 30, 60, ..., 420)
# END of bin = time_bin + 30 → gives 30, 60, ..., 450 (matching s0s28_timecourse_per_animal.csv)
clust_time["time_s"] = clust_time["time_bin"] + BIN_SECONDS

# Fill zeros for ALL 15 bins per animal per cluster
all_animals = df[["animal_id", "group", "experiment"]].drop_duplicates()
all_time_s = pd.DataFrame({"time_s": [(b + 1) * BIN_SECONDS for b in range(15)]})
all_clusters = pd.DataFrame({"cluster": list(CLUSTER_MAP.keys())})

# Cartesian product: every animal × cluster × time_s
index = (all_animals.assign(_k=1)
         .merge(all_clusters.assign(_k=1), on="_k")
         .merge(all_time_s.assign(_k=1), on="_k")
         .drop(columns="_k"))
index["time_bin"] = index["time_s"] - BIN_SECONDS

# Merge with actual data, fill NA → 0
clust_time_full = index.merge(
    clust_time[["animal_id", "cluster", "time_s", "pct"]],
    on=["animal_id", "cluster", "time_s"],
    how="left"
).fillna({"pct": 0.0})

clust_time_full.to_csv(OUT_TIME, index=False)
print(f"\nCluster timecourse (zero-filled): {len(clust_time_full)} rows → {OUT_TIME.name}")
print("time_s range:", clust_time_full["time_s"].min(), "to", clust_time_full["time_s"].max())
print("Bins per animal per cluster (all should be 15):")
print(clust_time_full.groupby(["cluster", "animal_id"])["time_s"].count().value_counts())
