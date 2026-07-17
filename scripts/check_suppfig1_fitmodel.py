"""Run fit_mixed_models on Supp Fig 1 clusters (same pipeline as figure_3.py)."""
import sys
import warnings
import pandas as pd
sys.path.insert(0, "d:/coping-dynamics-sequencing")
from src.statistics import fit_mixed_models
warnings.filterwarnings("ignore")

# Load Supp_cluster_time sheet from Raw_data.xlsx
try:
    wide = pd.read_excel(
        "d:/coping-dynamics-sequencing/data/Raw_data.xlsx",
        sheet_name="Supp_cluster_time", header=2
    )
    wide = wide.dropna(subset=["animal_id", "group", "cluster"]).copy()
    time_cols = [c for c in wide.columns if str(c).startswith("t_")]
    long = wide.melt(
        id_vars=["animal_id", "group", "project", "experiment", "cluster"],
        value_vars=time_cols, var_name="time_bin", value_name="percentage"
    )
    long["time_bin_numeric"] = long["time_bin"].str.extract(r"(\d+)").astype(int)
    long["percentage"] = pd.to_numeric(long["percentage"], errors="coerce")
    long = long[long["group"].isin(["Control", "ELS"])].copy()
    print("Source: Raw_data.xlsx Supp_cluster_time")
    print(f"Shape: {long.shape}  Animals: {long['animal_id'].nunique()}")
    print(f"Clusters: {sorted(long['cluster'].unique())}")
    print()
except Exception as e:
    print(f"Could not load Supp_cluster_time: {e}")
    print("Falling back to syllable_usage CSV...")
    raw = pd.read_csv("d:/coping-dynamics-sequencing/data/source/syllable_usage_per_timebin_30s.csv")
    raw.columns = raw.columns.str.strip()
    raw["animal_id"]  = raw["Animal"].astype(str).str.strip()
    raw["group"]      = raw["Condition"].str.strip()
    raw["experiment"] = raw["Experiment"].astype(int)
    raw["Syllable"]   = raw["Syllable"].astype(int)
    raw["Percentage"] = pd.to_numeric(raw["Percentage"])
    raw["time_bin_numeric"] = pd.to_numeric(raw["Time Bin"])
    raw = raw[raw["animal_id"] != "48.6"]
    INAC = {2, 4, 8, 9, 22, 31, 32, 33}
    MIX  = {7, 13, 17}
    frames = []
    for cl_name, sylls in [("Inaccurate tracking", INAC), ("Mix behaviors", MIX)]:
        sub = (raw[raw["Syllable"].isin(sylls)]
               .groupby(["animal_id","group","experiment","time_bin_numeric"])["Percentage"]
               .sum().reset_index()
               .assign(cluster=cl_name, percentage=lambda d: d["Percentage"]))
        frames.append(sub)
    long = pd.concat(frames, ignore_index=True)
    print(f"Shape: {long.shape}  Animals: {long['animal_id'].nunique()}")

print("Manuscript target: beta=-2.341, SE=0.942, z=-2.486, p=0.013")
print("="*70)

for cl in sorted(long["cluster"].unique()):
    sub = long[long["cluster"] == cl].copy()
    n_animals = sub["animal_id"].nunique()
    n_obs     = len(sub)
    print(f"\n--- {cl}  (n={n_animals} animals, {n_obs} obs) ---")
    try:
        results = fit_mixed_models(sub, "percentage")
        for analysis in ["Exp1", "Exp3", "Combined"]:
            rows = results[results["analysis"] == analysis]
            rows = rows[rows["parameter"].str.contains("ELS", na=False)]
            for _, r in rows.iterrows():
                tag = ""
                if abs(r["coef"] - (-2.341)) < 0.01:
                    tag = "  <-- beta MATCH"
                if abs(r["std_err"] - 0.942) < 0.01:
                    tag += "  SE MATCH!"
                print(f"  {analysis} {r['parameter']:35s}  beta={r['coef']:8.4f}  SE={r['std_err']:7.4f}  z={r['z']:8.4f}  p={r['p_value']:.4f}{tag}")
    except Exception as e:
        print(f"  ERROR: {e}")
