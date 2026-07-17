"""Replicate tracking exclusion stat using same GEE approach as Fig 3A."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod import families

# Load per-timebin data
tc = pd.read_csv("D:/coping-dynamics-sequencing/data/source/cluster_timecourse_per_animal.csv")
tc["stress"]     = (tc["group"] == "ELS").astype(int)
tc["experiment"] = tc["experiment"].astype(str)
exp_vals = sorted(tc["experiment"].unique())
tc["exp_bin"] = (tc["experiment"] == exp_vals[1]).astype(int)

# Also load raw syllable data to get inaccurate tracking per timebin
raw = pd.read_csv("D:/coping-dynamics-sequencing/data/source/syllable_usage_per_timebin_30s.csv")
raw.columns = raw.columns.str.strip()
raw["animal_id"]  = raw["Animal"].astype(str).str.strip()
raw["group"]      = raw["Condition"].str.strip()
raw["experiment"] = raw["Experiment"].astype(str).str.strip()
raw["Syllable"]   = raw["Syllable"].astype(int)
raw["Percentage"] = pd.to_numeric(raw["Percentage"])
raw["time_bin"]   = pd.to_numeric(raw["Time Bin"])

INACCURATE = {2, 4, 8, 9, 22, 31, 32, 33}
MIX        = {7, 13, 17}
EXCLUDED   = {"48.6"}

raw = raw[~raw["animal_id"].isin(EXCLUDED)]
raw["exp_bin"] = (raw["experiment"] == sorted(raw["experiment"].unique())[1]).astype(int)

# Sum inaccurate tracking % per animal per timebin
inac = raw[raw["Syllable"].isin(INACCURATE)].groupby(
    ["animal_id","group","experiment","exp_bin","time_bin"])["Percentage"].sum().reset_index()

# Zero-fill missing bins
all_animals = raw[["animal_id","group","experiment","exp_bin"]].drop_duplicates()
all_bins    = pd.DataFrame({"time_bin": sorted(raw["time_bin"].unique())})
full_idx = all_animals.assign(_k=1).merge(all_bins.assign(_k=1), on="_k").drop(columns="_k")
inac_full = full_idx.merge(inac[["animal_id","time_bin","Percentage"]], on=["animal_id","time_bin"], how="left").fillna({"Percentage": 0.0})
inac_full["stress"] = (inac_full["group"] == "ELS").astype(int)
inac_full = inac_full.sort_values(["animal_id","time_bin"]).reset_index(drop=True)

print(f"Inaccurate tracking data: {len(inac_full)} rows, {inac_full['animal_id'].nunique()} animals")
print(f"Control mean pct: {inac_full[inac_full['stress']==0]['Percentage'].mean():.2f}%")
print(f"ELS mean pct:     {inac_full[inac_full['stress']==1]['Percentage'].mean():.2f}%")

ms_beta, ms_se, ms_z, ms_p = -2.341, 0.942, -2.486, 0.013

def compare(label, ms_val, gee_val, tol=0.001):
    diff = abs(gee_val - ms_val)
    tag = "MATCH" if diff <= tol else ("CLOSE" if diff <= 0.05 else "DIFFERS")
    print(f"  {label:<6}  paragraph={ms_val:>8.4f}  GEE={gee_val:>8.4f}  diff={diff:.4f}  [{tag}]")

print("\n=== GEE (Gaussian, Exchangeable, animal clustering, per-timebin) ===")
endog = inac_full["Percentage"]
exog  = sm.add_constant(inac_full[["stress","exp_bin"]])
try:
    m = GEE(endog, exog, groups=inac_full["animal_id"],
            family=families.Gaussian(), cov_struct=sm.cov_struct.Exchangeable())
    res = m.fit()
    b, se, z, p = res.params["stress"], res.bse["stress"], res.tvalues["stress"], res.pvalues["stress"]
    print(f"  stress: beta={b:.4f}  SE={se:.4f}  z={z:.4f}  p={p:.4f}")
    print(f"  experiment: beta={res.params['exp_bin']:.4f}  SE={res.bse['exp_bin']:.4f}")
    compare("beta", ms_beta, b); compare("SE", ms_se, se)
    compare("z",    ms_z,    z); compare("p",  ms_p,  p)
except Exception as e:
    print(f"  failed: {e}")

print("\n=== GEE (Gaussian, Independence, animal clustering, per-timebin) ===")
try:
    m2 = GEE(endog, exog, groups=inac_full["animal_id"],
             family=families.Gaussian(), cov_struct=sm.cov_struct.Independence())
    res2 = m2.fit()
    b, se, z, p = res2.params["stress"], res2.bse["stress"], res2.tvalues["stress"], res2.pvalues["stress"]
    print(f"  stress: beta={b:.4f}  SE={se:.4f}  z={z:.4f}  p={p:.4f}")
    compare("beta", ms_beta, b); compare("SE", ms_se, se)
    compare("z",    ms_z,    z); compare("p",  ms_p,  p)
except Exception as e:
    print(f"  failed: {e}")
