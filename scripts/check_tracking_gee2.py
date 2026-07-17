"""Test different covariance types and model families for tracking exclusions."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod import families

raw = pd.read_csv("D:/coping-dynamics-sequencing/data/source/syllable_usage_per_timebin_30s.csv")
raw.columns = raw.columns.str.strip()
raw["animal_id"]  = raw["Animal"].astype(str).str.strip()
raw["group"]      = raw["Condition"].str.strip()
raw["experiment"] = raw["Experiment"].astype(str).str.strip()
raw["Syllable"]   = raw["Syllable"].astype(int)
raw["Percentage"] = pd.to_numeric(raw["Percentage"])
raw["time_bin"]   = pd.to_numeric(raw["Time Bin"])
raw = raw[~raw["animal_id"].isin({"48.6"})]
raw["exp_bin"] = (raw["experiment"] == sorted(raw["experiment"].unique())[1]).astype(int)
raw["stress"]  = (raw["group"] == "ELS").astype(int)

INACCURATE = {2, 4, 8, 9, 22, 31, 32, 33}
inac = raw[raw["Syllable"].isin(INACCURATE)].groupby(
    ["animal_id","group","experiment","exp_bin","stress","time_bin"])["Percentage"].sum().reset_index()

all_animals = raw[["animal_id","group","experiment","exp_bin","stress"]].drop_duplicates()
all_bins    = pd.DataFrame({"time_bin": sorted(raw["time_bin"].unique())})
inac_full = all_animals.assign(_k=1).merge(all_bins.assign(_k=1), on="_k").drop(columns="_k")
inac_full = inac_full.merge(inac[["animal_id","time_bin","Percentage"]], on=["animal_id","time_bin"], how="left").fillna({"Percentage": 0.0})
inac_full = inac_full.sort_values(["animal_id","time_bin"]).reset_index(drop=True)

ms_beta, ms_se, ms_z, ms_p = -2.341, 0.942, -2.486, 0.013

def compare(label, ms_val, gee_val, tol=0.001):
    diff = abs(gee_val - ms_val)
    tag = "MATCH" if diff <= tol else ("CLOSE" if diff <= 0.05 else "DIFFERS")
    print(f"  {label:<6}  ms={ms_val:>8.4f}  model={gee_val:>8.4f}  diff={diff:.4f}  [{tag}]")

endog = inac_full["Percentage"]
exog  = sm.add_constant(inac_full[["stress","exp_bin"]])

configs = [
    ("Gaussian Exchangeable ROBUST",  families.Gaussian(), sm.cov_struct.Exchangeable(), "robust"),
    ("Gaussian Exchangeable NAIVE",   families.Gaussian(), sm.cov_struct.Exchangeable(), "naive"),
    ("Gaussian Independence NAIVE",   families.Gaussian(), sm.cov_struct.Independence(), "naive"),
    ("Gamma Exchangeable ROBUST",     families.Gamma(link=sm.families.links.Log()), sm.cov_struct.Exchangeable(), "robust"),
    ("Gamma Exchangeable NAIVE",      families.Gamma(link=sm.families.links.Log()), sm.cov_struct.Exchangeable(), "naive"),
]

for name, fam, cov, cov_type in configs:
    try:
        m = GEE(endog, exog, groups=inac_full["animal_id"], family=fam, cov_struct=cov)
        res = m.fit(cov_type=cov_type)
        b, se, z, p = res.params["stress"], res.bse["stress"], res.tvalues["stress"], res.pvalues["stress"]
        tag_b = "MATCH" if abs(b-ms_beta)<=0.001 else "CLOSE" if abs(b-ms_beta)<=0.05 else "DIFF"
        tag_se = "MATCH" if abs(se-ms_se)<=0.001 else "CLOSE" if abs(se-ms_se)<=0.05 else "DIFF"
        print(f"\n[{name}]  beta={b:.4f}[{tag_b}]  SE={se:.4f}[{tag_se}]  z={z:.4f}  p={p:.4f}")
    except Exception as e:
        print(f"\n[{name}] FAILED: {e}")
