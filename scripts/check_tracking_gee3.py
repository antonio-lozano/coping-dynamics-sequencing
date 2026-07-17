"""Try experiment as clustering variable for tracking exclusions."""
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
inac_full = inac_full.sort_values(["experiment","animal_id","time_bin"]).reset_index(drop=True)

# Also per-animal total (n=82)
per_animal = inac_full.groupby(["animal_id","group","experiment","exp_bin","stress"])["Percentage"].sum().reset_index()
per_animal["pct_450"] = per_animal["Percentage"] / 15  # average across bins = pct of total
# Or total seconds: pct/100*30 per bin summed
per_animal["pct_total"] = inac_full.groupby("animal_id")["Percentage"].sum().values / 15

ms_beta, ms_se, ms_z, ms_p = -2.341, 0.942, -2.486, 0.013

def show(name, b, se, z, p):
    tag_b  = "MATCH" if abs(b-ms_beta)<=0.002 else "CLOSE" if abs(b-ms_beta)<=0.05 else "DIFF"
    tag_se = "MATCH" if abs(se-ms_se)<=0.002 else "CLOSE" if abs(se-ms_se)<=0.05 else "DIFF"
    tag_z  = "MATCH" if abs(z-ms_z)<=0.01 else "CLOSE" if abs(z-ms_z)<=0.1 else "DIFF"
    tag_p  = "MATCH" if abs(p-ms_p)<=0.002 else "CLOSE" if abs(p-ms_p)<=0.05 else "DIFF"
    print(f"  [{name}]")
    print(f"    beta={b:.4f}[{tag_b}]  SE={se:.4f}[{tag_se}]  z={z:.4f}[{tag_z}]  p={p:.4f}[{tag_p}]")

# GEE with experiment as clustering (2 clusters), per-timebin
endog = inac_full["Percentage"]
exog  = sm.add_constant(inac_full[["stress","exp_bin"]])
for cov_name, cov_struct, cov_type in [
    ("Exch ROBUST", sm.cov_struct.Exchangeable(), "robust"),
    ("Exch NAIVE",  sm.cov_struct.Exchangeable(), "naive"),
    ("Indep NAIVE", sm.cov_struct.Independence(), "naive"),
]:
    try:
        m = GEE(endog, exog, groups=inac_full["experiment"],
                family=families.Gaussian(), cov_struct=cov_struct)
        res = m.fit(cov_type=cov_type)
        show(f"GEE Gaussian exp-cluster {cov_name} (n=1230)",
             res.params["stress"], res.bse["stress"], res.tvalues["stress"], res.pvalues["stress"])
    except Exception as e:
        print(f"  [{cov_name}] failed: {e}")

# Per-animal LMM-style: simple OLS with experiment dummy
from statsmodels.formula.api import ols
print()
for yvar, ylab in [("pct_total", "pct_total (mean per bin)"), ("pct_450", "pct_450")]:
    m = ols(f"{yvar} ~ stress + exp_bin", data=per_animal).fit()
    b, se = m.params["stress"], m.bse["stress"]
    z, p = b/se, m.pvalues["stress"]
    show(f"OLS per-animal {ylab}", b, se, z, p)
