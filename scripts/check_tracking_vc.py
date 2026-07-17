"""Test mixedlm with vc_formula={"Experiment":"1"} — same spec as Antonio_final.py BFL models."""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

raw = pd.read_csv("D:/coping-dynamics-sequencing/data/source/syllable_usage_per_timebin_30s.csv")
raw.columns = raw.columns.str.strip()
raw["animal_id"]  = raw["Animal"].astype(str).str.strip()
raw["group"]      = raw["Condition"].str.strip()
raw["experiment"] = raw["Experiment"].astype(str).str.strip()
raw["Syllable"]   = raw["Syllable"].astype(int)
raw["Percentage"] = pd.to_numeric(raw["Percentage"])
raw["time_bin"]   = pd.to_numeric(raw["Time Bin"])
raw = raw[~raw["animal_id"].isin({"48.6"})]

INACCURATE = {2, 4, 8, 9, 22, 31, 32, 33}

# Per-animal total inaccurate tracking %
inac = raw[raw["Syllable"].isin(INACCURATE)].groupby(
    ["animal_id","group","experiment"])["Percentage"].sum().reset_index()
inac["pct"] = inac["Percentage"] / 15  # mean across 15 bins
inac["Condition"] = inac["group"]
inac["Animal"]    = inac["animal_id"]

ms_beta, ms_se, ms_z, ms_p = -2.341, 0.942, -2.486, 0.013

def show(name, params, bse, pvalues, key="Condition[T.ELS]"):
    # Try different key names
    for k in [key, "group[T.ELS]", "C(group)[T.ELS]", "Condition[T.ELS]"]:
        if k in params.index:
            b, se, p = params[k], bse[k], pvalues[k]
            z = b/se
            tag_b  = "MATCH" if abs(b-ms_beta)<=0.002 else f"DIFF({b:.4f})"
            tag_se = "MATCH" if abs(se-ms_se)<=0.002 else f"DIFF({se:.4f})"
            tag_z  = "MATCH" if abs(z-ms_z)<=0.01   else f"DIFF({z:.4f})"
            tag_p  = "MATCH" if abs(p-ms_p)<=0.002   else f"DIFF({p:.4f})"
            print(f"  [{name}] key={k}")
            print(f"    beta={b:.4f}[{tag_b}]  SE={se:.4f}[{tag_se}]  z={z:.4f}[{tag_z}]  p={p:.4f}[{tag_p}]")
            return
    print(f"  [{name}] key not found — available: {list(params.index)}")

print("=== Per-animal inaccurate tracking % ===")
print(f"n animals: {inac['animal_id'].nunique()}")
print(f"Control mean: {inac[inac['group']=='Control']['pct'].mean():.2f}%")
print(f"ELS mean:     {inac[inac['group']=='ELS']['pct'].mean():.2f}%")

# Model 1: groups=Animal, vc_formula={"Experiment": "1"}, reml=False (same as Antonio_final.py)
print("\n--- Model 1: Score~Condition+Experiment | groups=Animal, vc=Experiment, reml=False ---")
try:
    m = smf.mixedlm("pct ~ Condition + experiment", inac,
                    groups=inac["Animal"],
                    vc_formula={"experiment": "0+C(experiment)"}).fit(reml=False)
    show("vc_formula Experiment reml=F", m.params, m.bse, m.pvalues)
    print(f"    params: {dict(zip(m.params.index, m.params.values.round(4)))}")
except Exception as e:
    print(f"  failed: {e}")

# Model 2: groups=Animal, vc_formula={"Experiment": "1"}, reml=True
print("\n--- Model 2: same but reml=True ---")
try:
    m2 = smf.mixedlm("pct ~ Condition + experiment", inac,
                     groups=inac["Animal"],
                     vc_formula={"experiment": "0+C(experiment)"}).fit(reml=True)
    show("vc_formula Experiment reml=T", m2.params, m2.bse, m2.pvalues)
except Exception as e:
    print(f"  failed: {e}")

# Model 3: just groups=Animal (no vc), reml=False — per-animal
print("\n--- Model 3: pct~Condition | groups=Animal, no vc, reml=False ---")
try:
    m3 = smf.mixedlm("pct ~ Condition", inac,
                     groups=inac["Animal"]).fit(reml=False)
    show("simple reml=F", m3.params, m3.bse, m3.pvalues)
except Exception as e:
    print(f"  failed: {e}")

# Model 4: pct~Condition+experiment, groups=Animal, no vc, reml=False
print("\n--- Model 4: pct~Condition+experiment | groups=Animal, no vc, reml=False ---")
try:
    m4 = smf.mixedlm("pct ~ Condition + experiment", inac,
                     groups=inac["Animal"]).fit(reml=False)
    show("Condition+exp reml=F", m4.params, m4.bse, m4.pvalues)
except Exception as e:
    print(f"  failed: {e}")

# Model 5: pct~Condition, groups=Animal, no vc, reml=True
print("\n--- Model 5: pct~Condition | groups=Animal, reml=True ---")
try:
    m5 = smf.mixedlm("pct ~ Condition", inac,
                     groups=inac["Animal"]).fit(reml=True)
    show("Condition reml=T", m5.params, m5.bse, m5.pvalues)
except Exception as e:
    print(f"  failed: {e}")

# Also try per-timebin data with vc_formula
print("\n=== Per-timebin inaccurate tracking (n=1230) ===")
inac_tb = raw[raw["Syllable"].isin(INACCURATE)].groupby(
    ["animal_id","group","experiment","time_bin"])["Percentage"].sum().reset_index()
all_animals = raw[["animal_id","group","experiment"]].drop_duplicates()
all_bins    = pd.DataFrame({"time_bin": sorted(raw["time_bin"].unique())})
full = all_animals.assign(_k=1).merge(all_bins.assign(_k=1),on="_k").drop(columns="_k")
inac_tb_full = full.merge(inac_tb[["animal_id","time_bin","Percentage"]],
                          on=["animal_id","time_bin"],how="left").fillna({"Percentage":0.0})
inac_tb_full["Condition"] = inac_tb_full["group"]
inac_tb_full["Animal"]    = inac_tb_full["animal_id"]
inac_tb_full["time_s"]    = inac_tb_full["time_bin"] * 30

print(f"n rows: {len(inac_tb_full)}, n animals: {inac_tb_full['animal_id'].nunique()}")

# Model 6: timecourse LMM (same spec as Fig 3B-H) but on inaccurate tracking
print("\n--- Model 6: pct~Condition*time_s | groups=Animal, reml=True (timecourse LMM) ---")
try:
    m6 = smf.mixedlm("Percentage ~ Condition * time_s", inac_tb_full,
                     groups=inac_tb_full["Animal"]).fit(reml=True)
    show("LMM timecourse reml=T", m6.params, m6.bse, m6.pvalues, key="Condition[T.ELS]")
    # Also check main effect at time=0
    for k in m6.params.index:
        if "condition" in k.lower() or "els" in k.lower():
            print(f"    {k}: β={m6.params[k]:.4f}, SE={m6.bse[k]:.4f}, p={m6.pvalues[k]:.4f}")
except Exception as e:
    print(f"  failed: {e}")
