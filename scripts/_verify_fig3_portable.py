"""Portable Fig 3 verification: 3A GEE NB (cluster frequency) + 3B-H timecourse LMM.
Run from repo root: python scripts/_verify_fig3_portable.py
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod import families

REPO = Path(__file__).resolve().parents[1]
tc = pd.read_csv(REPO / "data/source/cluster_timecourse_per_animal.csv")
tc = tc[tc.group.isin(["Control", "ELS"])].copy()

# ---------- Fig 3A: GEE Negative Binomial ----------
tc["stress"] = (tc.group == "ELS").astype(int)
exp_vals = sorted(tc.experiment.astype(str).unique())
tc["exp_bin"] = (tc.experiment.astype(str) == exp_vals[1]).astype(int)
tc["frames_in_bin"] = (tc.pct / 100.0 * 750).round().astype(int)
fam = families.NegativeBinomial(alpha=1.0)

MS_3A = {"Freeze": (-0.204, 0.081, -2.510, 0.012),
         "Sniff":  (0.621, 0.231, 2.694, 0.007),
         "Turn":   (0.101, 0.032, 3.099, 0.002)}
print("=== Fig 3A  GEE NB (stress effect) ===")
print(f"{'cluster':<11}{'beta':>9}{'SE':>9}{'z':>9}{'p':>9}   manuscript")
for cl in ["Freeze", "Sniff", "Turn", "Locomotion", "Climb", "Groom", "Jump"]:
    sub = tc[tc.cluster == cl].sort_values(["animal_id", "time_s"]).reset_index(drop=True)
    exog = sm.add_constant(sub[["stress", "exp_bin"]])
    res = GEE(sub.frames_in_bin, exog, groups=sub.animal_id, family=fam,
              cov_struct=sm.cov_struct.Exchangeable()).fit()
    b, se, z, p = res.params.stress, res.bse.stress, res.tvalues.stress, res.pvalues.stress
    ref = f"  ms={MS_3A[cl]}" if cl in MS_3A else ""
    print(f"{cl:<11}{b:>9.4f}{se:>9.4f}{z:>9.4f}{p:>9.4f}{ref}")

# ---------- Fig 3B-H: timecourse LMM ----------
print("\n=== Fig 3B-H  LMM  pct ~ group*time_s  (ELS x time interaction) ===")
MS_3BH = {"Freeze": (-0.023, 0.005, -4.262), "Sniff": (-0.014, 0.004, -3.656),
          "Turn": (0.026, 0.007, 3.610), "Locomotion": (0.005, 0.002, 2.553)}
tc["grp"] = pd.Categorical(tc.group, categories=["Control", "ELS"])
print(f"{'cluster':<11}{'beta':>9}{'SE':>9}{'z':>9}{'p':>11}   manuscript")
for cl, panel in [("Freeze", "3B"), ("Sniff", "3C"), ("Turn", "3E"), ("Locomotion", "3F")]:
    sub = tc[tc.cluster == cl].copy()
    m = smf.mixedlm("pct ~ grp * time_s", sub, groups=sub.animal_id).fit(reml=True)
    k = "grp[T.ELS]:time_s"
    print(f"{cl:<11}{m.params[k]:>9.4f}{m.bse[k]:>9.4f}{m.tvalues[k]:>9.4f}{m.pvalues[k]:>11.2e}"
          f"   {panel} ms={MS_3BH[cl]}")
