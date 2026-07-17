"""Verify Fig 5 (resilience) stats: 5C 3-group frequency GEE NB, 5D-J timecourse LMM.
Groups: Control, ELS (vulnerable), ELS resilient. Reference = vulnerable ELS.
No gold report sheet exists locally for Fig 5 -> manuscript vs recompute only.
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod import families

REPO = Path(__file__).resolve().parents[1]
ds = pd.read_csv(REPO / "results/figure_data/figure_5_dynamics_scores.csv")
lab = ds[["animal", "group_ext"]].copy()
lab["animal"] = lab.animal.astype(str)

tc = pd.read_csv(REPO / "data/source/cluster_timecourse_per_animal.csv")
tc["animal_id"] = tc.animal_id.astype(str)
tc = tc.merge(lab, left_on="animal_id", right_on="animal", how="inner")
# 3-level group, reference = vulnerable "ELS"
tc["g3"] = pd.Categorical(tc.group_ext, categories=["ELS", "Control", "ELS resilient"])
tc["frames"] = (tc.pct / 100.0 * 750).round().astype(int)
fam = families.NegativeBinomial(alpha=1.0)

print("group_ext rows:", dict(tc.drop_duplicates('animal_id').group_ext.value_counts()))
print("\n=== Fig 5C  frequency GEE NB (ref = vulnerable ELS) ===")
print("manuscript: Freeze resVSvuln b=0.180/z=4.681; resVSctrl b=0.067/z=0.604/p=.546")
print(f"{'cluster':<11}{'contrast':<16}{'beta':>9}{'SE':>9}{'z':>8}{'p':>9}")
MS = {"Freeze":"resVSvuln 0.180/0.039/4.681; resVSctrl 0.067/0.111/0.604",
      "Sniff":"0.492/0.119/4.139","Groom":"-0.398/0.187/-2.122",
      "Locomotion":"-0.425/0.089/-4.755","Climb":"-0.196/0.075/-2.624"}
for cl in ["Freeze","Sniff","Groom","Turn","Locomotion","Climb","Jump"]:
    s = tc[tc.cluster==cl].sort_values(["animal_id","time_s"]).reset_index(drop=True)
    X = pd.get_dummies(s.g3, drop_first=True).astype(float)
    X = sm.add_constant(X)
    try:
        res = GEE(s.frames, X, groups=s.animal_id, family=fam,
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
        for c in ["group_ext[Control]" if False else "Control","ELS resilient"]:
            pass
        for name in [c for c in X.columns if c!="const"]:
            b,se,z,p = res.params[name],res.bse[name],res.tvalues[name],res.pvalues[name]
            print(f"{cl:<11}{name:<16}{b:>9.4f}{se:>9.4f}{z:>8.3f}{p:>9.4f}")
    except Exception as e:
        print(f"{cl}: FAIL {e}")
    if cl in MS: print(f"   ms-> {MS[cl]}")

print("\n=== Fig 5D  Freeze timecourse LMM (ref = vulnerable ELS) ===")
print("manuscript: res vs vuln b=-1.865/SE0.249/z=-7.484; res vs ctrl b=0.619/SE0.238/z=2.599")
for cl in ["Freeze","Turn","Sniff"]:
    s = tc[tc.cluster==cl].copy()
    m = smf.mixedlm("pct ~ g3 * time_s", s, groups=s.animal_id).fit(reml=True)
    print(f"-- {cl} interaction terms --")
    for k in m.params.index:
        if ":time_s" in k and "g3" in k:
            print(f"   {k:<34} b={m.params[k]:>8.3f} SE={m.bse[k]:>6.3f} z={m.tvalues[k]:>7.3f} p={m.pvalues[k]:.4f}")
