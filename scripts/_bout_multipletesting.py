"""Bout-duration MixedLM per behavior + multiple-testing correction.
Uniform reproducible model: bout_duration ~ group + Experiment, groups=Animal, reml=False.
Then Bonferroni and Benjamini-Hochberg (FDR) over the behavior family.
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
b = pd.read_csv(REPO / "results/statistical_reports/fig4_bout_cluster_per_animal.csv")

rows = []
for cl in ["Freezing", "Sniffing", "Turn", "Locomotion", "Climbing", "Jump", "Grooming"]:
    d = b[b.cluster == cl].copy()
    d["grp"] = pd.Categorical(d.group, categories=["Control", "ELS"])
    d["Experiment"] = d.Experiment.astype(int)
    m = smf.mixedlm("bout_duration ~ grp + Experiment", d, groups=d.Animal).fit(reml=False)
    k = "grp[T.ELS]"
    rows.append({"behavior": cl, "n": d.Animal.nunique(),
                 "beta": m.params[k], "SE": m.bse[k], "z": m.tvalues[k], "p_raw": m.pvalues[k]})
r = pd.DataFrame(rows)

def correct(df, m_label, family):
    sub = df[df.behavior.isin(family)].copy()
    p = sub.p_raw.values
    sub["p_bonf"] = np.minimum(p * len(family), 1.0)
    sub["p_BH"] = multipletests(p, method="fdr_bh")[1]
    print(f"\n===== Correction over {len(family)} behaviors ({m_label}) =====")
    print(f"{'behavior':<11}{'beta':>9}{'SE':>9}{'z':>8}{'p_raw':>9}{'p_Bonf':>9}{'p_BH':>9}  sig(BH)")
    for _, x in sub.sort_values('p_raw').iterrows():
        flag = "YES" if x.p_BH < 0.05 else "no"
        print(f"{x.behavior:<11}{x.beta:>9.4f}{x.SE:>9.4f}{x.z:>8.3f}"
              f"{x.p_raw:>9.4f}{x.p_bonf:>9.4f}{x.p_BH:>9.4f}   {flag}")

# 6-behavior family (exclude sparse Grooming, n=48)
fam6 = ["Freezing", "Sniffing", "Turn", "Locomotion", "Climbing", "Jump"]
fam7 = fam6 + ["Grooming"]
correct(r, "excl. Grooming", fam6)
correct(r, "all clusters", fam7)
