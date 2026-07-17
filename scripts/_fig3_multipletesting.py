"""Fig 3 multiple-testing correction.
3A: GEE NB stress effect (cluster frequency) across 7 clusters.
3B-H: timecourse LMM ELS x time interaction across 7 clusters.
Bonferroni + Benjamini-Hochberg (FDR) over the 7-cluster family for each.
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod import families
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
tc = pd.read_csv(REPO / "data/source/cluster_timecourse_per_animal.csv")
tc = tc[tc.group.isin(["Control", "ELS"])].copy()
CLUSTERS = ["Freeze", "Sniff", "Turn", "Locomotion", "Climb", "Groom", "Jump"]

# ---- 3A: GEE NB ----
tc["stress"] = (tc.group == "ELS").astype(int)
exp_vals = sorted(tc.experiment.astype(str).unique())
tc["exp_bin"] = (tc.experiment.astype(str) == exp_vals[1]).astype(int)
tc["frames"] = (tc.pct / 100.0 * 750).round().astype(int)
fam = families.NegativeBinomial(alpha=1.0)
a = []
for cl in CLUSTERS:
    s = tc[tc.cluster == cl].sort_values(["animal_id", "time_s"]).reset_index(drop=True)
    res = GEE(s.frames, sm.add_constant(s[["stress", "exp_bin"]]), groups=s.animal_id,
              family=fam, cov_struct=sm.cov_struct.Exchangeable()).fit()
    a.append({"cluster": cl, "beta": res.params.stress, "z": res.tvalues.stress, "p_raw": res.pvalues.stress})

# ---- 3B-H: timecourse LMM interaction ----
tc["grp"] = pd.Categorical(tc.group, categories=["Control", "ELS"])
d = []
for cl in CLUSTERS:
    s = tc[tc.cluster == cl].copy()
    m = smf.mixedlm("pct ~ grp * time_s", s, groups=s.animal_id).fit(reml=True)
    k = "grp[T.ELS]:time_s"
    d.append({"cluster": cl, "beta": m.params[k], "z": m.tvalues[k], "p_raw": m.pvalues[k]})

def report(title, rows):
    df = pd.DataFrame(rows)
    df["p_bonf"] = np.minimum(df.p_raw * len(df), 1.0)
    df["p_BH"] = multipletests(df.p_raw.values, method="fdr_bh")[1]
    print(f"\n===== {title} (family = {len(df)} clusters) =====")
    print(f"{'cluster':<11}{'beta':>9}{'z':>8}{'p_raw':>9}{'p_Bonf':>9}{'p_BH':>9}  sig_raw  sig_BH")
    for _, x in df.sort_values('p_raw').iterrows():
        print(f"{x.cluster:<11}{x.beta:>9.4f}{x.z:>8.3f}{x.p_raw:>9.4f}{x.p_bonf:>9.4f}{x.p_BH:>9.4f}"
              f"   {'Y' if x.p_raw<0.05 else '-':>3}     {'Y' if x.p_BH<0.05 else '-':>3}")

report("Fig 3A  cluster frequency (GEE NB, stress effect)", a)
report("Fig 3B-H  dynamics (LMM, ELS x time interaction)", d)
