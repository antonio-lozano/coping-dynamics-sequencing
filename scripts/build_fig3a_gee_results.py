"""
Run GEE Negative Binomial for all 7 clusters (Fig 3A) and save results to CSV.

Model: GEE NB(alpha=1.0), exchangeable correlation, animal_id grouping.
Response: frames_in_bin = pct/100 * 750  (per-timebin frame count proxy)
Predictors: Intercept, stress (0=Control, 1=ELS), exp_bin (0=Exp1, 1=Exp3)

Manuscript (Fig 3A):
  Freeze: beta=-0.204, SE=0.081, z=-2.510, p=0.012
  Sniff:  beta=+0.621, SE=0.231, z=2.694,  p=0.007
  Turn:   beta=+0.101, SE=0.032, z=3.099,  p=0.002

Output: results/statistical_reports/gee_freq_results.csv
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod import families

SOURCE = "D:/coping-dynamics-sequencing/data/source"
OUT    = "D:/coping-dynamics-sequencing/results/statistical_reports/gee_freq_results.csv"

tc = pd.read_csv(f"{SOURCE}/cluster_timecourse_per_animal.csv")
tc["stress"]  = (tc["group"] == "ELS").astype(int)
exp_vals      = sorted(tc["experiment"].astype(str).unique())
tc["exp_bin"] = (tc["experiment"].astype(str) == exp_vals[1]).astype(int)
# convert pct to frame count (30s bin × 25fps = 750 frames)
tc["frames_in_bin"] = (tc["pct"] / 100.0 * 750).round().astype(int)

fam = families.NegativeBinomial(alpha=1.0)

CLUSTERS = ["Freeze", "Sniff", "Turn", "Locomotion", "Climb", "Groom", "Jump"]

MS_REF = {
    "Freeze":     dict(beta=-0.204, SE=0.081, z=-2.510, p=0.012),
    "Sniff":      dict(beta=0.621,  SE=0.231, z=2.694,  p=0.007),
    "Turn":       dict(beta=0.101,  SE=0.032, z=3.099,  p=0.002),
}

rows = []
for cl in CLUSTERS:
    sub = (tc[tc["cluster"] == cl]
           .copy()
           .sort_values(["animal_id", "time_s"])
           .reset_index(drop=True))

    n_obs     = len(sub)
    n_animals = sub["animal_id"].nunique()
    n_ctrl    = (sub[sub["time_s"] == sub["time_s"].min()]["stress"] == 0).sum()
    n_els     = (sub[sub["time_s"] == sub["time_s"].min()]["stress"] == 1).sum()

    endog = sub["frames_in_bin"]
    exog  = sm.add_constant(sub[["stress", "exp_bin"]])

    result = {"cluster": cl, "n_obs": n_obs, "n_animals": n_animals,
              "n_control": int(n_ctrl), "n_els": int(n_els)}

    try:
        m   = GEE(endog, exog, groups=sub["animal_id"],
                  family=fam, cov_struct=sm.cov_struct.Exchangeable())
        res = m.fit()
        for param in ["const", "stress", "exp_bin"]:
            result[f"{param}_beta"] = round(float(res.params[param]), 4)
            result[f"{param}_SE"]   = round(float(res.bse[param]),    4)
            result[f"{param}_z"]    = round(float(res.tvalues[param]), 4)
            result[f"{param}_p"]    = round(float(res.pvalues[param]), 6)
        result["converged"] = True

        b, se, z, p = result["stress_beta"], result["stress_SE"], result["stress_z"], result["stress_p"]
        print(f"\n{cl} (n={n_animals} animals, {n_obs} obs)")
        print(f"  stress: beta={b:.4f}  SE={se:.4f}  z={z:.4f}  p={p:.4f}")
        if cl in MS_REF:
            ref = MS_REF[cl]
            for k, ms_val, got in [("beta", ref["beta"], b), ("SE", ref["SE"], se),
                                    ("z", ref["z"], z), ("p", ref["p"], p)]:
                diff = abs(got - ms_val)
                tag  = "✓" if diff < 0.002 else ("~" if diff < 0.05 else "✗")
                print(f"    {k}: manuscript={ms_val}  R={got:.4f}  {tag}")

    except Exception as e:
        result["converged"] = False
        print(f"\n{cl}: FAILED — {e}")

    rows.append(result)

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT, index=False)
print(f"\nSaved GEE results → {OUT}")
print(df_out[["cluster","stress_beta","stress_SE","stress_z","stress_p"]].to_string(index=False))
