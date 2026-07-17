"""Try GEE with per-timebin data, animal_id as clustering variable."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod import families

tc_df = pd.read_csv("D:/coping-dynamics-sequencing/data/source/cluster_timecourse_per_animal.csv")
tc_df["stress"]     = (tc_df["group"] == "ELS").astype(int)
tc_df["experiment"] = tc_df["experiment"].astype(str)
exp_vals = sorted(tc_df["experiment"].unique())
tc_df["exp_bin"] = (tc_df["experiment"] == exp_vals[1]).astype(int)
# frames per bin: pct/100 * 750 frames (30s * 25fps)
tc_df["frames_in_bin"] = (tc_df["pct"] / 100.0 * 750).round().astype(int)

ms = {
    "Freeze": dict(beta=-0.204, SE=0.081, z=-2.510, p=0.012),
    "Sniff":  dict(beta=0.621,  SE=0.231, z=2.694,  p=0.007),
    "Turn":   dict(beta=0.101,  SE=0.032, z=3.099,  p=0.002),
}

def compare(label, ms_val, gee_val, tol=0.001):
    diff = abs(gee_val - ms_val)
    tag = "MATCH" if diff <= tol else ("CLOSE" if diff <= 0.05 else "DIFFERS")
    print(f"    {label:<6}  paragraph={ms_val:>8.4f}  GEE={gee_val:>8.4f}  diff={diff:.4f}  [{tag}]")

fam = families.NegativeBinomial(alpha=1.0)

for cl, expected in ms.items():
    sub = tc_df[tc_df["cluster"] == cl].copy().sort_values(["animal_id","time_s"]).reset_index(drop=True)
    print(f"\n=== {cl} (n_obs={len(sub)}, n_animals={sub['animal_id'].nunique()}) ===")

    endog = sub["frames_in_bin"]
    exog  = sm.add_constant(sub[["stress","exp_bin"]])

    try:
        m = GEE(endog, exog, groups=sub["animal_id"],
                family=fam, cov_struct=sm.cov_struct.Exchangeable())
        res = m.fit()
        b   = res.params["stress"]
        se  = res.bse["stress"]
        z   = res.tvalues["stress"]
        p   = res.pvalues["stress"]
        exp_b  = res.params["exp_bin"]
        exp_se = res.bse["exp_bin"]
        exp_z  = res.tvalues["exp_bin"]
        print(f"  stress:     beta={b:.4f}  SE={se:.4f}  z={z:.4f}  p={p:.4f}")
        print(f"  experiment: beta={exp_b:.4f}  SE={exp_se:.4f}  z={exp_z:.4f}")
        compare("beta", expected["beta"], b)
        compare("SE",   expected["SE"],   se)
        compare("z",    expected["z"],    z)
        compare("p",    expected["p"],    p)
    except Exception as e:
        print(f"  failed: {e}")
