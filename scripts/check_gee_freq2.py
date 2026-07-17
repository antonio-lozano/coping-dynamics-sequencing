import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod import families

freq_df = pd.read_csv("D:/coping-dynamics-sequencing/data/source/cluster_frequency_per_animal.csv")
freq_df["stress"] = (freq_df["group"] == "ELS").astype(int)
freq_df["freq_frames"] = (freq_df["frequency_seconds"] * 25).round().astype(int)
freq_df["experiment"] = freq_df["experiment"].astype(str)

# Check experiments
print("Experiments:", sorted(freq_df["experiment"].unique()))
# Encode experiment as binary (0/1) since there are 2
exp_vals = sorted(freq_df["experiment"].unique())
freq_df["exp_bin"] = (freq_df["experiment"] == exp_vals[1]).astype(int)
print("exp_bin mapping:", dict(zip(exp_vals, [0,1])))

ms = {
    "Freeze": dict(beta=-0.204, SE=0.081, z=-2.510, p=0.012),
    "Sniff":  dict(beta=0.621,  SE=0.231, z=2.694,  p=0.007),
    "Turn":   dict(beta=0.101,  SE=0.032, z=3.099,  p=0.002),
}

def compare(label, ms_val, gee_val, tol=0.001):
    diff = abs(gee_val - ms_val)
    tag = "MATCH" if diff <= tol else ("CLOSE" if diff <= 0.05 else "DIFFERS")
    print(f"    {label:<6}  paragraph={ms_val:>8.4f}  GEE={gee_val:>8.4f}  diff={diff:.4f}  [{tag}]")

for cl, expected in ms.items():
    sub = freq_df[freq_df["cluster"] == cl].copy().reset_index(drop=True)
    print(f"\n=== {cl} (n={len(sub)}) ===")

    endog = sub["freq_frames"]
    exog  = sm.add_constant(sub[["stress", "exp_bin"]])

    fam = families.NegativeBinomial(alpha=1.0)
    m = GEE(endog, exog, groups=sub["experiment"],
            family=fam, cov_struct=sm.cov_struct.Exchangeable())
    res = m.fit()

    b  = res.params["stress"]
    se = res.bse["stress"]
    z  = res.tvalues["stress"]
    p  = res.pvalues["stress"]
    exp_b = res.params["exp_bin"]
    print(f"  stress:     beta={b:.4f}  SE={se:.4f}  z={z:.4f}  p={p:.4f}")
    print(f"  experiment: beta={exp_b:.4f}  SE={res.bse['exp_bin']:.4f}  z={res.tvalues['exp_bin']:.4f}")
    compare("beta", expected["beta"], b)
    compare("SE",   expected["SE"],   se)
    compare("z",    expected["z"],    z)
    compare("p",    expected["p"],    p)
