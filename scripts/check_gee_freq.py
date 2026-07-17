import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod import families

freq_df = pd.read_csv("D:/coping-dynamics-sequencing/data/source/cluster_frequency_per_animal.csv")
freq_df["stress"] = (freq_df["group"] == "ELS").astype(int)
freq_df["freq_frames"] = (freq_df["frequency_seconds"] * 25).round().astype(int)
freq_df["experiment"] = freq_df["experiment"].astype(str)

ms = {
    "Freeze":     dict(beta=-0.204, SE=0.081, z=-2.510, p=0.012),
    "Sniff":      dict(beta=0.621,  SE=0.231, z=2.694,  p=0.007),
    "Turn":       dict(beta=0.101,  SE=0.032, z=3.099,  p=0.002),
}

def compare(label, ms_val, gee_val, tol=0.001):
    diff = abs(gee_val - ms_val)
    tag = "MATCH" if diff <= tol else ("CLOSE" if diff <= 0.05 else "DIFFERS")
    print(f"    {label:<6}  paragraph={ms_val:>8.4f}  GEE={gee_val:>8.4f}  diff={diff:.4f}  [{tag}]")

for cl, expected in ms.items():
    sub = freq_df[freq_df["cluster"] == cl].copy().reset_index(drop=True)
    print(f"\n=== {cl} (n={len(sub)}) ===")

    # GEE NB with experiment as groups (clustering) and stress + experiment as covariates
    fam = families.NegativeBinomial()
    endog = sub["freq_frames"]
    exog  = sm.add_constant(sub[["stress", "experiment"]].assign(
        experiment=sub["experiment"].astype("category").cat.codes
    ))

    try:
        m = GEE(endog, exog, groups=sub["experiment"], family=fam, cov_struct=sm.cov_struct.Exchangeable())
        res = m.fit()
        print(res.summary().tables[1])
        b  = res.params["stress"]
        se = res.bse["stress"]
        z  = res.tvalues["stress"]
        p  = res.pvalues["stress"]
        print(f"  stress: beta={b:.4f}  SE={se:.4f}  z={z:.4f}  p={p:.4f}")
        compare("beta", expected["beta"], b)
        compare("SE",   expected["SE"],   se)
        compare("z",    expected["z"],    z)
        compare("p",    expected["p"],    p)
    except Exception as e:
        print(f"  GEE failed: {e}")
