"""Verify Figure 5C cluster-frequency stats.

5C is Figure 3A's model (GEE Negative Binomial on per-timebin frame counts,
groups=animal, exchangeable correlation, robust SE) extended to THREE groups:
Control / vulnerable ELS / resilient ELS.

Dual-reference fit (as in the original Frequency_indexes_bout_duration_resilience.py):
  - Control reference  -> Control/ELS  and  Control/Resilient contrasts
  - ELS reference      -> ELS/Resilient contrast

Run from repo root:
  python scripts/verify_fig5c_frequency.py
Writes results/statistical_reports/fig5c_frequency_gee.csv
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod import families

REPO = Path(__file__).resolve().parents[1]
FAM = families.NegativeBinomial(alpha=1.0)
CLUSTERS = ["Freeze", "Sniff", "Groom", "Turn", "Locomotion", "Climb", "Jump"]

# ---- data: Fig 3A inputs + resilience labels ----
tc = pd.read_csv(REPO / "data/source/cluster_timecourse_per_animal.csv")
tc["animal_id"] = tc.animal_id.astype(str)
tc["frames_in_bin"] = (tc.pct / 100.0 * 750).round().astype(int)
ds = pd.read_csv(REPO / "results/figure_data/figure_5_dynamics_scores.csv")
ds["animal"] = ds.animal.astype(str)
tc["g3"] = tc.animal_id.map(dict(zip(ds.animal, ds.group_ext)))   # Control / ELS / ELS resilient
exp_vals = sorted(tc.experiment.astype(str).unique())
tc["exp_bin"] = (tc.experiment.astype(str) == exp_vals[1]).astype(int)


def gee_fit(sub, ref):
    """GEE NB with `ref` as the reference group; returns fitted result."""
    others = [g for g in ["Control", "ELS", "ELS resilient"] if g != ref]
    cols = {}
    for g in others:
        cols["d_" + g.replace(" ", "_")] = (sub.g3 == g).astype(float)
    X = sub.assign(**cols)[list(cols) + ["exp_bin"]]
    X = sm.add_constant(X)
    res = GEE(sub.frames_in_bin, X, groups=sub.animal_id, family=FAM,
              cov_struct=sm.cov_struct.Exchangeable()).fit()
    return res


rows = []
for cl in CLUSTERS:
    sub = tc[tc.cluster == cl].sort_values(["animal_id", "time_s"]).reset_index(drop=True)
    ctrl_ref = gee_fit(sub, "Control")       # gives d_ELS, d_ELS_resilient (vs Control)
    els_ref = gee_fit(sub, "ELS")            # gives d_ELS_resilient (vs ELS)
    contrasts = [
        ("Control/ELS",       ctrl_ref, "d_ELS"),
        ("Control/Resilient", ctrl_ref, "d_ELS_resilient"),
        ("ELS/Resilient",     els_ref,  "d_ELS_resilient"),
    ]
    for name, res, key in contrasts:
        rows.append({
            "cluster": cl, "contrast": name,
            "beta": res.params[key], "SE": res.bse[key],
            "z": res.tvalues[key], "p": res.pvalues[key],
            "significant": res.pvalues[key] < 0.05,
        })

out = pd.DataFrame(rows)
out_path = REPO / "results/statistical_reports/fig5c_frequency_gee.csv"
out.to_csv(out_path, index=False)

pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print("Fig 5C frequency — GEE NB (Fig 3A recipe, 3 groups), dual-reference:\n")
for cl in CLUSTERS:
    block = out[out.cluster == cl]
    print(f"--- {cl} ---")
    for _, r in block.iterrows():
        flag = "*" if r.p < 0.05 else " "
        print(f"  {r.contrast:<18} beta={r.beta:>8.3f}  SE={r.SE:>6.3f}  z={r.z:>7.3f}  p={r.p:>8.4f} {flag}")
print(f"\nsaved -> {out_path}")
