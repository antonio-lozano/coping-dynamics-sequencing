"""
Reproduce Fig 5D-J timecourse × resilience MixedLM stats.

Model: percent ~ C(group_ext, Treatment('ELS')) * time_bin_numeric + C(experiment)
       random intercept per animal
ELS is the reference group so:
  - group_ext[T.Control]                  = Control baseline vs ELS baseline
  - group_ext[T.ELS resilient]            = ELS resilient baseline vs ELS baseline
  - group_ext[T.Control]:time_bin_numeric = Control slope difference vs ELS
  - group_ext[T.ELS resilient]:time_bin_numeric = Resilient slope difference vs ELS

Clusters: Freeze (D), Sniff (E), Groom (F), Turn (G), Locomotion (H), Climb (I), Jump (J)
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

REPO = Path(__file__).resolve().parents[1]

ELS_RESILIENT = {
    "2.4", "15.4", "26.4", "43.3", "43.5", "49.6",
    "123.3", "134.2", "148.4", "150.3", "150.5", "159.4",
}

CLUSTERS = ["Freeze", "Sniff", "Groom", "Turn", "Locomotion", "Climb", "Jump"]
PANELS   = dict(zip(CLUSTERS, "DEFGHIJ"))

# Manuscript reference values (ELS reference, time interaction)
MANUSCRIPT = {
    "Freeze": {"time_x_resilient": (-1.865, 0.249, -7.484), "time_x_ctrl": (0.619, 0.238, 2.599),
               "baseline_resilient": None},
    "Sniff":  {"baseline_resilient": (-0.338, 0.176, -1.920), "time_x_resilient": None},
    "Turn":   {"time_x_resilient": (1.622, 0.326, 4.967), "baseline_resilient": None},
}

tc = pd.read_csv(REPO / "data/source/cluster_timecourse_per_animal.csv")
tc["animal_id"] = tc["animal_id"].astype(str)
tc["group_ext"] = tc.apply(
    lambda r: "ELS resilient" if r["group"] == "ELS" and r["animal_id"] in ELS_RESILIENT
              else r["group"], axis=1
)
tc["time_bin_numeric"] = tc["time_bin"].astype(float)

rows = []
for cl in CLUSTERS:
    sub = tc[tc["cluster"] == cl].copy()
    # ELS as reference
    sub["group_ext"] = pd.Categorical(sub["group_ext"],
                                      categories=["ELS", "Control", "ELS resilient"])
    try:
        model = smf.mixedlm(
            "pct ~ C(group_ext, Treatment('ELS')) * time_bin_numeric + C(experiment)",
            sub,
            groups=sub["animal_id"],
            re_formula="1",
        ).fit(reml=False, method="lbfgs")

        for param in model.params.index:
            rows.append({
                "cluster": cl,
                "panel": PANELS[cl],
                "parameter": param,
                "coef": model.params[param],
                "se": model.bse[param],
                "z": model.tvalues[param],
                "p": model.pvalues[param],
            })
    except Exception as e:
        rows.append({"cluster": cl, "panel": PANELS[cl], "parameter": "FAILED",
                     "coef": None, "se": None, "z": None, "p": None})
        print(f"{cl}: FAILED — {e}")

df = pd.DataFrame(rows)
out_path = REPO / "results/statistical_reports/fig5_timecourse_mixedlm.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

# Print focused results
print(f"\nFig 5D-J — MixedLM (ELS reference), 3-group timecourse\n{'='*70}")
key_params = [
    "C(group_ext, Treatment('ELS'))[T.ELS resilient]",
    "C(group_ext, Treatment('ELS'))[T.Control]",
    "C(group_ext, Treatment('ELS'))[T.ELS resilient]:time_bin_numeric",
    "C(group_ext, Treatment('ELS'))[T.Control]:time_bin_numeric",
    "time_bin_numeric",
    "C(experiment)[T.3]",
]
labels = {
    "C(group_ext, Treatment('ELS'))[T.ELS resilient]":                  "Resilient baseline vs ELS",
    "C(group_ext, Treatment('ELS'))[T.Control]":                        "Control baseline vs ELS",
    "C(group_ext, Treatment('ELS'))[T.ELS resilient]:time_bin_numeric": "time × Resilient",
    "C(group_ext, Treatment('ELS'))[T.Control]:time_bin_numeric":       "time × Control",
    "time_bin_numeric":                                                   "time (ELS slope)",
    "C(experiment)[T.3]":                                                "Experiment",
}

for cl in CLUSTERS:
    panel = PANELS[cl]
    sub = df[df["cluster"] == cl]
    print(f"\nFig 5{panel} — {cl}")
    for p in key_params:
        row = sub[sub["parameter"] == p]
        if row.empty:
            continue
        r = row.iloc[0]
        if pd.isna(r["coef"]):
            continue
        sig = "*" if r["p"] < 0.05 else " "
        label = labels.get(p, p)
        print(f"  {label:<35} b={r['coef']:>8.3f}  SE={r['se']:>6.3f}  z={r['z']:>7.3f}  p={r['p']:>8.4f} {sig}")

    # Compare to manuscript if available
    ms = MANUSCRIPT.get(cl, {})
    if ms.get("time_x_resilient"):
        mb, mse, mz = ms["time_x_resilient"]
        row = sub[sub["parameter"] == "C(group_ext, Treatment('ELS'))[T.ELS resilient]:time_bin_numeric"]
        if not row.empty:
            r = row.iloc[0]
            print(f"  >>> Manuscript time*Resilient: b={mb}  SE={mse}  z={mz}")
            print(f"  >>> Computed:                  b={r['coef']:.3f}  SE={r['se']:.3f}  z={r['z']:.3f}")
    if ms.get("baseline_resilient"):
        mb, mse, mz = ms["baseline_resilient"]
        row = sub[sub["parameter"] == "C(group_ext, Treatment('ELS'))[T.ELS resilient]"]
        if not row.empty:
            r = row.iloc[0]
            print(f"  >>> Manuscript baseline*Resilient: b={mb}  SE={mse}  z={mz}")
            print(f"  >>> Computed:                      b={r['coef']:.3f}  SE={r['se']:.3f}  z={r['z']:.3f}")

print(f"\nSaved -> {out_path}")
