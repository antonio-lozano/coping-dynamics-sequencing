"""Check GEE on frequency_seconds per animal — same model as figure_3.py."""
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

freq_df = pd.read_csv("d:/coping-dynamics-sequencing/data/source/cluster_frequency_per_animal.csv")
freq_df["group"] = pd.Categorical(freq_df["group"], categories=["Control", "ELS"])

MS_REF = {
    "Freeze": dict(beta=-0.204, SE=0.081, z=-2.511, p=0.012),
    "Sniff":  dict(beta=-0.068, SE=0.185, z=-2.133, p=0.033),  # manuscript value
    "Turn":   dict(beta=0.100,  SE=0.032, z=3.099,  p=0.002),
}

print("GEE on frequency_seconds (per-animal total) — figure_3.py model")
print("="*80)
for cluster in ["Freeze", "Sniff", "Turn", "Locomotion"]:
    sub = freq_df[freq_df["cluster"] == cluster].dropna(subset=["frequency_seconds"])
    model = smf.gee(
        "frequency_seconds ~ C(group, Treatment('Control')) + experiment",
        data=sub,
        groups=sub["animal_id"],
        family=sm.families.NegativeBinomial(alpha=1.0),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    print(f"\n{cluster} (n={sub['animal_id'].nunique()} animals):")
    for p in model.params.index:
        b, se, z, pv = model.params[p], model.bse[p], model.tvalues[p], model.pvalues[p]
        print(f"  {p:45s}  beta={b:.4f}  SE={se:.4f}  z={z:.4f}  p={pv:.4f}")
    if cluster in MS_REF:
        ref = MS_REF[cluster]
        # find ELS param
        els_params = [p for p in model.params.index if "ELS" in p or "group" in p.lower()]
        if els_params:
            p = els_params[0]
            b = model.params[p]
            print(f"  => ELS effect: beta={b:.4f}  manuscript={ref['beta']}  match={'YES' if abs(b-ref['beta'])<0.002 else 'NO'}")
