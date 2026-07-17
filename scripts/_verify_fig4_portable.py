"""Portable Fig 4 verification via the old-code MixedLM rerun model
   Metric ~ group + Experiment, groups=Animal, reml=False
Compares to the pasted-manuscript values. Run from repo root.
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import pandas as pd, statsmodels.formula.api as smf

REPO = Path(__file__).resolve().parents[1]
SR = REPO / "results/statistical_reports"

def fit(df, metric):
    d = df.copy()
    d["grp"] = pd.Categorical(d["group"], categories=["Control", "ELS"])
    d["Experiment"] = d["Experiment"].astype(int)
    m = smf.mixedlm(f"{metric} ~ grp + Experiment", d, groups=d["Animal"]).fit(reml=False)
    k = "grp[T.ELS]"
    return m.params[k], m.bse[k], m.tvalues[k], m.pvalues[k], d["Animal"].nunique()

def show(label, df, metric, ms):
    b, se, z, p, n = fit(df, metric)
    print(f"{label:<20}{b:>10.4f}{se:>9.4f}{z:>9.3f}{p:>9.4f}  n={n}   ms={ms}")

div = pd.read_csv(SR / "fig4_diversity_per_animal.csv")
bovr = pd.read_csv(SR / "fig4_bout_overall_per_animal.csv")
bcl = pd.read_csv(SR / "fig4_bout_cluster_per_animal.csv")
tr = pd.read_csv(SR / "fig4_transition_per_animal.csv")

print(f"{'metric':<20}{'beta':>10}{'SE':>9}{'z':>9}{'p':>9}        manuscript")
print("--- Diversity (4E-I) ---")
show("4E simpson", div, "simpson", "(-0.013,0.007,-1.890,0.059)")
show("4F shannon", div, "shannon", "(ns)")
show("4G evenness", div, "evenness", "(ns)")
show("4H-I cui", div, "cui", "(+0.064,0.021,2.316,0.021)")
print("--- Bout durations (4J-Q) ---")
show("4? overall bout", bovr, "bout_mean", "(-)")
for cl, panel in [("Freezing", "4K"), ("Sniffing", "4L"), ("Turn", "4N")]:
    sub = bcl[bcl.cluster == cl].rename(columns={"bout_duration": "bd"})
    show(f"{panel} {cl} bout", sub, "bd",
         {"Freezing": "(-0.105,0.058,-1.804,0.071)", "Sniffing": "(0.322,0.124,2.591,0.010)",
          "Turn": "(0.159,0.058,2.729,0.006)"}[cl])
print("--- Transition (4T-W) ---")
show("4T lz", tr, "lz", "(ns)")
show("4U recurrence", tr, "recurrence", "(0.013,0.006,2.314,0.021)")
show("4V determinism", tr, "determinism", "(0.016,0.005,3.317,0.001)")
show("4W markov", tr, "markov", "(-0.041,0.020,-2.025,0.043)")
