"""
Find the exact model parameterization that reproduces Report.xlsx SE=0.00659 for Simpson.
Testing: numeric Experiment codings, groupby variations, cluster map differences.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import SYLLABLE_TIMEBIN_250MS

CLUSTER_MAP = {
    "Freezing": [0, 28], "Sniffing": [18, 20], "Grooming": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27], "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climbing": [111], "Jump": [23, 29, 30, 34],
}

raw = pd.read_csv(SYLLABLE_TIMEBIN_250MS)
raw = raw.rename(columns={"Time Bin": "Time_bin"})
raw = raw[raw["Condition"].isin(["Control", "ELS"])].copy()
raw["Animal"] = raw["Animal"].astype(str)
raw["Syllable"] = pd.to_numeric(raw["Syllable"], errors="coerce").astype(int)
s2c = {}
for c, ss in CLUSTER_MAP.items():
    for s in ss: s2c[int(s)] = c
raw["Cluster"] = raw["Syllable"].map(s2c).fillna("")

def make_metrics(df):
    """Exact original: groupby.size() then pivot."""
    freq_df = df.groupby(['Animal', 'Experiment', 'Condition', 'Cluster']).size().reset_index(name='Count')
    pivot = freq_df.pivot_table(index=['Animal','Condition','Experiment'],
                                columns='Cluster', values='Count', fill_value=0)
    total = pivot.sum(axis=1)
    rel = pivot.div(total, axis=0)
    ent = rel.apply(lambda r: scipy_entropy(r[r>0]), axis=1)
    nclus = (pivot > 0).sum(axis=1)
    simp = rel.apply(lambda r: 1-np.sum(r**2), axis=1)
    even = ent / np.log(nclus)
    df_out = pd.DataFrame({'Simpson': simp, 'Entropy': ent, 'Evenness': even}).reset_index()
    df_out['New_condition'] = df_out['Condition']
    return df_out

fm = make_metrics(raw)
print("n animals:", len(fm))
print("Gold Simpson: beta=-0.01344087 SE=0.00659293 z=-2.03867875 p=0.04148210")

# Test many experiment encodings
for exp_label, exp_vals in [
    ("int 1/3",   {1: 1, 3: 3}),
    ("int 0/1",   {1: 0, 3: 1}),
    ("float 0/1", {1: 0.0, 3: 1.0}),
    ("float 0/2", {1: 0.0, 3: 2.0}),
    ("float -1/1",{1: -1.0, 3: 1.0}),
]:
    fm2 = fm.copy()
    fm2['Exp'] = fm2['Experiment'].map(exp_vals)
    try:
        m = smf.mixedlm("Simpson ~ New_condition + Exp",
                        data=fm2, groups=fm2['Animal']).fit(reml=False)
        pk = "New_condition[T.ELS]"
        b, s, z, p = m.params[pk], m.bse[pk], m.tvalues[pk], m.pvalues[pk]
        print(f"  Exp={exp_label}: beta={b:.6f} SE={s:.6f} z={z:.6f} p={p:.6f}")
    except Exception as e:
        print(f"  Exp={exp_label}: ERROR {e}")

# Try: MixedLM with vc_formula (experiment as variance component)
print("\n--- With Experiment variance component ---")
for exp_col_name, exp_col_vals, exp_use_vc in [
    ("Exp_vc_numeric", {1:1, 3:3}, True),
    ("Exp_vc_binary",  {1:0, 3:1}, True),
]:
    fm2 = fm.copy()
    fm2['ExpVC'] = fm2['Experiment'].map(exp_col_vals).astype(float)
    fm2['ExpCat'] = fm2['Experiment'].astype(str)
    try:
        # MixedLM with Experiment as variance component (no fixed Exp effect)
        m = MixedLM.from_formula(
            "Simpson ~ New_condition",
            data=fm2,
            groups="Animal",
            vc_formula={"ExpCat": "0+C(ExpCat)"}
        ).fit(reml=False)
        pk = "New_condition[T.ELS]"
        b, s, z, p = m.params[pk], m.bse[pk], m.tvalues[pk], m.pvalues[pk]
        print(f"  vc_Exp: beta={b:.6f} SE={s:.6f} z={z:.6f} p={p:.6f}")
    except Exception as e:
        print(f"  vc_Exp error: {e}")

# Try: MixedLM with both fixed Exp and vc_Exp
print("\n--- MixedLM + fixed Exp + vc_Exp ---")
fm2 = fm.copy()
fm2['ExpCat'] = fm2['Experiment'].astype(str)
try:
    m = MixedLM.from_formula(
        "Simpson ~ New_condition + ExpCat",
        data=fm2,
        groups="Animal",
        vc_formula={"ExpCat": "0+C(ExpCat)"}
    ).fit(reml=False)
    pk = "New_condition[T.ELS]"
    b, s, z, p = m.params[pk], m.bse[pk], m.tvalues[pk], m.pvalues[pk]
    print(f"  fixed+vc_Exp: beta={b:.6f} SE={s:.6f} z={z:.6f} p={p:.6f}")
except Exception as e:
    print(f"  fixed+vc_Exp error: {e}")

# Try: REML=True versions
print("\n--- REML=True ---")
fm2 = fm.copy()
fm2['Exp_num'] = fm2['Experiment'].map({1:1, 3:3})
for exp_col in ['Exp_num']:
    try:
        m = smf.mixedlm(f"Simpson ~ New_condition + {exp_col}",
                        data=fm2, groups=fm2['Animal']).fit(reml=True)
        pk = "New_condition[T.ELS]"
        b, s, z, p = m.params[pk], m.bse[pk], m.tvalues[pk], m.pvalues[pk]
        print(f"  REML=True, {exp_col}: beta={b:.6f} SE={s:.6f} z={z:.6f} p={p:.6f}")
    except Exception as e:
        print(f"  REML=True error: {e}")

# Additional: what about including all 3-group model with ELS_resilient?
print("\n--- 3-group model (with ELS_resilient) ---")
els_resilient_list = ['2.4', '15.4', '26.4', '43.3', '43.5', '49.6',
                      '123.3', '134.2', '148.4', '150.3', '150.5', '159.4']
fm3 = fm.copy()
fm3['New_condition'] = fm3.apply(
    lambda r: 'ELS_resilient' if r['Animal'] in els_resilient_list
    else r['Condition'], axis=1
)
print("3-group counts:", fm3['New_condition'].value_counts().to_dict())
fm3['Exp_num'] = fm3['Experiment'].map({1:1, 3:3})
try:
    m = smf.mixedlm("Simpson ~ New_condition + Exp_num",
                    data=fm3, groups=fm3['Animal']).fit(reml=False)
    for pk in ['New_condition[T.ELS]', 'New_condition[T.ELS_resilient]']:
        if pk in m.params:
            b, s, z, p = m.params[pk], m.bse[pk], m.tvalues[pk], m.pvalues[pk]
            print(f"  3-group {pk}: beta={b:.8f} SE={s:.8f} z={z:.8f} p={p:.8f}")
except Exception as e:
    print(f"  3-group error: {e}")
