"""
Check unmapped syllables and try excluding them from denominator vs including them.
Also try exact original with all 3 conditions to see if the n=82 + 3-group model
gives SE=0.00659 for the ELS term.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
import statsmodels.formula.api as smf
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

all_syllables = sorted(raw["Syllable"].unique())
all_mapped = [s for ss in CLUSTER_MAP.values() for s in ss]
unmapped = [s for s in all_syllables if s not in all_mapped]
print(f"Unique syllables: {all_syllables}")
print(f"Unmapped syllables (empty string): {unmapped}")
print(f"Empty cluster fraction: {(raw['Cluster'] == '').mean():.4f} ({(raw['Cluster'] == '').sum()} rows)")

# Original: denominator INCLUDES empty string cluster
# Our code: same as original — pivot includes "" column, total = sum of all
freq_df = raw.groupby(['Animal', 'Experiment', 'Condition', 'Cluster']).size().reset_index(name='Count')
pivot = freq_df.pivot_table(index=['Animal','Condition','Experiment'],
                            columns='Cluster', values='Count', fill_value=0)
print(f"\nPivot columns (clusters + '' empty): {list(pivot.columns)}")
print(f"'' (empty) column present: {'' in pivot.columns}")

total_with_empty = pivot.sum(axis=1)
total_named_only = pivot.drop(columns=[''], errors='ignore').sum(axis=1)
print(f"\nTotal counts per animal (with empty): {total_with_empty.describe()}")
print(f"Total counts per animal (named only): {total_named_only.describe()}")

# Compute Simpson with and without empty cluster in denominator
rel_with_empty = pivot.div(total_with_empty, axis=0)
rel_named_only = pivot.drop(columns=[''], errors='ignore').div(total_named_only, axis=0)

simp_with_empty = rel_with_empty.apply(lambda r: 1 - np.sum(r**2), axis=1)
simp_named_only = rel_named_only.apply(lambda r: 1 - np.sum(r**2), axis=1)

fm_empty = simp_with_empty.reset_index().rename(columns={0: 'Simpson'})
fm_empty['New_condition'] = fm_empty['Condition']
fm_named = simp_named_only.reset_index().rename(columns={0: 'Simpson'})
fm_named['New_condition'] = fm_named['Condition']

print("\n--- Simpson with empty string in denominator (original approach) ---")
m = smf.mixedlm("Simpson ~ New_condition + Experiment", data=fm_empty,
                 groups=fm_empty['Animal']).fit(reml=False)
pk = "New_condition[T.ELS]"
print(f"  MixedLM: beta={m.params[pk]:.8f} SE={m.bse[pk]:.8f} z={m.tvalues[pk]:.8f} p={m.pvalues[pk]:.8f}")
print(f"  Gold:    beta=-0.01344087 SE=0.00659293 z=-2.03867875 p=0.04148210")

print("\n--- Simpson with named clusters only in denominator ---")
m2 = smf.mixedlm("Simpson ~ New_condition + Experiment", data=fm_named,
                  groups=fm_named['Animal']).fit(reml=False)
print(f"  MixedLM: beta={m2.params[pk]:.8f} SE={m2.bse[pk]:.8f} z={m2.tvalues[pk]:.8f} p={m2.pvalues[pk]:.8f}")

# Try: what if original CSV has all 3 conditions (Control, ELS, ELS_resilient)?
# Check if ELS_resilient animals in our CSV are labeled "ELS" or something else
els_resilient_list = ['2.4', '15.4', '26.4', '43.3', '43.5', '49.6',
                      '123.3', '134.2', '148.4', '150.3', '150.5', '159.4']
raw_all = pd.read_csv(SYLLABLE_TIMEBIN_250MS)
raw_all["Animal"] = raw_all["Animal"].astype(str)
raw_els = raw_all[raw_all["Condition"] == "ELS"]
print(f"\nELS animals in CSV: {sorted(raw_els['Animal'].unique())}")
print(f"ELS_resilient list: {els_resilient_list}")
resilient_in_csv = [a for a in els_resilient_list if a in raw_els['Animal'].unique()]
print(f"ELS_resilient animals in CSV (labeled as 'ELS'): {resilient_in_csv}")
