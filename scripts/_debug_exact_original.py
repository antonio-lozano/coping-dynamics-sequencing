"""
Reproduce the EXACT original model from Frequency_indexes_bout_duration_resilience.py,
but restricted to Control vs ELS only (no ELS_resilient split).

Original model:
  smf.mixedlm("Simpson ~ New_condition + Experiment",
               data=freq_metrics_reset,
               groups=freq_metrics_reset['Animal']).fit(reml=False)

Key differences from our OLS:
  1. MixedLM instead of OLS (random intercept per Animal)
  2. Frequency metrics computed from raw 250ms data via groupby.size() (all rows)
  3. Cluster assignment via JSON map (vs our hardcoded CLUSTER_MAP)

For Control vs ELS only: New_condition = Condition (Control or ELS, 41 each)
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import SYLLABLE_TIMEBIN_250MS

# Hardcoded cluster map (as in figure_4.py and our repo)
CLUSTER_MAP_REPO = {
    "Freezing": [0, 28],
    "Sniffing": [18, 20],
    "Grooming": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climbing": [111],
    "Jump": [23, 29, 30, 34],
}

# Load raw 250ms data
raw = pd.read_csv(SYLLABLE_TIMEBIN_250MS)
raw = raw.rename(columns={"Time Bin": "Time_bin", "Condition": "Condition"})
raw = raw[raw["Condition"].isin(["Control", "ELS"])].copy()
raw["Animal"] = raw["Animal"].astype(str)
raw["Syllable"] = pd.to_numeric(raw["Syllable"], errors="coerce").astype(int)

# Map syllables to clusters (repo CLUSTER_MAP)
s2c = {}
for c, ss in CLUSTER_MAP_REPO.items():
    for s in ss: s2c[int(s)] = c
raw["Cluster"] = raw["Syllable"].map(s2c).fillna("")

print(f"Total rows: {len(raw)}")
print(f"Animals: {raw['Animal'].nunique()}")
print(f"Conditions: {raw['Condition'].value_counts().to_dict()}")

# ── EXACT ORIGINAL: compute_frequency_metrics ─────────────────────────────────
# "freq_df = df.groupby(['Animal', 'Experiment', 'New_condition', 'Cluster']).size()"
# Here New_condition = Condition (Control or ELS, no split)
freq_df = raw.groupby(['Animal', 'Experiment', 'Condition', 'Cluster']).size().reset_index(name='Count')
pivot_df = freq_df.pivot_table(
    index=['Animal', 'Condition', 'Experiment'],
    columns='Cluster',
    values='Count',
    fill_value=0
)
total_counts = pivot_df.sum(axis=1)
relative_freq = pivot_df.div(total_counts, axis=0)

from scipy.stats import entropy as scipy_entropy
entropy_values = relative_freq.apply(lambda row: scipy_entropy(row[row > 0]), axis=1)
num_clusters = (pivot_df > 0).sum(axis=1)
max_entropy = np.log(num_clusters)
evenness = entropy_values / max_entropy
simpson = relative_freq.apply(lambda row: 1 - np.sum(row**2), axis=1)

freq_metrics = pd.DataFrame({
    'Entropy': entropy_values,
    'Evenness': evenness,
    'Simpson': simpson
})
freq_metrics_reset = freq_metrics.reset_index()
# Add New_condition column (same as Condition here — Control vs ELS only)
freq_metrics_reset['New_condition'] = freq_metrics_reset['Condition']

print(f"\nfreq_metrics_reset shape: {freq_metrics_reset.shape}")
print(freq_metrics_reset['New_condition'].value_counts())

# Gold standard from Report.xlsx
GOLD = {
    "Simpson":  {"beta": -0.013440869, "se": 0.006592931, "z": -2.038678748, "p": 0.0414821},
    "Entropy":  {"beta": -0.011573886, "se": 0.018343103, "z": -0.630966653, "p": 0.528062329},
    "Evenness": {"beta": -0.006497065, "se": 0.009731452, "z": -0.667635767, "p": 0.504366121},
}

for metric in ["Simpson", "Entropy", "Evenness"]:
    print(f"\n=== {metric.upper()} ===")
    g = GOLD[metric]
    print(f"  Gold: beta={g['beta']:.8f} SE={g['se']:.8f} z={g['z']:.8f} p={g['p']:.8f}")

    # Method 1: EXACT ORIGINAL — smf.mixedlm(groups=Animal)
    try:
        m1 = smf.mixedlm(f"{metric} ~ New_condition + Experiment",
                          data=freq_metrics_reset,
                          groups=freq_metrics_reset['Animal']).fit(reml=False)
        pk = "New_condition[T.ELS]"
        b1, s1, z1, p1 = m1.params[pk], m1.bse[pk], m1.tvalues[pk], m1.pvalues[pk]
        print(f"  MixedLM(~Cond+Exp, grp=Animal): beta={b1:.8f} SE={s1:.8f} z={z1:.8f} p={p1:.8f}")
        print(f"    beta_match={abs(b1-g['beta'])<=0.001*abs(g['beta'])}, SE_match={abs(s1-g['se'])<=0.01*abs(g['se'])}")
    except Exception as e:
        print(f"  MixedLM error: {e}")

    # Method 2: OLS (what we had before)
    freq_metrics_reset['cond_cat'] = pd.Categorical(freq_metrics_reset['New_condition'],
                                                      categories=['Control', 'ELS'])
    m2 = smf.ols(f"{metric} ~ C(cond_cat) + Experiment", data=freq_metrics_reset).fit()
    pk2 = "C(cond_cat)[T.ELS]"
    b2, s2, z2, p2 = m2.params[pk2], m2.bse[pk2], m2.tvalues[pk2], m2.pvalues[pk2]
    print(f"  OLS(~Cond+Exp):                  beta={b2:.8f} SE={s2:.8f} t={z2:.8f} p={p2:.8f}")

# ── Check: does including Experiment as numeric vs categorical matter? ─────────
print("\n\n=== CHECKING EXPERIMENT ENCODING ===")
print(f"Experiment values: {freq_metrics_reset['Experiment'].unique()}")
# If Experiment is numeric 1/3 in original vs "1"/"3" string
freq_metrics_reset['Exp_num'] = freq_metrics_reset['Experiment'].astype(float)
freq_metrics_reset['Exp_str'] = freq_metrics_reset['Experiment'].astype(str)

for exp_col in ['Exp_num', 'Exp_str']:
    try:
        m = smf.mixedlm(f"Simpson ~ New_condition + {exp_col}",
                          data=freq_metrics_reset,
                          groups=freq_metrics_reset['Animal']).fit(reml=False)
        pk = "New_condition[T.ELS]"
        b, s, z, p = m.params[pk], m.bse[pk], m.tvalues[pk], m.pvalues[pk]
        print(f"  MixedLM(Exp as {exp_col}): beta={b:.8f} SE={s:.8f} z={z:.8f} p={p:.8f}")
    except Exception as e:
        print(f"  Error with {exp_col}: {e}")
