import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import statsmodels
print("statsmodels version:", statsmodels.__version__)

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
import statsmodels.formula.api as smf

sys.path.insert(0, 'd:/coping-dynamics-sequencing')
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

freq_df = raw.groupby(['Animal', 'Experiment', 'Condition', 'Cluster']).size().reset_index(name='Count')
pivot = freq_df.pivot_table(index=['Animal','Condition','Experiment'], columns='Cluster', values='Count', fill_value=0)
total = pivot.sum(axis=1)
rel = pivot.div(total, axis=0)
simpson = rel.apply(lambda r: 1 - np.sum(r**2), axis=1)
fm = simpson.reset_index().rename(columns={0: 'Simpson'})

m = smf.mixedlm("Simpson ~ Condition + Experiment", data=fm, groups=fm['Animal']).fit(reml=False)
pk = "Condition[T.ELS]"
print(f"Simpson: beta={m.params[pk]:.8f} SE={m.bse[pk]:.8f} z={m.tvalues[pk]:.8f} p={m.pvalues[pk]:.8f}")
print(f"Gold:    beta=-0.01344087 SE=0.00659293 z=-2.03867875 p=0.04148210")

# Also test freezing bout
def compute_mean_bout_duration(subdf):
    subdf = subdf.sort_values('Time_bin')
    subdf['Cluster_change'] = (subdf['Cluster'] != subdf['Cluster'].shift(1)).astype(int)
    subdf['Bout'] = subdf['Cluster_change'].cumsum()
    return (subdf.groupby('Bout').size() * 0.25).mean()

def compute_cluster_bout_duration(subdf):
    subdf = subdf.sort_values('Time_bin')
    subdf['Cluster_change'] = (subdf['Cluster'] != subdf['Cluster'].shift(1)).astype(int)
    subdf['Bout'] = subdf['Cluster_change'].cumsum()
    return subdf.groupby(['Bout','Cluster']).size().groupby(level=1).mean() * 0.25

cbd = raw.groupby(['Animal','Experiment']).apply(compute_cluster_bout_duration).reset_index()
cbd.columns = ['Animal','Experiment','Cluster','MeanBoutDuration']
cbd = cbd.merge(raw[['Animal','Condition','Experiment']].drop_duplicates(), on=['Animal','Experiment'])
freeze_df = cbd[cbd['Cluster']=='Freezing'].copy()

m2 = smf.mixedlm("MeanBoutDuration ~ Condition + Experiment", data=freeze_df, groups=freeze_df['Animal']).fit(reml=False)
pk2 = "Condition[T.ELS]"
print(f"\nFreezing bout: beta={m2.params[pk2]:.8f} SE={m2.bse[pk2]:.8f} z={m2.tvalues[pk2]:.8f} p={m2.pvalues[pk2]:.8f}")
print(f"Gold:          beta=-0.10517400 SE=0.05256600 z=-2.00079200 p=0.04541500")
