import sys; sys.path.insert(0, 'd:/coping-dynamics-sequencing')
import numpy as np
import pandas as pd
from scipy.stats import entropy
from statsmodels.regression.mixed_linear_model import MixedLM
from src.config import SYLLABLE_TIMEBIN_250MS
import statsmodels.formula.api as smf

CLUSTER_MAP = {
    'Freezing': [0, 28], 'Sniffing': [18, 20], 'Grooming': [24],
    'Turn': [1, 3, 5, 6, 10, 15, 26, 27], 'Locomotion': [11, 12, 14, 16, 19, 21, 25],
    'Climbing': [111], 'Jump': [23, 29, 30, 34],
}

raw = pd.read_csv(SYLLABLE_TIMEBIN_250MS)
raw = raw.rename(columns={'Time Bin': 'time_bin', 'Condition': 'group'})
raw = raw[raw['group'].isin(['Control', 'ELS'])].copy()
raw['Animal'] = raw['Animal'].astype(str)
raw['Syllable'] = pd.to_numeric(raw['Syllable'], errors='coerce').astype(int)
s2c = {}
for c, ss in CLUSTER_MAP.items():
    for s in ss: s2c[int(s)] = c
raw['cluster'] = raw['Syllable'].map(s2c).fillna('')

idx = raw.groupby(['Animal', 'time_bin'])['Percentage'].idxmax()
pred = raw.loc[idx].sort_values(['Animal', 'time_bin']).reset_index(drop=True)
meta = pred[['Animal', 'group', 'Experiment']].drop_duplicates().reset_index(drop=True)

full_sequences = {str(a): grp.sort_values('time_bin')['cluster'].tolist() for a, grp in raw.groupby('Animal', sort=False)}

# Compute simpson
rows = []
for animal, seq in full_sequences.items():
    info = meta.set_index('Animal').to_dict('index')[animal]
    vals, counts = np.unique(seq, return_counts=True)
    p = counts / counts.sum()
    simpson = 1 - np.sum(p**2)
    rows.append({'Animal': animal, 'group': info['group'], 'Experiment': str(info['Experiment']), 'simpson': simpson})
df = pd.DataFrame(rows)
df['Stress'] = (df['group'] == 'ELS').astype(float)
df['Exp'] = (df['Experiment'] == '3').astype(float)

print('Fitting MixedLM...')
endog = df['simpson'].values
exog = np.column_stack([np.ones(len(df)), df['Stress'].values, df['Exp'].values])
groups = df['Animal'].values
try:
    model = MixedLM(endog, exog, groups=groups)
    result = model.fit(reml=False)
    print(result.summary())
    print(f"Stress beta={result.params[1]:.8f} SE={result.bse[1]:.8f} z={result.tvalues[1]:.8f} p={result.pvalues[1]:.8f}")
except Exception as e:
    print('MixedLM error:', type(e).__name__, e)

print('\nOLS results:')
df['group_cat'] = pd.Categorical(df['group'], categories=['Control', 'ELS'])
model2 = smf.ols("simpson ~ C(group_cat) + Experiment", data=df).fit()
param_key = "C(group_cat)[T.ELS]"
print(f"OLS: Stress beta={model2.params[param_key]:.8f} SE={model2.bse[param_key]:.8f} t={model2.tvalues[param_key]:.8f} p={model2.pvalues[param_key]:.8f}")
print(f"Gold:         beta=-0.01344087   SE=0.00659293   z=-2.03867875   p=0.04148210")
