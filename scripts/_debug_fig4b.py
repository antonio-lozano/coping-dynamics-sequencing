"""Find which model structure reproduces the Report.xlsx SE for simpson."""
import sys; sys.path.insert(0, 'd:/coping-dynamics-sequencing')
import numpy as np
import pandas as pd
from scipy.stats import entropy
from statsmodels.regression.mixed_linear_model import MixedLM
from src.config import SYLLABLE_TIMEBIN_250MS
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

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

GOLD_BETA = -0.01344087
GOLD_SE = 0.00659293
GOLD_Z = -2.03867875

print(f"Gold: beta={GOLD_BETA:.8f} SE={GOLD_SE:.8f} z={GOLD_Z:.8f}")

# Model A: MixedLM(simpson ~ Stress + Exp, groups=animal_id) -- 1 obs/group
endog = df['simpson'].values
exog_full = np.column_stack([np.ones(len(df)), df['Stress'].values, df['Exp'].values])
exog_stress = np.column_stack([np.ones(len(df)), df['Stress'].values])
result_a = MixedLM(endog, exog_full, groups=df['Animal'].values).fit(reml=False)
print(f"\nA: MixedLM(simpson~Stress+Exp, groups=animal): beta={result_a.params[1]:.8f} SE={result_a.bse[1]:.8f} z={result_a.tvalues[1]:.8f}")

# Model B: MixedLM(simpson ~ Stress, groups=Experiment) -- 2 groups, 41 obs each
result_b = MixedLM(endog, exog_stress, groups=df['Experiment'].values).fit(reml=False)
print(f"B: MixedLM(simpson~Stress, groups=Exp): beta={result_b.params[1]:.8f} SE={result_b.bse[1]:.8f} z={result_b.tvalues[1]:.8f}")

# Model C: MixedLM(simpson ~ Stress, groups=Experiment, exog_re=Stress)
try:
    exog_re = df['Stress'].values.reshape(-1,1)
    result_c = MixedLM(endog, exog_stress, groups=df['Experiment'].values, exog_re=exog_re).fit(reml=False)
    print(f"C: MixedLM(simpson~Stress, groups=Exp, re=Stress): beta={result_c.params[1]:.8f} SE={result_c.bse[1]:.8f} z={result_c.tvalues[1]:.8f}")
except Exception as e:
    print(f"C failed: {e}")

# Model D: MixedLM(simpson ~ Stress, groups=animal, vc_formula=Experiment)
try:
    df['ExpC'] = df['Experiment']
    result_d = MixedLM.from_formula("simpson ~ Stress", data=df, groups="Animal", vc_formula={"ExpC": "0+C(ExpC)"}).fit(reml=False)
    print(f"D: MixedLM(simpson~Stress, groups=animal, vc=Exp): beta={result_d.params[1]:.8f} SE={result_d.bse[1]:.8f} z={result_d.tvalues[1]:.8f}")
except Exception as e:
    print(f"D failed: {e}")

# Model E: REML=True
result_e = MixedLM(endog, exog_full, groups=df['Animal'].values).fit(reml=True)
print(f"E: MixedLM(~Stress+Exp, groups=animal, REML=True): beta={result_e.params[1]:.8f} SE={result_e.bse[1]:.8f} z={result_e.tvalues[1]:.8f}")

# Model F: No random effect (pure OLS using MixedLM framework with group var constrained to 0)
# This might explain "Experiment Var = 1" and "No. Groups = NA"

# Model G: OLS without experiment
df['group_cat'] = pd.Categorical(df['group'], categories=['Control', 'ELS'])
result_g = smf.ols("simpson ~ C(group_cat)", data=df).fit()
print(f"G: OLS(simpson~Stress only): beta={result_g.params['C(group_cat)[T.ELS]']:.8f} SE={result_g.bse['C(group_cat)[T.ELS]']:.8f} t={result_g.tvalues['C(group_cat)[T.ELS]']:.8f}")

# Model H: WLS or robust?
result_h = smf.ols("simpson ~ C(group_cat) + Experiment", data=df).fit(cov_type='HC3')
print(f"H: OLS(+Exp, HC3 robust): beta={result_h.params['C(group_cat)[T.ELS]']:.8f} SE={result_h.bse['C(group_cat)[T.ELS]']:.8f} t={result_h.tvalues['C(group_cat)[T.ELS]']:.8f}")
