"""
Fix the boundary convergence by initializing random effect variance
away from zero via start_params.
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import patsy

sys.path.insert(0, 'd:/coping-dynamics-sequencing')
from src.config import SYLLABLE_TIMEBIN_250MS

CLUSTER_MAP = {
    "Freezing": [0, 28], "Sniffing": [18, 20], "Grooming": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27], "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climbing": [111], "Jump": [23, 29, 30, 34],
}
moseq_df = pd.read_csv(SYLLABLE_TIMEBIN_250MS)
moseq_df.rename(columns={'Time Bin': 'Time_bin'}, inplace=True)
moseq_df = moseq_df[moseq_df['Condition'].isin(['Control', 'ELS'])].copy()
moseq_df['Animal'] = moseq_df['Animal'].astype(str)
moseq_df['Syllable'] = pd.to_numeric(moseq_df['Syllable'], errors='coerce').astype(int)
s2c = {int(s): c for c, ss in CLUSTER_MAP.items() for s in ss}
moseq_df['Cluster'] = moseq_df['Syllable'].map(s2c).fillna("")

def recurrence_plot(seq):
    seq = np.array(seq); n = len(seq)
    R = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if seq[i] == seq[j]: R[i, j] = 1
    return R

def recurrence_rate(R): return R.sum() / (R.shape[0] ** 2)

def diagonal_line_lengths(R, min_length=2):
    n = R.shape[0]; lengths = []
    for k in range(-n+1, n):
        diag = np.diag(R, k=k); count = 0
        for val in diag:
            if val == 1: count += 1
            else:
                if count >= min_length: lengths.append(count)
                count = 0
        if count >= min_length: lengths.append(count)
    return lengths

def determinism(R, min_length=2):
    lengths = diagonal_line_lengths(R, min_length)
    total = R.sum() - R.shape[0]
    return sum(lengths) / total if total > 0 else 0

def compute_markov_entropy_index(df, smoothing_factor=0.01):
    results = []
    for animal, grp in df.sort_values(['Animal','Time_bin']).groupby('Animal'):
        states = grp['Cluster'].tolist(); uniq = list(pd.unique(states))
        if len(uniq) == 1:
            results.append({'Animal': animal, 'MarkovEntropy': 0.0}); continue
        idx = {s: i for i, s in enumerate(uniq)}; M = len(uniq)
        counts = np.zeros((M, M))
        for a, b in zip(states, states[1:]): counts[idx[a], idx[b]] += 1
        counts += 0.01
        p = counts / counts.sum(axis=1, keepdims=True)
        pi = np.array([grp['Cluster'].value_counts().get(s, 0) for s in uniq]) / len(states)
        inner = np.array([-np.sum(row[row > 0] * np.log2(row[row > 0])) for row in p])
        results.append({'Animal': animal, 'MarkovEntropy': np.sum(pi * inner)})
    return pd.DataFrame(results)

print("Building metric dataframes...")
rr_list, det_list = [], []
for animal, group in moseq_df.groupby('Animal'):
    R = recurrence_plot(group['Cluster'].tolist())
    meta = {'Animal': animal, 'Condition': group['Condition'].iloc[0],
            'Experiment': group['Experiment'].iloc[0]}
    rr_list.append({**meta, 'Recurrence_Rate': recurrence_rate(R)})
    det_list.append({**meta, 'Determinism': determinism(R)})
rr_df = pd.DataFrame(rr_list)
det_df = pd.DataFrame(det_list)
entropy_df = compute_markov_entropy_index(moseq_df)
entropy_merged = pd.merge(entropy_df, moseq_df[['Animal','Condition','Experiment']].drop_duplicates(), on='Animal')

DATASETS = [
    ("Recurrence", rr_df,          "Recurrence_Rate", 0.01344, 0.00594, 2.264, 0.024),
    ("Determinism", det_df,         "Determinism",     0.01610, 0.00476, 3.382, 0.001),
    ("Markov",      entropy_merged, "MarkovEntropy",  -0.04119, 0.01759,-2.342, 0.019),
]
pk = "Condition[T.ELS]"

def fit_with_start(formula, df, re_var_init):
    """Fit MixedLM with random effect variance initialized to re_var_init."""
    model = smf.mixedlm(formula, df, groups=df["Animal"])
    # Get default start_params from a quick OLS fit then override variance
    res0 = smf.mixedlm(formula, df, groups=df["Animal"]).fit(reml=False, start_params=None)
    sp = res0.params_object.copy()
    # Override: set random effect covariance to re_var_init
    sp.cov_re = np.array([[re_var_init]])
    sp.vcomp = np.array([re_var_init])
    try:
        start = np.concatenate([res0.fe_params.values,
                                 np.array([re_var_init]),
                                 np.array([res0.scale])])
    except Exception:
        start = None
    return model.fit(reml=False, start_params=start)

print(f"\nGold: Recurrence SE=0.00594 p=0.024 | Determinism SE=0.00476 p=0.001 | Markov SE=0.01759 p=0.019")

# Try a range of re_var starting values
for re_init in [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05]:
    print(f"\n--- start re_var={re_init} ---")
    for name, df, metric, gold_b, gold_se, gold_z, gold_p in DATASETS:
        try:
            m = fit_with_start(f"{metric} ~ Condition + Experiment", df, re_init)
            se = m.bse[pk]; z = m.tvalues[pk]; p = m.pvalues[pk]
            flag = "✅" if abs(se - gold_se) < 0.001 else ("~" if abs(se - gold_se) < 0.002 else "❌")
            sig  = "✅" if (p < 0.05) == (gold_p < 0.05) else "❌"
            print(f"  {name:12s} SE={se:.5f}{flag} z={z:.3f} p={p:.5f}{sig}  (gold SE={gold_se:.5f})")
        except Exception as e:
            print(f"  {name:12s} ERROR: {e}")
