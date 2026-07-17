"""
Test methods to get stable SE for transition metrics MixedLM.
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

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

print("Building per-animal metric dataframes...")
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
    ("Recurrence", rr_df, "Recurrence_Rate", 0.01344, 0.00594, 2.264, 0.024),
    ("Determinism", det_df, "Determinism", 0.01610, 0.00476, 3.382, 0.001),
    ("Markov",    entropy_merged, "MarkovEntropy", -0.04119, 0.01759, -2.342, 0.019),
]

pk = "Condition[T.ELS]"

def fit_and_report(name, df, metric, gold_b, gold_se, gold_z, gold_p, label, result):
    b = result.params[pk]; se = result.bse[pk]
    z = result.tvalues[pk]; p = result.pvalues[pk]
    match_se = "✅" if abs(se - gold_se) < 0.001 else "❌"
    match_p  = "✅" if (p < 0.05) == (gold_p < 0.05) else "❌"
    print(f"  {name:12s} beta={b:.5f} SE={se:.5f}{match_se} z={z:.3f} p={p:.5f}{match_p}  (gold SE={gold_se:.5f} p={gold_p:.5f})")

# ── METHOD 1: Standard (baseline) ─────────────────────────────────────────────
print("\n=== Baseline (standard MixedLM, reml=False) ===")
for name, df, metric, *gold in DATASETS:
    m = smf.mixedlm(f"{metric} ~ Condition + Experiment", df, groups=df["Animal"]).fit(reml=False)
    fit_and_report(name, df, metric, *gold, "baseline", m)

# ── METHOD 2: free_re=True ────────────────────────────────────────────────────
print("\n=== free_re=True (unconstrained random effects) ===")
for name, df, metric, *gold in DATASETS:
    m = smf.mixedlm(f"{metric} ~ Condition + Experiment", df, groups=df["Animal"]).fit(reml=False, free_re=True)
    fit_and_report(name, df, metric, *gold, "free_re", m)

# ── METHOD 3: REML=True ───────────────────────────────────────────────────────
print("\n=== REML=True ===")
for name, df, metric, *gold in DATASETS:
    m = smf.mixedlm(f"{metric} ~ Condition + Experiment", df, groups=df["Animal"]).fit(reml=True)
    fit_and_report(name, df, metric, *gold, "reml", m)

# ── METHOD 4: Jackknife SE ────────────────────────────────────────────────────
print("\n=== Jackknife SE (delete-one-animal, leave-one-out) ===")
for name, df, metric, gold_b, gold_se, gold_z, gold_p in DATASETS:
    animals = df['Animal'].unique()
    betas = []
    for a in animals:
        sub = df[df['Animal'] != a]
        try:
            m = smf.mixedlm(f"{metric} ~ Condition + Experiment", sub, groups=sub["Animal"]).fit(reml=False)
            betas.append(m.params[pk])
        except Exception:
            pass
    n = len(betas)
    jk_se = np.sqrt((n - 1) / n * np.sum((np.array(betas) - np.mean(betas))**2))
    m_full = smf.mixedlm(f"{metric} ~ Condition + Experiment", df, groups=df["Animal"]).fit(reml=False)
    b = m_full.params[pk]
    z = b / jk_se
    from scipy import stats
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    match_se = "✅" if abs(jk_se - gold_se) < 0.001 else "❌"
    match_p  = "✅" if (p < 0.05) == (gold_p < 0.05) else "❌"
    print(f"  {name:12s} beta={b:.5f} SE={jk_se:.5f}{match_se} z={z:.3f} p={p:.5f}{match_p}  (gold SE={gold_se:.5f} p={gold_p:.5f})")

# ── METHOD 5: lbfgs optimizer with tighter tolerance ─────────────────────────
print("\n=== lbfgs optimizer (gtol=1e-8) ===")
for name, df, metric, *gold in DATASETS:
    m = smf.mixedlm(f"{metric} ~ Condition + Experiment", df, groups=df["Animal"]).fit(
        reml=False, method='lbfgs', optim_kwds={'gtol': 1e-8})
    fit_and_report(name, df, metric, *gold, "lbfgs_tight", m)
