"""
Likelihood Ratio Test for transition metrics.
Avoids SE/Hessian boundary issue entirely — p-value comes from
loglik difference, not from Wald SE. Valid even at the boundary.
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

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

print("Computing metrics...")
rr_list, det_list, lz_list = [], [], []
for animal, group in moseq_df.groupby('Animal'):
    R = recurrence_plot(group['Cluster'].tolist())
    meta = {'Animal': animal, 'Condition': group['Condition'].iloc[0],
            'Experiment': group['Experiment'].iloc[0]}
    rr_list.append({**meta, 'Recurrence_Rate': recurrence_rate(R)})
    det_list.append({**meta, 'Determinism': determinism(R)})
    lz_list.append({**meta, 'LZ_complexity': sum(1 for _ in group['Cluster'])})  # placeholder
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

print(f"\n{'Metric':<12}  {'beta':>8}  {'SE(Wald)':>9}  {'LRT chi2':>9}  {'p(LRT)':>8}  {'p(gold)':>8}  sig?")
print("-" * 70)
for name, df, metric, gold_b, gold_se, gold_z, gold_p in DATASETS:
    # Full model: Condition + Experiment
    m_full = smf.mixedlm(f"{metric} ~ Condition + Experiment", df, groups=df["Animal"]).fit(reml=False)
    # Reduced model: Experiment only (drop Condition)
    m_null = smf.mixedlm(f"{metric} ~ Experiment", df, groups=df["Animal"]).fit(reml=False)

    beta = m_full.params[pk]
    se_wald = m_full.bse[pk]

    # LRT statistic
    lrt = 2 * (m_full.llf - m_null.llf)
    p_lrt = stats.chi2.sf(lrt, df=1)

    sig = "YES" if p_lrt < 0.05 else "NO"
    match = "✅" if (p_lrt < 0.05) == (gold_p < 0.05) else "❌"
    print(f"{name:<12}  {beta:>8.5f}  {se_wald:>9.5f}  {lrt:>9.4f}  {p_lrt:>8.5f}  {gold_p:>8.5f}  {sig} {match}")

print(f"\nNote: LRT p-value is valid regardless of Wald SE/Hessian issues.")
print(f"SE(Wald) shown for reference only — inflated due to boundary convergence.")
