"""
Q1: Recompute transition metrics excluding unclassified syllables ("" cluster).
Uses same groupby order as original to preserve beta match.
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
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
        states = grp['Cluster'].tolist()
        states = [s for s in states if s != ""]  # exclude unclassified
        if len(states) < 2:
            results.append({'Animal': animal, 'MarkovEntropy': np.nan}); continue
        uniq = list(pd.unique(states))
        if len(uniq) == 1:
            results.append({'Animal': animal, 'MarkovEntropy': 0.0}); continue
        idx = {s: i for i, s in enumerate(uniq)}; M = len(uniq)
        counts = np.zeros((M, M))
        for a, b in zip(states, states[1:]): counts[idx[a], idx[b]] += 1
        counts += smoothing_factor
        p = counts / counts.sum(axis=1, keepdims=True)
        pi = np.array([states.count(s) for s in uniq]) / len(states)
        inner = np.array([-np.sum(row[row > 0] * np.log2(row[row > 0])) for row in p])
        results.append({'Animal': animal, 'MarkovEntropy': np.sum(pi * inner)})
    return pd.DataFrame(results)

pk = "Condition[T.ELS]"
gold = {
    "Recurrence":  (0.01344, 0.00594, 2.264,  0.024),
    "Determinism": (0.01610, 0.00476, 3.382,  0.001),
    "Markov":      (-0.04119, 0.01759, -2.342, 0.019),
}

print("Computing metrics (excluding '' cluster from sequences)...")
rr_list, det_list = [], []
for animal, group in moseq_df.groupby('Animal'):
    seq = [s for s in group['Cluster'].tolist() if s != ""]
    if len(seq) < 2:
        continue
    R = recurrence_plot(seq)
    meta = {'Animal': animal, 'Condition': group['Condition'].iloc[0],
            'Experiment': group['Experiment'].iloc[0]}
    rr_list.append({**meta, 'Recurrence_Rate': recurrence_rate(R)})
    det_list.append({**meta, 'Determinism': determinism(R)})
rr_df = pd.DataFrame(rr_list)
det_df = pd.DataFrame(det_list)
entropy_df = compute_markov_entropy_index(moseq_df)
entropy_merged = pd.merge(entropy_df.dropna(),
                          moseq_df[['Animal','Condition','Experiment']].drop_duplicates(), on='Animal')

print(f"\n{'Metric':<12} {'beta':>9} {'SE':>8} {'z':>7} {'p':>8}  {'gold_beta':>9} {'gold_SE':>8} {'gold_z':>7} {'gold_p':>8}  beta_match?")
print("-"*95)
for name, df, metric in [("Recurrence", rr_df, "Recurrence_Rate"),
                          ("Determinism", det_df, "Determinism"),
                          ("Markov", entropy_merged, "MarkovEntropy")]:
    m = smf.mixedlm(f"{metric} ~ Condition + Experiment", df, groups=df["Animal"]).fit(reml=False)
    b, se, z, p = m.params[pk], m.bse[pk], m.tvalues[pk], m.pvalues[pk]
    gb, gse, gz, gp = gold[name]
    bmatch = "✅" if abs(b - gb) < 0.0001 else "❌"
    print(f"{name:<12} {b:>9.5f} {se:>8.5f} {z:>7.3f} {p:>8.5f}  {gb:>9.5f} {gse:>8.5f} {gz:>7.3f} {gp:>8.5f}  {bmatch}")
