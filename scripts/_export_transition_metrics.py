"""Export per-animal transition metrics to CSV for lme4 in R."""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

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

def lempel_ziv_complexity(seq):
    n = len(seq); i = 0; c = 1; k = 1
    while True:
        if i + k > n: break
        substring = seq[i:i+k]
        found = any(seq[j:j+k] == substring for j in range(i))
        if found:
            k += 1
            if i + k > n: c += 1; break
        else:
            c += 1; i += k; k = 1
        if i >= n: break
    return c

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

print("Computing metrics (recurrence is slow)...")
rows = []
for animal, group in moseq_df.groupby('Animal'):
    R = recurrence_plot(group['Cluster'].tolist())
    rows.append({
        'Animal': animal,
        'Condition': group['Condition'].iloc[0],
        'Experiment': group['Experiment'].iloc[0],
        'LZ_complexity': lempel_ziv_complexity(group['Cluster'].tolist()),
        'Recurrence_Rate': recurrence_rate(R),
        'Determinism': determinism(R),
    })
out = pd.DataFrame(rows)

entropy_df = compute_markov_entropy_index(moseq_df)
out = out.merge(entropy_df, on='Animal')

out_path = 'd:/coping-dynamics-sequencing/results/figure_data/transition_metrics_per_animal.csv'
out.to_csv(out_path, index=False)
print(f"Saved {len(out)} rows to {out_path}")
print(out[['Animal','Condition','Experiment','Recurrence_Rate','Determinism','MarkovEntropy']].to_string())
