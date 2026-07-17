"""
Report.xlsx shows parameter name 'Stress' not 'Condition[T.ELS]'.
Test if using a binary numeric Stress variable (0=Control, 1=ELS)
instead of string Condition changes the MixedLM SE/convergence.
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
s2c = {}
for c, ss in CLUSTER_MAP.items():
    for s in ss: s2c[int(s)] = c
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
        counts += smoothing_factor
        p = counts / counts.sum(axis=1, keepdims=True)
        pi = np.array([grp['Cluster'].value_counts().get(s, 0) for s in uniq]) / len(states)
        inner = np.array([-np.sum(row[row > 0] * np.log2(row[row > 0])) for row in p])
        results.append({'Animal': animal, 'MarkovEntropy': np.sum(pi * inner)})
    return pd.DataFrame(results)

print("Computing metrics...")
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

# Add binary Stress variable (0=Control, 1=ELS)
for df in [rr_df, det_df, entropy_merged]:
    df['Stress'] = (df['Condition'] == 'ELS').astype(int)

print(f"\nGold: Recurrence SE=0.00594 z=2.264 p=0.024")
print(f"Gold: Determinism SE=0.00476 z=3.382 p=0.001")
print(f"Gold: Markov      SE=0.01759 z=-2.342 p=0.019")

print(f"\n--- String Condition[T.ELS] ---")
for name, df, metric in [("Recurrence", rr_df, "Recurrence_Rate"),
                          ("Determinism", det_df, "Determinism"),
                          ("Markov", entropy_merged, "MarkovEntropy")]:
    m = smf.mixedlm(f"{metric} ~ Condition + Experiment", df, groups=df["Animal"]).fit(reml=False)
    pk = "Condition[T.ELS]"
    print(f"  {name}: SE={m.bse[pk]:.5f} z={m.tvalues[pk]:.3f} p={m.pvalues[pk]:.5f}")

print(f"\n--- Binary numeric Stress (0/1) ---")
for name, df, metric in [("Recurrence", rr_df, "Recurrence_Rate"),
                          ("Determinism", det_df, "Determinism"),
                          ("Markov", entropy_merged, "MarkovEntropy")]:
    m = smf.mixedlm(f"{metric} ~ Stress + Experiment", df, groups=df["Animal"]).fit(reml=False)
    pk = "Stress"
    conv = getattr(m, 'converged', 'N/A')
    print(f"  {name}: SE={m.bse[pk]:.5f} z={m.tvalues[pk]:.3f} p={m.pvalues[pk]:.5f}  converged={conv}")
