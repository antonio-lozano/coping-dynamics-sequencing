"""
Check if excluding unmapped syllables ("" cluster) from sequences
changes per-animal metric distributions enough to explain the SE gap.
The gold SE implies residual SD ~0.038 for Recurrence; our data gives ~0.049.
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
def compute_markov_entropy_index(seqs_by_animal, df_meta, smoothing_factor=0.01):
    results = []
    for animal, states in seqs_by_animal.items():
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
gold = {"Recurrence": (0.01344, 0.00594, 0.024),
        "Determinism": (0.01610, 0.00476, 0.001),
        "Markov": (-0.04119, 0.01759, 0.019)}

for label, include_empty in [("WITH unmapped ('')", True), ("WITHOUT unmapped ('')", False)]:
    print(f"\n=== {label} ===")
    rr_list, det_list = [], []
    seqs_by_animal = {}
    for animal, group in moseq_df.sort_values('Time_bin').groupby('Animal'):
        seq = group['Cluster'].tolist()
        if not include_empty:
            seq = [s for s in seq if s != ""]
        if len(seq) < 2:
            continue
        seqs_by_animal[animal] = seq
        R = recurrence_plot(seq)
        meta = {'Animal': animal, 'Condition': group['Condition'].iloc[0],
                'Experiment': group['Experiment'].iloc[0]}
        rr_list.append({**meta, 'Recurrence_Rate': recurrence_rate(R)})
        det_list.append({**meta, 'Determinism': determinism(R)})

    rr_df = pd.DataFrame(rr_list)
    det_df = pd.DataFrame(det_list)
    meta_df = moseq_df[['Animal','Condition','Experiment']].drop_duplicates()
    entropy_df = compute_markov_entropy_index(seqs_by_animal, meta_df)
    entropy_merged = pd.merge(entropy_df, meta_df, on='Animal')

    print(f"  N animals: {len(rr_df)}")
    print(f"  Recurrence SD: {rr_df['Recurrence_Rate'].std():.5f}  "
          f"(gold implies ~0.038)")
    print(f"  Determinism SD: {det_df['Determinism'].std():.5f}")
    print(f"  Markov SD: {entropy_merged['MarkovEntropy'].std():.5f}")

    for name, df, metric in [("Recurrence", rr_df, "Recurrence_Rate"),
                              ("Determinism", det_df, "Determinism"),
                              ("Markov", entropy_merged, "MarkovEntropy")]:
        m = smf.mixedlm(f"{metric} ~ Condition + Experiment", df, groups=df["Animal"]).fit(reml=False)
        b, se, z, p = m.params[pk], m.bse[pk], m.tvalues[pk], m.pvalues[pk]
        gb, gse, gp = gold[name]
        flag = "✅" if abs(se - gse) < 0.001 else ("~" if abs(se - gse) < 0.002 else "❌")
        sig  = "✅" if (p < 0.05) == (gp < 0.05) else "❌"
        print(f"  {name:<12} beta={b:.5f} SE={se:.5f}{flag} z={z:.3f} p={p:.5f}{sig}  gold SE={gse:.5f}")
