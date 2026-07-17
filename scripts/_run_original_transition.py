"""
Run EXACT original transition metrics code (7. Transition metrics.py)
adapted to use repo CSV path and our cluster map instead of JSON.
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

# ── SECTION 1: LZ COMPLEXITY ──────────────────────────────────────────────────
def lempel_ziv_complexity(seq):
    n = len(seq); i = 0; c = 1; k = 1
    while True:
        if i + k > n: break
        substring = seq[i:i+k]
        found = any(seq[j:j+k] == substring for j in range(i))
        if found:
            k += 1
            if i + k > n:
                c += 1; break
        else:
            c += 1; i += k; k = 1
        if i >= n: break
    return c

lz_results = []
for animal, group in moseq_df.groupby('Animal'):
    seq = group['Cluster'].tolist()
    lz_results.append({
        'Animal': animal,
        'Condition': group['Condition'].iloc[0],
        'Experiment': group['Experiment'].iloc[0],
        'LZ_complexity': lempel_ziv_complexity(seq)
    })
lz_df = pd.DataFrame(lz_results)
model_lz = smf.mixedlm("LZ_complexity ~ Condition + Experiment", lz_df, groups=lz_df["Animal"]).fit(reml=False)
pk = "Condition[T.ELS]"
print(f"LZ:          beta={model_lz.params[pk]:.5f} SE={model_lz.bse[pk]:.5f} z={model_lz.tvalues[pk]:.3f} p={model_lz.pvalues[pk]:.5f}")
print(f"  Gold:      beta=-8.04878 SE=5.56673 z=-1.446 p=0.14821")

# ── SECTION 2: RECURRENCE RATE ────────────────────────────────────────────────
def recurrence_plot(seq):
    seq = np.array(seq); n = len(seq)
    R = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if seq[i] == seq[j]: R[i, j] = 1
    return R

def recurrence_rate(R):
    return R.sum() / (R.shape[0] ** 2)

print("\nComputing recurrence rate (slow O(n^2) per animal)...")
rr_results = []
for animal, group in moseq_df.groupby('Animal'):
    R = recurrence_plot(group['Cluster'].tolist())
    rr_results.append({
        'Animal': animal,
        'Condition': group['Condition'].iloc[0],
        'Experiment': group['Experiment'].iloc[0],
        'Recurrence_Rate': recurrence_rate(R)
    })
rr_df = pd.DataFrame(rr_results)
model_rr = smf.mixedlm("Recurrence_Rate ~ Condition + Experiment", rr_df, groups=rr_df["Animal"]).fit(reml=False)
print(f"Recurrence:  beta={model_rr.params[pk]:.5f} SE={model_rr.bse[pk]:.5f} z={model_rr.tvalues[pk]:.3f} p={model_rr.pvalues[pk]:.5f}")
print(f"  Gold:      beta=0.01344 SE=0.00594 z=2.264 p=0.02359")

# ── SECTION 3: DETERMINISM ────────────────────────────────────────────────────
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

det_results = []
for animal, group in moseq_df.groupby('Animal'):
    R = recurrence_plot(group['Cluster'].tolist())
    det_results.append({
        'Animal': animal,
        'Condition': group['Condition'].iloc[0],
        'Experiment': group['Experiment'].iloc[0],
        'Determinism': determinism(R)
    })
det_df = pd.DataFrame(det_results)
model_det = smf.mixedlm("Determinism ~ Condition + Experiment", det_df, groups=det_df["Animal"]).fit(reml=False)
print(f"Determinism: beta={model_det.params[pk]:.5f} SE={model_det.bse[pk]:.5f} z={model_det.tvalues[pk]:.3f} p={model_det.pvalues[pk]:.5f}")
print(f"  Gold:      beta=0.01610 SE=0.00476 z=3.382 p=0.00072")

# ── SECTION 4: MARKOV ENTROPY ─────────────────────────────────────────────────
def compute_markov_entropy_index(df, animal_col='Animal', state_col='Cluster',
                                  time_col='Time_bin', smoothing_factor=0.01):
    results = []
    df_sorted = df.sort_values([animal_col, time_col])
    for animal, grp in df_sorted.groupby(animal_col):
        states = grp[state_col].tolist()
        uniq = list(pd.unique(states))
        if len(uniq) == 1:
            results.append({animal_col: animal, 'MarkovEntropy': 0.0}); continue
        idx = {s: i for i, s in enumerate(uniq)}; M = len(uniq)
        counts = np.zeros((M, M), dtype=float)
        for a, b in zip(states, states[1:]): counts[idx[a], idx[b]] += 1
        counts += smoothing_factor
        p = counts / counts.sum(axis=1, keepdims=True)
        pi = np.array([grp[state_col].value_counts().get(s, 0) for s in uniq]) / len(states)
        inner = np.array([-np.sum(row[row > 0] * np.log2(row[row > 0])) for row in p])
        entropy = np.sum(pi * inner)
        results.append({animal_col: animal, 'MarkovEntropy': entropy})
    return pd.DataFrame(results)

entropy_df = compute_markov_entropy_index(moseq_df)
entropy_merged = pd.merge(entropy_df, moseq_df[['Animal','Condition','Experiment']].drop_duplicates(), on='Animal', how='left')
model_me = smf.mixedlm("MarkovEntropy ~ Condition + Experiment", entropy_merged, groups=entropy_merged['Animal']).fit(reml=False)
print(f"Markov:      beta={model_me.params[pk]:.5f} SE={model_me.bse[pk]:.5f} z={model_me.tvalues[pk]:.3f} p={model_me.pvalues[pk]:.5f}")
print(f"  Gold:      beta=-0.04119 SE=0.01759 z=-2.342 p=0.01917")
