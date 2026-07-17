"""
Profile likelihood SE for MixedLM transition metrics.
For each metric: sweep Condition beta over a grid, refit with it fixed,
collect profile log-likelihoods, fit quadratic → curvature → SE.
Valid even when Hessian is degenerate at the boundary.
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import patsy

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    ("Recurrence", rr_df,          "Recurrence_Rate", 0.01344, 0.00594, 2.264, 0.024),
    ("Determinism", det_df,         "Determinism",     0.01610, 0.00476, 3.382, 0.001),
    ("Markov",      entropy_merged, "MarkovEntropy",  -0.04119, 0.01759,-2.342, 0.019),
]
pk = "Condition[T.ELS]"

def profile_ll_se(formula, df, param_key, beta_mle, llf_max, n_points=40):
    """
    Compute profile log-likelihood SE for param_key.
    Sweeps beta over a grid around MLE, refits with constraint,
    fits quadratic to profile LL, returns SE from curvature.
    """
    # Build design matrices once
    y, X = patsy.dmatrices(formula.split('~')[1].strip() + ' ~ ' +
                            formula.split('~')[1].strip(),
                            df, return_type='dataframe')
    # Use offset approach: add an offset term to fix the condition coef
    # Create a binary ELS indicator
    els = (df['Condition'] == 'ELS').astype(float).values

    def profile_loglik(beta_fixed):
        # Offset the outcome by beta_fixed * ELS indicator
        df2 = df.copy()
        df2['_offset'] = beta_fixed * els
        df2['_y'] = df2[formula.split('~')[0].strip()] - df2['_offset']
        reduced_formula = '_y ~ Experiment'
        try:
            m = smf.mixedlm(reduced_formula, df2, groups=df2['Animal']).fit(reml=False)
            return m.llf
        except Exception:
            return -np.inf

    # Grid around MLE
    half_range = max(abs(beta_mle) * 3, 0.05)
    betas = np.linspace(beta_mle - half_range, beta_mle + half_range, n_points)
    llfs  = np.array([profile_loglik(b) for b in betas])

    # Keep finite values
    mask = np.isfinite(llfs)
    betas_f, llfs_f = betas[mask], llfs[mask]
    if len(betas_f) < 5:
        return np.nan

    # Fit quadratic: ll(b) ≈ a*(b - b_mle)^2 + c
    coeffs = np.polyfit(betas_f - beta_mle, llfs_f, 2)
    curvature = coeffs[0]  # should be negative (concave)
    if curvature >= 0:
        return np.nan
    se = np.sqrt(-1 / (2 * curvature))
    return se

from scipy import stats as spstats

print(f"\n{'Metric':<12}  {'beta':>9}  {'SE(Wald)':>9}  {'SE(profile)':>11}  {'z(profile)':>10}  {'p(profile)':>10}  {'p(gold)':>9}  sig?")
print("-" * 85)

for name, df, metric, gold_b, gold_se, gold_z, gold_p in DATASETS:
    formula = f"{metric} ~ Condition + Experiment"
    m_full = smf.mixedlm(formula, df, groups=df["Animal"]).fit(reml=False)
    beta_mle = m_full.params[pk]
    se_wald  = m_full.bse[pk]
    llf_max  = m_full.llf

    print(f"  computing profile for {name}...", end=' ', flush=True)
    se_prof = profile_ll_se(formula, df, pk, beta_mle, llf_max, n_points=50)
    if np.isnan(se_prof):
        print("FAILED")
        continue
    z_prof = beta_mle / se_prof
    p_prof = 2 * spstats.norm.sf(abs(z_prof))
    sig  = "YES" if p_prof < 0.05 else "NO"
    match = "✅" if (p_prof < 0.05) == (gold_p < 0.05) else "❌"
    print(f"done")
    print(f"  {name:<12}  {beta_mle:>9.5f}  {se_wald:>9.5f}  {se_prof:>11.5f}  "
          f"{z_prof:>10.3f}  {p_prof:>10.5f}  {gold_p:>9.5f}  {sig} {match}")
    print(f"  {'':12}  {'gold:':>9}  {gold_se:>9.5f}  {'':>11}  {gold_z:>10.3f}  {'':>10}  {'':>9}")
    print()
