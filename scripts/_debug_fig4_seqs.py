"""Check if pred_sequences vs full_sequences changes determinism/markov."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import SYLLABLE_TIMEBIN_250MS

CLUSTER_MAP = {
    "Freezing": [0, 28], "Sniffing": [18, 20], "Grooming": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27], "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climbing": [111], "Jump": [23, 29, 30, 34],
}

raw = pd.read_csv(SYLLABLE_TIMEBIN_250MS)
raw = raw.rename(columns={"Time Bin": "time_bin", "Condition": "group"})
raw = raw[raw["group"].isin(["Control", "ELS"])].copy()
raw["Animal"] = raw["Animal"].astype(str)
raw["Syllable"] = pd.to_numeric(raw["Syllable"], errors="coerce").astype(int)
s2c = {}
for c, ss in CLUSTER_MAP.items():
    for s in ss: s2c[int(s)] = c
raw["cluster"] = raw["Syllable"].map(s2c).fillna("")

idx = raw.groupby(["Animal", "time_bin"])["Percentage"].idxmax()
pred = raw.loc[idx].sort_values(["Animal", "time_bin"]).reset_index(drop=True)
meta = pred[["Animal", "group", "Experiment"]].drop_duplicates().reset_index(drop=True)

full_sequences = {str(a): grp.sort_values("time_bin")["cluster"].tolist()
                  for a, grp in raw.groupby("Animal", sort=False)}
pred_sequences = {str(a): grp["cluster"].tolist()
                  for a, grp in pred.groupby("Animal", sort=False)}

print(f"Full seq length (first animal): {len(list(full_sequences.values())[0])}")
print(f"Pred seq length (first animal): {len(list(pred_sequences.values())[0])}")

# Check if 250ms file has multiple rows per (animal, time_bin)
n_per_bin = raw.groupby(["Animal", "time_bin"]).size()
print(f"Max rows per (animal, time_bin): {n_per_bin.max()}")
print(f"Mean rows per (animal, time_bin): {n_per_bin.mean():.2f}")


def determinism(seq, min_length=2):
    arr = np.asarray(seq)
    n = len(arr)
    total = 0; diag_sum = 0
    for offset in range(-n + 1, n):
        diag = (arr[:n-abs(offset)] == arr[abs(offset):]) if offset >= 0 else (arr[-offset:] == arr[:n+offset])
        total += (int(diag.sum()) - n) if offset == 0 else int(diag.sum())
        run = 0
        for value in diag:
            if value: run += 1
            else:
                if run >= min_length: diag_sum += run
                run = 0
        if run >= min_length: diag_sum += run
    return float(diag_sum / total) if total > 0 else 0.0


def markov_entropy(seq, smoothing_factor=0.01):
    states = list(seq)
    unique_states = list(pd.unique(pd.Series(states)))
    if len(unique_states) == 1: return 0.0
    idx = {s: i for i, s in enumerate(unique_states)}
    counts = np.zeros((len(unique_states), len(unique_states)), dtype=float)
    for a, b in zip(states, states[1:]):
        counts[idx[a], idx[b]] += 1
    counts += smoothing_factor
    probs = counts / counts.sum(axis=1, keepdims=True)
    value_counts = pd.Series(states).value_counts()
    stationary = np.array([value_counts.get(s, 0) for s in unique_states], dtype=float) / len(states)
    inner = np.array([-np.sum(row[row > 0] * np.log2(row[row > 0])) for row in probs])
    return float(np.sum(stationary * inner))


GOLD_DET = {"beta": 0.016104788, "p": 0.000719144}
GOLD_MAR = {"beta": -0.041186979, "p": 0.019174938}

for seq_name, seqs in [("full", full_sequences), ("pred", pred_sequences)]:
    meta_map = meta.set_index("Animal").to_dict("index")
    rows = []
    # Only compute for a subset for speed check
    for animal, seq in seqs.items():
        info = meta_map[animal]
        rows.append({
            "Animal": animal, "group": info["group"], "Experiment": str(info["Experiment"]),
            "determinism": determinism(seq),
            "markov": markov_entropy(seq),
        })
    df = pd.DataFrame(rows)
    df["group_cat"] = pd.Categorical(df["group"], categories=["Control", "ELS"])

    for metric, gold in [("determinism", GOLD_DET), ("markov", GOLD_MAR)]:
        m = smf.ols(f"{metric} ~ C(group_cat) + Experiment", data=df).fit()
        pk = "C(group_cat)[T.ELS]"
        beta = m.params[pk]; p = m.pvalues[pk]
        diff_pct = abs(beta - gold["beta"]) / abs(gold["beta"]) * 100
        print(f"[{seq_name}] {metric}: beta={beta:.6f} (gold={gold['beta']:.6f}, diff={diff_pct:.1f}%) p={p:.6f}")

print("\nDone.")
