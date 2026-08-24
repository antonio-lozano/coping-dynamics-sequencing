"""BFA at syllable level, not cluster level.

von Ziegler et al. show that statistical power depends strongly on the number
of clusters, and settle on 25. Our transition analysis uses the seven
ethological clusters, which leaves only 42 off-diagonal transitions. This runs
the same permutation test on the retained MoSeq syllables, which is far closer
to their granularity, to test whether the missing signal is simply a
resolution problem.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
EUCLID12 = {
    "123.3",
    "134.2",
    "148.4",
    "15.4",
    "150.3",
    "150.5",
    "159.4",
    "2.4",
    "26.4",
    "43.3",
    "43.5",
    "49.6",
}
RNG = np.random.default_rng(13)
N_PERM = 10000

df = pd.read_csv(
    ROOT / "data/raw/moseq_syllables_per_frame.csv.gz",
    usecols=["name", "syllable", "group", "frame_index"],
)
print(f"{len(df)} frames, {df.name.nunique()} animals, {df.syllable.nunique()} distinct syllables")

# Keep the syllables the manuscript retains (frequency filter to 99.5% of frames).
usage = df.syllable.value_counts(normalize=True).sort_values(ascending=False)
keep = usage.cumsum().le(0.995)
kept = sorted(usage.index[keep])
print(f"retaining {len(kept)} syllables covering {usage[keep].sum() * 100:.1f}% of frames")
index = {s: i for i, s in enumerate(kept)}
K = len(kept)

meta, mats = [], []
for name, sub in df.sort_values(["name", "frame_index"]).groupby("name", sort=False):
    seq = sub.syllable.to_numpy()
    seq = seq[np.isin(seq, kept)]
    mat = np.zeros((K, K))
    if len(seq) >= 2:
        col = seq[np.concatenate(([True], seq[1:] != seq[:-1]))]
        for a, b in zip(col[:-1], col[1:]):
            mat[index[a], index[b]] += 1
    animal = name.split("Animal ")[-1].split("DLC")[0].replace("_", ".")
    meta.append({"animal": animal, "group": sub.group.iloc[0]})
    mats.append(mat)

meta = pd.DataFrame(meta)
mats = np.array(mats)
labels = meta.group.to_numpy()
print("groups:", meta.group.value_counts().to_dict())
occupied = (mats.sum(0) > 0).sum()
print(f"transition matrix {K}x{K} = {K * K} cells, {occupied} ever observed\n")


def bfa(X, labels):
    a = labels == meta.group.unique()[0]
    observed = np.abs(X[a].mean(0) - X[~a].mean(0)).sum()
    n_a, idx = a.sum(), np.arange(len(labels))
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        p = RNG.permutation(idx)
        null[k] = np.abs(X[p[:n_a]].mean(0) - X[p[n_a:]].mean(0)).sum()
    z = (observed - null.mean()) / null.std(ddof=1)
    pval = (np.sum(null >= observed) + 1) / (N_PERM + 1)
    return observed, z, pval, np.sum(null < observed) / N_PERM * 100


row = mats.sum(2, keepdims=True)
row[row == 0] = 1
tot = mats.sum((1, 2), keepdims=True)
tot[tot == 0] = 1
sets = {
    "raw transition counts": mats.reshape(len(mats), -1),
    "transition proportions": (mats / tot).reshape(len(mats), -1),
    "row-conditional": (mats / row).reshape(len(mats), -1),
}
print("=== BFA at syllable level ===")
for name, X in sets.items():
    obs, z, p, pct = bfa(X, labels)
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {name:<24s} d={obs:9.3f} percentile={pct:5.1f} z={z:6.2f} p={p:.2e} {star}")

print("\n=== BFL stratification (syllable level) ===")
for name, X in sets.items():
    is_c = labels == "Control"
    med_c = np.median(X[is_c], 0)
    med_e = np.median(X[~is_c], 0)
    score = np.log((np.abs(X - med_c).sum(1) + 1e-9) / (np.abs(X - med_e).sum(1) + 1e-9))
    lo, hi = score[is_c].min(), score[is_c].max()
    inside = set(meta.animal[(~is_c) & (score >= lo) & (score <= hi)])
    below = set(meta.animal[(~is_c) & (score < 0)])
    print(f"  {name}")
    print(
        f"     control-range: n={len(inside):2d} overlap={len(inside & EUCLID12) / 12 * 100:5.1f}%"
    )
    print(f"     score<0      : n={len(below):2d} overlap={len(below & EUCLID12) / 12 * 100:5.1f}%")
