"""Behavioural Flow Analysis (von Ziegler et al. 2024, Nat Methods) on our data.

Their pipeline, as described in the paper:

  BFA  group difference = Manhattan distance between the two group MEAN
       transition vectors, tested against a null built by permuting the group
       labels, scored with a right-tailed z-test on the null distribution.
       One number, so no multiple-testing correction.

  BFL  per-animal score comparing that animal's transition vector to the two
       group MEDIANS. Used for effect size, and to stratify responders from
       non-responders by whether the score falls inside the control range.

Our repository instead computes Euclidean distances on a row-normalised matrix,
embeds with MDS, and Welch-tests a log-ratio score. This script runs their
method so the two can be compared directly.
"""

import gzip
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from coping_dynamics.config import UPDATED_MOSEQ_PICKLE  # noqa: E402

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
CODES = list(range(1, 8))
BIN = 25 * 30
RNG = np.random.default_rng(13)
N_PERM = 10000

opener = gzip.open if UPDATED_MOSEQ_PICKLE.suffix == ".gz" else open
with opener(UPDATED_MOSEQ_PICKLE, "rb") as f:
    results = pickle.load(f)

meta, counts = [], []
for rec, payload in results.items():
    seq = np.asarray(payload.get("syllable", []))
    if len(seq) // BIN == 0:
        continue
    valid = seq[np.isin(seq, CODES)]
    mat = np.zeros((7, 7))
    if len(valid) >= 2:
        col = valid[np.concatenate(([True], valid[1:] != valid[:-1]))]
        for a, b in zip(col[:-1], col[1:]):
            mat[int(a) - 1, int(b) - 1] += 1
    meta.append({"animal": str(payload.get("Animal")), "group": str(payload.get("Condition"))})
    counts.append(mat)

meta = pd.DataFrame(meta)
counts = np.array(counts)
is_c = (meta.group == "Control").to_numpy()
is_e = (meta.group == "ELS").to_numpy()


def variants(counts):
    row = counts.sum(2, keepdims=True)
    row[row == 0] = 1
    tot = counts.sum((1, 2), keepdims=True)
    tot[tot == 0] = 1
    return {
        "raw transition counts": counts.reshape(len(counts), -1),
        "transition proportions (joint)": (counts / tot).reshape(len(counts), -1),
        "row-conditional probabilities": (counts / row).reshape(len(counts), -1),
    }


def bfa(X, labels):
    """Manhattan distance between group means, tested by label permutation."""
    a, b = labels == "Control", labels == "ELS"
    observed = np.abs(X[a].mean(0) - X[b].mean(0)).sum()
    null = np.empty(N_PERM)
    idx = np.arange(len(labels))
    n_a = a.sum()
    for k in range(N_PERM):
        perm = RNG.permutation(idx)
        pa = perm[:n_a]
        pb = perm[n_a:]
        null[k] = np.abs(X[pa].mean(0) - X[pb].mean(0)).sum()
    z = (observed - null.mean()) / null.std(ddof=1)
    p = (np.sum(null >= observed) + 1) / (N_PERM + 1)
    pct = (np.sum(null < observed) / N_PERM) * 100
    return observed, z, p, pct


def bfl(X, labels):
    """Per-animal Manhattan likeness to the two group medians (log ratio)."""
    med_c = np.median(X[labels == "Control"], axis=0)
    med_e = np.median(X[labels == "ELS"], axis=0)
    d_c = np.abs(X - med_c).sum(1)
    d_e = np.abs(X - med_e).sum(1)
    return np.log((d_c + 1e-9) / (d_e + 1e-9))


labels = meta.group.to_numpy()
print(f"{len(meta)} animals ({is_c.sum()} Control / {is_e.sum()} ELS), {N_PERM} permutations\n")
print("=== BFA: is there a Control vs ELS effect in behavioural flow? ===")
for name, X in variants(counts).items():
    obs, z, p, pct = bfa(X, labels)
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {name:<32s} d={obs:8.3f}  percentile={pct:5.1f}  z={z:6.2f}  p={p:.2e}  {star}")

print("\n=== BFL: resilience stratification ===")
for name, X in variants(counts).items():
    score = bfl(X, labels)
    ctrl = score[is_c]
    lo, hi = ctrl.min(), ctrl.max()
    # von Ziegler: inside the control range = non-responder (control-like).
    inside = set(meta.animal[is_e & (score >= lo) & (score <= hi)])
    below0 = set(meta.animal[is_e & (score < 0)])
    print(f"  {name}")
    print(
        f"     control-range rule : n={len(inside):2d}  "
        f"overlap with Euclid-12 = {len(inside & EUCLID12) / 12 * 100:5.1f}%"
    )
    print(
        f"     score < 0 rule     : n={len(below0):2d}  "
        f"overlap with Euclid-12 = {len(below0 & EUCLID12) / 12 * 100:5.1f}%"
    )
