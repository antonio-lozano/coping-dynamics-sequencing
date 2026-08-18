"""Does keeping self-transitions (dwell time) rescue the transition score?

The published feature collapses runs of identical states before counting
transitions, which forces the diagonal to zero and throws away how long each
behaviour persists. Figure 4 shows bout durations differ by group (freeze,
sniff and turn all significant), so that discarded diagonal is where the ELS
signal is most likely to sit. This compares collapsed against non-collapsed
matrices on the same footing as the rest of the sweep.
"""
import gzip
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import UPDATED_MOSEQ_PICKLE  # noqa: E402

EUCLID12 = {"123.3", "134.2", "148.4", "15.4", "150.3", "150.5",
            "159.4", "2.4", "26.4", "43.3", "43.5", "49.6"}
CODES = list(range(1, 8))
BIN = 25 * 30

opener = gzip.open if UPDATED_MOSEQ_PICKLE.suffix == ".gz" else open
with opener(UPDATED_MOSEQ_PICKLE, "rb") as f:
    results = pickle.load(f)

meta, coll, full = [], [], []
for rec, payload in results.items():
    seq = np.asarray(payload.get("syllable", []))
    if len(seq) // BIN == 0:
        continue
    valid = seq[np.isin(seq, CODES)]
    a = np.zeros((7, 7))   # collapsed: bout-to-bout, diagonal is zero
    b = np.zeros((7, 7))   # non-collapsed: frame-to-frame, diagonal is dwell
    if len(valid) >= 2:
        c = valid[np.concatenate(([True], valid[1:] != valid[:-1]))]
        for x, y in zip(c[:-1], c[1:]):
            a[int(x) - 1, int(y) - 1] += 1
        for x, y in zip(valid[:-1], valid[1:]):
            b[int(x) - 1, int(y) - 1] += 1
    meta.append({"animal": str(payload.get("Animal")),
                 "group": str(payload.get("Condition"))})
    coll.append(a)
    full.append(b)

meta = pd.DataFrame(meta)
coll, full = np.array(coll), np.array(full)
y = meta.group.to_numpy()
is_c, is_e = y == "Control", y == "ELS"


def rownorm(m):
    s = m.sum(2, keepdims=True); s[s == 0] = 1
    return m / s


def loocv(Z):
    ok = 0
    for i in range(len(y)):
        k = np.ones(len(y), bool); k[i] = False
        ok += int(LogisticRegression(max_iter=5000).fit(Z[k], y[k])
                  .predict(Z[i:i + 1])[0] == y[i])
    return ok / len(y) * 100


P_coll, P_full = rownorm(coll), rownorm(full)
off = ~np.eye(7, dtype=bool)
sets = {
    "collapsed row-cond (published)": P_coll.reshape(len(P_coll), -1),
    "NON-collapsed row-cond":         P_full.reshape(len(P_full), -1),
    "dwell only (diagonal)":          np.array([np.diag(m) for m in P_full]),
    "NON-collapsed off-diagonal":     P_full[:, off],
    "dwell + collapsed off-diag":     np.hstack([np.array([np.diag(m) for m in P_full]),
                                                 P_coll[:, off]]),
}

rows = []
for name, X in sets.items():
    for metric in ["euclidean", "cityblock", "correlation"]:
        D = cdist(X, X, metric=metric)
        co = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                 normalized_stress=False).fit_transform(D)
        d_c = np.linalg.norm(co - np.median(co[is_c], 0), axis=1)
        d_e = np.linalg.norm(co - np.median(co[is_e], 0), axis=1)
        score = np.log((d_c + 1e-9) / (d_e + 1e-9))
        res = set(meta.animal[is_e & (score < 0)])
        rows.append({"features": name, "metric": metric,
                     "LOOCV_mds2": loocv(co), "LOOCV_full": loocv(X),
                     "welch_p": ttest_ind(score[is_c], score[is_e],
                                          equal_var=False).pvalue,
                     "n_res": len(res),
                     "overlap_%": len(res & EUCLID12) / 12 * 100})

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
print(f"\nany p < 0.05 ? {(out.welch_p < 0.05).any()}")
best = out.loc[out.LOOCV_full.idxmax()]
print(f"best feature-space LOOCV: {best.features} / {best.metric} = {best.LOOCV_full:.1f}%")
out.to_csv(ROOT / "statistics" / "transition_dwell_control.csv", index=False)
