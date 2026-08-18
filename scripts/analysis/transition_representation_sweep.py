"""Fair version of the transition sweep.

The first pass scored LOOCV on features built from the group medians, which
leaks the labels into the representation and inflates accuracy. Here the
representation is always unsupervised - either the MDS embedding or the raw
feature vector - and the classifier only ever sees that. The dynamics score and
its Welch p are reported too, but they use the group medians by construction, so
they are descriptive of the published recipe rather than independent evidence.
"""
import gzip
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import ttest_ind
from sklearn.exceptions import ConvergenceWarning
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


def load():
    opener = gzip.open if UPDATED_MOSEQ_PICKLE.suffix == ".gz" else open
    with opener(UPDATED_MOSEQ_PICKLE, "rb") as f:
        results = pickle.load(f)
    meta, counts, stat = [], [], []
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
        meta.append({"animal": str(payload.get("Animal")),
                     "group": str(payload.get("Condition"))})
        counts.append(mat)
        pi = np.array([(valid == c).sum() for c in CODES], float)
        stat.append(pi / max(pi.sum(), 1))
    return pd.DataFrame(meta), np.array(counts), np.array(stat)


def features(counts, stat, kind):
    row = counts.sum(2, keepdims=True); row[row == 0] = 1
    P = counts / row
    tot = counts.sum((1, 2), keepdims=True); tot[tot == 0] = 1
    J = counts / tot
    off = ~np.eye(7, dtype=bool)
    return {
        "row-cond (current)":        P.reshape(len(P), -1),
        "row-cond sqrt":             np.sqrt(P).reshape(len(P), -1),
        "joint":                     J.reshape(len(J), -1),
        "joint sqrt":                np.sqrt(J).reshape(len(J), -1),
        "joint sqrt off-diag":       np.sqrt(J[:, off]),
        "joint sqrt + frequencies":  np.hstack([np.sqrt(J).reshape(len(J), -1), np.sqrt(stat)]),
        "frequencies only":          np.sqrt(stat),
    }[kind]


def distance(X, metric):
    if metric != "jensenshannon":
        return cdist(X, X, metric=metric)
    Xp = np.clip(X, 0, None); s = Xp.sum(1, keepdims=True); s[s == 0] = 1; Xp /= s
    n = len(Xp); D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = jensenshannon(Xp[i], Xp[j], base=2)
    return np.nan_to_num(D)


def loocv(Z, y):
    ok = 0
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        ok += int(LogisticRegression(max_iter=5000).fit(Z[m], y[m])
                  .predict(Z[i:i + 1])[0] == y[i])
    return ok / len(y) * 100


meta, counts, stat = load()
y = meta.group.to_numpy()
is_ctrl, is_els = y == "Control", y == "ELS"
rows = []
for kind in ["row-cond (current)", "row-cond sqrt", "joint", "joint sqrt",
             "joint sqrt off-diag", "joint sqrt + frequencies", "frequencies only"]:
    X = features(counts, stat, kind)
    for metric in ["euclidean", "cosine", "correlation", "cityblock", "jensenshannon"]:
        D = distance(X, metric)
        coords = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                     normalized_stress=False).fit_transform(D)
        d_c = np.linalg.norm(coords - np.median(coords[is_ctrl], 0), axis=1)
        d_e = np.linalg.norm(coords - np.median(coords[is_els], 0), axis=1)
        score = np.log((d_c + 1e-9) / (d_e + 1e-9))
        res = set(meta.animal[is_els & (score < 0)])
        rows.append({
            "features": kind, "metric": metric,
            "LOOCV_mds2": loocv(coords, y),
            "LOOCV_fullspace": loocv(X, y),
            "welch_p": ttest_ind(score[is_ctrl], score[is_els], equal_var=False).pvalue,
            "n_res": len(res),
            "overlap_%": len(res & EUCLID12) / 12 * 100,
        })

out = pd.DataFrame(rows)
out.to_csv(
    ROOT / "statistics" / "transition_representation_sweep.csv", index=False)
pd.set_option("display.width", 220)
cur = out[(out.features == "row-cond (current)") & (out.metric == "euclidean")]
print("CURRENT PUBLISHED SETTING")
print(cur.to_string(index=False), "\n")
print("=== ranked by honest Control/ELS separation (LOOCV in feature space) ===")
print(out.sort_values("LOOCV_fullspace", ascending=False).head(10).to_string(index=False))
print("\n=== ranked by LOOCV on the 2-D embedding (what the figure reports) ===")
print(out.sort_values("LOOCV_mds2", ascending=False).head(10).to_string(index=False))
