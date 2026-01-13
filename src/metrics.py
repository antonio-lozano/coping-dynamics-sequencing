"""Common metrics utilities."""
import numpy as np
import pandas as pd


def compute_transition_matrix(seq):
    """Compute transition counts after removing consecutive duplicates."""
    if len(seq) == 0:
        return pd.DataFrame()
    filtered = [seq[0]]
    for s in seq[1:]:
        if s != filtered[-1]:
            filtered.append(s)
    pairs = list(zip(filtered[:-1], filtered[1:]))
    states = sorted(set(filtered))
    TM = pd.DataFrame(0, index=states, columns=states, dtype=float)
    for a, b in pairs:
        TM.loc[a, b] += 1
    return TM


def diversity_scores(seq):
    """Return Simpson, Shannon, and evenness for a sequence of labels."""
    vals, counts = np.unique(seq, return_counts=True)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    p = counts / counts.sum()
    simpson = 1 - np.sum(p ** 2)
    shannon = -np.sum(p * np.log(p))
    evenness = shannon / np.log(len(vals)) if len(vals) > 0 else np.nan
    return simpson, shannon, evenness
