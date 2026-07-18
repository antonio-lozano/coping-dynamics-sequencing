# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""
Statistical analysis functions for coping-dynamics-sequencing manuscript.

This module provides reusable statistical functions used across figure generation scripts,
extracted from build_freezing_data_workbooks.py for better code organization and testing.

All functions maintain exact formulas and ddof specifications from original analysis.
No random seed is set here; seeding (if needed) is caller's responsibility.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation with ddof=1 (unbiased estimator for each group).

    Parameters
    ----------
    a : pd.Series
        First group values.
    b : pd.Series
        Second group values.

    Returns
    -------
    float
        Cohen's d effect size, or NaN if insufficient data.
    """
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled if pooled else np.nan


def summarize_by_group(
    df: pd.DataFrame,
    value_cols: list[str],
    group_cols: list[str],
    group_col: str = "group",
) -> pd.DataFrame:
    """
    Compute summary statistics (mean, SD, SEM, n) per group.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    value_cols : list[str]
        Columns to summarize (numeric).
    group_cols : list[str]
        Columns to group by.
    group_col : str, optional
        Column name for the main grouping variable (default: "group").

    Returns
    -------
    pd.DataFrame
        Summary with columns: group_cols + group_col + metric + [n, mean, sd, sem].
    """
    rows: list[dict] = []
    for keys, sub in df.groupby(group_cols, dropna=False) if group_cols else [((), df)]:
        if group_cols and not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys)) if group_cols else {}
        for value_col in value_cols:
            values = pd.to_numeric(sub[value_col], errors="coerce").dropna()
            rows.append(
                {
                    **base,
                    "metric": value_col,
                    "n": int(values.count()),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "sem": float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def fit_mixed_models(raw_df: pd.DataFrame, response_col: str) -> pd.DataFrame:
    """
    Fit mixed-effects linear models for timecourse data.

    Fits separate models for Exp1, Exp3, and combined data. Used for freezing and
    behavioral timecourse analysis (Figure 2, 3, 5, 6).

    Parameters
    ----------
    raw_df : pd.DataFrame
        Input data with columns: animal_id, group, experiment, time_bin_numeric, response_col.
    response_col : str
        Name of the response variable column.

    Returns
    -------
    pd.DataFrame
        Model results with columns: analysis, parameter, coef, std_err, z, p_value, ci_low, ci_high,
        aic, bic, log_likelihood, n_observations, n_animals.
    """
    model_rows: list[dict] = []
    data = raw_df.rename(columns={"group": "condition", response_col: "response"}).copy()
    data["condition"] = pd.Categorical(data["condition"], categories=["Control", "ELS"])

    # Per-experiment models: no vc_formula (matches original Ground_truth.py / Syllables_analysis.py)
    # Combined model: vc_formula={"experiment": "0 + C(experiment)"} controls for experiment as random effect
    definitions = [
        ("Exp1", data[data["experiment"] == 1], "response ~ condition * time_bin_numeric", False),
        ("Exp3", data[data["experiment"] == 3], "response ~ condition * time_bin_numeric", False),
        ("Combined", data, "response ~ condition * time_bin_numeric", True),
    ]

    for analysis, sub, formula, use_vc in definitions:
        try:
            model_kwargs = dict(groups=sub["animal_id"], re_formula="1")
            if use_vc:
                model_kwargs["vc_formula"] = {"experiment": "0 + C(experiment)"}
            md = smf.mixedlm(formula, sub, **model_kwargs)
            try:
                model = md.fit(reml=False)
            except Exception:
                # The default optimizer hits a singular Hessian at the
                # random-effect variance boundary for some clusters (e.g.
                # Locomotion/Climb/Jump combined); lbfgs converges and
                # reproduces the manuscript estimates.
                model = md.fit(reml=False, method="lbfgs")

            conf = model.conf_int()
            for param in model.params.index:
                model_rows.append(
                    {
                        "analysis": analysis,
                        "parameter": param.replace("condition", "group"),
                        "coef": model.params.get(param),
                        "std_err": model.bse.get(param),
                        "z": model.tvalues.get(param),
                        "p_value": model.pvalues.get(param),
                        "ci_low": conf.loc[param, 0] if param in conf.index else np.nan,
                        "ci_high": conf.loc[param, 1] if param in conf.index else np.nan,
                        "aic": model.aic,
                        "bic": model.bic,
                        "log_likelihood": model.llf,
                        "n_observations": int(model.nobs),
                        "n_animals": int(sub["animal_id"].nunique()),
                    }
                )
        except Exception as err:
            model_rows.append(
                {
                    "analysis": analysis,
                    "parameter": "model_failed",
                    "coef": np.nan,
                    "std_err": np.nan,
                    "z": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                    "log_likelihood": np.nan,
                    "n_observations": int(len(sub)),
                    "n_animals": int(sub["animal_id"].nunique()),
                    "note": str(err),
                }
            )

    return pd.DataFrame(model_rows)


# ============================================================================
# Diversity and complexity metrics (Figure 4, 6)
# ============================================================================


def compute_diversity_metrics(sequence: list[str]) -> dict[str, float]:
    """
    Compute Shannon entropy, Simpson diversity, evenness, and CUI for a behavior sequence.

    Parameters
    ----------
    sequence : list[str]
        List of behavior cluster names.

    Returns
    -------
    dict[str, float]
        Keys: shannon_entropy_index, simpson_index, evenness_index, cumulative_usage_index (CUI).
    """
    vals, counts = np.unique(sequence, return_counts=True)
    counts_map = dict(zip(vals, counts))
    order = sorted(counts_map)

    # Probability distribution
    probabilities = np.array([counts_map.get(cluster, 0) for cluster in order], dtype=float)
    probabilities = probabilities / probabilities.sum()

    # Shannon entropy (using ln, then normalize by log(n_clusters))
    nonzero = probabilities[probabilities > 0]
    shannon = stats.entropy(nonzero)  # Uses ln by default
    evenness = shannon / np.log(len(nonzero)) if len(nonzero) else np.nan

    # Simpson diversity
    simpson = 1 - np.sum(probabilities**2)

    # Cumulative Usage Index (CUI)
    named = np.array([counts_map.get(cluster, 0) for cluster in order if str(cluster).strip() != ""], dtype=float)
    named = named / counts.sum() if len(named) > 0 else named
    sorted_named = np.sort(named)[::-1] if len(named) > 0 else np.array([])
    cumulative = np.cumsum(sorted_named) if len(sorted_named) > 0 else np.array([])
    baseline = (len(cumulative) + 1) / (2 * len(cumulative)) if len(cumulative) > 0 else np.nan
    cui = (cumulative.mean() - baseline) / (1 - baseline) if len(cumulative) > 0 and baseline < 1 else np.nan

    return {
        "shannon_entropy_index": shannon,
        "simpson_index": simpson,
        "evenness_index": evenness,
        "cumulative_usage_index": cui,
    }


def compute_bout_duration(sequence: list[str]) -> pd.DataFrame:
    """
    Compute mean bout duration for each behavior cluster.

    A bout is a contiguous sequence of the same behavior.

    Parameters
    ----------
    sequence : list[str]
        List of behavior cluster names (time resolution: 250 ms per frame).

    Returns
    -------
    pd.DataFrame
        Columns: cluster, bout_duration_seconds (duration = n_frames * 0.25).
    """
    rows = []
    if not sequence:
        return pd.DataFrame(rows)

    prev = sequence[0]
    length = 1
    for cluster in sequence[1:] + ["__END__"]:
        if cluster == prev:
            length += 1
            continue
        rows.append(
            {
                "cluster": prev if str(prev).strip() else "Unmapped_or_excluded",
                "bout_duration_seconds": length * 0.25,
            }
        )
        prev = cluster
        length = 1

    return pd.DataFrame(rows)


# ============================================================================
# Transition sequence complexity metrics (Figure 4, 6)
# ============================================================================


def lempel_ziv_complexity(sequence: list[str]) -> int:
    """
    Compute Lempel-Ziv complexity of a sequence.

    Measures the minimum number of unique patterns needed to describe the sequence.
    Higher values indicate more complex, less predictable behavior sequences.

    Parameters
    ----------
    sequence : list[str]
        List of behavior cluster names.

    Returns
    -------
    int
        Lempel-Ziv complexity.
    """
    token_map = {value: i for i, value in enumerate(pd.unique(pd.Series(sequence)))}
    tokens = [token_map[x] for x in sequence]
    n, i, c, k = len(tokens), 0, 1, 1

    while True:
        if i + k > n:
            break
        sub = tokens[i : i + k]
        found = any(tokens[j : j + k] == sub for j in range(i))
        if found:
            k += 1
            if i + k > n:
                c += 1
                break
        else:
            c += 1
            i += k
            k = 1
        if i >= n:
            break

    return c


def recurrence_rate(sequence: list[str]) -> float:
    """
    Compute recurrence rate of a sequence.

    Proportion of time an animal repeats the same behavior consecutively.
    Higher values indicate more repetitive behavior.

    Parameters
    ----------
    sequence : list[str]
        List of behavior cluster names.

    Returns
    -------
    float
        Recurrence rate in [0, 1].
    """
    _, counts = np.unique(sequence, return_counts=True)
    n = len(sequence)
    return float(np.sum(counts * counts) / (n * n))


def determinism(sequence: list[str], min_length: int = 2) -> float:
    """
    Compute determinism of a sequence.

    Measures the proportion of sequence composed of repeated patterns.
    Higher values indicate more deterministic, predictable sequences.

    Parameters
    ----------
    sequence : list[str]
        List of behavior cluster names.
    min_length : int, optional
        Minimum pattern length (default: 2).

    Returns
    -------
    float
        Determinism in [0, 1].
    """
    arr = np.asarray(sequence)
    n = len(arr)
    total = 0
    diag_sum = 0

    for offset in range(-n + 1, n):
        diag = arr[: n - abs(offset)] == arr[abs(offset) :] if offset >= 0 else arr[-offset:] == arr[: n + offset]
        if offset == 0:
            total += int(diag.sum()) - n
        else:
            total += int(diag.sum())

        run = 0
        for value in diag:
            if value:
                run += 1
            else:
                if run >= min_length:
                    diag_sum += run
                run = 0
        if run >= min_length:
            diag_sum += run

    return float(diag_sum / total) if total > 0 else 0.0


def markov_entropy(sequence: list[str], smoothing_factor: float = 0.01) -> float:
    """
    Compute Markov entropy of a sequence based on transition probabilities.

    Measures uncertainty in behavioral transitions. Higher entropy indicates
    more variable, less predictable transitions between states.

    Parameters
    ----------
    sequence : list[str]
        List of behavior cluster names.
    smoothing_factor : float, optional
        Laplace smoothing factor (default: 0.01).

    Returns
    -------
    float
        Markov entropy.
    """
    unique_states = list(pd.unique(pd.Series(sequence)))
    if len(unique_states) == 1:
        return 0.0

    idx = {state: i for i, state in enumerate(unique_states)}
    counts = np.zeros((len(unique_states), len(unique_states)), dtype=float)

    for a, b in zip(sequence[:-1], sequence[1:]):
        counts[idx[a], idx[b]] += 1

    counts += smoothing_factor
    probs = counts / counts.sum(axis=1, keepdims=True)

    value_counts = pd.Series(sequence).value_counts()
    stationary = np.array([value_counts.get(state, 0) for state in unique_states], dtype=float) / len(sequence)

    inner = np.array([-np.sum(row[row > 0] * np.log2(row[row > 0])) for row in probs])

    return float(np.sum(stationary * inner))


def transition_sequence_metrics(sequence: list[str]) -> dict[str, float]:
    """
    Compute all transition-based complexity metrics for a sequence.

    Parameters
    ----------
    sequence : list[str]
        List of behavior cluster names.

    Returns
    -------
    dict[str, float]
        Keys: lempel_ziv_complexity, recurrence_rate, determinism, markov_entropy.
    """
    return {
        "lempel_ziv_complexity": lempel_ziv_complexity(sequence),
        "recurrence_rate": recurrence_rate(sequence),
        "determinism": determinism(sequence),
        "markov_entropy": markov_entropy(sequence, smoothing_factor=0.01),
    }


def metric_summary_and_tests(
    df: pd.DataFrame,
    value_col: str,
    by_cols: list[str],
    group_col: str = "group",
    group_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute summary statistics and pairwise statistical tests.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    value_col : str
        Column to summarize.
    by_cols : list[str]
        Columns to group by.
    group_col : str, optional
        Group variable column name (default: "group").
    group_order : list[str], optional
        Order of groups for comparisons (default: ["Control", "ELS"] or ["Control", "ELS", "ELS resilient"]).

    Returns
    -------
    pd.DataFrame
        Summary statistics and test results.
    """
    rows: list[dict] = []

    if group_order is None:
        # Infer from data
        unique_groups = sorted(df[group_col].dropna().unique().tolist())
        if "ELS resilient" in unique_groups:
            group_order = ["Control", "ELS", "ELS resilient"]
        else:
            group_order = ["Control", "ELS"]

    for keys, sub in df.groupby(by_cols, dropna=False) if by_cols else [((), df)]:
        if by_cols and not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(by_cols, keys)) if by_cols else {}

        # Summary per group
        for group in group_order:
            vals = pd.to_numeric(sub[sub[group_col] == group][value_col], errors="coerce").dropna()
            rows.append(
                {
                    **base,
                    "comparison_type": "summary",
                    "group": group,
                    "comparison": "",
                    "n": int(vals.count()),
                    "mean": float(vals.mean()) if len(vals) else np.nan,
                    "sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                    "sem": float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else np.nan,
                    "welch_t": np.nan,
                    "p_value": np.nan,
                    "cohens_d_left_minus_right": np.nan,
                }
            )

        # Pairwise comparisons
        for left_i, left in enumerate(group_order):
            for right in group_order[left_i + 1 :]:
                left_vals = pd.to_numeric(sub[sub[group_col] == left][value_col], errors="coerce").dropna()
                right_vals = pd.to_numeric(sub[sub[group_col] == right][value_col], errors="coerce").dropna()

                if len(left_vals) > 0 and len(right_vals) > 0:
                    t_stat, p_value = stats.ttest_ind(left_vals, right_vals, equal_var=False, nan_policy="omit")
                    d = cohens_d(left_vals, right_vals)

                    rows.append(
                        {
                            **base,
                            "comparison_type": "test",
                            "group": "",
                            "comparison": f"{left} vs {right}",
                            "n": int(len(left_vals) + len(right_vals)),
                            "mean": np.nan,
                            "sd": np.nan,
                            "sem": np.nan,
                            "welch_t": t_stat,
                            "p_value": p_value,
                            "cohens_d_left_minus_right": d,
                        }
                    )

    return pd.DataFrame(rows)
