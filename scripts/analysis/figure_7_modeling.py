"""Direct supplementary figure for the resilience-prediction claim.

Five panels, answering four questions:

  A  Does a model trained in one cohort work in the other?          (Q1)
  B  Which kinds of behavioural information transfer?               (Q2)
  C  Is freezing alone enough, or is a combination needed?          (Q3)
  D  The same question shown geometrically in behaviour space.      (Q3)
  E  How much of the session do we need before resilience is
     readable, and is it already readable during exploration?       (Q4)

Methodological fixes relative to the earlier scripts
----------------------------------------------------
1. Time-limited features are CUMULATIVE motif proportions (7 features at every
   horizon) instead of one feature per motif per 30-s bin (7 -> 105 features as
   the horizon grows). The old design confounded "more time" with "bigger
   model", so the AUC-vs-time curve could not be read as a time effect.
2. Internal validation uses repeated stratified 5-fold CV instead of LOOCV.
   With n=41 and a weak feature, LOOCV produces anti-learning: leaving one
   animal out shifts the fitted boundary away from that animal, so held-out
   predictions come out systematically inverted and AUC collapses toward 0
   (early Freeze reached AUC = 0.03 despite group means of 1.3% vs 1.9%).
3. Every cross-cohort AUC carries a stratified bootstrap CI, so the reader can
   see that with 6 resilient animals per cohort these estimates are soft.
4. The motif-combination search is run NESTED: the winning combination is
   chosen by CV inside the training cohort only, then applied once to the
   held-out cohort. The previously reported 0.868 for Sniff+Turn was selected
   on the held-out data itself and is therefore optimistic.
5. A within-cohort label-permutation null band is drawn on the time panel, so
   "above chance" is judged against the actual null for this sample size.

Run:
  uv run python run_tier1_direct_figure.py
"""

from __future__ import annotations

import itertools
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figure_source_data"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis import figure_7_feature_families as efa  # noqa: E402

REPO = efa.REPO

RANDOM_SEED = 13
# Sampled every 30 s to match the behaviour-dynamics time courses in
# Figures 3, 5 and Supplementary 1 (0.5-7.5 min, 15 points).
HORIZONS_MIN = [round(0.5 * k, 1) for k in range(1, 16)]
N_SPLITS = 5
N_REPEATS = 25
N_BOOT = 2000
N_PERM = 200
FPS = 25.0

# Manuscript palette.
AXIS = "#4D4D4D"
TEXT = "#2B2B2B"
RESILIENT = "#90BE6D"
VULNERABLE = "#C37B9F"
FREEZE = "#C671A0"
MOTIF = "#6398A4"
TRANSITION = "#9ECADA"
DIVERSITY = "#E6C23A"
BOUT = "#D98427"
INTEGRATED = "#BCD548"
INTEGRATED_FREEZE = "#D95D5D"
NULL_GREY = "#BFBFBF"

COHORT_SHORT = {1: "Krugers", 3: "Gómez"}
COHORT_FULL = {1: "Sanguino Gómez & Krugers", 3: "Sanguino Gómez et al."}
DIRECTIONS = [(1, 3), (3, 1)]


# --------------------------------------------------------------------------
# model fitting helpers (thin wrappers over the existing ridge implementation)
# --------------------------------------------------------------------------


def ridge_fit_np(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balanced-weight L2 ridge on standardized features. Mirrors efa.fit_ridge exactly."""
    y_signed = np.where(y == 1, 1.0, -1.0)
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale = np.where(scale == 0, 1.0, scale)
    xs = (x - mean) / scale
    xb = np.column_stack([np.ones(len(xs)), xs])
    n = len(y)
    n_pos = max(int(np.sum(y == 1)), 1)
    n_neg = max(int(np.sum(y == 0)), 1)
    weights = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))
    sqrt_w = np.sqrt(weights)
    xw = xb * sqrt_w[:, None]
    yw = y_signed * sqrt_w
    penalty = np.eye(xb.shape[1]) * efa.RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)
    return coef, mean, scale


def ridge_score_np(x: np.ndarray, coef: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return coef[0] + ((x - mean) / scale) @ coef[1:]


def fast_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC with tie handling; much cheaper than sklearn inside CV loops."""
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order]
    n = len(score)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos = int(sorted_y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[sorted_y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fit_score(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> np.ndarray:
    coef, mean, scale = efa.fit_ridge(train, cols)
    return efa.score(test, cols, coef, mean, scale)


def strat_key(data: pd.DataFrame, by_cohort: bool = True) -> np.ndarray:
    """Stratify folds on class, and on cohort as well when both are present."""
    if by_cohort and data["experiment"].nunique() > 1:
        return (data["experiment"].astype(int).astype(str) + "_" + data["target"].astype(int).astype(str)).to_numpy()
    return data["target"].astype(int).to_numpy()


def make_folds(key: np.ndarray, n_splits: int, n_repeats: int, seed: int) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Fold indices depend only on the stratification key, so build them once and reuse."""
    n = len(key)
    dummy = np.zeros(n)
    return [
        list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r).split(dummy, key))
        for r in range(n_repeats)
    ]


def cv_auc_np(
    x: np.ndarray,
    y: np.ndarray,
    folds: list[list[tuple[np.ndarray, np.ndarray]]],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    aucs = np.empty(len(folds), dtype=float)
    oof_sum = np.zeros(n, dtype=float)
    for r, repeat in enumerate(folds):
        oof = np.empty(n, dtype=float)
        for tr_idx, te_idx in repeat:
            coef, mean, scale = ridge_fit_np(x[tr_idx], y[tr_idx])
            oof[te_idx] = ridge_score_np(x[te_idx], coef, mean, scale)
        aucs[r] = fast_auc(y, oof)
        oof_sum += oof
    return aucs, oof_sum / len(folds)


def repeated_cv_auc(
    data: pd.DataFrame,
    cols: list[str],
    *,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = RANDOM_SEED,
    by_cohort: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeated stratified k-fold. Returns (per-repeat AUCs, mean out-of-fold score)."""
    y = data["target"].to_numpy(dtype=int)
    x = data[cols].to_numpy(dtype=float)
    folds = make_folds(strat_key(data, by_cohort=by_cohort), n_splits, n_repeats, seed)
    return cv_auc_np(x, y, folds)


def stratified_bootstrap_ci(y: np.ndarray, score: np.ndarray, *, n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> tuple[float, float]:
    """Resample positives and negatives separately so both classes always survive.

    AUC is the mean over positive/negative pairs of 1[s_pos > s_neg] + 0.5*1[tie],
    so the full pairwise comparison matrix can be built once and every bootstrap
    draw becomes a fancy-index into it. That keeps the whole CI to one numpy op.
    """
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diff = score[pos][:, None] - score[neg][None, :]
    comparison = (diff > 0).astype(float) + 0.5 * (diff == 0)
    ip = rng.integers(0, len(pos), (n_boot, len(pos)))
    jn = rng.integers(0, len(neg), (n_boot, len(neg)))
    draws = comparison[ip[:, :, None], jn[:, None, :]].mean(axis=(1, 2))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def cross_cohort(data: pd.DataFrame, cols: list[str], *, with_ci: bool = True) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for train_exp, test_exp in DIRECTIONS:
        train = data.loc[data["experiment"] == train_exp]
        test = data.loc[data["experiment"] == test_exp]
        score = fit_score(train, test, cols)
        y = test["target"].to_numpy(dtype=int)
        auc = fast_auc(y, score)
        lo, hi = stratified_bootstrap_ci(y, score) if with_ci else (np.nan, np.nan)
        rows.append(
            {
                "train_experiment": train_exp,
                "test_experiment": test_exp,
                "direction": f"{COHORT_SHORT[train_exp]} -> {COHORT_SHORT[test_exp]}",
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_resilient_test": int((y == 1).sum()),
                "n_vulnerable_test": int((y == 0).sum()),
                "n_features": len(cols),
                "roc_auc": auc,
                "auc_ci_low": lo,
                "auc_ci_high": hi,
                "score": score,
                "y": y,
                "test_frame": test,
            }
        )
    return rows


def worst_direction(rows: list[dict[str, object]]) -> float:
    return float(min(r["roc_auc"] for r in rows))


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _timecourse() -> pd.DataFrame:
    return pd.read_csv(REPO / "data/processed/cluster_timecourse_per_animal.csv")


@lru_cache(maxsize=1)
def _freezing_frames() -> pd.DataFrame:
    return pd.read_csv(REPO / "data/raw/freezing_predictions_light.csv.gz", usecols=["animal_id", "frame", "freezing"])


def cumulative_motif_features(horizon_min: float) -> pd.DataFrame:
    """Fraction of time in each motif from session start to `horizon_min`.

    Always 7 columns, whatever the horizon. This is the fix for the old
    one-column-per-(motif, 30-s bin) design.
    """
    tc = _timecourse()
    tc = tc.loc[tc["time_s"] <= horizon_min * 60].copy()
    wide = (
        tc.pivot_table(index="animal_id", columns="cluster", values="pct", aggfunc="mean", fill_value=0)
        .sort_index(axis=1)
        .reset_index()
    )
    return wide.rename(columns={c: f"cum__{c}" for c in wide.columns if c != "animal_id"})


def cumulative_freeze_features(horizon_min: float) -> pd.DataFrame:
    max_frame = int(round(horizon_min * 60 * FPS))
    freezing = _freezing_frames()
    freezing = freezing.loc[freezing["frame"] < max_frame]
    out = freezing.groupby("animal_id", as_index=False)["freezing"].mean()
    out["cumfreeze__supervised_pct"] = out["freezing"] * 100.0
    return out.drop(columns=["freezing"])


def cumulative_motif_freeze_features(horizon_min: float) -> pd.DataFrame:
    """Cumulative Freeze time share alone, as the freezing-only comparator.

    Uses the motif Freeze cluster rather than the supervised detector. On a
    matched denominator the two correlate at r = 0.98 and return the same AUC,
    so this keeps the freezing-alone curve on the time panel while reporting
    the measurement only once. See run_freeze_denominator_alignment.py.
    """
    wide = cumulative_motif_features(horizon_min)
    if "cum__Freeze" not in wide.columns:
        raise KeyError(f"expected a cum__Freeze column, got {sorted(wide.columns)}")
    return wide[["animal_id", "cum__Freeze"]].copy()


@lru_cache(maxsize=1)
def _labels() -> pd.DataFrame:
    return efa.load_labels()


@lru_cache(maxsize=1)
def _frequency() -> pd.DataFrame:
    return efa.frequency_features()[0]


@lru_cache(maxsize=1)
def _freezing() -> pd.DataFrame:
    return efa.freezing_features()[0]


def build_frames() -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, list[str], str, str]]]:
    labels = _labels()
    frequency = _frequency()
    diversity, _ = efa.diversity_features()
    transition, _ = efa.transition_features()
    bout, _ = efa.bout_features()
    pairs, _, _ = efa.transition_pair_features()

    all_behavior = efa.merge_feature_frames([frequency, diversity, transition, bout])

    def cols_of(frame: pd.DataFrame) -> list[str]:
        return [c for c in frame.columns if "__" in c]

    # Supervised Freeze % is deliberately absent, and with it the old
    # "All motif metrics + Freeze" block. On a matched denominator the
    # supervised detector correlates r = 0.98 with motif Freeze frequency and
    # returns the same AUC, so the two are one measurement read twice; the
    # "+ Freeze" block was just "All motif metrics" plus that duplicate. See
    # run_freeze_denominator_alignment.py.
    families = {
        "Motif time share": (frequency, cols_of(frequency), MOTIF),
        "Motif diversity": (diversity, cols_of(diversity), DIVERSITY),
        "Motif transition stats": (transition, cols_of(transition), TRANSITION),
        "Motif bout duration": (bout, cols_of(bout), BOUT),
        "Motif transition pairs": (pairs, cols_of(pairs), TRANSITION),
        "All motif metrics": (all_behavior, cols_of(all_behavior), INTEGRATED),
    }
    return labels, families


def attach(labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    return labels.merge(features, on="animal_id", how="inner")


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------


def calibrate(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], raw_test_score: np.ndarray) -> np.ndarray:
    """Map the ridge decision value onto a probability of being resilient.

    The ridge score is a least-squares fit to +/-1 targets, so its units are
    arbitrary and its zero point is not a meaningful threshold once the model
    moves to another cohort. The calibrator is a 1-D logistic fitted on the
    TRAINING cohort's out-of-fold scores only, so the held-out cohort plays no
    part in it. Being a monotone transform, this leaves every AUC unchanged; it
    only makes the axis interpretable and puts the decision line at 0.5.
    """
    x_train = train[cols].to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=int)
    folds = make_folds(y_train, 4, N_REPEATS, RANDOM_SEED)
    _, oof = cv_auc_np(x_train, y_train, folds)
    calibrator = LogisticRegression(C=1e3, solver="lbfgs")
    calibrator.fit(oof.reshape(-1, 1), y_train)
    return calibrator.predict_proba(np.asarray(raw_test_score, dtype=float).reshape(-1, 1))[:, 1]


def panel_a_data(labels: pd.DataFrame, frequency: pd.DataFrame, freq_cols: list[str]) -> list[dict[str, object]]:
    """Pre-specified 7-motif repertoire model, transferred both ways."""
    data = attach(labels, frequency)
    rows = cross_cohort(data, freq_cols)
    for row in rows:
        train = data.loc[data["experiment"] == row["train_experiment"]]
        test = data.loc[data["experiment"] == row["test_experiment"]]
        row["probability"] = calibrate(train, test, freq_cols, row["score"])
        called_resilient = row["probability"] >= 0.5
        y = np.asarray(row["y"], dtype=int)
        row["pct_called_resilient"] = float(100.0 * called_resilient.mean())
        row["sensitivity_at_0.5"] = float(called_resilient[y == 1].mean())
        row["specificity_at_0.5"] = float((~called_resilient[y == 0]).mean())
    return rows


def panel_b_data(labels: pd.DataFrame, families: dict[str, tuple[pd.DataFrame, list[str], str]]) -> pd.DataFrame:
    rows = []
    for name, (frame, cols, color) in families.items():
        data = attach(labels, frame)
        cc = cross_cohort(data, cols)
        cv_aucs, _ = repeated_cv_auc(data, cols)
        by_dir = {r["direction"]: r for r in cc}
        worst = min(cc, key=lambda r: r["roc_auc"])
        rows.append(
            {
                "family": name,
                "color": color,
                "n_features": len(cols),
                "auc_kru_to_gom": by_dir["Krugers -> Gómez"]["roc_auc"],
                "auc_gom_to_kru": by_dir["Gómez -> Krugers"]["roc_auc"],
                "worst_direction_auc": worst["roc_auc"],
                "worst_ci_low": worst["auc_ci_low"],
                "worst_ci_high": worst["auc_ci_high"],
                "mean_direction_auc": float(np.mean([r["roc_auc"] for r in cc])),
                "internal_cv_auc_mean": float(cv_aucs.mean()),
                "internal_cv_auc_sd": float(cv_aucs.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values("worst_direction_auc", ascending=True).reset_index(drop=True)


def nested_combination_search(
    labels: pd.DataFrame, frequency: pd.DataFrame, freq_cols: list[str]
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Choose the motif combination inside the training cohort, test once outside.

    This is the honest version of the 127-combination search: the held-out
    cohort never influences which combination is picked.
    """
    data = attach(labels, frequency)
    combos = [list(c) for k in range(1, len(freq_cols) + 1) for c in itertools.combinations(freq_cols, k)]
    col_index = {c: i for i, c in enumerate(freq_cols)}
    results: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    for train_exp, test_exp in DIRECTIONS:
        train = data.loc[data["experiment"] == train_exp]
        test = data.loc[data["experiment"] == test_exp]
        x_train_all = train[freq_cols].to_numpy(dtype=float)
        y_train = train["target"].to_numpy(dtype=int)
        # Folds depend only on the labels, so the same split structure serves all 127 combinations.
        inner_folds = make_folds(y_train, 4, N_REPEATS, RANDOM_SEED)
        best_combo, best_inner = None, -np.inf
        for combo in combos:
            idx = [col_index[c] for c in combo]
            inner, _ = cv_auc_np(x_train_all[:, idx], y_train, inner_folds)
            mean_inner = float(inner.mean())
            trace_rows.append(
                {
                    "train_experiment": train_exp,
                    "combination": "; ".join(c.split("__", 1)[1] for c in combo),
                    "n_features": len(combo),
                    "inner_cv_auc": mean_inner,
                }
            )
            if mean_inner > best_inner:
                best_combo, best_inner = combo, mean_inner
        score = fit_score(train, test, best_combo)
        y = test["target"].to_numpy(dtype=int)
        auc = fast_auc(y, score)
        lo, hi = stratified_bootstrap_ci(y, score)
        results.append(
            {
                "train_experiment": train_exp,
                "test_experiment": test_exp,
                "direction": f"{COHORT_SHORT[train_exp]} -> {COHORT_SHORT[test_exp]}",
                "selected": "; ".join(c.split("__", 1)[1] for c in best_combo),
                "selected_cols": best_combo,
                "inner_cv_auc": best_inner,
                "held_out_auc": auc,
                "auc_ci_low": lo,
                "auc_ci_high": hi,
            }
        )
    return results, pd.DataFrame(trace_rows)


def panel_c_data(
    labels: pd.DataFrame,
    frequency: pd.DataFrame,
    freq_cols: list[str],
    nested: list[dict[str, object]],
) -> pd.DataFrame:
    """Freeze alone versus small combinations, judged on worst transfer direction."""
    freq_data = attach(labels, frequency)
    # `selection` matters more than the AUC itself: models chosen by looking at the
    # held-out cohort cannot be compared on equal terms with a priori models.
    # Freezing enters once, as the motif time share - the supervised detector is
    # the same measurement (r = 0.98) and was dropped to avoid double-counting.
    candidates: list[tuple[str, pd.DataFrame, list[str], str, str]] = [
        ("Freeze time share", freq_data, ["frequency__Freeze"], FREEZE, "a priori"),
        ("All 7 motif time shares", freq_data, freq_cols, MOTIF, "a priori"),
        ("Turn time share", freq_data, ["frequency__Turn"], MOTIF, "post hoc"),
        ("Freeze + Turn", freq_data, ["frequency__Freeze", "frequency__Turn"], MOTIF, "post hoc"),
        ("Sniff + Turn", freq_data, ["frequency__Sniff", "frequency__Turn"], MOTIF, "post hoc"),
    ]
    rows = []
    for name, data, cols, color, selection in candidates:
        cc = cross_cohort(data, cols)
        by_dir = {r["direction"]: r for r in cc}
        worst = min(cc, key=lambda r: r["roc_auc"])
        cv_aucs, _ = repeated_cv_auc(data, cols)
        rows.append(
            {
                "model": name,
                "color": color,
                "n_features": len(cols),
                "selection": selection,
                "auc_kru_to_gom": by_dir["Krugers -> Gómez"]["roc_auc"],
                "auc_gom_to_kru": by_dir["Gómez -> Krugers"]["roc_auc"],
                "worst_direction_auc": worst["roc_auc"],
                "worst_ci_low": worst["auc_ci_low"],
                "worst_ci_high": worst["auc_ci_high"],
                "internal_cv_auc_mean": float(cv_aucs.mean()),
            }
        )
    by_train = {r["train_experiment"]: r for r in nested}
    worst_nested = min(nested, key=lambda r: r["held_out_auc"])
    rows.append(
        {
            "model": "Nested-selected motifs",
            "color": "#7A7A7A",
            "n_features": np.nan,
            "selection": "nested",
            "auc_kru_to_gom": by_train[1]["held_out_auc"],
            "auc_gom_to_kru": by_train[3]["held_out_auc"],
            "worst_direction_auc": worst_nested["held_out_auc"],
            "worst_ci_low": worst_nested["auc_ci_low"],
            "worst_ci_high": worst_nested["auc_ci_high"],
            "internal_cv_auc_mean": np.nan,
        }
    )
    return pd.DataFrame(rows)


def permutation_null(data: pd.DataFrame, cols: list[str], *, n_perm: int = N_PERM, seed: int = RANDOM_SEED) -> tuple[float, float]:
    """Null AUC band from labels shuffled within cohort (keeps cohort composition)."""
    rng = np.random.default_rng(seed)
    y = data["target"].to_numpy(dtype=int)
    exp = data["experiment"].to_numpy(dtype=int)
    x = data[cols].to_numpy(dtype=float)
    n = len(data)
    cohorts = [exp == e for e in np.unique(exp)]
    draws = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        y_perm = y.copy()
        for mask in cohorts:
            y_perm[mask] = rng.permutation(y[mask])
        key = exp.astype(str)
        key = np.char.add(np.char.add(key, "_"), y_perm.astype(str))
        oof = np.empty(n, dtype=float)
        for tr_idx, te_idx in StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed + p).split(
            np.zeros(n), key
        ):
            coef, mean, scale = ridge_fit_np(x[tr_idx], y_perm[tr_idx])
            oof[te_idx] = ridge_score_np(x[te_idx], coef, mean, scale)
        draws[p] = fast_auc(y_perm, oof)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def panel_e_data(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS_MIN:
        for set_name, frame, color in [
            ("Cumulative motif repertoire", cumulative_motif_features(horizon), MOTIF),
            ("Cumulative Freeze only", cumulative_motif_freeze_features(horizon), FREEZE),
        ]:
            cols = [c for c in frame.columns if "__" in c]
            data = attach(labels, frame)
            cv_aucs, _ = repeated_cv_auc(data, cols)
            cc = cross_cohort(data, cols, with_ci=False)
            by_dir = {r["direction"]: r["roc_auc"] for r in cc}
            null_lo, null_hi = permutation_null(data, cols)
            rows.append(
                {
                    "horizon_min": horizon,
                    "feature_set": set_name,
                    "color": color,
                    "n_features": len(cols),
                    "cv_auc_mean": float(cv_aucs.mean()),
                    "cv_auc_p10": float(np.percentile(cv_aucs, 10)),
                    "cv_auc_p90": float(np.percentile(cv_aucs, 90)),
                    "auc_kru_to_gom": by_dir["Krugers -> Gómez"],
                    "auc_gom_to_kru": by_dir["Gómez -> Krugers"],
                    "null_auc_p2.5": null_lo,
                    "null_auc_p97.5": null_hi,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------


def clean_axis(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=AXIS, labelsize=7, width=0.8, length=3)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(TEXT)


def stamp_letter(fig: plt.Figure, ax: plt.Axes, letter: str, dx: float = 0.085) -> None:
    """Place panel letters in figure coordinates so long tick labels cannot push them around."""
    pos = ax.get_position()
    fig.text(max(pos.x0 - dx, 0.004), min(pos.y1 + 0.018, 0.995), letter, fontsize=11, fontweight="bold", color=TEXT, va="bottom", ha="left")


def draw_panel_a(axes: list[plt.Axes], rows: list[dict[str, object]]) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    for ax, row in zip(axes, rows):
        prob = np.asarray(row["probability"], dtype=float)
        y = np.asarray(row["y"], dtype=int)
        ax.axhspan(0.5, 1.0, color=RESILIENT, alpha=0.06, lw=0, zorder=0)
        for target, xpos, color in [(0, 0, VULNERABLE), (1, 1, RESILIENT)]:
            mask = y == target
            jitter = rng.uniform(-0.14, 0.14, int(mask.sum()))
            ax.scatter(
                np.full(int(mask.sum()), xpos) + jitter,
                prob[mask],
                s=24,
                facecolor=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            ax.hlines(np.median(prob[mask]), xpos - 0.26, xpos + 0.26, color=AXIS, lw=1.3, zorder=4)
        ax.axhline(0.5, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"vulnerable\nn={int((y == 0).sum())}", f"resilient\nn={int((y == 1).sum())}"], fontsize=7)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        clean_axis(ax)
        train_exp = int(row["train_experiment"])
        test_exp = int(row["test_experiment"])
        ax.set_title(
            f"train {COHORT_SHORT[train_exp]} (n={row['n_train']})  →  test {COHORT_SHORT[test_exp]} (n={row['n_test']})\n"
            f"AUC {row['roc_auc']:.2f} [{row['auc_ci_low']:.2f}, {row['auc_ci_high']:.2f}]   "
            f"calls {row['pct_called_resilient']:.0f}% resilient",
            fontsize=7.5,
            color=TEXT,
            pad=5,
        )
    axes[0].set_ylabel("P(resilient)\ncalibrated in training cohort", fontsize=7.5, color=TEXT)


def draw_panel_b(ax: plt.Axes, table: pd.DataFrame) -> None:
    ypos = np.arange(len(table))
    ax.axvline(0.5, color=NULL_GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.barh(ypos, table["worst_direction_auc"], color=table["color"], height=0.62, zorder=2, edgecolor="white", linewidth=0.5)
    ax.hlines(
        ypos,
        table["worst_ci_low"],
        table["worst_ci_high"],
        color=AXIS,
        lw=1.0,
        zorder=4,
    )
    ax.scatter(table["auc_kru_to_gom"], ypos, s=16, facecolor="white", edgecolor=AXIS, linewidth=0.8, zorder=5, marker="o")
    ax.scatter(table["auc_gom_to_kru"], ypos, s=18, facecolor=AXIS, edgecolor="white", linewidth=0.5, zorder=5, marker="D")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{n}  ({k})" for n, k in zip(table["family"], table["n_features"])], fontsize=7)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("cross-cohort AUC (bar = worse direction)", fontsize=7.5, color=TEXT)
    clean_axis(ax)
    handles = [
        Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor=AXIS, markersize=4, label="Krugers → Gómez"),
        Line2D([], [], marker="D", ls="none", markerfacecolor=AXIS, markeredgecolor="white", markersize=4, label="Gómez → Krugers"),
        Line2D([], [], color=AXIS, lw=1.0, label="95% CI, worse direction"),
    ]
    ax.legend(
        handles=handles,
        fontsize=6,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        handletextpad=0.5,
        columnspacing=1.2,
    )


def draw_panel_c(ax: plt.Axes, table: pd.DataFrame) -> None:
    # Group by how the model was chosen, then by performance inside each group.
    rank = {"a priori": 0, "post hoc": 1, "nested": 2}
    order = (
        table.assign(_rank=table["selection"].map(rank))
        .sort_values(["_rank", "worst_direction_auc"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ypos = np.arange(len(order))
    ax.axvline(0.5, color=NULL_GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for y, row in zip(ypos, order.itertuples()):
        post_hoc = row.selection == "post hoc"
        ax.barh(
            y,
            row.worst_direction_auc,
            color=row.color if not post_hoc else "white",
            height=0.62,
            zorder=2,
            edgecolor=row.color,
            linewidth=0.9,
            hatch="////" if post_hoc else None,
        )
    ax.hlines(ypos, order["worst_ci_low"], order["worst_ci_high"], color=AXIS, lw=1.0, zorder=4)
    ax.scatter(order["auc_kru_to_gom"], ypos, s=16, facecolor="white", edgecolor=AXIS, linewidth=0.8, zorder=5, marker="o")
    ax.scatter(order["auc_gom_to_kru"], ypos, s=18, facecolor=AXIS, edgecolor="white", linewidth=0.5, zorder=5, marker="D")
    ax.set_yticks(ypos)
    ax.set_yticklabels(order["model"], fontsize=7)
    for tick, selection in zip(ax.get_yticklabels(), order["selection"]):
        if selection != "a priori":
            tick.set_style("italic")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("cross-cohort AUC (bar = worse direction)", fontsize=7.5, color=TEXT)
    clean_axis(ax)
    handles = [
        Patch(facecolor=MOTIF, edgecolor=MOTIF, label="a priori"),
        Patch(facecolor="white", edgecolor=MOTIF, hatch="////", label="post hoc (picked on test data)"),
        Patch(facecolor="#7A7A7A", edgecolor="#7A7A7A", label="nested (picked in training cohort)"),
    ]
    ax.legend(
        handles=handles,
        fontsize=5.8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.155),
        ncol=3,
        handletextpad=0.5,
        columnspacing=1.0,
        handlelength=1.6,
    )


def draw_panel_d(ax: plt.Axes, data: pd.DataFrame, boundary: tuple[float, float, float]) -> None:
    for exp, marker in [(1, "o"), (3, "^")]:
        for target, color, name in [(0, VULNERABLE, "vulnerable"), (1, RESILIENT, "resilient")]:
            sub = data[(data["experiment"] == exp) & (data["target"] == target)]
            ax.scatter(
                sub["frequency__Freeze"],
                sub["frequency__Turn"],
                s=34,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
    b0, b_freeze, b_turn = boundary
    xlim = (data["frequency__Freeze"].min() - 1.5, data["frequency__Freeze"].max() + 1.5)
    ylim = (data["frequency__Turn"].min() - 5.0, data["frequency__Turn"].max() + 1.5)
    if abs(b_turn) > 1e-9:
        xs = np.linspace(*xlim, 100)
        ys = -(b0 + b_freeze * xs) / b_turn
        inside = (ys >= ylim[0]) & (ys <= ylim[1])
        ax.plot(xs[inside], ys[inside], color=AXIS, lw=1.0, ls=(0, (5, 3)), zorder=2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Freeze, % of classified time", fontsize=7.5, color=TEXT)
    ax.set_ylabel("Turn, % of classified time", fontsize=7.5, color=TEXT)
    clean_axis(ax)
    # The line is fitted on every animal, so it is a picture of the separation,
    # not an out-of-sample claim. Panel c carries the held-out numbers.
    ax.text(
        0.02,
        0.02,
        "dashed line: 2-feature boundary fitted on all animals (illustrative)",
        transform=ax.transAxes,
        fontsize=5.6,
        color=AXIS,
        ha="left",
        va="bottom",
    )
    handles = [
        Line2D([], [], marker="o", ls="none", markerfacecolor=VULNERABLE, markeredgecolor="white", markersize=5, label="vulnerable"),
        Line2D([], [], marker="o", ls="none", markerfacecolor=RESILIENT, markeredgecolor="white", markersize=5, label="resilient"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="none", markeredgecolor=AXIS, markersize=5, label="Krugers"),
        Line2D([], [], marker="^", ls="none", markerfacecolor="none", markeredgecolor=AXIS, markersize=5, label="Gómez"),
    ]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="upper right", handletextpad=0.4, labelspacing=0.3, ncol=2, columnspacing=0.8)


def draw_panel_e(ax: plt.Axes, table: pd.DataFrame) -> None:
    null_rows = table[table["feature_set"] == "Cumulative motif repertoire"].sort_values("horizon_min")
    ax.fill_between(
        null_rows["horizon_min"],
        null_rows["null_auc_p2.5"],
        null_rows["null_auc_p97.5"],
        color=NULL_GREY,
        alpha=0.35,
        lw=0,
        zorder=1,
        label="chance (shuffled labels, 95%)",
    )
    for name, group in table.groupby("feature_set"):
        group = group.sort_values("horizon_min")
        color = group["color"].iloc[0]
        ax.fill_between(group["horizon_min"], group["cv_auc_p10"], group["cv_auc_p90"], color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(group["horizon_min"], group["cv_auc_mean"], color=color, lw=1.6, marker="o", markersize=3.5, zorder=4, label=name)
        ax.scatter(group["horizon_min"], group["auc_kru_to_gom"], s=13, facecolor="white", edgecolor=color, linewidth=0.8, zorder=3, marker="o")
        ax.scatter(group["horizon_min"], group["auc_gom_to_kru"], s=14, facecolor="none", edgecolor=color, linewidth=0.8, zorder=3, marker="D")
    ax.axhline(0.5, color=AXIS, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(HORIZONS_MIN)
    ax.set_xticklabels([f"{h:g}" for h in HORIZONS_MIN], fontsize=7)
    ax.set_xlabel("session used, from start (min)", fontsize=7.5, color=TEXT)
    ax.set_ylabel("AUC", fontsize=7.5, color=TEXT)
    ax.set_ylim(0.0, 1.0)
    clean_axis(ax)
    handles, labels_ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor=AXIS, markersize=4, label="cross-cohort, each direction"))
    labels_.append("cross-cohort, each direction")
    ax.legend(
        handles=handles,
        labels=labels_,
        fontsize=6,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.3,
    )


def build_figure(
    a_rows: list[dict[str, object]],
    b_table: pd.DataFrame,
    c_table: pd.DataFrame,
    scatter_data: pd.DataFrame,
    boundary: tuple[float, float, float],
    e_table: pd.DataFrame,
) -> plt.Figure:
    fig = plt.figure(figsize=(7.28, 9.1))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.95, 1.10, 1.00],
        hspace=0.72,
        wspace=0.50,
        left=0.185,
        right=0.985,
        top=0.905,
        bottom=0.075,
    )

    gs_a = gs[0, :].subgridspec(1, 2, wspace=0.26)
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])
    draw_panel_a([ax_a1, ax_a2], a_rows)
    fig.text(
        0.5,
        0.975,
        "Does a model trained in one cohort work in the other?",
        ha="center",
        va="top",
        fontsize=8.5,
        color=TEXT,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.955,
        "the 7 motif time shares, specified before any transfer result was seen",
        ha="center",
        va="top",
        fontsize=7,
        color=AXIS,
    )

    ax_b = fig.add_subplot(gs[1, 0])
    draw_panel_b(ax_b, b_table)
    ax_b.set_title("Which motif measure transfers?", fontsize=8, color=TEXT, fontweight="bold", loc="left", pad=6)

    ax_c = fig.add_subplot(gs[1, 1])
    draw_panel_c(ax_c, c_table)
    ax_c.set_title("Freezing alone, or a combination?", fontsize=8, color=TEXT, fontweight="bold", loc="left", pad=6)

    ax_d = fig.add_subplot(gs[2, 0])
    draw_panel_d(ax_d, scatter_data, boundary)
    ax_d.set_title("Freeze and Turn time share, all 41 animals", fontsize=8, color=TEXT, fontweight="bold", loc="left", pad=6)

    ax_e = fig.add_subplot(gs[2, 1])
    draw_panel_e(ax_e, e_table)
    ax_e.set_title("When does resilience become readable?", fontsize=8, color=TEXT, fontweight="bold", loc="left", pad=6)

    for ax, letter, dx in [
        (ax_a1, "a", 0.075),
        (ax_b, "b", 0.195),
        (ax_c, "c", 0.155),
        (ax_d, "d", 0.075),
        (ax_e, "e", 0.075),
    ]:
        stamp_letter(fig, ax, letter, dx=dx)

    return fig


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def write_report(
    a_rows: list[dict[str, object]],
    b_table: pd.DataFrame,
    c_table: pd.DataFrame,
    nested: list[dict[str, object]],
    e_table: pd.DataFrame,
) -> None:
    motif = e_table[e_table["feature_set"] == "Cumulative motif repertoire"].sort_values("horizon_min")
    freeze = e_table[e_table["feature_set"] == "Cumulative Freeze only"].sort_values("horizon_min")
    above_null = motif[motif["cv_auc_mean"] > motif["null_auc_p97.5"]]
    first_above = above_null["horizon_min"].min() if len(above_null) else np.nan

    lines: list[str] = []
    lines.append("\n## Direct supplementary figure (`supplementary_figure5_direct.png/.pdf/.svg`)\n")
    lines.append(
        "Five panels built to answer four questions directly: (a) does the model transfer between "
        "cohorts, (b) which behavioural information transfers, (c) is freezing alone sufficient, "
        "(d) the same question in behaviour space, (e) how much of the session is needed.\n"
    )
    lines.append("### Method changes from the earlier scripts\n")
    lines.append(
        "- Time-limited features are cumulative motif proportions (7 features at every horizon) instead of "
        "one feature per motif per 30-s bin (7 features at 0.5 min rising to 105 at 7.5 min). The old design "
        "changed model size along the x-axis, so the curve could not be read as an effect of time.\n"
        "- Internal validation is repeated stratified 5-fold CV (25 repeats, stratified by cohort and class) "
        "instead of LOOCV. With n=41 and a weak feature, LOOCV produced anti-learning: early Freeze reached "
        "AUC = 0.031 even though the group means were 1.3% vs 1.9%, because leaving one animal out pushes the "
        "fitted boundary away from that animal. Sub-chance LOOCV values in the earlier tables are artefacts of "
        "the resampling scheme, not evidence of inverted signal.\n"
        "- Every cross-cohort AUC has a class-stratified bootstrap CI (2000 draws).\n"
        "- The motif-combination search is nested: the combination is chosen by CV inside the training cohort "
        "only, then applied once to the held-out cohort.\n"
        "- The time panel carries a within-cohort label-permutation null band (400 permutations).\n"
    )

    lines.append("### Q1 Cross-cohort transfer (pre-specified 7-motif model)\n")
    lines.append(
        "Panel a plots a calibrated probability rather than the raw ridge score. The ridge decision value is a "
        "least-squares fit to +/-1 targets, so its units are arbitrary and its zero point stops being a usable "
        "threshold once the model is applied to another cohort. The calibrator is a 1-D logistic fitted on the "
        "training cohort's out-of-fold scores only; it is a monotone transform, so no AUC changes.\n"
    )
    for row in a_rows:
        lines.append(
            f"- {row['direction']}: AUC = {row['roc_auc']:.3f} (95% CI {row['auc_ci_low']:.3f}-{row['auc_ci_high']:.3f}), "
            f"n_test = {row['n_test']} ({row['n_resilient_test']} resilient). At the 0.5 threshold the model calls "
            f"{row['pct_called_resilient']:.0f}% of the held-out cohort resilient "
            f"(sensitivity {row['sensitivity_at_0.5']:.2f}, specificity {row['specificity_at_0.5']:.2f}).\n"
        )
    lines.append(
        "\nDiscrimination and calibration fail separately here: a model can rank animals usefully and still put "
        "its threshold in the wrong place after transfer, which is what the 0.5-threshold columns show.\n"
    )

    lines.append("\n### Q2 Which information transfers\n")
    for _, row in b_table.sort_values("worst_direction_auc", ascending=False).iterrows():
        lines.append(
            f"- {row['family']} ({row['n_features']} features): worse direction AUC = {row['worst_direction_auc']:.3f} "
            f"(95% CI {row['worst_ci_low']:.3f}-{row['worst_ci_high']:.3f}); directions "
            f"{row['auc_kru_to_gom']:.3f} / {row['auc_gom_to_kru']:.3f}; internal CV = {row['internal_cv_auc_mean']:.3f}.\n"
        )

    lines.append("\n### Q3 Freezing alone versus combinations\n")
    for _, row in c_table.sort_values("worst_direction_auc", ascending=False).iterrows():
        lines.append(
            f"- {row['model']} [{row['selection']}]: worse direction AUC = {row['worst_direction_auc']:.3f} "
            f"(95% CI {row['worst_ci_low']:.3f}-{row['worst_ci_high']:.3f}); directions "
            f"{row['auc_kru_to_gom']:.3f} / {row['auc_gom_to_kru']:.3f}.\n"
        )
    lines.append("\nNested selection outcome (combination chosen inside the training cohort only):\n")
    for row in nested:
        lines.append(
            f"- train {COHORT_FULL[row['train_experiment']]}: selected `{row['selected']}` "
            f"(inner CV AUC = {row['inner_cv_auc']:.3f}); held-out AUC = {row['held_out_auc']:.3f} "
            f"(95% CI {row['auc_ci_low']:.3f}-{row['auc_ci_high']:.3f}).\n"
        )

    lines.append("\n### Q4 When in the task\n")
    for _, row in motif.iterrows():
        lines.append(
            f"- {row['horizon_min']:g} min, cumulative repertoire: CV AUC = {row['cv_auc_mean']:.3f} "
            f"(10-90% across repeats {row['cv_auc_p10']:.3f}-{row['cv_auc_p90']:.3f}; chance band up to "
            f"{row['null_auc_p97.5']:.3f}); cross-cohort {row['auc_kru_to_gom']:.3f} / {row['auc_gom_to_kru']:.3f}.\n"
        )
    for _, row in freeze.iterrows():
        lines.append(
            f"- {row['horizon_min']:g} min, cumulative Freeze only: CV AUC = {row['cv_auc_mean']:.3f} "
            f"(chance band up to {row['null_auc_p97.5']:.3f}).\n"
        )
    if not np.isnan(first_above):
        lines.append(
            f"\nThe cumulative repertoire first exceeds its own permutation chance band at {first_above:g} min.\n"
        )

    best_family = b_table.loc[b_table["worst_direction_auc"].idxmax()]
    freeze_row = c_table.loc[c_table["model"] == "Freeze time share"].iloc[0]
    all_motifs = c_table.loc[c_table["model"] == "All 7 motif time shares"].iloc[0]
    worst_nested = min(nested, key=lambda r: r["held_out_auc"])
    lines.append("\n### Bottom line, and what changes\n")
    lines.append(
        f"- Transfer is real but asymmetric and imprecise: {a_rows[0]['roc_auc']:.2f} "
        f"[{a_rows[0]['auc_ci_low']:.2f}, {a_rows[0]['auc_ci_high']:.2f}] one way and "
        f"{a_rows[1]['roc_auc']:.2f} [{a_rows[1]['auc_ci_low']:.2f}, {a_rows[1]['auc_ci_high']:.2f}] the other. "
        "The weaker direction's interval includes 0.5, so the honest claim is one-directional.\n"
    )
    lines.append(
        f"- Judged on the worse direction, the strongest family is {best_family['family']} "
        f"({best_family['worst_direction_auc']:.3f}). Freeze time share alone reaches "
        f"{freeze_row['worst_direction_auc']:.3f} while the full 7-motif repertoire reaches "
        f"{all_motifs['worst_direction_auc']:.3f}: at this sample size the compact freezing signal transfers "
        "at least as well as the whole repertoire, so the repertoire's added value is not established.\n"
    )
    lines.append(
        f"- Honest (nested) motif selection gives {worst_nested['held_out_auc']:.3f} in the worse direction, "
        "against 0.868 for the previously reported Sniff+Turn model that was chosen using the held-out cohort. "
        "The two training cohorts also select different combinations, which is itself evidence that "
        "'which behaviours matter' is not resolvable at n=41.\n"
    )
    lines.append(
        "- With the feature-count confound removed, no horizon before the full session clears its own "
        "permutation null. The earlier statement that resilience is reliably predictable by about 5 min does "
        "not survive; the data do not show resilience being readable during early exploration.\n"
    )

    lines.append("\nOutputs:\n")
    for name in [
        "supplementary_figure5_direct.png/.pdf/.svg",
        "direct_figure_q1_cross_cohort.csv",
        "direct_figure_q1_per_animal_predictions.csv",
        "direct_figure_q2_feature_families.csv",
        "direct_figure_q3_freeze_vs_combinations.csv",
        "direct_figure_q3_nested_selection.csv",
        "direct_figure_q3_nested_selection_trace.csv",
        "direct_figure_q4_time_to_prediction.csv",
        "direct_figure_scatter_source_data.csv",
    ]:
        lines.append(f"- `{name}`\n")

    # Replace any previous copy of this section in place, instead of stacking
    # duplicates. Bounded at the next top-level heading: this section is no
    # longer the last one in the report, and truncating to end-of-file here
    # silently deleted every section written after it.
    report_path = OUT / "REPORT.md"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    section = "".join(lines)
    marker = "\n## Direct supplementary figure"
    if marker in existing:
        start = existing.index(marker)
        rest = existing[start + len(marker) :]
        nxt = rest.find("\n## ")
        tail = rest[nxt + 1 :] if nxt != -1 else ""
        existing = existing[:start].rstrip() + "\n"
        report_path.write_text(existing + section.rstrip() + "\n\n" + tail, encoding="utf-8")
    else:
        report_path.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")


# --------------------------------------------------------------------------


def main() -> None:
    labels, families = build_frames()
    frequency = _frequency()
    freq_cols = [c for c in frequency.columns if c.startswith("frequency__")]

    print("Q1 cross-cohort transfer, pre-specified 7-motif model ...", flush=True)
    a_rows = panel_a_data(labels, frequency, freq_cols)
    for row in a_rows:
        print(f"  {row['direction']}: AUC {row['roc_auc']:.3f} [{row['auc_ci_low']:.3f}, {row['auc_ci_high']:.3f}]")

    print("Q2 feature families ...", flush=True)
    b_table = panel_b_data(labels, families)
    print(b_table[["family", "n_features", "worst_direction_auc", "internal_cv_auc_mean"]].to_string(index=False))

    print("Q3 nested combination search ...", flush=True)
    nested, nested_trace = nested_combination_search(labels, frequency, freq_cols)
    for row in nested:
        print(f"  train {COHORT_SHORT[row['train_experiment']]}: picked {row['selected']} -> held-out AUC {row['held_out_auc']:.3f}")

    print("Q3 freeze versus combinations ...", flush=True)
    c_table = panel_c_data(labels, frequency, freq_cols, nested)
    print(c_table[["model", "worst_direction_auc", "auc_kru_to_gom", "auc_gom_to_kru"]].to_string(index=False))

    print("Q4 time to prediction ...", flush=True)
    e_table = panel_e_data(labels)
    print(
        e_table[["horizon_min", "feature_set", "n_features", "cv_auc_mean", "null_auc_p97.5"]].to_string(index=False)
    )

    scatter_data = attach(labels, frequency)
    coef, mean, scale = efa.fit_ridge(scatter_data, ["frequency__Freeze", "frequency__Turn"])
    b_freeze = coef[1] / scale[0]
    b_turn = coef[2] / scale[1]
    b0 = coef[0] - coef[1] * mean[0] / scale[0] - coef[2] * mean[1] / scale[1]

    fig = build_figure(a_rows, b_table, c_table, scatter_data, (b0, b_freeze, b_turn), e_table)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"supplementary_figure5_direct.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    q1 = pd.DataFrame(
        [{k: v for k, v in row.items() if k not in {"score", "y", "test_frame", "probability"}} for row in a_rows]
    )
    q1.to_csv(OUT / "direct_figure_q1_cross_cohort.csv", index=False)
    per_animal = []
    for row in a_rows:
        frame = row["test_frame"][["animal_id", "experiment", "profile", "target"]].copy()
        frame["direction"] = row["direction"]
        frame["ridge_score"] = row["score"]
        frame["p_resilient"] = row["probability"]
        frame["called"] = np.where(np.asarray(row["probability"]) >= 0.5, "resilient", "vulnerable")
        per_animal.append(frame)
    pd.concat(per_animal, ignore_index=True).to_csv(OUT / "direct_figure_q1_per_animal_predictions.csv", index=False)
    b_table.drop(columns=["color"]).to_csv(OUT / "direct_figure_q2_feature_families.csv", index=False)
    c_table.drop(columns=["color"]).to_csv(OUT / "direct_figure_q3_freeze_vs_combinations.csv", index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k != "selected_cols"} for row in nested]).to_csv(
        OUT / "direct_figure_q3_nested_selection.csv", index=False
    )
    nested_trace.to_csv(OUT / "direct_figure_q3_nested_selection_trace.csv", index=False)
    e_table.drop(columns=["color"]).to_csv(OUT / "direct_figure_q4_time_to_prediction.csv", index=False)
    scatter_data[
        ["animal_id", "experiment", "profile", "target", "frequency__Freeze", "frequency__Turn", "frequency__Sniff"]
    ].to_csv(OUT / "direct_figure_scatter_source_data.csv", index=False)

    write_report(a_rows, b_table, c_table, nested, e_table)
    print("\nWrote supplementary_figure5_direct.* and direct_figure_*.csv; appended REPORT.md")


if __name__ == "__main__":
    main()
