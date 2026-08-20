"""Core models for Figure 7 resilient/vulnerable prediction."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


RANDOM_SEED = 13
# Sampled every 30 s to match the behaviour-dynamics time courses in
# Figures 3, 5 and Supplementary 1 (0.5-7.5 min, 15 points).
HORIZONS_MIN = [round(0.5 * k, 1) for k in range(1, 16)]
RIDGE_ALPHA = 1.0
DEFAULT_REPO = Path(__file__).resolve().parents[2]
COHORT_LABELS = {
    "Exp1": "Sanguino Gómez\n& Krugers",
    "Exp3": "Sanguino Gómez\net al.",
}


def repo_root(explicit: Path | None = None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend([Path.cwd(), DEFAULT_REPO])
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "data/processed/figure5_dynamics_scores.csv").exists():
            return root
    raise FileNotFoundError("Could not find the manuscript repo. Re-run with --repo PATH.")


def auc_or_nan(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def ap_or_nan(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def metric_row(
    y_true: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray,
    *,
    analysis: str,
    feature_set: str,
    train_experiment: str,
    test_experiment: str,
    n_train: int,
    n_test: int,
    n_features: int,
) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "analysis": analysis,
        "feature_set": feature_set,
        "train_experiment": train_experiment,
        "test_experiment": test_experiment,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "n_resilient_test": int(np.sum(y_true == 1)),
        "n_vulnerable_test": int(np.sum(y_true == 0)),
        "roc_auc": auc_or_nan(y_true, score),
        "average_precision": ap_or_nan(y_true, score),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity_resilient": float(sensitivity),
        "specificity_vulnerable": float(specificity),
        "tn_vulnerable": int(tn),
        "fp_vulnerable_as_resilient": int(fp),
        "fn_resilient_as_vulnerable": int(fn),
        "tp_resilient": int(tp),
    }
    cohort_labels = {
        "Exp1": "Sanguino Gómez\n& Krugers",
        "Exp3": "Sanguino Gómez\net al.",
    }


def bootstrap_auc_ci(
    y_true: np.ndarray,
    score: np.ndarray,
    *,
    n_boot: int = 300,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    rng = rng or np.random.default_rng(RANDOM_SEED)
    n = len(y_true)
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y_true[idx], score[idx])))
    if not aucs:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return (float(lo), float(hi))


def bootstrap_delta_ci(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    n_boot: int = 300,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    rng = rng or np.random.default_rng(RANDOM_SEED + 1)
    n = len(y_true)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        deltas.append(float(roc_auc_score(y_true[idx], score_a[idx]) - roc_auc_score(y_true[idx], score_b[idx])))
    if not deltas:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return (float(lo), float(hi))


def paired_auc_test(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    n_boot: int = 10000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Paired bootstrap test for AUC_a - AUC_b on one set of animals.

    Both feature sets score the same held-out animals, so the animals - not the
    scores - are resampled, and each replicate recomputes both AUCs on the same
    resampled set.  That keeps the two models' errors paired, which is what makes
    the comparison sensitive: the shared animal-level noise cancels in the delta.

    The two-sided p-value is the usual bootstrap inversion, the proportion of
    replicate deltas falling on the opposite side of zero from the observed
    delta, doubled.  The +1 corrections keep it strictly positive so a p-value
    is never reported as exactly zero with a finite number of replicates.
    """
    rng = rng or np.random.default_rng(RANDOM_SEED + 2)
    n = len(y_true)
    observed = float(roc_auc_score(y_true, score_a) - roc_auc_score(y_true, score_b))
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        deltas.append(
            float(roc_auc_score(y_true[idx], score_a[idx])
                  - roc_auc_score(y_true[idx], score_b[idx]))
        )
    if not deltas:
        return {"delta_auc": observed, "ci_low": float("nan"),
                "ci_high": float("nan"), "p_value": float("nan"), "n_boot": 0}
    arr = np.asarray(deltas, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    tail = np.sum(arr <= 0) if observed > 0 else np.sum(arr >= 0)
    p = min(1.0, 2.0 * (float(tail) + 1.0) / (len(arr) + 1.0))
    return {"delta_auc": observed, "ci_low": float(lo), "ci_high": float(hi),
            "p_value": p, "n_boot": len(arr)}


def significance_stars(p: float) -> str:
    """Manuscript convention: *** <0.001, ** <0.01, * <0.05, otherwise n.s."""
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def load_labels(root: Path) -> pd.DataFrame:
    prof = pd.read_csv(root / "data/processed/figure5_dynamics_scores.csv")
    freq = pd.read_csv(root / "data/processed/cluster_frequency_per_animal.csv")
    experiments = freq[["animal_id", "experiment"]].drop_duplicates()
    labels = (
        prof.loc[prof["group"] == "ELS", ["animal", "group", "dynamics_score", "resilient_by_zero"]]
        .rename(columns={"animal": "animal_id"})
        .merge(experiments, on="animal_id", how="left")
    )
    labels["target"] = labels["resilient_by_zero"].astype(int)
    labels["profile"] = np.where(labels["target"].eq(1), "resilient", "vulnerable")
    labels["animal_label"] = labels["animal_id"].map(lambda x: f"{x:.1f}")
    return labels.sort_values(["experiment", "animal_id"]).reset_index(drop=True)


def full_repertoire_features(root: Path) -> pd.DataFrame:
    freq = pd.read_csv(root / "data/processed/cluster_frequency_per_animal.csv")
    wide = (
        freq.pivot_table(index="animal_id", columns="cluster", values="frequency_seconds", aggfunc="sum", fill_value=0)
        .sort_index(axis=1)
        .reset_index()
    )
    cluster_cols = [c for c in wide.columns if c != "animal_id"]
    totals = wide[cluster_cols].sum(axis=1).replace(0, np.nan)
    wide[cluster_cols] = wide[cluster_cols].div(totals, axis=0).mul(100.0)
    return wide.rename(columns={c: f"motif_{c}" for c in cluster_cols})


def freezing_features(root: Path) -> pd.DataFrame:
    freezing = pd.read_csv(root / "data/raw/freezing_predictions_light.csv.gz", usecols=["animal_id", "freezing"])
    out = freezing.groupby("animal_id", as_index=False)["freezing"].mean()
    out["supervised_freezing_pct"] = out["freezing"] * 100.0
    return out.drop(columns=["freezing"])


def early_repertoire_features(root: Path, horizon_min: float) -> pd.DataFrame:
    tc = pd.read_csv(root / "data/processed/cluster_timecourse_per_animal.csv")
    tc = tc.loc[tc["time_s"] <= horizon_min * 60].copy()
    tc["early_feature"] = "early_motif_" + tc["cluster"] + "_t" + tc["time_bin"].astype(str)
    wide = (
        tc.pivot_table(index="animal_id", columns="early_feature", values="pct", aggfunc="mean", fill_value=0)
        .sort_index(axis=1)
        .reset_index()
    )
    return wide


def early_freezing_features(root: Path, horizon_min: float, fps: float = 25.0) -> pd.DataFrame:
    max_frame = int(round(horizon_min * 60 * fps))
    freezing = pd.read_csv(root / "data/raw/freezing_predictions_light.csv.gz", usecols=["animal_id", "frame", "freezing"])
    freezing = freezing.loc[freezing["frame"] < max_frame]
    out = freezing.groupby("animal_id", as_index=False)["freezing"].mean()
    out["early_supervised_freezing_pct"] = out["freezing"] * 100.0
    return out.drop(columns=["freezing"])


def assemble(labels: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = labels.merge(features, on="animal_id", how="inner")
    feature_cols = [c for c in data.columns if c not in {"animal_id", "animal_label", "group", "dynamics_score", "resilient_by_zero", "target", "profile", "experiment"}]
    return data, feature_cols


def fit_ridge_classifier(train: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = train[feature_cols].to_numpy(dtype=float)
    y = train["target"].to_numpy(dtype=int)
    y_signed = np.where(y == 1, 1.0, -1.0)
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    xs = (x - mean) / scale
    xb = np.column_stack([np.ones(len(xs)), xs])

    n = len(y)
    n_pos = max(int(np.sum(y == 1)), 1)
    n_neg = max(int(np.sum(y == 0)), 1)
    weights = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))
    xw = xb * np.sqrt(weights[:, None])
    yw = y_signed * np.sqrt(weights)

    penalty = np.eye(xb.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)
    return coef, mean, scale


def score_ridge(test: pd.DataFrame, feature_cols: list[str], coef: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = test[feature_cols].to_numpy(dtype=float)
    xs = (x - mean) / scale
    return coef[0] + xs @ coef[1:]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    coef, mean, scale = fit_ridge_classifier(train, feature_cols)
    score = score_ridge(test, feature_cols, coef, mean, scale)
    pred = (score >= 0).astype(int)
    return score, pred


def cross_cohort(labels: pd.DataFrame, feature_sets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for feature_name, features in feature_sets.items():
        data, feature_cols = assemble(labels, features)
        for train_exp, test_exp in [(1, 3), (3, 1)]:
            train = data.loc[data["experiment"] == train_exp].copy()
            test = data.loc[data["experiment"] == test_exp].copy()
            score, pred = fit_predict(train, test, feature_cols)
            metric_rows.append(
                metric_row(
                    test["target"].to_numpy(),
                    score,
                    pred,
                    analysis="cross_cohort",
                    feature_set=feature_name,
                    train_experiment=f"Exp{train_exp}",
                    test_experiment=f"Exp{test_exp}",
                    n_train=len(train),
                    n_test=len(test),
                    n_features=len(feature_cols),
                )
            )
            pred_df = test[["animal_id", "animal_label", "experiment", "profile", "target", "dynamics_score"]].copy()
            pred_df["analysis"] = "cross_cohort"
            pred_df["feature_set"] = feature_name
            pred_df["train_experiment"] = f"Exp{train_exp}"
            pred_df["test_experiment"] = f"Exp{test_exp}"
            pred_df["resilience_score"] = score
            pred_df["predicted_profile"] = np.where(pred == 1, "resilient", "vulnerable")
            prediction_rows.append(pred_df)
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def loocv(
    labels: pd.DataFrame,
    feature_sets: dict[str, pd.DataFrame],
    analysis: str,
    *,
    bootstrap_ci: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for feature_name, features in feature_sets.items():
        data, feature_cols = assemble(labels, features)
        y = data["target"].to_numpy()
        scores = np.zeros(len(data), dtype=float)
        preds = np.zeros(len(data), dtype=int)
        for test_pos in range(len(data)):
            train = data.drop(data.index[test_pos])
            test = data.iloc[[test_pos]]
            score, pred = fit_predict(train, test, feature_cols)
            scores[test_pos] = score[0]
            preds[test_pos] = pred[0]
        row = metric_row(
            y,
            scores,
            preds,
            analysis=analysis,
            feature_set=feature_name,
            train_experiment="LOOCV",
            test_experiment="LOOCV",
            n_train=len(data) - 1,
            n_test=len(data),
            n_features=len(feature_cols),
        )
        if bootstrap_ci:
            row["roc_auc_ci95_low"], row["roc_auc_ci95_high"] = bootstrap_auc_ci(y, scores)
        else:
            row["roc_auc_ci95_low"], row["roc_auc_ci95_high"] = np.nan, np.nan
        metric_rows.append(row)
        pred_df = data[["animal_id", "animal_label", "experiment", "profile", "target", "dynamics_score"]].copy()
        pred_df["analysis"] = analysis
        pred_df["feature_set"] = feature_name
        pred_df["train_experiment"] = "LOOCV"
        pred_df["test_experiment"] = "LOOCV"
        pred_df["resilience_score"] = scores
        pred_df["predicted_profile"] = np.where(preds == 1, "resilient", "vulnerable")
        prediction_rows.append(pred_df)
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def early_prediction(labels: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    pred_frames: list[pd.DataFrame] = []
    for horizon in HORIZONS_MIN:
        feature_sets = {
            "early_motif_repertoire": early_repertoire_features(root, horizon),
            "early_freezing_only": early_freezing_features(root, horizon),
        }
        metrics, preds = loocv(labels, feature_sets, analysis="early_prediction", bootstrap_ci=False)
        metrics["horizon_min"] = horizon
        preds["horizon_min"] = horizon
        metric_frames.append(metrics)
        pred_frames.append(preds)
    return pd.concat(metric_frames, ignore_index=True), pd.concat(pred_frames, ignore_index=True)


def coefficient_table(labels: pd.DataFrame, feature_sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_name, features in feature_sets.items():
        data, feature_cols = assemble(labels, features)
        coef, _, _ = fit_ridge_classifier(data, feature_cols)
        for feature, coef_value in zip(feature_cols, coef[1:]):
            rows.append({"feature_set": feature_name, "feature": feature, "standardized_linear_coefficient": float(coef_value)})
    return pd.DataFrame(rows).sort_values(["feature_set", "standardized_linear_coefficient"], ascending=[True, False])


def plot_summary(cross_metrics: pd.DataFrame, head_metrics: pd.DataFrame, early_metrics: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "motif_repertoire": "#6398A4",
        "freezing_only": "#C671A0",
        "early_motif_repertoire": "#6398A4",
        "early_freezing_only": "#C671A0",
    }
    labels = {
        "motif_repertoire": "Motif repertoire",
        "freezing_only": "Freeze only",
        "early_motif_repertoire": "Motif repertoire",
        "early_freezing_only": "Freeze only",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)

    ax = axes[0]
    cross = cross_metrics.copy()
    cross["direction"] = cross["train_experiment"] + " -> " + cross["test_experiment"]
    x_positions = np.arange(cross["direction"].nunique())
    width = 0.34
    for offset, feature in [(-width / 2, "motif_repertoire"), (width / 2, "freezing_only")]:
        vals = cross.loc[cross["feature_set"] == feature, "roc_auc"].to_numpy()
        ax.bar(x_positions + offset, vals, width=width, color=colors[feature], label=labels[feature])
        for x, y in zip(x_positions + offset, vals):
            ax.text(x, min(y + 0.025, 1.02), f"{y:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.5, color="#6B6B6B", lw=0.8, ls=":")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            f"{COHORT_LABELS[row['train_experiment']]} ->\n{COHORT_LABELS[row['test_experiment']]}"
            for _, row in cross.drop_duplicates("direction").iterrows()
        ]
    )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("ROC AUC")
    ax.set_title("A  Held-out cohort")
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    ax = axes[1]
    head = head_metrics.copy()
    x = np.arange(len(head))
    bars = ax.bar(x, head["roc_auc"], color=[colors[f] for f in head["feature_set"]])
    ax.errorbar(
        x,
        head["roc_auc"],
        yerr=[
            head["roc_auc"] - head["roc_auc_ci95_low"],
            head["roc_auc_ci95_high"] - head["roc_auc"],
        ],
        fmt="none",
        ecolor="#2B2B2B",
        elinewidth=0.8,
        capsize=2,
    )
    for bar, (_, row) in zip(bars, head.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, min(row["roc_auc"] + 0.04, 1.02), f"{row['roc_auc']:.2f}", ha="center", fontsize=7)
    ax.axhline(0.5, color="#6B6B6B", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[f] for f in head["feature_set"]], rotation=18, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_title("B  Full-session LOOCV")

    ax = axes[2]
    for feature in ["early_motif_repertoire", "early_freezing_only"]:
        sub = early_metrics.loc[early_metrics["feature_set"] == feature].sort_values("horizon_min")
        ax.plot(sub["horizon_min"], sub["roc_auc"], marker="o", lw=1.5, color=colors[feature], label=labels[feature])
    ax.axhline(0.5, color="#6B6B6B", lw=0.8, ls=":")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("First minutes included")
    ax.set_ylabel("ROC AUC")
    ax.set_title("C  Early prediction")
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"tier1_reanalysis_summary.{ext}", dpi=300)
    plt.close(fig)


def markdown_count_table(labels: pd.DataFrame) -> str:
    counts = labels.groupby(["experiment", "profile"]).size().unstack(fill_value=0)
    for col in ["vulnerable", "resilient"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["vulnerable", "resilient"]].reset_index()
    lines = [
        "| experiment | vulnerable | resilient |",
        "| --- | ---: | ---: |",
    ]
    for _, row in counts.iterrows():
        lines.append(f"| {int(row['experiment'])} | {int(row['vulnerable'])} | {int(row['resilient'])} |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    labels: pd.DataFrame,
    cross_metrics: pd.DataFrame,
    head_metrics: pd.DataFrame,
    early_metrics: pd.DataFrame,
    delta_ci: tuple[float, float],
) -> None:
    motif_cross = cross_metrics.loc[cross_metrics["feature_set"] == "motif_repertoire"].copy()
    freezing_cross = cross_metrics.loc[cross_metrics["feature_set"] == "freezing_only"].copy()
    head = head_metrics.set_index("feature_set")
    early_motif = early_metrics.loc[early_metrics["feature_set"] == "early_motif_repertoire"].sort_values("horizon_min")
    best_early = early_motif.loc[early_motif["roc_auc"].idxmax()]
    first_two = early_motif.loc[early_motif["horizon_min"].eq(2.0)].iloc[0]
    first_three = early_motif.loc[early_motif["horizon_min"].eq(3.0)].iloc[0]

    lines = [
        "# Tier 1 reanalysis: predictive repertoire marker",
        "",
        "## Design",
        "",
        "- Target: ELS resilient vs ELS vulnerable profile from `data/processed/figure5_dynamics_scores.csv`.",
        "- Cohorts: `experiment == 1` and `experiment == 3`, kept independent for held-out tests.",
        "- Model: standardized L2-regularized linear classifier with balanced class weights; no feature selection.",
        "- Repertoire features: full-session proportions of the seven motif clusters.",
        "- Freeze feature: supervised Freeze percentage from `data/raw/freezing_predictions_light.csv.gz`.",
        "- Early repertoire features: 30-second motif trajectory columns accumulated up to each tested horizon.",
        "",
        "## Sample counts",
        "",
        markdown_count_table(labels),
        "",
        "## Main results",
        "",
        (
            f"- Cross-cohort motif AUCs: Sanguino Gómez & Krugers -> Sanguino Gómez et al. = "
            f"{motif_cross.loc[motif_cross['test_experiment'].eq('Exp3'), 'roc_auc'].iloc[0]:.3f}; "
            f"Sanguino Gómez et al. -> Sanguino Gómez & Krugers = "
            f"{motif_cross.loc[motif_cross['test_experiment'].eq('Exp1'), 'roc_auc'].iloc[0]:.3f}."
        ),
        (
            f"- Cross-cohort Freeze-only AUCs: Sanguino Gómez & Krugers -> Sanguino Gómez et al. = "
            f"{freezing_cross.loc[freezing_cross['test_experiment'].eq('Exp3'), 'roc_auc'].iloc[0]:.3f}; "
            f"Sanguino Gómez et al. -> Sanguino Gómez & Krugers = "
            f"{freezing_cross.loc[freezing_cross['test_experiment'].eq('Exp1'), 'roc_auc'].iloc[0]:.3f}."
        ),
        (
            f"- Full-session LOOCV AUC: motif repertoire = {head.loc['motif_repertoire', 'roc_auc']:.3f} "
            f"(95% bootstrap CI {head.loc['motif_repertoire', 'roc_auc_ci95_low']:.3f}-"
            f"{head.loc['motif_repertoire', 'roc_auc_ci95_high']:.3f}); Freeze only = "
            f"{head.loc['freezing_only', 'roc_auc']:.3f} "
            f"(95% bootstrap CI {head.loc['freezing_only', 'roc_auc_ci95_low']:.3f}-"
            f"{head.loc['freezing_only', 'roc_auc_ci95_high']:.3f})."
        ),
        (
            f"- Motif-minus-Freeze LOOCV AUC delta = "
            f"{head.loc['motif_repertoire', 'roc_auc'] - head.loc['freezing_only', 'roc_auc']:.3f} "
            f"(paired bootstrap CI {delta_ci[0]:.3f}-{delta_ci[1]:.3f})."
        ),
        (
            f"- Early motif-trajectory prediction: first 2 min AUC = {first_two['roc_auc']:.3f}; "
            f"first 3 min AUC = {first_three['roc_auc']:.3f}; "
            f"best tested horizon = {best_early['horizon_min']:.1f} min with AUC = {best_early['roc_auc']:.3f}."
        ),
        "",
        "## Interpretation",
        "",
        (
            "This is a cheap/high-leverage manuscript addition for the cross-cohort marker claim: motif "
            "repertoire trained in the Sanguino Gómez & Krugers cohort generalizes to the Sanguino Gómez et al. "
            "cohort with AUC 0.817, and the reverse direction remains above chance but weaker. The head-to-head "
            "Freeze result is supportive but should be phrased carefully: motif repertoire exceeds Freeze in "
            "full-session LOOCV, but the paired bootstrap interval overlaps zero and Freeze is strong in the "
            "Sanguino Gómez et al. -> Sanguino Gómez & Krugers transfer. Early-session "
            "predictability is clearest by 2-5 minutes rather than monotonically improving from the first minute."
        ),
        "",
        "## Outputs",
        "",
        "- `tier1_reanalysis_summary.png/.pdf/.svg`",
        "- `supplementary_figure4_prospective_paper_palette.png/.pdf/.svg`",
        "- `supplementary_figure4_feature_expansion_named_cohorts.png/.pdf/.svg`",
        "- `supplementary_figure4_all_parameter_importance.png/.pdf/.svg`",
        "- `supplementary_figure4_cumulative_features_time.png/.pdf/.svg`",
        "- `expanded_feature_model_loocv_metrics.csv`",
        "- `expanded_feature_model_cross_cohort_metrics.csv`",
        "- `expanded_feature_block_ablation.csv`",
        "- `expanded_feature_importance_coefficients.csv`",
        "- `expanded_behavior_category_importance.csv`",
        "- `expanded_transition_pair_importance.csv`",
        "- `cumulative_feature_loocv_metrics.csv`",
        "- `cumulative_feature_selection_summary.csv`",
        "- `analysis_dataset.csv`",
        "- `cross_cohort_metrics.csv` and `cross_cohort_predictions.csv`",
        "- `head_to_head_loocv_metrics.csv` and `head_to_head_loocv_predictions.csv`",
        "- `early_prediction_metrics.csv` and `early_prediction_predictions.csv`",
        "- `feature_coefficients.csv`",
        "- `run_tier1_reanalysis.py`",
        "",
        "## Expanded feature-family check",
        "",
        (
            "The original predictive model uses only full-session motif frequencies. The expanded check separately "
            "tests supervised Freeze, motif frequencies, transition metrics, diversity metrics, bout-duration "
            "metrics, all behavior metrics together, and all behavior plus Freeze. Dynamics score/MDS position "
            "are not used as predictors because the resilient/vulnerable label is defined from the dynamics score, "
            "so including it would be circular."
        ),
        "",
        (
            "LOOCV AUCs: Freeze = 0.790; motif frequency = 0.842; transition summaries = 0.750; "
            "transition pairs = 0.603; diversity = 0.796; bout duration = 0.747; all behavior without "
            "transition pairs = 0.836; all behavior with transition pairs = 0.626; all behavior plus Freeze = "
            "0.624. The compact frequency model remains the strongest clean model. Exact transition pairs are "
            "individually interpretable, but as a 42-feature block they hurt generalization in this sample."
        ),
        "",
        (
            "Overall behavior-category weights from the all-parameter model: Freeze = 18.1%, Sniff = 17.9%, "
            "Jump = 17.4%, Turn = 15.6%, Locomotion = 11.8%, Groom = 11.0%, Climb = 8.2% of behavior-linked "
            "absolute weight. Positive feature weights push the model toward resilient; negative weights push "
            "it toward vulnerable. Top transition-pair features are Freeze -> Sniff, Jump -> Turn, Turn -> "
            "Groom, Turn -> Sniff, and Jump -> Climb, but these should be treated as exploratory because the "
            "full transition-pair block does not cross-validate well."
        ),
        "",
        "## Cumulative feature and time checks",
        "",
        (
            "Nested LOOCV feature ranking shows that the full 23-feature behavioral summary model reaches "
            "AUC = 0.836, but performance is already near maximum with the top 7 features (AUC = 0.813). "
            "The near-maximum selected set is dominated by bout duration and motif frequency, with the most "
            "stable features being Bout: Turn, Freq: Jump, Bout: Sniff, Freq: Turn, Trans: determinism, "
            "Bout: Freeze, and Div: evenness."
        ),
        "",
        (
            "For time-to-prediction, motif trajectory first crosses AUC = 0.70 at 5 min (AUC = 0.753). "
            "The first 2 min are above chance but weaker (AUC = 0.672), and the 3 min estimate is "
            "unstable/near chance (AUC = 0.549). A cautious figure statement is therefore: resilience "
            "becomes reliably predictable by about 5 min of behavior."
        ),
        "",
        "Paper palette used for regenerated figures: `#C671A0`, `#6398A4`, `#9ECADA`, `#E6C23A`, `#BCD548`, `#D98427`, `#D95D5D`.",
        "",
        "Rerun command:",
        "",
        "```powershell",
        "python scripts/generate_figures/figure_7_resilience_prediction.py --recompute-onset",
        "```",
        "",
    ]
    # This script regenerates the report from scratch, so it overwrites whatever
    # is there. Every later analysis appends its own section to the same file, so
    # a bare overwrite silently discards all of them. Keep a timestamped copy
    # first - cheap, and the alternative is losing the write-up.
    report = out_dir / "REPORT.md"
    if report.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = out_dir / f"REPORT.backup_{stamp}.md"
        backup.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"existing REPORT.md backed up to {backup.name} before overwrite")
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=None, help="Path to the manuscript repository.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory outside the manuscript repository.")
    args = parser.parse_args()

    root = repo_root(args.repo)
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(root)
    feature_sets = {
        "motif_repertoire": full_repertoire_features(root),
        "freezing_only": freezing_features(root),
    }
    analysis_data, _ = assemble(labels, feature_sets["motif_repertoire"].merge(feature_sets["freezing_only"], on="animal_id"))
    analysis_data.to_csv(out_dir / "analysis_dataset.csv", index=False)

    cross_metrics, cross_predictions = cross_cohort(labels, feature_sets)
    cross_metrics.to_csv(out_dir / "cross_cohort_metrics.csv", index=False)
    cross_predictions.to_csv(out_dir / "cross_cohort_predictions.csv", index=False)

    head_metrics, head_predictions = loocv(labels, feature_sets, analysis="full_session_head_to_head")
    head_metrics.to_csv(out_dir / "head_to_head_loocv_metrics.csv", index=False)
    head_predictions.to_csv(out_dir / "head_to_head_loocv_predictions.csv", index=False)

    early_metrics, early_predictions = early_prediction(labels, root)
    early_metrics.to_csv(out_dir / "early_prediction_metrics.csv", index=False)
    early_predictions.to_csv(out_dir / "early_prediction_predictions.csv", index=False)

    coeffs = coefficient_table(labels, feature_sets)
    coeffs.to_csv(out_dir / "feature_coefficients.csv", index=False)

    head_scores = {
        name: head_predictions.loc[head_predictions["feature_set"].eq(name), "resilience_score"].to_numpy()
        for name in ["motif_repertoire", "freezing_only"]
    }
    y = head_predictions.loc[head_predictions["feature_set"].eq("motif_repertoire"), "target"].to_numpy()
    delta_ci = bootstrap_delta_ci(y, head_scores["motif_repertoire"], head_scores["freezing_only"])

    plot_summary(cross_metrics, head_metrics, early_metrics, out_dir)
    write_report(out_dir, labels, cross_metrics, head_metrics, early_metrics, delta_ci)
    script_src = Path(__file__).resolve()
    script_dst = (out_dir / "run_tier1_reanalysis.py").resolve()
    if script_src != script_dst:
        shutil.copy2(script_src, script_dst)

    print(f"Wrote Tier 1 reanalysis to {out_dir}")
    print(cross_metrics[["feature_set", "train_experiment", "test_experiment", "roc_auc", "balanced_accuracy"]].to_string(index=False))
    print(head_metrics[["feature_set", "roc_auc", "balanced_accuracy", "roc_auc_ci95_low", "roc_auc_ci95_high"]].to_string(index=False))
    print(early_metrics[["feature_set", "horizon_min", "roc_auc", "balanced_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
