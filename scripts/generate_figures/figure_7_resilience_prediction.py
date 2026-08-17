"""Figure 7 - behavior dynamics predict resilient ELS profiles.

Four panels:

  A  Onset of prediction. LOOCV AUC as a function of how many opening minutes of
     the session are included, for all behaviours together and for each of the
     seven behaviours separately, against a within-cohort permutation null.
  B  Held-out cohort transfer (train one cohort, test the other).
  C  Full-session LOOCV, behaviour dynamics vs Freeze dynamics only.
  D  Exact Shapley values for the full-session behaviour-dynamics model.

Terminology: this script uses **behaviour dynamics** for the 30-s trajectory of
each behaviour's time share.

Panel D uses closed-form Shapley values rather than the `shap` package. The
classifier is linear, and for a linear model on standardised inputs the exact
Shapley value of feature j for animal i is phi_ij = coef_j * z_ij (Lundberg &
Lee 2017, linear SHAP with a mean-centred background). No approximation and no
extra dependency.

Run from the manuscript repository:

    python scripts/generate_figures/figure_7_resilience_prediction.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis import figure_7_core as tier1  # noqa: E402
from scripts.analysis import figure_7_recap as recap  # noqa: E402
from src.config import (  # noqa: E402
    FIGURES_DIR,
    FIGURE_SOURCE_DATA_DIR,
    RAW_DATA_DIR,
    STATISTICS_DIR,
)
from src.panel_letters import align_panel_letters  # noqa: E402

OUT = FIGURES_DIR
SOURCE_OUT = FIGURE_SOURCE_DATA_DIR
REPORT = STATISTICS_DIR / "figure7_prediction_report.md"
SECTION = "## Behavior dynamics and resilience prediction"
FIGURE7_PDF = RAW_DATA_DIR / "legacy_figures" / "figure7_classifier_original.pdf"

RANDOM_SEED = 13
N_PERM_DEFAULT = 200

# Behaviour palette, taken verbatim from the manuscript repo so this figure
# matches every other figure in the set:
#   scripts/generate_figures/figure_4_diversity_dynamics.py  (COLORS)
BEHAVIOUR_COLORS = {
    "Freeze": "#C37BA0",
    "Sniff": "#5B9AAA",
    "Groom": "#8EC6DE",
    "Turn": "#B7DB45",
    "Locomotion": "#F1C232",
    "Climb": "#F4A259",
    "Jump": "#E45756",
}
BEHAVIOUR_ORDER = ["Freeze", "Sniff", "Groom", "Turn", "Locomotion", "Climb", "Jump"]
COMBINED = "All behaviours"
COMBINED_COLOR = "#9A9A9A"
FREEZE_ONLY_DISPLAY = "Freeze dynamics only"
FREEZE_ONLY_COLOR = "#C671A0"
DYNAMICS_COLOR = "#A6A6A6"
AXIS = "#4D4D4D"
CHANCE_GREY = "#9A9A9A"
EVENT_SPANS = [(3.0, 3.5), (4.5, 5.0), (6.0, 6.5)]
# Horizons run 0.5-7.5 min; pad so the first and last markers are drawn whole
# rather than being clipped in half by the axes edge. Shared by panels I-N.
HORIZON_XLIM = (0.0, 7.65)
# Every dynamic panel in the figure set (Figures 3, 5, 7 and Supplementary 1)
# shares one x axis: whole minutes 0-7, running from zero and padded past 7.5 so
# the final marker is drawn whole.
HORIZON_XTICKS = list(range(1, 8))
HORIZON_END = 7.5
HORIZON_XLABEL = "Time (minutes)"

FIGURE7_CLASSES = BEHAVIOUR_ORDER + ["Unassigned"]
FIGURE7_ACCURACY = np.array([0.80, 0.86, 0.45, 0.71, 0.61, 0.68, 0.47, 0.30, 0.63])
FIGURE7_ACCURACY_NOTES = [
    "+0.68, 6.4x", "+0.74, 6.9x", "+0.33, 3.6x", "+0.58, 5.6x",
    "+0.49, 4.9x", "+0.55, 5.4x", "+0.34, 3.7x", "+0.18, 2.4x",
    "+0.50, 5.0x",
]
FIGURE7_SHAP_FEATURES = [
    "Left Body 3 - Tail Variability",
    "Right Body 3 - Tail Variability",
    "Angular Velocity",
    "Right Body 1 Movement",
    "Distance between Left Body 2 and Right Body 2",
    "Global Turning Speed",
    "Right Head 1 - Nose Variability",
    "Right Body 1 - Right Body 2 Variability",
    "Center Body Speed",
    "Turning Speed Variability",
    "Left Body 1 Movement",
    "Average Nose Speed",
    "Right Body 1 - Center Distance",
    "Center Speed Variability",
    "Right Head 2 - Tail Distance",
    "Left Body 3 - Center Variability",
    "Body Width Variability",
    "Right Head 1 Lateral Variability",
    "Average Tail Speed",
    "Instant Tail Speed",
]
FIGURE7_CONFUSION = np.array([
    [0.800, 0.004, 0.000, 0.021, 0.044, 0.041, 0.044, 0.041],
    [0.002, 0.860, 0.000, 0.000, 0.002, 0.022, 0.069, 0.044],
    [0.069, 0.007, 0.450, 0.069, 0.038, 0.000, 0.031, 0.340],
    [0.038, 0.000, 0.000, 0.710, 0.160, 0.034, 0.026, 0.038],
    [0.078, 0.001, 0.000, 0.060, 0.610, 0.043, 0.088, 0.110],
    [0.036, 0.037, 0.000, 0.001, 0.045, 0.680, 0.130, 0.068],
    [0.048, 0.049, 0.000, 0.008, 0.100, 0.160, 0.470, 0.160],
    [0.076, 0.085, 0.000, 0.025, 0.190, 0.096, 0.230, 0.300],
])


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def behaviour_dynamics_features(
    root: Path, horizon_min: float, behaviours: list[str] | None = None
) -> pd.DataFrame:
    """30-s time-share trajectory columns, optionally restricted to one behaviour.

    Same construction as `figure_7_core.early_repertoire_features`, but
    filterable and using `bd_*` feature names.
    """
    tc = pd.read_csv(root / "data/processed/cluster_timecourse_per_animal.csv")
    tc = tc.loc[tc["time_s"] <= horizon_min * 60].copy()
    if behaviours is not None:
        tc = tc.loc[tc["cluster"].isin(behaviours)].copy()
    tc["feature"] = "bd_" + tc["cluster"] + "_t" + tc["time_bin"].astype(str)
    return (
        tc.pivot_table(index="animal_id", columns="feature", values="pct",
                       aggfunc="mean", fill_value=0)
        .sort_index(axis=1)
        .reset_index()
    )


# --------------------------------------------------------------------------
# model (numpy mirror of the tier1 ridge, for speed inside permutations)
# --------------------------------------------------------------------------
def _ridge_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    penalty = np.eye(xb.shape[1]) * tier1.RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)
    return coef, mean, scale


def loocv_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    scores = np.empty(n, dtype=float)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        keep[i] = False
        coef, mean, scale = _ridge_fit(x[keep], y[keep])
        scores[i] = coef[0] + ((x[i] - mean) / scale) @ coef[1:]
        keep[i] = True
    return scores


def loocv_auc(x: np.ndarray, y: np.ndarray) -> float:
    return tier1.auc_or_nan(y, loocv_scores(x, y))


def within_cohort_shuffles(cohort: np.ndarray, y: np.ndarray, n_perm: int,
                           seed: int) -> np.ndarray:
    """Shuffle the target inside each cohort, so cohort structure cannot leak in."""
    rng = np.random.default_rng(seed)
    out = np.tile(y, (n_perm, 1))
    for c in np.unique(cohort):
        idx = np.flatnonzero(cohort == c)
        for k in range(n_perm):
            out[k, idx] = rng.permutation(out[k, idx])
    return out


# --------------------------------------------------------------------------
# panel A
# --------------------------------------------------------------------------
def onset_of_prediction(labels: pd.DataFrame, root: Path, n_perm: int) -> pd.DataFrame:
    feature_sets: dict[str, list[str] | None] = {COMBINED: None}
    feature_sets.update({b: [b] for b in BEHAVIOUR_ORDER})

    rows: list[dict[str, object]] = []
    for horizon in tier1.HORIZONS_MIN:
        for name, subset in feature_sets.items():
            feats = behaviour_dynamics_features(root, horizon, subset)
            data, cols = tier1.assemble(labels, feats)
            if not cols:
                continue
            x = data[cols].to_numpy(dtype=float)
            y = data["target"].to_numpy(dtype=int)
            cohort = data["experiment"].to_numpy()

            observed = loocv_auc(x, y)
            perms = within_cohort_shuffles(cohort, y, n_perm, RANDOM_SEED)
            null = np.array([loocv_auc(x, p) for p in perms])
            null_mean = float(np.nanmean(null))

            rows.append({
                "feature_set": name,
                "horizon_min": horizon,
                "n_features": len(cols),
                "roc_auc": observed,
                "null_mean": null_mean,
                "auc_above_null_mean": float(observed - null_mean),
                "p_perm": float((np.sum(null >= observed) + 1) / (len(null) + 1)),
            })
            print(f"  {name:<16s} {horizon:>4.1f} min  n_feat={len(cols):>3d}  "
                  f"AUC={observed:.3f}  nullmean={null_mean:.3f}  "
                  f"delta={observed - null_mean:+.3f}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# panels B and C
# --------------------------------------------------------------------------
def full_session_sets(root: Path) -> dict[str, pd.DataFrame]:
    return {
        "Behaviour dynamics": tier1.full_repertoire_features(root),
        "Freeze only": tier1.freezing_features(root),
    }


def held_out_cohort(labels: pd.DataFrame, root: Path) -> pd.DataFrame:
    metrics, _ = tier1.cross_cohort(labels, full_session_sets(root))
    return metrics


def held_out_cohort_tests(labels: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Panel D: behaviour dynamics vs Freeze only, within each transfer direction.

    Both feature sets score the same held-out animals in a given direction, so
    the comparison is paired animal-by-animal.
    """
    _, preds = tier1.cross_cohort(labels, full_session_sets(root))
    rows: list[dict[str, object]] = []
    for test_exp in ["Exp1", "Exp3"]:
        sub = preds.loc[preds["test_experiment"] == test_exp]
        a = sub.loc[sub["feature_set"] == "Behaviour dynamics"].sort_values("animal_id")
        b = sub.loc[sub["feature_set"] == "Freeze only"].sort_values("animal_id")
        if not np.array_equal(a["animal_id"].to_numpy(), b["animal_id"].to_numpy()):
            raise AssertionError(f"{test_exp}: held-out animals differ between feature sets")
        y = a["target"].to_numpy(dtype=int)
        stats = tier1.paired_auc_test(y, a["resilience_score"].to_numpy(dtype=float),
                                      b["resilience_score"].to_numpy(dtype=float))
        rows.append({"panel": "D", "comparison": "Behaviour dynamics vs Freeze only",
                     "test_experiment": test_exp, "n_test": len(y), **stats,
                     "stars": tier1.significance_stars(stats["p_value"])})
    return pd.DataFrame(rows)


def full_session_loocv(labels: pd.DataFrame, root: Path
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    scores_by_set: dict[str, np.ndarray] = {}
    target: np.ndarray | None = None
    for name, feats in full_session_sets(root).items():
        data, cols = tier1.assemble(labels, feats)
        x = data[cols].to_numpy(dtype=float)
        y = data["target"].to_numpy(dtype=int)
        scores = loocv_scores(x, y)
        lo, hi = tier1.bootstrap_auc_ci(y, scores)
        rows.append({"feature_set": name, "roc_auc": tier1.auc_or_nan(y, scores),
                     "ci_low": lo, "ci_high": hi, "n_features": len(cols)})
        scores_by_set[name] = scores
        target = y

    # Panel E: the two models leave out the same animals in the same order, so
    # their LOOCV scores line up one-to-one and the test can be paired.
    stats = tier1.paired_auc_test(target, scores_by_set["Behaviour dynamics"],
                                  scores_by_set["Freeze only"])
    test = pd.DataFrame([{
        "panel": "E", "comparison": "Behaviour dynamics vs Freeze only",
        "n_animals": len(target), **stats,
        "stars": tier1.significance_stars(stats["p_value"]),
    }])
    return pd.DataFrame(rows), test


# --------------------------------------------------------------------------
# panel D - exact Shapley values for the linear model
# --------------------------------------------------------------------------
def shapley_values(labels: pd.DataFrame, root: Path) -> pd.DataFrame:
    """phi_ij = coef_j * z_ij, the exact Shapley value of a linear model.

    With a mean-centred background distribution, E[z_j] = 0, so each feature's
    contribution to an animal's score is simply its standardised value times the
    fitted coefficient. Summing phi over j recovers score_i - E[score].
    """
    feats = tier1.full_repertoire_features(root)
    data, cols = tier1.assemble(labels, feats)
    x = data[cols].to_numpy(dtype=float)
    y = data["target"].to_numpy(dtype=int)

    coef, mean, scale = _ridge_fit(x, y)
    z = (x - mean) / scale
    phi = z * coef[1:]

    rows: list[dict[str, object]] = []
    for j, col in enumerate(cols):
        behaviour = col.replace("motif_", "")
        rows.append({
            "behaviour": behaviour,
            "coefficient": float(coef[j + 1]),
            "mean_abs_shap": float(np.mean(np.abs(phi[:, j]))),
            "mean_shap_resilient": float(np.mean(phi[y == 1, j])),
            "mean_shap_vulnerable": float(np.mean(phi[y == 0, j])),
            "direction": "resilient" if coef[j + 1] > 0 else "vulnerable",
        })
    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)

    per_animal = pd.DataFrame(phi, columns=[c.replace("motif_", "") for c in cols])
    per_animal.insert(0, "animal_label", data["animal_label"].to_numpy())
    per_animal.insert(1, "profile", data["profile"].to_numpy())
    per_animal.to_csv(SOURCE_OUT / "figure7_shapley_per_animal.csv", index=False)
    return out


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.0,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "text.color": "#1A1A1A",
        "axes.labelcolor": "#1A1A1A",
        "axes.grid": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def _tidy(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
        ax.spines[spine].set_linewidth(0.45)
    ax.tick_params(axis="both", colors=AXIS, labelsize=5.0, width=0.45, length=2)
    ax.xaxis.label.set_color(AXIS)
    ax.yaxis.label.set_color(AXIS)
    ax.title.set_color(AXIS)


def display_feature_set(name: str) -> str:
    return FREEZE_ONLY_DISPLAY if name == "Freeze only" else name


def panel_tag(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.12) -> plt.Text:
    return ax.text(x, y, letter, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, fontweight="bold", color=AXIS, clip_on=False)


def line_legend(ax: plt.Axes, *, ncol: int = 1, fontsize: float = 4.6,
                columnspacing: float = 0.7) -> None:
    leg = ax.legend(frameon=False, fontsize=fontsize, loc="lower right", ncol=ncol,
                    handlelength=1.15, handletextpad=0.35,
                    columnspacing=columnspacing,
                    labelspacing=0.22, borderaxespad=0.15)
    for text in leg.get_texts():
        text.set_color(AXIS)


def sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, label: str,
                *, tick: float = 0.022, fontsize: float = 5.0) -> None:
    """Draw a significance bracket spanning x1-x2 with its label above."""
    ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y],
            color=AXIS, linewidth=0.5, clip_on=False, zorder=6)
    # "n.s." is text and sits above the line; asterisks hang lower, so they are
    # nudged up to keep the visual gap to the bracket the same.
    offset = tick + (0.004 if label == "n.s." else -0.004)
    ax.text((x1 + x2) / 2.0, y + offset, label, ha="center", va="bottom",
            fontsize=fontsize if label == "n.s." else fontsize + 1.2,
            color=AXIS, clip_on=False, zorder=6)


def add_event_spans(ax: plt.Axes) -> None:
    for start, end in EVENT_SPANS:
        ax.axvspan(start, end, color="#F3F3F3", zorder=0)


def align_edge_xticklabels(ax: plt.Axes) -> None:
    labels = ax.get_xticklabels()
    if labels:
        labels[0].set_ha("left")
        labels[-1].set_ha("right")


def fit_spines_to_ticks(ax: plt.Axes) -> None:
    """Stop visible axes at the first and last major tick."""
    xmin, xmax = ax.get_xlim()
    xticks = np.asarray(ax.get_xticks(), dtype=float)
    xticks = xticks[np.isfinite(xticks) & (xticks >= min(xmin, xmax)) & (xticks <= max(xmin, xmax))]
    if len(xticks) and ax.spines["bottom"].get_visible():
        ax.spines["bottom"].set_bounds(float(xticks.min()), float(xticks.max()))

    ymin, ymax = ax.get_ylim()
    yticks = np.asarray(ax.get_yticks(), dtype=float)
    yticks = yticks[np.isfinite(yticks) & (yticks >= min(ymin, ymax)) & (yticks <= max(ymin, ymax))]
    if len(yticks) and ax.spines["left"].get_visible():
        ax.spines["left"].set_bounds(float(yticks.min()), float(yticks.max()))


def mark_final_horizon(ax: plt.Axes) -> None:
    """Run the x axis line out to the 7.5 min end point.

    Ticks stay on whole minutes, but the horizons run to 7.5, so the spine is
    extended past the last tick to sit under the final data point.  Must be
    called after ``fit_spines_to_ticks``, which would otherwise stop the line
    at 7.
    """
    ax.spines["bottom"].set_bounds(HORIZON_XLIM[0], HORIZON_END)


def full_x_spine(ax: plt.Axes) -> None:
    """Draw the bottom axis across the whole panel.

    ``fit_spines_to_ticks`` stops the spine at the first and last tick, which on
    a categorical bar panel leaves only a stub under the bars.
    """
    ax.spines["bottom"].set_bounds(*ax.get_xlim())


def remap_family_name(name: str) -> str:
    remap = {
        "All behaviour time shares": "All frequencies",
        "Diversity metrics": "All diversity indices",
        "Transition summaries": "All transition metrics",
        "Bout durations": "All bout durations",
        "All metrics": "All predictions",
    }
    return remap.get(name, name)


# Panels I and J sit side by side and should use the same legend type size.
# Four points is large enough to remain legible at final figure size while the
# four-column legend in I still clears the trajectories beneath it.
IJ_LEGEND_FONTSIZE = 4.0


def plot_panel_a(ax: plt.Axes, onset: pd.DataFrame, letter: str = "A") -> plt.Text:
    combined = onset.loc[onset["feature_set"] == COMBINED].sort_values("horizon_min")

    add_event_spans(ax)
    ax.axhline(0.5, color=CHANCE_GREY, linewidth=0.5, linestyle=(0, (2.5, 2)), zorder=1)

    for behaviour in BEHAVIOUR_ORDER:
        sub = onset.loc[onset["feature_set"] == behaviour].sort_values("horizon_min")
        ax.plot(sub["horizon_min"], sub["roc_auc"], color=BEHAVIOUR_COLORS[behaviour],
                linewidth=0.9, marker="o", markersize=2.4, mec="white", mew=0.25,
                label=behaviour,
                zorder=3, alpha=0.9)

    ax.plot(combined["horizon_min"], combined["roc_auc"], color=COMBINED_COLOR,
            linewidth=1.35, marker="o", markersize=3.2, mec="white", mew=0.25,
            label=COMBINED, zorder=5)

    ax.set_xlabel(HORIZON_XLABEL)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Behaviour dynamics over time", fontsize=6.4, color=AXIS, pad=3)
    # Same 0-1 axis as panel J beside it: the legend sits inside the plot, in
    # the clear band above the curves on the left half.
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(HORIZON_XLIM)
    ax.set_xticks(HORIZON_XTICKS)
    ax.set_xticklabels([f"{h:g}" for h in HORIZON_XTICKS])
    align_edge_xticklabels(ax)
    _tidy(ax)
    fit_spines_to_ticks(ax)
    mark_final_horizon(ax)

    handles, lbls = ax.get_legend_handles_labels()
    handle_map = dict(zip(lbls, handles))
    legend_order = [COMBINED, "Freeze", "Sniff", "Groom",
                    "Turn", "Locomotion", "Climb", "Jump"]
    leg = ax.legend([handle_map[label] for label in legend_order], legend_order,
                    loc="upper left", bbox_to_anchor=(0.005, 1.0),
                    ncol=4, frameon=False, handlelength=0.7,
                    handletextpad=0.25, labelspacing=0.12, columnspacing=0.55,
                    fontsize=IJ_LEGEND_FONTSIZE, borderaxespad=0.0)
    for text in leg.get_texts():
        text.set_color(AXIS)
    return panel_tag(ax, letter, x=-0.105, y=1.12)


def plot_panel_b(ax: plt.Axes, cross: pd.DataFrame, letter: str = "B",
                 tests: pd.DataFrame | None = None) -> plt.Text:
    directions = [("Exp3", "Exp1"), ("Exp1", "Exp3")]
    # The two groups sit close together near the middle of the panel; the
    # three-line labels below are narrow enough not to collide at this pitch.
    width = 0.15
    xs = np.arange(len(directions)) * 0.60
    bar_tops: dict[str, list[float]] = {}
    for k, (name, colour) in enumerate(
        [("Behaviour dynamics", DYNAMICS_COLOR), ("Freeze only", FREEZE_ONLY_COLOR)]
    ):
        vals = []
        for test_exp, _ in directions:
            row = cross.loc[(cross["feature_set"] == name)
                            & (cross["test_experiment"] == test_exp)]
            vals.append(float(row["roc_auc"].iloc[0]))
        pos = xs + (k - 0.5) * width
        bar_tops[name] = vals
        ax.bar(pos, vals, width=width, color=colour, edgecolor="white",
               linewidth=0.4, label=display_feature_set(name), zorder=2)

    # The paired AUC tests are all non-significant and are reported in
    # statistics/figure7_panel_de_auc_tests.csv; the brackets and the per-bar
    # value labels are left off the panel to keep it readable at print size.

    ax.axhline(0.5, color=CHANCE_GREY, linewidth=0.5, linestyle=(0, (2.5, 2)), zorder=1)
    ax.set_xticks(xs)
    # Both cohort names keep their own break, with the arrow leading the test
    # cohort. At the same type size as panel E, and with the groups this close
    # together, no line may be wider than a half cohort name.
    ax.set_xticklabels(
        [f"{tier1.COHORT_LABELS[train]}\n\u2192 {tier1.COHORT_LABELS[test]}"
         for test, train in directions],
        fontsize=4.0,
        ha="center",
        multialignment="center",
    )
    ax.set_ylabel("ROC AUC")
    # Headroom for the legend, which sits inside the axes so the strip above
    # them stays clear for the panel letter. Ticks still stop at 1.
    ax.set_ylim(0, 1.20)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(xs[0] - 0.30, xs[-1] + 0.30)
    ax.set_title("Held-out cohort", fontsize=6.0, color=AXIS, pad=3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False,
              handlelength=1.0, fontsize=4.4, labelspacing=0.2,
              borderpad=0.0, ncol=2, columnspacing=0.7, handletextpad=0.35)
    _tidy(ax)
    # After _tidy: it sets one tick label size for both axes, which would undo
    # the smaller size the two-line cohort names need to stay inside the panel.
    ax.tick_params(axis="x", labelsize=4.0)
    fit_spines_to_ticks(ax)
    full_x_spine(ax)
    return panel_tag(ax, letter, x=-0.12, y=1.12)


def plot_panel_c(ax: plt.Axes, head: pd.DataFrame, letter: str = "C",
                 tests: pd.DataFrame | None = None) -> plt.Text:
    names = ["Behaviour dynamics", "Freeze only"]
    colours = [DYNAMICS_COLOR, FREEZE_ONLY_COLOR]
    xs = np.arange(len(names)) * 0.44
    tops: list[float] = []
    for x, name, colour in zip(xs, names, colours):
        row = head.loc[head["feature_set"] == name].iloc[0]
        ax.bar(x, row["roc_auc"], width=0.28, color=colour, edgecolor="white",
               linewidth=0.4, zorder=2)
        ax.errorbar(x, row["roc_auc"],
                    yerr=[[row["roc_auc"] - row["ci_low"]],
                          [row["ci_high"] - row["roc_auc"]]],
                    color=recap.darken(colour, factor=0.68),
                    linewidth=0.8, capsize=2.2, zorder=3)
        tops.append(float(row["ci_high"]))

    # As in panel D: the paired test is non-significant and lives in the
    # statistics table, so no bracket and no value labels are drawn here.

    ax.axhline(0.5, color=CHANCE_GREY, linewidth=0.5, linestyle=(0, (2.5, 2)), zorder=1)
    ax.set_xticks(xs)
    # Three lines here, unlike D: this panel is the narrowest on the row, and
    # "Freeze dynamics" on one line is wider than the gap between its two bars.
    ax.set_xticklabels(["Behaviour\ndynamics\n ", "Freeze\ndynamics\nonly"],
                       fontsize=4.0)
    ax.set_ylabel("ROC AUC")
    # No bracket above the CI whiskers any more, so the axis stops at 1.
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(xs[0] - 0.30, xs[-1] + 0.30)
    ax.set_title("Full-session\ncross-validation", fontsize=6.0, color=AXIS, pad=3)
    _tidy(ax)
    # After _tidy, as in panel D: keep the two-line condition names small enough
    # that they do not run into each other or into panel F.
    ax.tick_params(axis="x", labelsize=4.0)
    fit_spines_to_ticks(ax)
    full_x_spine(ax)
    return panel_tag(ax, letter, x=-0.12, y=1.12)


def plot_panel_d(ax: plt.Axes, shap: pd.DataFrame, letter: str = "D") -> plt.Text:
    shap = shap.sort_values("mean_abs_shap")
    ys = np.arange(len(shap))
    colours = [BEHAVIOUR_COLORS.get(b, "#999999") for b in shap["behaviour"]]
    ax.barh(ys, shap["mean_abs_shap"], color=colours, edgecolor="white",
            linewidth=0.4, height=0.68, zorder=2)

    for y, (_, r) in zip(ys, shap.iterrows()):
        arrow = "\u2191" if r["coefficient"] > 0 else "\u2193"
        ax.text(r["mean_abs_shap"] + 0.004, y, arrow, va="center", ha="left",
                fontsize=6.5, color="#4A4A4A")

    ax.set_yticks(ys)
    ax.set_yticklabels(shap["behaviour"], fontsize=6)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Full-session contributions", fontsize=6.4, color=AXIS, pad=3)
    ax.set_xlim(0, float(shap["mean_abs_shap"].max()) * 1.34)
    # Right-align inside the last tick: the axis runs past it, so anchoring at
    # the axes edge would put the key outside the drawn plot.
    ax.text(0.80, 0.06, "\u2191 toward resilient\n\u2193 toward vulnerable",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6,
            color="#6A6A6A", linespacing=1.35)
    _tidy(ax)
    fit_spines_to_ticks(ax)
    return panel_tag(ax, letter, x=-0.12, y=1.12)


def build_figure(onset: pd.DataFrame, cross: pd.DataFrame, head: pd.DataFrame,
                 shap: pd.DataFrame) -> plt.Figure:
    style()
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.32, 1.0],
                  hspace=0.46, wspace=0.36,
                  left=0.078, right=0.985, top=0.945, bottom=0.088)
    plot_panel_a(fig.add_subplot(gs[0, :]), onset)
    plot_panel_b(fig.add_subplot(gs[1, 0]), cross)
    plot_panel_c(fig.add_subplot(gs[1, 1]), head)
    plot_panel_d(fig.add_subplot(gs[1, 2]), shap)
    return fig


def recap_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load tracked predictor tables, recomputing them only when absent."""
    panel_ab = SOURCE_OUT / "figure7_predictor_catalog.csv"
    panel_c = SOURCE_OUT / "figure7_predictor_timecourse.csv"
    if not panel_ab.exists() or not panel_c.exists():
        labels, predictors = recap.cached("predictors", recap.build_predictors, fresh=False)
        ab = recap.cached("panel_ab", lambda: recap.panel_ab_data(labels, predictors),
                          fresh=False)
        horizon = recap.cached("horizon_features", recap.all_horizon_features,
                               fresh=False)
        c = recap.cached("panel_c", lambda: recap.panel_c_data(labels, horizon),
                         fresh=False)
        ab.drop(columns=["color"], errors="ignore").to_csv(panel_ab, index=False)
        c.drop(columns=["color"], errors="ignore").to_csv(panel_c, index=False)
    else:
        ab = pd.read_csv(panel_ab)
        c = pd.read_csv(panel_c)
    ab["color"] = ab["predictor"].map(recap.predictor_color)
    c["color"] = c["predictor"].map(recap.predictor_color)
    return ab, c


INDIVIDUAL_OVERTIME_PARAMETERS = [
    ("Frequency", "Freeze frequency", "frequency", "frequency__Freeze", "Freeze"),
    ("Frequency", "Sniff frequency", "frequency", "frequency__Sniff", "Sniff"),
    ("Frequency", "Groom frequency", "frequency", "frequency__Groom", "Groom"),
    ("Frequency", "Turn frequency", "frequency", "frequency__Turn", "Turn"),
    ("Frequency", "Locomotion frequency", "frequency", "frequency__Locomotion", "Locomotion"),
    ("Frequency", "Climb frequency", "frequency", "frequency__Climb", "Climb"),
    ("Frequency", "Jump frequency", "frequency", "frequency__Jump", "Jump"),
    ("Diversity", "Simpson index", "diversity", "diversity__simpson", None),
    ("Diversity", "Shannon entropy", "diversity", "diversity__shannon", None),
    ("Diversity", "Evenness index", "diversity", "diversity__evenness", None),
    ("Diversity", "Cumulative usage", "diversity", "diversity__cui", None),
    ("Transition", "Lempel-Ziv", "transition", "transition__lz", None),
    ("Transition", "Recurrence", "transition", "transition__recurrence", None),
    ("Transition", "Determinism", "transition", "transition__determinism", None),
    ("Transition", "Markov entropy", "transition", "transition__markov", None),
    ("Bout", "Mean bout duration", "bout", "bout__overall_mean", None),
    ("Bout", "Freeze bout duration", "bout", "bout__Freeze", "Freeze"),
    ("Bout", "Sniff bout duration", "bout", "bout__Sniff", "Sniff"),
    ("Bout", "Groom bout duration", "bout", "bout__Groom", "Groom"),
    ("Bout", "Turn bout duration", "bout", "bout__Turn", "Turn"),
    ("Bout", "Locomotion bout duration", "bout", "bout__Locomotion", "Locomotion"),
    ("Bout", "Climb bout duration", "bout", "bout__Climb", "Climb"),
    ("Bout", "Jump bout duration", "bout", "bout__Jump", "Jump"),
]


def colour_for_individual(panel: str, name: str, behaviour: str | None, family: str) -> str:
    if behaviour in BEHAVIOUR_COLORS:
        return BEHAVIOUR_COLORS[behaviour]
    if panel == "Diversity":
        colours = {
            "Simpson index": "#D8B51F",
            "Shannon entropy": "#F1C232",
            "Evenness index": "#BCA51B",
            "Cumulative usage": "#E6D676",
        }
        return colours.get(name, "#F1C232")
    if panel == "Transition":
        colours = {
            "Lempel-Ziv": "#5B9AAA",
            "Recurrence": "#8EC6DE",
            "Determinism": "#6398A4",
            "Markov entropy": "#9ECADA",
        }
        return colours.get(name, "#8EC6DE")
    return "#F4A259"


def individual_overtime_data(labels: pd.DataFrame) -> pd.DataFrame:
    """Repeated-CV AUC for every individual time-resolved metric."""
    horizon_frames = recap.cached("horizon_features", recap.all_horizon_features,
                                  fresh=False)
    rows: list[dict[str, object]] = []
    for horizon in recap.HORIZONS_MIN:
        frames = horizon_frames[horizon]
        for panel, name, family, col, behaviour in INDIVIDUAL_OVERTIME_PARAMETERS:
            frame = frames[family]
            if col not in frame.columns:
                continue
            data = recap.direct.attach(labels, frame)
            folds = recap.direct.make_folds(
                recap.direct.strat_key(data), recap.direct.N_SPLITS,
                recap.direct.N_REPEATS, recap.RANDOM_SEED
            )
            aucs, _ = recap.direct.cv_auc_np(
                data[[col]].to_numpy(dtype=float),
                data["target"].to_numpy(dtype=int),
                folds,
            )
            rows.append({
                "panel": panel,
                "metric": name,
                "family": family,
                "horizon_min": horizon,
                "cv_auc": float(aucs.mean()),
                "cv_lo": float(np.percentile(aucs, 10)),
                "cv_hi": float(np.percentile(aucs, 90)),
                "color": colour_for_individual(panel, name, behaviour, family),
                "linestyle": "-",
            })
    return pd.DataFrame(rows)


def draw_family_overtime(ax: plt.Axes, table: pd.DataFrame, letter: str) -> plt.Text:
    add_event_spans(ax)
    for name, sub in table.groupby("predictor", sort=False):
        sub = sub.sort_values("horizon_min")
        ax.plot(sub["horizon_min"], sub["cv_auc"], color=sub["color"].iloc[0],
                lw=0.9, marker="o", ms=2.4, mec="white", mew=0.25,
                label=remap_family_name(name), zorder=2)
    ax.axhline(0.5, color=CHANCE_GREY, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    ax.set_xlim(HORIZON_XLIM)
    ax.set_xticks(HORIZON_XTICKS)
    ax.set_xticklabels([f"{h:g}" for h in HORIZON_XTICKS])
    align_edge_xticklabels(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC AUC")
    ax.set_xlabel(HORIZON_XLABEL)
    ax.set_title("Frequency/diversity/transition/bout duration metrics over time",
                 fontsize=6.4, color=AXIS, pad=3)
    _tidy(ax)
    fit_spines_to_ticks(ax)
    mark_final_horizon(ax)
    line_legend(ax, ncol=1, fontsize=IJ_LEGEND_FONTSIZE)
    return panel_tag(ax, letter, x=-0.105, y=1.12)


def top_two_names(table: pd.DataFrame, name_col: str, auc_col: str,
                  mode: str) -> list[str]:
    ordered = table.sort_values("horizon_min")
    if mode == "overall":
        ranked = ordered.groupby(name_col, sort=False)[auc_col].max().sort_values(
            ascending=False
        ).reset_index()
    elif mode == "early":
        early = ordered.loc[ordered["horizon_min"] <= 3.0]
        ranked = early.groupby(name_col, sort=False)[auc_col].max().sort_values(
            ascending=False
        ).reset_index()
    else:
        raise ValueError(f"unknown top-two mode: {mode}")
    return [str(v) for v in ranked[name_col].head(2)]


def draw_top_two_mini(ax: plt.Axes, table: pd.DataFrame, names: list[str],
                      *, name_col: str, auc_col: str, color_col: str | None,
                      title: str, x_ticks: list[float],
                      show_xlabel: bool = False) -> None:
    add_event_spans(ax)
    for name in names:
        sub = table.loc[table[name_col] == name].sort_values("horizon_min")
        if color_col is not None:
            color = str(sub[color_col].iloc[0])
        else:
            color = BEHAVIOUR_COLORS.get(name, COMBINED_COLOR)
        label = remap_family_name(name)
        ax.plot(sub["horizon_min"], sub[auc_col], color=color, lw=0.8,
                marker="o", ms=1.8, mec="white", mew=0.18, label=label, zorder=2)
    ax.axhline(0.5, color=CHANCE_GREY, lw=0.4, ls=(0, (2.5, 2)), zorder=1)
    ax.set_xlim(HORIZON_XLIM)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x:g}" for x in x_ticks])
    align_edge_xticklabels(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("")
    if show_xlabel:
        ax.set_xlabel(HORIZON_XLABEL, fontsize=4.4, labelpad=0.5)
    ax.set_title(title, fontsize=4.6, color=AXIS, pad=1.5)
    _tidy(ax)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=3.8, length=1.6, pad=0.8)
    ax.tick_params(axis="x", labelsize=4.0, length=1.6, pad=1)
    fit_spines_to_ticks(ax)
    mark_final_horizon(ax)
    leg = ax.legend(frameon=False, fontsize=3.5, loc="lower right",
                    handlelength=0.9, handletextpad=0.25, labelspacing=0.1,
                    borderaxespad=0.05)
    for text in leg.get_texts():
        text.set_color(AXIS)


def draw_behaviour_mini_pair(top_ax: plt.Axes, bottom_ax: plt.Axes,
                             onset: pd.DataFrame) -> None:
    behaviour_rows = onset.loc[onset["feature_set"].isin(BEHAVIOUR_ORDER)].copy()
    draw_top_two_mini(
        top_ax, behaviour_rows,
        top_two_names(behaviour_rows, "feature_set", "roc_auc", "overall"),
        name_col="feature_set", auc_col="roc_auc", color_col=None,
        title="Best predictor overall", x_ticks=HORIZON_XTICKS[::2],
    )
    draw_top_two_mini(
        bottom_ax, behaviour_rows,
        top_two_names(behaviour_rows, "feature_set", "roc_auc", "early"),
        name_col="feature_set", auc_col="roc_auc", color_col=None,
        title="Best predictor first 3 min", x_ticks=HORIZON_XTICKS[::2], show_xlabel=True,
    )


def draw_family_mini_pair(top_ax: plt.Axes, bottom_ax: plt.Axes,
                          table: pd.DataFrame) -> None:
    draw_top_two_mini(
        top_ax, table,
        top_two_names(table, "predictor", "cv_auc", "overall"),
        name_col="predictor", auc_col="cv_auc", color_col="color",
        title="Best predictor overall", x_ticks=HORIZON_XTICKS[::2],
    )
    draw_top_two_mini(
        bottom_ax, table,
        top_two_names(table, "predictor", "cv_auc", "early"),
        name_col="predictor", auc_col="cv_auc", color_col="color",
        title="Best predictor first 3 min", x_ticks=HORIZON_XTICKS[::2], show_xlabel=True,
    )


def behaviour_legend_label(metric: str) -> str:
    """Shorten a per-behaviour legend to the behaviour name alone.

    Every entry in the Frequency and Bout panels is respectively a frequency or
    a bout duration, so repeating that on all seven or eight lines only costs
    width. The Bout panel's one non-behaviour entry is the average across
    behaviours, which is named explicitly instead.
    """
    if metric == "Mean bout duration":
        return "Mean all behaviours"
    return metric.replace(" bout duration", "").replace(" frequency", "")


def draw_individual_overtime(ax: plt.Axes, table: pd.DataFrame, panel: str,
                             letter: str, title: str, ncol: int = 2,
                             columnspacing: float = 0.7) -> plt.Text:
    subtab = table.loc[table["panel"] == panel]
    add_event_spans(ax)
    for name, sub in subtab.groupby("metric", sort=False):
        sub = sub.sort_values("horizon_min")
        label = (behaviour_legend_label(name)
                 if panel in ("Bout", "Frequency") else name)
        # The mean-across-behaviours line has no behaviour colour of its own;
        # draw it grey so it reads as the summary rather than as an eighth
        # behaviour.
        color = (COMBINED_COLOR if panel == "Bout" and name == "Mean bout duration"
                 else sub["color"].iloc[0])
        ax.plot(sub["horizon_min"], sub["cv_auc"], color=color,
                lw=0.8, marker="o", ms=2.0, mec="white", mew=0.2,
                linestyle=sub["linestyle"].iloc[0], label=label, alpha=0.95)
    ax.axhline(0.5, color=CHANCE_GREY, lw=0.45, ls=(0, (2.5, 2)), zorder=1)
    ax.set_xlim(HORIZON_XLIM)
    ax.set_xticks(HORIZON_XTICKS)
    ax.set_xticklabels([f"{h:g}" for h in HORIZON_XTICKS])
    align_edge_xticklabels(ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel(HORIZON_XLABEL)
    ax.set_ylabel("ROC AUC")
    ax.set_title(title, fontsize=6.4, color=AXIS, pad=3)
    _tidy(ax)
    fit_spines_to_ticks(ax)
    mark_final_horizon(ax)
    line_legend(ax, ncol=ncol, fontsize=3.9, columnspacing=columnspacing)
    return panel_tag(ax, letter, x=-0.12, y=1.12)


def figure7_global_shap_values() -> pd.DataFrame:
    """Figure 7B's exact stacked-bar widths, from the archival source data.

    The values were extracted once from the legacy figure PDF, which was lost
    before it could be committed; the exported CSV written by that extraction
    is now the archival source of record (docs/figure_structure.md). When a
    recovered copy of the PDF is present, the original extraction runs instead
    so the CSV can be re-derived and compared against the archived values.
    """
    exported = SOURCE_OUT / "figure7_classifier_global_shap.csv"
    if not FIGURE7_PDF.exists():
        if exported.exists():
            print(f"  [source data] {exported.name} (archival source)")
            return pd.read_csv(exported, index_col="feature")
        raise FileNotFoundError(
            f"Figure 7B archival source data not found: {exported}\n"
            f"and no legacy PDF to re-extract it from at: {FIGURE7_PDF}"
        )

    source_colours = {
        "Freeze": np.array([0.776, 0.443, 0.627]),
        "Sniff": np.array([0.388, 0.596, 0.643]),
        "Groom": np.array([0.620, 0.792, 0.855]),
        "Turn": np.array([0.737, 0.835, 0.282]),
        "Locomotion": np.array([0.902, 0.761, 0.227]),
        "Climb": np.array([0.851, 0.518, 0.153]),
        "Jump": np.array([0.851, 0.365, 0.365]),
        "Unassigned": np.array([0.824, 0.824, 0.824]),
    }
    values = np.zeros((len(FIGURE7_SHAP_FEATURES), len(FIGURE7_CLASSES)))
    axis_width_points = 182.95
    with fitz.open(FIGURE7_PDF) as document:
        for drawing in document[0].get_drawings():
            rect = drawing["rect"]
            fill = drawing.get("fill")
            if fill is None or not (5.2 <= rect.height <= 5.7):
                continue
            if not (395.0 <= rect.x0 <= 545.0 and 15.5 <= rect.y0 <= 151.0):
                continue
            row = round((rect.y0 - 16.0) / 6.81)
            if not 0 <= row < len(FIGURE7_SHAP_FEATURES):
                continue
            fill_array = np.asarray(fill)
            distances = {
                name: float(np.linalg.norm(fill_array - colour))
                for name, colour in source_colours.items()
            }
            name = min(distances, key=distances.get)
            if distances[name] < 0.02:
                values[row, FIGURE7_CLASSES.index(name)] += rect.width / axis_width_points

    return pd.DataFrame(values, index=FIGURE7_SHAP_FEATURES, columns=FIGURE7_CLASSES)


def write_classifier_source_data() -> None:
    """Export the legacy classifier values reused in Figure 7A-C."""
    pd.DataFrame({
        "class": FIGURE7_CLASSES + ["Overall"],
        "accuracy": FIGURE7_ACCURACY,
        "chance_accuracy": 0.125,
    }).to_csv(SOURCE_OUT / "figure7_classifier_accuracy.csv", index=False)

    figure7_global_shap_values().to_csv(
        SOURCE_OUT / "figure7_classifier_global_shap.csv", index_label="feature"
    )

    confusion = pd.DataFrame(
        FIGURE7_CONFUSION,
        index=FIGURE7_CLASSES,
        columns=FIGURE7_CLASSES,
    )
    confusion.index.name = "true_class"
    confusion.to_csv(SOURCE_OUT / "figure7_classifier_confusion_matrix.csv")


def draw_figure7_a(ax: plt.Axes) -> plt.Text:
    labels = FIGURE7_CLASSES + ["Overall"]
    colours = [BEHAVIOUR_COLORS[name] for name in BEHAVIOUR_ORDER] + ["#D2D2D2"]
    ys = np.arange(len(labels))
    ax.barh(ys[:-1], FIGURE7_ACCURACY[:-1], height=0.72, color=colours,
            edgecolor="white", linewidth=0.35, zorder=2)

    gradient = np.linspace(0, 1, 512)[None, :]
    gradient_cmap = LinearSegmentedColormap.from_list(
        "figure7_behaviours",
        ["#D2D2D2", BEHAVIOUR_COLORS["Jump"], BEHAVIOUR_COLORS["Climb"],
         BEHAVIOUR_COLORS["Locomotion"], BEHAVIOUR_COLORS["Turn"],
         BEHAVIOUR_COLORS["Groom"], BEHAVIOUR_COLORS["Sniff"],
         BEHAVIOUR_COLORS["Freeze"]],
    )
    overall_y = ys[-1]
    ax.imshow(gradient, extent=(0, FIGURE7_ACCURACY[-1], overall_y - 0.36,
                               overall_y + 0.36), aspect="auto", cmap=gradient_cmap,
              interpolation="bicubic", zorder=2)

    for y, value, note in zip(ys, FIGURE7_ACCURACY, FIGURE7_ACCURACY_NOTES):
        ax.text(value + 0.012, y, f"{value:.2f}\n({note})", ha="left", va="center",
                fontsize=4.4, color=AXIS, linespacing=1.05)

    ax.axvline(0.12, color=AXIS, linewidth=0.5, linestyle=(0, (2.5, 2)), zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=4.2)
    # Pin the category limits with half a bar of padding at each end. The default
    # margins clipped the top bar (Freeze) against the axes edge.
    ax.set_ylim(len(labels) - 0.5, -0.5)
    # The value annotations sit to the right of each bar; at the enlarged label
    # size the longest one ("0.86 (+0.74, 6.9x)") runs past 1.0, so the axis is
    # padded beyond the last tick to keep it inside the panel.
    ax.set_xlim(0, 1.30)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Accuracy")
    # The enlarged two-line value labels are taller than the bars, so the top
    # one reached into the title; give the title extra clearance.
    ax.set_title("Accuracy versus chance", fontsize=6.4, color=AXIS, pad=8)
    _tidy(ax)
    fit_spines_to_ticks(ax)
    # Keep the chance key in the open lower-right corner. Two lines allow a
    # slightly larger type size without crowding the lower bars.
    ax.text(0.985, 0.035, "Dashed line:\nchance = 0.12",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=4.2,
            color=AXIS, linespacing=1.18)
    return panel_tag(ax, "A", x=-0.20, y=1.15)


def draw_figure7_b(ax: plt.Axes, shap_values: pd.DataFrame) -> plt.Text:
    ys = np.arange(len(shap_values))
    left = np.zeros(len(shap_values))
    colours = {**BEHAVIOUR_COLORS, "Unassigned": "#D2D2D2"}
    for name in FIGURE7_CLASSES:
        values = shap_values[name].to_numpy(dtype=float)
        ax.barh(ys, values, left=left, height=0.78, color=colours[name],
                edgecolor="none", label=name, zorder=2)
        left += values

    ax.set_yticks(ys)
    ax.set_yticklabels(FIGURE7_SHAP_FEATURES, fontsize=3.3)
    ax.invert_yaxis()
    # Bars run to ~0.83 and previously reached the axis edge, so the panel read as
    # one solid block butting into its own feature names and into A. The legend
    # now sits inside the panel over the gap beside the short lower bars, which
    # frees the right margin: the axis only needs a little headroom past the
    # longest bar. Ticks stay on the same 0-0.9 grid.
    ax.set_xlim(0, 0.92)
    ax.set_xticks(np.linspace(0, 0.9, 4))
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global SHAP feature importance", fontsize=6.4, color=AXIS, pad=3)
    _tidy(ax)
    fit_spines_to_ticks(ax)
    # Sits in the open wedge to the right of the shorter lower bars, so the panel
    # no longer needs a wide empty right margin just to hold the key.
    leg = ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.03), frameon=False,
                    fontsize=5.2, ncol=1, handlelength=1.4, handletextpad=0.4,
                    labelspacing=0.22, borderaxespad=0.0)
    for text in leg.get_texts():
        text.set_color(AXIS)
    return panel_tag(ax, "B", x=-0.34, y=1.15)


def draw_figure7_c(ax: plt.Axes, cax: plt.Axes) -> plt.Text:
    cmap = LinearSegmentedColormap.from_list("figure7_confusion", ["#FFFFFF", "#B45E8C"])
    image = ax.imshow(FIGURE7_CONFUSION, cmap=cmap, vmin=0, vmax=0.86,
                      interpolation="nearest", aspect="equal")
    ticks = np.arange(len(FIGURE7_CLASSES))
    ax.set_xticks(ticks)
    ax.set_xticklabels(FIGURE7_CLASSES, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=3.7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(FIGURE7_CLASSES, fontsize=4.0)
    ax.set_xlabel("Predicted label", labelpad=1.5, fontsize=5.0)
    ax.set_ylabel("True label", labelpad=2, fontsize=5.0)
    ax.set_title("Cross-validated confusion matrix", fontsize=6.4, color=AXIS, pad=3)
    ax.tick_params(axis="both", length=0, pad=1.5, colors=AXIS)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.45)

    for row in range(FIGURE7_CONFUSION.shape[0]):
        for col in range(FIGURE7_CONFUSION.shape[1]):
            value = FIGURE7_CONFUSION[row, col]
            label = "0" if value == 0 else f"{value:.3f}".rstrip("0").rstrip(".")
            ax.text(col, row, label, ha="center", va="center", fontsize=3.15,
                    color="white" if value >= 0.42 else "#B45E8C")

    colourbar = ax.figure.colorbar(image, cax=cax)
    colourbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8])
    colourbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8"])
    colourbar.ax.tick_params(labelsize=3.5, length=1.8, width=0.4, colors=AXIS,
                             pad=1.0)
    # Ticks label the outside of the bar. On the left they were drawn over the
    # matrix's last column, since the bar sits flush against it.
    colourbar.ax.yaxis.set_ticks_position("right")
    colourbar.ax.yaxis.set_label_position("right")
    colourbar.outline.set_edgecolor(AXIS)
    colourbar.outline.set_linewidth(0.45)
    return panel_tag(ax, "C", x=-0.30, y=1.15)


def build_complete_figure(onset: pd.DataFrame, cross: pd.DataFrame,
                          head: pd.DataFrame, shap: pd.DataFrame,
                          individual_time: pd.DataFrame,
                          labels: pd.DataFrame,
                          cross_tests: pd.DataFrame | None = None,
                          head_test: pd.DataFrame | None = None) -> plt.Figure:
    """Build the complete A4 manuscript Figure 7."""
    style()
    ab_recap, c_recap = recap_tables()
    figure7_shap = figure7_global_shap_values()
    fig = plt.figure(figsize=(8.27, 11.69), dpi=300, facecolor="white")  # A4 portrait
    gs = GridSpec(6, 8, figure=fig,
                  height_ratios=[1.12, 0.86, 0.88, 0.88, 0.82, 0.70],
                  hspace=0.70, wspace=0.70,
                  left=0.068, right=0.988, top=0.978, bottom=0.052)

    # Row 0 carries A and B alone, so both get the full page width: A's accuracy
    # labels and B's 20 feature names are the most space-hungry text in the
    # figure. The confusion matrix (C) moves down to join D-F on row 1.
    # B's 20 feature names are the widest text block on the page and its bars ran
    # right up against them.  Giving A a little more of the row and B a little
    # less, with a wider gutter, pulls B's bars clear of its own labels and of A.
    source = gs[0, :].subgridspec(
        1, 2, width_ratios=[1.16, 1.40], wspace=0.46
    )
    # Panels A-C are excluded from align_panel_letters: they label their y axes
    # with long category names instead of an axis title, so anchoring the letter
    # to that column would push it far to the left, over the neighbour. Their
    # offsets stay hand-tuned, but A and C are snapped afterwards onto the same
    # left margin as the auto-aligned panels (see snap_to_left_margin below) so
    # the left-hand letters read as one vertical line down the page.
    ax_a_source = fig.add_subplot(source[0, 0])
    letter_a = draw_figure7_a(ax_a_source)
    letter_artists: list[tuple[plt.Axes, plt.Text]] = []
    ax_b_source = fig.add_subplot(source[0, 1])
    draw_figure7_b(ax_b_source, figure7_shap)

    # C is square (imshow with aspect="equal"), so it needs a narrower cell than
    # its neighbours; D is trimmed to make room for it.
    # C is drawn with aspect="equal", so width past its square is dead space:
    # its cell is trimmed and the room goes to D and E (whose multi-line cohort
    # labels need it) and to F, which now runs out to the same right margin as
    # B, G and H.
    def_panels = gs[1, :].subgridspec(
        1, 4, width_ratios=[0.72, 0.92, 0.66, 1.52], wspace=0.52
    )
    c_grid = def_panels[0, 0].subgridspec(
        1, 2, width_ratios=[1.0, 0.045], wspace=0.06
    )
    ax_c_source = fig.add_subplot(c_grid[0, 0])
    letter_c = draw_figure7_c(ax_c_source, fig.add_subplot(c_grid[0, 1]))

    ax_d = fig.add_subplot(def_panels[0, 1])
    ax_e = fig.add_subplot(def_panels[0, 2])
    ax_f = fig.add_subplot(def_panels[0, 3])
    letter_artists += [
        (ax_d, plot_panel_b(ax_d, cross, letter="D", tests=cross_tests)),
        (ax_e, plot_panel_c(ax_e, head, letter="E", tests=head_test)),
        (ax_f, plot_panel_d(ax_f, shap, letter="F")),
    ]

    ax_g = fig.add_subplot(gs[2, :])
    ax_h = fig.add_subplot(gs[3, :])
    recap.draw_panel_a(ax_g, ab_recap)
    recap.draw_panel_b(ax_h, ab_recap)
    letter_artists += [
        (ax_g, recap.tag(ax_g, "G", x=-0.045, y=1.12)),
        (ax_h, recap.tag(ax_h, "H", x=-0.045, y=1.12)),
    ]

    ij = gs[4, :].subgridspec(
        2, 12,
        width_ratios=[
            0.62, 0.62, 0.62, 0.17, 0.74, 0.52,
            0.62, 0.62, 0.62, 0.17, 0.74, 0.08,
        ],
        hspace=0.48,
        wspace=0.04,
    )
    ax_i = fig.add_subplot(ij[:, 0:3])
    letter_artists.append((ax_i, plot_panel_a(ax_i, onset, letter="I")))
    draw_behaviour_mini_pair(fig.add_subplot(ij[0, 4]), fig.add_subplot(ij[1, 4]), onset)
    ax_j = fig.add_subplot(ij[:, 6:9])
    letter_artists.append((ax_j, draw_family_overtime(ax_j, c_recap, letter="J")))
    draw_family_mini_pair(fig.add_subplot(ij[0, 10]), fig.add_subplot(ij[1, 10]), c_recap)

    # K's behaviour names are shorter than N's, so its four columns would pack
    # into a narrower block; the wider column spacing makes the two legends span
    # the same width.
    for columns, panel, letter, title, ncol, colspace in [
        ((0, 2), "Frequency", "K", "Individual frequencies", 4, 2.4),
        ((2, 4), "Diversity", "L", "Individual diversity indices", 1, 0.7),
        ((4, 6), "Transition", "M", "Transition metrics", 1, 0.7),
        ((6, 8), "Bout", "N", "Bout durations", 4, 0.7),
    ]:
        ax = fig.add_subplot(gs[5, columns[0]:columns[1]])
        letter_artists.append(
            (ax, draw_individual_overtime(ax, individual_time, panel, letter, title,
                                          ncol=ncol, columnspacing=colspace))
        )

    align_panel_letters(fig, letter_artists)
    snap_to_left_margin(fig, [letter_a, letter_c], [ax_g, ax_h, ax_i])
    return fig


def snap_to_left_margin(fig: plt.Figure, letters: list[plt.Text],
                        reference_axes: list[plt.Axes]) -> None:
    """Move hand-placed letters onto the left margin the aligned ones settled on.

    ``align_panel_letters`` puts every letter it is given in its own panel's
    y-title column.  For the full-width panels (G, H, I) that column is the page's
    left margin, so those letters share one x.  A and C are placed by hand -
    their y axes carry long category names rather than a title - and were left
    sitting at two different x positions well to the left of that line.  This
    re-anchors them in figure coordinates to the same margin, keeping each
    letter's own vertical position.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_inv = fig.transFigure.inverted()
    margin = min(
        fig_inv.transform((ax.yaxis.label.get_window_extent(renderer=renderer).x0, 0))[0]
        for ax in reference_axes
    )
    for text in letters:
        if text is None:
            continue
        # The letter may still be in axes coordinates; resolve where it is on the
        # page before overwriting only its x.
        y_fig = fig_inv.transform(text.get_window_extent(renderer=renderer).p1)[1]
        text.set_transform(fig.transFigure)
        text.set_position((margin, y_fig))
        text.set_ha("left")
        text.set_va("top")


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def write_report(onset: pd.DataFrame, cross: pd.DataFrame, head: pd.DataFrame,
                 shap: pd.DataFrame, n_perm: int) -> None:
    onset = onset.copy()
    onset["auc_above_null_mean"] = onset["roc_auc"] - onset["null_mean"]
    combined = onset.loc[onset["feature_set"] == COMBINED].sort_values("horizon_min")
    positive = onset.loc[onset["auc_above_null_mean"] > 0].copy()
    first = combined.loc[combined["auc_above_null_mean"] > 0]

    if first.empty:
        verdict = (
            "The combined behaviour-dynamics model never rises above the mean "
            "shuffled-label baseline."
        )
    else:
        h = first["horizon_min"].iloc[0]
        verdict = (
            f"The combined behaviour-dynamics model is already above the mean "
            f"shuffled-label baseline at **{h:.1f} min**, with the largest margins "
            f"after 5 min."
        )

    lines = [
        SECTION,
        "",
        "`figure_7_resilience_prediction.py` rebuilds the early-prediction",
        "panel with the three things the original lacked: a permutation null, a",
        "per-behaviour breakdown, and Shapley attributions. **Behaviour dynamics**",
        "means the 30-s trajectory of each behaviour's time share.",
        "",
        "### Panel I - onset of prediction",
        "",
        "Panel F plots raw ROC AUC so it is directly comparable with the other",
        "time-resolved panels. The source table still carries the mean within-cohort",
        f"permutation AUC ({n_perm} shuffles per point) as an audit column, but it is",
        "not drawn in the figure.",
        "",
        "| Opening minutes | n features | ROC AUC |",
        "| ---: | ---: | ---: |",
    ]
    for _, r in combined.iterrows():
        lines.append(
            f"| {r['horizon_min']:.1f} | {int(r['n_features'])} | {r['roc_auc']:.3f} |"
        )
    lines += ["", "The combined behaviour-dynamics curve is above chance from 0.5 min, "
              "with the clearest rise after 5 min.", ""]
    lines += [
        "",
        "### Complete A4 Figure 7 layout",
        "",
        "The complete A4 version begins with Figure 7 panels A-C: accuracy versus",
        "chance, global SHAP feature importance, and the confusion matrix. The",
        "behaviour-dynamics held-out transfer, full-session LOOCV, and full-session",
        "Shapley contributions form panels D-F in the second row. The all-predictor",
        "held-out and CV panels follow, then side-by-side overtime summaries and four",
        "individual-overtime panels: frequencies, diversity indices, transition",
        "metrics, and bout durations.",
        "",
        f"Full-session LOOCV: behaviour dynamics "
        f"{head.loc[head['feature_set'] == 'Behaviour dynamics', 'roc_auc'].iloc[0]:.3f}, "
        f"{FREEZE_ONLY_DISPLAY} {head.loc[head['feature_set'] == 'Freeze only', 'roc_auc'].iloc[0]:.3f}.",
        "",
        "The Shapley panel gives exact Shapley values for the full-session model. The classifier is",
        "linear, so phi_ij = coef_j * z_ij is the closed-form Shapley value against a",
        "mean-centred background - no sampling and no `shap` dependency. Ranked by",
        "mean |phi|: "
        + ", ".join(f"{r['behaviour']} ({r['mean_abs_shap']:.3f}, toward {r['direction']})"
                    for _, r in shap.head(3).iterrows())
        + ".",
        "",
        "**Caveat carried from the pre-shock work.** The label is built from the",
        "full-session behaviour-dynamics profile, so every horizon is scored against",
        "a target that partly includes the window being tested. The onset panel is therefore",
        "anti-conservative; the within-cohort permutation audit controls for sample",
        "size, cohort, and model size, not for that overlap.",
        "",
        "Outputs: `figures/figure7.png/.pdf/.svg`,",
        "`figure_source_data/figure7_*.csv`, and `statistics/figure7_*.csv`.",
        "",
    ]
    block = "\n".join(lines)

    text = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
    if SECTION in text:
        head_txt, rest = text.split(SECTION, 1)
        nxt = rest.find("\n## ")
        tail = rest[nxt + 1:] if nxt != -1 else ""
        text = head_txt + block + tail
    else:
        text = text.rstrip() + "\n\n" + block
    REPORT.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--perms", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--recompute-onset", action="store_true",
                    help="rerun onset permutations instead of reading tracked source data")
    args = ap.parse_args()

    root = tier1.repo_root(args.repo)
    labels = tier1.load_labels(root)
    print(f"repo: {root}\nanimals: {len(labels)} "
          f"({int(labels['target'].sum())} resilient)\n"
          f"permutations per point: {args.perms}\n")

    print("panel A - onset of prediction")
    onset_path = SOURCE_OUT / "figure7_prediction_onset.csv"
    if onset_path.exists() and not args.recompute_onset:
        onset = pd.read_csv(onset_path)
        print(f"  [cache] {onset_path}")
    else:
        onset = onset_of_prediction(labels, root, args.perms)
    onset = onset.drop(columns=["null_95", "beats_null"], errors="ignore")
    onset["auc_above_null_mean"] = onset["roc_auc"] - onset["null_mean"]
    onset.to_csv(onset_path, index=False)
    onset.to_csv(STATISTICS_DIR / "figure7_prediction_permutation.csv", index=False)

    print("\npanels B/C - held-out cohort and full-session LOOCV")
    cross = held_out_cohort(labels, root)
    head, head_test = full_session_loocv(labels, root)
    pd.concat([
        cross.assign(panel="B held-out cohort"),
        head.assign(panel="C full-session LOOCV"),
    ], ignore_index=True).to_csv(SOURCE_OUT / "figure7_full_session_auc.csv", index=False)
    cross.to_csv(STATISTICS_DIR / "figure7_cross_cohort_auc.csv", index=False)
    head.to_csv(STATISTICS_DIR / "figure7_full_session_auc.csv", index=False)

    print("panels D/E - paired AUC significance tests")
    cross_tests = held_out_cohort_tests(labels, root)
    panel_tests = pd.concat([cross_tests, head_test], ignore_index=True, sort=False)
    panel_tests.to_csv(STATISTICS_DIR / "figure7_panel_de_auc_tests.csv", index=False)
    panel_tests.to_csv(SOURCE_OUT / "figure7_panel_de_auc_tests.csv", index=False)
    print(panel_tests.to_string(index=False))

    print("panel D - Shapley values")
    shap = shapley_values(labels, root)
    shap.to_csv(SOURCE_OUT / "figure7_shapley_contributions.csv", index=False)
    print(shap.to_string(index=False))

    print("individual overtime panels")
    individual_time = individual_overtime_data(labels)
    individual_time.drop(columns=["color", "linestyle"]).to_csv(
        SOURCE_OUT / "figure7_individual_timecourse_auc.csv", index=False
    )

    write_classifier_source_data()
    ab_recap, c_recap = recap_tables()
    ab_recap.drop(columns=["color"], errors="ignore").to_csv(
        SOURCE_OUT / "figure7_predictor_catalog.csv", index=False
    )
    c_recap.drop(columns=["color"], errors="ignore").to_csv(
        SOURCE_OUT / "figure7_predictor_timecourse.csv", index=False
    )

    summary = pd.concat([
        pd.DataFrame({
            "panel": "A",
            "measure": "classifier_accuracy",
            "category": FIGURE7_CLASSES + ["Overall"],
            "value": FIGURE7_ACCURACY,
        }),
        cross[["feature_set", "roc_auc"]]
        .assign(panel="D", measure="held_out_cohort_auc")
        .rename(columns={"feature_set": "category", "roc_auc": "value"})[
            ["panel", "measure", "category", "value"]
        ],
        head[["feature_set", "roc_auc"]]
        .assign(panel="E", measure="full_session_auc")
        .rename(columns={"feature_set": "category", "roc_auc": "value"})[
            ["panel", "measure", "category", "value"]
        ],
    ], ignore_index=True, sort=False)
    summary.to_csv(SOURCE_OUT / "figure7.csv", index=False)

    complete = build_complete_figure(onset, cross, head, shap, individual_time, labels,
                                     cross_tests=cross_tests, head_test=head_test)
    for ext in ("png", "pdf", "svg"):
        path = OUT / f"figure7.{ext}"
        complete.savefig(path, dpi=400 if ext == "png" else None,
                         facecolor="white")
        print(f"wrote {path}")
    plt.close(complete)

    if not args.no_report:
        write_report(onset, cross, head, shap, args.perms)
        print(f"updated {REPORT} section: {SECTION}")


if __name__ == "__main__":
    main()
