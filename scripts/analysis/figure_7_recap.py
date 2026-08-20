# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""Predictor-catalog analyses used by manuscript Figure 7.

Same three-panel grammar as tier1_reanalysis_summary.pdf, with the two-series
comparison (motif repertoire vs freeze only) replaced by the full predictor
catalogue:

  A  Held-out cohort   - train one cohort, test the other, both directions
  B  Full-session CV   - all 41 animals, internal cross-validation with CIs
  C  Early prediction  - AUC against how many minutes of session are used

Predictor catalogue
-------------------
  1. Each individual behaviour time share                   (1 feature each)
  2. All behaviour time shares                              (7 features)
  3. Transition summaries: Lempel-Ziv, recurrence,
     determinism, Markov entropy                            (4 features)
  4. Diversity: Simpson, Shannon, evenness, CUI             (4 features)
  5. Bout durations                                         (8 features)
  6. Integrated block: all metrics                          (23 features)

Dynamics score and MDS position are excluded: the resilient/vulnerable label is
defined from the dynamics score, so using them as predictors would be circular.

Supervised Freeze % is excluded as well, for a different reason: on a matched
denominator it correlates r = 0.98 with motif Freeze frequency and returns the
same AUC, so carrying both would double-count one measurement rather than add
independent evidence. See run_freeze_denominator_alignment.py.

Two deliberate departures from the original summary figure
----------------------------------------------------------
1. Panel B uses repeated stratified 5-fold CV, not LOOCV. At n=41 with a weak
   feature LOOCV anti-learns - leaving one animal out shifts the fitted
   boundary away from it, so AUC collapses toward 0 (early Freeze reached 0.03
   despite group means of 1.3% vs 1.9%). The sub-chance values in the original
   panel C are artefacts of that resampling scheme.
2. Panel C features are CUMULATIVE metrics recomputed on the truncated
   sequence, so every horizon uses the same number of features. The original
   used one column per (motif, 30-s bin), growing 7 -> 105 features across the
   x-axis, which confounded "more time" with "bigger model".

Panel C metrics are recomputed from data/raw/syllable_usage_per_timebin_250ms.csv
using the same construction as manuscript Figure 4 (full mapped sequence,
truncated at each horizon), so the final horizon reproduces the published
full-session values.

Run through ``scripts/generate_figures/figure_7_resilience_prediction.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figure_source_data"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis import figure_7_feature_families as efa  # noqa: E402
from scripts.analysis import figure_7_modeling as direct  # noqa: E402

from src.statistics import (  # noqa: E402
    compute_diversity_metrics,
    compute_bout_duration,
    transition_sequence_metrics,
)

RANDOM_SEED = direct.RANDOM_SEED
# Sampled every 30 s to match the behaviour-dynamics time courses in
# Figures 3, 5 and Supplementary 1 (0.5-7.5 min, 15 points).
HORIZONS_MIN = [round(0.5 * k, 1) for k in range(1, 16)]
CACHE_DIR = REPO / ".cache" / "figure7"

# --------------------------------------------------------------------------
# manuscript style (scripts/generate_figures/figure_4_diversity_dynamics.py)
# --------------------------------------------------------------------------
plt.rcParams["axes.grid"] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

AXIS = "#4D4D4D"
CHANCE_GREY = "#9A9A9A"

# Behaviour colours, identical to manuscript Figure 4.
BEHAVIOR_COLORS = {
    "Freeze": "#C37BA0",
    "Sniff": "#5B9AAA",
    "Groom": "#8EC6DE",
    "Turn": "#B7DB45",
    "Locomotion": "#F1C232",
    "Climb": "#F4A259",
    "Jump": "#E45756",
}
BLOCK_COLORS = {
    "All behaviour time shares": "#5B9AAA",
    "Diversity metrics": "#F1C232",
    "Transition summaries": "#8EC6DE",
    "Bout durations": "#F4A259",
    "All metrics": "#B7DB45",
}

# Terminology follows the manuscript: "frequency" (cluster_frequency_per_animal.csv,
# Fig. 4 usage panels), the diversity indices as named in Fig. 4E-H, the
# transition metrics as named in Fig. 4T-W, and bout duration as in Fig. 4J-Q.
FAMILY_COLORS = {
    "freezing": "#C37BA0",
    "frequency": "#5B9AAA",
    "diversity": "#F1C232",
    "transition": "#8EC6DE",
    "bout": "#F4A259",
    "combined": "#B7DB45",
}

# (display name, source column, family) in figure order. Blocks are inserted
# after the family they summarise, see build_predictors.
SINGLE_PARAMETERS = [
    # Supervised Freeze % is deliberately excluded. Once put on the same
    # denominator as the motif measure it correlates with it at r = 0.98 and
    # gives the same AUC (0.783 worse-direction either way) - it is the same
    # measurement via a different detector, not independent evidence. See
    # run_freeze_denominator_alignment.py.
    ("Freeze frequency", "frequency__Freeze", "frequency"),
    ("Sniff frequency", "frequency__Sniff", "frequency"),
    ("Groom frequency", "frequency__Groom", "frequency"),
    ("Turn frequency", "frequency__Turn", "frequency"),
    ("Locomotion frequency", "frequency__Locomotion", "frequency"),
    ("Climb frequency", "frequency__Climb", "frequency"),
    ("Jump frequency", "frequency__Jump", "frequency"),
    ("Simpson index", "diversity__simpson", "diversity"),
    ("Shannon entropy index", "diversity__shannon", "diversity"),
    ("Evenness index", "diversity__evenness", "diversity"),
    ("Cumulative usage index", "diversity__cui", "diversity"),
    ("Lempel-Ziv complexity", "transition__lz", "transition"),
    ("Recurrence rate", "transition__recurrence", "transition"),
    ("Determinism", "transition__determinism", "transition"),
    ("Markov entropy", "transition__markov", "transition"),
    ("Mean bout duration", "bout__overall_mean", "bout"),
    # The stored bout table uses the long cluster names.
    ("Freeze bout duration", "bout__Freezing", "bout"),
    ("Sniff bout duration", "bout__Sniffing", "bout"),
    ("Groom bout duration", "bout__Grooming", "bout"),
    ("Turn bout duration", "bout__Turn", "bout"),
    ("Locomotion bout duration", "bout__Locomotion", "bout"),
    ("Climb bout duration", "bout__Climbing", "bout"),
    ("Jump bout duration", "bout__Jump", "bout"),
]


def predictor_color(name: str) -> str:
    """Return the stable manuscript color for a predictor label."""
    if name in BLOCK_COLORS:
        return BLOCK_COLORS[name]
    block_families = {
        "All frequencies": "frequency",
        "All diversity indices": "diversity",
        "All transition metrics": "transition",
        "All bout durations": "bout",
        "All parameters": "combined",
    }
    if name in block_families:
        return FAMILY_COLORS[block_families[name]]
    for display, _, family in SINGLE_PARAMETERS:
        if display == name:
            return FAMILY_COLORS[family]
    raise KeyError(f"Unknown Figure 7 predictor: {name}")

# efa.SYLLABLE_TO_CLUSTER maps to these short names, so the panel C
# recomputation uses them throughout. (The repo's stored bout table uses the
# long forms - "Freezing", "Sniffing", "Grooming", "Climbing" - but that table
# is only read for panels A and B, via efa.bout_features.)
BEHAVIORS = ["Freeze", "Sniff", "Groom", "Turn", "Locomotion", "Climb", "Jump"]


def darken(color: str, factor: float = 0.62) -> tuple[float, float, float]:
    """Same hue, lower luminance - used so each error bar matches its own bar."""
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def style_axis(ax: plt.Axes, labelsize: float = 5.2) -> None:
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
        ax.spines[spine].set_linewidth(0.45)
    ax.tick_params(axis="both", colors=AXIS, labelsize=labelsize, width=0.45, length=2)
    ax.xaxis.label.set_color(AXIS)
    ax.yaxis.label.set_color(AXIS)
    ax.title.set_color(AXIS)


def tag(ax: plt.Axes, letter: str, x: float, y: float = 1.10) -> plt.Text:
    return ax.text(x, y, letter, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, fontweight="bold", color=AXIS, clip_on=False)


def cached(name: str, builder, *, fresh: bool):
    path = CACHE_DIR / f"{name}.pkl"
    if not fresh and path.exists():
        print(f"  [cache] {name}", flush=True)
        return pd.read_pickle(path)
    t0 = time.time()
    obj = builder()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(obj, path)
    print(f"  [computed] {name} ({time.time() - t0:.1f}s)", flush=True)
    return obj


# --------------------------------------------------------------------------
# full-session predictor catalogue (panels A and B)
# --------------------------------------------------------------------------


def build_predictors():
    """Ordered predictor catalogue: name -> (frame, columns, colour, kind)."""
    labels = efa.load_labels()
    frequency, _ = efa.frequency_features()
    diversity, _ = efa.diversity_features()
    transition, _ = efa.transition_features()
    bout, _ = efa.bout_features()
    freezing, _ = efa.freezing_features()

    def cols_of(frame: pd.DataFrame) -> list[str]:
        return [c for c in frame.columns if "__" in c]

    all_metrics = efa.merge_feature_frames([frequency, diversity, transition, bout])
    all_plus_freeze = all_metrics.merge(freezing, on="animal_id", how="inner")

    pooled = all_plus_freeze
    available = set(cols_of(pooled))
    missing = [c for _, c, _ in SINGLE_PARAMETERS if c not in available]
    if missing:
        raise KeyError(
            f"catalogued parameter(s) absent from the feature frames: {missing}. "
            f"Available: {sorted(available)}"
        )

    # Blocks are placed immediately after the family they summarise, so the
    # x-axis reads family by family.
    blocks_after = {
        "frequency": ("All frequencies", frequency, cols_of(frequency), "frequency"),
        "diversity": ("All diversity indices", diversity, cols_of(diversity), "diversity"),
        "transition": ("All transition metrics", transition, cols_of(transition), "transition"),
        "bout": ("All bout durations", bout, cols_of(bout), "bout"),
    }

    predictors: dict[str, tuple] = {}
    for i, (name, col, family) in enumerate(SINGLE_PARAMETERS):
        predictors[name] = (pooled, [col], FAMILY_COLORS[family], "single")
        is_last_of_family = (
            i + 1 == len(SINGLE_PARAMETERS) or SINGLE_PARAMETERS[i + 1][2] != family
        )
        if is_last_of_family and family in blocks_after:
            bname, bframe, bcols, bfamily = blocks_after[family]
            predictors[bname] = (bframe, bcols, FAMILY_COLORS[bfamily], "block")

    predictors["All parameters"] = (
        all_metrics, cols_of(all_metrics), FAMILY_COLORS["combined"], "block",
    )
    return labels, predictors


def panel_ab_data(labels: pd.DataFrame, predictors: dict) -> pd.DataFrame:
    """Cross-cohort (panel A) and internal CV (panel B) for every predictor."""
    rows = []
    for name, (frame, cols, color, kind) in predictors.items():
        data = direct.attach(labels, frame)
        cc = direct.cross_cohort(data, cols)
        by_dir = {r["direction"]: r for r in cc}
        # Fold layout depends only on the stratification key, identical for all
        # predictors, so build it once and reuse.
        folds = direct.make_folds(
            direct.strat_key(data), direct.N_SPLITS, direct.N_REPEATS, RANDOM_SEED
        )
        y = data["target"].to_numpy(dtype=int)
        aucs, oof = direct.cv_auc_np(data[cols].to_numpy(dtype=float), y, folds)
        # The plotted point and its interval must be the SAME statistic. The
        # bootstrap resamples animals from the averaged out-of-fold score, so
        # the point estimate has to be the AUC of that same score - not the
        # mean of the per-repeat AUCs, which is a different quantity and can
        # fall outside the interval entirely for weak predictors.
        point = direct.fast_auc(y, oof)
        ci_lo, ci_hi = direct.stratified_bootstrap_ci(y, oof)
        kru = by_dir["Krugers -> Gómez"]
        gom = by_dir["Gómez -> Krugers"]
        rows.append(
            {
                "predictor": name,
                "kind": kind,
                "color": color,
                "n_features": len(cols),
                "auc_kru_to_gom": kru["roc_auc"],
                "ci_kru_lo": kru["auc_ci_low"], "ci_kru_hi": kru["auc_ci_high"],
                "auc_gom_to_kru": gom["roc_auc"],
                "ci_gom_lo": gom["auc_ci_low"], "ci_gom_hi": gom["auc_ci_high"],
                "worst_direction_auc": min(r["roc_auc"] for r in cc),
                "cv_auc": point,
                "cv_ci_low": ci_lo,
                "cv_ci_high": ci_hi,
                # Kept for transparency: the spread across CV partitions is
                # fold noise only, and is much narrower than the animal-level
                # sampling uncertainty the bootstrap captures.
                "cv_auc_repeat_mean": float(aucs.mean()),
                "cv_auc_repeat_sd": float(aucs.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# time-resolved predictors (panel C)
# --------------------------------------------------------------------------


def _timecourse() -> pd.DataFrame:
    return pd.read_csv(REPO / "data/processed/cluster_timecourse_per_animal.csv")


def cumulative_frequency(sub: pd.DataFrame) -> pd.DataFrame:
    """Time-weighted cumulative frequency per behaviour, from the 250 ms table.

    Two wrong ways to build this, both of which were tried:

      * counting rows of the 250 ms table - that table holds one row per
        syllable present in a bin, so a count weights a bin by how many
        syllables it contains, giving an occurrence count, not a frequency;
      * averaging the 30-s timecourse `pct` across bins - an unweighted bin
        mean, which disagrees with the published frequency by up to 25
        percentage points for Turn because bins differ in how much mapped data
        they hold.

    The correct quantity uses `Percentage`, the share of each 250 ms bin
    occupied by that syllable, converted to seconds and normalised over the
    seven named behaviours - identical to how efa.frequency_features derives
    frequency from `frequency_seconds`.
    """
    sub = sub.loc[sub["cluster"].isin(BEHAVIORS)].copy()
    sub["seconds"] = sub["Percentage"] / 100.0 * 0.25
    wide = sub.pivot_table(index="Animal", columns="cluster", values="seconds",
                           aggfunc="sum", fill_value=0)
    totals = wide.sum(axis=1).replace(0, np.nan)
    wide = wide.div(totals, axis=0).mul(100.0)
    wide = wide.reindex(columns=BEHAVIORS, fill_value=0.0).sort_index(axis=1).reset_index()
    wide = wide.rename(columns={"Animal": "animal_id"})
    wide["animal_id"] = wide["animal_id"].astype(float)
    return wide.rename(columns={c: f"frequency__{c}" for c in wide.columns if c != "animal_id"})


def _raw_250ms() -> pd.DataFrame:
    """Full mapped 250 ms sequence table, exactly as manuscript Figure 4 builds it."""
    raw = pd.read_csv(REPO / "data/raw/syllable_usage_per_timebin_250ms.csv")
    raw = raw.rename(columns={"Time Bin": "time_bin", "Condition": "group"})
    raw = raw[raw["group"].isin(["Control", "ELS"])].copy()
    raw["Syllable"] = pd.to_numeric(raw["Syllable"], errors="coerce").astype(int)
    raw["cluster"] = raw["Syllable"].map(efa.SYLLABLE_TO_CLUSTER).fillna("")
    return raw


def horizon_features(raw: pd.DataFrame, horizon_min: float) -> dict[str, pd.DataFrame]:
    """Recompute every metric family on the sequence truncated at `horizon_min`.

    Figure 4 computes diversity, bout and transition metrics on the full mapped
    sequence (not the predominant-per-bin one), so the same construction is used
    here. Truncating the sequence rather than adding one column per time bin
    keeps the feature count constant across horizons.
    """
    sub = raw.loc[raw["time_bin"] <= horizon_min * 60]
    div_rows, trans_rows, bout_rows = [], [], []
    for animal, group in sub.groupby("Animal", sort=False):
        # Unmapped syllables map to "" and are KEPT in the sequence: Figure 4
        # computes diversity and transition metrics on the full mapped sequence
        # including them, so dropping them here would not reproduce the
        # published full-session values.
        seq = group["cluster"].tolist()
        if not seq:
            continue
        animal_id = float(animal)

        div = compute_diversity_metrics(seq)
        div_rows.append(
            {"animal_id": animal_id,
             "diversity__simpson": div["simpson_index"],
             "diversity__shannon": div["shannon_entropy_index"],
             "diversity__evenness": div["evenness_index"],
             "diversity__cui": div["cumulative_usage_index"]}
        )

        tm = transition_sequence_metrics(seq)
        trans_rows.append(
            {"animal_id": animal_id,
             "transition__lz": tm["lempel_ziv_complexity"],
             "transition__recurrence": tm["recurrence_rate"],
             "transition__determinism": tm["determinism"],
             "transition__markov": tm["markov_entropy"]}
        )

        # compute_bout_duration returns one row PER BOUT (column
        # bout_duration_seconds), so the per-cluster value is a groupby mean.
        bouts = compute_bout_duration(seq)
        if len(bouts):
            bouts = bouts[bouts["cluster"].isin(BEHAVIORS)]
        per_cluster = (
            bouts.groupby("cluster")["bout_duration_seconds"].mean()
            if len(bouts) else pd.Series(dtype=float)
        )
        row = {
            "animal_id": animal_id,
            "bout__overall_mean": float(bouts["bout_duration_seconds"].mean()) if len(bouts) else 0.0,
        }
        for b in BEHAVIORS:
            row[f"bout__{b}"] = float(per_cluster.get(b, 0.0))
        bout_rows.append(row)

    frames = {
        # Time-weighted from the 250 ms Percentage column - see cumulative_frequency.
        "frequency": cumulative_frequency(sub),
        "diversity": pd.DataFrame(div_rows).fillna(0.0),
        "transition": pd.DataFrame(trans_rows).fillna(0.0),
        "bout": pd.DataFrame(bout_rows).fillna(0.0),
        "freezing": cumulative_freeze(horizon_min),
    }
    return frames


def cumulative_freeze(horizon_min: float) -> pd.DataFrame:
    frame = direct.cumulative_freeze_features(horizon_min)
    return frame.rename(columns={"cumfreeze__supervised_pct": "freezing__supervised_pct"})


def validate_final_horizon(frames: dict[str, pd.DataFrame]) -> None:
    """Check the recomputation reproduces the published full-session metrics.

    Panel C rests on the assumption that truncating the sequence and recomputing
    is the same operation Figure 4 performed, just on less data. At the final
    horizon there is no truncation, so the two must agree.
    """
    print("  validation at full session (published vs recomputed):", flush=True)
    checks = [
        ("diversity", efa.diversity_features()[0], ["simpson", "shannon", "evenness", "cui"]),
        # Frequency is the family that was previously built the wrong way, so
        # it is explicitly covered here now.
        ("frequency", efa.frequency_features()[0], BEHAVIORS),
    ]
    for family, published, metrics in checks:
        merged = published.merge(frames[family], on="animal_id", suffixes=("_pub", "_new"))
        for metric in metrics:
            a = merged[f"{family}__{metric}_pub"].to_numpy(dtype=float)
            b = merged[f"{family}__{metric}_new"].to_numpy(dtype=float)
            r = float(np.corrcoef(a, b)[0, 1])
            print(f"    {family}/{metric:<12} r = {r:.4f}   max|diff| = {np.max(np.abs(a - b)):.4f}",
                  flush=True)


def all_horizon_features() -> dict[float, dict[str, pd.DataFrame]]:
    """Per-horizon feature frames for every family.

    Cached on its own, separately from any AUC computation, because the
    sequence metrics (Lempel-Ziv, determinism, Markov entropy) are quadratic in
    sequence length and dominate the runtime. Downstream analyses - the
    early-window combination search, SHAP - reuse this without recomputing.
    """
    raw = _raw_250ms()
    out: dict[float, dict[str, pd.DataFrame]] = {}
    for horizon in HORIZONS_MIN:
        out[horizon] = horizon_features(raw, horizon)
        print(f"    horizon {horizon} min features done", flush=True)
    validate_final_horizon(out[HORIZONS_MIN[-1]])
    return out


def horizon_family_sets(frames: dict[str, pd.DataFrame]) -> dict[str, tuple]:
    """Family name -> (frame, columns) for one horizon."""
    def cols_of(frame: pd.DataFrame) -> list[str]:
        return [c for c in frame.columns if "__" in c]

    frequency, diversity = frames["frequency"], frames["diversity"]
    transition, bout = frames["transition"], frames["bout"]
    all_metrics = efa.merge_feature_frames([frequency, diversity, transition, bout])
    # Supervised Freeze % dropped - see the note on SINGLE_PARAMETERS.
    return {
        "All behaviour time shares": (frequency, cols_of(frequency)),
        "Diversity metrics": (diversity, cols_of(diversity)),
        "Transition summaries": (transition, cols_of(transition)),
        "Bout durations": (bout, cols_of(bout)),
        "All metrics": (all_metrics, cols_of(all_metrics)),
    }


def panel_c_data(labels: pd.DataFrame, horizon_frames: dict) -> pd.DataFrame:
    """AUC against session length, for each predictor family."""
    rows = []
    for horizon in HORIZONS_MIN:
        families = horizon_family_sets(horizon_frames[horizon])
        for name, (frame, cols) in families.items():
            data = direct.attach(labels, frame)
            folds = direct.make_folds(
                direct.strat_key(data), direct.N_SPLITS, direct.N_REPEATS, RANDOM_SEED
            )
            aucs, _ = direct.cv_auc_np(
                data[cols].to_numpy(dtype=float), data["target"].to_numpy(dtype=int), folds
            )
            cc = direct.cross_cohort(data, cols, with_ci=False)
            rows.append(
                {
                    "horizon_min": horizon,
                    "predictor": name,
                    "color": BLOCK_COLORS[name],
                    "n_features": len(cols),
                    "cv_auc": float(aucs.mean()),
                    "cv_lo": float(np.percentile(aucs, 10)),
                    "cv_hi": float(np.percentile(aucs, 90)),
                    "auc_kru_to_gom": cc[0]["roc_auc"],
                    "auc_gom_to_kru": cc[1]["roc_auc"],
                }
            )
        print(f"    horizon {horizon} min done", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# drawing - mirrors tier1_reanalysis_summary.pdf
# --------------------------------------------------------------------------


def _x_labels(ax: plt.Axes, table: pd.DataFrame) -> None:
    """Shared x-axis treatment: blocks named in italic to separate them."""
    ax.set_xticks(np.arange(len(table)))
    labels = ax.set_xticklabels(
        table["predictor"], fontsize=4.3, rotation=42, ha="right", rotation_mode="anchor"
    )
    for label, kind in zip(labels, table["kind"]):
        if kind == "block":
            label.set_style("italic")
            label.set_fontweight("bold")
    ax.set_xlim(-0.7, len(table) - 0.3)


def draw_panel_a(ax: plt.Axes, table: pd.DataFrame) -> None:
    """Both transfer directions, one pair of bars per predictor."""
    x = np.arange(len(table))
    width = 0.38
    ax.bar(x - width / 2, table["auc_kru_to_gom"], width=width, color=table["color"],
           edgecolor="none", alpha=0.95, zorder=2)
    ax.bar(x + width / 2, table["auc_gom_to_kru"], width=width, color=table["color"],
           edgecolor="white", linewidth=0.3, alpha=0.45, zorder=2, hatch="////")
    ax.axhline(0.5, color=CHANCE_GREY, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    _x_labels(ax, table)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("ROC AUC", fontsize=5.8, labelpad=2)
    ax.set_title("Held-out cohort", fontsize=6.4, color=AXIS, pad=3)
    style_axis(ax, labelsize=5.0)
    handles = [
        Patch(facecolor="#BFBFBF", edgecolor="none",
              label="Sanguino Gómez & Krugers $\\rightarrow$ Sanguino Gómez et al."),
        Patch(facecolor="#BFBFBF", edgecolor="white", lw=0.3, alpha=0.45, hatch="////",
              label="Sanguino Gómez et al. $\\rightarrow$ Sanguino Gómez & Krugers"),
    ]
    leg = ax.legend(handles=handles, frameon=False, fontsize=4.5, loc="upper right",
                    handlelength=1.0, handleheight=0.75, handletextpad=0.4,
                    labelspacing=0.28, borderaxespad=0.1, ncol=2, columnspacing=1.0)
    for text in leg.get_texts():
        text.set_color(AXIS)


def draw_panel_b(ax: plt.Axes, table: pd.DataFrame) -> None:
    """One bar per predictor with a bootstrap CI, all 41 animals.

    Same x order as panel A so the two rows can be read against each other.
    """
    x = np.arange(len(table))
    ax.bar(x, table["cv_auc"], width=0.7, color=table["color"], edgecolor="none",
           alpha=0.95, zorder=2)
    # Percentile bootstrap intervals are not guaranteed to bracket the point
    # estimate exactly, so guard the tiny numerical case rather than letting
    # matplotlib reject a marginally negative arm.
    lo_arm = np.clip(table["cv_auc"] - table["cv_ci_low"], 0.0, None)
    hi_arm = np.clip(table["cv_ci_high"] - table["cv_auc"], 0.0, None)
    # Drawn one at a time so each interval takes a darkened version of its own
    # bar colour; errorbar's ecolor is scalar and cannot vary per point.
    for xi, value, lo, hi, color in zip(x, table["cv_auc"], lo_arm, hi_arm, table["color"]):
        ax.errorbar(
            xi, value, yerr=[[lo], [hi]], fmt="none", ecolor=darken(color),
            elinewidth=0.55, capsize=1.0, capthick=0.55, zorder=3,
        )
    ax.axhline(0.5, color=CHANCE_GREY, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    _x_labels(ax, table)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("ROC AUC", fontsize=5.8, labelpad=2)
    ax.set_title("Full-session cross-validation", fontsize=6.4, color=AXIS, pad=3)
    style_axis(ax, labelsize=5.0)


# The panel C table is cached from before the terminology fix, so its family
# names are remapped at draw time rather than paying a full recompute.
PANEL_C_RENAME = {
    "All behaviour time shares": "All frequencies",
    "Diversity metrics": "All diversity indices",
    "Transition summaries": "All transition metrics",
    "Bout durations": "All bout durations",
    "All metrics": "All parameters",
}


def draw_panel_c(ax: plt.Axes, table: pd.DataFrame) -> None:
    """AUC against how much of the session is used."""
    for name, sub in table.groupby("predictor", sort=False):
        sub = sub.sort_values("horizon_min")
        ax.plot(sub["horizon_min"], sub["cv_auc"], color=sub["color"].iloc[0],
                lw=0.9, marker="o", ms=2.4, mec="white", mew=0.3,
                label=PANEL_C_RENAME.get(name, name), zorder=2)
    ax.axhline(0.5, color=CHANCE_GREY, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    ax.set_xlim(0.0, 8.0)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("First minutes included", fontsize=5.8, labelpad=1.5)
    ax.set_ylabel("ROC AUC", fontsize=5.8, labelpad=2)
    ax.set_title("Early prediction", fontsize=6.4, color=AXIS, pad=3)
    style_axis(ax, labelsize=5.0)
    leg = ax.legend(
        frameon=False, fontsize=4.5, loc="lower right", ncol=2, columnspacing=1.0,
        handlelength=1.3, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2,
    )
    for text in leg.get_texts():
        text.set_color(AXIS)


def build_figure(ab_table: pd.DataFrame, c_table: pd.DataFrame) -> plt.Figure:
    """One panel per row, full width, so every parameter name stays legible."""
    fig = plt.figure(figsize=(7.09, 7.4))  # 180 mm wide, three stacked rows
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.0, 1.0, 0.92],
        left=0.085, right=0.985, top=0.965, bottom=0.055, hspace=0.95,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])

    draw_panel_a(ax_a, ab_table)
    draw_panel_b(ax_b, ab_table)
    draw_panel_c(ax_c, c_table)

    tag(ax_a, "A", x=-0.062, y=1.14)
    tag(ax_b, "B", x=-0.062, y=1.14)
    tag(ax_c, "C", x=-0.062, y=1.14)
    return fig


def build_figure_compact(ab_table: pd.DataFrame, c_table: pd.DataFrame) -> plt.Figure:
    """Same data, but panel C kept narrow.

    Panels A and B need the full width for 30 parameter names; panel C only has
    six x positions, so stretching it to the same width flattens the curves and
    makes the horizon differences hard to read. Here it keeps its natural
    aspect and the legend moves outside.
    """
    fig = plt.figure(figsize=(7.09, 6.9))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.0, 1.0, 0.85],
        left=0.085, right=0.985, top=0.965, bottom=0.055, hspace=0.95,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    # Panel C occupies the left ~42% of the bottom row; the legend sits beside it.
    inner = gs[2, 0].subgridspec(1, 2, width_ratios=[0.42, 0.58], wspace=0.04)
    ax_c = fig.add_subplot(inner[0, 0])
    ax_legend = fig.add_subplot(inner[0, 1])
    ax_legend.axis("off")

    draw_panel_a(ax_a, ab_table)
    draw_panel_b(ax_b, ab_table)
    draw_panel_c(ax_c, c_table)

    # Move panel C's legend out of the axes so the curves are unobstructed.
    if ax_c.get_legend() is not None:
        ax_c.get_legend().remove()
    handles, labels = ax_c.get_legend_handles_labels()
    leg = ax_legend.legend(
        handles, labels, frameon=False, fontsize=5.0, loc="center left",
        bbox_to_anchor=(0.02, 0.5), handlelength=1.4, handletextpad=0.5,
        labelspacing=0.42, borderaxespad=0.0,
    )
    for text in leg.get_texts():
        text.set_color(AXIS)

    tag(ax_a, "A", x=-0.062, y=1.14)
    tag(ax_b, "B", x=-0.062, y=1.14)
    tag(ax_c, "C", x=-0.148, y=1.14)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplementary Figure 4 recapitulation")
    parser.add_argument("--fresh", action="store_true",
                        help="recompute every statistic instead of reading the cache")
    args = parser.parse_args()
    started = time.time()

    labels, predictors = cached("predictors", build_predictors, fresh=args.fresh)
    print(f"n = {len(labels)} animals; {len(predictors)} predictors")

    ab_table = cached("panel_ab", lambda: panel_ab_data(labels, predictors), fresh=args.fresh)
    print("panels A and B done")

    horizon_frames = cached("horizon_features", all_horizon_features, fresh=args.fresh)
    c_table = cached("panel_c", lambda: panel_c_data(labels, horizon_frames), fresh=args.fresh)
    print("panel C done")

    fig = build_figure(ab_table, c_table)
    stem = OUT / "supplementary_figure4_recap"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig_compact = build_figure_compact(ab_table, c_table)
    stem_compact = OUT / "supplementary_figure4_recap_compact"
    for ext in ("png", "pdf", "svg"):
        fig_compact.savefig(stem_compact.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig_compact)
    print(f"wrote {stem_compact}.png/.pdf/.svg", flush=True)

    ab_table.drop(columns=["color"]).to_csv(OUT / "supp4_recap_panelAB_cross_cohort_and_cv.csv", index=False)
    c_table.drop(columns=["color"]).to_csv(OUT / "supp4_recap_panelC_time_to_prediction.csv", index=False)

    print("\nHeld-out cohort (panel A):")
    for _, r in ab_table.iterrows():
        print(f"  {r['predictor']:<28} {r['auc_kru_to_gom']:.3f} / {r['auc_gom_to_kru']:.3f}"
              f"   worse = {r['worst_direction_auc']:.3f}")
    print("\nFull-session CV (panel B):")
    for _, r in ab_table.sort_values("cv_auc", ascending=False).iterrows():
        print(f"  {r['predictor']:<28} {r['cv_auc']:.3f}  [{r['cv_ci_low']:.3f}-{r['cv_ci_high']:.3f}]")
    print(f"\nwrote {stem}.png/.pdf/.svg   ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
