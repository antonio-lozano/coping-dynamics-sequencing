# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gómez and Antonio Lozano
"""Regenerate Supplementary Figure 3 distance-metric controls for Figure 5."""

from __future__ import annotations

from pathlib import Path
import gzip
import pickle
import sys
import zipfile
from io import BytesIO

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.spatial.distance import cityblock, correlation, cosine
from scipy.stats import ttest_ind

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    COPING_DATA2_ZIP as COPING_DATA2,
    COPING2_MEMBER_UPDATED_RESULTS,
    FIGURE_DATA_DIR as OUTPUT_DIR,
    UPDATED_RESULTS_PKL as ORIGINAL_EQUIPO_RESULTS,
)

LEGACY_FIGURES_DIR = REPO_ROOT / "figures"

AXIS = "#4D4D4D"
CONTROL = "#F9C74F"
ELS = "#C37BA0"
RESILIENT = "#90BE6D"
PALETTE = {"Control": CONTROL, "ELS": ELS}
BOX_YLABEL_X = -0.58
BOX_TAG_X = -0.58
BOX_TOP_LIMITS = {
    "Correlation": 1.5,
    "Cosine": 1.3,
    "Minkowski": 1.5,
    "Manhattan": 2.0,
}
METRICS = [
    ("Correlation", correlation, "A", "B"),
    ("Cosine", cosine, "C", "D"),
    ("Minkowski", lambda a, b: float(np.sum(np.abs(a - b) ** 2.5) ** (1.0 / 2.5)), "F", "G"),
    ("Manhattan", cityblock, "H", "I"),
]
FIG5_RESILIENT = {
    "2.4",
    "15.4",
    "26.4",
    "43.3",
    "43.5",
    "49.6",
    "123.3",
    "134.2",
    "148.4",
    "150.3",
    "150.5",
    "159.4",
}


def read_zip_pickle(zip_path: Path, member: str) -> object:
    with zipfile.ZipFile(zip_path) as zf:
        return pickle.loads(zf.read(member))


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def metric_mds_smacof(
    dist: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
    n_init: int = 4,
    max_iter: int = 300,
    eps: float = 1e-3,
) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    n = dist.shape[0]
    best_coords = None
    best_stress = np.inf
    for _ in range(n_init):
        coords = rng.uniform(size=n * n_components).reshape(n, n_components)
        old_stress = None
        for _iter in range(max_iter):
            distances = pairwise_euclidean(coords)
            stress = 0.5 * np.square(dist - distances).sum()
            distances[distances == 0] = 1e-5
            ratio = dist / distances
            bmat = -ratio
            bmat[np.arange(n), np.arange(n)] += ratio.sum(axis=1)
            coords = bmat.dot(coords) / n
            norm = np.sqrt(np.square(coords).sum(axis=1)).sum()
            normalized = stress / max(norm, 1e-12)
            if old_stress is not None and old_stress - normalized < eps:
                break
            old_stress = normalized
        if stress < best_stress:
            best_stress = stress
            best_coords = coords.copy()
    if best_coords is None:
        raise RuntimeError("Metric MDS failed to converge.")
    return best_coords


def loocv_logistic(coords: np.ndarray, labels: np.ndarray, c_value: float = 1.0) -> float:
    y = np.array([0.0 if label == "Control" else 1.0 for label in labels])
    design = np.column_stack([np.ones(len(coords)), coords])
    correct = 0
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        x_train = design[train]
        y_train = y[train]

        def objective(weights: np.ndarray) -> float:
            logits = x_train @ weights
            nll = np.logaddexp(0, logits).sum() - y_train @ logits
            penalty = 0.5 / c_value * np.square(weights[1:]).sum()
            return float(nll + penalty)

        result = minimize(objective, np.zeros(design.shape[1]), method="BFGS")
        prob = 1.0 / (1.0 + np.exp(-(design[i] @ result.x)))
        correct += int((prob >= 0.5) == bool(y[i]))
    return correct / len(y)


def load_feature_matrix() -> tuple[pd.DataFrame, np.ndarray]:
    if ORIGINAL_EQUIPO_RESULTS.exists():
        opener = gzip.open if ORIGINAL_EQUIPO_RESULTS.suffix == ".gz" else open
        with opener(ORIGINAL_EQUIPO_RESULTS, "rb") as f:
            results = pickle.load(f)
    else:
        results = read_zip_pickle(COPING_DATA2, COPING2_MEMBER_UPDATED_RESULTS)

    valid_codes = list(range(1, 8))
    bin_size = 25 * 30
    rows = []
    features = []
    for rec, payload in results.items():
        seq = np.asarray(payload.get("syllable", []))
        n_bins = len(seq) // bin_size
        if n_bins == 0:
            continue
        seq = seq[: n_bins * bin_size].reshape(n_bins, bin_size)
        feature = []
        for chunk in seq:
            feature.extend([np.sum(chunk == code) / bin_size * 100.0 for code in valid_codes])
        rows.append(
            {
                "recording": str(rec),
                "animal": str(payload.get("Animal")),
                "group": str(payload.get("Condition")),
            }
        )
        features.append(feature)
    return pd.DataFrame(rows), np.asarray(features, dtype=float)


def load_transition_features() -> tuple[pd.DataFrame, np.ndarray]:
    """Per-animal first-order transition-probability matrix (7x7) between behavioral
    clusters, flattened to a 49-dim feature (von Ziegler-style behavioural flow).
    Consecutive identical frames are collapsed to a bout-level state sequence."""
    if ORIGINAL_EQUIPO_RESULTS.exists():
        opener = gzip.open if ORIGINAL_EQUIPO_RESULTS.suffix == ".gz" else open
        with opener(ORIGINAL_EQUIPO_RESULTS, "rb") as f:
            results = pickle.load(f)
    else:
        results = read_zip_pickle(COPING_DATA2, COPING2_MEMBER_UPDATED_RESULTS)

    codes = list(range(1, 8))
    bin_size = 25 * 30
    rows = []
    features = []
    for rec, payload in results.items():
        seq = np.asarray(payload.get("syllable", []))
        if len(seq) // bin_size == 0:
            continue
        valid = seq[np.isin(seq, codes)]
        mat = np.zeros((7, 7), dtype=float)
        if len(valid) >= 2:
            collapsed = valid[np.concatenate(([True], valid[1:] != valid[:-1]))]
            for a, b in zip(collapsed[:-1], collapsed[1:]):
                mat[int(a) - 1, int(b) - 1] += 1
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        rows.append(
            {
                "recording": str(rec),
                "animal": str(payload.get("Animal")),
                "group": str(payload.get("Condition")),
            }
        )
        features.append((mat / row_sums).flatten())
    return pd.DataFrame(rows), np.asarray(features, dtype=float)


def profile_from_distance(meta: pd.DataFrame, dist: np.ndarray, metric_name: str) -> tuple[pd.DataFrame, dict]:
    """Build an MDS dynamics-score profile + summary row from a precomputed distance matrix."""
    coords = metric_mds_smacof(dist, random_state=42)
    prof = meta.copy()
    prof["metric"] = metric_name
    prof[["mds1", "mds2"]] = coords
    median_ctrl = prof.loc[prof["group"] == "Control", ["mds1", "mds2"]].median().to_numpy()
    median_els = prof.loc[prof["group"] == "ELS", ["mds1", "mds2"]].median().to_numpy()
    d_ctrl = np.linalg.norm(coords - median_ctrl, axis=1)
    d_els = np.linalg.norm(coords - median_els, axis=1)
    prof["dynamics_score"] = np.log((d_ctrl + 1e-9) / (d_els + 1e-9))
    prof["resilient_by_zero"] = (prof["group"] == "ELS") & (prof["dynamics_score"] < 0)
    loocv = loocv_logistic(coords, meta["group"].to_numpy())
    prof["loocv_accuracy"] = loocv
    ctrl = prof.loc[prof["group"] == "Control", "dynamics_score"]
    els = prof.loc[prof["group"] == "ELS", "dynamics_score"]
    t_stat, p_value = ttest_ind(ctrl, els, equal_var=False, nan_policy="omit")
    summary = {
        "metric": metric_name,
        "loocv_accuracy": loocv,
        "control_n": int(ctrl.count()),
        "els_n": int(els.count()),
        "control_mean_score": float(ctrl.mean()),
        "els_mean_score": float(els.mean()),
        "welch_t_control_vs_els": float(t_stat),
        "welch_p_value": float(p_value),
    }
    return prof, summary


def pairwise_distance(features: np.ndarray, metric_name: str, func) -> np.ndarray:
    n = len(features)
    dist = np.zeros((n, n), dtype=float)
    if metric_name == "Jensen-Shannon":
        denom = features.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        work = features / denom
    else:
        work = features
    for i in range(n):
        for j in range(n):
            value = func(work[i], work[j])
            dist[i, j] = 0.0 if np.isnan(value) else float(value)
    return dist


def metric_profiles(meta: pd.DataFrame, features: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_rows = []
    summary_rows = []
    labels = meta["group"].to_numpy()
    for metric_name, func, _, _ in METRICS:
        dist = pairwise_distance(features, metric_name, func)
        coords = metric_mds_smacof(dist, random_state=42)
        prof = meta.copy()
        prof["metric"] = metric_name
        prof[["mds1", "mds2"]] = coords
        median_ctrl = prof.loc[prof["group"] == "Control", ["mds1", "mds2"]].median().to_numpy()
        median_els = prof.loc[prof["group"] == "ELS", ["mds1", "mds2"]].median().to_numpy()
        d_ctrl = np.linalg.norm(coords - median_ctrl, axis=1)
        d_els = np.linalg.norm(coords - median_els, axis=1)
        prof["dynamics_score"] = np.log((d_ctrl + 1e-9) / (d_els + 1e-9))
        prof["resilient_by_zero"] = (prof["group"] == "ELS") & (prof["dynamics_score"] < 0)
        loocv = loocv_logistic(coords, labels)
        ctrl = prof.loc[prof["group"] == "Control", "dynamics_score"]
        els = prof.loc[prof["group"] == "ELS", "dynamics_score"]
        t_stat, p_value = ttest_ind(ctrl, els, equal_var=False, nan_policy="omit")
        prof["loocv_accuracy"] = loocv
        profile_rows.append(prof)
        summary_rows.append(
            {
                "metric": metric_name,
                "loocv_accuracy": loocv,
                "control_n": int(ctrl.count()),
                "els_n": int(els.count()),
                "control_mean_score": float(ctrl.mean()),
                "els_mean_score": float(els.mean()),
                "welch_t_control_vs_els": float(t_stat),
                "welch_p_value": float(p_value),
            }
        )
    return pd.concat(profile_rows, ignore_index=True), pd.DataFrame(summary_rows)


def threshold_audit(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric_name, sub in profiles[profiles["group"] == "ELS"].groupby("metric", sort=False):
        zero_ids = set(sub.loc[sub["dynamics_score"] < 0, "animal"])
        mean_threshold = float(sub["dynamics_score"].mean())
        median_threshold = float(sub["dynamics_score"].median())
        mean_ids = set(sub.loc[sub["dynamics_score"] < mean_threshold, "animal"])
        median_ids = set(sub.loc[sub["dynamics_score"] < median_threshold, "animal"])
        rows.append(
            {
                "metric": metric_name,
                "els_n": int(len(sub)),
                "zero_threshold": 0.0,
                "zero_n": int(len(zero_ids)),
                "zero_matches_figure5_euclidean": zero_ids == FIG5_RESILIENT,
                "zero_extra_vs_figure5_euclidean": ", ".join(sorted(zero_ids - FIG5_RESILIENT)),
                "zero_missing_vs_figure5_euclidean": ", ".join(sorted(FIG5_RESILIENT - zero_ids)),
                "els_mean_score": mean_threshold,
                "mean_threshold_n": int(len(mean_ids)),
                "mean_added_vs_zero": ", ".join(sorted(mean_ids - zero_ids)),
                "mean_removed_vs_zero": ", ".join(sorted(zero_ids - mean_ids)),
                "els_median_score": median_threshold,
                "median_threshold_n": int(len(median_ids)),
                "median_added_vs_zero": ", ".join(sorted(median_ids - zero_ids)),
                "median_removed_vs_zero": ", ".join(sorted(zero_ids - median_ids)),
            }
        )
    return pd.DataFrame(rows)


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


def tag(ax: plt.Axes, letter: str, x: float = -0.16, y: float = 1.10) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="center", va="top", fontsize=8, fontweight="bold", color=AXIS, clip_on=False)


def match_fig5_mds_proportions(ax: plt.Axes) -> None:
    ax.set_box_aspect(0.76)


def align_box_to_mds(mds_ax: plt.Axes, box_ax: plt.Axes) -> None:
    mds_pos = mds_ax.get_position()
    box_pos = box_ax.get_position()
    gap = 0.112
    new_x0 = mds_pos.x0 + mds_pos.width + gap
    new_width = min(box_pos.width * 0.92, 0.073)
    box_ax.set_position([new_x0, mds_pos.y0, new_width, mds_pos.height])


def plot_mds(ax: plt.Axes, data: pd.DataFrame, metric_name: str, letter: str) -> None:
    match_fig5_mds_proportions(ax)
    x = data["mds1"].to_numpy()
    y = data["mds2"].to_numpy()
    score = data["dynamics_score"].to_numpy()
    margin = 0.10 * max(np.ptp(x), np.ptp(y))
    gx, gy = np.mgrid[x.min() - margin : x.max() + margin : 150j, y.min() - margin : y.max() + margin : 150j]
    gz = griddata(np.column_stack([x, y]), score, (gx, gy), method="cubic")
    nn = griddata(np.column_stack([x, y]), score, (gx, gy), method="nearest")
    gz = np.where(np.isnan(gz), nn, gz)
    vabs = max(abs(np.nanmin(score)), abs(np.nanmax(score)))
    levels = np.linspace(-vabs, vabs, 19)
    cmap = mcolors.LinearSegmentedColormap.from_list("dyn", [CONTROL, "#FFFFFF", ELS])
    cf = ax.contourf(gx, gy, np.clip(gz, -vabs + 1e-9, vabs - 1e-9), levels=levels, cmap=cmap, alpha=0.62)
    cs = ax.contour(gx, gy, gz, levels=levels, colors=AXIS, linewidths=0.32, alpha=0.82)
    ax.clabel(cs, levels[::2], inline=True, fontsize=3.8, fmt="%.2g", colors=AXIS)
    for group in ["Control", "ELS"]:
        sub = data[data["group"] == group]
        ax.scatter(sub["mds1"], sub["mds2"], s=21, color=PALETTE[group], edgecolor=AXIS, linewidth=0.30, alpha=0.88, label=group, zorder=3)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=PALETTE[group], markeredgecolor="none", markersize=4.0, label=group)
        for group in ["Control", "ELS"]
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, facecolor="white", edgecolor="#D0D0D0", fontsize=5.1, borderpad=0.16, handlelength=0.55, handletextpad=0.38)
    cax = ax.inset_axes([1.05, 0.0, 0.042, 1.0])
    cbar = plt.colorbar(cf, cax=cax)
    cbar.ax.yaxis.set_label_position("left")
    cbar.set_label("Dynamic Similarity Score", fontsize=4.8, color=AXIS, labelpad=1.8)
    cbar.ax.tick_params(labelsize=4.4, width=0.35, length=1.5, colors=AXIS)
    cbar.outline.set_linewidth(0.35)
    ax.text(0.985, -0.145, f"LOOCV Acc:\n{data['loocv_accuracy'].iloc[0]*100:.1f}%", transform=ax.transAxes, ha="right", va="top", fontsize=4.6, color=AXIS, clip_on=False, bbox=dict(facecolor="white", edgecolor="#BDBDBD", linewidth=0.35, pad=1.4))
    ax.set_title(f"MDS Plot ({metric_name} Distance)", fontsize=6.2, pad=3)
    ax.set_xlabel("MDS Dimension 1", fontsize=5.7, labelpad=1)
    ax.set_ylabel("MDS Dimension 2", fontsize=5.7, labelpad=1)
    style_axis(ax)
    tag(ax, letter)


def plot_box(ax: plt.Axes, data: pd.DataFrame, metric_name: str, letter: str) -> None:
    groups = ["Control", "ELS"]
    y_max = np.nanmax(data["dynamics_score"])
    y_min = np.nanmin(data["dynamics_score"])
    pad = (y_max - y_min) * 0.18
    y0 = y_min - pad
    y1 = BOX_TOP_LIMITS.get(metric_name, y_max + pad)
    resilient_vals = data.loc[(data["group"] == "ELS") & (data["dynamics_score"] < 0), "dynamics_score"].dropna()
    if not resilient_vals.empty:
        rect_pad = (y1 - y0) * 0.018
        rect_y0 = max(y0, float(resilient_vals.min()) - rect_pad)
        ax.add_patch(
            plt.Rectangle(
                (1 - 0.18, rect_y0),
                0.36,
                min(0.0, y1) - rect_y0,
                facecolor=(*mcolors.to_rgb(RESILIENT), 0.10),
                edgecolor=RESILIENT,
                linewidth=0.75,
                linestyle=(0, (2.0, 1.4)),
                zorder=1.5,
            )
        )
    for i, group in enumerate(groups):
        sub = data[data["group"] == group].dropna(subset=["dynamics_score"]).copy()
        vals = sub["dynamics_score"].to_numpy()
        color = PALETTE[group]
        ax.boxplot(
            vals,
            positions=[i],
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": (*mcolors.to_rgb(color), 0.15), "edgecolor": color, "linewidth": 0.75},
            whiskerprops={"color": color, "linewidth": 0.75},
            capprops={"color": color, "linewidth": 0.75},
            medianprops={"color": color, "linewidth": 0.95},
        )
        rng = np.random.default_rng(12 + i)
        jitter = rng.normal(i, 0.024, len(vals))
        ax.scatter(jitter, vals, s=6.5, color=color, alpha=0.78, linewidth=0, zorder=3)
    ax.set_ylim(y0, y1)
    ticks = [tick for tick in ax.get_yticks() if y0 <= tick <= y1]
    if y1 not in ticks:
        ticks.append(y1)
    ax.set_yticks(sorted(ticks))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(groups, fontsize=5.8)
    ax.tick_params(axis="x", pad=4)
    ax.set_ylabel(f"log ({metric_name} score)", fontsize=5.0, labelpad=4.5)
    ax.yaxis.set_label_coords(BOX_YLABEL_X, 0.5)
    style_axis(ax, labelsize=5.0)
    p_value = ttest_ind(
        data[data["group"] == "Control"]["dynamics_score"],
        data[data["group"] == "ELS"]["dynamics_score"],
        equal_var=False,
        nan_policy="omit",
    ).pvalue
    if p_value < 0.05:
        y = y_max + pad * 0.28
        h = pad * 0.18
        ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], color=AXIS, linewidth=0.55, clip_on=False)
        ax.text(0.5, y + h * 1.1, "*", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=AXIS, clip_on=False)
    tag(ax, letter, x=BOX_TAG_X)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta, features = load_feature_matrix()
    profiles, summary = metric_profiles(meta, features)

    # Transition-probability MDS (panels J-K): von Ziegler-style behavioural flow.
    tmeta, tfeatures = load_transition_features()
    tprof, tsummary = profile_from_distance(tmeta, pairwise_euclidean(tfeatures), "Transition")
    trans_res = set(tprof.loc[tprof["resilient_by_zero"], "animal"])
    overlap = len(trans_res & FIG5_RESILIENT) / len(FIG5_RESILIENT) * 100.0
    tsummary["overlap_pct_with_fig5_euclidean"] = overlap
    print(
        f"Transition MDS: resilient n={len(trans_res)}, "
        f"overlap with Fig 5 Euclidean classification = {overlap:.1f}%"
    )
    profiles = pd.concat([profiles, tprof], ignore_index=True)
    summary = pd.concat([summary, pd.DataFrame([tsummary])], ignore_index=True)

    profiles.to_csv(OUTPUT_DIR / "supplementary_figure_3_distance_scores.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "supplementary_figure_3_distance_summary.csv", index=False)
    threshold_audit(profiles).to_csv(OUTPUT_DIR / "supplementary_figure_3_threshold_audit.csv", index=False)

    fig = plt.figure(figsize=(8.27, 11.69), dpi=300, facecolor="white")
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        left=0.075,
        right=0.955,
        top=0.965,
        bottom=0.66,
        wspace=0.78,
        hspace=0.30,
        width_ratios=[2.02, 0.74, 2.02, 0.74],
    )
    positions = [(0, 0), (0, 2), (1, 0), (1, 2)]
    for (metric_name, _, mds_letter, box_letter), (row, col) in zip(METRICS, positions):
        sub = profiles[profiles["metric"] == metric_name].copy()
        mds_ax = fig.add_subplot(gs[row, col])
        box_ax = fig.add_subplot(gs[row, col + 1])
        plot_mds(mds_ax, sub, metric_name, mds_letter)
        plot_box(box_ax, sub, metric_name, box_letter)
        align_box_to_mds(mds_ax, box_ax)

    # Transition-probability panels (J, K) on a new row below.
    gs_trans = fig.add_gridspec(
        nrows=1,
        ncols=4,
        left=0.075,
        right=0.955,
        top=0.625,
        bottom=0.47,
        wspace=0.78,
        width_ratios=[2.02, 0.74, 2.02, 0.74],
    )
    trans_sub = profiles[profiles["metric"] == "Transition"].copy()
    trans_mds_ax = fig.add_subplot(gs_trans[0, 0])
    trans_box_ax = fig.add_subplot(gs_trans[0, 1])
    plot_mds(trans_mds_ax, trans_sub, "Transition Prob., Euclidean", "J")
    plot_box(trans_box_ax, trans_sub, "Transition", "K")
    align_box_to_mds(trans_mds_ax, trans_box_ax)

    pdf = OUTPUT_DIR / "supplementary_figure_3_distances.pdf"
    svg = OUTPUT_DIR / "supplementary_figure_3_distances.svg"
    png = OUTPUT_DIR / "supplementary_figure_3_distances.png"
    fig.savefig(pdf)
    fig.savefig(svg)
    fig.savefig(png, dpi=300)
    LEGACY_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LEGACY_FIGURES_DIR / "supplementary_figure3.pdf")
    fig.savefig(LEGACY_FIGURES_DIR / "supplementary_figure3.svg")
    fig.savefig(LEGACY_FIGURES_DIR / "supplementary_figure3.png", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf}")
    print(f"Saved {svg}")
    print(f"Saved {png}")
    print(f"Saved {LEGACY_FIGURES_DIR / 'supplementary_figure3.pdf'}")


if __name__ == "__main__":
    main()
