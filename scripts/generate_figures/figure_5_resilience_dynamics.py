# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gómez and Antonio Lozano
"""Regenerate manuscript Figure 5 in the final Figure 3/4 A4 style."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import gzip
import pickle
import sys
import zipfile

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    BFL_SCORES_XLSX,
    COPING_DATA_ZIP as COPING_DATA,
    COPING_DATA2_ZIP as COPING_DATA2,
    COPING_MEMBER_BFL_SCORES,
    COPING2_MEMBER_TIMEBIN_30S,
    COPING2_MEMBER_UPDATED_RESULTS,
    FIGURE_DATA_DIR as OUTPUT_DIR,
    SYLLABLE_TIMEBIN_30S,
    UPDATED_RESULTS_PKL as ORIGINAL_EQUIPO_RESULTS,
)
from src.statistics import fit_mixed_models

LEGACY_FIGURES_DIR = REPO_ROOT / "figures"

AXIS = "#4D4D4D"
CONTROL = "#F9C74F"
ELS = "#C37BA0"
RESILIENT = "#90BE6D"
PALETTE = {"Control": CONTROL, "ELS": ELS, "ELS resilient": RESILIENT}
GROUP_ORDER = ["Control", "ELS", "ELS resilient"]
EVENT_SPANS = [(3.5, 4.0), (5.0, 5.5), (6.5, 7.0)]

ELS_RESILIENT = {
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

CLUSTER_MAP = {
    "Freeze": [0, 28],
    "Sniff": [18, 20],
    "Groom": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climb": [111],
    "Jump": [23, 29, 30, 34],
}
ORDER = ["Freeze", "Sniff", "Groom", "Turn", "Locomotion", "Climb", "Jump"]


def read_zip_csv(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        return pd.read_csv(BytesIO(zf.read(member)))


def read_zip_excel(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        return pd.read_excel(BytesIO(zf.read(member)))


def read_zip_pickle(zip_path: Path, member: str) -> object:
    with zipfile.ZipFile(zip_path) as zf:
        return pickle.loads(zf.read(member))


def group_ext(animal: str, group: str) -> str:
    animal = str(animal)
    if group == "ELS" and animal in ELS_RESILIENT:
        return "ELS resilient"
    return group


def load_cluster_time() -> pd.DataFrame:
    if SYLLABLE_TIMEBIN_30S.exists():
        raw = pd.read_csv(SYLLABLE_TIMEBIN_30S)
    else:
        raw = read_zip_csv(COPING_DATA2, COPING2_MEMBER_TIMEBIN_30S)
    raw = raw.rename(columns={"Condition": "group", "Time Bin": "time_bin"})
    raw["Animal"] = raw["Animal"].astype(str)
    raw["Syllable"] = pd.to_numeric(raw["Syllable"], errors="coerce").astype(int)
    raw["time_bin"] = pd.to_numeric(raw["time_bin"], errors="coerce").astype(int)
    raw["group_ext"] = raw.apply(lambda r: group_ext(r["Animal"], r["group"]), axis=1)

    frames = []
    base_cols = ["Animal", "group", "group_ext", "Experiment", "time_bin"]
    base = raw[base_cols].drop_duplicates()
    for cluster, syllables in CLUSTER_MAP.items():
        grid = base.assign(_key=1).merge(pd.DataFrame({"Syllable": syllables, "_key": 1}), on="_key").drop(columns="_key")
        complete = grid.merge(raw, on=base_cols + ["Syllable"], how="left")
        complete["Percentage"] = complete["Percentage"].fillna(0.0)
        frames.append(complete.groupby(base_cols, as_index=False)["Percentage"].sum().assign(cluster=cluster))
    out = pd.concat(frames, ignore_index=True)
    out["time_min"] = (out["time_bin"] + 30) / 60.0
    return out.rename(columns={"Animal": "animal", "Percentage": "percent"})


def load_bfl() -> pd.DataFrame:
    if BFL_SCORES_XLSX.exists():
        bfl = pd.read_excel(BFL_SCORES_XLSX)
    else:
        bfl = read_zip_excel(COPING_DATA, COPING_MEMBER_BFL_SCORES)
    bfl = bfl[["Animal", "Condition", "Score", "Experiment"]].dropna(subset=["Animal", "Condition", "Score"])
    bfl["animal"] = bfl["Animal"].astype(float).map(lambda v: str(v).rstrip("0").rstrip(".") if "." in str(v) else str(v))
    bfl["animal"] = bfl["Animal"].map(lambda v: f"{float(v):.1f}")
    bfl = bfl.rename(columns={"Condition": "group", "Score": "score", "Experiment": "experiment"})
    bfl["group_ext"] = bfl.apply(lambda r: group_ext(r["animal"], r["group"]), axis=1)
    return bfl


def summarize_time(cluster_time: pd.DataFrame) -> pd.DataFrame:
    return (
        cluster_time.groupby(["group_ext", "cluster", "time_bin", "time_min"])["percent"]
        .agg(["mean", "sem"])
        .reset_index()
    )


def frequency_points(cluster_time: pd.DataFrame) -> pd.DataFrame:
    return (
        cluster_time.assign(seconds=cluster_time["percent"] / 100.0 * 30.0)
        .groupby(["animal", "group_ext", "cluster"], as_index=False)["seconds"]
        .sum()
    )


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def classical_mds(dist: np.ndarray, n_components: int = 2) -> np.ndarray:
    n = dist.shape[0]
    centered = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centered @ (dist * dist) @ centered
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1][:n_components]
    eigvals = np.maximum(eigvals[order], 0)
    return eigvecs[:, order] * np.sqrt(eigvals)


def metric_mds_smacof(
    dist: np.ndarray,
    n_components: int = 2,
    random_state: int = 0,
    n_init: int = 4,
    max_iter: int = 300,
    eps: float = 1e-3,
) -> np.ndarray:
    """Small metric-MDS SMACOF implementation matching sklearn's original Figure 7 recipe."""
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


def rescale_mds_to_reference_space(coords: np.ndarray) -> np.ndarray:
    """Match the coordinate window used by the original trimmed Figure 5 MDS panel."""
    target_xlim = (-0.45, 0.70)
    target_ylim = (-0.50, 0.55)
    target_center = np.array([0.0, 0.0])
    centered = coords - coords.mean(axis=0)
    span = np.ptp(centered, axis=0)
    target_span = np.array([target_xlim[1] - target_xlim[0], target_ylim[1] - target_ylim[0]])
    scale = float(np.min(target_span / span))
    return centered * scale + target_center


def loocv_nearest_centroid(coords: np.ndarray, labels: np.ndarray) -> float:
    correct = 0
    for i in range(len(labels)):
        train = np.arange(len(labels)) != i
        centroids = {}
        for label in np.unique(labels):
            centroids[label] = coords[train & (labels == label)].mean(axis=0)
        pred = min(centroids, key=lambda label: np.linalg.norm(coords[i] - centroids[label]))
        correct += int(pred == labels[i])
    return correct / len(labels)


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


def mds_profiles_from_updated_results() -> tuple[pd.DataFrame, np.ndarray, float]:
    if ORIGINAL_EQUIPO_RESULTS.exists():
        opener = gzip.open if ORIGINAL_EQUIPO_RESULTS.suffix == ".gz" else open
        with opener(ORIGINAL_EQUIPO_RESULTS, "rb") as f:
            results = pickle.load(f)
    else:
        results = read_zip_pickle(COPING_DATA2, COPING2_MEMBER_UPDATED_RESULTS)
    valid_codes = list(range(1, 8))
    fps = 25
    bin_seconds = 30
    bin_size = fps * bin_seconds

    profiles = []
    features = []
    for rec, data in results.items():
        seq = np.asarray(data.get("syllable", []))
        n_bins = int(len(seq) / bin_size)
        if n_bins == 0:
            continue
        seq = seq[: n_bins * bin_size]
        arr = seq.reshape(n_bins, bin_size)
        feature = []
        for chunk in arr:
            feature.extend([np.sum(chunk == code) / float(bin_size) * 100.0 for code in valid_codes])
        animal = str(data.get("Animal"))
        group = str(data.get("Condition"))
        profiles.append({"recording": str(rec), "animal": animal, "group": group})
        features.append(feature)

    feature_matrix = np.asarray(features, dtype=float)
    dist = pairwise_euclidean(feature_matrix)
    d_min = np.min(dist)
    d_max = np.max(dist)
    dist = (dist - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(dist)
    coords = metric_mds_smacof(dist, random_state=42)
    prof = pd.DataFrame(profiles)
    prof[["mds1", "mds2"]] = coords
    prof["group_ext"] = prof.apply(lambda r: group_ext(r["animal"], r["group"]), axis=1)

    median_ctrl = prof.loc[prof["group"] == "Control", ["mds1", "mds2"]].median().to_numpy()
    median_els = prof.loc[prof["group"] == "ELS", ["mds1", "mds2"]].median().to_numpy()
    d_ctrl = np.linalg.norm(coords - median_ctrl, axis=1)
    d_els = np.linalg.norm(coords - median_els, axis=1)
    prof["dynamics_score"] = np.log((d_ctrl + 1e-9) / (d_els + 1e-9))

    loocv = loocv_logistic(coords, prof["group"].to_numpy())
    return prof, coords, loocv


def style_axis(ax: plt.Axes, labelsize: float = 6.0) -> None:
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
        ax.spines[spine].set_linewidth(0.45)
    ax.tick_params(axis="both", colors=AXIS, labelsize=labelsize, width=0.45, length=2)
    ax.xaxis.label.set_color(AXIS)
    ax.yaxis.label.set_color(AXIS)
    ax.title.set_color(AXIS)


def tag(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.12) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, fontweight="bold", color=AXIS, clip_on=False)


def add_epochs(ax: plt.Axes) -> None:
    for start, end in EVENT_SPANS:
        ax.axvspan(start, end, color="#EDEDED", zorder=0)


def sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, h: float | None = None, text: str = "*") -> None:
    lo, hi = ax.get_ylim()
    h = h if h is not None else (hi - lo) * 0.035
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=AXIS, linewidth=0.55, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.12, text, ha="center", va="bottom", fontsize=9, fontweight="bold", color=AXIS, clip_on=False)


def legend_boxes(groups: list[str]) -> list[mpatches.Patch]:
    return [mpatches.Patch(facecolor=PALETTE[group], edgecolor="none", label=group) for group in groups]


def time_panel_legend(ax: plt.Axes, corner: str = "upper_right") -> None:
    anchors = {
        "upper_right": (0.88, 0.955, 0.905),
        "upper_left": (0.25, 0.955, 0.905),
        "lower_right": (0.88, 0.095, 0.045),
        "lower_left": (0.25, 0.095, 0.045),
    }
    x_center, y_top, y_bottom = anchors[corner]
    handles = [
        plt.Line2D([0], [0], color=PALETTE[group], marker="o", markersize=2.2, linewidth=1.0, label=group)
        for group in GROUP_ORDER
    ]
    top_leg = ax.legend(
        handles=handles[:2],
        loc="center",
        bbox_to_anchor=(x_center, y_top),
        bbox_transform=ax.transAxes,
        ncol=2,
        frameon=False,
        fontsize=4.15,
        handlelength=1.0,
        handletextpad=0.28,
        columnspacing=0.45,
        borderaxespad=0,
    )
    ax.add_artist(top_leg)
    ax.legend(
        handles=[handles[2]],
        loc="center",
        bbox_to_anchor=(x_center, y_bottom),
        bbox_transform=ax.transAxes,
        ncol=1,
        frameon=False,
        fontsize=4.15,
        handlelength=1.0,
        handletextpad=0.28,
        borderaxespad=0,
    )


def boxplot_dynamic_score(ax: plt.Axes, prof: pd.DataFrame) -> None:
    pos = ax.get_position()
    width = pos.width * 0.74
    ax.set_position([pos.x0, pos.y0, width, pos.height])

    groups = ["Control", "ELS"]
    y_min, y_max = -1.05, 1.50
    els_resilient_vals = prof.loc[(prof["group"] == "ELS") & (prof["dynamics_score"] < 0), "dynamics_score"].dropna()
    if not els_resilient_vals.empty:
        rect_pad = (y_max - y_min) * 0.018
        rect_y0 = max(y_min, float(els_resilient_vals.min()) - rect_pad)
        rect_y1 = 0.0
        ax.add_patch(
            mpatches.Rectangle(
                (1 - 0.18, rect_y0),
                0.36,
                rect_y1 - rect_y0,
                facecolor=(*mcolors.to_rgb(RESILIENT), 0.10),
                edgecolor=RESILIENT,
                linewidth=0.85,
                linestyle=(0, (2.2, 1.5)),
                zorder=1.5,
            )
        )
    for i, group in enumerate(groups):
        sub = prof[prof["group"] == group].dropna(subset=["dynamics_score"]).copy()
        vals = sub["dynamics_score"].to_numpy()
        color = PALETTE[group]
        ax.boxplot(
            vals,
            positions=[i],
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": (*mcolors.to_rgb(color), 0.15), "edgecolor": color, "linewidth": 1.15},
            whiskerprops={"color": color, "linewidth": 1.15},
            capprops={"color": color, "linewidth": 1.15},
            medianprops={"color": color, "linewidth": 1.35},
        )
        rng = np.random.default_rng(42 + i)
        jitter = rng.normal(i, 0.022, size=len(vals))
        ax.scatter(jitter, vals, s=8.0, color=color, alpha=0.78, linewidth=0, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(groups)
    ax.set_ylabel("log (Euclidean score)", fontsize=6.0, labelpad=2)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(-1.0, 1.51, 0.5))
    style_axis(ax, labelsize=5.8)
    ax.tick_params(axis="x", labelsize=7.5, pad=2)
    sig_bracket(ax, 0, 1, 1.27, h=0.05)
    tag(ax, "B", x=-0.18, y=1.11)


def threshold_audit(prof: pd.DataFrame) -> pd.DataFrame:
    els = prof[prof["group"] == "ELS"].copy()
    zero_ids = set(els.loc[els["dynamics_score"] < 0, "animal"])
    mean_threshold = float(els["dynamics_score"].mean())
    median_threshold = float(els["dynamics_score"].median())
    mean_ids = set(els.loc[els["dynamics_score"] < mean_threshold, "animal"])
    median_ids = set(els.loc[els["dynamics_score"] < median_threshold, "animal"])
    return pd.DataFrame(
        [
            {
                "metric": "Euclidean",
                "els_n": int(len(els)),
                "zero_threshold": 0.0,
                "zero_n": int(len(zero_ids)),
                "zero_matches_green": zero_ids == ELS_RESILIENT,
                "els_mean_score": mean_threshold,
                "mean_threshold_n": int(len(mean_ids)),
                "mean_added_vs_zero": ", ".join(sorted(mean_ids - zero_ids)),
                "mean_removed_vs_zero": ", ".join(sorted(zero_ids - mean_ids)),
                "els_median_score": median_threshold,
                "median_threshold_n": int(len(median_ids)),
                "median_added_vs_zero": ", ".join(sorted(median_ids - zero_ids)),
                "median_removed_vs_zero": ", ".join(sorted(zero_ids - median_ids)),
            }
        ]
    )


def plot_mds(ax: plt.Axes, prof: pd.DataFrame, loocv: float) -> None:
    pos = ax.get_position()
    fig = ax.figure
    target_width = min(pos.width, pos.height * fig.get_figheight() / fig.get_figwidth() * 1.18)
    ax.set_position([pos.x0, pos.y0, target_width, pos.height])

    x = prof["mds1"].to_numpy()
    y = prof["mds2"].to_numpy()
    score = prof["dynamics_score"].to_numpy()
    margin = 0.12 * max(np.ptp(x), np.ptp(y))
    gx, gy = np.mgrid[x.min() - margin : x.max() + margin : 160j, y.min() - margin : y.max() + margin : 160j]
    gz = griddata(np.column_stack([x, y]), score, (gx, gy), method="cubic")
    nn = griddata(np.column_stack([x, y]), score, (gx, gy), method="nearest")
    gz = np.where(np.isnan(gz), nn, gz)
    cmap = mcolors.LinearSegmentedColormap.from_list("dyn", [CONTROL, "#FFFFFF", ELS])
    vabs = 1.2
    levels = np.arange(-vabs, vabs + 0.001, 0.15)
    gz_fill = np.clip(gz, -vabs + 1e-6, vabs - 1e-6)
    cf = ax.contourf(gx, gy, gz_fill, levels=levels, cmap=cmap, vmin=-vabs, vmax=vabs, alpha=0.62, extend="neither")
    cs = ax.contour(gx, gy, gz, levels=levels, colors=AXIS, linewidths=0.32, alpha=0.82)
    ax.clabel(cs, levels[::2], inline=True, fontsize=3.9, fmt="%.2f", colors=AXIS)
    for group in ["Control", "ELS"]:
        sub = prof[prof["group"] == group]
        ax.scatter(sub["mds1"], sub["mds2"], s=24, color=PALETTE[group], edgecolor=AXIS, linewidth=0.35, alpha=0.88, label=group, zorder=3)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=PALETTE[group],
            markeredgecolor="none",
            markeredgewidth=0.0,
            markersize=4.4,
            label=group,
        )
        for group in ["Control", "ELS"]
    ]
    leg = ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        fancybox=False,
        fontsize=5.7,
        borderaxespad=0.22,
        handletextpad=0.62,
        borderpad=0.18,
        handlelength=0.55,
    )
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("white")
    cax = ax.inset_axes([1.04, 0.0, 0.052, 1.0])
    cbar = plt.colorbar(cf, cax=cax)
    cbar.set_label("Dynamic Similarity Score", fontsize=5.2, color=AXIS)
    cbar.set_ticks(np.arange(-1.2, 1.21, 0.3))
    cbar.ax.tick_params(labelsize=4.6, width=0.35, length=1.5, colors=AXIS)
    cbar.outline.set_linewidth(0.35)
    ax.text(1.035, -0.085, f"LOOCV Acc:\n{loocv*100:.1f}%", transform=ax.transAxes, ha="left", va="top", fontsize=4.8, color=AXIS, clip_on=False, bbox=dict(facecolor="white", edgecolor="#BDBDBD", linewidth=0.35, pad=1.5))
    ax.set_xlim(-0.45, 0.70)
    ax.set_ylim(-0.50, 0.55)
    ax.set_xlabel("MDS Dimension 1", fontsize=6.0)
    ax.set_ylabel("MDS Dimension 2", fontsize=6.0)
    ax.set_title("MDS Plot (Euclidean Distance)", fontsize=6.3, pad=4)
    style_axis(ax, labelsize=5.2)
    tag(ax, "A", x=-0.14, y=1.10)


def plot_frequency(ax: plt.Axes, freq: pd.DataFrame) -> None:
    summary = freq.groupby(["group_ext", "cluster"])["seconds"].agg(["mean", "sem"]).reset_index()
    x = np.arange(len(ORDER))
    width = 0.22
    offsets = [-width, 0, width]
    for off, group in zip(offsets, GROUP_ORDER):
        vals = summary[summary["group_ext"] == group].set_index("cluster").reindex(ORDER)
        ax.bar(x + off, vals["mean"], width=width, color=PALETTE[group], edgecolor="white", linewidth=0.25, label=group, zorder=2)
        ax.errorbar(x + off, vals["mean"], yerr=vals["sem"], fmt="none", ecolor=AXIS, elinewidth=0.55, capsize=1.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, fontsize=6.0)
    ax.set_ylabel("Frequency (s)", fontsize=6.2)
    ax.set_ylim(0, 300)
    ax.set_yticks(np.arange(0, 301, 50))
    ax.legend(handles=legend_boxes(GROUP_ORDER), loc="upper right", ncol=3, frameon=False, fontsize=5.5, handlelength=1.0, handletextpad=0.35, columnspacing=0.75)
    style_axis(ax)
    tag(ax, "C", x=-0.08, y=1.10)
    bracket_gap = 0.045
    for idx in [0, 1, 3]:
        y = float(summary[summary["cluster"] == ORDER[idx]]["mean"].max() + summary[summary["cluster"] == ORDER[idx]]["sem"].max() + 8)
        right_edge = idx - bracket_gap if idx in [0, 3] else idx
        sig_bracket(ax, idx - width, right_edge, y, h=5)
        if idx in [0, 3]:
            sig_bracket(ax, idx + bracket_gap, idx + width, y, h=5)


def plot_time(ax: plt.Axes, summary: pd.DataFrame, cluster: str, letter: str, ylim: tuple[float, float], yticks: list[float], star_x: float | None = None, legend_corner: str = "upper_right") -> None:
    add_epochs(ax)
    data = summary[summary["cluster"] == cluster]
    for group in GROUP_ORDER:
        sub = data[data["group_ext"] == group].sort_values("time_min")
        if sub.empty:
            continue
        ax.plot(sub["time_min"], sub["mean"], color=PALETTE[group], linewidth=0.85, marker="o", markersize=2.0, label=group, zorder=3)
        ax.fill_between(sub["time_min"].to_numpy(), (sub["mean"] - sub["sem"]).to_numpy(), (sub["mean"] + sub["sem"]).to_numpy(), color=PALETTE[group], alpha=0.18, linewidth=0, zorder=2)
    if star_x is not None:
        ax.text(star_x, 0.992, "*", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=11, color=AXIS, fontweight="bold")
    ax.set_xlim(0.75, 7.25)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xticks(np.arange(1, 8))
    ax.set_title(cluster, fontsize=7, pad=5, y=1.075)
    ax.set_xlabel("Time (minutes)", fontsize=5.8, labelpad=1)
    ax.set_ylabel("% of time in cluster", fontsize=5.8, labelpad=2)
    style_axis(ax, labelsize=5.4)
    time_panel_legend(ax, legend_corner)
    tag(ax, letter, x=-0.14, y=1.10)


def export_source_data(prof: pd.DataFrame, freq: pd.DataFrame, output_dir: Path) -> None:
    """Export Figure 5 source data: behavioral dynamics, resilience classification, and frequencies."""
    rows = []

    # Add dynamics score summary (from prof dataframe)
    if "dynamics_score" in prof.columns and "group" in prof.columns:
        for group in prof["group"].unique():
            group_data = prof[prof["group"] == group]["dynamics_score"]
            if len(group_data) > 0:
                rows.append({
                    "metric": "Behavioral dynamics score",
                    "group": group,
                    "mean": group_data.mean(),
                    "std": group_data.std(),
                    "sem": group_data.sem(),
                    "n": len(group_data),
                })

    # Add behavior frequency summary (from freq dataframe)
    # freq has columns: Animal, group_ext, cluster, seconds
    if not freq.empty and "cluster" in freq.columns and "group_ext" in freq.columns:
        for cluster in freq["cluster"].unique():
            for group in ["Control", "ELS", "ELS resilient"]:
                group_freq = freq[(freq["cluster"] == cluster) & (freq["group_ext"] == group)]
                if not group_freq.empty and "seconds" in group_freq.columns:
                    freq_vals = group_freq["seconds"]
                    if len(freq_vals) > 0:
                        rows.append({
                            "metric": f"{cluster} frequency",
                            "group": group,
                            "mean_seconds": freq_vals.mean(),
                            "std_seconds": freq_vals.std() if len(freq_vals) > 1 else 0,
                            "sem_seconds": freq_vals.sem() if len(freq_vals) > 1 else 0,
                            "n": len(freq_vals),
                        })

    df = pd.DataFrame(rows)
    if not df.empty:
        output_csv = output_dir / "source_data_figure5.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")
    else:
        print("Warning: No source data generated for Figure 5")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cluster_time = load_cluster_time()
    time_summary = summarize_time(cluster_time)
    freq = frequency_points(cluster_time)
    prof, _, loocv = mds_profiles_from_updated_results()
    prof.assign(resilient_by_zero=(prof["group"] == "ELS") & (prof["dynamics_score"] < 0)).to_csv(
        OUTPUT_DIR / "figure_5_dynamics_scores.csv", index=False
    )
    threshold_audit(prof).to_csv(OUTPUT_DIR / "figure_5_threshold_audit.csv", index=False)

    fig = plt.figure(figsize=(8.27, 11.69), dpi=300, facecolor="white")
    gs = fig.add_gridspec(
        nrows=4,
        ncols=14,
        left=0.075,
        right=0.955,
        top=0.955,
        bottom=0.20,
        wspace=0.28,
        hspace=0.42,
        height_ratios=[1.18, 1.0, 1.0, 1.0],
    )
    axA = fig.add_subplot(gs[0, 0:9])
    axB = fig.add_subplot(gs[0, 10:14])
    axC = fig.add_subplot(gs[1, 0:9])
    axD = fig.add_subplot(gs[1, 10:14])
    axE = fig.add_subplot(gs[2, 0:4])
    axF = fig.add_subplot(gs[2, 5:9])
    axG = fig.add_subplot(gs[2, 10:14])
    axH = fig.add_subplot(gs[3, 0:4])
    axI = fig.add_subplot(gs[3, 5:9])
    axJ = fig.add_subplot(gs[3, 10:14])

    plot_mds(axA, prof, loocv)
    boxplot_dynamic_score(axB, prof)
    plot_frequency(axC, freq)
    plot_time(axD, time_summary, "Freeze", "D", (0, 70), list(range(0, 71, 10)), star_x=4.0, legend_corner="lower_right")
    # Sniff: p=0.047 but BH_FDR=0.109 -> not significant after correction, no star.
    plot_time(axE, time_summary, "Sniff", "E", (0, 25), list(range(0, 26, 5)))
    plot_time(axF, time_summary, "Groom", "F", (0, 0.5), [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plot_time(axG, time_summary, "Turn", "G", (0, 70), list(range(0, 71, 10)), star_x=4.0, legend_corner="lower_right")
    # Locomotion: no corresponding contrast reported in the manuscript, no star.
    plot_time(axH, time_summary, "Locomotion", "H", (0, 14), list(range(0, 15, 2)))
    plot_time(axI, time_summary, "Climb", "I", (0, 14), list(range(0, 15, 2)))
    plot_time(axJ, time_summary, "Jump", "J", (0, 4), [0, 1, 2, 3, 4])

    pdf = OUTPUT_DIR / "figure_5_resilience_dynamics.pdf"
    png = OUTPUT_DIR / "figure_5_resilience_dynamics.png"
    svg = OUTPUT_DIR / "figure_5_resilience_dynamics.svg"
    fig.savefig(pdf)
    fig.savefig(svg)
    fig.savefig(png, dpi=300)
    LEGACY_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LEGACY_FIGURES_DIR / "figure5.pdf")
    fig.savefig(LEGACY_FIGURES_DIR / "figure5.svg")
    fig.savefig(LEGACY_FIGURES_DIR / "figure5.png", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf}")
    print(f"Saved {svg}")
    print(f"Saved {png}")
    print(f"Saved {LEGACY_FIGURES_DIR / 'figure5.pdf'}")

    # Export source data
    print("Computing Figure 5 source statistics...")
    export_source_data(prof, freq, OUTPUT_DIR)


if __name__ == "__main__":
    main()
