# SPDX-License-Identifier: MIT
"""Hierarchical clustering of animals by behavioural profile.

Groups animals by how similar their behaviour is, without using the condition
label, and then shows where Control, vulnerable ELS and resilient ELS animals
fall in the resulting tree. It answers a question the MDS plot only hints at:
does an unsupervised grouping of the mice recover the experimental design, and
do the resilient animals sit with the controls?

Two views are written:

  * a dendrogram with each leaf coloured by profile, plus the litter of origin,
    so family clustering is visible at the same time;
  * a clustered heatmap of the 30 s behavioural profile, with the same tree.

    python scripts/generate_animal_dendrogram.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "supplementary_media" / "clustering"
STATS = ROOT / "statistics"

CLUSTERS = ["Freeze", "Sniff", "Groom", "Turn", "Locomotion", "Climb", "Jump"]
PROFILE_COLORS = {"Control": "#F9C74F", "ELS vulnerable": "#C37BA0", "ELS resilient": "#5B9AAA"}
AXIS = "#4D4D4D"
RESILIENT = {
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


def behavioural_profiles() -> tuple[pd.DataFrame, np.ndarray]:
    """Per-animal 30 s time-share profile, the Figure 5 feature space."""
    timecourse = pd.read_csv(ROOT / "data/processed/cluster_timecourse_per_animal.csv")
    wide = timecourse.pivot_table(
        index="animal_id",
        columns=["cluster", "time_bin"],
        values="pct",
        aggfunc="mean",
        fill_value=0.0,
    )
    wide = wide.sort_index(axis=1)
    freq = pd.read_csv(ROOT / "data/processed/cluster_frequency_per_animal.csv")
    meta = freq[["animal_id", "group", "experiment"]].drop_duplicates("animal_id")
    meta = meta.set_index("animal_id").loc[wide.index].reset_index()
    meta["animal"] = meta["animal_id"].astype(str)
    meta["litter"] = meta["animal"].str.split(".").str[0]
    meta["profile"] = np.where(
        meta["group"] == "Control",
        "Control",
        np.where(meta["animal"].isin(RESILIENT), "ELS resilient", "ELS vulnerable"),
    )
    return meta, wide.to_numpy(dtype=float)


def enrichment(labels: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster in sorted(set(labels)):
        member = meta[labels == cluster]
        counts = member["profile"].value_counts()
        rows.append(
            {
                "cluster": cluster,
                "n": len(member),
                "Control": int(counts.get("Control", 0)),
                "ELS vulnerable": int(counts.get("ELS vulnerable", 0)),
                "ELS resilient": int(counts.get("ELS resilient", 0)),
                "n_litters": member["litter"].nunique(),
                "dominant": counts.idxmax() if len(counts) else "",
                "purity_%": round(counts.max() / len(member) * 100, 1) if len(member) else 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="ward")
    parser.add_argument("--k", type=int, default=3, help="clusters to cut the tree into")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    meta, features = behavioural_profiles()
    distances = pdist(features, metric="euclidean")
    tree = linkage(distances, method=args.method)
    labels = fcluster(tree, t=args.k, criterion="maxclust")
    meta["tree_cluster"] = labels

    # ---- dendrogram ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=200, facecolor="white")
    dend = dendrogram(
        tree,
        labels=meta["animal"].to_numpy(),
        ax=ax,
        color_threshold=None,
        above_threshold_color="#B0B0B0",
        leaf_font_size=6.5,
    )
    profile_of = dict(zip(meta["animal"], meta["profile"]))
    ax.set_ylabel("Ward linkage distance", fontsize=10, color=AXIS)
    ax.set_title(
        "Animals clustered by 30 s behavioural profile (condition not used in the clustering)",
        fontsize=11,
        color=AXIS,
        pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelsize=8)
    ax.legend(
        handles=[plt.Line2D([], [], color=c, lw=6, label=k) for k, c in PROFILE_COLORS.items()],
        frameon=False,
        fontsize=9,
        loc="upper right",
    )
    fig.tight_layout()
    # Colour the leaves after the layout pass: tight_layout regenerates the tick
    # artists, so colours applied before it are discarded.
    for tick in ax.get_xticklabels():
        tick.set_color(PROFILE_COLORS[profile_of[tick.get_text()]])
        tick.set_rotation(90)
        tick.set_fontweight("bold")
    fig.savefig(OUT / "animal_dendrogram.png", facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "animal_dendrogram.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # ---- clustered heatmap ----------------------------------------------
    order = dend["leaves"]
    share = pd.read_csv(ROOT / "data/processed/cluster_frequency_per_animal.csv")
    share = share.pivot_table(
        index="animal_id",
        columns="cluster",
        values="frequency_seconds",
        aggfunc="sum",
        fill_value=0.0,
    )
    share = share.div(share.sum(axis=1), axis=0) * 100
    share = share.reindex(meta["animal_id"]).reindex(columns=CLUSTERS).fillna(0.0)

    fig, ax = plt.subplots(figsize=(7.5, 11), dpi=200, facecolor="white")
    image = ax.imshow(
        share.to_numpy()[order], aspect="auto", cmap="magma_r", interpolation="nearest"
    )
    ax.set_xticks(range(len(CLUSTERS)))
    ax.set_xticklabels(CLUSTERS, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([meta["animal"].iloc[i] for i in order], fontsize=5.5)
    for tick, i in zip(ax.get_yticklabels(), order):
        tick.set_color(PROFILE_COLORS[meta["profile"].iloc[i]])
    ax.set_title(
        "Time share per behaviour, ordered by the dendrogram", fontsize=11, color=AXIS, pad=10
    )
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("% of session", fontsize=9, color=AXIS)
    fig.tight_layout()
    fig.savefig(OUT / "animal_profile_heatmap.png", facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "animal_profile_heatmap.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # ---- tables ----------------------------------------------------------
    table = enrichment(labels, meta)
    table.to_csv(STATS / "animal_dendrogram_clusters.csv", index=False)
    meta[["animal", "group", "profile", "litter", "experiment", "tree_cluster"]].to_csv(
        OUT / "animal_cluster_assignments.csv", index=False
    )

    print(f"{len(meta)} animals, {args.k} clusters ({args.method} linkage)\n")
    print(table.to_string(index=False))
    same_litter = squareform(distances)
    litter = meta["litter"].to_numpy()
    within = np.array(
        [
            same_litter[i, j]
            for i in range(len(litter))
            for j in range(i + 1, len(litter))
            if litter[i] == litter[j]
        ]
    )
    between = np.array(
        [
            same_litter[i, j]
            for i in range(len(litter))
            for j in range(i + 1, len(litter))
            if litter[i] != litter[j]
        ]
    )
    print(f"\nprofile distance within litters : {within.mean():.2f} (n={len(within)} pairs)")
    print(f"profile distance between litters: {between.mean():.2f} (n={len(between)} pairs)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
