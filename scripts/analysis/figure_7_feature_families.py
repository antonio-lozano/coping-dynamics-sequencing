"""Feature-family construction and models used by manuscript Figure 7."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figure_source_data"
RIDGE_ALPHA = 1.0

AXIS = "#4D4D4D"
TEXT = "#2B2B2B"
FREEZING = "#C671A0"
MOTIF = "#6398A4"
TRANSITION = "#9ECADA"
DIVERSITY = "#E6C23A"
INTEGRATED = "#BCD548"
BOUT = "#D98427"
INTEGRATED_FREEZE = "#D95D5D"

DISPLAY_ORDER = ["Freeze", "Sniff", "Groom", "Turn", "Locomotion", "Climb", "Jump"]
CLUSTER_MAP = {
    "Freeze": [0, 28],
    "Sniff": [18, 20],
    "Groom": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climb": [111],
    "Jump": [23, 29, 30, 34],
}
SYLLABLE_TO_CLUSTER = {
    syllable: cluster for cluster, syllables in CLUSTER_MAP.items() for syllable in syllables
}

BLOCK_COLORS = {
    "freezing_only": FREEZING,
    "frequency": MOTIF,
    "transition": TRANSITION,
    "transition_pairs": TRANSITION,
    "diversity": DIVERSITY,
    "bout_duration": BOUT,
    "behavior_integrated": INTEGRATED,
    "behavior_integrated_with_pairs": INTEGRATED,
    "behavior_plus_freezing": INTEGRATED_FREEZE,
}

BLOCK_LABELS = {
    "freezing_only": "Freeze",
    "frequency": "Motif freq.",
    "transition": "Transition summary",
    "transition_pairs": "Transition pairs",
    "diversity": "Diversity",
    "bout_duration": "Bout duration",
    "behavior_integrated": "All behavior",
    "behavior_integrated_with_pairs": "All behavior + pairs",
    "behavior_plus_freezing": "Behavior + Freeze",
}

COHORT_LABELS = {
    "Exp1": "Sanguino Gómez\n& Krugers",
    "Exp3": "Sanguino Gómez\net al.",
}
CLUSTER_COLOR_LOOKUP = {
    "Freeze": FREEZING,
    "Sniff": MOTIF,
    "Groom": TRANSITION,
    "Turn": DIVERSITY,
    "Locomotion": INTEGRATED,
    "Climb": BOUT,
    "Jump": INTEGRATED_FREEZE,
}


def load_labels() -> pd.DataFrame:
    prof = pd.read_csv(REPO / "data/processed/figure5_dynamics_scores.csv")
    freq = pd.read_csv(REPO / "data/processed/cluster_frequency_per_animal.csv")
    experiments = freq[["animal_id", "experiment"]].drop_duplicates()
    labels = (
        prof.loc[prof["group"] == "ELS", ["animal", "group", "dynamics_score", "resilient_by_zero"]]
        .rename(columns={"animal": "animal_id"})
        .merge(experiments, on="animal_id", how="left")
    )
    labels["target"] = labels["resilient_by_zero"].astype(int)
    labels["profile"] = np.where(labels["target"].eq(1), "resilient", "vulnerable")
    return labels.sort_values(["experiment", "animal_id"]).reset_index(drop=True)


def frequency_features() -> tuple[pd.DataFrame, dict[str, str]]:
    freq = pd.read_csv(REPO / "data/processed/cluster_frequency_per_animal.csv")
    wide = freq.pivot_table(
        index="animal_id",
        columns="cluster",
        values="frequency_seconds",
        aggfunc="sum",
        fill_value=0,
    )
    totals = wide.sum(axis=1).replace(0, np.nan)
    wide = wide.div(totals, axis=0).mul(100.0).reset_index()
    rename = {c: f"frequency__{c}" for c in wide.columns if c != "animal_id"}
    return wide.rename(columns=rename), {v: "frequency" for v in rename.values()}


def freezing_features() -> tuple[pd.DataFrame, dict[str, str]]:
    freezing = pd.read_csv(
        REPO / "data/raw/freezing_predictions_light.csv.gz", usecols=["animal_id", "freezing"]
    )
    out = freezing.groupby("animal_id", as_index=False)["freezing"].mean()
    out["freezing__supervised_pct"] = out["freezing"] * 100.0
    return out.drop(columns=["freezing"]), {"freezing__supervised_pct": "freezing"}


def transition_features() -> tuple[pd.DataFrame, dict[str, str]]:
    trans = pd.read_csv(REPO / "statistics/fig4_transition_per_animal.csv").rename(
        columns={"Animal": "animal_id"}
    )
    cols = ["lz", "recurrence", "determinism", "markov"]
    out = trans[["animal_id", *cols]].copy()
    rename = {c: f"transition__{c}" for c in cols}
    return out.rename(columns=rename), {v: "transition" for v in rename.values()}


def transition_pair_features() -> tuple[pd.DataFrame, dict[str, str], dict[str, tuple[str, str]]]:
    raw = pd.read_csv(REPO / "data/raw/syllable_usage_per_timebin_250ms.csv")
    raw = raw.rename(columns={"Time Bin": "time_bin", "Condition": "group"})
    raw = raw[raw["group"].isin(["Control", "ELS"])].copy()
    raw["Syllable"] = pd.to_numeric(raw["Syllable"], errors="coerce").astype(int)
    raw["cluster"] = raw["Syllable"].map(SYLLABLE_TO_CLUSTER)
    raw = raw.dropna(subset=["cluster"])
    idx = raw.groupby(["Animal", "time_bin"])["Percentage"].idxmax()
    dominant = raw.loc[idx].sort_values(["Animal", "time_bin"])

    rows = []
    transition_feature_map: dict[str, tuple[str, str]] = {}
    for animal, sub in dominant.groupby("Animal", sort=False):
        seq = sub["cluster"].tolist()
        if not seq:
            continue
        compressed = [seq[0]]
        for cluster in seq[1:]:
            if cluster != compressed[-1]:
                compressed.append(cluster)
        counts = {(a, b): 0.0 for a in DISPLAY_ORDER for b in DISPLAY_ORDER if a != b}
        outgoing = {a: 0.0 for a in DISPLAY_ORDER}
        for a, b in zip(compressed[:-1], compressed[1:]):
            if a in outgoing and b in DISPLAY_ORDER and a != b:
                counts[(a, b)] += 1.0
                outgoing[a] += 1.0
        row = {"animal_id": float(animal)}
        for a in DISPLAY_ORDER:
            for b in DISPLAY_ORDER:
                if a == b:
                    continue
                feature = f"transition_pair__{a}_to_{b}"
                row[feature] = counts[(a, b)] / outgoing[a] if outgoing[a] else 0.0
                transition_feature_map[feature] = (a, b)
        rows.append(row)
    out = pd.DataFrame(rows)
    return (
        out,
        {c: "transition_pairs" for c in out.columns if c != "animal_id"},
        transition_feature_map,
    )


def diversity_features() -> tuple[pd.DataFrame, dict[str, str]]:
    div = pd.read_csv(REPO / "statistics/fig4_diversity_per_animal.csv").rename(
        columns={"Animal": "animal_id"}
    )
    cols = ["simpson", "shannon", "evenness", "cui"]
    out = div[["animal_id", *cols]].copy()
    rename = {c: f"diversity__{c}" for c in cols}
    return out.rename(columns=rename), {v: "diversity" for v in rename.values()}


def bout_features() -> tuple[pd.DataFrame, dict[str, str]]:
    overall = pd.read_csv(REPO / "statistics/fig4_bout_overall_per_animal.csv").rename(
        columns={"Animal": "animal_id"}
    )
    overall = overall[["animal_id", "bout_mean"]].rename(
        columns={"bout_mean": "bout__overall_mean"}
    )
    cluster = pd.read_csv(REPO / "statistics/fig4_bout_cluster_per_animal.csv").rename(
        columns={"Animal": "animal_id"}
    )
    wide = cluster.pivot_table(
        index="animal_id", columns="cluster", values="bout_duration", aggfunc="mean", fill_value=0
    )
    wide = wide.reset_index()
    rename = {c: f"bout__{c}" for c in wide.columns if c != "animal_id"}
    out = overall.merge(wide.rename(columns=rename), on="animal_id", how="outer")
    block_map = {"bout__overall_mean": "bout_duration"}
    block_map.update({v: "bout_duration" for v in rename.values()})
    return out, block_map


def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="animal_id", how="inner")
    return out


def fit_ridge(train: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = train[cols].to_numpy(dtype=float)
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


def score(
    test: pd.DataFrame, cols: list[str], coef: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    xs = (test[cols].to_numpy(dtype=float) - mean) / scale
    return coef[0] + xs @ coef[1:]


def metric_row(
    y: np.ndarray,
    scores: np.ndarray,
    pred: np.ndarray,
    model_name: str,
    train: str,
    test: str,
    n_features: int,
) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "train_experiment": train,
        "test_experiment": test,
        "n_test": len(y),
        "n_features": n_features,
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity_resilient": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity_vulnerable": tn / (tn + fp) if (tn + fp) else np.nan,
        "tn_vulnerable": int(tn),
        "fp_vulnerable_as_resilient": int(fp),
        "fn_resilient_as_vulnerable": int(fn),
        "tp_resilient": int(tp),
    }


def evaluate_loocv(
    data: pd.DataFrame, cols: list[str], model_name: str
) -> tuple[dict[str, object], pd.DataFrame]:
    scores = np.zeros(len(data))
    preds = np.zeros(len(data), dtype=int)
    for i in range(len(data)):
        train = data.drop(data.index[i])
        test = data.iloc[[i]]
        coef, mean, scale = fit_ridge(train, cols)
        s = score(test, cols, coef, mean, scale)[0]
        scores[i] = s
        preds[i] = int(s >= 0)
    y = data["target"].to_numpy()
    row = metric_row(y, scores, preds, model_name, "LOOCV", "LOOCV", len(cols))
    pred_df = data[["animal_id", "experiment", "profile", "target", "dynamics_score"]].copy()
    pred_df["model"] = model_name
    pred_df["resilience_score"] = scores
    pred_df["predicted_profile"] = np.where(preds == 1, "resilient", "vulnerable")
    return row, pred_df


def evaluate_cross(
    data: pd.DataFrame, cols: list[str], model_name: str
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    rows = []
    pred_frames = []
    for train_exp, test_exp in [(1, 3), (3, 1)]:
        train = data[data["experiment"].eq(train_exp)]
        test = data[data["experiment"].eq(test_exp)]
        coef, mean, scale = fit_ridge(train, cols)
        scores = score(test, cols, coef, mean, scale)
        preds = (scores >= 0).astype(int)
        rows.append(
            metric_row(
                test["target"].to_numpy(),
                scores,
                preds,
                model_name,
                f"Exp{train_exp}",
                f"Exp{test_exp}",
                len(cols),
            )
        )
        pred = test[["animal_id", "experiment", "profile", "target", "dynamics_score"]].copy()
        pred["model"] = model_name
        pred["train_experiment"] = f"Exp{train_exp}"
        pred["test_experiment"] = f"Exp{test_exp}"
        pred["resilience_score"] = scores
        pred["predicted_profile"] = np.where(preds == 1, "resilient", "vulnerable")
        pred_frames.append(pred)
    return rows, pd.concat(pred_frames, ignore_index=True)


def coefficients(data: pd.DataFrame, cols: list[str], block_map: dict[str, str]) -> pd.DataFrame:
    coef, _, _ = fit_ridge(data, cols)
    out = pd.DataFrame(
        {
            "feature": cols,
            "block": [block_map[c] for c in cols],
            "standardized_linear_weight": coef[1:],
        }
    )
    out["abs_weight"] = out["standardized_linear_weight"].abs()
    return out.sort_values("abs_weight", ascending=False)


def clean_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.7, length=2.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.7)


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.15,
        1.08,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=TEXT,
    )


def short_feature_name(feature: str) -> str:
    label = (
        feature.replace("frequency__", "Freq: ")
        .replace("transition__", "Trans: ")
        .replace("transition_pair__", "Pair: ")
        .replace("diversity__", "Div: ")
        .replace("bout__", "Bout: ")
        .replace("freezing__", "Freeze: ")
        .replace("_", " ")
    )
    label = (
        label.replace("_to_", " -> ")
        .replace("Freezing", "Freeze")
        .replace("Grooming", "Groom")
        .replace("Sniffing", "Sniff")
        .replace("Climbing", "Climb")
    )
    return label


def feature_behavior_links(
    feature: str, transition_feature_map: dict[str, tuple[str, str]]
) -> list[str]:
    if feature.startswith("frequency__"):
        return [feature.split("__", 1)[1]]
    if feature.startswith("bout__"):
        name = feature.split("__", 1)[1]
        return [] if name == "overall_mean" else [short_feature_name(name).replace("Bout: ", "")]
    if feature.startswith("transition_pair__"):
        return list(transition_feature_map.get(feature, ()))
    return []


def behavior_category_importance(
    coefs: pd.DataFrame, transition_feature_map: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    rows = []
    for _, row in coefs.iterrows():
        links = feature_behavior_links(str(row["feature"]), transition_feature_map)
        if not links:
            continue
        share = float(row["standardized_linear_weight"]) / len(links)
        for behavior in links:
            rows.append(
                {"behavior": behavior, "signed_weight_share": share, "abs_weight_share": abs(share)}
            )
    out = (
        pd.DataFrame(rows)
        .groupby("behavior", as_index=False)
        .agg(
            signed_weight=("signed_weight_share", "sum"),
            total_abs_weight=("abs_weight_share", "sum"),
        )
    )
    total = out["total_abs_weight"].sum()
    out["percent_of_behavior_linked_weight"] = (
        out["total_abs_weight"] / total * 100.0 if total else np.nan
    )
    return out.sort_values("total_abs_weight", ascending=False)


def transition_pair_importance(
    coefs: pd.DataFrame, transition_feature_map: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    sub = coefs[coefs["feature"].str.startswith("transition_pair__")].copy()
    if sub.empty:
        return sub
    sub["from_behavior"] = sub["feature"].map(lambda f: transition_feature_map[f][0])
    sub["to_behavior"] = sub["feature"].map(lambda f: transition_feature_map[f][1])
    sub["transition"] = sub["from_behavior"] + " -> " + sub["to_behavior"]
    return sub.sort_values("abs_weight", ascending=False)


def plot_figure(
    loocv_metrics: pd.DataFrame,
    cross_metrics: pd.DataFrame,
    ablation: pd.DataFrame,
    coefs: pd.DataFrame,
    behavior_importance: pd.DataFrame,
    pair_importance: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.3,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.8,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": AXIS,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(10.2, 5.7))
    gs = fig.add_gridspec(
        2, 3, left=0.065, right=0.99, bottom=0.105, top=0.93, wspace=0.43, hspace=0.50
    )
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = [
        fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)
    ]

    order = [
        "freezing_only",
        "frequency",
        "transition",
        "transition_pairs",
        "diversity",
        "bout_duration",
        "behavior_integrated",
        "behavior_integrated_with_pairs",
        "behavior_plus_freezing",
    ]
    sub = loocv_metrics.set_index("model").loc[order].reset_index()
    x = np.arange(len(sub))
    colors = [BLOCK_COLORS[m] for m in sub["model"]]
    ax_a.bar(x, sub["roc_auc"], color=colors, edgecolor="white", linewidth=0.4)
    ax_a.axhline(0.5, color=AXIS, linewidth=0.65, linestyle=(0, (1.5, 1.5)))
    for i, row in sub.iterrows():
        ax_a.text(
            i,
            min(row["roc_auc"] + 0.025, 1.03),
            f"{row['roc_auc']:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.6,
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([BLOCK_LABELS[m] for m in sub["model"]], rotation=28, ha="right")
    ax_a.set_ylim(0, 1.05)
    ax_a.set_ylabel("LOOCV ROC AUC")
    ax_a.set_title("Feature-family comparison", pad=4)
    clean_axis(ax_a)
    panel_label(ax_a, "A")

    directions = [("Exp1", "Exp3"), ("Exp3", "Exp1")]
    models = [
        "freezing_only",
        "frequency",
        "behavior_integrated_with_pairs",
        "behavior_plus_freezing",
    ]
    width = 0.19
    base_x = np.arange(len(directions))
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(models))
    for offset, model_name in zip(offsets, models):
        vals = []
        for train, test in directions:
            vals.append(
                cross_metrics.loc[
                    cross_metrics["model"].eq(model_name)
                    & cross_metrics["train_experiment"].eq(train)
                    & cross_metrics["test_experiment"].eq(test),
                    "roc_auc",
                ].iloc[0]
            )
        ax_b.bar(
            base_x + offset,
            vals,
            width=width,
            color=BLOCK_COLORS[model_name],
            edgecolor="white",
            linewidth=0.35,
            label=BLOCK_LABELS[model_name],
        )
    ax_b.axhline(0.5, color=AXIS, linewidth=0.65, linestyle=(0, (1.5, 1.5)))
    ax_b.set_xticks(base_x)
    ax_b.set_xticklabels(
        [f"{COHORT_LABELS[train]} ->\n{COHORT_LABELS[test]}" for train, test in directions]
    )
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("Held-out ROC AUC")
    ax_b.set_title("Cross-cohort transfer", pad=4)
    ax_b.legend(frameon=False, loc="lower right", ncol=1)
    clean_axis(ax_b)
    panel_label(ax_b, "B")

    ab = ablation.sort_values("auc_loss_when_removed")
    y = np.arange(len(ab))
    ax_c.barh(
        y,
        ab["auc_loss_when_removed"],
        color=[BLOCK_COLORS.get(m, AXIS) for m in ab["removed_block"]],
        edgecolor="white",
        linewidth=0.4,
    )
    ax_c.axvline(0, color=AXIS, linewidth=0.7)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([BLOCK_LABELS.get(m, m) for m in ab["removed_block"]])
    ax_c.set_xlabel("AUC change when removed")
    ax_c.set_title("Block ablation", pad=4)
    clean_axis(ax_c)
    panel_label(ax_c, "C")

    beh = behavior_importance.sort_values("total_abs_weight")
    y = np.arange(len(beh))
    beh_colors = [CLUSTER_COLOR_LOOKUP.get(b, AXIS) for b in beh["behavior"]]
    ax_d.barh(
        y,
        beh["percent_of_behavior_linked_weight"],
        color=beh_colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(beh["behavior"])
    ax_d.set_xlabel("Share of behavior-linked weight (%)")
    ax_d.set_title("Overall behavior-category weight", pad=4)
    clean_axis(ax_d)
    panel_label(ax_d, "D")

    top_pairs = pair_importance.head(10).iloc[::-1]
    y = np.arange(len(top_pairs))
    colors = [CLUSTER_COLOR_LOOKUP.get(v, TRANSITION) for v in top_pairs["from_behavior"]]
    ax_e.barh(
        y, top_pairs["standardized_linear_weight"], color=colors, edgecolor="white", linewidth=0.4
    )
    ax_e.axvline(0, color=AXIS, linewidth=0.7)
    ax_e.set_yticks(y)
    ax_e.set_yticklabels(top_pairs["transition"])
    ax_e.set_xlabel("Standardized linear weight")
    ax_e.set_title("Top transition-pair features", pad=4)
    clean_axis(ax_e)
    panel_label(ax_e, "E")

    top = coefs.head(12).iloc[::-1]
    y = np.arange(len(top))
    colors = [
        BLOCK_COLORS.get(block if block != "freezing" else "freezing_only", AXIS)
        for block in top["block"]
    ]
    ax_f.barh(y, top["standardized_linear_weight"], color=colors, edgecolor="white", linewidth=0.4)
    ax_f.axvline(0, color=AXIS, linewidth=0.7)
    ax_f.set_yticks(y)
    ax_f.set_yticklabels(top["display_feature"])
    ax_f.set_xlabel("Standardized linear weight")
    ax_f.set_title("Top all-parameter features", pad=4)
    clean_axis(ax_f)
    panel_label(ax_f, "F")

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"supplementary_figure4_all_parameter_importance.{ext}", dpi=600)
    plt.close(fig)


def main() -> None:
    labels = load_labels()
    frequency, frequency_map = frequency_features()
    freezing, freezing_map = freezing_features()
    transition, transition_map = transition_features()
    transition_pairs, transition_pair_map, transition_feature_map = transition_pair_features()
    diversity, diversity_map = diversity_features()
    bout, bout_map = bout_features()

    behavior = merge_feature_frames([frequency, transition, diversity, bout])
    behavior_with_pairs = merge_feature_frames(
        [frequency, transition, transition_pairs, diversity, bout]
    )
    behavior_freezing = behavior_with_pairs.merge(freezing, on="animal_id", how="inner")
    feature_sets = {
        "freezing_only": (freezing, freezing_map),
        "frequency": (frequency, frequency_map),
        "transition": (transition, transition_map),
        "transition_pairs": (transition_pairs, transition_pair_map),
        "diversity": (diversity, diversity_map),
        "bout_duration": (bout, bout_map),
        "behavior_integrated": (
            behavior,
            {**frequency_map, **transition_map, **diversity_map, **bout_map},
        ),
        "behavior_integrated_with_pairs": (
            behavior_with_pairs,
            {**frequency_map, **transition_map, **transition_pair_map, **diversity_map, **bout_map},
        ),
        "behavior_plus_freezing": (
            behavior_freezing,
            {
                **frequency_map,
                **transition_map,
                **transition_pair_map,
                **diversity_map,
                **bout_map,
                **freezing_map,
            },
        ),
    }

    loocv_rows = []
    loocv_preds = []
    cross_rows = []
    cross_preds = []
    for name, (features, _) in feature_sets.items():
        data = labels.merge(features, on="animal_id", how="inner")
        cols = [c for c in data.columns if "__" in c]
        row, pred = evaluate_loocv(data, cols, name)
        loocv_rows.append(row)
        loocv_preds.append(pred)
        rows, preds = evaluate_cross(data, cols, name)
        cross_rows.extend(rows)
        cross_preds.append(preds)

    loocv_metrics = pd.DataFrame(loocv_rows)
    cross_metrics = pd.DataFrame(cross_rows)
    loocv_predictions = pd.concat(loocv_preds, ignore_index=True)
    cross_predictions = pd.concat(cross_preds, ignore_index=True)

    behavior_data = labels.merge(behavior_with_pairs, on="animal_id", how="inner")
    behavior_cols = [c for c in behavior_data.columns if "__" in c]
    block_map = feature_sets["behavior_integrated_with_pairs"][1]
    full_auc = loocv_metrics.loc[
        loocv_metrics["model"].eq("behavior_integrated_with_pairs"), "roc_auc"
    ].iloc[0]
    ablation_rows = []
    for block in ["frequency", "transition", "transition_pairs", "diversity", "bout_duration"]:
        keep = [c for c in behavior_cols if block_map[c] != block]
        row, _ = evaluate_loocv(behavior_data, keep, f"without_{block}")
        ablation_rows.append(
            {
                "removed_block": block,
                "full_behavior_auc": full_auc,
                "auc_without_block": row["roc_auc"],
                "auc_loss_when_removed": full_auc - row["roc_auc"],
            }
        )
    ablation = pd.DataFrame(ablation_rows)
    coefs = coefficients(behavior_data, behavior_cols, block_map)
    coefs.insert(1, "display_feature", coefs["feature"].map(short_feature_name))
    behavior_importance = behavior_category_importance(coefs, transition_feature_map)
    pair_importance = transition_pair_importance(coefs, transition_feature_map)

    loocv_metrics.to_csv(OUT / "expanded_feature_model_loocv_metrics.csv", index=False)
    cross_metrics.to_csv(OUT / "expanded_feature_model_cross_cohort_metrics.csv", index=False)
    loocv_predictions.to_csv(OUT / "expanded_feature_model_loocv_predictions.csv", index=False)
    cross_predictions.to_csv(
        OUT / "expanded_feature_model_cross_cohort_predictions.csv", index=False
    )
    ablation.to_csv(OUT / "expanded_feature_block_ablation.csv", index=False)
    coefs.to_csv(OUT / "expanded_feature_importance_coefficients.csv", index=False)
    behavior_importance.to_csv(OUT / "expanded_behavior_category_importance.csv", index=False)
    pair_importance.to_csv(OUT / "expanded_transition_pair_importance.csv", index=False)
    plot_figure(loocv_metrics, cross_metrics, ablation, coefs, behavior_importance, pair_importance)
    print("LOOCV model comparison")
    print(
        loocv_metrics[["model", "n_features", "roc_auc", "balanced_accuracy"]].to_string(
            index=False
        )
    )
    print("\nBlock ablation")
    print(ablation.to_string(index=False))
    print("\nTop features")
    print(
        coefs.head(12)[["display_feature", "block", "standardized_linear_weight"]].to_string(
            index=False
        )
    )
    print("\nBehavior category importance")
    print(behavior_importance.to_string(index=False))
    print("\nTop transition pairs")
    print(
        pair_importance.head(12)[
            ["transition", "standardized_linear_weight", "abs_weight"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
