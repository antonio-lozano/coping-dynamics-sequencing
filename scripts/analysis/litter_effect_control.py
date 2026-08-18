# SPDX-License-Identifier: MIT
"""Litter-effect control for the ELS design.

Reviewers of early-life-stress work reliably ask whether littermates were
treated as independent. In this dataset the litter is recoverable from the
animal identifier: the digits before the decimal point are the dam, so 2.3 and
2.4 are siblings, as are 11.5 and 11.6.

Two things are reported.

1. The structure itself: how many litters, how many animals each contributed,
   and how resilient and vulnerable animals are distributed across them. If
   resilience clustered within a few litters, the subgroup could be a family
   effect rather than an individual one, so that is tested explicitly against a
   permutation null.

2. A refit of every animal-level model with litter as the random intercept.
   The published models group by animal, but these metrics have exactly one
   observation per animal, so that random effect has nothing to estimate and
   sits on the variance boundary - which is also why the standard errors were
   unstable between runs. Litter is the grouping factor the design actually
   has.

    python scripts/analysis/litter_effect_control.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "statistics"
RNG = np.random.default_rng(13)
N_PERM = 20000

# The twelve resilient ELS animals, from the Figure 5 Euclidean classification.
RESILIENT = {"2.4", "15.4", "26.4", "43.3", "43.5", "49.6",
             "123.3", "134.2", "148.4", "150.3", "150.5", "159.4"}


def litter_of(animal: str) -> str:
    return str(animal).split(".")[0]


def load_animal_metrics() -> pd.DataFrame:
    """One row per animal: diversity, bout and transition metrics."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fig4", ROOT / "scripts" / "generate_figures" / "figure_4_diversity_dynamics.py")
    fig4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fig4)

    _p, _ps, sequences, meta = fig4.load_sequences()
    metrics, _usage = fig4.compute_frequency_metrics(sequences, meta)
    transitions = fig4.transition_metrics(sequences, meta)
    bouts = fig4.bout_table(sequences, meta)
    mean_bout = (bouts.groupby(["Animal", "cluster"], as_index=False)["bout_duration"]
                 .mean().pivot(index="Animal", columns="cluster", values="bout_duration"))

    df = metrics.merge(transitions.drop(columns=["group", "Experiment"]), on="Animal")
    df = df.merge(mean_bout, on="Animal", how="left")
    df["litter"] = df["Animal"].map(litter_of)
    df["resilient"] = df["Animal"].isin(RESILIENT)
    df["profile"] = np.where(df["group"] == "Control", "Control",
                             np.where(df["resilient"], "ELS resilient", "ELS vulnerable"))
    return df


def structure_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sizes = df.groupby("litter").size()
    rows.append({"quantity": "animals", "value": len(df)})
    rows.append({"quantity": "litters", "value": df["litter"].nunique()})
    rows.append({"quantity": "animals per litter (mean)", "value": round(sizes.mean(), 2)})
    rows.append({"quantity": "animals per litter (max)", "value": int(sizes.max())})
    rows.append({"quantity": "litters spanning both groups",
                 "value": int((df.groupby("litter")["group"].nunique() > 1).sum())})
    els = df[df["group"] == "ELS"]
    rows.append({"quantity": "ELS litters", "value": els["litter"].nunique()})
    rows.append({"quantity": "litters containing a resilient animal",
                 "value": els.loc[els["resilient"], "litter"].nunique()})
    rows.append({"quantity": "resilient animals", "value": int(els["resilient"].sum())})
    mixed = els.groupby("litter")["resilient"].nunique()
    rows.append({"quantity": "ELS litters split resilient/vulnerable",
                 "value": int((mixed > 1).sum())})
    return pd.DataFrame(rows)


def resilience_clustering_test(df: pd.DataFrame) -> dict:
    """Do resilient animals cluster within litters more than chance?

    Statistic: number of distinct litters the resilient animals come from.
    Fewer litters than expected would mean resilience runs in families.
    """
    els = df[df["group"] == "ELS"].reset_index(drop=True)
    litters = els["litter"].to_numpy()
    n_res = int(els["resilient"].sum())
    observed = els.loc[els["resilient"], "litter"].nunique()
    null = np.empty(N_PERM, dtype=int)
    index = np.arange(len(els))
    for k in range(N_PERM):
        pick = RNG.choice(index, size=n_res, replace=False)
        null[k] = len(set(litters[pick]))
    p_clustered = (np.sum(null <= observed) + 1) / (N_PERM + 1)
    return {"resilient_n": n_res, "observed_litters": observed,
            "expected_litters": float(null.mean()),
            "p_more_clustered_than_chance": float(p_clustered)}


METRICS = {
    "simpson": "Simpson diversity", "shannon": "Shannon entropy",
    "evenness": "Evenness", "cui": "Cumulative usage index",
    "lz": "Lempel-Ziv complexity", "recurrence": "Recurrence rate",
    "determinism": "Determinism", "markov": "Markov entropy",
    "Freezing": "Freeze bout duration", "Sniffing": "Sniff bout duration",
    "Turn": "Turn bout duration",
}


def refit_with_litter(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, label in METRICS.items():
        if column not in df.columns:
            continue
        d = df.dropna(subset=[column]).copy()
        d["Condition"] = pd.Categorical(d["group"], categories=["Control", "ELS"])
        for grouping, name in (("Animal", "animal (published)"), ("litter", "litter")):
            try:
                res = smf.mixedlm(f"Q('{column}') ~ Condition + Experiment", d,
                                  groups=d[grouping]).fit(reml=False)
                key = "Condition[T.ELS]"
                rows.append({
                    "metric": label, "random_effect": name,
                    "beta": res.params[key], "SE": res.bse[key],
                    "z": res.tvalues[key], "p": res.pvalues[key],
                    "n_groups": d[grouping].nunique(),
                })
            except Exception as exc:  # noqa: BLE001
                rows.append({"metric": label, "random_effect": name,
                             "beta": np.nan, "SE": np.nan, "z": np.nan,
                             "p": np.nan, "n_groups": d[grouping].nunique(),
                             "note": str(exc)[:60]})
    return pd.DataFrame(rows)


def main() -> None:
    df = load_animal_metrics()
    OUT.mkdir(parents=True, exist_ok=True)

    structure = structure_report(df)
    print("LITTER STRUCTURE")
    print(structure.to_string(index=False))

    counts = (df.groupby(["profile", "litter"]).size().reset_index(name="n")
              .pivot(index="litter", columns="profile", values="n").fillna(0).astype(int))
    counts.to_csv(OUT / "litter_composition.csv")

    test = resilience_clustering_test(df)
    print("\nDOES RESILIENCE RUN IN FAMILIES?")
    print(f"  {test['resilient_n']} resilient animals come from "
          f"{test['observed_litters']} litters "
          f"(chance expectation {test['expected_litters']:.1f})")
    print(f"  permutation p (more clustered than chance) = "
          f"{test['p_more_clustered_than_chance']:.3f}")

    fits = refit_with_litter(df)
    fits.to_csv(OUT / "litter_effect_refits.csv", index=False)
    wide = fits.pivot(index="metric", columns="random_effect",
                      values=["beta", "SE", "p"])
    print("\nMODELS REFITTED WITH LITTER AS THE RANDOM EFFECT")
    pd.set_option("display.width", 200)
    print(wide.round(4).to_string())

    flips = []
    for metric, sub in fits.groupby("metric"):
        p = dict(zip(sub.random_effect, sub.p))
        if len(p) == 2 and not any(np.isnan(list(p.values()))):
            if (p["animal (published)"] < 0.05) != (p["litter"] < 0.05):
                flips.append((metric, p["animal (published)"], p["litter"]))
    print("\nCONCLUSIONS THAT CHANGE:",
          "none" if not flips else "")
    for metric, a, b in flips:
        print(f"  {metric}: p {a:.4f} (animal) -> {b:.4f} (litter)")
    print(f"\nwrote {OUT / 'litter_effect_refits.csv'}")
    print(f"wrote {OUT / 'litter_composition.csv'}")


if __name__ == "__main__":
    main()
