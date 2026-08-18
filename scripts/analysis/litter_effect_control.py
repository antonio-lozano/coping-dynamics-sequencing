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



def litter_variance_components(df: pd.DataFrame) -> pd.DataFrame:
    """How much of each metric's variance sits between litters?

    Fits metric ~ Condition + Experiment with a litter random intercept and
    reports the intraclass correlation, sigma2_litter / (sigma2_litter +
    sigma2_residual). This asks whether litter matters at all, rather than
    merely controlling for it: an ICC near zero means littermates are no more
    alike than unrelated mice, and the pseudoreplication concern is empirically
    empty for that measure.

    A likelihood-ratio test against the same model without the random intercept
    gives a p-value. The null sits on the variance boundary, so the usual chi2
    reference is conservative - the honest reading is that a significant test is
    real and a non-significant one is weak evidence either way.
    """
    rows = []
    for column, label in METRICS.items():
        if column not in df.columns:
            continue
        d = df.dropna(subset=[column]).copy()
        d["Condition"] = pd.Categorical(d["group"], categories=["Control", "ELS"])
        formula = f"Q('{column}') ~ Condition + Experiment"
        try:
            # Both fits must use ML, not REML: REML likelihoods are not comparable
            # across models with different random-effect structures, which made
            # the earlier likelihood ratios come out negative.
            mixed = smf.mixedlm(formula, d, groups=d["litter"]).fit(reml=False)
            plain = smf.ols(formula, d).fit()
            var_litter = float(np.asarray(mixed.cov_re)[0, 0])
            var_resid = float(mixed.scale)
            icc = var_litter / (var_litter + var_resid) if var_litter + var_resid else np.nan
            lr = 2 * (mixed.llf - plain.llf)
            from scipy.stats import chi2
            p_lr = 0.5 * chi2.sf(max(lr, 0.0), df=1)   # boundary-corrected
            rows.append({"metric": label, "ICC_litter": icc,
                         "var_litter": var_litter, "var_residual": var_resid,
                         "LR_stat": lr, "p_litter_effect": p_lr})
        except Exception as exc:  # noqa: BLE001
            rows.append({"metric": label, "ICC_litter": np.nan, "var_litter": np.nan,
                         "var_residual": np.nan, "LR_stat": np.nan,
                         "p_litter_effect": np.nan, "note": str(exc)[:50]})
    return pd.DataFrame(rows)


def sibling_similarity_test(df: pd.DataFrame) -> dict:
    """Are littermates more behaviourally alike than unrelated mice?

    Compares the mean Euclidean distance between behavioural profiles for
    sibling pairs against a null built by shuffling litter labels, which keeps
    the litter size distribution intact.
    """
    profiles = pd.read_csv(ROOT / "data/processed/cluster_timecourse_per_animal.csv")
    wide = profiles.pivot_table(index="animal_id", columns=["cluster", "time_bin"],
                                values="pct", aggfunc="mean", fill_value=0.0).sort_index(axis=1)
    animals = [str(a) for a in wide.index]
    features = wide.to_numpy(dtype=float)
    from scipy.spatial.distance import squareform, pdist
    dist = squareform(pdist(features))
    litter = np.array([a.split(".")[0] for a in animals])

    def mean_sibling_distance(labels: np.ndarray) -> float:
        vals = [dist[i, j] for i in range(len(labels)) for j in range(i + 1, len(labels))
                if labels[i] == labels[j]]
        return float(np.mean(vals)) if vals else np.nan

    observed = mean_sibling_distance(litter)
    null = np.empty(N_PERM // 4)
    shuffled = litter.copy()
    for k in range(len(null)):
        RNG.shuffle(shuffled)
        null[k] = mean_sibling_distance(shuffled)
    p = (np.sum(null <= observed) + 1) / (len(null) + 1)
    return {"observed_sibling_distance": observed,
            "null_mean_distance": float(np.nanmean(null)),
            "p_siblings_more_alike": float(p)}


def litter_size_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Does the number of siblings tested predict behaviour?"""
    d = df.copy()
    d["litter_n"] = d.groupby("litter")["Animal"].transform("size")
    d["Condition"] = pd.Categorical(d["group"], categories=["Control", "ELS"])
    rows = []
    for column, label in METRICS.items():
        if column not in d.columns:
            continue
        sub = d.dropna(subset=[column])
        try:
            res = smf.mixedlm(f"Q('{column}') ~ Condition + Experiment + litter_n",
                              sub, groups=sub["litter"]).fit(reml=False)
            rows.append({"metric": label, "beta_litter_n": res.params["litter_n"],
                         "SE": res.bse["litter_n"], "p": res.pvalues["litter_n"]})
        except Exception:  # noqa: BLE001
            continue
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

    variance = litter_variance_components(df)
    variance.to_csv(OUT / "litter_variance_components.csv", index=False)
    print("\nIS THERE A LITTER EFFECT AT ALL?  (ICC = share of variance between litters)")
    print(variance.round(4).to_string(index=False))

    sib = sibling_similarity_test(df)
    print("\nARE LITTERMATES MORE ALIKE?")
    print(f"  sibling pairs mean profile distance = {sib['observed_sibling_distance']:.2f}")
    print(f"  shuffled-litter null                = {sib['null_mean_distance']:.2f}")
    print(f"  permutation p                       = {sib['p_siblings_more_alike']:.4f}")

    size = litter_size_effect(df)
    print("\nDOES LITTER SIZE PREDICT BEHAVIOUR?")
    print(size.round(4).to_string(index=False))
    size.to_csv(OUT / "litter_size_effect.csv", index=False)

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
