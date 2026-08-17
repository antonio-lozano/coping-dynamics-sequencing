"""Regenerate the Figure 6 resilience statistics (diversity metrics + bout duration),
3-group: Control / vulnerable ELS / resilient ELS.

Reproduces the manuscript resilience statistics from the bundled source data.
Self-contained: reads only repo data.

Model per metric (and per cluster for bout duration):
  MixedLM(metric ~ C(New_condition, Treatment(reference=R)) + Experiment, groups=Animal, reml=False)
  fit twice (R = 'Control' and R = 'ELS') so every pairwise contrast is read off a row:
    vulnerable - control   = Control-ref  [T.ELS]
    resilient  - control   = Control-ref  [T.ELS_resilient]
    resilient  - vulnerable = ELS-ref     [T.ELS_resilient]

Benjamini-Hochberg FDR is applied within each contrast type:
  - diversity metrics: across the 6 metrics
  - bout duration:     across the 7 clusters   (matches the manuscript Fig 6M/N/P values)

Run: python scripts/derive_tables/fig6_resilience_stats.py
Writes statistics/fig6_diversity_resilience_stats.csv
       statistics/fig6_bout_resilience_stats.csv
       statistics/fig6_transition_resilience_stats.csv
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import entropy
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.statistics import (
    determinism,
    markov_entropy,
    recurrence_rate,
)
from src.statistics import (
    lempel_ziv_complexity as lz_complexity,
)

CSV = REPO / "data" / "raw" / "syllable_usage_per_timebin_250ms.csv"
OUT_DIR = REPO / "statistics"
OUT_DIV = OUT_DIR / "fig6_diversity_resilience_stats.csv"
OUT_BOUT = OUT_DIR / "fig6_bout_resilience_stats.csv"
OUT_TRANS = OUT_DIR / "fig6_transition_resilience_stats.csv"

CLUSTER_MAP = {
    "Freezing": [0, 28],
    "Sniffing": [18, 20],
    "Grooming": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climbing": [111],
    "Jump": [23, 29, 30, 34],
}
S2C = {s: c for c, ss in CLUSTER_MAP.items() for s in ss}
ELS_RESILIENT = [
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
]
ELS_VULN = [
    "15.6",
    "23.4",
    "25.1",
    "32.2",
    "32.4",
    "49.5",
    "55.3",
    "55.5",
    "56.3",
    "63.3",
    "69.1",
    "69.2",
    "69.3",
    "69.5",
    "73.1",
    "73.4",
    "75.1",
    "88.2",
    "88.3",
    "128.4",
    "133.3",
    "123.1",
    "134.4",
    "144.3",
    "144.5",
    "153.4",
    "154.4",
    "159.3",
    "159.5",
]
CLUSTERS = ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]


def load():
    df = pd.read_csv(CSV)
    df.columns = df.columns.str.strip()
    df.rename(columns={"Time Bin": "Time_bin"}, inplace=True)
    df["Animal"] = df["Animal"].astype(str)
    df["Cluster"] = df["Syllable"].astype(int).map(S2C).fillna("")  # unmapped -> "" (as original)
    df["New_condition"] = df.apply(
        lambda r: (
            "ELS_resilient"
            if r.Animal in ELS_RESILIENT
            else ("ELS" if r.Animal in ELS_VULN else r["Condition"])
        ),
        axis=1,
    )
    return df


def diversity_table(df):
    f = (
        df.groupby(["Animal", "Experiment", "New_condition", "Cluster"])
        .size()
        .reset_index(name="Count")
    )
    piv = f.pivot_table(
        index=["Animal", "Experiment", "New_condition"],
        columns="Cluster",
        values="Count",
        fill_value=0,
    )
    tot = piv.sum(axis=1)
    rel = piv.div(tot, axis=0)
    H = rel.apply(lambda x: entropy(x[x > 0]), axis=1)
    E = H / np.log((piv > 0).sum(axis=1))
    S = rel.apply(lambda x: 1 - np.sum(x**2), axis=1)
    idx = H.index
    div = pd.DataFrame(
        {
            "Animal": idx.get_level_values(0),
            "Experiment": idx.get_level_values(1),
            "New_condition": idx.get_level_values(2),
            "Entropy": H.values,
            "Evenness": E.values,
            "Simpson": S.values,
        }
    )
    # CUI (named clusters only)
    fc = (
        df[df.Cluster != ""]
        .groupby(["Animal", "Experiment", "New_condition", "Cluster"])
        .size()
        .reset_index(name="Count")
    )
    tota = (
        fc.groupby(["Animal", "Experiment", "New_condition"])["Count"]
        .sum()
        .reset_index(name="Total")
    )
    fc = fc.merge(tota, on=["Animal", "Experiment", "New_condition"])
    fc["rel"] = fc.Count / fc.Total

    def cui(s):
        rels = np.sort(s.values)[::-1]
        cum = rels.cumsum()
        n = len(cum)
        base = (n + 1) / (2 * n)
        return (cum.mean() - base) / (1 - base)

    cu = (
        fc.groupby(["Animal", "Experiment", "New_condition"])["rel"]
        .apply(cui)
        .reset_index(name="CUI")
    )
    div = div.merge(cu, on=["Animal", "Experiment", "New_condition"], how="left")
    # Markov entropy per animal (matches original). Deliberately NOT the shared
    # src.statistics.markov_entropy: this one uses Laplace smoothing 0.002, the
    # published value behind the MarkovEntropy column, while the transition
    # table below uses the shared implementation at 0.01 (MarkovEntropyIdx).
    rows = []
    for a, grp in df.groupby("Animal"):
        sts = grp.sort_values("Time_bin")["Cluster"].tolist()
        uni = list(pd.unique(sts))
        M = len(uni)
        if M < 2:
            rows.append({"Animal": a, "MarkovEntropy": 0.0})
            continue
        idxm = {s: i for i, s in enumerate(uni)}
        cnt = np.zeros((M, M))
        for x, y in zip(sts, sts[1:]):
            cnt[idxm[x], idxm[y]] += 1
        cnt += 0.002
        P = cnt / cnt.sum(axis=1, keepdims=True)
        pi = np.array([sts.count(s) for s in uni]) / len(sts)
        ent = [-np.sum(P[i, P[i] > 0] * np.log2(P[i, P[i] > 0])) for i in range(M)]
        rows.append({"Animal": a, "MarkovEntropy": float(np.dot(pi, ent))})
    mk = pd.DataFrame(rows).merge(
        df[["Animal", "Experiment", "New_condition"]].drop_duplicates(), on="Animal"
    )
    div = div.merge(
        mk[["Animal", "Experiment", "MarkovEntropy"]], on=["Animal", "Experiment"], how="left"
    )
    return div


def bout_table(df):
    return (
        df.groupby(["Animal", "Experiment"])
        .apply(
            lambda d: (
                d.sort_values("Time_bin")
                .assign(chg=lambda x: (x["Cluster"] != x["Cluster"].shift()).astype(int))
                .assign(Bout=lambda x: x["chg"].cumsum())
                .groupby(["Bout", "Cluster"])
                .size()
                .mul(0.25)
                .groupby(level=1)
                .mean()
            )
        )
        .reset_index(name="MeanBoutDuration")
        .merge(
            df[["Animal", "Experiment", "New_condition"]].drop_duplicates(),
            on=["Animal", "Experiment"],
            how="left",
        )
    )


def markov_entropy_seq(seq, smoothing=0.01):
    return markov_entropy(seq, smoothing_factor=smoothing)


def transition_table(df):
    """Predominant-cluster sequence per animal (max Percentage per time bin), then 4 metrics."""
    idx = df.groupby(["Animal", "Time_bin"])["Percentage"].idxmax()
    pred = df.loc[idx].sort_values(["Animal", "Time_bin"])
    rows = []
    for a, g in pred.groupby("Animal", sort=False):
        seq = g["Cluster"].tolist()
        info = g.iloc[0]
        rows.append(
            {
                "Animal": a,
                "Experiment": info["Experiment"],
                "New_condition": info["New_condition"],
                "LZ_complexity": lz_complexity(seq),
                "Recurrence": recurrence_rate(seq),
                "Determinism": determinism(seq),
                "MarkovEntropyIdx": markov_entropy_seq(seq),
            }
        )
    return pd.DataFrame(rows)


def contrasts(data, metric):
    """Return the three pairwise contrasts from dual-reference MixedLM."""
    d = data.dropna(subset=[metric]).copy()
    mc = smf.mixedlm(
        f"{metric} ~ C(New_condition, Treatment(reference='Control')) + Experiment",
        d,
        groups=d["Animal"],
    ).fit(reml=False)
    me = smf.mixedlm(
        f"{metric} ~ C(New_condition, Treatment(reference='ELS')) + Experiment",
        d,
        groups=d["Animal"],
    ).fit(reml=False)
    kE = "C(New_condition, Treatment(reference='Control'))[T.ELS]"
    kRc = "C(New_condition, Treatment(reference='Control'))[T.ELS_resilient]"
    kRe = "C(New_condition, Treatment(reference='ELS'))[T.ELS_resilient]"
    return {
        "vuln_vs_control": (mc.params[kE], mc.bse[kE], mc.tvalues[kE], float(mc.pvalues[kE])),
        "resilient_vs_control": (
            mc.params[kRc],
            mc.bse[kRc],
            mc.tvalues[kRc],
            float(mc.pvalues[kRc]),
        ),
        "resilient_vs_vuln": (me.params[kRe], me.bse[kRe], me.tvalues[kRe], float(me.pvalues[kRe])),
    }


def build(items, label_col):
    """items: list of (name, contrasts_dict). Build long table + BH-FDR within contrast type."""
    rows = []
    for name, c in items:
        for ctype, (b, se, z, p) in c.items():
            rows.append(
                {
                    label_col: name,
                    "contrast": ctype,
                    "beta": round(b, 4),
                    "SE": round(se, 4),
                    "z": round(z, 4),
                    "p": p,
                }
            )
    out = pd.DataFrame(rows)
    out["p_BH_FDR"] = np.nan
    for ctype in out.contrast.unique():
        m = out.contrast == ctype
        out.loc[m, "p_BH_FDR"] = multipletests(out.loc[m, "p"], method="fdr_bh")[1]
    out["p"] = out["p"].round(4)
    out["p_BH_FDR"] = out["p_BH_FDR"].round(4)
    out["sig_BH"] = out["p_BH_FDR"] < 0.05
    return out


def main():
    df = load()
    n = df.drop_duplicates("Animal")["New_condition"].value_counts().to_dict()
    print(f"Source: {CSV.relative_to(REPO)}  |  groups: {n}\n")

    # diversity
    div = diversity_table(df)
    div_items = [
        (v, contrasts(div, v)) for v in ["Entropy", "Evenness", "Simpson", "CUI", "MarkovEntropy"]
    ]
    div_out = build(div_items, "metric")
    OUT_DIV.parent.mkdir(parents=True, exist_ok=True)
    div_out.to_csv(OUT_DIV, index=False)

    # bout
    bt = bout_table(df)
    bout_items = [(cl, contrasts(bt[bt.Cluster == cl], "MeanBoutDuration")) for cl in CLUSTERS]
    bout_out = build(bout_items, "cluster")
    bout_out.to_csv(OUT_BOUT, index=False)

    def show(title, dd, key):
        print(f"\n===== {title} =====")
        print(
            f"  {key:<13}{'contrast':<22}{'beta':>8}{'SE':>8}{'z':>8}{'p':>9}{'p_BH':>9}{'sig':>5}"
        )
        for _, r in dd.iterrows():
            print(
                f"  {r[key]:<13}{r.contrast:<22}{r.beta:>8.3f}{r.SE:>8.3f}{r.z:>8.3f}"
                f"{r.p:>9.4f}{r.p_BH_FDR:>9.4f}{('*' if r.sig_BH else ''):>5}"
            )

    # transition metrics
    tr = transition_table(df)
    tr_items = [
        (v, contrasts(tr, v))
        for v in ["LZ_complexity", "Recurrence", "Determinism", "MarkovEntropyIdx"]
    ]
    tr_out = build(tr_items, "metric")
    tr_out.to_csv(OUT_TRANS, index=False)

    show("DIVERSITY metrics (BH-FDR across 5 metrics, per contrast)", div_out, "metric")
    show("BOUT DURATION (BH-FDR across 7 clusters, per contrast)", bout_out, "cluster")
    show("TRANSITION metrics (BH-FDR across 4 metrics, per contrast)", tr_out, "metric")
    print(
        "\nReport.xlsx transition targets: LZ ELS/Resilient 8.68/8.65/1.00 p=0.316; "
        "Recurrence ELS/Resilient -0.029/0.0107/-2.73 p=0.006"
    )

    # quick manuscript check (Fig 6M/N/P)
    def g(cl, ct):
        r = bout_out[(bout_out.cluster == cl) & (bout_out.contrast == ct)].iloc[0]
        return f"b={r.beta:+.3f} SE={r.SE:.3f} z={r.z:+.3f} p={r.p:.3f} BH={r.p_BH_FDR:.3f}"

    print("\nManuscript Fig 6 cross-check:")
    print(
        f"  Freeze resilient-vs-vuln : {g('Freezing', 'resilient_vs_vuln')}   (MS 0.263/0.089/2.969/0.003, BH 0.023)"
    )
    print(
        f"  Turn   resilient-vs-vuln : {g('Turn', 'resilient_vs_vuln')}   (MS -0.201/0.089/-2.264/0.024, BH 0.085)"
    )
    print(
        f"  Sniff  vuln-vs-control   : {g('Sniffing', 'vuln_vs_control')}   (MS 0.420/0.127/3.318/<.001, BH 0.004)"
    )
    print(
        f"\nsaved -> {OUT_DIV.relative_to(REPO)}\nsaved -> {OUT_BOUT.relative_to(REPO)}\nsaved -> {OUT_TRANS.relative_to(REPO)}"
    )


if __name__ == "__main__":
    main()
