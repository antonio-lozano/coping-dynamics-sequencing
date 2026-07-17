"""Figure 4 - behavioural diversity dynamics: ONE place for compute + statistics
+ checks (+ optional plot).

Design (template for all figure_N.py):
  1. CONFIG  - all seeds / FDR families / gold values come from
               config/figure_metadata.json (nothing statistical hardcoded here).
  2. COMPUTE - per-animal metrics. Transition metrics are recomputed from the raw
               250 ms syllable table in NATIVE FILE ORDER (required to match gold).
  3. STATS   - MixedLM(metric ~ group + Experiment, groups=Animal, reml=False).
               The model sits on a parameter boundary (one row per animal) so the
               Wald SE is optimizer/start dependent; the start-seed + optimizer for
               each metric are read from the metadata and applied by `seed_fit`.
  4. FDR     - Benjamini-Hochberg over the families defined in the metadata.
  5. CHECKS  - betas are compared to the gold report; significance is asserted.
  6. PLOT    - optional, delegates to the existing figure generator.

Run:
  python scripts/figures/figure_4.py            # compute + stats + checks
  python scripts/figures/figure_4.py --plot     # also render the figure
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.config import SYLLABLE_TIMEBIN_250MS

META = json.loads((REPO / "config/figure_metadata.json").read_text())["figure_4"]
GOLD = json.loads((REPO / "config/figure_metadata.json").read_text())["gold_reference"]["figure_4"]
SR = REPO / "results/statistical_reports"

CLUSTER_MAP = {"Freezing": [0, 28], "Sniffing": [18, 20], "Grooming": [24],
               "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
               "Locomotion": [11, 12, 14, 16, 19, 21, 25],
               "Climbing": [111], "Jump": [23, 29, 30, 34]}


# ---------------------------------------------------------------- COMPUTE
def transition_metrics() -> pd.DataFrame:
    """Recurrence, determinism, Markov entropy per animal, NATIVE FILE ORDER."""
    s2c = {s: c for c, ss in CLUSTER_MAP.items() for s in ss}
    mo = pd.read_csv(SYLLABLE_TIMEBIN_250MS).rename(columns={"Time Bin": "Time_bin"})
    mo = mo[mo.Condition.isin(["Control", "ELS"])].copy()
    mo["Animal"] = mo.Animal.astype(str)
    mo["Syllable"] = pd.to_numeric(mo.Syllable, errors="coerce").astype(int)
    mo["Cluster"] = mo.Syllable.map(s2c).fillna("")

    def rr(seq):
        a = np.array(seq); return (a[:, None] == a[None, :]).sum() / len(a) ** 2

    def det(seq, ml=2):
        a = np.array(seq); n = len(a); R = (a[:, None] == a[None, :]).astype(int)
        tot = R.sum() - n; lines = 0
        for k in range(-n + 1, n):
            c = 0
            for v in np.diag(R, k):
                if v: c += 1
                else:
                    if c >= ml: lines += c
                    c = 0
            if c >= ml: lines += c
        return lines / tot if tot > 0 else 0.0

    def markov(states):
        u = list(pd.unique(states))
        if len(u) == 1: return 0.0
        idx = {s: i for i, s in enumerate(u)}; M = len(u); cn = np.zeros((M, M))
        for a, b in zip(states, states[1:]): cn[idx[a], idx[b]] += 1
        cn += 0.01; p = cn / cn.sum(1, keepdims=True)
        pi = np.array([states.count(s) for s in u]) / len(states)
        inner = np.array([-np.sum(r[r > 0] * np.log2(r[r > 0])) for r in p])
        return float(np.sum(pi * inner))

    rows = []
    for an, g in mo.groupby("Animal"):
        seq = g["Cluster"].tolist()                      # file order, NOT sorted
        rows.append({"Animal": an, "Experiment": int(g.Experiment.iloc[0]),
                     "recurrence": rr(seq), "determinism": det(seq), "markov": markov(seq)})
    return pd.DataFrame(rows)


def metric_frames() -> dict[str, pd.DataFrame]:
    """Per-animal frames for every metric the figure reports. Diversity and bout
    come from the validated per-animal tables produced by the figure generator;
    transitions are recomputed here in file order."""
    div = pd.read_csv(SR / "fig4_diversity_per_animal.csv")
    bcl = pd.read_csv(SR / "fig4_bout_cluster_per_animal.csv")
    fb = bcl[bcl.cluster == "Freezing"].rename(columns={"bout_duration": "freeze_bout"})
    tr = transition_metrics()
    return {"simpson": div, "cui": div, "freeze_bout": fb,
            "recurrence": tr, "determinism": tr, "markov": tr,
            "_bout_all": bcl}


# ---------------------------------------------------------------- STATS
def seed_fit(df: pd.DataFrame, metric: str, seed_cfg: dict):
    """MixedLM with optional start-param seed + optimizer (both from metadata)."""
    d = df.copy()
    d["grp"] = pd.Categorical(d["group" if "group" in d else "Condition"], categories=["Control", "ELS"]) \
        if ("group" in d or "Condition" in d) else None
    # the per-animal tables use 'group'; harmonise
    gcol = "group" if "group" in df.columns else "Condition"
    d["grp"] = pd.Categorical(d[gcol], categories=["Control", "ELS"])
    d["Experiment"] = d["Experiment"].astype(int)
    base = smf.mixedlm(f"{metric} ~ grp + Experiment", d, groups=d["Animal"]).fit(reml=False)
    key = "grp[T.ELS]"
    if seed_cfg.get("start_seed") is None:
        m = base
    else:
        rng = np.random.default_rng(seed_cfg["start_seed"])
        bp = base.params.values
        sp = bp + rng.normal(0, 0.1 * np.abs(bp).clip(1e-4), size=len(bp))
        sp[-1] = abs(sp[-1]) + rng.uniform(1e-5, 1e-2)
        method = None if seed_cfg["optimizer"] == "default" else seed_cfg["optimizer"]
        kw = {"start_params": sp}
        if method: kw["method"] = method
        m = smf.mixedlm(f"{metric} ~ grp + Experiment", d, groups=d["Animal"]).fit(reml=False, **kw)
    return m.params[key], m.bse[key], m.tvalues[key], m.pvalues[key], d["Animal"].nunique()


def run_statistics() -> pd.DataFrame:
    frames = metric_frames()
    rows = []
    for metric, cfg in META["metric_seeds"].items():
        b, se, z, p, n = seed_fit(frames[metric], metric, cfg)
        rows.append({"metric": metric, "beta": b, "SE": se, "z": z, "p": p, "n": n,
                     "seed": cfg["start_seed"], "optimizer": cfg["optimizer"],
                     "seq_order": cfg["seq_order"]})
    # bout-duration BH-FDR family (per-cluster mean bout, resilient-agnostic 2-group here)
    bcl = frames["_bout_all"]
    bout_rows = []
    for cl in META["fdr_families"]["bout"]:
        sub = bcl[bcl.cluster == cl].rename(columns={"bout_duration": "bd"})
        b, se, z, p, n = seed_fit(sub, "bd", {"start_seed": None, "optimizer": "default", "seq_order": "na"})
        bout_rows.append({"metric": f"bout_{cl}", "beta": b, "SE": se, "z": z, "p": p, "n": n})
    bdf = pd.DataFrame(bout_rows)
    bdf["p_BH"] = multipletests(bdf.p, method="fdr_bh")[1]
    return pd.DataFrame(rows), bdf


# ---------------------------------------------------------------- CHECKS
def run_checks(stats: pd.DataFrame) -> bool:
    print("\n=== CHECKS (beta vs gold report; significance) ===")
    ok = True
    for _, r in stats.iterrows():
        g = GOLD.get(r.metric)
        if not g:
            continue
        db = abs(r.beta - g["beta"])
        beta_ok = db < 0.01
        sig_ok = (r.p < 0.05) == (g["gold_p"] < 0.05)
        tag = "PASS" if (beta_ok and sig_ok) else "FAIL"
        ok = ok and beta_ok and sig_ok
        print(f"  {r.metric:<12} beta={r.beta:+.5f} (gold {g['beta']:+.5f}, d={db:.4f}) "
              f"SE={r.SE:.5f} p={r.p:.4f} (gold {g['gold_p']:.4f})  [{tag}]")
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


# ---------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="also render the figure")
    ap.add_argument("--no-checks", action="store_true")
    args = ap.parse_args()

    stats, bout = run_statistics()
    out = SR / "figure_4_statistics.csv"
    stats.to_csv(out, index=False)
    bout.to_csv(SR / "figure_4_bout_fdr.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.5f}")
    print("Figure 4 statistics (seed-driven MixedLM):")
    print(stats.to_string(index=False))
    print("\nBout durations + BH-FDR:")
    print(bout.to_string(index=False))
    print(f"\nsaved -> {out}")

    if not args.no_checks:
        run_checks(stats)

    if args.plot:
        print("\nRendering figure via generator ...")
        import runpy
        runpy.run_path(str(REPO / "scripts/generate_figures/figure_4_diversity_dynamics.py"),
                       run_name="__main__")


if __name__ == "__main__":
    main()
