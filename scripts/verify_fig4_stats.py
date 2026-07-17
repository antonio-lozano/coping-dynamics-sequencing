"""
Reproduce Figure 4 statistics and compare to Report.xlsx gold standard.
Metrics: diversity (simpson/shannon/evenness/CUI), bout durations, transition metrics.
Model: MixedLM(metric ~ Stress + Experiment, groups=animal_id)
       With 1 obs/animal this is effectively OLS + experiment covariate.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import SYLLABLE_TIMEBIN_250MS

CLUSTER_MAP = {
    "Freezing": [0, 28],
    "Sniffing": [18, 20],
    "Grooming": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climbing": [111],
    "Jump": [23, 29, 30, 34],
}

DISPLAY_ORDER = ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]

# Gold standard from Report.xlsx
GOLD = {
    "diversity": {
        "simpson":  {"beta": -0.013440869, "se": 0.006592931, "z": -2.038678748, "p": 0.0414821},
        "shannon":  {"beta": -0.011573886, "se": 0.018343103, "z": -0.630966653, "p": 0.528062329},
        "evenness": {"beta": -0.006497065, "se": 0.009731452, "z": -0.667635767, "p": 0.504366121},
        "cui":      {"beta":  0.063645788, "se": 0.021445504, "z":  2.967791679, "p": 0.002999475},
    },
    "bout": {
        "Overall":     {"beta":  0.044643567, "se": 0.034937455, "z":  1.277813938, "p": 0.201315039},
        "Freezing":    {"beta": -0.105173988, "se": 0.052566191, "z": -2.00079151,  "p": 0.045414863},
        "Sniffing":    {"beta":  0.322323093, "se": 0.128634553, "z":  2.505727158, "p": 0.012219988},
        "Grooming":    {"beta":  0.013740377, "se": 0.010360041, "z":  1.326286001, "p": 0.184744982},
        "Turn":        {"beta":  0.159255617, "se": 0.051574764, "z":  3.087859328, "p": 0.002016039},
        "Locomotion":  {"beta":  0.003169017, "se": 0.015414888, "z":  0.205581577, "p": 0.837117769},
        "Climbing":    {"beta":  0.017676532, "se": 0.038136176, "z":  0.463510856, "p": 0.642998235},
        "Jump":        {"beta": -0.015732322, "se": 0.024630672, "z": -0.638728906, "p": 0.522999305},
    },
    "transition": {
        "lz":          {"beta": -8.048780488, "se": 5.56672732,  "z": -1.445872956, "p": 0.148212838},
        "recurrence":  {"beta":  0.013440869, "se": 0.00593759,  "z":  2.263690785, "p": 0.02359314},
        "determinism": {"beta":  0.016104788, "se": 0.004761667, "z":  3.382174516, "p": 0.000719144},
        "markov":      {"beta": -0.041186979, "se": 0.017585398, "z": -2.342112429, "p": 0.019174938},
    }
}


def syllable_to_cluster():
    out = {}
    for cluster, syllables in CLUSTER_MAP.items():
        for s in syllables:
            out[int(s)] = cluster
    return out


def load_sequences():
    raw = pd.read_csv(SYLLABLE_TIMEBIN_250MS)
    raw = raw.rename(columns={"Time Bin": "time_bin", "Condition": "group"})
    raw = raw[raw["group"].isin(["Control", "ELS"])].copy()
    raw["Animal"] = raw["Animal"].astype(str)
    raw["Syllable"] = pd.to_numeric(raw["Syllable"], errors="coerce").astype(int)
    s2c = syllable_to_cluster()
    raw["cluster"] = raw["Syllable"].map(s2c).fillna("")

    full_sequences = {
        str(animal): group.sort_values("time_bin")["cluster"].tolist()
        for animal, group in raw.groupby("Animal", sort=False)
    }

    idx = raw.groupby(["Animal", "time_bin"])["Percentage"].idxmax()
    pred = raw.loc[idx].sort_values(["Animal", "time_bin"]).reset_index(drop=True)
    meta = pred[["Animal", "group", "Experiment"]].drop_duplicates().reset_index(drop=True)
    return full_sequences, meta


def compute_diversity(sequences, meta):
    meta_map = meta.set_index("Animal").to_dict("index")
    rows = []
    for animal, seq in sequences.items():
        info = meta_map[animal]
        vals, counts = np.unique(seq, return_counts=True)
        counts_map = dict(zip(vals, counts))
        order = sorted(counts_map)
        p = np.array([counts_map.get(c, 0) for c in order], dtype=float)
        p = p / p.sum()
        nonzero = p[p > 0]
        shannon = entropy(nonzero)
        evenness = shannon / np.log(len(nonzero)) if len(nonzero) else np.nan
        simpson = 1 - np.sum(p**2)
        named_p = np.array([counts_map.get(c, 0) for c in order if str(c).strip() != ""], dtype=float)
        named_p = named_p / counts.sum()
        sorted_p = np.sort(named_p)[::-1]
        cum = np.cumsum(sorted_p)
        baseline = (len(cum) + 1) / (2 * len(cum))
        cui = (cum.mean() - baseline) / (1 - baseline)
        rows.append({
            "Animal": animal,
            "group": info["group"],
            "Experiment": info["Experiment"],
            "simpson": simpson,
            "shannon": shannon,
            "evenness": evenness,
            "cui": cui,
        })
    return pd.DataFrame(rows)


def compute_bouts(sequences, meta):
    meta_map = meta.set_index("Animal").to_dict("index")
    rows = []
    for animal, seq in sequences.items():
        info = meta_map[animal]
        prev = seq[0]; length = 1
        for c in seq[1:] + ["__END__"]:
            if c == prev:
                length += 1
                continue
            rows.append({"Animal": animal, "group": info["group"],
                          "Experiment": info["Experiment"],
                          "cluster": prev, "bout_duration": length * 0.25})
            prev = c; length = 1
    return pd.DataFrame(rows)


def _bout_lengths(seq):
    lengths = []
    prev = seq[0]; length = 1
    for c in seq[1:] + ["__END__"]:
        if c == prev:
            length += 1
        else:
            lengths.append(length)
            prev = c; length = 1
    return lengths


def recurrence_rate(seq):
    _, counts = np.unique(seq, return_counts=True)
    n = len(seq)
    return float(np.sum(counts * counts) / (n * n))


def determinism(seq, min_length=2):
    arr = np.asarray(seq)
    n = len(arr)
    total = 0; diag_sum = 0
    for offset in range(-n + 1, n):
        diag = (arr[:n-abs(offset)] == arr[abs(offset):]) if offset >= 0 else (arr[-offset:] == arr[:n+offset])
        if offset == 0:
            total += int(diag.sum()) - n
        else:
            total += int(diag.sum())
        run = 0
        for value in diag:
            if value:
                run += 1
            else:
                if run >= min_length:
                    diag_sum += run
                run = 0
        if run >= min_length:
            diag_sum += run
    return float(diag_sum / total) if total > 0 else 0.0


def lz_complexity(seq):
    token_map = {v: i for i, v in enumerate(pd.unique(pd.Series(seq)))}
    tokens = [token_map[x] for x in seq]
    n, i, c, k = len(tokens), 0, 1, 1
    while True:
        if i + k > n:
            break
        sub = tokens[i:i+k]
        found = any(tokens[j:j+k] == sub for j in range(i))
        if found:
            k += 1
            if i + k > n:
                c += 1; break
        else:
            c += 1; i += k; k = 1
        if i >= n:
            break
    return c


def markov_entropy(seq, smoothing_factor=0.01):
    states = list(seq)
    unique_states = list(pd.unique(pd.Series(states)))
    if len(unique_states) == 1:
        return 0.0
    idx = {s: i for i, s in enumerate(unique_states)}
    counts = np.zeros((len(unique_states), len(unique_states)), dtype=float)
    for a, b in zip(states, states[1:]):
        counts[idx[a], idx[b]] += 1
    counts += smoothing_factor
    probs = counts / counts.sum(axis=1, keepdims=True)
    value_counts = pd.Series(states).value_counts()
    stationary = np.array([value_counts.get(s, 0) for s in unique_states], dtype=float) / len(states)
    inner = np.array([-np.sum(row[row > 0] * np.log2(row[row > 0])) for row in probs])
    return float(np.sum(stationary * inner))


def compute_transitions(sequences, meta):
    meta_map = meta.set_index("Animal").to_dict("index")
    rows = []
    for animal, seq in sequences.items():
        info = meta_map[animal]
        rows.append({
            "Animal": animal,
            "group": info["group"],
            "Experiment": info["Experiment"],
            "lz": lz_complexity(seq),
            "recurrence": recurrence_rate(seq),
            "determinism": determinism(seq),
            "markov": markov_entropy(seq),
        })
    return pd.DataFrame(rows)


def fit_mixedlm(df, metric):
    """MixedLM with animal as grouping (1 obs/animal) + experiment fixed effect."""
    sub = df.dropna(subset=[metric]).copy()
    sub["Stress"] = (sub["group"] == "ELS").astype(float)
    sub["Exp"] = (sub["Experiment"] == "Exp3").astype(float)
    endog = sub[metric].values
    exog = np.column_stack([np.ones(len(sub)), sub["Stress"].values, sub["Exp"].values])
    groups = sub["Animal"].values
    try:
        model = MixedLM(endog, exog, groups=groups)
        result = model.fit(reml=False)
        beta = result.params[1]
        se = result.bse[1]
        z = result.tvalues[1]
        p = result.pvalues[1]
        n = int(model.nobs)
        return beta, se, z, p, n
    except Exception as e:
        print(f"    MixedLM error: {e}")
        return None, None, None, None, None


def fit_ols(df, metric):
    """OLS with group + experiment (matches figure_4.py export_source_data)."""
    sub = df.dropna(subset=[metric]).copy()
    sub["group"] = pd.Categorical(sub["group"], categories=["Control", "ELS"])
    try:
        model = smf.ols(
            f"{metric} ~ C(group, Treatment('Control')) + Experiment",
            data=sub
        ).fit()
        param_key = "C(group, Treatment('Control'))[T.ELS]"
        beta = model.params[param_key]
        se = model.bse[param_key]
        t = model.tvalues[param_key]
        p = model.pvalues[param_key]
        n = int(model.nobs)
        return beta, se, t, p, n
    except Exception as e:
        return None, None, None, None, None


def tag(val, gold, tol_rel=0.01):
    if val is None or gold is None:
        return "N/A"
    if abs(val - gold) <= tol_rel * abs(gold) + 1e-10:
        return "CONFIRMED"
    elif abs(val - gold) <= 0.05 * abs(gold) + 1e-6:
        return "CLOSE"
    else:
        return "NOT REPRODUCED"


def compare(name, beta, se, z, p, gold_dict, n=None):
    g = gold_dict
    print(f"  beta: {beta:.6f}  gold: {g['beta']:.6f}  [{tag(beta, g['beta'])}]")
    print(f"  SE:   {se:.6f}  gold: {g['se']:.6f}  [{tag(se, g['se'])}]")
    print(f"  z:    {z:.6f}  gold: {g['z']:.6f}  [{tag(z, g['z'])}]")
    print(f"  p:    {p:.6f}  gold: {g['p']:.6f}  [{tag(p, g['p'])}]")
    if n: print(f"  n={n}")


def main():
    print("Loading sequences...")
    sequences, meta = load_sequences()
    print(f"  Animals: {len(sequences)}")
    print(f"  Groups: {meta['group'].value_counts().to_dict()}")
    print(f"  Experiments: {meta['Experiment'].value_counts().to_dict()}")

    # --- Diversity metrics ---
    print("\n=== DIVERSITY METRICS (MixedLM vs OLS) ===")
    metrics_df = compute_diversity(sequences, meta)
    for metric in ["simpson", "shannon", "evenness", "cui"]:
        print(f"\n--- {metric.upper()} ---")
        beta_m, se_m, z_m, p_m, n_m = fit_mixedlm(metrics_df, metric)
        beta_o, se_o, t_o, p_o, n_o = fit_ols(metrics_df, metric)
        print(f"  [MixedLM]  beta={beta_m:.6f}  SE={se_m:.6f}  z={z_m:.6f}  p={p_m:.6f}  n={n_m}")
        print(f"  [OLS]      beta={beta_o:.6f}  SE={se_o:.6f}  t={t_o:.6f}  p={p_o:.6f}  n={n_o}")
        g = GOLD["diversity"][metric]
        print(f"  [Report]   beta={g['beta']:.6f}  SE={g['se']:.6f}  z={g['z']:.6f}  p={g['p']:.6f}")
        # Tag both
        for label, b, s, v in [("MixedLM", beta_m, se_m, z_m), ("OLS", beta_o, se_o, t_o)]:
            bt = tag(b, g["beta"]); st = tag(s, g["se"]); vt = tag(v, g["z"]); pt = tag(p_m if label=="MixedLM" else p_o, g["p"])
            print(f"  [{label}] => beta:{bt} SE:{st} stat:{vt} p:{pt}")

    # --- Bout durations ---
    print("\n=== BOUT DURATIONS ===")
    bouts_df = compute_bouts(sequences, meta)
    overall = bouts_df.groupby(["Animal", "group", "Experiment"])["bout_duration"].mean().reset_index()
    cluster_means = (
        bouts_df[bouts_df["cluster"].isin(DISPLAY_ORDER)]
        .groupby(["Animal", "group", "Experiment", "cluster"], as_index=False)["bout_duration"]
        .mean()
    )
    # Overall
    print("\n--- Overall ---")
    beta_m, se_m, z_m, p_m, n_m = fit_mixedlm(overall.rename(columns={"bout_duration": "bout"}), "bout")
    g = GOLD["bout"]["Overall"]
    print(f"  [MixedLM]  beta={beta_m:.6f}  SE={se_m:.6f}  z={z_m:.6f}  p={p_m:.6f}  n={n_m}")
    print(f"  [Report]   beta={g['beta']:.6f}  SE={g['se']:.6f}  z={g['z']:.6f}  p={g['p']:.6f}")
    print(f"  => beta:{tag(beta_m, g['beta'])} SE:{tag(se_m, g['se'])} z:{tag(z_m, g['z'])} p:{tag(p_m, g['p'])}")

    for cluster in DISPLAY_ORDER:
        print(f"\n--- {cluster} ---")
        sub = cluster_means[cluster_means["cluster"] == cluster].copy()
        beta_m, se_m, z_m, p_m, n_m = fit_mixedlm(sub.rename(columns={"bout_duration": "bout"}), "bout")
        g = GOLD["bout"][cluster]
        print(f"  [MixedLM]  beta={beta_m:.6f}  SE={se_m:.6f}  z={z_m:.6f}  p={p_m:.6f}  n={n_m}")
        print(f"  [Report]   beta={g['beta']:.6f}  SE={g['se']:.6f}  z={g['z']:.6f}  p={g['p']:.6f}")
        print(f"  => beta:{tag(beta_m, g['beta'])} SE:{tag(se_m, g['se'])} z:{tag(z_m, g['z'])} p:{tag(p_m, g['p'])}")

    # --- Transition metrics ---
    print("\n=== TRANSITION METRICS ===")
    trans_df = compute_transitions(sequences, meta)
    for metric in ["lz", "recurrence", "determinism", "markov"]:
        print(f"\n--- {metric.upper()} ---")
        beta_m, se_m, z_m, p_m, n_m = fit_mixedlm(trans_df, metric)
        g = GOLD["transition"][metric]
        print(f"  [MixedLM]  beta={beta_m:.6f}  SE={se_m:.6f}  z={z_m:.6f}  p={p_m:.6f}  n={n_m}")
        print(f"  [Report]   beta={g['beta']:.6f}  SE={g['se']:.6f}  z={g['z']:.6f}  p={g['p']:.6f}")
        print(f"  => beta:{tag(beta_m, g['beta'])} SE:{tag(se_m, g['se'])} z:{tag(z_m, g['z'])} p:{tag(p_m, g['p'])}")

    # Save comparison tables
    print("\nSaving results...")
    rows = []
    metrics_df2 = compute_diversity(sequences, meta)
    for metric in ["simpson", "shannon", "evenness", "cui"]:
        beta_m, se_m, z_m, p_m, n_m = fit_mixedlm(metrics_df2, metric)
        g = GOLD["diversity"][metric]
        rows.append({"category": "diversity", "metric": metric,
                     "beta": beta_m, "se": se_m, "stat": z_m, "p": p_m, "n": n_m,
                     "gold_beta": g["beta"], "gold_se": g["se"], "gold_z": g["z"], "gold_p": g["p"],
                     "beta_tag": tag(beta_m, g["beta"]), "p_tag": tag(p_m, g["p"])})
    overall2 = compute_bouts(sequences, meta).groupby(["Animal","group","Experiment"])["bout_duration"].mean().reset_index()
    b, s, z, p, n = fit_mixedlm(overall2.rename(columns={"bout_duration":"bout"}), "bout")
    g = GOLD["bout"]["Overall"]
    rows.append({"category":"bout","metric":"Overall","beta":b,"se":s,"stat":z,"p":p,"n":n,
                 "gold_beta":g["beta"],"gold_se":g["se"],"gold_z":g["z"],"gold_p":g["p"],
                 "beta_tag":tag(b,g["beta"]),"p_tag":tag(p,g["p"])})
    bouts2 = compute_bouts(sequences, meta)
    cm2 = bouts2[bouts2["cluster"].isin(DISPLAY_ORDER)].groupby(["Animal","group","Experiment","cluster"],as_index=False)["bout_duration"].mean()
    for cluster in DISPLAY_ORDER:
        sub = cm2[cm2["cluster"]==cluster].copy()
        b, s, z, p, n = fit_mixedlm(sub.rename(columns={"bout_duration":"bout"}), "bout")
        g = GOLD["bout"][cluster]
        rows.append({"category":"bout","metric":cluster,"beta":b,"se":s,"stat":z,"p":p,"n":n,
                     "gold_beta":g["beta"],"gold_se":g["se"],"gold_z":g["z"],"gold_p":g["p"],
                     "beta_tag":tag(b,g["beta"]),"p_tag":tag(p,g["p"])})
    trans2 = compute_transitions(sequences, meta)
    for metric in ["lz","recurrence","determinism","markov"]:
        b, s, z, p, n = fit_mixedlm(trans2, metric)
        g = GOLD["transition"][metric]
        rows.append({"category":"transition","metric":metric,"beta":b,"se":s,"stat":z,"p":p,"n":n,
                     "gold_beta":g["beta"],"gold_se":g["se"],"gold_z":g["z"],"gold_p":g["p"],
                     "beta_tag":tag(b,g["beta"]),"p_tag":tag(p,g["p"])})
    out_path = REPO / "results" / "statistical_reports" / "fig4_verification_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
