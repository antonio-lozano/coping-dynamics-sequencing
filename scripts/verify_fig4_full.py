"""
Figure 4 full statistical verification.
Runs OLS(metric ~ group + Experiment) for diversity, bout duration, and transition metrics.
Compares to Report.xlsx gold standard (betas should match exactly; SE differs by model).
Saves results CSV for the R Excel report builder.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import SYLLABLE_TIMEBIN_250MS

CLUSTER_MAP = {
    "Freezing": [0, 28], "Sniffing": [18, 20], "Grooming": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27], "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climbing": [111], "Jump": [23, 29, 30, 34],
}
DISPLAY_ORDER = ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]

GOLD = {
    "diversity": {
        "simpson":  {"beta": -0.013440869, "se": 0.006592931, "z": -2.038678748, "p": 0.0414821,   "n": 82},
        "shannon":  {"beta": -0.011573886, "se": 0.018343103, "z": -0.630966653, "p": 0.528062329, "n": 82},
        "evenness": {"beta": -0.006497065, "se": 0.009731452, "z": -0.667635767, "p": 0.504366121, "n": 82},
        "cui":      {"beta":  0.063645788, "se": 0.021445504, "z":  2.967791679, "p": 0.002999475, "n": 82},
    },
    "bout": {
        "Overall":    {"beta":  0.044643567, "se": 0.034937455, "z":  1.277813938, "p": 0.201315039, "n": 82},
        "Freezing":   {"beta": -0.105173988, "se": 0.052566191, "z": -2.00079151,  "p": 0.045414863, "n": 82},
        "Sniffing":   {"beta":  0.322323093, "se": 0.128634553, "z":  2.505727158, "p": 0.012219988, "n": 82},
        "Grooming":   {"beta":  0.013740377, "se": 0.010360041, "z":  1.326286001, "p": 0.184744982, "n": 48},
        "Turn":       {"beta":  0.159255617, "se": 0.051574764, "z":  3.087859328, "p": 0.002016039, "n": 82},
        "Locomotion": {"beta":  0.003169017, "se": 0.015414888, "z":  0.205581577, "p": 0.837117769, "n": 82},
        "Climbing":   {"beta":  0.017676532, "se": 0.038136176, "z":  0.463510856, "p": 0.642998235, "n": 81},
        "Jump":       {"beta": -0.015732322, "se": 0.024630672, "z": -0.638728906, "p": 0.522999305, "n": 80},
    },
    "transition": {
        "lz":          {"beta": -8.048780488, "se": 5.56672732,  "z": -1.445872956, "p": 0.148212838, "n": 82},
        "recurrence":  {"beta":  0.013440869, "se": 0.00593759,  "z":  2.263690785, "p": 0.02359314,  "n": 82},
        "determinism": {"beta":  0.016104788, "se": 0.004761667, "z":  3.382174516, "p": 0.000719144, "n": 82},
        "markov":      {"beta": -0.041186979, "se": 0.017585398, "z": -2.342112429, "p": 0.019174938, "n": 82},
    }
}


def load_data():
    raw = pd.read_csv(SYLLABLE_TIMEBIN_250MS)
    raw = raw.rename(columns={"Time Bin": "time_bin", "Condition": "group"})
    raw = raw[raw["group"].isin(["Control", "ELS"])].copy()
    raw["Animal"] = raw["Animal"].astype(str)
    raw["Syllable"] = pd.to_numeric(raw["Syllable"], errors="coerce").astype(int)
    s2c = {}
    for c, ss in CLUSTER_MAP.items():
        for s in ss: s2c[int(s)] = c
    raw["cluster"] = raw["Syllable"].map(s2c).fillna("")
    idx = raw.groupby(["Animal", "time_bin"])["Percentage"].idxmax()
    pred = raw.loc[idx].sort_values(["Animal", "time_bin"]).reset_index(drop=True)
    meta = pred[["Animal", "group", "Experiment"]].drop_duplicates().reset_index(drop=True)
    full_sequences = {str(a): grp.sort_values("time_bin")["cluster"].tolist()
                      for a, grp in raw.groupby("Animal", sort=False)}
    return full_sequences, meta


def compute_diversity(sequences, meta):
    meta_map = meta.set_index("Animal").to_dict("index")
    rows = []
    for animal, seq in sequences.items():
        info = meta_map[animal]
        vals, counts = np.unique(seq, return_counts=True)
        p = counts / counts.sum()
        nonzero = p[p > 0]
        shannon = entropy(nonzero)
        evenness = shannon / np.log(len(nonzero)) if len(nonzero) > 1 else np.nan
        simpson = 1 - np.sum(p**2)
        named_p = np.array([counts[list(vals).index(c)] for c in vals if str(c).strip() != ""], dtype=float)
        named_p = named_p / counts.sum()
        sorted_p = np.sort(named_p)[::-1]
        cum = np.cumsum(sorted_p)
        baseline = (len(cum) + 1) / (2 * len(cum))
        cui = (cum.mean() - baseline) / (1 - baseline)
        rows.append({"Animal": animal, "group": info["group"],
                     "Experiment": str(info["Experiment"]),
                     "simpson": simpson, "shannon": shannon, "evenness": evenness, "cui": cui})
    return pd.DataFrame(rows)


def compute_bouts(sequences, meta):
    meta_map = meta.set_index("Animal").to_dict("index")
    rows = []
    for animal, seq in sequences.items():
        info = meta_map[animal]
        prev = seq[0]; length = 1
        for c in seq[1:] + ["__END__"]:
            if c == prev: length += 1
            else:
                rows.append({"Animal": animal, "group": info["group"],
                             "Experiment": str(info["Experiment"]),
                             "cluster": prev, "bout_duration": length * 0.25})
                prev = c; length = 1
    return pd.DataFrame(rows)


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
        total += (int(diag.sum()) - n) if offset == 0 else int(diag.sum())
        run = 0
        for value in diag:
            if value: run += 1
            else:
                if run >= min_length: diag_sum += run
                run = 0
        if run >= min_length: diag_sum += run
    return float(diag_sum / total) if total > 0 else 0.0


def lz_complexity(seq):
    token_map = {v: i for i, v in enumerate(pd.unique(pd.Series(seq)))}
    tokens = [token_map[x] for x in seq]
    n, i, c, k = len(tokens), 0, 1, 1
    while True:
        if i + k > n: break
        sub = tokens[i:i+k]
        found = any(tokens[j:j+k] == sub for j in range(i))
        if found:
            k += 1
            if i + k > n: c += 1; break
        else:
            c += 1; i += k; k = 1
        if i >= n: break
    return c


def markov_entropy(seq, smoothing_factor=0.01):
    states = list(seq)
    unique_states = list(pd.unique(pd.Series(states)))
    if len(unique_states) == 1: return 0.0
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
        rows.append({"Animal": animal, "group": info["group"],
                     "Experiment": str(info["Experiment"]),
                     "lz": lz_complexity(seq),
                     "recurrence": recurrence_rate(seq),
                     "determinism": determinism(seq),
                     "markov": markov_entropy(seq)})
    return pd.DataFrame(rows)


def fit_ols(df, metric):
    sub = df.dropna(subset=[metric]).copy()
    sub["group_cat"] = pd.Categorical(sub["group"], categories=["Control", "ELS"])
    model = smf.ols(f"{metric} ~ C(group_cat) + Experiment", data=sub).fit()
    pk = "C(group_cat)[T.ELS]"
    return model.params[pk], model.bse[pk], model.tvalues[pk], model.pvalues[pk], int(model.nobs)


def tag_val(val, gold, rel_tol=0.01, abs_tol=1e-6):
    if val is None or gold is None: return "N/A"
    if abs(val - gold) <= rel_tol * abs(gold) + abs_tol: return "CONFIRMED"
    if abs(val - gold) <= 0.05 * abs(gold) + 1e-4: return "CLOSE"
    return "NOT REPRODUCED"


def print_compare(metric, beta, se, stat, p, n, gold):
    g = gold
    print(f"  beta: {beta:+.6f}  gold: {g['beta']:+.6f}  [{tag_val(beta, g['beta'])}]")
    print(f"  SE:   {se:.6f}   gold: {g['se']:.6f}  [{tag_val(se, g['se'])}]")
    print(f"  stat: {stat:+.6f}  gold: {g['z']:+.6f}  [{tag_val(stat, g['z'])}]")
    print(f"  p:    {p:.6f}  gold: {g['p']:.6f}  [{tag_val(p, g['p'])}]")
    print(f"  n:    {n}  gold: {g['n']}")


def main():
    print("Loading data...")
    sequences, meta = load_data()
    print(f"  n animals: {len(sequences)}, groups: {meta['group'].value_counts().to_dict()}")

    results = []

    # --- Diversity metrics ---
    print("\n=== DIVERSITY METRICS ===")
    div_df = compute_diversity(sequences, meta)
    for metric in ["simpson", "shannon", "evenness", "cui"]:
        print(f"\n--- {metric.upper()} ---")
        beta, se, stat, p, n = fit_ols(div_df, metric)
        g = GOLD["diversity"][metric]
        print_compare(metric, beta, se, stat, p, n, g)
        results.append({"category": "diversity", "metric": metric,
                        "beta": beta, "se": se, "stat": stat, "p": p, "n": n,
                        "gold_beta": g["beta"], "gold_se": g["se"], "gold_z": g["z"], "gold_p": g["p"],
                        "beta_tag": tag_val(beta, g["beta"]), "se_tag": tag_val(se, g["se"]),
                        "p_tag": tag_val(p, g["p"])})

    # --- Bout durations ---
    print("\n=== BOUT DURATIONS ===")
    bouts_df = compute_bouts(sequences, meta)
    overall = bouts_df.groupby(["Animal", "group", "Experiment"])["bout_duration"].mean().reset_index()
    print("\n--- Overall ---")
    beta, se, stat, p, n = fit_ols(overall.rename(columns={"bout_duration": "bval"}), "bval")
    g = GOLD["bout"]["Overall"]
    print_compare("Overall", beta, se, stat, p, n, g)
    results.append({"category": "bout", "metric": "Overall",
                    "beta": beta, "se": se, "stat": stat, "p": p, "n": n,
                    "gold_beta": g["beta"], "gold_se": g["se"], "gold_z": g["z"], "gold_p": g["p"],
                    "beta_tag": tag_val(beta, g["beta"]), "se_tag": tag_val(se, g["se"]),
                    "p_tag": tag_val(p, g["p"])})

    cluster_means = (bouts_df[bouts_df["cluster"].isin(DISPLAY_ORDER)]
                     .groupby(["Animal", "group", "Experiment", "cluster"], as_index=False)["bout_duration"].mean())
    for cluster in DISPLAY_ORDER:
        print(f"\n--- {cluster} ---")
        sub = cluster_means[cluster_means["cluster"] == cluster].copy()
        sub["bval"] = sub["bout_duration"]
        beta, se, stat, p, n = fit_ols(sub, "bval")
        g = GOLD["bout"][cluster]
        print_compare(cluster, beta, se, stat, p, n, g)
        results.append({"category": "bout", "metric": cluster,
                        "beta": beta, "se": se, "stat": stat, "p": p, "n": n,
                        "gold_beta": g["beta"], "gold_se": g["se"], "gold_z": g["z"], "gold_p": g["p"],
                        "beta_tag": tag_val(beta, g["beta"]), "se_tag": tag_val(se, g["se"]),
                        "p_tag": tag_val(p, g["p"])})

    # --- Transition metrics ---
    print("\n=== TRANSITION METRICS ===")
    trans_df = compute_transitions(sequences, meta)
    for metric in ["lz", "recurrence", "determinism", "markov"]:
        print(f"\n--- {metric.upper()} ---")
        beta, se, stat, p, n = fit_ols(trans_df, metric)
        g = GOLD["transition"][metric]
        print_compare(metric, beta, se, stat, p, n, g)
        results.append({"category": "transition", "metric": metric,
                        "beta": beta, "se": se, "stat": stat, "p": p, "n": n,
                        "gold_beta": g["beta"], "gold_se": g["se"], "gold_z": g["z"], "gold_p": g["p"],
                        "beta_tag": tag_val(beta, g["beta"]), "se_tag": tag_val(se, g["se"]),
                        "p_tag": tag_val(p, g["p"])})

    # Save
    out_dir = REPO / "results" / "statistical_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig4_verification_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Also save per-animal data for Excel report
    div_df["Stress"] = (div_df["group"] == "ELS").astype(int)
    div_df.to_csv(out_dir / "fig4_diversity_per_animal.csv", index=False)
    overall.rename(columns={"bout_duration": "bout_mean"}).to_csv(out_dir / "fig4_bout_overall_per_animal.csv", index=False)
    cluster_means.to_csv(out_dir / "fig4_bout_cluster_per_animal.csv", index=False)
    trans_df.to_csv(out_dir / "fig4_transition_per_animal.csv", index=False)
    print("Saved per-animal data CSVs.")


if __name__ == "__main__":
    main()
