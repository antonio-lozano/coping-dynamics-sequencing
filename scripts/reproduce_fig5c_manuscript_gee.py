"""Reproduce the manuscript's Figure 5C cluster-frequency GEE and document a
transcription error in the reported "resilient vs vulnerable" statistics.

WHAT THE MANUSCRIPT DID
-----------------------
The original analysis (E:\\#Deeplabcut_project\\Data\\Code\\Frequency_resilience.py)
fits, per behavioral cluster, ONE GEE Negative-Binomial model on per-animal
syllable counts normalized to a common total of 11250 frames:

    Count ~ New_condition + Experiment
    family      = NegativeBinomial(alpha=1.0)
    cov_struct  = Exchangeable          (robust / population-averaged SE)
    groups      = Animal
    reference   = Control               (so New_condition[T.ELS], [T.ELS_resilient])

Input: data/source/updated_results.pkl.gz  (bundled in this repo). Each record
holds an "Animal", "Experiment" and a "syllable" array of cluster codes 1-7
(Freezing=1 ... Jump=7). This script is fully self-contained — no E: drive needed.

THE ERROR
---------
The manuscript's Fig 5C paragraph reports the "resilient vs vulnerable" effect
for every cluster, but the numbers it cites are actually the **Experiment**
(Exp1 vs Exp3) covariate row of each model — NOT the resilient-vs-vulnerable
contrast. (The single "comparable to controls" number it cites IS correct: that
is the New_condition[T.ELS_resilient] row.)

The correct resilient-vs-vulnerable contrast is New_condition[T.ELS] minus
New_condition[T.ELS_resilient] (computed here via t_test, as in the original
script). Only Freezing is significant; sniffing/grooming/locomotion/climbing are
non-significant, and grooming/locomotion are even directionally opposite to the
manuscript text.

The GEE is deterministic — there is no random seed; these values are exact and
repeatable.

Run from repo root:
    python scripts/reproduce_fig5c_manuscript_gee.py
Writes results/statistical_reports/fig5c_manuscript_vs_correct.csv
"""
import warnings, gzip, pickle
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
PKL  = REPO / "data" / "source" / "updated_results.pkl.gz"
OUT  = REPO / "results" / "statistical_reports" / "fig5c_manuscript_vs_correct.csv"
OUT_FIXED = REPO / "results" / "statistical_reports" / "fig5c_corrected_report.csv"

NORMALIZE_TARGET = 11250
SYLL = {1: "Freezing", 2: "Sniffing", 3: "Grooming", 4: "Turn",
        5: "Locomotion", 6: "Climbing", 7: "Jump"}

# Resilience classification (12 resilient ELS, 29 vulnerable ELS) — matches
# figure_5_dynamics_scores.csv group_ext and the original Frequency_resilience.py.
ELS_RESILIENT = {"2.4", "15.4", "26.4", "43.3", "43.5", "49.6",
                 "123.3", "134.2", "148.4", "150.3", "150.5", "159.4"}
ELS_VULNERABLE = {"15.6", "23.4", "25.1", "32.2", "32.4", "49.5", "55.3", "55.5",
                  "56.3", "63.3", "69.1", "69.2", "69.3", "69.5", "73.1", "73.4",
                  "75.1", "88.2", "88.3", "128.4", "133.3", "123.1", "134.4",
                  "144.3", "144.5", "153.4", "154.4", "159.3", "159.5"}

# Manuscript Fig 5C "resilient vs vulnerable" values, as printed in the paper.
# (Only 5 of the 7 clusters are reported; Turn and Jump are not mentioned.)
MANUSCRIPT = {
    "Freezing":   (0.180, 0.039, 4.681),
    "Sniffing":   (0.492, 0.119, 4.139),
    "Grooming":   (-0.398, 0.187, -2.122),
    "Locomotion": (-0.425, 0.089, -4.755),
    "Climbing":   (-0.196, 0.075, -2.624),
}

# BH-FDR family = the 7 behavioral clusters (one resilient-vs-vulnerable test each),
# NOT 14 (i.e. not also counting the resilient-vs-control contrast).
ALL_CLUSTERS = ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]


def classify(animal, original):
    if animal in ELS_RESILIENT:
        return "ELS_resilient"
    if animal in ELS_VULNERABLE:
        return "ELS"
    return original


def build_dataframe():
    with gzip.open(PKL, "rb") as fh:
        results = pickle.load(fh)
    records = results.values() if isinstance(results, dict) else results
    rows = []
    for rec in records:
        animal = str(rec.get("Animal", "?"))
        cond = classify(animal, rec.get("Condition", rec.get("New_condition", "?")))
        exp = rec.get("Experiment", "?")
        uniq, counts = np.unique(rec.get("syllable", []), return_counts=True)
        total = counts.sum()
        norm = counts / total * NORMALIZE_TARGET if total else counts
        for code, n in zip(uniq, norm):
            if code in SYLL:
                rows.append({"Animal": animal, "Experiment": exp,
                             "New_condition": cond, "Syllable": SYLL[code], "Count": n})
    return pd.DataFrame(rows)


def fit_cluster(df, cluster):
    sub = df[df["Syllable"] == cluster].copy()
    return smf.gee("Count ~ New_condition + Experiment", data=sub, groups=sub["Animal"],
                   family=sm.families.NegativeBinomial(alpha=1.0),
                   cov_struct=sm.cov_struct.Exchangeable()).fit()


def resilient_minus_vulnerable(model):
    """Correct contrast: New_condition[T.ELS_resilient] - New_condition[T.ELS]."""
    names = list(model.params.index)
    vec = np.zeros(len(names))
    vec[names.index("New_condition[T.ELS]")] = 1          # ELS
    vec[names.index("New_condition[T.ELS_resilient]")] = -1   # - resilient
    t = model.t_test(vec)                                  # gives ELS - resilient
    beta = -float(np.ravel(t.effect)[0])                  # flip -> resilient - vulnerable
    se = float(np.ravel(t.sd)[0])
    z = -float(np.ravel(t.tvalue)[0])
    p = float(np.ravel(t.pvalue)[0])
    return beta, se, z, p


def main():
    df = build_dataframe()
    n_by_group = df.drop_duplicates("Animal")["New_condition"].value_counts().to_dict()
    print(f"Source: {PKL.relative_to(REPO)}  |  n animals = {df.Animal.nunique()}  |  {n_by_group}")
    print("GEE Negative Binomial is deterministic (no random seed).\n")

    rows = []
    report_rows = []   # corrected, properly-labeled report (mirrors Report.xlsx layout)
    for cl in ALL_CLUSTERS:
        m = fit_cluster(df, cl)
        ic_b, ic_se, ic_z, ic_p = m.params["Intercept"], m.bse["Intercept"], m.tvalues["Intercept"], float(m.pvalues["Intercept"])
        ke = "New_condition[T.ELS]"
        ce_b, ce_se, ce_z, ce_p = m.params[ke], m.bse[ke], m.tvalues[ke], float(m.pvalues[ke])
        exp_b, exp_se, exp_z = m.params["Experiment"], m.bse["Experiment"], m.tvalues["Experiment"]
        exp_p = float(m.pvalues["Experiment"])
        kr = "New_condition[T.ELS_resilient]"
        rc_b, rc_se, rc_z, rc_p = m.params[kr], m.bse[kr], m.tvalues[kr], float(m.pvalues[kr])
        cb, cse, cz, cp = resilient_minus_vulnerable(m)
        man = MANUSCRIPT.get(cl)

        # --- corrected report rows (labels attached to the RIGHT values) ---
        report_rows += [
            {"cluster": cl, "Parameter": "Intercept",            "Coef.": ic_b, "Std. Err.": ic_se, "z": ic_z, "P>|z|": ic_p},
            {"cluster": cl, "Parameter": "Control/ELS",          "Coef.": ce_b, "Std. Err.": ce_se, "z": ce_z, "P>|z|": ce_p},
            {"cluster": cl, "Parameter": "Control/Resilient",    "Coef.": rc_b, "Std. Err.": rc_se, "z": rc_z, "P>|z|": rc_p},
            # FIXED: this row now holds the real resilient-vs-vulnerable contrast (resilient - vulnerable)
            {"cluster": cl, "Parameter": "ELS/Resilient",        "Coef.": cb,   "Std. Err.": cse,   "z": cz,   "P>|z|": cp},
            # FIXED: this row now holds the Experiment (Exp1 vs Exp3) coefficient
            {"cluster": cl, "Parameter": "Experiment",           "Coef.": exp_b,"Std. Err.": exp_se,"z": exp_z,"P>|z|": exp_p},
        ]

        print(f"=== {cl} (n_obs={int(m.nobs)}) ===")
        if man:
            mb, mse, mz = man
            print(f"  manuscript 'resilient vs vulnerable' : beta={mb:+.3f} SE={mse:.3f} z={mz:+.3f}")
            print(f"    -> is actually the EXPERIMENT row   : beta={exp_b:+.3f} SE={exp_se:.3f} z={exp_z:+.3f} p={exp_p:.4g}   [mislabeled]")
        else:
            print(f"  (not reported in manuscript)         : EXPERIMENT row beta={exp_b:+.3f} SE={exp_se:.3f} z={exp_z:+.3f} p={exp_p:.4g}")
        print(f"  CORRECT resilient - vulnerable        : beta={cb:+.3f} SE={cse:.3f} z={cz:+.3f} p={cp:.4f}")
        print(f"  resilient - control (T.ELS_resilient) : beta={rc_b:+.3f} SE={rc_se:.3f} z={rc_z:+.3f} p={rc_p:.4f}\n")

        rows.append({
            "cluster": cl,
            "reported_in_manuscript": man is not None,
            "manuscript_beta": man[0] if man else np.nan,
            "manuscript_SE": man[1] if man else np.nan,
            "manuscript_z": man[2] if man else np.nan,
            "experiment_row_beta": round(exp_b, 4), "experiment_row_SE": round(exp_se, 4),
            "experiment_row_z": round(exp_z, 4), "experiment_row_p": exp_p,
            "manuscript_equals_experiment_row": (man is not None and abs(exp_b - man[0]) < 0.01 and
                                                 abs(exp_se - man[1]) < 0.01 and
                                                 abs(exp_z - man[2]) < 0.05),
            "correct_res_vs_vuln_beta": round(cb, 4), "correct_res_vs_vuln_SE": round(cse, 4),
            "correct_res_vs_vuln_z": round(cz, 4), "correct_res_vs_vuln_p": cp,
            "res_vs_control_beta": round(rc_b, 4), "res_vs_control_SE": round(rc_se, 4),
            "res_vs_control_z": round(rc_z, 4), "res_vs_control_p": rc_p,
        })

    out = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR across the family of 7 clusters (one resilient-vs-vulnerable
    # test per cluster). Applied to (a) the manuscript's reported quantity (= Experiment-row
    # p's) and (b) the correct resilient-vs-vulnerable contrast p-values.
    out["experiment_row_p_BH"] = multipletests(out["experiment_row_p"], method="fdr_bh")[1]
    out["correct_res_vs_vuln_p_BH"] = multipletests(out["correct_res_vs_vuln_p"], method="fdr_bh")[1]
    out["correct_sig_BH_0.05"] = out["correct_res_vs_vuln_p_BH"] < 0.05

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("Benjamini-Hochberg FDR (family = 7 clusters, one res-vs-vuln test each):")
    print(f"  {'cluster':<11}{'Exp-row p':>12}{'Exp p_BH':>11}"
          f"{'correct p':>12}{'correct p_BH':>14}{'sig?':>6}")
    for _, r in out.iterrows():
        print(f"  {r.cluster:<11}{r.experiment_row_p:>12.2e}{r['experiment_row_p_BH']:>11.4f}"
              f"{r['correct_res_vs_vuln_p']:>12.4f}{r['correct_res_vs_vuln_p_BH']:>14.4f}"
              f"{('YES' if r['correct_sig_BH_0.05'] else 'no'):>6}")

    # --- corrected report (Report.xlsx layout, transposition FIXED) ---
    rep = pd.DataFrame(report_rows)
    # BH-FDR over the 7 ELS/Resilient contrast rows, written onto those rows only
    elsres_mask = rep["Parameter"] == "ELS/Resilient"
    rep["P_BH_FDR"] = np.nan
    rep.loc[elsres_mask, "P_BH_FDR"] = multipletests(rep.loc[elsres_mask, "P>|z|"], method="fdr_bh")[1]
    for c in ["Coef.", "Std. Err.", "z", "P>|z|", "P_BH_FDR"]:
        rep[c] = rep[c].astype(float).round(4)
    rep.to_csv(OUT_FIXED, index=False)

    reported = out[out["reported_in_manuscript"]]
    all_match = bool(reported["manuscript_equals_experiment_row"].all())
    print(f"\nAll {len(reported)} reported manuscript values equal the Experiment row: {all_match}")
    sig = ', '.join(out.loc[out['correct_sig_BH_0.05'], 'cluster']) or 'NONE'
    print(f"After BH-FDR over 7 clusters, real resilient-vs-vulnerable frequency effect(s): {sig}.")

    print("\nCORRECTED report (labels fixed; 'ELS/Resilient' = resilient - vulnerable contrast):")
    print(f"  {'cluster':<11}{'Parameter':<19}{'Coef.':>9}{'SE':>9}{'z':>9}{'P>|z|':>10}{'P_BH':>9}")
    for _, r in rep.iterrows():
        bh = "" if pd.isna(r["P_BH_FDR"]) else f"{r['P_BH_FDR']:>9.4f}"
        print(f"  {r.cluster:<11}{r.Parameter:<19}{r['Coef.']:>9.3f}{r['Std. Err.']:>9.3f}{r['z']:>9.3f}{r['P>|z|']:>10.4f}{bh}")

    print(f"\nsaved -> {OUT.relative_to(REPO)}")
    print(f"saved -> {OUT_FIXED.relative_to(REPO)}")


if __name__ == "__main__":
    main()
