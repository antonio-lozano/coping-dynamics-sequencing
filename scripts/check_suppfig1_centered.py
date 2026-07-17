"""Test centered vs uncentered time for Supp Fig 1 — why beta=-2.341 vs -2.147."""
import sys, warnings, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, "d:/coping-dynamics-sequencing")
warnings.filterwarnings("ignore")

wide = pd.read_excel(
    "d:/coping-dynamics-sequencing/data/Raw_data.xlsx",
    sheet_name="Supp_cluster_time", header=2
)
wide = wide.dropna(subset=["animal_id", "group", "cluster"]).copy()
time_cols = [c for c in wide.columns if str(c).startswith("t_")]
long = wide.melt(
    id_vars=["animal_id", "group", "project", "experiment", "cluster"],
    value_vars=time_cols, var_name="time_bin", value_name="percentage"
)
long["time_bin_numeric"] = long["time_bin"].str.extract(r"(\d+)").astype(int)
long["percentage"] = pd.to_numeric(long["percentage"], errors="coerce")
long = long[long["group"].isin(["Control", "ELS"])].copy()

mean_t = long["time_bin_numeric"].mean()
long["time_c"] = long["time_bin_numeric"] - mean_t
print(f"Mean time_bin_numeric = {mean_t}s  -> centering at {mean_t}")
print("Manuscript: beta=-2.341, SE=0.942, z=-2.486, p=0.013")
print()

for cl in ["Inaccurate tracking", "Mix behaviors"]:
    sub = long[long["cluster"] == cl].copy()
    n_a = sub["animal_id"].nunique()
    print(f"=== {cl}  (n={n_a} animals) ===")

    # Uncentered (same as fit_mixed_models)
    try:
        m1 = smf.mixedlm(
            "percentage ~ C(group, Treatment('Control')) * time_bin_numeric",
            sub, groups=sub["animal_id"],
            vc_formula={"experiment": "0 + C(experiment)"}
        ).fit(reml=False)
        p1 = "C(group, Treatment('Control'))[T.ELS]"
        b1, se1 = m1.params[p1], m1.bse[p1]
        z1 = b1 / se1
        pv1 = m1.pvalues[p1]
        print(f"  Uncentered  (at t=0):    beta={b1:.4f}  SE={se1:.4f}  z={z1:.4f}  p={pv1:.4f}")
    except Exception as e:
        print(f"  Uncentered ERROR: {e}")

    # Centered at mean time
    try:
        m2 = smf.mixedlm(
            "percentage ~ C(group, Treatment('Control')) * time_c",
            sub, groups=sub["animal_id"],
            vc_formula={"experiment": "0 + C(experiment)"}
        ).fit(reml=False)
        p2 = "C(group, Treatment('Control'))[T.ELS]"
        b2, se2 = m2.params[p2], m2.bse[p2]
        z2 = b2 / se2
        pv2 = m2.pvalues[p2]
        print(f"  Centered    (at t=240s): beta={b2:.4f}  SE={se2:.4f}  z={z2:.4f}  p={pv2:.4f}")
        if abs(b2 - (-2.341)) < 0.05:
            print("    *** beta close to manuscript -2.341 ***")
        if abs(se2 - 0.942) < 0.05:
            print("    *** SE close to manuscript 0.942 ***")
        if abs(z2 - (-2.486)) < 0.05:
            print("    *** z close to manuscript -2.486 ***")
    except Exception as e:
        print(f"  Centered ERROR: {e}")

    # Simple per-animal mean (no time)
    try:
        per_animal = sub.groupby(["animal_id", "group", "experiment"])["percentage"].mean().reset_index()
        m3 = smf.mixedlm(
            "percentage ~ C(group, Treatment('Control'))",
            per_animal, groups=per_animal["animal_id"],
            vc_formula={"experiment": "0 + C(experiment)"}
        ).fit(reml=False)
        p3 = "C(group, Treatment('Control'))[T.ELS]"
        b3, se3 = m3.params[p3], m3.bse[p3]
        z3 = b3 / se3
        pv3 = m3.pvalues[p3]
        print(f"  Per-animal mean (no time): beta={b3:.4f}  SE={se3:.4f}  z={z3:.4f}  p={pv3:.4f}")
        if abs(b3 - (-2.341)) < 0.05:
            print("    *** beta close to manuscript -2.341 ***")
        if abs(se3 - 0.942) < 0.05:
            print("    *** SE close to manuscript 0.942 ***")
    except Exception as e:
        print(f"  Per-animal mean ERROR: {e}")

    print()
