"""
Verify Fig 2A-C freezing LMM statistics against manuscript and Report.xlsx.

Manuscript claims:
  Fig 2A (Exp1 - Sanguino-Gomez and Krugers 2024):
    time:       beta=3.768, SE=0.142, z=26.489, p<0.001
    ELS x time: beta=-0.887, SE=0.181, z=-4.898, p<0.001

  Fig 2B (Exp3 - Sanguino-Gomez et al 2024):
    time:       beta=4.412, SE=0.191, z=23.091, p<0.001
    ELS x time: beta=-0.721, SE=0.270, z=-2.667, p=0.008

  Combined (Fig 2C):
    ELS x time: beta=-0.822, SE=0.154, z=-5.327, p<0.001

Report.xlsx Ground_truth sheet:
  Exp1 time:       beta=3.721, SE=0.128, z=29.048   <-- DIFFERS from manuscript
  Exp3 time:       beta=4.412, SE=0.191, z=23.091   <-- matches
  Combined ELS*t:  beta=-0.822, SE=0.154, z=-5.327  <-- matches
"""
import sys
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

SOURCE = repo_root / "data" / "source"
FREEZING_DIR = SOURCE / "freezing_predictions"
INDEX_CSV = SOURCE / "animal_groups.csv"
FREQ_CSV = SOURCE / "cluster_frequency_per_animal.csv"
BIN_SECONDS = 30
FPS = 30

EXCLUDE = {"animal_48_6", "48_6"}


def _normalize(name):
    return str(name).strip().replace(" ", "_")


def _short_id(name):
    norm = _normalize(str(name))
    m = re.search(r"Animal[_ ]?(\d+(?:[_-]\d+)?)", norm, re.IGNORECASE)
    if m:
        return m.group(1).replace("_", ".").replace("-", ".")
    parts = norm.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 else norm


def _is_excluded(name):
    n = _normalize(name).lower()
    return n in EXCLUDE


# Load group map
idx = pd.read_csv(INDEX_CSV)
group_map = {}
for _, row in idx.iterrows():
    raw = str(row["name"]).strip()
    norm = _normalize(raw)
    grp = str(row["group"])
    group_map[raw] = grp
    group_map[norm] = grp
    sid = _short_id(raw)
    group_map[sid] = grp

# Experiment assignment
exp_map_df = pd.read_csv(FREQ_CSV)[["animal_id", "experiment"]].drop_duplicates()
exp_map = dict(zip(exp_map_df["animal_id"].astype(str), exp_map_df["experiment"]))

# Load freezing
records = []
bin_size = int(FPS * BIN_SECONDS)
for csv_path in sorted(FREEZING_DIR.glob("*_freezing_predictions_only.csv")):
    base = csv_path.name.replace("_freezing_predictions_only.csv", "")
    if _is_excluded(base):
        continue
    df = pd.read_csv(csv_path)
    if "Freezing_Jen_0-125_threshold" not in df.columns:
        continue
    norm_base = _normalize(base)
    sid = _short_id(base)
    grp = group_map.get(base) or group_map.get(norm_base) or group_map.get(sid) or "Unknown"
    if grp not in ("Control", "ELS"):
        continue
    frame = df["Unnamed: 0"] if "Unnamed: 0" in df.columns else np.arange(len(df))
    freeze = df["Freezing_Jen_0-125_threshold"].to_numpy(dtype=float)
    # Bin
    n_bins = len(freeze) // bin_size
    for b in range(n_bins):
        chunk = freeze[b * bin_size:(b + 1) * bin_size]
        records.append({
            "animal_id": sid,
            "group": grp,
            "time_bin": b + 1,
            "freezing_pct": chunk.mean() * 100.0,
        })

long_df = pd.DataFrame(records)

# Experiment assignment via cluster_frequency
def _get_exp(sid):
    try:
        v = float(sid.replace("_", "."))
        return exp_map.get(str(v), exp_map.get(sid, None))
    except (ValueError, TypeError):
        return exp_map.get(sid, None)

long_df["experiment"] = long_df["animal_id"].map(_get_exp)
missing_exp = long_df["experiment"].isna().sum()
if missing_exp > 0:
    print(f"WARNING: {missing_exp} rows with unknown experiment assignment")

long_df = long_df.dropna(subset=["experiment"])
long_df["experiment"] = long_df["experiment"].astype(int)

print(f"\nData: {long_df['animal_id'].nunique()} animals, {len(long_df)} observations")
print(f"Exp1 animals: {long_df[long_df['experiment']==1]['animal_id'].nunique()}")
print(f"Exp3 animals: {long_df[long_df['experiment']==3]['animal_id'].nunique()}")
print()

# Fit LMMs matching src/statistics.py fit_mixed_models()
long_df["condition"] = pd.Categorical(long_df["group"], categories=["Control", "ELS"])

def fit_lmm(sub, label, use_vc=False):
    formula = "freezing_pct ~ condition * time_bin"
    try:
        if use_vc:
            m = smf.mixedlm(formula, sub, groups=sub["animal_id"],
                            re_formula="1",
                            vc_formula={"experiment": "0 + C(experiment)"}).fit(reml=False)
        else:
            m = smf.mixedlm(formula, sub, groups=sub["animal_id"],
                            re_formula="1").fit(reml=False)
        print(f"=== {label} (n={sub['animal_id'].nunique()} animals, {len(sub)} obs) ===")
        params_of_interest = ["time_bin", "condition[T.ELS]:time_bin"]
        for p in m.params.index:
            if any(k in p for k in ["time_bin", "Intercept", "condition"]):
                ci = m.conf_int()
                ci_lo = ci.loc[p, 0] if p in ci.index else float("nan")
                ci_hi = ci.loc[p, 1] if p in ci.index else float("nan")
                print(f"  {p:40s}: beta={m.params[p]:.4f}, SE={m.bse[p]:.4f}, "
                      f"z={m.tvalues[p]:.4f}, p={m.pvalues[p]:.4f}, CI=[{ci_lo:.3f}, {ci_hi:.3f}]")
        print()
    except Exception as e:
        print(f"ERROR in {label}: {e}\n")

fit_lmm(long_df[long_df["experiment"] == 1].copy(), "Exp1 (Sanguino-Gomez & Krugers 2024)")
fit_lmm(long_df[long_df["experiment"] == 3].copy(), "Exp3 (Sanguino-Gomez et al 2024)")
fit_lmm(long_df.copy(), "Combined", use_vc=True)

print("\n--- MANUSCRIPT REPORTED VALUES ---")
print("Fig 2A Exp1 time:       beta=3.768, SE=0.142, z=26.489, p<0.001")
print("Fig 2A Exp1 ELS x time: beta=-0.887, SE=0.181, z=-4.898, p<0.001")
print("Fig 2B Exp3 time:       beta=4.412, SE=0.191, z=23.091, p<0.001")
print("Fig 2B Exp3 ELS x time: beta=-0.721, SE=0.270, z=-2.667, p=0.008")
print("Combined ELS x time:    beta=-0.822, SE=0.154, z=-5.327, p<0.001")
print("\n--- REPORT.xlsx Ground_truth sheet ---")
print("Exp1 time:              beta=3.721, SE=0.128, z=29.048")
print("Exp3 time:              beta=4.412, SE=0.191, z=23.091  [matches manuscript]")
print("Combined ELS x time:   beta=-0.822, SE=0.154, z=-5.327  [matches manuscript]")
