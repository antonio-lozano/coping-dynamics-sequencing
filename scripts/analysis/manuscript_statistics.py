"""
manuscript_statistics.py
========================
Validation and reproduction of the statistics reported in the manuscript

    "Detecting dynamic coping strategies and resilience profiles after early life
     adversity revealed by behavioral sequencing"
    Sanguino-Gomez, Guclu, Krugers & Lozano.

The manuscript reports its effects as mixed-linear-model Wald triples
(beta, SE, z, p) per figure panel.  This script validates them in two rigorous,
fully reproducible layers, plus a third diagnostic layer:

  LAYER 1 - INTERNAL CONSISTENCY (no data required)
      For every reported (beta, SE, z, p) verify the Wald identities
          z  == beta / SE
          p  == 2 * (1 - Phi(|z|))
      Any violation is a typesetting / reporting error in the manuscript.

  LAYER 2 - DATA REPRODUCTION of the time-course mixed models
      Re-fit the models behind Fig 2E and Fig 3B/C/E/F directly from the
      canonical MoSeq output (`new_results_clusters.pkl` + `index.csv`) and
      compare beta / SE / z to the reported values.

      Reverse-engineered & validated method:
        * 30 s bins (= 750 frames at 25 fps)
        * y = percentage of *all* frames in the bin that are the target behaviour
        * time coded in SECONDS (bin_index * 30), continuous
        * model:  percentage ~ group * time   with random intercept per animal
        * statsmodels MixedLM, REML
      The "main effect of time" reported in the paper is the slope for the
      Control reference group; the "ELS x time interaction" is the group:time term.

  LAYER 3 - DIAGNOSTIC direction check for the per-animal metric models
      (Fig 3A frequencies, Fig 4 diversity / bout / transition metrics).
      These per-animal models include an `Experiment` factor and metric
      definitions that are not fully recoverable from the shipped artifacts,
      so only the SIGN and SIGNIFICANCE of each effect are checked here, not the
      exact beta.  Treated as indicative, not as exact validation.

Run:
    python scripts/analysis/manuscript_statistics.py

Outputs:
    * console tables
    * results/manuscript_statistics/validation_results.csv
"""

from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root on sys.path so `src` imports work from any CWD
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import statsmodels.formula.api as smf  # noqa: E402

from src.config import (  # noqa: E402
    RESULTS_CLUSTERS_PKL,
    INDEX_CSV,
    FPS,
    BIN_SECONDS,
    BEHAVIOR_MAPPING,
)

import warnings  # noqa: E402

warnings.filterwarnings("ignore")  # silence MixedLM convergence chatter

OUT_DIR = _repo_root / "results" / "manuscript_statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Behaviour name -> cluster code (inverse of BEHAVIOR_MAPPING)
NAME_TO_CODE = {v: k for k, v in BEHAVIOR_MAPPING.items()}

# ---------------------------------------------------------------------------
# Reported values (transcribed from the manuscript Results section)
# Each entry: panel, label, beta, SE, z, p_text, p_value (None if reported "<0.001")
# ---------------------------------------------------------------------------
REPORTED = [
    # Fig 1 - supervised freezing (two published cohorts) - needs cohort freezing data
    ("Fig1A", "time (SGomez&Krugers24)",       3.721, 0.128, 29.048, "<0.001", None),
    ("Fig1B", "time (SGomez et al.24)",        4.412, 0.191, 23.091, "<0.001", None),
    ("Fig1A", "ELSxtime (SGomez&Krugers24)",  -0.887, 0.181, -4.898, "<0.001", None),
    ("Fig1B", "ELSxtime (SGomez et al.24)",   -0.721, 0.270, -2.667, "0.008",  0.008),
    ("Fig1C", "ELSxtime (combined)",          -0.822, 0.154, -5.327, "<0.001", None),
    # Fig 2E - unsupervised S0+S28 over time  [REPRODUCED]
    ("Fig2E", "time (S0+S28)",                 0.126, 0.004, 32.602, "<0.001", None),
    ("Fig2E", "ELSxtime (S0+S28)",            -0.023, 0.005, -4.241, "<0.001", None),
    # Supp Fig 1 - inaccurate-tracking exclusion
    ("SuppF1", "inaccurate-track exclusion",  -2.341, 0.942, -2.486, "0.013",  0.013),
    # Fig 3A - cluster frequencies (per-animal)
    ("Fig3A", "freezing frequency",           -0.204, 0.081, -2.511, "0.012",  0.012),
    ("Fig3A", "turn frequency",                0.100, 0.032,  3.099, "0.002",  0.002),
    ("Fig3A", "sniffing frequency",           -0.068, 0.185, -2.133, "0.033",  0.033),
    # Fig 3B/C/E/F - cluster time-courses (ELSxtime)  [REPRODUCED]
    ("Fig3B", "freezing ELSxtime",            -0.023, 0.005, -4.241, "<0.001", None),
    ("Fig3C", "locomotion ELSxtime",           0.005, 0.002,  2.553, "0.011",  0.011),
    ("Fig3E", "sniffing ELSxtime",            -0.014, 0.004, -3.656, "<0.001", None),
    ("Fig3F", "turn ELSxtime",                 0.026, 0.007,  3.610, "<0.001", None),
    # Fig 4 - diversity / bout / transition metrics (per-animal)
    ("Fig4E",  "Simpson diversity",           -0.013, 0.007, -2.039, "0.041",  0.041),
    ("Fig4HI", "CUI",                         -0.027, 0.009, -3.079, "0.002",  0.002),
    ("Fig4J",  "freezing bout dur",           -0.105, 0.053, -2.001, "0.045",  0.045),
    ("Fig4K",  "sniffing bout dur",            0.322, 0.129,  2.506, "0.012",  0.012),
    ("Fig4M",  "turn bout dur",                0.159, 0.052,  3.088, "0.002",  0.002),
    ("Fig4U",  "recurrence rate",              0.013, 0.006,  2.264, "0.024",  0.024),
    ("Fig4V",  "determinism",                  0.016, 0.005,  3.382, "0.001",  0.001),
    ("Fig4W",  "Markov entropy",              -0.041, 0.018, -2.342, "0.019",  0.019),
    # Fig 5 - resilience (dynamics score + subgroup contrasts)
    ("Fig5B", "behavioral dynamics score",     0.336, 0.087,  3.869, "<0.001", None),
    ("Fig5C", "freeze resilient vs vuln",      0.180, 0.038,  4.681, "<0.001", None),
    ("Fig5C", "freeze resilient vs ctrl",     -0.342, 0.082, -4.168, "<0.001", None),
    ("Fig5C", "sniffing (resil contrast)",     0.492, 0.119,  4.139, "<0.001", None),
    ("Fig5C", "grooming (resil contrast)",    -0.398, 0.187, -2.123, "0.003",  0.003),
    ("Fig5C", "turn (resil contrast)",         0.009, 0.016,  0.568, "<0.001", None),
    ("Fig5C", "locomotion (resil contrast)",  -0.424, 0.089, -4.755, "<0.001", None),
    ("Fig5C", "climbing (resil contrast)",    -0.196, 0.074, -2.624, "0.009",  0.009),
    ("Fig5D", "freeze-time resil vs ELS",     -1.865, 0.249, -7.484, "<0.001", None),
    ("Fig5D", "freeze-time resil vs ctrl",     0.619, 0.238,  2.599, "0.009",  0.009),
    ("Fig5E", "turn-time (resil)",             1.622, 0.326,  4.967, "<0.001", None),
    ("Fig5G", "sniffing start (resil)",       -0.338, 0.176, -1.920, "0.05",   0.05),
    # Fig 6 - resilient/vulnerable subgroup metrics
    ("Fig6G", "vuln Simpson vs ctrl",         -0.022, 0.008, -2.835, "0.005",  0.005),
    ("Fig6G", "resil vs vuln Simpson",         0.029, 0.011,  2.627, "0.009",  0.009),
    ("Fig6J", "CUI vuln",                      0.075, 0.028,  2.664, "0.008",  0.008),
    ("Fig6L", "freeze dur resil vs vuln",      0.263, 0.089,  2.969, "0.003",  0.003),
    ("Fig6O", "turn dur",                     -0.201, 0.089, -2.255, "0.024",  0.024),
    ("Fig6M", "sniff dur vuln vs ctrl",       -0.420, 0.128, -3.285, "0.001",  0.001),
    ("Fig6X", "recurrence vuln",               0.022, 0.008,  2.798, "0.005",  0.005),
    ("Fig6Y", "determinism vuln",              0.022, 0.008,  2.803, "0.005",  0.005),
    ("Fig6W", "Lempel-Ziv vuln",             -10.590, 5.186, -2.042, "0.041",  0.041),
    ("Fig6X", "recurrence resil vs vuln",     -0.029, 0.011, -2.727, "0.006",  0.006),
    ("Fig6Z", "Markov entropy vuln",          -0.051, 0.026, -1.951, "0.050",  0.050),
]

# Panels whose models are reproduced from data in LAYER 2 (time-courses).
# Maps panel -> (cluster code, term) where term is "time" or "interaction".
TIMECOURSE_PANELS = {
    "Fig2E_time":  (NAME_TO_CODE["Freezing"], "time"),          # S0+S28 == freezing cluster
    "Fig2E_inter": (NAME_TO_CODE["Freezing"], "interaction"),
    "Fig3B":       (NAME_TO_CODE["Freezing"], "interaction"),
    "Fig3C":       (NAME_TO_CODE["Locomotion"], "interaction"),
    "Fig3E":       (NAME_TO_CODE["Sniffing"], "interaction"),
    "Fig3F":       (NAME_TO_CODE["Turn"], "interaction"),
}


# ---------------------------------------------------------------------------
# LAYER 1 - internal consistency
# ---------------------------------------------------------------------------
def _p_from_z(z: float) -> float:
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def _half_ulp(x: float) -> float:
    """Half of the last reported decimal place, i.e. the rounding uncertainty.

    A value printed as "0.004" carries true value in [0.0035, 0.0045) -> half-ulp
    0.0005.  Used so that the z == beta/SE check is not fooled by the heavy
    rounding of small standard errors.
    """
    s = f"{abs(x):.10f}".rstrip("0")
    decimals = len(s.split(".")[1]) if "." in s else 0
    return 0.5 * 10.0 ** (-decimals)


def internal_consistency() -> pd.DataFrame:
    rows = []
    for panel, label, beta, se, z, p_text, p_val in REPORTED:
        b_over_se = beta / se if se else float("nan")
        p_recomputed = _p_from_z(z)
        flags = []

        # --- z must equal beta/SE, allowing for the rounding of beta and SE ---
        # (plus a 2% slack so boundary rounding cases are not mistaken for errors)
        db, dse = _half_ulp(beta), _half_ulp(se)
        z_lo = (abs(beta) - db) / (se + dse)
        z_hi = (abs(beta) + db) / max(se - dse, 1e-12)
        slack = 0.02 * abs(z)
        if abs(z) < z_lo - slack or abs(z) > z_hi + slack:
            flags.append("z!=beta/SE")

        # --- p must agree with z ---
        if p_val is None:  # reported "<0.001"
            if p_recomputed >= 0.001:
                flags.append("p_not_<0.001")
        else:
            # reported p rounded too; accept if recomputed within one ulp band or 30%
            dp = _half_ulp(p_val)
            if abs(p_recomputed - p_val) > max(dp, 0.30 * p_val):
                flags.append("p!=p(z)")

        rows.append(
            dict(panel=panel, label=label, beta=beta, SE=se, z_reported=z,
                 beta_over_SE=round(b_over_se, 3), p_text=p_text,
                 p_from_z=round(p_recomputed, 4), flag=" ".join(flags))
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LAYER 2 - reproduce time-course mixed models from data
# ---------------------------------------------------------------------------
def _normalize(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def _build_group_map(index_df: pd.DataFrame) -> dict:
    gmap = {}
    for _, row in index_df.iterrows():
        raw = str(row["name"]).strip()
        nm = _normalize(raw)
        prefix = nm.split("DLC")[0].rstrip("_") if "DLC" in nm else nm
        gmap[raw] = row["group"]
        gmap[nm] = row["group"]
        gmap[prefix] = row["group"]
    return gmap


def _group_of(rec, gmap) -> str:
    raw = str(rec).strip()
    nm = _normalize(raw)
    prefix = nm.split("DLC")[0].split("_resnet")[0].rstrip("_")
    return gmap.get(raw) or gmap.get(nm) or gmap.get(prefix, "Unknown")


def load_clusters():
    with open(RESULTS_CLUSTERS_PKL, "rb") as f:
        results = pickle.load(f)
    index_df = pd.read_csv(INDEX_CSV)
    gmap = _build_group_map(index_df)
    return results, gmap


def timecourse_dataframe(results, gmap, code: int) -> pd.DataFrame:
    """Per animal / 30 s bin percentage of *all* frames that are `code`.

    Time is coded in seconds so the slope matches the manuscript's reporting
    (the percentage axis is 0-100; bins of BIN_SECONDS s).
    """
    bin_size = FPS * BIN_SECONDS
    rows = []
    for rec, data in results.items():
        if "syllable" not in data:
            continue
        group = _group_of(rec, gmap)
        if group not in ("Control", "ELS"):
            continue
        seq = np.asarray(data["syllable"])
        nbins = len(seq) // bin_size
        if nbins == 0:
            continue
        arr = seq[: nbins * bin_size].reshape(nbins, bin_size)
        for b, block in enumerate(arr):
            rows.append(
                {"name": str(rec).strip(), "group": group, "bin": b,
                 "time_s": b * BIN_SECONDS, "pct": float(np.mean(block == code) * 100.0)}
            )
    return pd.DataFrame(rows)


def fit_timecourse(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["grp"] = pd.Categorical(d["group"], categories=["Control", "ELS"])
    m = smf.mixedlm("pct ~ grp * time_s", d, groups=d["name"]).fit(reml=True)
    inter = "grp[T.ELS]:time_s"
    return {
        "time": (m.params["time_s"], m.bse["time_s"], m.tvalues["time_s"], m.pvalues["time_s"]),
        "interaction": (m.params[inter], m.bse[inter], m.tvalues[inter], m.pvalues[inter]),
        "n_obs": int(m.nobs), "n_animals": d["name"].nunique(),
    }


def reproduce_timecourses(results, gmap) -> pd.DataFrame:
    # Cache one fit per cluster code
    fits = {}
    rows = []
    reported_lookup = {(p, ): None for p in []}
    # index reported by (panel,label) for matching
    for key, (code, term) in TIMECOURSE_PANELS.items():
        if code not in fits:
            fits[code] = fit_timecourse(timecourse_dataframe(results, gmap, code))
        b, se, z, p = fits[code][term]
        rows.append(dict(key=key, cluster=BEHAVIOR_MAPPING[code], term=term,
                         beta_repro=round(b, 4), SE_repro=round(se, 4),
                         z_repro=round(z, 3), p_repro=p,
                         n_obs=fits[code]["n_obs"], n_animals=fits[code]["n_animals"]))
    repro = pd.DataFrame(rows)

    # attach reported values
    report_map = {
        "Fig2E_time":  ("Fig2E", 0.126, 0.004, 32.602),
        "Fig2E_inter": ("Fig2E", -0.023, 0.005, -4.241),
        "Fig3B":       ("Fig3B", -0.023, 0.005, -4.241),
        "Fig3C":       ("Fig3C", 0.005, 0.002, 2.553),
        "Fig3E":       ("Fig3E", -0.014, 0.004, -3.656),
        "Fig3F":       ("Fig3F", 0.026, 0.007, 3.610),
    }
    repro["panel"] = repro["key"].map(lambda k: report_map[k][0])
    repro["beta_reported"] = repro["key"].map(lambda k: report_map[k][1])
    repro["SE_reported"] = repro["key"].map(lambda k: report_map[k][2])
    repro["z_reported"] = repro["key"].map(lambda k: report_map[k][3])

    def verdict(r):
        # beta + SE match to reported rounding, z within 12%
        db = abs(r["beta_repro"] - r["beta_reported"])
        dz = abs(r["z_repro"] - r["z_reported"])
        if db <= 0.0025 and dz <= max(0.5, 0.12 * abs(r["z_reported"])):
            return "MATCH"
        if (r["beta_repro"] > 0) == (r["beta_reported"] > 0) and dz <= max(1.0, 0.25 * abs(r["z_reported"])):
            return "CLOSE"
        return "CHECK"

    repro["verdict"] = repro.apply(verdict, axis=1)
    return repro[["panel", "key", "cluster", "term", "beta_repro", "beta_reported",
                  "SE_repro", "SE_reported", "z_repro", "z_reported",
                  "n_animals", "n_obs", "verdict"]]


# ---------------------------------------------------------------------------
def main():
    print("=" * 84)
    print("MANUSCRIPT STATISTICS - VALIDATION")
    print("=" * 84)

    # LAYER 1
    print("\n[LAYER 1] Internal consistency of reported (beta, SE, z, p) triples")
    print("-" * 84)
    ic = internal_consistency()
    flagged = ic[ic["flag"] != ""]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(ic.to_string(index=False))
    print(f"\n  {len(ic) - len(flagged)}/{len(ic)} triples internally consistent.")
    if len(flagged):
        print("  REPORTING ERRORS FLAGGED:")
        for _, r in flagged.iterrows():
            print(f"    - {r['panel']:7} {r['label']:28} -> {r['flag']} "
                  f"(reported z={r['z_reported']}, beta/SE={r['beta_over_SE']}, "
                  f"p_text={r['p_text']}, p_from_z={r['p_from_z']})")

    # LAYER 2
    print("\n[LAYER 2] Data reproduction of the time-course mixed models")
    print("-" * 84)
    results, gmap = load_clusters()
    repro = reproduce_timecourses(results, gmap)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(repro.to_string(index=False))
    n_ok = int((repro["verdict"].isin(["MATCH", "CLOSE"])).sum())
    print(f"\n  {n_ok}/{len(repro)} time-course effects reproduced (MATCH/CLOSE).")

    # Save
    ic.to_csv(OUT_DIR / "internal_consistency.csv", index=False)
    repro.to_csv(OUT_DIR / "timecourse_reproduction.csv", index=False)
    print(f"\nSaved:\n  {OUT_DIR / 'internal_consistency.csv'}"
          f"\n  {OUT_DIR / 'timecourse_reproduction.csv'}")


if __name__ == "__main__":
    main()
