"""Build the final statistical report workbook (report/STATISTICAL_REPORT_FINAL.xlsx).

Starts from the curated template shape and:
  1. Rebuilds the three Figure 4 sheets from the default MixedLM fits with
     Combined + per-experiment blocks per metric.
  2. Applies house styling to every sheet: #4d4d4d header/title accent (white
     text) and a mustard-yellow highlight on significant p-values (P>|z| < .05).

Run:  python scripts/build_statistical_report.py
"""
import os, sys, shutil, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.cell.cell import MergedCell
import importlib.util

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = Path(os.getenv("COPING_DYNAMICS_REPORT_TEMPLATE", REPO / "report" / "STATISTICAL_REPORT_TEMPLATE.xlsx"))
OUT = REPO / "report" / "STATISTICAL_REPORT_FINAL.xlsx"

# ---- house style ----
ACCENT = "FF4D4D4D"        # dark grey accent for titles/headers
ACCENT_SUB = "FF6E6E6E"    # lighter grey for sub-section (experiment) labels
SIG = "FFF2C94C"           # mustard yellow for significant p-values
WHITE = Font(color="FFFFFFFF", bold=True, size=10)
WHITE_SM = Font(color="FFFFFFFF", bold=True, size=9)
fill_accent = PatternFill("solid", fgColor=ACCENT)
fill_sub = PatternFill("solid", fgColor=ACCENT_SUB)
fill_sig = PatternFill("solid", fgColor=SIG)
CENTER = Alignment(horizontal="center")

# ---- load the Fig 4 data pipeline ----
spec = importlib.util.spec_from_file_location(
    "fig4", REPO / "scripts" / "generate_figures" / "figure_4_diversity_dynamics.py")
fig4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(fig4)
pred, pred_seq, full_seq, meta = fig4.load_sequences()
metrics, _usage = fig4.compute_frequency_metrics(full_seq, meta)
bouts = fig4.bout_table(full_seq, meta)
transitions = fig4.transition_metrics(full_seq, meta)
mean_bouts = (bouts.groupby(["Animal", "group", "Experiment", "cluster"], as_index=False)
              ["bout_duration"].mean())

HDR = ["Parameter", "Coef.", "Std. Err.", "z", "P>|z|", "[0.025", "0.975]"]
PARAM_LABEL = {"Intercept": "Intercept", "Condition[T.ELS]": "Stress", "Experiment": "Experiment"}


def fit(df, metric, combined):
    d = df.dropna(subset=[metric]).copy()
    d["Condition"] = pd.Categorical(d["group"], categories=["Control", "ELS"])
    formula = f"{metric} ~ Condition + Experiment" if combined else f"{metric} ~ Condition"
    m = smf.mixedlm(formula, d, groups=d["Animal"]).fit(reml=False)
    ci = m.conf_int()
    rows = []
    for p in m.params.index:
        if p == "Group Var":
            continue
        rows.append([PARAM_LABEL.get(p, p), float(m.params[p]), float(m.bse[p]),
                     float(m.tvalues[p]), float(m.pvalues[p]), float(ci.loc[p, 0]), float(ci.loc[p, 1])])
    return rows, int(m.nobs)


def write_block(ws, top, left, title, table_source, metric):
    """Write one metric block: title, then Combined/Exp1/Exp2 sub-tables stacked."""
    c0 = left
    tc = ws.cell(top, c0, title); tc.fill = fill_accent; tc.font = WHITE
    for j in range(1, 7):
        ws.cell(top, c0 + j).fill = fill_accent
    r = top + 1
    exp_specs = [("Combined datasets", True, table_source),
                 ("Experiment 1 (Sanguino-Gomez & Krugers, 2024)", False, table_source[table_source["Experiment"] == 1]),
                 ("Experiment 2 (Sanguino-Gomez et al., 2024)", False, table_source[table_source["Experiment"] == 3])]
    for label, combined, src in exp_specs:
        sc = ws.cell(r, c0, label); sc.fill = fill_sub; sc.font = WHITE_SM
        for j in range(1, 7):
            ws.cell(r, c0 + j).fill = fill_sub
        r += 1
        for j, h in enumerate(HDR):
            hc = ws.cell(r, c0 + j, h); hc.fill = fill_accent; hc.font = WHITE_SM; hc.alignment = CENTER
        r += 1
        try:
            rows, n = fit(src, metric, combined)
        except Exception as e:
            ws.cell(r, c0, f"model failed: {e}"); r += 2; continue
        for row in rows:
            is_effect = row[0] == "Stress"   # only highlight the effect of interest
            for j, val in enumerate(row):
                cell = ws.cell(r, c0 + j, round(val, 6) if isinstance(val, float) else val)
                if j == 4 and is_effect and isinstance(val, float) and val < 0.05:
                    cell.fill = fill_sig
                    cell.font = Font(bold=True)
            r += 1
        ws.cell(r, c0, "No. Observations"); ws.cell(r, c0 + 1, n)
        r += 2  # blank spacer between experiment blocks


FIG4_SHEETS = {
    "Fig.4E-H_frequency_metrics": [("Simpson Index", metrics, "simpson"),
                                   ("Shannon entropy", metrics, "shannon"),
                                   ("Evenness", metrics, "evenness"),
                                   ("Cumulative usage index", metrics, "cui")],
    "Fig.4T-W_transition_metrics": [("Lempel-Ziv complexity", transitions, "lz"),
                                    ("Recurrence rate", transitions, "recurrence"),
                                    ("Determinism", transitions, "determinism"),
                                    ("Markov entropy", transitions, "markov")],
    "Fig.4J-Q_bout_duration": [(c, mean_bouts[mean_bouts["cluster"] == c].rename(columns={"bout_duration": c}), c)
                               for c in ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]],
}


def rebuild_fig4(wb):
    for sheet, blocks in FIG4_SHEETS.items():
        ws = wb[sheet]
        # unmerge everything (merged title cells are read-only), then clear
        for rng in list(ws.merged_cells.ranges):
            ws.unmerge_cells(str(rng))
        for row in ws.iter_rows():
            for cell in row:
                cell.value = None
                cell.fill = PatternFill()
        left = 1
        for title, src, metric in blocks:
            write_block(ws, 1, left, title, src, metric)
            left += 9  # 7 data cols + 2 gap


def restyle_all(wb, skip):
    """Recolor grey header/title fills to #4D4D4D and recolor the template's
    curated significance highlights (pink) to mustard. Fig 4 sheets are skipped
    (already styled in rebuild_fig4)."""
    GREYS = {"FFEDEDED", "FFF5F5F5", "EDEDED", "F5F5F5"}
    PINKS = {"FFF2CEEF", "F2CEEF", "FFF2CEE0"}
    for ws in wb.worksheets:
        if ws.title in skip:
            continue
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell, MergedCell):
                    continue
                f = cell.fill
                if not (f and f.patternType):
                    continue
                rgb = f.fgColor.rgb
                if rgb in GREYS:
                    cell.fill = fill_accent
                    cell.font = Font(color="FFFFFFFF", bold=True, size=cell.font.size or 10)
                elif rgb in PINKS:                       # curated significance -> mustard
                    cell.fill = fill_sig
                    cell.font = Font(bold=True)


def main():
    if not TEMPLATE.exists():
        raise FileNotFoundError(
            f"Report template not found: {TEMPLATE}. "
            "Set COPING_DYNAMICS_REPORT_TEMPLATE to an existing workbook."
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE, OUT)
    wb = openpyxl.load_workbook(OUT)
    print("Rebuilding Figure 4 sheets (combined + per-experiment)...")
    rebuild_fig4(wb)
    print("Restyling all sheets (#4D4D4D accent, mustard significance)...")
    restyle_all(wb, skip=set(FIG4_SHEETS))
    wb.save(OUT)
    print(f"Saved: {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
