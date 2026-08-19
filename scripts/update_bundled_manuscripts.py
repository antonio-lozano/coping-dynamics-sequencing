#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Bring the two bundled manuscript copies up to the current statistics.

Why this exists
---------------
The per-animal models behind Figures 4, 5B and 6 once asked for a random
intercept for animal on tables that hold exactly one row per animal. That
random effect is not identifiable, so the reported standard errors were read
off a singular Hessian and moved between runs. ``src.statistics`` now fits
those models as OLS with standard errors clustered by litter, which changed
every SE, z and p (never a coefficient) and moved a few claims across the
significance boundary.

The corrected numbers are already in ``statistics/*.csv``,
``report/statistical_report.xlsx`` and the working manuscript. The two copies
bundled under ``report/`` still quoted the old values, so this script rewrites
their Results statistics in place:

    report/Sanguino-Gómez_coping_strategies_Nature_Neuroscience_highlighted.docx
    report/Sanguino-Gómez_coping_strategies_publication_ready_highlighted.docx

Every replacement is highlighted bright green (yellow already marks the
submission additions). Claims that lost significance are reworded, and the
Figure 4K freezing-bout and 4W Markov-entropy quotes are dropped to match the
working manuscript, which reports both panels as not significant without
numbers. ``scripts/check_manuscript.py`` verifies the result against the
statistics exports and the workbook.

This is a one-time migration for the August 2026 model change: the hard-coded
before strings match the bundled copies as they were then and silently stop
matching after any later edit. Run ``check_manuscript.py``, not this script,
to prove the copies are current.

Run:
    python scripts/update_bundled_manuscripts.py
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import docx
from docx.enum.text import WD_COLOR_INDEX
from docx.text.paragraph import Paragraph
from docx.text.run import Run

REPO = Path(__file__).resolve().parents[1]
TARGETS = [
    REPO / "report" / "Sanguino-Gómez_coping_strategies_Nature_Neuroscience_highlighted.docx",
    REPO / "report" / "Sanguino-Gómez_coping_strategies_publication_ready_highlighted.docx",
]
HL = WD_COLOR_INDEX.BRIGHT_GREEN

MI = "−"  # true minus sign, used throughout the bundled copies
EN = "–"  # en dash, as in "Fig. 4J–Q" and "Lempel–Ziv"

problems: list[str] = []
applied: list[str] = []


def _run_after(anchor: Run, text: str, source: Run, highlight: bool) -> Run:
    element = deepcopy(source._element)
    for child in list(element):
        if child.tag.split("}")[-1] in {"t", "br", "tab", "cr", "noBreakHyphen"}:
            element.remove(child)
    anchor._element.addnext(element)
    run = Run(element, anchor._parent)
    run.text = text
    if highlight:
        run.font.highlight_color = HL
    return run


def replace(paragraph: Paragraph, old: str, new: str, *, label: str) -> bool:
    """Swap `old` for a highlighted `new`, splitting runs where needed.

    Already-applied edits count as applied, so the script can be re-run after
    a value is added to the table without disturbing the earlier replacements.
    """
    runs = list(paragraph.runs)
    full = "".join(r.text for r in runs)
    if new in full:
        applied.append(f"{label} (already applied)")
        return True
    start = full.find(old)
    if start == -1:
        return False
    end = start + len(old)
    spans, pos = [], 0
    for r in runs:
        spans.append((pos, pos + len(r.text), r))
        pos += len(r.text)
    inserted = False
    for s, e, r in spans:
        if e <= start or s >= end:
            continue
        before = r.text[: max(start, s) - s]
        after = r.text[min(end, e) - s:]
        if not inserted:
            r.text = before
            anchor = _run_after(r, new, r, True)
            if after:
                _run_after(anchor, after, r, False)
            inserted = True
        else:
            r.text = before + after
    applied.append(label)
    return True


# (old, new, label). Sources of truth for the new values:
#   statistics/stats_figure4_diversity_MixedLM.csv     4E, 4H
#   statistics/stats_figure4_bouts_MixedLM.csv         4K, 4L, 4N
#   statistics/stats_figure4_transition_MixedLM.csv    4U, 4V, 4W
#   statistics/stats_figure5_dynamics.csv              5B
#   statistics/stats_figure5_timecourse_MixedLM.csv    5D, 5E, 5G
#   statistics/fig6_*_resilience_stats.csv             6G, 6M, 6N, 6P, 6W-6Z
#   report/statistical_report.xlsx                     every BH-FDR value
EDITS: list[tuple[str, str, str]] = [
    # -- Figure 3F: unchanged model, but the BH-FDR family holds 6 tests, not
    #    7 - the Groom timecourse model does not converge under the locked
    #    environment and is excluded from the correction.
    (
        "β = 0.005, SE = 0.002, z = 2.553, p = 0.011, BH_FDR p = 0.019; Fig. 3F",
        "β = 0.005, SE = 0.002, z = 2.553, p = 0.011, BH_FDR p = 0.016; Fig. 3F",
        "Fig 3F BH-FDR family size",
    ),
    # -- Figure 4 diversity and bouts ---------------------------------------
    (
        f"β = {MI}0.013, SE = 0.007, z = {MI}1.915, p = 0.055; Fig. 4E",
        f"β = {MI}0.013, SE = 0.007, z = {MI}1.893, p = 0.058; Fig. 4E",
        "Fig 4E Simpson",
    ),
    (
        f"β = 0.064, SE = 0.028, z = 2.250, p = 0.024; Fig. 4H{EN}I",
        f"β = 0.064, SE = 0.030, z = 2.096, p = 0.036; Fig. 4H{EN}I",
        "Fig 4H CUI",
    ),
    (
        f"freezing bouts were shorter (β = {MI}0.094, SE = 0.033, z = {MI}2.824, "
        "p = 0.005, BH-FDR p = 0.017; Fig. 4K), whereas sniffing (β = 0.321, "
        "SE = 0.126, z = 2.540, p = 0.011, BH-FDR p = 0.026; Fig. 4L) and turning "
        "bouts (β = 0.164, SE = 0.058, z = 2.824, p = 0.005, BH-FDR p = 0.017; "
        "Fig. 4N) were longer. Thus, ELS altered the balance of passive and "
        "active bouts and increased dominance by the most-used motifs, while the "
        "reduction in Simpson diversity remained a statistical trend.",
        "freezing bout duration did not differ significantly (Fig. 4K), whereas "
        "sniffing (β = 0.321, SE = 0.147, z = 2.182, p = 0.029, BH-FDR p = 0.102; "
        "Fig. 4L) and turning bouts (β = 0.164, SE = 0.071, z = 2.318, p = 0.020, "
        "BH-FDR p = 0.102; Fig. 4N) were nominally longer but did not survive FDR "
        "correction. Thus, ELS increased dominance by the most-used motifs, while "
        "the bout-level prolongation of active behaviors was nominal and the "
        "reduction in Simpson diversity remained a statistical trend.",
        "Fig 4K/4L/4N bouts",
    ),
    # -- Figure 4 transitions ------------------------------------------------
    (
        "recurrence showed a trend toward an increase in ELS animals (β = 0.013, "
        "SE = 0.007, z = 1.908, p = 0.056; Fig. 4U). Determinism was "
        "significantly higher in ELS animals (β = 0.016, SE = 0.007, z = 2.167, "
        f"p = 0.030; Fig. 4V), whereas Markov entropy was unchanged (β = {MI}0.041, "
        f"SE = 0.025, z = {MI}1.638, p = 0.102; Fig. 4W). These results support "
        "greater predictability of ELS sequences through increased determinism, "
        "while evidence for higher recurrence was near threshold and neither "
        "sequence complexity nor transition entropy differed significantly.",
        "recurrence showed a trend toward an increase in ELS animals (β = 0.013, "
        "SE = 0.007, z = 1.893, p = 0.058; Fig. 4U). Determinism showed a similar "
        "trend (β = 0.016, SE = 0.009, z = 1.778, p = 0.075; Fig. 4V), whereas "
        "Markov entropy was unchanged (Fig. 4W). These results point toward more "
        "repetitive and predictable ELS sequences, although recurrence and "
        "determinism remained trends and neither sequence complexity nor "
        "transition entropy differed significantly.",
        "Fig 4U/4V/4W transitions",
    ),
    # -- Figure 5 ------------------------------------------------------------
    (
        "β = 0.336, SE = 0.087, z = 3.869, p < 0.001; Fig. 5B",
        "β = 0.336, SE = 0.104, z = 3.222, p = 0.001; Fig. 5B",
        "Fig 5B dynamics score",
    ),
    (
        "β = 1.865, SE = 0.241, z = 7.747",
        "β = 1.865, SE = 0.249, z = 7.484",
        "Fig 5D freeze res-vs-vuln",
    ),
    (
        f"β = {MI}1.622, SE = 0.315, z = {MI}5.141",
        f"β = {MI}1.622, SE = 0.326, z = {MI}4.967",
        "Fig 5G turn res-vs-vuln",
    ),
    (
        "The sniffing interaction was nominal (β = 0.338, SE = 0.170, z = 1.988, "
        "p = 0.047) but was not significant after FDR correction (BH-FDR "
        "p = 0.109; Fig. 5E).",
        "The sniffing interaction showed a trend (β = 0.338, SE = 0.176, "
        "z = 1.920, p = 0.055) and was not significant after FDR correction "
        "(BH-FDR p = 0.128; Fig. 5E).",
        "Fig 5E sniff res-vs-vuln",
    ),
    # -- Figure 6 ------------------------------------------------------------
    (
        f"β = {MI}0.022, SE = 0.008, z = {MI}2.818, p = 0.005, BH-FDR p = 0.024; Fig. 6G",
        f"β = {MI}0.022, SE = 0.007, z = {MI}3.017, p = 0.003, BH-FDR p = 0.010; Fig. 6G",
        "Fig 6G Simpson vuln-vs-ctrl",
    ),
    (
        "β = 0.029, SE = 0.011, z = 2.641, p = 0.008, BH-FDR p = 0.041",
        "β = 0.029, SE = 0.010, z = 3.091, p = 0.002, BH-FDR p = 0.008",
        "Fig 6G Simpson res-vs-vuln",
    ),
    (
        "Resilient animals had longer freezing bouts than vulnerable animals "
        "(β = 0.263, SE = 0.089, z = 2.972, p = 0.003, BH-FDR p = 0.021; Fig. 6M).",
        "Resilient animals had nominally longer freezing bouts than vulnerable "
        "animals (β = 0.263, SE = 0.100, z = 2.644, p = 0.008, BH-FDR p = 0.057; "
        "Fig. 6M), which did not survive FDR correction.",
        "Fig 6M freeze bout res-vs-vuln",
    ),
    (
        f"were nominal (β = {MI}0.201, SE = 0.090, z = {MI}2.247, p = 0.025) but "
        "did not survive FDR correction (BH-FDR p = 0.086; Fig. 6P)",
        f"were nominal (β = {MI}0.201, SE = 0.103, z = {MI}1.963, p = 0.050) but "
        "did not survive FDR correction (BH-FDR p = 0.134; Fig. 6P)",
        "Fig 6P turn bout res-vs-vuln",
    ),
    (
        "β = 0.420, SE = 0.131, z = 3.207, p = 0.001, BH-FDR p = 0.005; Fig. 6N",
        "β = 0.420, SE = 0.178, z = 2.364, p = 0.018, BH-FDR p = 0.042; Fig. 6N",
        "Fig 6N sniff bout vuln-vs-ctrl",
    ),
    (
        "Relative to controls, vulnerable ELS animals showed lower "
        f"Lempel{EN}Ziv complexity (β = {MI}8.965, SE = 4.417, z = {MI}2.030, "
        "p = 0.042, BH-FDR p = 0.042; Fig. 6W), higher recurrence (β = 0.026, "
        "SE = 0.009, z = 3.025, p = 0.003, BH-FDR p = 0.010; Fig. 6X), higher "
        "determinism (β = 0.010, SE = 0.003, z = 2.799, p = 0.005, BH-FDR "
        f"p = 0.010; Fig. 6Y), and lower Markov entropy (β = {MI}0.044, "
        f"SE = 0.021, z = {MI}2.100, p = 0.036, BH-FDR p = 0.042; Fig. 6Z).",
        "Relative to controls, vulnerable ELS animals showed higher recurrence "
        "(β = 0.026, SE = 0.008, z = 3.237, p = 0.001, BH-FDR p = 0.005; "
        "Fig. 6X) and nominally higher determinism (β = 0.010, SE = 0.004, "
        "z = 2.204, p = 0.028, BH-FDR p = 0.055; Fig. 6Y), alongside a trend "
        f"toward lower Lempel{EN}Ziv complexity (β = {MI}8.965, SE = 5.224, "
        f"z = {MI}1.716, p = 0.086, BH-FDR p = 0.115; Fig. 6W); Markov entropy "
        f"did not differ significantly (β = {MI}0.044, SE = 0.029, z = {MI}1.502, "
        "p = 0.133, BH-FDR p = 0.133; Fig. 6Z).",
        "Fig 6W/6X/6Y/6Z transitions vuln-vs-ctrl",
    ),
    (
        f"(β = {MI}0.033, SE = 0.012, z = {MI}2.755, p = 0.006, BH-FDR p = 0.024)",
        f"(β = {MI}0.033, SE = 0.010, z = {MI}3.253, p = 0.001, BH-FDR p = 0.005)",
        "Fig 6X recurrence res-vs-vuln",
    ),
]


def main() -> int:
    for target in TARGETS:
        doc = docx.Document(str(target))
        done_before = len(applied)
        for old, new, label in EDITS:
            hit = False
            for paragraph in doc.paragraphs:
                if replace(paragraph, old, new, label=f"{target.name}: {label}"):
                    hit = True
                    break
            if not hit:
                problems.append(f"NOT FOUND in {target.name}: [{label}]")
        doc.save(str(target))
        print(f"{target.name}: {len(applied) - done_before}/{len(EDITS)} edits applied")

    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
