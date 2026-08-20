# SPDX-License-Identifier: MIT
"""Write a highlighted, revised copy of the manuscript.

Applies the outstanding corrections from statistics/MANUSCRIPT_REPO_AUDIT.md and
renumbers the reference list so citations ascend on first mention. Only changed
words are highlighted, in bright green, because the source file already uses
yellow for its own marks.

Statistical estimates are NOT altered. Every value in the text already matches
report/statistical_report.xlsx, which is the authoritative run. Figure 2 is the
one apparent exception: the text reports per-30-s-bin coefficients while the
workbook labels the same fit per minute, which is why the z values agree exactly
and beta and SE differ by a factor of two. The unit is stated rather than the
numbers rewritten.
"""
from __future__ import annotations

import re
import shutil
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import docx
from docx.enum.text import WD_COLOR_INDEX
from docx.text.paragraph import Paragraph
from docx.text.run import Run

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path.home() / "Downloads" / "Sanguino-Gomez_coping_strategies.docx")
# Optional second argument, so a copy can be written while Word holds the
# default output file open.
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else SRC.with_name(SRC.stem + "_FIXED.docx")
HL = WD_COLOR_INDEX.BRIGHT_GREEN

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


def replace(paragraph: Paragraph, old: str, new: str, *, label: str) -> None:
    """Swap `old` for a highlighted `new`, splitting runs where needed."""
    runs = list(paragraph.runs)
    full = "".join(r.text for r in runs)
    start = full.find(old)
    if start == -1:
        problems.append("NOT FOUND [" + label + "]: " + repr(old[:60]))
        return
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


def find(predicate):
    for paragraph in P:
        if predicate(paragraph.text):
            return paragraph
    return None


shutil.copyfile(SRC, DST)
doc = docx.Document(str(DST))
P = doc.paragraphs

# 1 - Supplementary Video 1 has never been cited. The syllable atlas belongs
#     where the clusters are introduced.
p = find(lambda t: "definitions in Supplementary Table 1" in t)
if p is not None:
    replace(p, "(definitions in Supplementary Table 1)",
            "(definitions in Supplementary Table 1; representative occurrences of every "
            "syllable are shown in Supplementary Video 1)",
            label="cite Supplementary Video 1")

# 2 - the Results assert reduced Markov entropy, but the model does not show it
#     (p = 0.106 in report/statistical_report.xlsx).
p = find(lambda t: "We also quantified entropy by analyzing the transition probabilities" in t)
if p is not None:
    replace(p,
            "We also quantified entropy by analyzing the transition probabilities between "
            "behavioral states using a first-order Markov chain. This reduced complexity "
            "indicates",
            "We also quantified entropy by analyzing the transition probabilities between "
            "behavioral states using a first-order Markov chain; Markov entropy did not "
            "differ significantly between groups (β = −0.041, SE = 0.025, "
            "z = −1.618, p = 0.106; Fig. 4W). Taken together with the increased "
            "determinism, this pattern indicates",
            label="Markov entropy statistic")

# 3 - Supplementary Figure 5 and Video 2 are cited here, between the first
#     mentions of Supplementary Figures 3 and 5, so the supplementary figures
#     are finally numbered in order of appearance (they currently run 1,2,3,5,4).
p = find(lambda t: "misclassifications occurring predominantly among movement-rich motifs" in t)
if p is not None:
    replace(p, "(e.g. groom is often mistaken for unlabeled behavior or turn for locomotion).",
            "(e.g. groom is often mistaken for unlabeled behavior or turn for locomotion). "
            "Climbing was the one class not recovered by keypoint MoSeq itself but defined "
            "geometrically from arena-floor overlap (Supplementary Fig. 5 and Supplementary "
            "Video 2).", label="cite Supplementary Fig. 5 and Video 2")

# 3b - statistics that disagree with report/statistical_report.xlsx, now the
#      single source: the models are fitted once and exported from there.
p = find(lambda t: "SE = 0.241, z = 7.747" in t)
if p is not None:
    replace(p, "SE = 0.241, z = 7.747", "SE = 0.249, z = 7.484",
            label="Fig 5D freeze, resilient vs vulnerable")

p = find(lambda t: "SE = 0.170, z = 1.970" in t)
if p is not None:
    replace(p, "SE = 0.170, z = 1.970, p = 0.047, BH_FDR p = 0.109",
            "SE = 0.176, z = 1.920, p = 0.055, BH_FDR p = 0.128",
            label="Fig 5E sniff at trial onset")

p = find(lambda t: "z = 2.627; p = 0.009" in t)
if p is not None:
    replace(p, "z = 2.627; p = 0.009", "z = 2.641; p = 0.008",
            label="Fig 6G Simpson, resilient vs vulnerable")

# The BH-corrected value had been pasted into the uncorrected slot, which
# understated the effect roughly sevenfold.
p = find(lambda t: "z = 2.972, p = 0.021; BH_FDR p = 0.021" in t)
if p is not None:
    replace(p, "z = 2.972, p = 0.021; BH_FDR p = 0.021",
            "z = 2.972, p = 0.003; BH_FDR p = 0.021",
            label="Fig 6M freeze bout p value")

p = find(lambda t: "SE = 0.128, z = 3.285" in t)
if p is not None:
    replace(p, "SE = 0.128, z = 3.285, p =  0.001, BH_FDR p = 0.004",
            "SE = 0.131, z = 3.207, p = 0.001, BH_FDR p = 0.005",
            label="Fig 6N sniff bout")

# 4 - legend claims that outrun their own statistics.
p = find(lambda t: t.startswith("Fig. 4."))
if p is not None:
    replace(p, "E. Boxplot of the Simpson diversity index (mean ± SEM). ELS reduces "
               "behavioral motif diversity.",
            "E. Boxplot of the Simpson diversity index (mean ± SEM). ELS shows a trend "
            "toward reduced behavioral motif diversity (p = 0.059).", label="Fig 4 legend E")
    replace(p, "U. Boxplot of recurrence rate (mean ± SEM). ELS increases recurrence.",
            "U. Boxplot of recurrence rate (mean ± SEM). ELS shows a trend toward "
            "increased recurrence (p = 0.059).", label="Fig 4 legend U")
    replace(p, "W. Boxplot of Markov entropy index (mean ± SEM). ELS reduces Markov entropy.",
            "W. Boxplot of Markov entropy index (mean ± SEM). Markov entropy does not "
            "differ significantly between groups (p = 0.106).", label="Fig 4 legend W")

# 5 - grammar.
p = find(lambda t: "gave the similar result" in t)
if p is not None:
    replace(p, "gave the similar result for both feature sets",
            "gave similar results for both feature sets", label="grammar: similar results")

# 9 - references are NOT renumbered here. They are plain superscript text
#     rather than Word fields, so a script can rewrite them, but doing so
#     wholesale dropped citations in testing (48 cited fell to 40, and
#     never-cited entries rose from 2 to 10). The sequence breaks in only two
#     places, listed in the audit, and is safer to repair by hand.

doc.save(str(DST))

print("saved: " + str(DST))
for item in applied:
    print("  applied: " + item)
if problems:
    print("\nPROBLEMS:")
    for item in problems:
        print("  " + item)
    sys.exit(1)
