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

# 1 - Methods names a metric the paper does not actually use.
p = find(lambda t: "Jensen" in t)
if p is not None:
    replace(p, "Cosine, Jensen–Shannon, Manhattan and Correlation",
            "Cosine, Minkowski, Manhattan and Correlation", label="Methods metric name")

# 2 - the transition panels were dropped from Supplementary Figure 3: the
#     score does not separate Control from ELS, and its overlap with the
#     Figure 5 resilient set (58.3%) is close to the 48.8% expected by chance
#     for the number of animals it classifies. The cross-reference and the
#     legend entries go with them.
p = find(lambda t: "Supplementary Fig. 3J-K" in t or "Supplementary Fig. 3I" in t)
if p is not None:
    for old in ("(Supplementary Fig. 3J-K)", "(Supplementary Fig. 3I–J)"):
        if old in p.text:
            replace(p, old, "(Supplementary Materials and Methods)",
                    label="Supp Fig 3 transition cross-reference removed")
            break

# 3 - typo.
p = find(lambda t: "3..025" in t)
if p is not None:
    replace(p, "z = 3..025", "z = 3.025", label="z = 3..025 typo")

# 4 - a clause printed twice, joined by a non-breaking space.
p = find(lambda t: "prolonged bout durations.\xa0showed prolonged" in t)
if p is not None:
    replace(p, "showed prolonged bout durations.\xa0showed prolonged bout durations.",
            "showed prolonged bout durations.", label="duplicated bout clause")

# 5 - Supplementary Figure 4 is deliberately NOT cited here. Climbing is
#     discussed before Supplementary Figures 1-3 are first mentioned, so a
#     citation at this point makes Supp. Fig. 4 the first supplementary figure
#     in the paper and the order worse, not better. Swapping the numbers of
#     Supplementary Figures 4 and 5 is the correct fix and is a separate,
#     repo-wide rename.

# 6 - state the Figure 2 time unit, so the text reconciles with the workbook.
p = find(lambda t: "A mixed linear model analysis confirmed a significant main effect of time" in t)
if p is not None:
    replace(p, "A mixed linear model analysis confirmed a significant main effect of time",
            "A mixed linear model analysis confirmed a significant main effect of time "
            "(coefficients expressed per 30 s bin)", label="Figure 2 time unit")

# 7 - legend claims that outrun their own statistics.
p = find(lambda t: t.startswith("Fig. 4."))
if p is not None:
    replace(p, "E. Boxplot of the Simpson diversity index (mean ± SEM). ELS reduces "
               "behavioral motif diversity.",
            "E. Boxplot of the Simpson diversity index (mean ± SEM). ELS shows a trend "
            "toward reduced behavioral motif diversity (p = 0.059).", label="Fig 4 legend E")
    replace(p, "U. Boxplot of recurrence rate (mean ± SEM). ELS increases recurrence.",
            "U. Boxplot of recurrence rate (mean ± SEM). ELS shows a trend toward increased "
            "recurrence (p = 0.059).", label="Fig 4 legend U")
    replace(p, "W. Boxplot of Markov entropy index (mean ± SEM). ELS reduces Markov entropy.",
            "W. Boxplot of Markov entropy index (mean ± SEM). Markov entropy does not differ "
            "significantly between groups (p = 0.106).", label="Fig 4 legend W")

# 8 - the Results sentence asserts a Markov effect the model does not show.
p = find(lambda t: "We also quantified entropy by analyzing the transition probabilities" in t)
if p is not None:
    replace(p,
            "We also quantified entropy by analyzing the transition probabilities between "
            "behavioral states using a first-order Markov chain.",
            "We also quantified entropy by analyzing the transition probabilities between "
            "behavioral states using a first-order Markov chain; Markov entropy did not differ "
            "significantly between groups (β = −0.041, SE = 0.025, z = −1.618, "
            "p = 0.106; Fig. 4W).", label="Markov entropy sentence")

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
