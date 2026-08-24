#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Align the bundled manuscripts' figure legends with the statistics.

``update_bundled_manuscripts.py`` corrected every quoted statistic in the
Results text, but the figure legends state several of the same effects as
plain findings even though the statistics behind them are trends or not
significant at all. ``check_manuscript.py`` cannot catch this: legends carry
no beta/SE/z triple to parse, so the only guard is reading them against
``statistics/*.csv`` - which is what this script encodes.

Corrected against the current exports:

    Fig 4E   trend p updated 0.055 -> 0.058     stats_figure4_diversity_MixedLM.csv
    Fig 4K   "freeze in shorter bouts" -> NS    stats_figure4_bouts_MixedLM.csv (p = 0.147)
    Fig 4U   trend p updated 0.056 -> 0.058     stats_figure4_transition_MixedLM.csv
    Fig 4V   "enhances determinism" -> trend    stats_figure4_transition_MixedLM.csv (p = 0.075)
    Fig 5H   resilient-vs-vulnerable claim cut  stats_figure5_timecourse_MixedLM.csv (p = 0.299)
    Fig 6W   "reduced complexity" -> trend      fig6_transition_resilience_stats.csv (p = 0.086)
    Fig 6Z   "lower Markov entropy" -> NS       fig6_transition_resilience_stats.csv (p = 0.133)
    Results  vulnerable-subgroup summary        same file: LZ p = 0.086 and Markov
             sentence claimed five significant  p = 0.133 are not significant, and
             effects where only two exist       determinism survives only uncorrected

Like its predecessor this is a one-time migration: the before strings match
the bundled copies as they are today and silently stop matching after any
later edit. Every replacement is highlighted bright green.

Run:
    python scripts/update_bundled_legends.py
"""

from __future__ import annotations

import docx
from update_bundled_manuscripts import EN, TARGETS, applied, problems, replace

EDITS: list[tuple[str, str, str]] = [
    # -- Figure 4 legend -----------------------------------------------------
    (
        "ELS showed a trend toward lower behavioral motif diversity (p = 0.055).",
        "ELS showed a trend toward lower behavioral motif diversity (p = 0.058).",
        "Fig 4E legend trend p",
    ),
    (
        "ELS mice freeze in shorter bouts.",
        "Freeze bout duration did not differ significantly between groups.",
        "Fig 4K legend (p = 0.147)",
    ),
    (
        "ELS showed a trend toward higher recurrence (p = 0.056).",
        "ELS showed a trend toward higher recurrence (p = 0.058).",
        "Fig 4U legend trend p",
    ),
    (
        "ELS enhances determinism.",
        "ELS showed a trend toward higher determinism (p = 0.075).",
        "Fig 4V legend",
    ),
    # -- Figure 5 legend -----------------------------------------------------
    # The vulnerable-vs-control locomotion-by-time interaction is significant
    # (p = 0.006) but resilient animals differ from neither group (resilient
    # vs vulnerable p = 0.299, resilient vs control p = 0.342), so the claim
    # that the effect "was absent in resilient animals" is unsupported.
    (
        "ELS mice showed locomotion suppression during shocks that was absent "
        "in resilient animals.",
        "Locomotion declined more steeply over the session in controls than in "
        "vulnerable ELS mice; resilient animals did not differ significantly "
        "from either group.",
        "Fig 5H legend",
    ),
    # -- Figure 6 legend -----------------------------------------------------
    (
        "When analyzed as a subgroup, vulnerable ELS mice show reduced "
        "behavioral complexity compared to controls.",
        "When analyzed as a subgroup, vulnerable ELS mice show a trend toward "
        "reduced behavioral complexity compared to controls (p = 0.086).",
        "Fig 6W legend",
    ),
    (
        "Vulnerable ELS animals show lower Markov entropy than controls; "
        "resilient mice do not differ significantly from either group.",
        "Markov entropy did not differ significantly across groups.",
        "Fig 6Z legend (p = 0.133)",
    ),
    # -- Results: vulnerable-subgroup transition summary ----------------------
    (
        "When the vulnerable subgroup was analyzed separately, however, it "
        f"showed significantly lower Simpson diversity and Lempel{EN}Ziv "
        "complexity, higher recurrence and determinism, and lower Markov "
        "entropy than controls.",
        "When the vulnerable subgroup was analyzed separately, however, it "
        "showed significantly lower Simpson diversity and higher recurrence "
        "than controls, together with nominally higher determinism and a "
        f"trend toward lower Lempel{EN}Ziv complexity; Markov entropy did not "
        "differ significantly.",
        "Results vulnerable-subgroup summary",
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
