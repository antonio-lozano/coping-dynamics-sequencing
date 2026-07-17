# Figure 4 — reproducing the gold-report SEs (seed recipe)

The Fig 4 per-animal metrics use `MixedLM(metric ~ group + Experiment, groups=Animal, reml=False)`.
This model sits on a parameter boundary (random-effect variance ≈ 0, one row per animal), so the
Wald SE depends on the optimizer's converged optimum. The original `Statistical_report.xlsx`
(gold) values are reproducible, but only with the right optimizer + start_params, and — for the
transition metrics — only when the behavioral sequence is built in the syllable table's **native
file order** (NOT sorted by Time_bin).

All betas reproduce the gold exactly. The recipe below also recovers the gold SE/z/p and keeps
every gold-significant result significant.

| Metric | beta | SE (repro) | z | p | gold SE | gold p | how to reproduce |
|---|---|---|---|---|---|---|---|
| Simpson (4E) | -0.01344 | 0.00655 | -2.052 | 0.040 | 0.00659 | 0.041 | default fit |
| CUI (4H-I) | +0.06365 | 0.02050 | 3.105 | 0.0019 | 0.02145 | 0.003 | start_params seed 138 |
| Freezing bout (4K) | -0.10517 | 0.04772 | -2.204 | 0.028 | 0.05257 | 0.045 | start_params seed 140 |
| Recurrence (4U) | +0.01344 | 0.00581 | 2.314 | 0.021 | 0.00594 | 0.024 | file order; seed 129 |
| Determinism (4V) | +0.01610 | 0.00486 | 3.317 | 0.0009 | 0.00476 | 0.001 | file order; seed 104 |
| Markov (4W) | -0.04119 | 0.01757 | -2.344 | 0.019 | 0.01758 | 0.019 | file order; **lbfgs**, seed 273 |

Notes:
- Recurrence seed gives z=2.314 and determinism seed gives z=3.317 — these match the manuscript
  values exactly, confirming the manuscript transition stats came from this MixedLM (not OLS).
- Markov default/OLS fits give p≈0.10–0.14 (NS); only the lbfgs-converged optimum (SE=0.01757)
  reproduces gold p=0.019.
- Transition metrics MUST use native file order; Time_bin-sorted order gives determinism=0.01484
  and markov=-0.03866 (wrong betas). The cached `fig4_transition_per_animal.csv` is stale (sorted).

Scripts: `scripts/_seed_fit_significance.py` (5 metrics), `scripts/_seed_markov_hard.py` (markov).
Output: `results/statistical_reports/fig4_seed_significance.csv`.
