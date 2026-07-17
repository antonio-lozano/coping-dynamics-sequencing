# Results Text Audit

Source text: pasted Results draft supplied to Codex.

This audit compares the pasted manuscript statistics against the checked-in
verification notes and current statistical CSV outputs in this portable repo.

## Highest Priority Corrections

### Figure 5C frequency contrasts

The older manuscript-era Figure 5C frequency paragraph had a known transcription
problem: several values labelled as "resilient vs vulnerable" were actually the
Experiment covariate row. The current pasted text has already fixed the two
claims it keeps for Freezing and Turn.

Use these corrected values from `fig5c_corrected_report.csv`:

| Claim | Correct value |
| --- | --- |
| Resilient ELS freeze more than vulnerable ELS | beta = 0.4088, SE = 0.1176, z = 3.4759, p = 0.0005, BH-FDR p = 0.0018 |
| Resilient ELS do not differ from controls for freeze | beta = 0.0671, SE = 0.1111, z = 0.6038, p = 0.5460 |
| Resilient ELS turn less than vulnerable ELS | beta = -0.1445, SE = 0.0326, z = -4.4375, p = 0.000009, BH-FDR p = 0.0001 |
| Resilient ELS do not differ from controls for turn | beta = -0.0037, SE = 0.0341, z = -0.1100, p = 0.9124 |

Do not reuse older "resilient vs vulnerable" claims for sniffing, grooming,
locomotion, or climbing. In the corrected analysis they are not significant.

### Figure 5D-J time-course signs

The pasted text has sign/contrast inconsistencies for the resilience time-course
model.

`validate_source_data.py` records the manuscript-era expected model terms as:

| Effect | Validator value |
| --- | --- |
| Fig 5D Freeze time x Resilient | beta = -1.865, SE = 0.249, z = -7.484 |
| Fig 5D Freeze time x Resilient vs Control | beta = 0.619, SE = 0.238, z = 2.599 |
| Fig 5G Turn time x Resilient | beta = 1.622, SE = 0.326, z = 4.967 |
| Fig 5E Sniff baseline x Resilient | beta = -0.338, SE = 0.176, z = -1.920 |

The pasted paragraph currently reports freeze as positive 1.865, turn as
negative -1.622, and sniff baseline as positive 0.338 while describing less
sniffing. Either flip the signs to match the model term above, or explicitly
state that the contrast has been reversed. As written, the signs and prose are
not consistently labelled.

### Figure 6 CUI

The pasted text says the higher CUI in ELS animals originated from the vulnerable
subgroup:

`beta = 0.075, SE = 0.028, z = 2.664, p = 0.008`

That is stale for the current self-contained Figure 6 reproduction. Current
`fig6_diversity_resilience_stats.csv` says:

| Contrast | Current CUI result |
| --- | --- |
| vulnerable vs control | beta = 0.0119, SE = 0.0125, z = 0.9506, p = 0.3418, BH-FDR p = 0.3418 |
| resilient vs control | beta = -0.0092, SE = 0.0167, z = -0.5483, p = 0.5835, BH-FDR p = 0.7283 |
| resilient vs vulnerable | beta = -0.0210, SE = 0.0172, z = -1.2215, p = 0.2219, BH-FDR p = 0.3250 |

So the Figure 6 CUI sentence should be removed or rewritten as no reliable CUI
difference across the three resilience groups.

## Model-Dependent / Old-Report Values

### Figure 4 diversity and bout metrics

The pasted Figure 4 values mostly match the archived `Report.xlsx` pathway, but
the repo documentation flags the old model as problematic for one-row-per-animal
summary metrics. `fig4_verification_note.md` recommends OLS for strict
reproducibility.

If matching the old report, the pasted values are acceptable:

| Metric | Old-report value |
| --- | --- |
| Simpson | beta = -0.013, SE = 0.0066, z = -2.039, p = 0.041 |
| CUI | beta = 0.064, SE = 0.021, z = 2.968, p = 0.003 |
| Freezing bout | beta = -0.105, SE = 0.053, z = -2.001, p = 0.045 |
| Sniffing bout | beta = 0.322, SE = 0.129, z = 2.506, p = 0.012 |
| Turn bout | beta = 0.159, SE = 0.052, z = 3.088, p = 0.002 |

If using the current OLS-reproducible path, use:

| Metric | Current OLS value |
| --- | --- |
| Simpson | beta = -0.0134, SE = 0.0076, t = -1.772, p = 0.080 |
| CUI | beta = 0.0636, SE = 0.0289, t = 2.204, p = 0.030 |
| Freezing bout | beta = -0.1052, SE = 0.0618, t = -1.702, p = 0.093 |
| Sniffing bout | beta = 0.3223, SE = 0.1311, t = 2.458, p = 0.016 |
| Turn bout | beta = 0.1593, SE = 0.0602, t = 2.645, p = 0.010 |

The biological direction is stable, but the significance language changes for
Simpson and freezing bout duration under the recommended reproducible model.

### Figure 4 transition metrics

The pasted transition conclusions are directionally consistent. Current
`stats_figure4_transition_MixedLM.csv` gives:

| Metric | Current value |
| --- | --- |
| LZ complexity | beta = -8.049, SE = 5.549, z = -1.450, p = 0.147 |
| Recurrence | beta = 0.0134, SE = 0.0058, z = 2.314, p = 0.0207 |
| Determinism | beta = 0.0161, SE = 0.0049, z = 3.317, p = 0.0009 |
| Markov entropy | beta = -0.0412, SE = 0.0203, z = -2.025, p = 0.0429 |

The pasted Markov entropy p = 0.019 is old-report-specific; current generated
output is p = 0.0429.

## Values That Look OK

The following sections align with current outputs or archived validation notes:

- Figure 1/2 SimBA/freezing validation: r = 0.95, R2 = 0.90, p < 0.001, and
  the Figure 2A-C freezing progression statistics are confirmed in
  `DIFF_REPORT.txt`.
- Figure 2 unsupervised S0/S28 freezing validation: overall direction and
  values match the checked outputs after rounding.
- Figure 3 behavior clusters: freeze, sniff, turn frequency effects and the
  time interactions for freeze, sniff, turn, and locomotion match current
  `stats_figure3A_GEE.csv` and `stats_figure3_overtime.csv` after rounding.
- Figure 5 dynamics score, LOOCV accuracy, and 12/41 resilient prevalence match
  checked outputs.
- Figure 6 Simpson diversity, bout duration, and transition results mostly
  match the current self-contained reproduction, except for the CUI sentence
  called out above.
- Figure 7 XGBoost/SHAP values are marked hand-curated / not script-generated
  in `DIFF_REPORT.txt`; they should be treated as not independently verified by
  the current scripts.

## Non-Data Text Cleanup

The pasted text contains mojibake/encoding artifacts that should be fixed before
manuscript use. In plain ASCII terms:

- Replace corrupted R-squared text with `R2` or `R^2`.
- Replace corrupted beta symbols with `beta`.
- Replace corrupted multiplication/cross symbols with `x`.
- Replace corrupted minus signs with `-`.
- Fix corrupted accented names, for example `Sanguino-Gomez`.
- Fix corrupted apostrophes and quotation marks.
