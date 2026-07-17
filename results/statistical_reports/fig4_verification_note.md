# Figure 4 Statistics Verification Note

Paper: "Detecting dynamic coping strategies and resilience profiles after early life adversity revealed by behavioral sequencing"

Gold-standard report checked: `E:\#Deeplabcut_project\Report\Report.xlsx`

Repository checked: `D:\coping-dynamics-sequencing`

## Scope

Figure 4 is a two-group comparison:

- Control
- ELS

For Figure 4, ELS includes all ELS animals, including the animals later labelled ELS_resilient in subgroup analyses.

Input size:

- 82 animals total
- 41 Control
- 41 ELS
- Experiment 1: 50 animals
- Experiment 3: 32 animals

The old resilience labels are not used as separate groups for Figure 4.

## Data and Old-Code Path

The old code checked was:

`E:\#Deeplabcut_project\Code\Frequency_indexes_bout_duration_resilience.py`

The old Figure 4 code computes frequency metrics from the raw 250 ms syllable table:

`D:\coping-dynamics-sequencing\data\source\syllable_usage_per_timebin_250ms.csv`

The old-code frequency path is:

```python
freq_df = df.groupby(["Animal", "Experiment", "Condition", "Cluster"]).size().reset_index(name="Count")
pivot_df = freq_df.pivot_table(
    index=["Animal", "Condition", "Experiment"],
    columns="Cluster",
    values="Count",
    fill_value=0,
)
total_counts = pivot_df.sum(axis=1)
relative_freq = pivot_df.div(total_counts, axis=0)
```

Unmapped syllables are assigned to the blank cluster `""`. The blank cluster is included in Simpson, Shannon, and evenness denominators. For CUI, the old code drops blank clusters before calculating cumulative usage.

The old-code statistical model is:

```python
smf.mixedlm("Metric ~ group + Experiment", data=df, groups=df["Animal"]).fit(reml=False)
```

In the old script the variable is called `New_condition`, but for this Figure 4 check it was collapsed to the required two groups: Control vs ELS.

## Why Report.xlsx and R Differ

The current R model used for verification is equivalent to:

```r
lm(metric ~ group + experiment)
```

This is also the current Figure 4 generator's recommended approach, because Figure 4 metrics have one summary row per animal. A MixedLM with `groups=Animal` is not identifiable in the usual way when each animal contributes only one row.

The current Figure 4 generator explicitly notes this:

`scripts/generate_figures/figure_4_diversity_dynamics.py`

```python
# MixedLM is unidentifiable here (one observation per animal).
# Original Statistical_report used OLS(group + experiment).
```

Therefore:

- beta estimates are stable across Report.xlsx, old-code MixedLM, and current R/OLS.
- SE, z/t, and p differ because the old report uses an archived MixedLM-style output for a one-row-per-animal model.

## Full Comparison

Control vs ELS only, n=82.

| Figure | Metric | Report.xlsx | Old-code MixedLM rerun | Current R/OLS-equivalent |
| --- | --- | --- | --- | --- |
| 4E | Simpson | b=-0.013441, SE=0.006593, z=-2.038679, p=0.041482, n=82 | b=-0.013441, SE=0.007111, z=-1.890036, p=0.058753, n=82 | b=-0.013441, SE=0.007586, t=-1.771758, p=0.080291, n=82 |
| 4F | Shannon | b=-0.011574, SE=0.018343, z=-0.630967, p=0.528062, n=82 | b=-0.011574, SE=0.018596, z=-0.622384, p=0.533690, n=82 | b=-0.011574, SE=0.019311, t=-0.599353, p=0.550653, n=82 |
| 4G | Evenness | b=-0.006497, SE=0.009731, z=-0.667636, p=0.504366, n=82 | b=-0.006497, SE=0.009438, z=-0.688367, p=0.491222, n=82 | b=-0.006497, SE=0.010423, t=-0.623326, p=0.534866, n=82 |
| 4H-I | CUI | b=+0.063646, SE=0.021446, z=+2.967792, p=0.002999, n=82 | b=+0.063646, SE=0.027481, z=+2.315999, p=0.020558, n=82 | b=+0.063646, SE=0.028879, t=+2.203862, p=0.030445, n=82 |
| 4K | Freezing bout | b=-0.105174, SE=0.052566, z=-2.000792, p=0.045415, n=82 | b=-0.105174, SE=0.058312, z=-1.803648, p=0.071287, n=82 | b=-0.105174, SE=0.061778, t=-1.702451, p=0.092603, n=82 |
| 4L | Sniffing bout | b=+0.322323, SE=0.128635, z=+2.505727, p=0.012220, n=82 | b=+0.322323, SE=0.124423, z=+2.590552, p=0.009582, n=82 | b=+0.322323, SE=0.131144, t=+2.457787, p=0.016169, n=82 |
| 4N | Turn bout | b=+0.159256, SE=0.051575, z=+3.087859, p=0.002016, n=82 | b=+0.159256, SE=0.058356, z=+2.729044, p=0.006352, n=82 | b=+0.159256, SE=0.060199, t=+2.645491, p=0.009839, n=82 |

## n=81 Check

Hypothesis tested: Report.xlsx might have been run with one animal missing.

Result: no. Report.xlsx metadata says n=82 for all Figure 4 blocks. Leave-one-animal-out tests do not identify a single missing animal that explains all report values.

Best single-animal exclusions differed by metric:

| Metric | Best n=81 dropped animal |
| --- | --- |
| Simpson | 11.6 |
| CUI | 157.4 |
| Freezing bout | 136.2 |
| Sniffing bout | 88.2 |
| Turn bout | 69.5 |

Because different metrics require different dropped animals, the report was not simply generated with n=81.

## Directionality Check

| Text claim | Beta sign | Direction check |
| --- | --- | --- |
| ELS lower Simpson diversity | negative | OK |
| Shannon unchanged | p > 0.49 in old-code/R reruns | OK |
| Evenness unchanged | p > 0.49 in old-code/R reruns | OK |
| ELS higher CUI | positive | OK; pasted manuscript sign must be positive |
| ELS shorter freezing bouts | negative | OK |
| ELS longer sniffing bouts | positive | OK |
| ELS longer turning bouts | positive | OK |

## Manuscript Implication

The pasted manuscript CUI statistic is incorrect if it reports:

```text
b = -0.027, SE = 0.009, z = -3.079, p = 0.002
```

The correct direction is positive. If using Report.xlsx as the reference, use:

```text
b = 0.064, SE = 0.021, z = 2.968, p = 0.003
```

If using the current reproducible R/OLS model, use:

```text
b = 0.064, SE = 0.029, t = 2.204, p = 0.030
```

Recommended for strict reproducibility: use the current R/OLS model values, because the old MixedLM has one observation per animal and is not a stable model for SE/z/p inference.

Recommended for matching the old manuscript report exactly: use Report.xlsx values, but note that the available old code rerun does not reproduce all Report.xlsx SE/z/p values exactly.

## Scripts Used

- `scripts/verify_results_fig4_section_R.R`
- `scripts/debug_fig4_control_vs_els_oldcode.py`
- `scripts/debug_fig4_n81_check.py`
- `scripts/_debug_exact_original.py`
- `scripts/_debug_exact2.py`

