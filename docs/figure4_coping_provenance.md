# Figure 4 Provenance

Figure 4 is regenerated from the bundled 250 ms MoSeq syllable-usage table:

- `data/raw/syllable_usage_per_timebin_250ms.csv`
- the behavioral cluster map in
  `scripts/generate_figures/figure_4_diversity_dynamics.py`

The same input table is used for diversity metrics, bout-duration metrics,
representative ethograms, behavioral barcodes, chord diagrams, and transition
metrics.

## Analysis Chain

1. Syllables are mapped to seven hand-curated behavioral clusters:
   Freezing, Sniffing, Grooming, Turn, Locomotion, Climbing, and Jump.
2. Frequency metrics are computed per animal from the full 250 ms table.
3. Bout durations are computed from consecutive cluster assignments.
4. Transition metrics are computed from each animal's cluster sequence:
   Lempel-Ziv complexity, recurrence rate, determinism, and Markov entropy.
5. Combined mixed-effects models are fit with condition and experiment terms,
   grouping observations by animal.

## Unassigned Syllables

Syllables not included in the behavioral cluster map are assigned to a blank
cluster. This blank cluster is included in Shannon entropy, evenness, Simpson
index, overall bout duration, Lempel-Ziv complexity, recurrence, determinism,
and Markov entropy.

For cumulative usage index, total usage is computed with the blank cluster
included, and the blank cluster is removed only before calculating the final
index. This rule is required for reproducible Figure 4 values.

## Outputs

- Figure: `figures/figure4.pdf`, `figures/figure4.svg`, `figures/figure4.png`
- Figure source data: `figure_source_data/figure4.csv`
- Statistical tables: `statistics/stats_figure4_*_MixedLM.csv`
- Per-animal model inputs: `statistics/fig4_*_per_animal.csv`
