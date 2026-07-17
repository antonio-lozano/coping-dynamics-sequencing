# Figure 4 COping Provenance

This note records the checked data path for manuscript Figure 4.

## Bundled Inputs

The current repository is self-contained. Figure 4 is regenerated from:

- `dataset/source/syllable_usage_per_timebin_250ms.csv`
- the built-in behavioral cluster map in
  `scripts/generate_figures/figure_4_diversity_dynamics.py`

## Actual Figure 4 Data Chain

1. The upstream MoSeq cleaning step starts from the per-frame syllable table and produces:
   - `new_moseq_df_final.csv`
   - `Syllable_per_timebin_final(250ms).csv`
   - predominant-syllable tables for ethogram/barcode display

2. The frequency and bout-duration analysis uses:
   - `Syllable_per_timebin_final(250ms).csv`
   - the behavioral cluster map
   - per-animal frequency metrics
   - per-cluster bout duration summaries

3. The transition analysis uses the same 250 ms table and behavioral cluster map
   to compute LZ complexity, recurrence, determinism, and Markov entropy.

## Important Implementation Detail

The COping scripts do not simply discard syllables absent from the behavioral
cluster JSON. They assign unmatched syllables to a blank cluster.

This blank/unassigned cluster is included in:

- Shannon entropy
- Evenness
- Simpson index
- overall bout duration
- transition metrics: LZ complexity, recurrence, determinism, Markov entropy

For CUI, the original script computes total usage including the blank cluster,
then removes the blank cluster before calculating the cumulative usage index.
This detail is necessary to reproduce the archived workbook values.

## Verification

The corrected Figure 4 generator reproduces the archived COping group means for:

- Simpson index
- Shannon entropy
- Evenness
- CUI
- overall bout duration
- LZ complexity
- recurrence rate
- determinism
- Markov entropy

The generator is `scripts/generate_figures/figure_4_diversity_dynamics.py`.
