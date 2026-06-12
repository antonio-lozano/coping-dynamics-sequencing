# Figure 4 COping Provenance

This note records the checked data path for manuscript Figure 4.

## Archives Checked

- `H:\Downloads\Coping_data2.zip\COping\Code`
- `H:\Downloads\Coping_data.zip`
- `H:\Downloads\London\JEN_STORAGE_archives`
- `H:\antonio\keypoint_moseq_project\code\shapley\Behavioral_clusters_a_mano_definitivo_no_mix_inaccurate.json`

The London/JEN February folders contain raw videos, DLC CSVs, and pickle files
for context/cue/training storage. They are upstream raw/tracking material, not
the scripts that assemble Figure 4.

## Actual Figure 4 Data Chain

1. `COping/Code/1. MoSeq data cleaning, time binning and predominant syllable extraction.py`
   starts from `moseq_df_final.csv` and produces:
   - `new_moseq_df_final.csv`
   - `Syllable_per_timebin_final(250ms).csv`
   - predominant-syllable tables for ethogram/barcode display

2. `COping/Code/6. Frequency metrics and bout duration.py` uses:
   - `Syllable_per_timebin_final(250ms).csv`
   - `Behavioral_clusters_a_mano_definitivo_no_mix_inaccurate.json`
   - saved result workbooks under `COping/Code/Results/Frequency_metrics`
   - saved result workbooks under `COping/Code/Results/Bout_duration`

3. `COping/Code/7. Transition metrics.py` uses the same 250 ms table and JSON,
   saving transition workbooks under `COping/Code/Results/Transition_metrics`.

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

The generator is:

`scripts\figure_4_diversity_dynamics\generate_figure_4_diversity_dynamics.py`

