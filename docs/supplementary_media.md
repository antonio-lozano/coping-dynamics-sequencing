# Behavioral atlas and supplementary media

The [grid atlas](../supplementary_media/Supplementary_Video_1_MoSeq_syllable_atlas_grid.mp4) plays 25 representative motifs simultaneously in a 5 × 5 grid. The [sequential atlas](../supplementary_media/Supplementary_Video_1_MoSeq_syllable_atlas.mp4) presents them one at a time. The README's animated preview links to the grid video.

Both movies use 24 selected MoSeq syllables plus the derived climbing label, grouped by seven behavioral classes. They illustrate representative behavior, not the complete fitted-state inventory. The grid is 22.2 seconds at 1920 × 1080; the sequential movie is 63 seconds at 960 × 540. Both use silent H.264 video at 25 fps.

## Sources

Download `syllable_clips`, `ethograms` and `barcodes` from [Figshare](https://doi.org/10.6084/m9.figshare.33439885), using the [dataset downloader](datasets.md).

`Supplementary_Video_1_source_index.csv` records hashes, labels and selected occurrences for the atlas. Occurrence numbers are one-based within concatenated 20-frame clips: occurrence `n` selects `[20*(n-1), 20*n)`. They are not raw-recording frame indices. `per_animal_index.csv` maps recording IDs to bundled ethograms and barcodes.

## Regeneration

Use a disposable checkout and explicit source paths:

```bash
uv run python -m coping_dynamics.datasets syllable_clips --output ../coping-data --extract
uv run python scripts/generate_supplementary_video_1_grid.py \
  --clip-dir PATH/TO/EXTRACTED/CLIPS --output ../new-grid-atlas.mp4
```

The grid generator verifies clips against the source index. The sequential generator additionally requires `--skeleton-gif`; supply the matching skeleton animation. Keep outputs separate from original media.

## Reuse

Study media are [CC BY 4.0](../LICENSE-DATA). Credit the authors and cite the [preprint](https://doi.org/10.1101/2025.09.01.673507); indicate modifications when adapting a figure or video.
