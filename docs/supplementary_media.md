# Supplementary audiovisual material

## Selected submission file

`supplementary_media/Supplementary_Video_1_MoSeq_syllable_atlas.mp4` is the
recommended audiovisual supplement. It combines three 20-frame representative
pose-overlaid occurrences for each available common MoSeq syllable (0–34) with
the corresponding canonical 14-point skeleton trajectory. The geometry-derived
climbing class (111) is shown without a MoSeq skeleton. Syllables are ordered by
the seven curated behavioral classes used in the manuscript; motifs excluded
from those classes remain explicitly labeled as unassigned.

The movie is 16:9 H.264/yuv420p, 960 × 540 pixels, 25 frames s⁻¹, silent,
and below Nature's 30 MB per-video limit. Its editable title and legend are in
`report/SIGuide.docx`. Source paths, sizes and SHA-256 hashes are recorded in
`supplementary_media/Supplementary_Video_1_source_index.csv`.

## Grid view

`supplementary_media/Supplementary_Video_1_MoSeq_syllable_atlas_grid.mp4` shows
the same 25 cluster-mapped syllables playing simultaneously in a 5 × 5 grid
(1920 × 1080 pixels, H.264/yuv420p, 25 frames s⁻¹, silent, below 30 MB), so the
whole repertoire can be compared at a glance: tiles are ordered and
color-coded by behavioral cluster, and each tile loops the same three
representative occurrences the sequential atlas uses. It is built by
`scripts/generate_supplementary_video_1_grid.py` from the identical archived
clips, each verified against the SHA-256 recorded in
`Supplementary_Video_1_source_index.csv`, which therefore remains the single
provenance record for both videos.

## Scope decision

The fitted label space contained 96 syllables, 87 of which were observed in the
bundled per-frame table; after the manuscript's 99.5% coverage filter, 38 were
retained for downstream analyses. These are model-derived syllables, not 87
independently validated ethological behaviors. The archived audiovisual source
currently available on the mounted drives contains clips for syllables 0–34 and
the derived climbing class. The submission therefore presents the concise,
interpretable common-syllable atlas and does not imply that all 87 observed
labels are distinct behavior categories.

The original raw videos referenced by the historical scripts were on an
unmounted storage volume. If that volume becomes available, additional retained
syllables can be rendered for repository quality control, but a large rare-label
dump is not recommended as journal Supplementary Information.

## Regeneration

Run the generator with explicit source locations:

```bash
python scripts/generate_supplementary_video_1.py \
  --clip-dir PATH/TO/video_clips \
  --skeleton-gif PATH/TO/skeleton_trajectories.gif
```

The generator verifies source presence, writes the indexed provenance table and
encodes the movie without audio. The source clip files contain concatenated
20-frame representative occurrences with the 14-point pose overlay and a white
dot marking frames assigned to the displayed syllable.
