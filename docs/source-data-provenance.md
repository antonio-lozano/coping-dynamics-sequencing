# Source data and recording identity

The study's upstream recordings and analysis inputs are distributed through [Figshare](https://doi.org/10.6084/m9.figshare.33439885). See the [dataset guide](datasets.md) for selective downloads and file layout.

The dataset provides raw videos, raw and filtered DeepLabCut tracks, network assets, keypoint-MoSeq configuration/results, freezing predictions and representative clips. Bundled compact tables in `data/raw/` support downstream analysis without downloading the complete video collection.

Recording IDs, frame indices and acquisition rates determine how these sources join. Preserve original names and frame order; never pair recordings by directory listing position. The keypoint-MoSeq archive includes a session index and syllable metadata. Freezing predictions contain one CSV per recording.

The atlas clip index, `supplementary_media/Supplementary_Video_1_source_index.csv`, records file hashes and selected clip occurrences. Its occurrence numbers refer to concatenated clips, not raw-video frame numbers.

Different cohorts and derived tables can contain different recording counts. Use the explicit cohort mappings in the figure generators and the [data dictionary](data_dictionary.md), rather than assuming every archive has the same analysis membership.

Code is MIT licensed; data and media are CC BY 4.0. Retain original author credits. Cite the [preprint](https://doi.org/10.1101/2025.09.01.673507).
