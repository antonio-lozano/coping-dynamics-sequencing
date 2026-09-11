# Study datasets

The dataset is deposited at [Figshare](https://doi.org/10.6084/m9.figshare.33439885). It is embargoed until publication. Private review links are not distributed in this repository. Obtain authorized archives directly from the authors during the embargo.

## Verify and extract authorized archives

```bash
uv run python -m coping_dynamics.datasets --list
uv run python -m coping_dynamics.datasets freezing_predictions keypoint_moseq --output ../coping-data --extract
```

Place authorized ZIP archives in the `--output` directory before running the command. Missing embargoed archives fail before any network request. `--list` performs no download. Omit `--extract` to retain ZIPs only. Use `all` explicitly to verify all seven archives (about 2.9 GiB). Archives are checked for size and ZIP integrity. Pinned SHA-256 values, where present in `coping_dynamics/datasets.json`, are verified; every completed file's SHA-256 is printed. Existing archives are verified and reused, never overwritten. Extraction requires a new directory and rejects unsafe archive paths.

| Archive | Contents | Approx. size |
| --- | --- | ---: |
| `freezing_predictions` | Per-frame SimBA freezing predictions for 98 recordings | 2.5 MiB |
| `keypoint_moseq` | Configuration, session index, PCA, results, per-frame table and syllable metadata | 125 MiB |
| `dlc` | Raw/filtered DeepLabCut pose tables and trained network assets | 683 MiB |
| `videos` | 98 raw behavioral recordings | 1.80 GiB |
| `syllable_clips` | Representative syllable clips | 291 MiB |
| `ethograms` | Per-recording ethogram visualizations | 3.8 MiB |
| `barcodes` | Per-recording barcode visualizations | 9.1 MiB |

## Use Figshare files in analyses

Archives retain their original top-level folder. For example, after the command above:

```text
../coping-data/
  freezing_predictions.zip
  freezing_predictions/freezing_predictions/*.csv
  keypoint_moseq.zip
  keypoint_moseq/keypoint_moseq/moseq_df.csv
  keypoint_moseq/keypoint_moseq/results.h5
```

Point the freezing-table workflow at the extracted predictions:

```bash
uv run python scripts/derive_tables/freezing_predictions_light.py \
  --input ../coping-data/freezing_predictions/freezing_predictions \
  --output ../coping-freezing-tables
```

With `--output`, the generator writes compact tables into a new directory without modifying bundled references. Omitting it retains the checkout-based regeneration behavior. The external archive and bundled tables are distinct inputs: do not silently replace one with the other.

For Behavior Studio, obtain authorized `dlc` and optionally `videos` archives, extract them, and select the relevant extracted tracking/video folder in the Analysis tab. Match recording identities and frame counts; choose the matching network configuration when processing raw video. See the [app guide](../apps/behavior-dlc-classifier/GUIDE.md).

The raw `moseq_df.csv` is an upstream table, not necessarily the processed schema expected by the root saved-model classifier. The [first-recording tutorial](first-recording.md) uses the bundled compatible table and model.

## Troubleshooting

- On interruption, rerun the command; incomplete temporary downloads are discarded.
- If an existing file fails validation, move it aside before retrying. The downloader will not replace it.
- If extraction already exists, choose a new output directory or omit `--extract`.
- If Figshare reports unavailable access, open the collection link to check availability. Do not substitute files from an unrelated dataset.

Figshare file IDs, names and sizes are recorded in the package's dataset catalog. Analysis runs use local inputs; they do not contact Figshare implicitly.
