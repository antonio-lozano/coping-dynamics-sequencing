# Data availability


> Model weights and app demo recordings are separate assets. During the data embargo, obtain the authorized archive from the authors; run `python scripts/install_app_assets.py /path/to/behavior-studio-assets.zip` from the repository root before following bundled-model examples. The public dataset reference is DOI `10.6084/m9.figshare.33439885`. No private download link is included.

Study data are available on [Figshare](https://doi.org/10.6084/m9.figshare.33439885) under **CC BY 4.0**. The collection includes 98 raw videos, DeepLabCut pose estimates and network assets, keypoint-MoSeq outputs, supervised freezing predictions, syllable clips, ethograms and barcodes.

Use the [dataset download guide](docs/datasets.md) to select archives and use the files in the app or analysis code. No Google Drive account or Drive download tool is required.

The repository also bundles compact inputs in `data/raw/`, generated tables in `data/processed/`, plotted values in `figure_source_data/` and statistical outputs in `statistics/`. Workbooks are `report/raw_data.xlsx` and `report/statistical_report.xlsx`. These bundled tables support the quickstart and downstream figure workflows without downloading raw videos.

Please cite the [study preprint](https://doi.org/10.1101/2025.09.01.673507) when using the data. See [LICENSE-DATA](LICENSE-DATA) for attribution and reuse terms.
