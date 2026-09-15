# Code availability


> Model weights and app demo recordings are separate assets. During the data embargo, obtain the authorized archive from the authors; run `python scripts/install_app_assets.py /path/to/behavior-studio-assets.zip` from the repository root before following bundled-model examples. The public dataset reference is DOI `10.6084/m9.figshare.33439885`. No private download link is included.

[Coping Dynamics Sequencing](https://github.com/antonio-lozano/coping-dynamics-sequencing) provides the analysis package, Behavior Studio desktop app, research website and tutorials under the [MIT license](LICENSE).

Start with the [installation guide](docs/installation.md) and [first-recording tutorial](docs/first-recording.md). The app uses its own environment: see the [Behavior Studio guide](apps/behavior-dlc-classifier/GUIDE.md).

Figure generators live in `scripts/generate_figures/`; shared analysis and classifier code live in `coping_dynamics/`. Use the [isolated reproduction runner](REPRODUCIBILITY.md) to generate outputs without overwriting bundled references.

Additional upstream data are distributed through [Figshare](https://doi.org/10.6084/m9.figshare.33439885); see [download instructions](docs/datasets.md).

Please cite the [bioRxiv preprint](https://doi.org/10.1101/2025.09.01.673507). [CITATION.cff](CITATION.cff) provides machine-readable citation metadata.
