# Installation


> Model weights and app demo recordings are separate assets. During the data embargo, obtain the authorized archive from the authors; run `python scripts/install_app_assets.py /path/to/behavior-studio-assets.zip` from the repository root before following bundled-model examples. The public dataset reference is DOI `10.6084/m9.figshare.33439885`. No private download link is included.

## Choose an environment

| Use | Installation | Required assets |
| --- | --- | --- |
| Manuscript downstream reproduction | Source checkout + root `uv.lock` | Bundled study inputs and scripts |
| Shared Python metrics | Installed `coping_dynamics` package | Your explicitly supplied data |
| Root behavior classifier | Root environment | Compatible explicit model and feature data |
| Companion DLC/freezing application | Its own environment under `apps/behavior-dlc-classifier` | Compatible tracks/video/models |
| Raw-video pose estimation | Separate compatible DeepLabCut setup | Matching tracking project and network |

These environments are not interchangeable. In particular, installing the root package does not install DeepLabCut or prove that the archived tracking networks can run on your hardware.

## Recommended: locked study checkout

```bash
git clone --branch main https://github.com/antonio-lozano/coping-dynamics-sequencing.git
cd coping-dynamics-sequencing
uv sync --locked
uv run python scripts/check_reproducibility.py
```

Install `uv` using its [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/) if needed. Record `git rev-parse HEAD` with analyses so the exact software version can be recovered.

The project declares Python 3.10–3.11; `.python-version` selects the locked interpreter. `uv sync --locked` refuses a stale lockfile rather than silently resolving new versions. A lock records dependency resolution, not identical operating-system libraries or guaranteed identical numerical bytes.

macOS is the supported desktop target for this version. Full Windows/Linux desktop support is planned for the next version.

## Alternative installations

From the checkout root:

```bash
python -m venv .venv
```

Activate `.venv` with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell, then:

```bash
python -m pip install -r requirements.txt
python scripts/check_reproducibility.py
```

`requirements.txt` is exported from the lock and includes the local editable package. Alternatively:

```bash
conda env create -f environment.yml
conda activate coping-dynamics
python scripts/check_reproducibility.py
```

The conda file is separately maintained; do not describe it as the identical uv environment without verifying its resolved dependencies.

## Package versus checkout

```bash
python -m pip install .
```

This installs the Python package and `behavior-classifier` entry point. Wheels and source distributions contain the root Python package and metadata, not the study's data/model assets, `scripts/` or companion application. Use the Git checkout for those workflows. Manuscript commands in this documentation require a checkout. For classifier work, pass explicit input, model and output paths; do not assume installation provides pretrained assets.

Package installation outside a checkout needs its own verification: import modules, run `behavior-classifier --help`, and exercise an explicit-input workflow from an unrelated directory. Editable-checkout tests alone do not establish wheel usability.

## Companion application

Follow [its guide](../apps/behavior-dlc-classifier/GUIDE.md) in its own directory/environment. Do not run its lockfile installation in the root environment. Its tests and model provenance are separate from manuscript reproduction.

Use [DeepLabCut's official installation guide](https://deeplabcut.github.io/DeepLabCut/docs/installation.html) for pose-estimation backend and hardware setup; this repository's root environment is not a substitute. For your own setup, raw-video tracking remains unverified until the real video, tracking network and compatible environment have been exercised together; the retained macOS runs are described below.

## Tracking installation: readiness versus completion

Use the platform-specific commands in [Tracking new videos](../apps/behavior-dlc-classifier/GUIDE.md#tracking-new-videos).
The Apple Silicon recipe is separate from the inherited Linux/Windows recipe;
the latter is not yet a verified fresh installation. Preserve the documented
CPU index, explicit `tensorpack`/`tf-slim` requirements and macOS PyTables override.
Do not copy dependency pins between these platforms or into the root environment.
Linux x86-64 also requires glibc 2.28 or newer for the pinned CPU PyTorch wheel
(`manylinux_2_28`; check `ldd --version`). Python 3.10 dependency resolution
was independently checked for that target, not installation or tracking.
The older `manylinux_2_17` target fails resolution; see the guide before installing.

From `apps/behavior-dlc-classifier`, after installing its tracking environment:

```bash
uv run --python .venv-dlc/bin/python --no-project python -c "import deeplabcut; print(deeplabcut.__version__)"
```

On Windows use `.venv-dlc/Scripts/python.exe` instead. Expect `2.3.11` for the
bundled TensorFlow-era network. An import failure is an installation failure;
a successful import is only readiness, not proof of tracking.

| Check | Evidence required | Does not establish |
| --- | --- | --- |
| Companion tests | Run `uv run pytest` in the companion directory/environment; retain failures and skips | Raw-video tracking or pose accuracy |
| Tracking import | Correct interpreter and DeepLabCut version, no import exception | Compatible network execution |
| Fresh real-video tracking | Use a disposable video/project copy with no pre-existing tracks; run the guide's `run-from-raw` command; retain raw/filtered HDF5 and CSV | Independent accuracy or generalization |
| Output integrity | Full video frame count, 14 named body parts × x/y/likelihood (42 columns), finite coordinates/likelihoods; source video/network/config hashes unchanged | Agreement with human labels |

Use the frame count and body-part names of your own recording when validating outputs.

## Versions and upgrades

Record `git rev-parse HEAD` with every result to identify the exact software version. See the root
[change history](../CHANGELOG.md) and the separate
[companion changelog](../apps/behavior-dlc-classifier/CHANGELOG.md).

To evaluate a newer candidate, keep your existing checkout, environment and
outputs intact. Clone the repository into a different directory, record its
commit, run `uv sync --locked`, verify bundled integrity and execute the
[first-recording walkthrough](first-recording.md) into a new output directory.
Compare results before migrating your own workflow. Do not reuse or overwrite
archived models, reference results or an old environment across candidates.

The selected API documentation is not a promise of semantic-version compatibility
between development revisions. A wheel still needs explicit compatible data and
models. Keep root and companion environments separate; follow the companion guide
for tracking upgrades instead of changing the root lockfile.

## Troubleshooting

- **Clone denied:** check your existing GitHub account access; do not copy credentials into reports.
- **Lock check fails:** preserve the error and use the declared interpreter; do not update the scientific environment merely to bypass it.
- **Missing model or input:** check the workflow's asset paths and provenance. Installing a wheel does not download these assets.
- **Manifest mismatch:** inspect the changed paths and compare with the candidate reference before regenerating a manifest.
- **Statistical warnings or numerical differences:** retain logs and investigate sample/method/environment differences; do not suppress them as a successful result.
- **DLC/GUI imports fail:** use the companion or tracking environment and its documented dependencies, not an unplanned root dependency upgrade.

## Check that you installed the intended environment

From the checkout root:

```bash
uv run python --version
uv run python -c "import coping_dynamics; print(coping_dynamics.__file__)"
uv run behavior-classifier --help
git rev-parse HEAD
```

Expect Python 3.11 for the locked checkout, an import path in this checkout, and CLI usage containing `predict`. Save the revision with your outputs. These checks establish interpreter/import/entry-point readiness, not full workflow success. Next run [one complete bundled recording](first-recording.md); no video, GPU or tracking installation is required for that example.

Next: [tutorials](tutorials.md) and [reproducibility evidence](../REPRODUCIBILITY.md).
