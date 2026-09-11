# Reproducibility

## What counts as evidence?

| Check | Establishes | Does not establish |
| --- | --- | --- |
| Committed manifest hashes | Bundled bytes agree with their recorded manifest | Regeneration or scientific correctness |
| Successful rebuild | Implemented downstream commands execute on supplied inputs | Agreement with original manuscript or independent replication |
| Same-machine double build | Selected outputs repeat in that environment | All-platform byte identity |
| Numeric/reference comparison | Explicit values agree under declared criteria | Validity of an unexplained method change |

The root rebuild starts from bundled predictions/sequence tables and other tracked inputs. It does not rerun all raw-video pose estimation, MoSeq model fitting or historical classifier analyses. Hand-assembled schematics and archival exports remain separate provenance items. The companion application in `apps/behavior-dlc-classifier/` is a separate workflow.

## Reading the checks

The automatic **Software checks** workflow runs fast tests, imports, compilation,
metadata and input integrity. The separate, manually dispatched **Scientific reference
comparison** workflow rebuilds and compares archived results; differences fail that
workflow and are never waived by the software badge. Interpret its outcomes as follows:

- **Not started:** no runner steps executed (for example, an account billing or
  runner-availability restriction). No test result can be inferred.
- **Software failure:** inspect the failing installation, test or packaging step.
- **Reference disagreement:** generation ran, but one or more outputs differ from
  the archived baseline. Inspect the comparison report before changing methods
  or reference files.
- **Incomplete report:** the rebuild did not produce a final evidence report.
  Check its logs for interruption or an execution error.

Software execution, repeat-run determinism, and agreement with archived scientific
results are separate checks. None should be substituted for another.

## Install and check the snapshot

From the checkout root:

```bash
uv sync --locked
uv run python scripts/check_reproducibility.py
uv run pytest -m "not slow" -q
```

See [installation](docs/installation.md) for alternatives. The checker verifies required files, the 98 per-animal freezing prediction CSVs, manifest hashes and repository hygiene. Record command output and candidate commit; do not report a pass solely because a check is defined in CI.

## Rebuild safely against a frozen reference

Use a new directory **outside** this checkout:

```bash
uv run python -m coping_dynamics.reproduction --output ../coping-reproduction
```

The runner copies the actual working source files (including uncommitted changes),
records their SHA256 hashes, and executes generators only in the isolated copy.
It retains the original source and manifest, step logs, generated artifacts and
`evidence.json`. Existing output directories are rejected rather than overwritten.
The candidate snapshot records exactly which source bytes ran.

CSV comparisons require identical column/row order and identifiers, with numeric
roundoff limits `rtol=1e-7`, `atol=1e-10`. Missing values must remain missing.
These limits do not authorize scientific changes. Other changed artifacts are
reported for review, not silently accepted as equivalent. A successful process
alone does not imply agreement: the runner exits nonzero for `FAILED`,
`REVIEW_REQUIRED` or `PARTIAL`; only full `PASS` exits zero. This pass concerns
the bundled downstream pipeline, not raw-video reconstruction or original-paper
scientific validity. See the [preprint](https://doi.org/10.1101/2025.09.01.673507) for study methods.

`--skip-figures` executes a partial tables/reports workflow, never a full pass.
The inherited `scripts/run_all.py` remains an **in-place artifact authoring**
command. It updates the manifest before checking it, so it must not be used as
proof of agreement with the frozen reference. Run it only in disposable working
copies when intentionally generating candidate artifacts.

## Targeted commands

Run only in the disposable checkout described above:

```bash
uv run python scripts/run_all_figures.py
uv run python scripts/build_raw_data_workbook.py
uv run python scripts/build_statistical_report.py
uv run pytest -m slow -q
```

The existing slow tests double-build the raw-data workbook and Supplementary Figure 4. They do not cover every figure or statistical table. Rendering uses fixed metadata where configured, but numerical libraries, fonts and platforms can still differ. Do not claim universal byte identity without measurements.

## Record a reproducible run

For each workflow retain:

- Candidate commit, operating system, Python version and lockfile identity.
- Exact command, input hashes, selected model and parameters.
- Logs, warnings, elapsed time and expected versus produced outputs.
- Reference comparison criteria, observed numerical differences and explanations.
- Status: **tested**, **failed**, **blocked** (name missing prerequisite), or **unverified**.

Record numerical disagreement before changing code or baselines. In particular, a different cohort, exclusion rule, model formula, random-effects structure, temporal resolution or validation split is a scientific difference—not a rounding tolerance.

See [tutorials](docs/tutorials.md) for individual workflows, [figure inventory](docs/figure_structure.md) for generated/static boundaries.

## Model refitting versus cached figure replay

The isolated full runner explicitly passes `--recompute` to Figure 7, including
predictor recap tables. The ordinary figure command permits cached replay for
cosmetic edits; it is not equivalent to full model reproduction. Output write
observations are recorded separately from byte equality, and unchanged copied
artifacts are not credited as regenerated. A write alone also does not prove
refitting: retain the explicit command and model-stage logs. Static panels and
archived classifier explanations still have the narrower provenance scope
described in the figure inventory.

### Slow tests preserve the checkout

`uv run pytest -m slow -q` now copies the checkout to a temporary directory before
its double-build tests. The fixture verifies source hashes afterward. A plain
`pytest` run no longer regenerates tracked workbook/figure artifacts in place.
The legacy `scripts/run_all.py` remains an explicit in-place authoring command;
use the isolated reproduction entry point for frozen-reference comparison.

## Manuscript authority

`scripts/check_manuscript.py` is a legacy checker with a hard-coded DOCX claim registry. It is not an automatic check of the submitted PDF. It remains available for explicit historical use, not as a green claim of submitted-manuscript agreement. No result tolerances or archived tables are changed to make CI pass.
