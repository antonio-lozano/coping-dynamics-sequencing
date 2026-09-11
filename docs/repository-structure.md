# Repository structure

## Product boundaries

| Directory | Responsibility | Entry point |
| --- | --- | --- |
| `coping_dynamics/` | Installable shared analysis and saved-model inference | `behavior-classifier` |
| `apps/behavior-dlc-classifier/` | Independently packaged desktop/video product | `freezing-dlc`, `freezing-dlc-gui` |
| `docs/` | User guides, tutorials, API and provenance | [Documentation home](index.md) |
| `scripts/` | Checkout-only analysis, generation and maintenance commands | [Tutorials](tutorials.md) |
| `tests/` | Analysis and repository contracts | `uv run pytest -m "not slow"` |

The analysis package stays at its established import path: moving it into `src/` would change checkout-based script behavior without improving this boundary. The app already has its own `src/freezing_dlc/`, lockfile, optional dependencies, tests and launchers. Do not merge its environment into the study environment.

## Scientific assets are stable

`data/`, `classifier/`, `figure_source_data/`, `statistics/`, `figures/`, `report/` and `supplementary_media/` keep their provenance-bound locations. No archived model or source bytes are changed by this migration. Original filenames inside bundled datasets and model projects are identifiers, not cosmetic naming opportunities.

## Canonical commands

From the repository root:

```bash
uv sync --locked
uv run behavior-classifier --help
```


In a separate terminal/environment:

```bash
cd apps/behavior-dlc-classifier
uv sync --locked --extra ml --extra video
uv run freezing-dlc --help
uv run freezing-dlc-gui
```

## Compatibility

Python module names, distribution names and console commands are unchanged. A relative symlink retains the former checkout location `tools/behavior_dlc_classifier` on systems with symlink support. New commands, CI and documentation use canonical locations and do not require these aliases. On Windows without Git symlink support, use the canonical directories; the old paths may be checked out as small link-text files.

An existing editable app environment may point through the old alias. Resync from the canonical app directory after upgrading. Root wheels continue to contain Python code only, not the website, study inputs or trained models. App wheels contain `freezing_dlc`; use a checkout for bundled models and sample videos.
