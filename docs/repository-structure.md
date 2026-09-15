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

Python module names, distribution names and console commands are unchanged. The former checkout location `tools/behavior_dlc_classifier` contains a navigation README, not a symlink. Use `apps/behavior-dlc-classifier` for installation and execution on every platform. No Developer Mode or link-creation privilege is required to clone the repository.

Existing editable installations may still point through the former alias. Resync from the canonical app directory after upgrading. Root wheels contain Python code, not the website, study inputs or trained models. App models and examples are installed separately using the checksummed asset installer.
