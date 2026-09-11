# Contributing

Thanks for contributing! Please open an issue for substantial changes, preserve author credits, and keep analysis inputs unchanged.

See the [repository map](docs/repository-structure.md) for package boundaries and stable asset paths. Start new user documentation from [the documentation home](docs/index.md).

## Development setup

```bash
uv sync --locked
uv run pytest -m "not slow" -q
uv run ruff check coping_dynamics scripts tests behavior_classifier.py
uv run ruff format --check coping_dynamics scripts tests behavior_classifier.py
```

Use a feature worktree and review changes before committing. The companion tool has its own environment and tests; root checks do not replace them. Slow determinism tests use isolated temporary copies. Direct generators modify tracked artifacts, so use a disposable checkout and preserve reference outputs first.

## Scientific changes

- Preserve source data, model hashes, cohort membership and provenance.
- Record a scientific discrepancy before changing a method or reference result. Explain the original and implemented methods, affected panels/results, and evidence.
- Never adjust exclusions, formulae, metrics, validation splits, thresholds or test expectations merely to recover a reported result or passing check.
- Keep historical reproduction separate from exploratory corrected analyses; label both clearly.
- Never silently rewrite manuscripts or replace archived models during tests or examples.
- Treat statistical warnings, failed fits and unavailable source stages as evidence requiring investigation, not success.

Add meaningful tests for boundary cases, cohort/sample integrity and scientific formula contracts. Use small fixtures for focused tests and real study inputs for documented workflow verification. Synthetic tests are supporting evidence, not a substitute for full-data execution.

## Generated artifacts and dependencies

Do not refresh `MANIFEST.csv` to hide unexplained differences. Capture the old manifest, commands, input hashes, output differences and numerical comparison before accepting a regenerated artifact set. Avoid source-data deletion, history rewrites or ad hoc dependency upgrades. Document any archive format migration and verify content equivalence.

`requirements.txt` is exported from the lockfile; follow its header when regenerating it. The conda environment is separate. Keep publication dependencies stable unless a specific supported change requires otherwise.

## Review evidence

A change summary should state: what changed; scientific impact or why none; tested commands and candidate/environment; real data used; outputs compared; known failures; blocked and unverified workflows. See [the evidence contract](REPRODUCIBILITY.md). A green integrity check alone is not a complete reproducibility claim.
