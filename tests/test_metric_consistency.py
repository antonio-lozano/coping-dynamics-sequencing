# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""Cross-implementation agreement tests for the sequence metrics.

The transition-metric quartet (Lempel-Ziv, recurrence rate, determinism,
Markov entropy) historically existed as one copy per consumer: the Figure 4
script, the Figure 6 script, the Figure 6 statistics derivation, and
``src.statistics``. These tests assert all consumers produce bit-identical
values, so the copies can be (and now are) a single canonical implementation.

Two look-alikes are DELIBERATELY different published behaviors, pinned here so
nobody "fixes" them into the shared version:

- ``fig6_resilience_stats.diversity_table`` embeds a second Markov entropy at
  Laplace smoothing 0.002 (column ``MarkovEntropy``), not the quartet's 0.01.
- The derive script normalizes CUI by named-cluster totals; the figure scripts
  and ``src.statistics`` normalize by all-cluster totals (unmapped included).
"""

import importlib.util
from pathlib import Path

import pytest

from src.statistics import (
    determinism,
    lempel_ziv_complexity,
    markov_entropy,
    recurrence_rate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

SEQUENCES = [
    ["A"] * 10,
    ["A", "B"] * 5,
    ["A", "A", "B", "C", "A", "B", "B", "C", "A", "A"],
    ["A", "B", "C", "A", "B", "C", "A", "C", "B", "A", "A", "B"],
    ["", "A", "A", "", "B", "A", "", "", "C", "A"],  # unmapped syllables kept as ""
]


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fig4():
    return _load("fig4_metrics", "scripts/generate_figures/figure_4_diversity_dynamics.py")


@pytest.fixture(scope="module")
def fig6():
    return _load("fig6_metrics", "scripts/generate_figures/figure_6_resilience_diversity.py")


@pytest.fixture(scope="module")
def derive():
    return _load("fig6_stats_metrics", "scripts/derive_tables/fig6_resilience_stats.py")


@pytest.mark.parametrize("seq_index", range(len(SEQUENCES)))
def test_quartet_agrees_across_all_consumers(fig4, fig6, derive, seq_index):
    seq = SEQUENCES[seq_index]
    for module in (fig4, fig6):
        assert module.lz_complexity(seq) == lempel_ziv_complexity(seq)
        assert module.recurrence_rate(seq) == recurrence_rate(seq)
        assert module.determinism(seq) == determinism(seq)
        assert module.markov_entropy(seq) == markov_entropy(seq)
    assert derive.lz_complexity(seq) == lempel_ziv_complexity(seq)
    assert derive.recurrence_rate(seq) == recurrence_rate(seq)
    assert derive.determinism(seq) == determinism(seq)
    assert derive.markov_entropy_seq(seq) == markov_entropy(seq)
    assert derive.markov_entropy_seq(seq, smoothing=0.01) == markov_entropy(
        seq, smoothing_factor=0.01
    )


def test_diversity_table_markov_smoothing_is_deliberately_different(derive):
    """The MarkovEntropy column in fig6_diversity_resilience_stats.csv uses
    smoothing 0.002. Consolidating it to 0.01 would change a published
    statistic; this test documents the divergence as intentional."""
    import inspect

    source = inspect.getsource(derive.diversity_table)
    assert "0.002" in source, (
        "diversity_table no longer contains the 0.002-smoothing Markov entropy; "
        "if this was consolidated, the published MarkovEntropy statistics changed"
    )
