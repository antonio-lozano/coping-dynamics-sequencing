# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""Golden-value tests for the canonical metric implementations in
``src.statistics``.

The values below were computed from the implementations as published and pin
their exact behavior: any refactor (e.g. consolidating duplicated metric code
into this module) must keep every number bit-for-bit, because the manuscript
statistics and the MANIFEST hashes depend on them.

Known quirk, pinned deliberately: ``determinism`` counts the zero-offset
diagonal in the numerator but excludes it from the denominator, so highly
repetitive sequences score above 1. Changing that would change published
numbers and is the authors' call, not a refactor's.
"""

import numpy as np
import pandas as pd
import pytest

from src.statistics import (
    cohens_d,
    compute_bout_duration,
    compute_diversity_metrics,
    determinism,
    lempel_ziv_complexity,
    markov_entropy,
    recurrence_rate,
    transition_sequence_metrics,
)

CONSTANT = ["A"] * 10
ALTERNATING = ["A", "B"] * 5
MIXED = ["A", "A", "B", "C", "A", "B", "B", "C", "A", "A"]
THREE_STATE = ["A", "B", "C", "A", "B", "C", "A", "C", "B", "A", "A", "B"]

GOLDEN = {
    "constant": (CONSTANT, 3, 1.0, 1.0888888888888888, 0.0),
    "alternating": (ALTERNATING, 4, 0.5, 1.25, 0.022923493099305151),
    "mixed": (MIXED, 6, 0.38, 0.8571428571428571, 0.81371360499857359),
    "three_state": (
        THREE_STATE,
        7,
        0.34722222222222221,
        0.73684210526315785,
        1.1254632682527661,
    ),
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_transition_metrics_golden(name):
    sequence, lz, rr, det, me = GOLDEN[name]
    assert lempel_ziv_complexity(sequence) == lz
    assert recurrence_rate(sequence) == pytest.approx(rr, rel=1e-15, abs=1e-15)
    assert determinism(sequence) == pytest.approx(det, rel=1e-15, abs=1e-15)
    assert markov_entropy(sequence) == pytest.approx(me, rel=1e-15, abs=1e-15)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_transition_sequence_metrics_bundles_the_same_functions(name):
    sequence = GOLDEN[name][0]
    bundle = transition_sequence_metrics(sequence)
    assert bundle["lempel_ziv_complexity"] == lempel_ziv_complexity(sequence)
    assert bundle["recurrence_rate"] == recurrence_rate(sequence)
    assert bundle["determinism"] == determinism(sequence)
    assert bundle["markov_entropy"] == markov_entropy(sequence)


def test_diversity_metrics_uniform_two_states():
    result = compute_diversity_metrics(["A", "A", "B", "B"])
    assert result["shannon_entropy_index"] == pytest.approx(np.log(2), rel=1e-15)
    assert result["simpson_index"] == pytest.approx(0.5, rel=1e-15)
    assert result["evenness_index"] == pytest.approx(1.0, rel=1e-15)
    assert result["cumulative_usage_index"] == pytest.approx(0.0, abs=1e-15)


def test_diversity_metrics_with_blank_labels():
    result = compute_diversity_metrics(["A", "A", "A", "B", "C", "C", "", ""])
    assert result["shannon_entropy_index"] == pytest.approx(1.3208883431493221, rel=1e-15)
    assert result["simpson_index"] == pytest.approx(0.71875, rel=1e-15)
    assert result["evenness_index"] == pytest.approx(0.95281953111478324, rel=1e-15)
    assert result["cumulative_usage_index"] == pytest.approx(-0.24999999999999975, rel=1e-12)


def test_bout_duration_splits_runs_and_names_blanks():
    result = compute_bout_duration(["A", "A", "B", "", "C"])
    assert result.to_dict("records") == [
        {"cluster": "A", "bout_duration_seconds": 0.5},
        {"cluster": "B", "bout_duration_seconds": 0.25},
        {"cluster": "Unmapped_or_excluded", "bout_duration_seconds": 0.25},
        {"cluster": "C", "bout_duration_seconds": 0.25},
    ]


def test_bout_duration_empty_sequence():
    assert compute_bout_duration([]).empty


def test_markov_entropy_single_state_is_zero():
    assert markov_entropy(["A", "A", "A"]) == 0.0


def test_cohens_d_unit_shift():
    d = cohens_d(pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 3.0, 4.0]))
    assert d == pytest.approx(-1.0, rel=1e-15)


def test_cohens_d_insufficient_data_is_nan():
    assert np.isnan(cohens_d(pd.Series([1.0]), pd.Series([2.0, 3.0])))
