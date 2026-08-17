# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""Import smoke tests: every shared module and every locked scientific
dependency must import cleanly in the locked environment."""

import importlib

import pytest

REPO_MODULES = [
    "coping_dynamics",
    "coping_dynamics.config",
    "coping_dynamics.panel_letters",
    "coping_dynamics.plotting",
    "coping_dynamics.statistics",
]

THIRD_PARTY = [
    "matplotlib",
    "numpy",
    "openpyxl",
    "pandas",
    "scipy",
    "seaborn",
    "shap",
    "sklearn",
    "statsmodels",
    "xgboost",
]


@pytest.mark.parametrize("name", REPO_MODULES + THIRD_PARTY)
def test_module_imports(name):
    importlib.import_module(name)


def test_config_paths_resolve_inside_repo():
    from coping_dynamics import config

    repo_root = config.PROJECT_ROOT.resolve()
    for attr in dir(config):
        value = getattr(config, attr)
        if attr.isupper() and hasattr(value, "resolve") and hasattr(value, "is_absolute"):
            resolved = value.resolve()
            assert repo_root == resolved or repo_root in resolved.parents, (
                f"config.{attr} = {value} escapes the repository"
            )
