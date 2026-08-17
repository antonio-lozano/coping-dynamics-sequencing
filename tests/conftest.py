# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""Shared test setup: figures must never try to open a display."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
