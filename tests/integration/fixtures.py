# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pytest

import moeabench as mb


def _load_example_experiment(zip_name: str):
    exp = mb.experiment()
    exp.load(str(Path("examples") / zip_name))
    return exp


@pytest.fixture
def paired_experiments():
    """Fresh pair of canonical experiments derived from example_full assets."""
    exp1 = _load_example_experiment("example_full_exp1.zip")
    exp2 = _load_example_experiment("example_full_exp2.zip")
    return exp1, exp2


@pytest.fixture
def canonical_mop(paired_experiments):
    exp1, _ = paired_experiments
    return exp1.mop


@pytest.fixture
def canonical_gt(canonical_mop):
    return canonical_mop.pf(n_points=1000)
