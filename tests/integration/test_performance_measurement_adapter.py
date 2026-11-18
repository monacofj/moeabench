# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke the performance measurement adapter with a real, tiny MOEA run.

This does not establish a performance baseline or certify solver quality.
"""
import numpy as np

from moeabench.performance_stability import _measure_case


def test_explicit_case_measures_one_real_solver_run():
    case = {
        "id": "tiny-real-measurement",
        "mop": {"name": "DTLZ2", "kwargs": {"M": 2}},
        "moea": {"name": "NSGA2deap", "kwargs": {"population": 12, "generations": 2}},
        "metric": "igd",
        "reference_front": [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
    }
    result = _measure_case(case, seed=7)
    assert np.isfinite(result)
    assert result >= 0
