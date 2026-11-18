# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Additional issue #96 guards: nonnegative metric scores are mandatory."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import moeabench as mb
from moeabench import performance_stability as ps


def test_statistical_assessment_rejects_negative_observations():
    case = {
        "id": "example",
        "baseline_values": [0.15, 0.15, 0.15],
        "max_relative_degradation": 0.01,
        "max_absolute_degradation": 0.01,
    }
    for invalid in ([-0.1, 0.1, 0.1], [np.nan, 0.1, 0.1]):
        with pytest.raises(ValueError, match="nonnegative"):
            ps.assess_case(case, invalid)


def test_measurement_adapter_rejects_negative_metric(monkeypatch):
    problem = object()
    algorithm = object()
    monkeypatch.setattr(mb.mops, "DTLZ2", lambda **kwargs: problem)
    monkeypatch.setattr(mb.moeas, "NSGA2deap",
                        lambda **kwargs: algorithm)

    class FakeExperiment:
        def __init__(self, mop, moea):
            assert mop is problem and moea is algorithm

        def run(self, repeat, silent):
            assert repeat == 1 and silent is True

        def front(self):
            return np.array([[0.1, 0.9]], dtype=float)

    monkeypatch.setattr(mb, "experiment", FakeExperiment)
    monkeypatch.setattr(mb.metrics, "igd", lambda *args, **kwargs: -0.01)

    case = {
        "id": "negative-result",
        "mop": {"name": "DTLZ2", "kwargs": {"M": 2}},
        "moea": {"name": "NSGA2deap", "kwargs": {}},
        "metric": "igd",
        "reference_front": [[0, 1], [1, 0]],
    }
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ps._measure_case(case, 7)
