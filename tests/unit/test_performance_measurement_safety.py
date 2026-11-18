# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Preserved negative/nonfinite score guards against the current measurement API."""
from __future__ import annotations
import numpy as np
import pytest
import moeabench as mb
from moeabench import certification as cert


def test_statistical_assessment_rejects_negative_observations():
    case=cert._default_cases()[0]
    for invalid in ([-0.1]*8, [np.nan]*8, [np.inf]*8):
        with pytest.raises(ValueError):
            cert.assess_paired(case, invalid, [0.5]*8, [0.5]*64)
        with pytest.raises(ValueError):
            cert.assess_paired(case, [0.5]*8, invalid, [0.5]*64)


def test_measurement_adapter_rejects_negative_metric(monkeypatch):
    problem=object()
    algorithm=object()
    monkeypatch.setattr(mb.mops, "DTLZ2", lambda **kwargs: problem)
    monkeypatch.setattr(mb.moeas, "NSGA2", lambda **kwargs: algorithm)
    class FakeExperiment:
        def __init__(self, mop, moea):
            assert mop is problem and moea is algorithm
        def run(self, repeat, silent):
            assert repeat == 1 and silent
        def front(self):
            return np.array([[0.1, 0.9, 0.5]])
    monkeypatch.setattr(mb, "experiment", FakeExperiment)
    monkeypatch.setattr(mb.metrics, "igdplus", lambda *a, **kw: -0.01)
    case=cert._default_cases()[0]
    with pytest.raises(ValueError, match="Invalid igdplus"):
        cert._measure(case, case["seeds"][0], np.array([[0,1,0], [1,0,0]]))
