# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: GT trace reordering must not reassign per-point clinical markers."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import moeabench as mb
from moeabench.diagnostics import qscore


class _Experiment:
    """Minimal GT-bearing source with deliberately unequal front sizes."""

    def __init__(self):
        self.mop = type("Mop", (), {"name": "DTLZ2"})()
        self.moea = type("Moea", (), {"population": 100})()
        self.runs = [object()]
        self.name = "Experimental"
        self._front = np.column_stack((
            np.linspace(0.1, 0.9, 100),
            np.linspace(0.9, 0.1, 100),
            np.linspace(0.2, 0.8, 100),
        ))
        self._gt = np.column_stack((
            np.linspace(0.0, 1.0, 500),
            np.linspace(1.0, 0.0, 500),
            np.linspace(0.0, 0.5, 500),
        ))

    def front(self):
        return self._front

    def optimal_front(self):
        return self._gt


@pytest.mark.scope_integration
@pytest.mark.level_basic
@pytest.mark.parametrize("mode", ["static", "interactive"])
@pytest.mark.parametrize("explicit", [False, True])
def test_clinical_markers_follow_source_after_gt_reordering(monkeypatch, mode, explicit):
    exp = _Experiment()
    called = []

    def marker_scores(source, **kwargs):
        called.append(source)
        if source is exp:
            return np.full(100, 0.75)
        # Deliberately distinct clinical class for a positional GT array:
        # equal lengths alone would NOT catch incorrect source associations.
        if isinstance(source, np.ndarray) and source.shape == (500, 3):
            return np.full(500, 0.25)
        raise AssertionError(f"unexpected marker source {type(source).__name__}")

    monkeypatch.setattr(qscore, "q_closeness_points", marker_scores)
    inputs = (exp, exp.optimal_front()) if explicit else (exp,)
    plot = mb.view.topology(*inputs, markers=True, mode=mode, show=False)
    try:
        assert len(plot.vet_pts) == (3 if explicit else 2)
        assert len(plot.marker_styles) == len(plot.vet_pts)
        assert plot.gt_flags[0] is True  # inferred reference reordered first
        assert len(plot.vet_pts[0]) == 500
        assert plot.marker_styles[0] is None  # GT is context, not classified data
        # The empirical source moved to trace index 1.
        assert len(plot.vet_pts[1]) == 100
        assert plot.marker_styles[1]["symbol"] == ["circle"] * 100
        assert plot.marker_styles[1]["size"] == [6] * 100
        assert called[0] is exp
        if explicit:
            assert called[1] is inputs[1]
            assert plot.marker_styles[2]["symbol"] == ["circle-open"] * 500
        if mode == "interactive":
            assert len(plot.figure.data) >= 2
    finally:
        plt.close("all")


@pytest.mark.scope_integration
@pytest.mark.level_basic
def test_mismatched_clinical_marker_count_is_not_assigned(monkeypatch):
    exp = _Experiment()
    monkeypatch.setattr(qscore, "q_closeness_points", lambda source, **kwargs: np.ones(7))
    with pytest.warns(RuntimeWarning, match="expected 100"):
        plot = mb.view.topology(exp, markers=True, mode="static", show=False)
    try:
        assert plot.marker_styles[1] is None
    finally:
        plt.close("all")


@pytest.mark.scope_integration
@pytest.mark.level_basic
def test_trace_order_unchanged_when_gt_disabled(monkeypatch):
    exp = _Experiment()
    monkeypatch.setattr(qscore, "q_closeness_points", lambda source, **kwargs: np.full(100, 0.9))
    plot = mb.view.topology(exp, markers=True, show_gt=False, mode="static", show=False)
    try:
        assert len(plot.vet_pts) == 1
        np.testing.assert_array_equal(plot.vet_pts[0], exp.front())
        assert plot.marker_styles[0]["symbol"] == ["circle"] * 100
    finally:
        plt.close("all")
