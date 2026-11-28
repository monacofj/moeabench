# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New source selection observation tests; never run 448 scalarizations in unit."""
from types import SimpleNamespace
import numpy as np
import pytest

from moeabench.mops import DTLZ5
from moeabench.mops import _dtlz56_front as front
from scripts.audit.dtlz5_final_selection_trace import (
    trace_original_dtlz5_selection,
)
from scripts.audit.dtlz56_paired_gt_evidence import _original_objectives


def test_research_tracer_readonly_wrapper_and_independent_objectives(monkeypatch):
    point = np.array([0.2, 1.0, 0.0, 0.5], dtype=float)
    objective, _ = _original_objectives(
        point[None, :], problem="DTLZ5", m=4, k=1,
    )
    original_scalarizer = front.scalarized_candidate
    original_reconstructor = front.decision_vector
    calls = []

    def fake_scalarizer(weights, max_g, **kwargs):
        calls.append("scalar")
        return SimpleNamespace(
            objectives=objective[0].copy(), g=0.0, scale=1.0,
            abs_gap=1e-7, nodes=3, angular_active_faces=(1, 0),
        )

    def fake_reconstruct(candidate, **kwargs):
        calls.append("reconstruct")
        return point.copy()

    def fake_ps(self, *, n_points):
        candidate = front.scalarized_candidate(
            np.ones(4), 0.25, abs_tol=1e-7,
        )
        answer = front.decision_vector(
            candidate, M=4, K=1, problem="DTLZ5",
        )
        return np.repeat(answer[None, :], n_points, axis=0)

    monkeypatch.setattr(front, "scalarized_candidate", fake_scalarizer)
    monkeypatch.setattr(front, "decision_vector", fake_reconstruct)
    monkeypatch.setattr(DTLZ5, "ps", fake_ps)
    report = trace_original_dtlz5_selection()
    assert calls == ["scalar", "reconstruct"]
    assert report["num_scalarized_candidates"] == 1
    assert report["max_absolute_objective_reconstruction_discrepancy"] == 0
    assert report["selected_rows"][0]["trace"]["scalarization_nodes"] == 3
    assert report["selected_rows"][0]["trace"]["active_faces"] == [1, 0]
    assert report["scientific_certification"] == "unverified"
    # Tracing never permanently replaces the functions.
    assert front.scalarized_candidate is fake_scalarizer
    assert front.decision_vector is fake_reconstruct
    assert original_scalarizer is not fake_scalarizer
    assert original_reconstructor is not fake_reconstruct


@pytest.mark.parametrize("n", [0, 8, 17, True])
def test_invalid_point_counts_rejected_without_sampler(n, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No production sampling allowed")
    monkeypatch.setattr(DTLZ5, "ps", forbidden)
    with pytest.raises(ValueError):
        trace_original_dtlz5_selection(n_points=n)


def test_tracer_restores_observers_after_failed_source(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("provisional source failure")
    original = front.scalarized_candidate
    reconstructor = front.decision_vector
    monkeypatch.setattr(DTLZ5, "ps", fail)
    with pytest.raises(RuntimeError):
        trace_original_dtlz5_selection()
    assert front.scalarized_candidate is original
    assert front.decision_vector is reconstructor
