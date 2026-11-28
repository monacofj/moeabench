# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New inexpensive tests of fixed source preference sensitivity protocol."""
import numpy as np
import pytest
from types import SimpleNamespace
from scripts.audit import dtlz5_candidate157_tolerance_sweep as sweep


def test_sobol_source_preference_matches_archived_scalarizer_weights():
    observed = sweep.original_source_ray_157()
    assert np.array_equal(observed, sweep.SOURCE_WEIGHT_VALUES)
    assert observed.shape == (4,)


def test_sweep_preserves_distinct_tolerances_and_never_certifies(monkeypatch):
    def fake_scalarize(weights, max_g, *, abs_tol, max_nodes):
        assert max_nodes == 30000
        return SimpleNamespace(
            nodes=2, abs_gap=abs_tol, g=0.0,
            objectives=np.ones(4), angular_active_faces=(0, 1),
        )
    def fake_decision(candidate, **kwargs):
        return np.array([0.25, 0.5, 0.5, 0.5])
    monkeypatch.setattr(sweep, "original_source_ray_157", lambda: np.ones(4))
    monkeypatch.setattr(sweep, "scalarized_candidate", fake_scalarize)
    monkeypatch.setattr(sweep, "decision_vector", fake_decision)
    result = sweep.sweep_source_ray_157(tolerances=(1e-7,1e-9))
    assert [row["tolerance"] for row in result["observations"]] == [1e-7,1e-9]
    assert all(row["scientific_membership"] == "unverified"
               for row in result["observations"])
    assert result["scientific_certification"] == "unverified"


@pytest.mark.parametrize("bad", [0,-1,float("nan"),float("inf")])
def test_reject_nonpositive_or_nonfinite_tolerance(bad,monkeypatch):
    monkeypatch.setattr(sweep, "original_source_ray_157", lambda: np.ones(4))
    with pytest.raises(ValueError):
        sweep.sweep_source_ray_157(tolerances=(bad,))
