# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent full continuous g=0 curve dominance evidence (#175)."""
import json

import numpy as np
import pytest

from moeabench.mops import DTLZ5, DTLZ6
from moeabench.mops._dtlz56_g0_curve import audit_g0_curve_dominance


def _interior_curve_point():
    t = np.pi/4
    return np.array([.5*np.cos(t), .5*np.cos(t),
                     np.sqrt(.5)*np.cos(t), np.sin(t)])


@pytest.mark.parametrize("klass", [DTLZ5, DTLZ6])
def test_analytic_g0_curve_witness_for_strictly_dominated_positive_g(klass):
    z = _interior_curve_point()
    y = 1.1*z
    result = audit_g0_curve_dominance(y)
    assert result.status == "witness"
    w = np.asarray(result.witness)
    assert np.all(w <= y)
    assert np.any(w < y)
    np.testing.assert_allclose(np.linalg.norm(w), 1, atol=1e-14)
    assert result.candidate_g == pytest.approx(.1, abs=1e-13)

    mop = klass(M=4)
    report = mop.assess_gt(reference=np.vstack([z, y]), probes=16)
    assert report.curve_dominance.status == "unverified"
    assert report.curve_dominance.value["witness_count"] == 1
    assert report.curve_dominance.value["numerical_witnesses"][0]["index"] == 1
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize("klass", [DTLZ5, DTLZ6])
def test_g0_curve_exclusion_does_not_claim_global_pareto_validity(klass):
    # Research #65: known feasible positive-g vector *not* dominated by
    # any point on the full continuous g=0 curve, but dominated
    # along the same radial direction at smaller positive g.
    y = np.array([0.5893373, 0.4183464, 1.0554343, 0.3573078])
    check = audit_g0_curve_dominance(y)
    assert check.status == "no_curve_dominator"
    report = klass(M=4).assess_gt(reference=y[None, :], probes=16)
    assert report.curve_dominance.value["no_curve_dominator"] == 1
    assert report.curve_dominance.status == "unverified"
    assert report.validity.status != "pass"


def test_exact_g0_curve_point_is_not_falsely_called_dominated():
    result = audit_g0_curve_dominance(_interior_curve_point())
    assert result.status != "witness"


def test_curve_numeric_equality_remains_unverified():
    y = _interior_curve_point()
    y[0] += 2e-10
    result = audit_g0_curve_dominance(y, tolerance=1e-9)
    assert result.status == "unverified"


def test_curve_oracle_rejects_invalid_inputs():
    for candidate in ([np.nan, 1.], [-1., 1.], [1.]):
        with pytest.raises(ValueError):
            audit_g0_curve_dominance(candidate)
    with pytest.raises(ValueError):
        audit_g0_curve_dominance([.5, .5], tolerance=0)


def test_curve_oracle_is_independent_of_production_sampler(monkeypatch):
    import moeabench.mops._dtlz56_front as production
    monkeypatch.setattr(
        production, "scalarized_candidate",
        lambda *a, **k: pytest.fail("production scalarizer reused"))
    assert audit_g0_curve_dominance(1.1*_interior_curve_point()).status == "witness"
