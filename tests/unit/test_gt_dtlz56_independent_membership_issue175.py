# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent DTLZ5/6 GT checks: no production sampler as its oracle."""
import numpy as np
import pytest

from moeabench.mops import DTLZ5, DTLZ6
from moeabench.mops._dtlz56_independent_audit import inspect_dtlz56_geometry


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_known_g_zero_curve_has_proved_global_pareto_membership(cls, m):
    mop = cls(M=m)
    # Use only the analytically established g=0 curve: its Euclidean
    # norm is 1, the global minimum feasible radius of the original MOP.
    angles = np.linspace(0.0, np.pi/2, 7)
    coeff = np.array([2**(-(m-2)/2)] +
                     [2**(-(m-j)/2) for j in range(2, m)])
    curve = np.column_stack((
        np.cos(angles)[:, None] * coeff[None, :],
        np.sin(angles)))
    outcome = mop.assess_gt(reference=curve, probes=16)
    assert outcome.validity.status == "pass", outcome.validity.to_dict()
    assert outcome.validity.value["certified_g0"] == len(curve)
    assert outcome.validity.value["unverified_positive_g"] == 0


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
def test_full_original_gt_positively_identifies_g_nonzero_without_false_pass(cls):
    mop = cls(M=4)
    outcome = mop.assess_gt(n_points=12, probes=16)
    assert outcome.validity.status == "unverified", outcome.validity.to_dict()
    assert outcome.validity.value["violations"] == 0
    assert outcome.validity.value["unverified_positive_g"] > 0
    assert outcome.components.status == "pass"
    assert outcome.components.value["g_zero"] >= 2
    assert outcome.components.value["g_positive"] > 0


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
def test_same_direction_larger_radius_is_provably_dominated(cls):
    mop = cls(M=4)
    g0 = mop.pf(n_points=2)
    # Artificially enlarged g=0 angular ray: feasible for positive
    # g but strongly dominated by its original g=0 vector.
    enlarged = g0 * 1.1
    outcome = mop.assess_gt(reference=enlarged, probes=16)
    assert outcome.validity.status == "fail"
    assert outcome.validity.value["strictly_radially_dominated"] == 2


@pytest.mark.parametrize("cls", [DTLZ5, DTLZ6])
def test_outside_original_feasible_radius_is_rejected(cls):
    mop = cls(M=4)
    gmax = mop.K/4 if cls is DTLZ5 else float(mop.K)
    outside = mop.pf(n_points=2) * (gmax+2)
    outcome = mop.assess_gt(reference=outside, probes=16)
    assert outcome.validity.status == "fail"
    assert outcome.validity.value["outside_feasible_image"] == 2


def test_angular_necessity_detects_impossible_positive_sphere_direction():
    # At g=0 all inner angles are pi/4, independent of the MOP evaluator.
    # A unit vector with a different inner angle is outside the
    # original feasible image despite satisfying the unit-norm check.
    mop = DTLZ5(M=4)
    impossible = np.array([[1., 0., 0., 0.]])
    result = mop.assess_gt(reference=impossible, probes=16)
    assert result.validity.status == "fail"
    assert result.validity.value["outside_feasible_image"] == 1


def test_custom_subclass_never_inherits_original_dtlz56_oracle():
    class CustomDTLZ5(DTLZ5):
        pass
    outcome = CustomDTLZ5(M=4).assess_gt(reference=np.ones((2,4)), probes=16)
    assert outcome.validity.status == "unverified"


def test_independent_oracle_is_not_importing_or_invoking_production_scalarizer(monkeypatch):
    import moeabench.mops._dtlz56_front as production
    monkeypatch.setattr(production, "scalarized_candidate",
                        lambda *args, **kwargs: pytest.fail("production oracle reused"))
    f = DTLZ5(M=3).pf(n_points=7)
    evidence = inspect_dtlz56_geometry(
        f, m=3, k=10, problem="DTLZ5")
    assert evidence["violations"] == 0
    assert evidence["certified_g0"] == len(f)
