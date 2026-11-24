# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent regression cases for singular original DTLZ5/6 angles (#175).

These are *new* tests. Previously accepted GT regression suites and
frozen baselines remain unchanged.
"""
import numpy as np
import pytest

from moeabench.mops import DTLZ5, DTLZ6
from moeabench.mops._dtlz56_independent_audit import (
    inspect_dtlz56_geometry,
)


@pytest.mark.parametrize("cls,kind", [(DTLZ5, "DTLZ5"), (DTLZ6, "DTLZ6")])
@pytest.mark.parametrize("m", [2, 3, 4, 5, 10])
def test_last_axis_g0_is_feasible_and_individually_pareto(cls, kind, m):
    """theta_1=pi/2 makes all remaining angles genuinely undefined."""
    endpoint = np.zeros((1, m))
    endpoint[0, -1] = 1.0
    mop = cls(M=m)
    result = inspect_dtlz56_geometry(
        endpoint, m=m, k=mop.K, problem=kind)
    assert result["outside_feasible_image"] == 0
    assert result["strictly_radially_dominated"] == 0
    assert result["certified_g0"] == 1
    assert result["unverified_positive_g"] == 0
    audit = mop.assess_gt(reference=endpoint, probes=16)
    # A valid individual point is not a certificate of complete M>=4 PF.
    assert audit.validity.status == (
        "pass" if m <= 3 else "unverified")
    assert audit.validity.value["certified_g0"] == 1
    assert audit.validity.value["outside_feasible_image"] == 0


@pytest.mark.parametrize("cls,kind", [(DTLZ5, "DTLZ5"), (DTLZ6, "DTLZ6")])
@pytest.mark.parametrize("m", [3, 4, 5])
def test_positive_g_last_axis_is_feasible_but_radially_dominated(
        cls, kind, m):
    """The same singular ray at g>0 has a feasible smaller-radius witness."""
    endpoint = np.zeros((1, m))
    endpoint[0, -1] = 1.1
    mop = cls(M=m)
    result = inspect_dtlz56_geometry(
        endpoint, m=m, k=mop.K, problem=kind)
    assert result["outside_feasible_image"] == 0
    assert result["strictly_radially_dominated"] == 1
    assert result["certified_g0"] == 0
    assert mop.assess_gt(reference=endpoint, probes=16).validity.status == "fail"


@pytest.mark.parametrize("cls,kind", [(DTLZ5, "DTLZ5"), (DTLZ6, "DTLZ6")])
def test_true_angular_infeasibility_is_not_hidden_by_singularity_fix(cls, kind):
    mop = cls(M=4)
    # This unit vector has a defined inner theta_2=pi/2, which is
    # impossible at g=0, even though one other inner angle is singular.
    impossible = np.array([[0., 0., 1., 0.]])
    result = inspect_dtlz56_geometry(
        impossible, m=4, k=mop.K, problem=kind)
    assert result["outside_feasible_image"] == 1
    assert mop.assess_gt(reference=impossible, probes=16).validity.status == "fail"


@pytest.mark.parametrize("cls,kind", [(DTLZ5, "DTLZ5"), (DTLZ6, "DTLZ6")])
def test_original_evaluator_generates_singular_last_axis_independently(cls, kind):
    mop = cls(M=5)
    decision = np.empty((1, mop.N), dtype=float)
    decision[:, :mop.M-1] = .5
    decision[:, 0] = 1.
    decision[:, mop.M-1:] = .5 if kind == "DTLZ5" else 0.
    objective = mop.evaluation(decision)["F"]
    expected = np.zeros_like(objective)
    expected[0, -1] = 1.0
    np.testing.assert_allclose(objective, expected, atol=1e-14)
    result = inspect_dtlz56_geometry(
        expected, m=mop.M, k=mop.K, problem=kind)
    assert result["certified_g0"] == 1
