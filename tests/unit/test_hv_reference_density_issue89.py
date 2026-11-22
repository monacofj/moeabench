# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent finite-sampler HV evidence; no historical tests are changed."""

import numpy as np
import pytest

from scripts.audit.hv_reference_density_issue89 import (
    evaluate_case,
    independent_hv2,
    exact_continuous_hv2,
)


def test_independent_2d_oracle_has_a_closed_form_rectangle_union():
    gt = np.array([[0.0, 0.5], [0.5, 0.0]])
    p = np.array([[0.25, 0.25]])
    a = independent_hv2(gt, [0, 0], [.5, .5])
    b = independent_hv2(p, [0, 0], [.5, .5])
    assert a == pytest.approx(.21, abs=1e-12)
    assert b == pytest.approx(.36, abs=1e-12)
    assert b / a == pytest.approx(12 / 7)


@pytest.mark.parametrize("name,n", [
    ("DTLZ1", 2000),
    ("DTLZ1", 10000),
    ("DTLZ2", 2000),
])
def test_actual_sampler_gt_density_underestimates_true_pf_hv(name, n):
    """True-front insertion changes the denominator without a solver defect."""
    evidence = evaluate_case(name, n)
    assert evidence["gt_points"] == n
    assert evidence["independent_h_abs"] > 1.0
    assert evidence["public_h_abs"] == pytest.approx(
        evidence["independent_h_abs"], rel=2e-10, abs=1e-12
    )
    assert evidence["increment"] > 0.0
    assert evidence["independent_hv_gt"] < exact_continuous_hv2(name)
    assert evidence["independent_hv_evaluated"] < exact_continuous_hv2(name)
    assert 0.0 < evidence["continuous_bound_ratio"] < 1.0

def test_exact_continuous_denominators_follow_proved_2d_front_geometry():
    assert exact_continuous_hv2("DTLZ1") == pytest.approx(.71)
    assert exact_continuous_hv2("DTLZ2") == pytest.approx(
        1.21 - np.pi / 4
    )
