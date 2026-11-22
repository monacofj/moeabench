# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent degenerate three-objective GT audit for issue #89.

Adds coverage only; existing regression tests and calibrated references
remain untouched.
"""

import numpy as np
import pytest

from scripts.audit.hv_dpf1_degenerate_issue89 import (
    evaluate_case,
    exact_true_dpf1_hv3,
    independent_dpf1_hv3,
)


def test_true_dpf1_three_objective_hv_from_exact_integral():
    assert exact_true_dpf1_hv3() == pytest.approx(
        (1.1**3 - 0.1**3) / 3 + 0.1 * 1.1**2, abs=1e-13
    )
    endpoints = np.array([[0, .5, .5], [.5, 0, 0]])
    observed = independent_dpf1_hv3(
        endpoints, ideal=np.zeros(3), nadir=np.array([.5, .5, .5])
    )
    # Two endpoint cuboids with identical y,z coordinates in the second.
    assert observed == pytest.approx(.1 * .1 * 1.1 + .1 * 1.1 * 1.1)


@pytest.mark.parametrize("count,check_public", [
    (2000, True),
    (10000, False),
])
def test_actual_dpf1_degenerate_gt_has_finite_sampling_gap(count, check_public):
    evidence = evaluate_case(count, check_public=check_public)
    assert evidence["sampled_gt_h_abs"] > 1.0
    assert 0 < evidence["continuous_bound_ratio"] < 1
    assert evidence["increment"] > 0
    assert evidence["independent_hv_gt"] < exact_true_dpf1_hv3()
    if check_public:
        assert evidence["public_h_abs"] == pytest.approx(
            evidence["sampled_gt_h_abs"], rel=2e-9, abs=1e-11
        )
