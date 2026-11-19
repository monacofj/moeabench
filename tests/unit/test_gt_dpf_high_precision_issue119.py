# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent precision and cross-dimensional tests; no regression edits."""
import pytest
from scripts.audit.gt_dpf_high_precision_issue119 import precision_probe,dpf5_dimension_probe


def test_mp_reference_recovers_values_lost_by_naive_subtraction():
    pytest.importorskip('mpmath')
    r=precision_probe(128,160)
    assert r["naive_zero_but_mp_positive"] > 0
    assert r["mp_positive_count"] >= r["naive_zero_but_mp_positive"]
    assert r["stable_zero_but_mp_positive"] <= r["naive_zero_but_mp_positive"]


def test_precision_reference_stabilizes_with_more_digits():
    pytest.importorskip('mpmath')
    a=precision_probe(64,120)
    b=precision_probe(64,200)
    assert a["mp_positive_count"] == b["mp_positive_count"]
    assert a["mp_to_float64_zero_count"] == b["mp_to_float64_zero_count"]


@pytest.mark.parametrize("m",[3,4,5,6,10])
def test_canonical_spherical_reconstruction_is_dimensionality_consistent(m):
    r=dpf5_dimension_probe(m,128)
    assert r["P"]["reference_norm2_minus_one_max_abs"] < 1e-10
    assert r["D"]["reference_norm2_minus_one_max_abs"] < 1e-10
    assert r["D"]["max_abs_component_difference"] < 1e-10
    assert not r["globally_complete_certified"]
