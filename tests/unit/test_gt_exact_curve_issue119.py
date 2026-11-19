# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New M=2 analytic oracle tests; old regression tests are untouched."""
import numpy as np
import pytest

from scripts.audit.gt_exact_curve_issue119 import exact_curve_stats, compare


def test_line_endpoints_have_proved_mean_and_radius():
    p=np.array([[0.5,0.0],[0.0,0.5]])
    v=exact_curve_stats(p,"DTLZ1")
    length=np.sqrt(2)/2
    np.testing.assert_allclose(v["continuous_mean_euclidean_nearest_distance"],
                               length/4,rtol=0,atol=1e-14)
    np.testing.assert_allclose(v["exact_continuous_covering_radius"],
                               length/2,rtol=0,atol=1e-14)
    assert v["endpoints_included"]


def test_quarter_circle_endpoints_have_proved_chord_radius_and_mean():
    p=np.array([[1.0,0.0],[0.0,1.0]])
    v=exact_curve_stats(p,"DTLZ2")
    np.testing.assert_allclose(v["exact_continuous_covering_radius"],
                               2*np.sin(np.pi/8),rtol=0,atol=1e-14)
    np.testing.assert_allclose(
        v["continuous_mean_euclidean_nearest_distance"],
        8*(1-np.cos(np.pi/8))/(np.pi/2),rtol=0,atol=1e-14)


@pytest.mark.parametrize("name",["DTLZ1","DTLZ2"])
def test_empirical_reference_max_is_lower_bound_on_proved_continuous_radius(name):
    x=compare(name,32,candidate_pool=96,seed=11,ref_sizes=(128,512),iterations=3)
    for method,item in x["methods"].items():
        v=item["exact"]
        assert v["point_count"]==32
        assert v["exact_continuous_covering_radius"]>0
        assert v["continuous_mean_euclidean_nearest_distance"]>0
        for stats in item["reference_estimates"].values():
            assert stats["sampled_max_lower_bound"]<=v["exact_continuous_covering_radius"]+1e-12
            assert stats["max_gap_to_exact"]>=-1e-12


def test_invalid_points_and_off_surface_rejected():
    with pytest.raises(ValueError,match="Off-simplex"):
        exact_curve_stats(np.array([[0.2,0.2]]),"DTLZ1")
    with pytest.raises(ValueError,match="Off-hypersphere"):
        exact_curve_stats(np.array([[0.6,0.6]]),"DTLZ2")
    with pytest.raises(ValueError):
        exact_curve_stats(np.ones((3,3)),"DTLZ2")
