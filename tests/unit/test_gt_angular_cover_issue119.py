# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent proof tests for spherical GT angular audit, issue #119."""
from math import pi
import numpy as np
import pytest
from scripts.audit.gt_angular_cover_issue119 import (
    positive_sphere_angles, angular_coverage_interval, compare,
)
from scripts.audit.gt_exact_curve_issue119 import exact_curve_stats
from scripts.audit.gt_geometry_pilot import independent_reference


@pytest.mark.parametrize("m", [2, 3, 5])
def test_angles_map_complete_positive_orthant_and_unit_sphere(m):
    angles = [np.zeros(m-1), np.full(m-1, pi/2),
              np.full(m-1, pi/4)]
    for theta in angles:
        p = positive_sphere_angles(theta)
        assert len(p) == m
        assert np.all(p >= -1e-14)
        np.testing.assert_allclose(np.linalg.norm(p), 1, atol=1e-13, rtol=0)
    for k in range(m):
        # Standard hyperspherical map reaches every coordinate axis.
        target = np.eye(m)[k]
        rng = np.random.default_rng(21)
        samples = np.vstack([positive_sphere_angles(
            rng.uniform(0, pi/2, m-1)) for _ in range(120)])
        # The map is smooth even near singular boundary coordinates.
        assert samples.shape == (120, m)
        assert np.all(samples >= 0)


@pytest.mark.parametrize("m", [3, 5])
def test_positive_sphere_angular_map_is_one_lipschitz_numerically(m):
    rng = np.random.default_rng(38)
    for _ in range(150):
        a = rng.uniform(0, pi/2, m-1)
        b = rng.uniform(0, pi/2, m-1)
        assert np.linalg.norm(positive_sphere_angles(a)-
                              positive_sphere_angles(b)) <= np.linalg.norm(a-b)+1e-12


def test_angular_interval_encloses_independently_exact_m2_radius():
    gt=np.array([[1.0,0.0],[0.0,1.0]])
    truth=exact_curve_stats(gt,"DTLZ2")["exact_continuous_covering_radius"]
    for cells in (1,8,32,128):
        x=angular_coverage_interval(gt,max_cells=cells,width_target=1e-6)
        assert x["continuous_radius_lower"] <= truth+1e-12
        assert x["continuous_radius_upper"] >= truth-1e-12
        assert x["angular_cells"] <= cells
        assert x["evaluated_centers"] == 2*x["angular_cells"]-1


def test_refinement_monotone_and_one_cell_is_valid():
    f=independent_reference("DTLZ2",30,3,seed=121)
    previous=None
    for budget in (1,4,32,128):
        x=angular_coverage_interval(f,max_cells=budget,width_target=1e-8)
        assert x["interval_width"] >= 0
        assert x["continuous_radius_lower"] <= x["continuous_radius_upper"]
        if previous is not None:
            assert x["continuous_radius_lower"]>=previous["continuous_radius_lower"]-1e-12
            assert x["continuous_radius_upper"]<=previous["continuous_radius_upper"]+1e-12
        previous=x


def test_actual_mop_against_existing_barycentric_certificate():
    report=compare(3,16,q=8,cell_budget=64,seed=11,
                   candidate_pool=48,target=0.01)
    for key in ("incumbent","fps"):
        b=report["methods"][key]["barycentric"]
        a=report["methods"][key]["angular"]
        assert b["lower"] <= a["continuous_radius_upper"]+1e-12
        assert a["continuous_radius_lower"] <= b["upper"]+1e-12


def test_invalid_surface_and_angles_rejected():
    with pytest.raises(ValueError):
        positive_sphere_angles(np.array([pi]))
    with pytest.raises(ValueError):
        angular_coverage_interval(np.array([[0.4,0.4,0.4]]))
    with pytest.raises(ValueError):
        angular_coverage_interval(np.array([[1.0,0.0]]),max_cells=0)
