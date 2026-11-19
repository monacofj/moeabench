# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent #119 local angular tests; existing regressions are untouched."""
from math import pi

import numpy as np
import pytest

from scripts.audit.gt_exact_curve_issue119 import exact_curve_stats
from scripts.audit.gt_geometry_pilot import independent_reference
from scripts.audit.gt_local_angular_issue119 import (
    local_weights, local_radius, tighter_angular_interval, compare,
)
from scripts.audit.gt_angular_cover_issue119 import (
    angular_coverage_interval, positive_sphere_angles,
)


@pytest.mark.parametrize("m",[2,3,5,8])
def test_local_weight_is_upper_bound_on_exact_tangent_norm(m):
    rng=np.random.default_rng(819)
    for _ in range(35):
        lo=rng.uniform(0,pi/3,m-1)
        hi=lo+rng.uniform(0,pi/2-lo)
        weights=local_weights(lo)
        for _ in range(6):
            theta=rng.uniform(lo,hi)
            exact=np.r_[1.0,np.cumprod(np.cos(theta[:-1]))]
            assert np.all(exact <= weights + 1e-12)
        assert np.all(weights <= 1+1e-12)


@pytest.mark.parametrize("m",[2,3,5])
def test_local_chord_bound_on_random_points_inside_cells(m):
    rng=np.random.default_rng(31)
    for _ in range(40):
        a=rng.uniform(0,pi/3,m-1)
        b=a+rng.uniform(0,pi/2-a)
        center=positive_sphere_angles((a+b)/2)
        radius=local_radius(a,b)
        assert radius <= np.linalg.norm(b-a)/2 + 1e-12
        for _ in range(40):
            point=positive_sphere_angles(rng.uniform(a,b))
            assert np.linalg.norm(point-center) <= radius + 1e-12


def test_local_m2_bound_encloses_exact_continuous_radius():
    gt=np.array([[1.0,0.0],[0.0,1.0]])
    exact=exact_curve_stats(gt,"DTLZ2")["exact_continuous_covering_radius"]
    for cells in (1,8,64):
        res=tighter_angular_interval(gt,max_cells=cells,width_target=1e-8)
        assert res["continuous_radius_lower"] <= exact + 1e-12
        assert res["continuous_radius_upper"] >= exact - 1e-12
        assert res["angular_cells"] <= cells
        assert res["evaluated_centers"] == 2*res["angular_cells"]-1


def test_local_refinement_is_monotone_and_valid():
    front=independent_reference("DTLZ2",32,3,seed=22)
    previous=None
    for max_cells in (1,4,32,128):
        current=tighter_angular_interval(front,max_cells=max_cells,
                                         width_target=1e-9)
        assert current["interval_width"] >= 0
        if previous:
            assert current["continuous_radius_lower"] >= previous["continuous_radius_lower"]-1e-12
            assert current["continuous_radius_upper"] <= previous["continuous_radius_upper"]+1e-12
        previous=current


def test_same_actual_mop_gt_has_valid_intersected_intervals():
    result=compare(3,24,q=8,cell_budget=64,seed=101,
                   candidate_pool=96,target=1e-4)
    for name in ("incumbent","fps"):
        info=result["methods"][name]
        combined=info["combined"]
        for key in ("original_angular","local_angular"):
            res=info[key]
            assert combined["lower"] >= res["continuous_radius_lower"]-1e-12
            assert combined["upper"] <= res["continuous_radius_upper"]+1e-12
        bary=info["barycentric"]
        assert combined["lower"] >= bary["lower"]-1e-12
        assert combined["upper"] <= bary["upper"]+1e-12


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        local_radius(np.array([0.3]),np.array([0.2]))
    with pytest.raises(ValueError):
        local_weights(np.array([pi]))
    with pytest.raises(ValueError):
        tighter_angular_interval(np.array([[0.2,0.2,0.2]]))
