# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent research tests for M=2 exact grids; no old regressions touched."""
import numpy as np
import pytest

from scripts.audit.gt_exact_grids_issue119 import (
    exact_arclength_grid, theoretical_optimum, compare,
)
from scripts.audit.gt_exact_curve_issue119 import exact_curve_stats


@pytest.mark.parametrize("name",["DTLZ1","DTLZ2"])
@pytest.mark.parametrize("budget",[2,3,16,64])
@pytest.mark.parametrize("endpoints",[True,False])
def test_grids_attain_closed_form_continuous_optima(name,budget,endpoints):
    p=exact_arclength_grid(name,budget,endpoints=endpoints)
    truth=theoretical_optimum(name,budget,endpoints=endpoints)
    actual=exact_curve_stats(p,name)
    assert actual["point_count"]==budget
    assert actual["endpoints_included"]==endpoints
    for key,expected in truth.items():
        np.testing.assert_allclose(actual[key],expected,rtol=0,atol=1e-12)


@pytest.mark.parametrize("name",["DTLZ1","DTLZ2"])
def test_midcell_grids_improve_both_continuous_metrics_at_fixed_budget(name):
    n=20
    constrained=theoretical_optimum(name,n,endpoints=True)
    free=theoretical_optimum(name,n,endpoints=False)
    for metric in constrained:
        assert free[metric] < constrained[metric]


def test_exact_sampled_control_reuses_unchanged_dtlz1_mop():
    case=compare("DTLZ1",16,seed=23,pool_size=64)
    assert set(case["methods"])=={
        "incumbent","fps","equal_arclength_with_axes",
        "equal_arclength_midpoints",
    }
    incumbent=case["methods"]["incumbent"]["observed"]
    endpoint=case["methods"]["equal_arclength_with_axes"]["observed"]
    assert endpoint["exact_continuous_covering_radius"] < incumbent["exact_continuous_covering_radius"]


def test_bad_budget_and_geometry_rejected():
    with pytest.raises(ValueError):
        exact_arclength_grid("DTLZ4",10,endpoints=True)
    with pytest.raises(ValueError):
        theoretical_optimum("DTLZ1",1,endpoints=False)
