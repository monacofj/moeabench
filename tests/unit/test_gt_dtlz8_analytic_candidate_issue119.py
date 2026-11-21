# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent analytic DTLZ8 Pareto candidate and covering controls (#119).

Exercises the actual DTLZ8 evaluator but DOES NOT install the candidate as
production PS/PF or rewrite an existing reference/baseline/regression test.
"""

import numpy as np
import pytest

from scripts.audit.gt_dtlz8_analytic_candidate_issue119 import (
    _arm, _central, analyze, candidate, choose_complete_grids,
    continuous_real_bound,
)
from scripts.audit.gt_remaining_mop_matrix_issue119 import (
    dtlz8_same_t_pareto_witness,
)


@pytest.mark.parametrize("m", [3, 5, 10])
@pytest.mark.parametrize("budget", [128, 512])
def test_candidate_reconstructs_published_pareto_objectives(m, budget):
    first=candidate(m,budget)
    second=candidate(m,budget)
    objectives=first["objectives"]
    assert objectives.shape==(budget,m)
    assert first["decisions"].shape==(budget,first["mop"].N)
    assert np.array_equal(first["decisions"],second["decisions"])
    assert np.array_equal(objectives,second["objectives"])
    assert np.all((first["decisions"]>=0)&(first["decisions"]<=1))
    assert len(np.unique(objectives,axis=0))==budget
    assert first["replay_max_abs"] < 1e-12
    np.testing.assert_allclose(
        objectives, dtlz8_same_t_pareto_witness(objectives),
        rtol=0,atol=1e-12
    )


@pytest.mark.parametrize("m", [3, 5, 10])
def test_candidate_independent_probes_respect_real_grid_bound(m):
    sample=analyze(m,128,probe_count=128)
    assert sample["requested"]==sample["returned"]==sample["numerical_unique"]==128
    assert sample["independent_probe_global_max"] <= (
        sample["mathematical_real_cover_upper"]["union"]+1e-10
    )
    assert sample["ieee_roundoff_enclosure_proved"] is False
    assert sample["runtime_MOP_generator_modified"] is False


@pytest.mark.parametrize("m", [3, 5, 10])
def test_complete_grid_bound_improves_on_nested_refinement(m):
    coarse=continuous_real_bound(m,2,2,2)
    fine=continuous_real_bound(m,3,3,3)
    assert fine["center"] < coarse["center"]
    assert fine["arms"] < coarse["arms"]
    assert fine["union"] < coarse["union"]
    chosen,bound=choose_complete_grids(m,128)
    assert chosen[0]+(m-1)*chosen[1]*chosen[2]==128
    assert bound["union"] <= coarse["union"]+1e-14


def test_arm_and_central_join_at_the_published_constraint_boundary():
    m=5
    central=_central([1/3],m)
    for j in range(m-1):
        for s in (0.0,0.5,1.0):
            arm=_arm([1/3],[s],m,j)
            np.testing.assert_allclose(arm,central,rtol=0,atol=1e-15)


def test_small_budget_cannot_claim_complete_grid_guarantee():
    for m in (3,5,10):
        with pytest.raises(ValueError, match="Complete DTLZ8 component grids"):
            candidate(m,2+4*(m-1)-1)


@pytest.mark.parametrize("m", [3, 5, 10])
def test_analytical_candidate_not_limited_by_historical_csv_size(m):
    data = candidate(m, 4096)
    assert data["objectives"].shape == (4096, m)
    assert len(np.unique(data["objectives"],axis=0)) == 4096
    assert data["replay_max_abs"] < 1e-12
