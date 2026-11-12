# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The proposed DPF5 measure preserves decision-parameter branch mass."""
import numpy as np
import pytest
from moeabench.mops import DPF5

from calibration.scripts.study_dpf5_independent_pf_issue78 import (
    fresh_dpf5_pf,run_study,
)


def test_independent_parameter_sample_is_seeded_spherical_and_no_dedup():
    F,mask=fresh_dpf5_pf(4096,seed=123)
    G,second=fresh_dpf5_pf(4096,seed=123)
    H,third=fresh_dpf5_pf(4096,seed=124)
    assert np.array_equal(F,G)
    assert np.array_equal(mask,second)
    assert not np.array_equal(F,H)
    assert np.all(F>=0)
    assert np.allclose(np.linalg.norm(F,axis=1),1,atol=1e-12,rtol=0)
    assert len(np.unique(F,axis=0))==len(F)
    assert 0.64<np.mean(mask)<0.69
    # D branch exactly obeys its collapsed objective constraint.
    np.testing.assert_allclose(F[mask,0],F[mask,1],atol=0,rtol=0)
    assert not np.allclose(F[~mask,0],F[~mask,1])


def test_bad_reference_sample_parameters_are_rejected():
    for n in (0,1,63,200001):
        with pytest.raises(ValueError,match="bounded"):
            fresh_dpf5_pf(n,seed=123)
    with pytest.raises(ValueError,match="seed"):
        fresh_dpf5_pf(64,seed=-1)
    with pytest.raises(ValueError,match="ONLY"):
        fresh_dpf5_pf(64,seed=3,mop=DPF5(M=5,D=2))


def test_independent_reference_requires_archival_support_match_and_has_finite_residue():
    report=run_study(n_points=(256,),seeds=(123,124),ks=(40,))
    assert report["status"]=="NONCERTIFIED_DPF5_IID_DECISION_PARAMETER_PF_PROPOSAL"
    assert report["archived_GT_exact_unique_100_grid_support_reproduced"]
    assert report["archived_GT_points"]==3367
    assert report["analytical_branch_D_mass"]==pytest.approx(2/3)
    assert not report["area_uniformity_claimed"]
    assert not report["baseline_or_GT_modified"]
    assert not report["certified"]
    for entry in report["rows"]:
        assert entry["unique_objective_count"]==256
        assert entry["raw_dist_median"]>0
        assert entry["ideal_res"]["40"]>0
    with pytest.raises(ValueError,match="distinct"):
        run_study(n_points=(128,128),seeds=(1,),ks=(40,))
