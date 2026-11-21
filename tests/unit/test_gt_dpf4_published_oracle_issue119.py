# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Positive mathematical properties of the independent published DPF4 oracle.

The implementation under review is *not* required to reproduce a known
defect; diagnostics report any discrepancy without pinning it in tests.
"""
import numpy as np
import pytest

from scripts.audit.gt_dpf4_published_equations_issue119 import (
    _nonnegative_unit_hypersphere, audit, published_dpf4,
)
from moeabench.mops.DPF4 import DPF4


@pytest.mark.parametrize("m,d",[(3,2),(5,2),(6,3),(7,4)])
def test_independent_published_dpf4_oracle_is_pareto_shaped(m,d):
    mop=DPF4(M=m,D=d)
    rng=np.random.default_rng(11942)
    x=rng.uniform(0.1,0.9,size=(128,mop.N))
    x[:,d-1:]=0.5
    f=published_dpf4(x,m,d,mop._chaos_pool)
    assert f.shape==(128,m)
    assert np.isfinite(f).all()
    assert np.all(f>=0)
    angular=_nonnegative_unit_hypersphere(x[:,:d-1]*np.pi/2)
    np.testing.assert_allclose(np.linalg.norm(angular,axis=1),1,rtol=0,atol=2e-14)
    eta=np.sort(mop._chaos_pool[:m-d])
    raw=np.sqrt(np.maximum(f[:,d-1:],0))
    essential_last=1-angular[:,-1]
    np.testing.assert_allclose(raw.sum(axis=1),
                               essential_last+eta.sum(),rtol=0,atol=2e-13)


@pytest.mark.parametrize("m,d",[(3,2),(5,2),(6,3),(7,4)])
def test_dpf4_discrepancy_audit_is_reproducible_not_a_negative_regression(m,d):
    a=audit(m,d,sample_count=80)
    b=audit(m,d,sample_count=80)
    assert a==b
    assert a["cases"]==80
    assert a["max_objective_difference_to_independent_published"]>=0
    assert a["source_projection_telescope_error"]>=0
    assert a["no_production_changes"]
    if d==2:
        probe=a["small_angle_numerical_probe"]
        assert probe["stable_published_first_objective"]>0
        assert probe["current_first_objective"]>=0


def test_dpf4_published_thresholds_require_ordering_for_multiple_projections():
    mop=DPF4(M=5,D=2)
    eta=np.sort(mop._chaos_pool[:3])
    assert np.all(np.diff(eta)>0)
    expected=published_dpf4(mop.ps(n_points=128),5,2,mop._chaos_pool)
    assert expected.shape==(128,5)
