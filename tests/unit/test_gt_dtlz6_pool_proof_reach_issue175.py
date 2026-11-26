# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New bounded exact-proof and source-covering tests; no expensive PS in unit."""
import numpy as np
import pytest

from scripts.audit.dtlz6_pool_proof_reach import (
    classify_dtlz6_decision,_discrete_cover_distance,audit_source,
)


def test_unit_radius_original_dtlz6_is_proved():
    assert classify_dtlz6_decision((.25,.5,.5,0.))[0]=="proved_pf"


def test_exact_irrational_g_uses_conservative_support_theorem():
    result=classify_dtlz6_decision((0.,1.,1.,.5))
    assert result[0] in ("proved_pf","unverified")
    assert result[0] != "dominated"


@pytest.mark.parametrize("bad", [
    (0.,.5,.5), (0.,.5,.5,0.,0.),
    (0.,.5,.5,1.1),(0.,float("nan"),.5,0.),
])
def test_invalid_decisions_rejected(bad):
    with pytest.raises(ValueError):
        classify_dtlz6_decision(bad)


def test_discrete_cover_radius_is_a_source_observation_not_pareto_proof():
    pool=np.array([(0.,0.),(1.,0.),(0.,1.)])
    assert _discrete_cover_distance(pool,pool)==0.
    assert _discrete_cover_distance(pool,pool[:1])>0.
    assert _discrete_cover_distance(pool,np.empty((0,2))) is None


def test_invalid_sample_count_rejected_without_production_call(monkeypatch):
    from moeabench.mops import DTLZ6
    def forbidden(*args,**kwargs):
        raise AssertionError("No production sampling in unit tests")
    monkeypatch.setattr(DTLZ6,"ps",forbidden)
    with pytest.raises(ValueError):
        audit_source(n_points=8)
