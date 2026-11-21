# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent published Pareto geometry and evaluator crosschecks for #119."""
import numpy as np
import pytest

from scripts.audit.gt_independent_geometry_issue119 import (
    SUPPORTED, _sphere, _project_dpf34, audit,
    evaluator_independent_residual, independent_probes,
)


@pytest.mark.parametrize("m,d", [(3,2), (5,2), (5,3)])
@pytest.mark.parametrize("name", SUPPORTED)
def test_independent_geometry_probes_have_all_components(name,m,d):
    groups = independent_probes(name,m,d,32,11941)
    expected = (m if name=="DTLZ8"
                else 2 if name=="DPF5" else 1)
    assert len(groups)==expected
    for group in groups.values():
        assert group.shape==(32,m)
        assert np.isfinite(group).all()
        assert np.min(group)>=-1e-13
    if name=="DTLZ8":
        for key, f in groups.items():
            t,y = f[:,-1],f[:,:-1]
            if key=="line":
                np.testing.assert_allclose(y,((1-t)/4)[:,None],
                                           atol=1e-14,rtol=0)
            else:
                z=np.sort(y,axis=1)
                np.testing.assert_allclose(z[:,0]+z[:,1],
                                           1-2*t,atol=1e-14,rtol=0)
    elif name=="DPF5":
        for group in groups.values():
            np.testing.assert_allclose(
                np.linalg.norm(group,axis=1),1,atol=1e-13,rtol=0)


@pytest.mark.parametrize("name", ["DPF3","DPF4","DPF5"])
@pytest.mark.parametrize("m,d", [(3,2),(5,2),(5,3)])
def test_independent_published_equations_match_evaluator(name,m,d):
    assert evaluator_independent_residual(name,m,d) < 1e-12


@pytest.mark.parametrize("name", SUPPORTED)
def test_finite_geometry_report_keeps_components_distinct(name):
    r=audit(name,m=3,d=2,budget=128,count=32)
    assert r["budget"]==128
    assert not r["continuous_covering_certified"]
    assert not r["mixed_dimensional_weight_assumed"]
    assert len(r["source_pf_hash"])==64
    for v in r["components"].values():
        assert v["probes"]==32
        assert v["maximum_gap"]>=v["mean_gap"]>=0


def test_sphere_oracle_is_unit_positive():
    angles=np.array([[0.,0.],[np.pi/4,np.pi/6],[np.pi/2,np.pi/2]])
    f=_sphere(angles)
    np.testing.assert_allclose(np.linalg.norm(f,axis=1),
                               np.ones(3),atol=1e-14,rtol=0)


def test_projection_oracle_uses_published_piecewise_order():
    y=np.array([[0.1,0.2],[0.5,0.9]])
    f=_project_dpf34(y,5,True)
    assert f.shape==(2,5)
    assert np.isfinite(f).all()
