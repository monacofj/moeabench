# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent conditional angular covering checks for published DPF5.

Do not confuse a real-arithmetic complete-grid covering theorem with an
IEEE float64-certified upper bound. Historical regression files unchanged.
"""

from itertools import product

import numpy as np
import pytest
from scipy.spatial import cKDTree

from moeabench.mops.DPF5 import DPF5


@pytest.mark.parametrize("m,d",[(3,2),(5,2),(6,3),(6,4)])
def test_both_dpf5_branches_retain_complete_two_angle_grid(m,d):
    mop=DPF5(M=m,D=d)
    ps=mop.ps(n_points=128)
    F=mop.evaluation(ps)["F"]
    assert len(ps)==128
    assert len(np.unique(F,axis=0))==128

    for label in ("P","D"):
        q = m-1 if label=="P" else d-1
        lo, hi = ((0.0,np.nextafter(1/3,0.0)) if label=="P"
                  else (1/3,1.0))
        angular=np.asarray(list(product((0.0,1.0),repeat=q)))
        angular[:,0]=lo+(hi-lo)*angular[:,0]
        x=np.full((len(angular),mop.N),0.5)
        x[:,:q]=angular
        x[:,m-1]=x[:,0]
        targets=mop.evaluation(x)["F"]

        selector=ps[:,0]<1/3 if label=="P" else ps[:,0]>=1/3
        samples=F[selector]
        assert len(samples)>0
        dist=cKDTree(samples).query(targets,k=1)[0]
        # Grid samples may have duplicate *mathematical* images, but the
        # geometric support must remain intact after deduplication.
        assert np.max(dist)<3e-14

        # Published hyperspherical objectives are 1-Lipschitz in branch
        # angular coordinates. An entire 2-point-per-axis tensor grid
        # gives a conservative conditional real-arithmetic upper bound.
        rng=np.random.default_rng(550+m*10+d+(0 if label=="P" else 100))
        probes=rng.uniform(size=(256,q))
        probes[:,0]=lo+(hi-lo)*probes[:,0]
        qp=np.full((len(probes),mop.N),0.5)
        qp[:,:q]=probes
        qp[:,m-1]=qp[:,0]
        probe_F=mop.evaluation(qp)["F"]
        bound=(np.pi/4)*np.sqrt((hi-lo)**2+(q-1))
        if label=="P":
            bound+=(1/3-hi)*np.pi/2
        observed=np.max(cKDTree(samples).query(probe_F,k=1)[0])
        assert observed<=bound+3e-13


@pytest.mark.parametrize("m,d",[(3,2),(5,2),(6,3),(6,4)])
def test_corrected_dpf5_has_deterministic_branch_cardinality(m,d):
    mop=DPF5(M=m,D=d)
    for n in (64,128):
        ps=mop.ps(n_points=n)
        replay=mop.ps(n_points=n)
        assert np.array_equal(ps,replay)
        F=mop.evaluation(ps)["F"]
        assert len(np.unique(F,axis=0))==n
        np.testing.assert_allclose(np.linalg.norm(F,axis=1),1,rtol=0,atol=3e-14)
        assert np.any(ps[:,0]<1/3)
        assert np.any(ps[:,0]>=1/3)
