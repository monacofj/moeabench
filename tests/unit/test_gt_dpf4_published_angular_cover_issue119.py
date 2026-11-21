# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Published DPF4 global Pareto geometry and angular covering (issue 119).

This is an independent research oracle for the *published sorted* eta
projection. It deliberately does not call or change the production
DPF4 evaluator for the expected published objective values.
"""
from itertools import product

import numpy as np
import pytest
from scipy.spatial import cKDTree

from moeabench.mops.DPF4 import DPF4
from scripts.audit.gt_remaining_mop_matrix_issue119 import (
    _published_dpf4_at_ps,
)


@pytest.mark.parametrize("m,d,k",[(5,2,33),(6,3,9),(7,4,5)])
def test_dpf4_published_angular_tensor_cover_respects_analytic_bound(m,d,k):
    mop=DPF4(M=m,D=d)
    q=d-1
    angles=np.asarray(
        list(product(np.linspace(0.0,1.0,k),repeat=q)),
        dtype=float,
    ).reshape(-1,q)
    ps=np.full((len(angles),mop.N),0.5)
    ps[:,:q]=angles
    published=_published_dpf4_at_ps(ps,m,d,mop._chaos_pool[:m-d])
    assert published.shape==(k**q,m)
    assert np.all(np.isfinite(published))
    assert np.all(published>=0)

    rng=np.random.default_rng(350+m*17+d)
    trials=np.full((2048,mop.N),0.5)
    trials[:,:q]=rng.random((2048,q))
    true_pf=_published_dpf4_at_ps(trials,m,d,mop._chaos_pool[:m-d])
    distances=cKDTree(published).query(true_pf,k=1)[0]
    mathematical_bound=np.sqrt(q)*np.pi/(2*(k-1))
    # A finite probe measures a lower bound on the worst gap; the
    # *mathematical* upper bound comes from the independent 2-Lipschitz
    # proof, not from these random probes.
    assert float(distances.max())<=mathematical_bound+1e-12


def test_dpf4_published_sorted_projection_is_different_from_unsorted_source():
    mop=DPF4(M=5,D=2)
    ps=mop.ps(128)
    published=_published_dpf4_at_ps(ps,mop.M,mop.D,mop._chaos_pool[:mop.M-mop.D])
    existing=mop.evaluation(ps)["F"]
    assert np.max(np.abs(published-existing))>0.5
    # No historical regression is altered, and no replacement GT
    # is recorded by this independent source comparison.
