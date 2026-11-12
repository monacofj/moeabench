# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent geometry and source-row provenance tests for DPF5's frozen GT."""
import numpy as np

from moeabench.mops import DPF5
from calibration.scripts.audit_dpf5_frozen_gt_measure_issue78 import (
    classify_branch,
)


def test_dpf5_branch_detection_is_geometrically_grounded():
    mop=DPF5(D=2,M=3)
    X=mop.ps(n_points=1024)
    F=mop.evaluation(X)["F"]
    label=classify_branch(F)
    physical_D=X[:,0]>1/3
    # Every D objective is on the f1=f2 diagonal. A P meridian can also
    # intersect it and is explicitly ambiguous in the archive audit.
    assert np.all(label[physical_D])
    assert np.sum(label&~physical_D)>0
    assert np.mean(label)>=np.mean(physical_D)
    assert len(np.unique(F[physical_D],axis=0)) < np.sum(physical_D)/5


def test_recorded_branch_weights_not_equivalent_to_uniform_objectives():
    X=DPF5(D=2,M=3).ps(n_points=1024)
    F=DPF5(D=2,M=3).evaluation(X)["F"]
    raw_branch=X[:,0]>1/3
    mass=np.mean(raw_branch)
    P_u=np.unique(F[~raw_branch],axis=0)
    D_u=np.unique(F[raw_branch],axis=0)
    uniq_mass=len(D_u)/(len(P_u)+len(D_u))
    assert mass>.6 and uniq_mass<.1
    assert mass/uniq_mass>10
