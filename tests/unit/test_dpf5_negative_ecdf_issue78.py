# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Negative-null DPF5 ECDF weights need not match ideal PF branch weights."""
import numpy as np
import pytest

from calibration.scripts.audit_dpf5_negative_ecdf_issue78 import (
    weighted_quantiles, frozen_dpf5_counts, study,
)


def test_weighted_empirical_quantile_honors_multiplicity():
    values=[1,2,4]
    weights=[1,3,1]
    assert weighted_quantiles(values,weights,[0,.5,.8,1]).tolist()==[1,2,2,4]
    assert weighted_quantiles(values,[1,1,1],[.5]).tolist()==[2]
    for invalid in ([1,-1,1],[0,0,0],[1,float("nan"),1]):
        with pytest.raises(ValueError,match="Invalid"):
            weighted_quantiles(values,invalid,[.5])
    with pytest.raises(ValueError,match="Invalid"):
        weighted_quantiles(values,weights,[-.1])


def test_frozen_gt_weights_are_exactly_reconstructed_without_guessing():
    import json
    from calibration.scripts.study_dpf5_independent_pf_issue78 import FROZEN,PACKAGE
    source=json.loads(FROZEN.read_text())
    gt=np.asarray(source["problems"]["DPF5"]["gt_reference"])
    with np.load(PACKAGE,allow_pickle=False) as pkg:
        ideal=np.asarray(pkg["ideal"])
        nadir=np.asarray(pkg["nadir"])
    raw,P,D=frozen_dpf5_counts(gt,ideal,nadir)
    assert len(gt)==3367
    assert np.sum(raw)==10000
    assert np.count_nonzero(P)==3300
    assert np.count_nonzero(D)==67
    assert np.sum(P)==3300
    assert np.sum(D)==6700
    assert np.all(raw==P+D)


def test_same_blurred_source_has_different_negative_null_branch_weights():
    r=study(K=100,reference_n=256,paired=False)
    assert r["status"]=="NONCERTIFIED_DPF5_NULL_ECDF_WEIGHTING_AUDIT"
    assert not r["scientific_measure_approved"]
    assert not r["baseline_modified"]
    assert r["paired_real_moea"]==[]
    assert r["frozen_GT_points"]==3367
    assert r["distinct_D_rows"]==67
    assert r["original_D_decision_rows"]==6700
    d=r["null_ecdfs"]["distinct_frozen_GT_rows"]
    w=r["null_ecdfs"]["original_decision_grid_multiplicity"]
    assert d["branch_D_anchor_probability"]==pytest.approx(67/3367)
    assert w["branch_D_anchor_probability"]==pytest.approx(.67)
    assert d["corrected_rand_ecdf_length"]==200
    assert w["corrected_rand_ecdf_length"]==200
    assert d["corrected_rand_ecdf_sha256"]!=w["corrected_rand_ecdf_sha256"]
    assert r["ideal_residue"]>0
