# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Counterfactual laws compare exactly the same DPF5 analytical source/GT."""
import numpy as np
import pytest

from calibration.scripts.study_dpf5_branch_measure_issue78 import (
    weighted_median, compare_grid,
)


def test_weighted_median_rejects_invalid_and_matches_empirical_lower_median():
    assert weighted_median([3,1,2],[1,1,1])==2.0
    assert weighted_median([0,2,8],[1,3,1])==2.0
    assert weighted_median([0,2,8],[3,1,1])==0.0
    assert weighted_median([0,2,8],[.25,.5,.25])==2.0
    for values, weights in [
        ([],[]),([1,2],[1]),([1,np.nan],[1,1]),
        ([1,2],[0,0]),([1,2],[1,-1]),
    ]:
        with pytest.raises(ValueError,match="Invalid"):
            weighted_median(values,weights)


def test_dpf5_same_grid_changes_branch_measure_not_pf_or_gt():
    report=compare_grid(16,ks=(40,))
    assert report["status"]=="DPF5_MEASURE_SENSITIVITY_NOT_NORMATIVE"
    assert report["frozen_baseline_version"]=="0.14.0"
    assert report["raw_objective_count"]==256
    assert report["raw_branch_D_probability"]==pytest.approx(10/16)
    assert report["unique_P_count"]==96
    assert report["unique_D_count"]==10
    assert report["dedup_branch_D_probability"]==pytest.approx(10/106)
    assert report["illustrative_stratified_branch_D_probability"]==.5
    assert report["physical_PF_radial_max_error"]<1e-12
    k=report["K"]["40"]
    assert k["s_K"]>0
    assert len(k["ideal_res_by_measure"])==3
    assert all(v>=0 and np.isfinite(v) for v in k["ideal_res_by_measure"].values())
    assert k["max_minus_min_ideal_res"]>=0


def test_outside_grid_or_missing_k_refused():
    with pytest.raises(ValueError,match="Bounded"):
        compare_grid(7)
    with pytest.raises(ValueError,match="positive K"):
        compare_grid(16,ks=(-1,))


def test_dpf5_counterfactual_baselines_only_adjust_one_closeness_record():
    import copy
    from calibration.scripts.study_dpf5_paired_q_issue78 import candidate_for_law

    old={
        "version":"0.14.0",
        "problems":{
            "DPF5":{
                "gt_reference":[[0.0,1.0],[1.0,0.0]],
                "40":{
                    "closeness":{"ideal_res":0.0,
                                 "rand_ecdf":[.1,.2,.3],
                                 "rand50":.2},
                    "headway":{"method":"archived"},
                },
            },
            "DPF1":{"40":{"closeness":{"ideal_res":0.0}}},
        },
    }
    frozen=copy.deepcopy(old)
    name="uniform_decision_grid_pushforward"
    candidate=candidate_for_law(old,40,.15,name)
    assert old==frozen
    assert candidate["version"]==old["version"]
    assert candidate["problems"]["DPF1"]==old["problems"]["DPF1"]
    assert candidate["problems"]["DPF5"]["gt_reference"]==old["problems"]["DPF5"]["gt_reference"]
    assert candidate["problems"]["DPF5"]["40"]["headway"]==old["problems"]["DPF5"]["40"]["headway"]
    observed=candidate["problems"]["DPF5"]["40"]["closeness"]
    assert observed["rand_ecdf"]==pytest.approx([0.,.05,.15])
    assert observed["rand50"]==pytest.approx(.05)
    assert observed["ideal_res"]==pytest.approx(.15)
    assert observed["ideal_res_method"]=="noncertified_DPF5_"+name
    with pytest.raises(ValueError,match="Unsupported"):
        candidate_for_law(old,40,.15,"unknown")
