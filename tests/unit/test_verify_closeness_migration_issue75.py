# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent verification that migration cannot silently alter unrelated FAIR data."""
import copy
import pytest
from calibration.scripts.verify_closeness_migration import verify


def _examples():
    record = {
        "closeness": {"rand_ecdf":[0.10,0.40,0.90], "rand50":0.4,
                      "p95":0.88, "ideal_res":0.0, "sigma":0.03},
        "headway":{"rand50":0.3}, "cov":{"rand50":0.2},
        "gap":{"rand50":0.3}, "reg":{"rand50":0.4}, "bal":{"rand50":0.5},
    }
    original={"version":"0.14.0","problems":{
        "DTLZ1":{"20":copy.deepcopy(record), "gt_reference":[[0.5,0.5,0.0]]},
        "DTLZ2":{"20":copy.deepcopy(record), "gt_reference":[[1.0,0.0,0.0]]},
    }}
    candidate=copy.deepcopy(original)
    new=candidate["problems"]["DTLZ1"]["20"]["closeness"]
    epsilon=0.05
    new["rand_ecdf"]=[0.05,0.35,0.85]
    new["rand50"]=0.35
    new["p95"]=0.83
    new["ideal_res"]=epsilon
    new["ideal_res_method"]="independent_analytical_pf"
    return original,candidate


def test_preserves_unrelated_metric_and_problem_values():
    a,b=_examples()
    result=verify(a,b,["DTLZ1"])
    assert result["k_counts"] == {"DTLZ1":1}
    assert result["untouched_problem_count"] == 1
    assert result["status"] == "PASS_PROVISIONAL_CLOSENESS_ONLY"


def test_rejects_corruption_of_unrelated_fair():
    a,b=_examples()
    b["problems"]["DTLZ1"]["20"]["cov"]["rand50"]=0.99
    with pytest.raises(ValueError,match="Unrelated FAIR"):
        verify(a,b,["DTLZ1"])


def test_rejects_corruption_of_untargeted_mop():
    a,b=_examples()
    b["problems"]["DTLZ2"]["20"]["gap"]["rand50"]=0.99
    with pytest.raises(ValueError,match="Untargeted problem"):
        verify(a,b,["DTLZ1"])


def test_rejects_corruption_of_corrected_ecdf():
    a,b=_examples()
    b["problems"]["DTLZ1"]["20"]["closeness"]["rand_ecdf"][0]=0.15
    with pytest.raises(AssertionError):
        verify(a,b,["DTLZ1"])


def test_versioned_area_measure_candidate_is_independently_verified():
    original,candidate=_examples()
    close=candidate["problems"]["DTLZ1"]["20"]["closeness"]
    close.update({
        "ideal_res_method":"uniform_normalized_objective_pf_area_v1",
        "reference_measure":"uniform_intrinsic_area_normalized_objective_pf",
        "ideal_res_sampler":"uniform_simplex_area_dirichlet_111",
        "ideal_res_sampling_seed":123,
        "ideal_res_sample_count":4096,
    })
    result=verify(original,candidate,["DTLZ1"])
    assert result["k_counts"]=={"DTLZ1":1}
    assert result["untouched_problem_count"]==1
    assert (result["k_records"]["DTLZ1"]["20"]["ideal_res_method"]
            =="uniform_normalized_objective_pf_area_v1")

    close["reference_measure"]="uniform_decision_variables"
    with pytest.raises(ValueError,match="Area measure provenance"):
        verify(original,candidate,["DTLZ1"])
    close["reference_measure"]="uniform_intrinsic_area_normalized_objective_pf"
    close["ideal_res_sampling_seed"]=-1
    with pytest.raises(ValueError,match="Area RNG provenance"):
        verify(original,candidate,["DTLZ1"])
    close["ideal_res_sampling_seed"]=123
    close["unexpected_metric"]="invented"
    with pytest.raises(ValueError,match="Unrecognized new Closeness"):
        verify(original,candidate,["DTLZ1"])


def test_area_record_clips_ecdf_and_rejects_unknown_methods():
    from calibration.scripts.migrate_closeness_residue import (
        residue_update_record, NORMALIZED_AREA_METHOD,
    )
    original={"rand_ecdf":[.1,.3,.9],"rand50":.3,"p95":.8,"ideal_res":0.0}
    provenance={"law":"uniform_simplex_area_dirichlet_111",
                "seed":17,"n_fresh":4096}
    migrated=residue_update_record(
        original,.2,method=NORMALIZED_AREA_METHOD,provenance=provenance)
    assert migrated["rand_ecdf"]==pytest.approx([0.,.1,.7])
    assert migrated["rand50"]==pytest.approx(.1)
    assert migrated["p95"]==pytest.approx(.6)
    assert migrated["ideal_res_sampling_seed"]==17
    assert migrated["ideal_res_sample_count"]==4096
    assert original["rand_ecdf"]==[.1,.3,.9]
    with pytest.raises(ValueError,match="provenance"):
        residue_update_record(original,.2,method=NORMALIZED_AREA_METHOD)
    with pytest.raises(ValueError,match="Unknown ideal residue"):
        residue_update_record(original,.2,method="undefined")
    with pytest.raises(ValueError,match="already been corrected"):
        residue_update_record(migrated,.2,method=NORMALIZED_AREA_METHOD,
                              provenance=provenance)
