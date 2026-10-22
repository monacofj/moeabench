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
