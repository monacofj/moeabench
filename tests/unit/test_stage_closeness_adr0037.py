# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Guard staged ADR 0037-only residue correction; never adopt surface-area law."""
import copy

import numpy as np
import pytest

from calibration.scripts.stage_closeness_adr0037 import (
    revise_closeness_record, stage, verify,
)


def _data():
    old = {
        "version": "0.14.0",
        "schema": "baseline-history",
        "problems": {
            "Toy": {
                "gt_reference": [[0.0, 1.0], [1.0, 0.0]],
                "20": {"closeness": {
                    "ideal_res": 0.0,
                    "rand_ecdf": [.05, .25, .70],
                    "rand50": .25,
                    "p95": .60,
                    "sigma": .4,
                }, "headway": {"method": "historic", "anchor": .9},
                    "cov": {"rand50": .5}},
            },
            "Unsupported": {"gt_reference": [[0., 1.], [1., 0.]],
                            "20": {"closeness": {"ideal_res": 0.0}}},
        },
    }
    new = copy.deepcopy(old)
    new["problems"]["Toy"]["20"]["closeness"] = revise_closeness_record(
        old["problems"]["Toy"]["20"]["closeness"], .1)
    report = {
        "status":"PROVISIONAL_ADR0037_INDEPENDENT_MOP_PF_NOT_CERTIFIED",
        "objective_area_uniform":False,
        "supported":{"Toy":{"K":{"20":{"epsilon":.1}}}},
        "unsupported":{"Unsupported":"No analytical oracle"},
        "K_records_changed":1,
    }
    return old, new, report


def test_closeness_only_correction_preserves_historical_records():
    old, new, report = _data()
    out = verify(old, new, report)
    assert out["status"] == "PASS_PROVISIONAL_ONLY"
    assert out["K_records_checked"] == 1
    assert old["version"] == new["version"] == "0.14.0"
    assert new["problems"]["Unsupported"] == old["problems"]["Unsupported"]
    assert new["problems"]["Toy"]["20"]["headway"] == old["problems"]["Toy"]["20"]["headway"]
    assert new["problems"]["Toy"]["20"]["closeness"]["rand_ecdf"] == pytest.approx(
        [0., .15, .60])


def test_independent_verifier_refuses_unrelated_edits_and_forged_epsilon():
    old, new, report = _data()
    wrong=copy.deepcopy(new)
    wrong["problems"]["Unsupported"]["20"]["closeness"]["ideal_res"] = .4
    with pytest.raises(ValueError, match="Unsupported MOP"):
        verify(old,wrong,report)

    wrong=copy.deepcopy(new)
    wrong["problems"]["Toy"]["20"]["headway"]["anchor"] = .7
    with pytest.raises(ValueError, match="Unrelated FAIR"):
        verify(old,wrong,report)

    wrong=copy.deepcopy(new)
    wrong["problems"]["Toy"]["20"]["closeness"]["rand_ecdf"][-1] += .01
    with pytest.raises(AssertionError):
        verify(old,wrong,report)

    wrong=copy.deepcopy(new)
    wrong["problems"]["Toy"]["20"]["closeness"]["reference_measure"]="uniform_area"
    with pytest.raises(ValueError,match="Unexpected Closeness"):
        verify(old,wrong,report)

    wrong=copy.deepcopy(new)
    wrong["version"]="0.16.1"
    with pytest.raises(ValueError,match="metadata"):
        verify(old,wrong,report)


def test_double_subtraction_unmeasured_zero_and_invalid_ecdf_rejected():
    source=_data()[0]["problems"]["Toy"]["20"]["closeness"]
    migrated=revise_closeness_record(source, .1)
    with pytest.raises(ValueError, match="double correction"):
        revise_closeness_record(migrated, .1)
    with pytest.raises(ValueError, match="positive"):
        revise_closeness_record(source, 0.0)
    with pytest.raises(ValueError, match="Invalid frozen Closeness ECDF"):
        revise_closeness_record({**source,"rand_ecdf":[.5,.2,.8]}, .1)
    assert source["ideal_res"] == 0.0


def test_absent_analytical_source_is_explicitly_unsupported(tmp_path):
    original=_data()[0]
    result, report=stage(original,["Unsupported"],packages=tmp_path)
    assert result==original
    assert "Unsupported" in report["unsupported"]
    assert report["K_records_changed"]==0
    assert verify(original,result,report)["K_records_checked"]==0
    with pytest.raises(ValueError,match="distinct"):
        stage(original,["Unsupported","Unsupported"],packages=tmp_path)
