# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New unit tests: exact proof-reach classes and grouped observations."""
from collections import Counter

import pytest

from scripts.audit.dtlz5_pool_proof_reach import (
    classify_original_decision, evaluate_proof_reach,
)
from scripts.audit.dtlz5_row10_exact_dominance import SOURCE_ROW_10

PF_FACE = (0.20, 1.0, 1.0, 1.0)
UNKNOWN = (0.20, 0.99999999, 1.0, 0.68)
ANCHOR = (0.20, 0.5, 0.5, 0.5)


def test_known_exact_membership_negative_and_abstention():
    assert classify_original_decision(PF_FACE)[0] == "proved_pf"
    assert classify_original_decision(ANCHOR)[0] == "proved_pf"
    assert classify_original_decision(SOURCE_ROW_10) == (
        "dominated","exact_off_ray_source_157")
    assert classify_original_decision(UNKNOWN)[0] in ("dominated","unverified")


@pytest.mark.parametrize("bad",[
    (0, 0, 0), (0, 0, 0, 0, 0),
    (0, 0, -0.1, 0), (0, 0, 0, float("nan")),
])
def test_invalid_sources_fail_closed(bad):
    with pytest.raises(ValueError):
        classify_original_decision(bad)


def test_all_source_rows_are_counted_without_gaining_certification():
    rows = [PF_FACE]*100 + [SOURCE_ROW_10]*100 + [ANCHOR]*248
    report = {
        "candidate_pool_decision_hex":[
            [float(v).hex() for v in x] for x in rows
        ],
        "candidate_pool_source_families":(
            ["sobol"]*256 + ["gaussian"]*128 + ["gamma"]*64
        ),
        "selected_rows":[
            {
                "selected_index":i,
                "decision_hex":[float(v).hex() for v in rows[i]],
                "trace":{"scalarizer_candidate_index":i},
            }
            for i in range(16)
        ],
    }
    out = evaluate_proof_reach(report)
    assert out["source_candidate_count"] == 448
    assert out["pool_counts"] == {
        "proved_pf":348,"dominated":100,"unverified":0,
    }
    assert sum(out["selected_counts"].values()) == 16
    for family, count in (("sobol",256),("gaussian",128),("gamma",64)):
        assert sum(out["by_source_family"][family].values()) == count
    assert sum(sum(v.values()) for v in out["by_radial_stratum"].values()) == 448
    assert out["scientific_certification"] == "unverified"
    assert out["finite_gt_coverage"] == "unverified"


def test_reject_incomplete_or_forged_pool():
    with pytest.raises(ValueError):
        evaluate_proof_reach({
            "candidate_pool_decision_hex":[],
            "candidate_pool_source_families":[],
            "selected_rows":[],
        })
