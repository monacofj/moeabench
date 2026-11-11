# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scientific audit is conservative; ND/GT closeness never certifies area law."""
import copy
import numpy as np
import pytest

from calibration.scripts.audit_pf_sources_issue75 import (
    LAWS, _nd_evidence, classify, run_audit,
)


def _sample_nd(dominated=0):
    return {
        "points":100,
        "unique_points":100,
        "duplicate_points":0,
        "within_sample_dominated":dominated,
        "within_sample_nondominated_fraction":1-dominated/100,
    }


def test_all_fourteen_sources_have_explicit_nonuniform_provenance():
    assert set(LAWS)=={
        "DPF1","DPF2","DPF3","DPF4","DPF5",
        "DTLZ1","DTLZ2","DTLZ3","DTLZ4","DTLZ5",
        "DTLZ6","DTLZ7","DTLZ8","DTLZ9",
    }
    assert "meshgrid" in LAWS["DPF5"][0]
    assert "one_third" in LAWS["DPF5"][0]
    assert "NDS_filter" in LAWS["DPF3"][0]
    assert "static" in LAWS["DTLZ8"][0]
    assert "disconnected" in LAWS["DTLZ7"][1]


def test_no_probability_law_is_automatically_certified_by_geometric_overlap():
    for name in LAWS:
        label=classify(name,analytical_supported=True,
                       candidates=_sample_nd(),feasibility=None,branch=None)
        assert not label.startswith("CERTIFIED")
        assert label.startswith(("HOLD_","CONDITIONAL_"))
    assert classify("DPF3",analytical_supported=False,
                    candidates=_sample_nd(),feasibility=None,branch=None
                   )=="NO_COMPATIBLE_INDEPENDENT_ANALYTICAL_PF"
    assert classify("DTLZ8",analytical_supported=False,
                    candidates=_sample_nd(),feasibility=None,branch=None
                   )=="NO_COMPATIBLE_INDEPENDENT_ANALYTICAL_PF"


def test_dominated_or_infeasible_analytical_samples_must_be_held():
    assert classify("DTLZ1",analytical_supported=True,candidates=_sample_nd(1),
                    feasibility=None,branch=None
                   )=="HOLD_DOMINATED_ANALYTICAL_POINTS"
    assert classify("DTLZ9",analytical_supported=True,candidates=_sample_nd(),
                    feasibility={"infeasible":1},branch=None
                   )=="HOLD_INFEASIBLE_ANALYTICAL_POINTS"


def test_nd_oracle_detects_dominance_duplicates_and_validity():
    data=np.array([[0.1,0.1],[0.1,0.1],[0.2,0.2],[0.0,0.4]])
    result=_nd_evidence(data)
    assert result["duplicate_points"]==1
    assert result["unique_points"]==3
    assert result["within_sample_dominated"]==1
    assert result["within_sample_nondominated_fraction"]==pytest.approx(2/3)
    with pytest.raises(ValueError,match="Invalid"):
        _nd_evidence(np.array([[np.nan,0],[1.,2.]]))


def test_inventory_rejects_partial_archive_and_unbounded_sampling():
    with pytest.raises(ValueError,match="exactly"):
        run_audit({"problems":{"DTLZ1":{}}})
    with pytest.raises(ValueError,match="bounded"):
        run_audit({"problems":{name:{} for name in LAWS}},n_points=99999)
