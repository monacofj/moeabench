# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New final-selected paired GT audit tests; no historical regressions edited."""
import json
import numpy as np
import pytest

from scripts.audit import dtlz56_final_selected_small_gt as audit
from scripts.audit.dtlz56_paired_gt_evidence import _original_objectives


def _synthetic_original_decisions(problem):
    x=np.zeros((16,4),dtype=float)
    x[:,0]=np.linspace(0,1,16)
    x[:,1:3]=.5
    x[:,3]=.5 if problem=="DTLZ5" else 0.
    # Two explicit nonzero-distance cases.
    x[-2:,3]=1.
    x[-2,1]=1.
    x[-2,2]=0.
    x[-1,1]=1.
    x[-1,2]=1.
    return x


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_readonly_final_selected_analysis_reports_geometry_and_limits(problem):
    x=_synthetic_original_decisions(problem)
    y,g=_original_objectives(x,problem=problem,m=4,k=1)
    result=audit.analyze_final_selected(
        x,y,problem=problem,m=4,k=1,
    )
    assert result["n_points"]==16
    assert result["source"]=="unchanged_production_mop.ps_final_maximin_selection"
    assert result["radial"]["exact_numeric_zero_count"]==14
    assert result["radial"]["positive_count"]==2
    assert result["radial"]["positive_row_indices"]==[14,15]
    assert result["radial"]["maximum_positive_g"]==pytest.approx(
        .25 if problem=="DTLZ5" else 1.)
    assert len(result["radial"]["quantiles"])==5
    assert len(result["geometry_observational"]["objective_component_span"])==4
    assert np.isfinite(result["geometry_observational"][
        "objective_range_normalized_nearest_neighbor_median"])
    assert result["geometry_observational"][
        "max_original_spherical_norm_residual"]<1e-12
    assert result["exact_rational_g_decision_image_ledger"]["n_points"]==16
    assert result["whole_gt_global_membership"]=="unverified"
    assert result["continuous_front_coverage"]=="unverified"
    assert result["indicator_fitness"]=="unverified"
    assert result["scientific_certification"]=="unverified"
    json.dumps(result,allow_nan=False)


def test_independent_alignment_mismatch_rejects_result():
    x=_synthetic_original_decisions("DTLZ5")
    y,_=_original_objectives(x,problem="DTLZ5",m=4,k=1)
    y[3,0]+=0.01
    with pytest.raises(ValueError,match="alignment"):
        audit.analyze_final_selected(
            x,y,problem="DTLZ5",m=4,k=1,
        )


@pytest.mark.parametrize("problem",["DTLZ5","DTLZ6"])
def test_paired_capture_calls_public_ps_once_and_replays_original_rows(
    problem,monkeypatch,tmp_path,
):
    from moeabench.mops import DTLZ5,DTLZ6
    cls=DTLZ5 if problem=="DTLZ5" else DTLZ6
    calls=[]
    def mocked_ps(self,*,n_points):
        calls.append(n_points)
        return _synthetic_original_decisions(problem)
    monkeypatch.setattr(cls,"ps",mocked_ps)
    bundle=tmp_path/"paired.npz"
    output=tmp_path/"audit.json"
    report=audit.capture_and_audit(
        problem=problem,m=4,k=1,n_points=16,
        bundle=bundle,report=output,
    )
    assert calls==[16]
    assert bundle.exists() and output.exists()
    stored=json.loads(output.read_text())
    assert stored["source_expected_total_scalarizations"]==448
    assert stored["source_ray_pool_size"]==256
    assert stored["paired_artifact_sha256"]==report["paired_artifact_sha256"]
    assert stored["scientific_certification"]=="unverified"
    assert stored["radial"]["positive_count"]==2
    with pytest.raises(FileExistsError):
        audit.capture_and_audit(
            problem=problem,m=4,k=1,n_points=16,
            bundle=bundle,report=output,
        )
    assert calls==[16]


@pytest.mark.parametrize("bad",[
    {"problem":"DTLZ1","m":4,"k":1,"n_points":16},
    {"problem":"DTLZ5","m":3,"k":1,"n_points":16},
    {"problem":"DTLZ5","m":4,"k":2,"n_points":16},
    {"problem":"DTLZ5","m":4,"k":1,"n_points":8},
    {"problem":"DTLZ5","m":4,"k":1,"n_points":True},
])
def test_invalid_configs_rejected_before_expensive_ps(tmp_path,bad):
    with pytest.raises(ValueError):
        audit.capture_and_audit(
            **bad,bundle=tmp_path/"p.npz",
            report=tmp_path/"r.json",
        )


def test_proven_dominated_source_decision_blocks_experimental_reference():
    # Make both inner angles strictly interior at positive original g,
    # triggering the pre-existing *exact rational* same-ray witness.
    x=_synthetic_original_decisions("DTLZ5")
    x[-1,1]=.99
    x[-1,2]=.999
    x[-1,3]=.7
    y,_=_original_objectives(x,problem="DTLZ5",m=4,k=1)
    result=audit.analyze_final_selected(
        x,y,problem="DTLZ5",m=4,k=1,
    )
    assert result["proven_dominated_decision_row_indices"]==[15]
    assert result["experimental_reference_gate"]=="blocked_proven_dominated_decision"
    assert result["scientific_certification"]=="unverified"


def test_no_known_dominator_never_implies_certification():
    x=_synthetic_original_decisions("DTLZ5")
    y,_=_original_objectives(x,problem="DTLZ5",m=4,k=1)
    result=audit.analyze_final_selected(
        x,y,problem="DTLZ5",m=4,k=1,
    )
    assert result["proven_dominated_decision_row_indices"]==[]
    assert result["experimental_reference_gate"]=="unverified_no_known_dominator"
    assert result["continuous_front_coverage"]=="unverified"
