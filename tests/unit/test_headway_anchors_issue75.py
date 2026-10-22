# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Protect ADR 0049 anchors from obsolete one-population baseline generation."""

import numpy as np
import pytest

from calibration.scripts import validate_headway_anchors as headway_oracle
from calibration.scripts import generate_baselines as offline
from moeabench.diagnostics import calibration, fair, qscore


def _reference():
    x=np.linspace(0,1,41)
    return np.column_stack((x,1-x,np.zeros_like(x)))


def test_all_adr0049_analytic_controls_and_unavailable_initial_data():
    report=headway_oracle.validate_analytic_controls()
    assert report["perfect"]["headway"] == pytest.approx(0.0)
    assert report["perfect"]["q_headway"] == pytest.approx(1.0)
    assert report["improved"]["headway"] == pytest.approx(0.25)
    assert 0 < report["improved"]["q_headway"] < 1
    assert report["unchanged"]["headway"] == pytest.approx(1.0)
    assert report["unchanged"]["q_headway"] == pytest.approx(0.0)
    assert report["regressed"]["headway"] == pytest.approx(2.0)
    assert report["regressed"]["q_headway"] == pytest.approx(0.0)
    assert report["missing_P0"]["headway"] is None


def test_single_population_calibration_helpers_never_compute_headway():
    gt=_reference()
    pop=gt[::4]
    centroids=pop
    ref_hist=np.ones(len(pop),dtype=float)/len(pop)
    sidecar=calibration._calc_all(pop,gt,0.1,pop,centroids,ref_hist)
    result=offline.calculate_metrics(pop,gt,0.1,pop,centroids,ref_hist)
    assert set(sidecar) == {"cov","gap","reg","bal"}
    assert set(result) == set(sidecar)
    assert all(np.isfinite(v) for v in sidecar.values())


def test_new_sidecar_baseline_exposes_analytic_headway_anchors(monkeypatch):
    gt=_reference()
    monkeypatch.setattr(calibration,"N_SAMPLES_ECDF",3)
    monkeypatch.setattr(calibration,"N_IDEAL_DRAWS",2)
    records=calibration._generate_baselines("Toy",gt,[10])
    rec=records["10"]["headway"]
    assert rec == {
        "ideal":0.0, "no_progress":1.0,
        "method":"analytic_anchors_adr0049",
        "status":"no_empirical_longitudinal_baseline",
    }
    assert "rand_ecdf" not in rec
    assert "rand50" not in rec


def test_existing_headway_quality_mapping_is_analytic_not_empirical(monkeypatch):
    # It must not load the obsolete random/ideal Headway distributions.
    from moeabench.diagnostics import baselines
    def forbidden(*args,**kwargs):
        raise AssertionError("Q_HEADWAY improperly borrowed a frozen baseline")
    monkeypatch.setattr(baselines,"get_baseline_values",forbidden)
    monkeypatch.setattr(baselines,"get_baseline_ecdf",forbidden)
    for h in [0.0,0.01,0.2,0.5,1.0,2.0]:
        measured=qscore.q_headway(h).value
        assert measured == pytest.approx(headway_oracle.expected_q(h))
    assert np.isnan(qscore.q_headway(float("nan")).value)


def test_real_two_run_headway_scores_are_independently_reproducible():
    report=headway_oracle.validate_real_pairs((7,8),population=16,generations=8)
    assert len(report)==2
    assert report[0]["GT_sha256"] == report[1]["GT_sha256"]
    assert report[0]["P0_sha256"] != report[1]["P0_sha256"]
    for trial in report:
        assert 0 <= trial["q_headway"] <= 1
        assert trial["P0_sha256"] != trial["Pf_sha256"]
        assert trial["q_headway"] == pytest.approx(headway_oracle.expected_q(trial["headway"]))
