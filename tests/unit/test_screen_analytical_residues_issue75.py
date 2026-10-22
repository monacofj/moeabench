# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Screening does not assume every packaged MOP supports analytical residue."""

import numpy as np

from calibration.scripts import screen_analytical_residues as screen


def test_all_mop_screening_preserves_unsupported_with_reason(monkeypatch, tmp_path):
    gt = np.column_stack((np.linspace(0,1,41),1-np.linspace(0,1,41)))
    for name in ("Supported","Unsupported"):
        pkg = (tmp_path/"moeabench/diagnostics/resources/references"/name)
        pkg.mkdir(parents=True)
        np.savez(pkg/"calibration_package.npz",gt_norm=gt)
    source={"problems":{
        "Supported":{"gt_reference":gt.tolist()},
        "Unsupported":{"gt_reference":gt.tolist()},
    }}

    def synthetic_sampler(problem, archived, package):
        if problem == "Unsupported":
            return None
        return (archived[:-1] + archived[1:])/2.0

    monkeypatch.setattr(screen,"_analytical_ideal_from_package",synthetic_sampler)
    result=screen.inventory(source,k_values=(10,20),root=tmp_path)
    assert result["total"] == 2
    assert result["supported_count"] == 1
    assert result["unsupported_count"] == 1
    assert result["supported"]["Supported"]["k"]["10"]["ideal_res"] > 0
    assert result["supported"]["Supported"]["k"]["20"]["ideal_res"] > 0
    assert "No independently verified" in result["unsupported"]["Unsupported"]["reason"]
    assert result["status"] == "NONCERTIFIED_ANALYTICAL_COMPATIBILITY_SCREEN"


def test_gt_mismatch_never_receives_analytical_residue(tmp_path):
    gt=np.column_stack((np.linspace(0,1,41),1-np.linspace(0,1,41)))
    pkg=tmp_path/"moeabench/diagnostics/resources/references"/"Wrong"
    pkg.mkdir(parents=True)
    np.savez(pkg/"calibration_package.npz",gt_norm=gt+1)
    source={"problems":{"Wrong":{"gt_reference":gt.tolist()}}}
    result=screen.inventory(source,k_values=(10,),root=tmp_path)
    assert result["supported_count"] == 0
    assert "does not match" in result["unsupported"]["Wrong"]["reason"]


def test_unequal_sample_cardinality_accepted_only_with_compatible_tail(monkeypatch):
    """DPF5-like grids are allowed; DPF3-like severe tail gaps are not."""
    from calibration.scripts import generate_baselines as generator
    gt = np.column_stack((np.linspace(0,1,41),1-np.linspace(0,1,41)))
    pkg = {"ideal": np.array([0.0,0.0]), "nadir": np.array([1.0,1.0])}
    expected = (gt[:-1] + gt[1:])/2

    class GoodGrid:
        def pf(self,n_points=100):
            # Rounds requested cardinality down to a square just as
            # DPF5's Pareto-set grid does.
            x=np.linspace(0,1,int(np.sqrt(n_points))**2)
            return np.column_stack((x,1-x))

    import moeabench.mops as mops
    monkeypatch.setattr(mops,"ToyGrid",GoodGrid,raising=False)
    # The sampler extension is separately tested; this case isolates
    # geometry acceptance at variable cardinality.
    import moeabench.diagnostics.calibration as cal
    monkeypatch.setattr(cal,"independent_analytical_reference",
                        lambda mop, reference: expected)
    out=generator._analytical_ideal_from_package("ToyGrid",gt,pkg)
    np.testing.assert_allclose(out,expected)

    class BadTail(GoodGrid):
        def pf(self,n_points=100):
            front=super().pf(n_points)
            front[-4:,0]+=10.0
            return front

    monkeypatch.setattr(mops,"ToyGrid",BadTail,raising=False)
    assert generator._analytical_ideal_from_package("ToyGrid",gt,pkg) is None
