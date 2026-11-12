# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""No publication: current outward source vs frozen DPF5 half-blur null."""
import copy
import numpy as np

from calibration.scripts.study_dpf5_outward_blur_issue83 import (
    _in_memory_candidates, study,
)
from calibration.scripts.study_dpf5_independent_pf_issue78 import FROZEN


def test_new_outward_sigma_preserves_2s_k_and_original_anchor_law():
    frozen,candidates,info=_in_memory_candidates(K=100,reference_n=256,
                                                 reference_seed=11)
    assert info["status"]=="NONCERTIFIED_DPF5_CORRECTED_OUTWARD_BLUR_PAIRED_STUDY"
    assert info["GT_distinct_points"]==3367
    assert info["GT_distinct_P"]==3300
    assert info["GT_distinct_D"]==67
    assert info["GT_negative_null_D_anchor_fraction"]==67/3367
    assert info["GT_independent_ideal_D_theoretical_mass"]==2/3
    assert info["all_blurs_strictly_dominated_by_parent"]
    assert abs(info["sigma_controller_observed_median_u"]-2)<.1
    assert abs(info["independent_final_blur_median_u"]-2)<.2
    assert info["new_sigma"]>0
    assert info["independent_ideal_residue"]>0
    assert not info["null_weighting_revised"]
    assert not info["historical_GT_changed"]
    assert not info["baseline_files_changed"]
    assert not info["certified"]
    assert set(candidates)=={
        "historical_null_only_ideal_residue_fixed",
        "corrected_one_sided_outward_null",
    }
    k="100"
    old=frozen["problems"]["DPF5"][k]
    for name,candidate in candidates.items():
        assert candidate["version"]==frozen["version"]
        assert candidate["problems"]["DPF5"]["gt_reference"]==frozen["problems"]["DPF5"]["gt_reference"]
        assert candidate["problems"]["DPF1"]==frozen["problems"]["DPF1"]
        assert candidate["problems"]["DPF5"][k]["headway"]==old["headway"]
        for key in ("cov","gap","reg","bal"):
            assert candidate["problems"]["DPF5"][k][key]==old[key]
        assert candidate["problems"]["DPF5"][k]["closeness"]["ideal_res"]>0
    assert frozen["problems"]["DPF5"][k]["closeness"]["ideal_res"]==0.0
    assert len(candidates["corrected_one_sided_outward_null"]
               ["problems"]["DPF5"][k]["closeness"]["rand_ecdf"])==200


def test_unpaired_outward_study_does_not_run_optimizers_or_write_baselines():
    a=study(K=100,reference_n=256,reference_seed=11,paired=False)
    b=study(K=100,reference_n=256,reference_seed=11,paired=False)
    assert a["paired_real_moea"]==[]
    assert not a["certified"]
    assert not a["baseline_files_changed"]
    assert a["new_sigma"]==b["new_sigma"]
    assert a["independent_ideal_residue"]==b["independent_ideal_residue"]
    assert a["blur_objectives_sha256"]==b["blur_objectives_sha256"]
    assert a["new_outward_null_ecdf_sha256"]==b["new_outward_null_ecdf_sha256"]
