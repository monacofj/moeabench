# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent regression for issue-75 reservoir-cardinality sensitivity."""

import numpy as np
import pytest

from calibration.scripts.study_reservoir_cardinality_issue75 import (
    compare_reservoir_factors,
)


def _gt():
    x=np.linspace(0,1,121)
    return np.column_stack((x,1-x,np.zeros_like(x)))


class KnownFront:
    def pf(self,n_points):
        x=np.linspace(0,1,n_points)
        # An offset from the GT makes nearest-GT distances unambiguous and
        # nonzero, while leaving the position-variable ordering fixed.
        return np.column_stack((x,1-x,0.002+0.002*x))


def test_cardinality_study_keeps_k_seed_and_final_n_fixed():
    report=compare_reservoir_factors(
        _gt(),KnownFront(),n_fresh=50,
        factors=(2,4,8),seeds=(7,8,9),k_values=(20,100),
    )
    assert report["n_fresh"] == 50
    assert set(report["factors"]) == {"2","4","8"}
    for factor,rec in report["factors"].items():
        assert rec["requested_pf_points"] == 121 + int(factor)*50
        assert rec["unique_fresh_candidates"] >= 50
        assert sum(rec["candidate_regions_f1"]) == rec["unique_fresh_candidates"]
        assert len(rec["replicates"]) == 3
        for trial in rec["replicates"]:
            assert sum(trial["regions_f1"]) == 50
            assert set(trial["epsilons"]) == {"20","100"}
            assert all(v>0 for v in trial["epsilons"].values())
    summary=report["all_factors_summary"]["100"]
    assert len(summary["factor_medians"]) == 3
    assert summary["factor_median_range"]["min"] > 0
    assert report == compare_reservoir_factors(
        _gt(),KnownFront(),n_fresh=50,
        factors=(2,4,8),seeds=(7,8,9),k_values=(20,100),
    )


def test_reservoir_study_rejects_nonindependent_candidate_pool():
    class RepeatsGT:
        def pf(self,n_points):
            return _gt().copy()

    with pytest.raises(ValueError,match="only .* unique fresh"):
        compare_reservoir_factors(_gt(),RepeatsGT(),n_fresh=50,
                                  factors=(2,),seeds=(7,))
