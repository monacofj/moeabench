# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cross-law real-front comparisons must not hide Q clipping as stability."""
import numpy as np
import pytest
from calibration.scripts.study_cross_sampler_moea_issue75 import (
    summarize_law_shift,
)


def _samples():
    return {
        "library_123":{"eps_distance_median":0.05},
        "library_124":{"eps_distance_median":0.06},
        "independent_123":{"eps_distance_median":0.07},
        "independent_124":{"eps_distance_median":0.08},
    }


def _trial(q, generation="-1"):
    return [{"seed":7,"fronts":{generation:{
        "q_by_analytical_seed":dict(zip(_samples(),q)),
        "raw_closeness_median":0.15,
        "archived_zero_residue_q":0.4,
        "front_sha256":"fixed_front_not_modified",
    }}}]


def test_cross_source_q_shift_uses_medians_not_extreme_seed_pair():
    r=summarize_law_shift(_trial([.6,.62,.72,.74]),_samples())
    assert r["non_clipped_fronts"]==1
    assert r["clipped_all_zero_fronts"]==0
    assert r["max_cross_law_q_median_shift_nonclipped"]==pytest.approx(.12)
    assert r["max_within_law_q_seed_span_nonclipped"]==pytest.approx(.02)
    assert r["cross_law_shift_exceeds_within_law_seed_span"] is True
    assert r["max_cross_law_epsilon_median_shift"]==pytest.approx(.02)
    assert r["not_certified"]


def test_clipped_zero_must_not_pretend_independent_methods_agree():
    r=summarize_law_shift(_trial([0,0,0,0]),_samples())
    assert r["clipped_all_zero_fronts"]==1
    assert r["non_clipped_fronts"]==0
    assert r["max_cross_law_q_median_shift_nonclipped"] is None
    assert r["max_within_law_q_seed_span_nonclipped"] is None
    assert r["cross_law_shift_exceeds_within_law_seed_span"] is None


def test_cross_source_evidence_requires_replicates_for_both_laws():
    with pytest.raises(ValueError,match="replicated"):
        summarize_law_shift(_trial([0,.1,.2,.3]),
                            {"library_123":{"eps_distance_median":.01},
                             "independent_123":{"eps_distance_median":.02}})
    broken=_trial([.6,.62,.72,.74])
    broken[0]["fronts"]["-1"]["q_by_analytical_seed"].pop("independent_124")
    with pytest.raises(ValueError,match="Missing cross-law"):
        summarize_law_shift(broken,_samples())
