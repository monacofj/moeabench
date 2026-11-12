# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""DPF5 experimental protocol: two explicit laws, no baseline publication."""
import numpy as np
import pytest

from calibration.scripts.validate_dpf5_closeness_protocol_issue81 import (
    frozen_reference, sample_independent_ideal, study
)


def test_frozen_GT_reconstructed_exactly_from_original_100_by_100_grid():
    frozen,gt,ideal,nadir,mop=frozen_reference()
    assert frozen["version"]=="0.14.0"
    assert gt.shape==(3367,3)
    assert np.all(nadir>ideal)
    raw=np.asarray(mop.evaluation(mop.ps(n_points=10000))["F"])
    rebuilt=np.unique((raw-ideal)/(nadir-ideal),axis=0)
    np.testing.assert_allclose(rebuilt,np.unique(gt,axis=0),atol=1e-12,rtol=0)


def test_ideal_sample_is_private_seeded_pushforward_without_deduplication():
    _,_,_,_,mop=frozen_reference()
    before=np.random.get_state()
    a,mask=sample_independent_ideal(mop,size=4096,seed=123)
    b,mask_b=sample_independent_ideal(mop,size=4096,seed=123)
    after=np.random.get_state()
    assert all(np.array_equal(x,y) if isinstance(x,np.ndarray) else x==y
               for x,y in zip(before,after))
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(mask,mask_b)
    assert a.shape==(4096,3)
    assert np.all(a>=0)
    assert np.allclose(np.linalg.norm(a,axis=1),1.,atol=1e-12,rtol=0)
    assert 0.64<float(np.mean(mask))<0.69
    assert len(np.unique(a,axis=0))==len(a)


def test_small_nonnormative_multiseed_study_stays_within_explicit_scope():
    j=study(ks=(100,),ideal_sizes=(256,),ideal_seeds=(123,124),
            fit_seeds=(1100,1101),blur_seeds=(1600,1601))
    assert j["status"]=="NONCERTIFIED_DPF5_TWO_REFERENCE_STABILITY_PROTOCOL"
    assert j["mop"]=="DPF5" and j["M"]==3 and j["D"]==2
    assert j["frozen_GT_count"]==3367
    assert j["archived_GT_exact_100_by_100_unique_PF"]
    assert j["ideal_branch_D_theoretical_mass"]==pytest.approx(2/3)
    assert j["failure_branch_D_anchor_mass"]==pytest.approx(67/3367)
    assert j["no_baseline_or_GT_or_audit_report_modified"]
    assert j["no_normative_release"]
    assert j["no_statistical_acceptance_threshold_invented"]
    assert j["sigma_target_normalized_distance"]==2.
    assert j["ecdf_storage_order_statistics"]==200
    assert set(j["results_by_K"])=={"100"}
    v=j["results_by_K"]["100"]
    assert v["sigma_fit_summary"]["n"]==2
    assert v["independent_null_median_summary"]["n"]==4
    assert v["ideal_epsilon_by_size"]["256"]["summary"]["n"]==2
    assert v["illustrative_epsilon_for_ecdf"]>0
    assert v["s_K"]>0
    for trial in v["trials"]:
        assert abs(trial["controller_median_u"]-2.)<.1
        assert trial["sigma"]>0
        assert len(trial["independent_null_draws"])==2
        assert all(x["strict_parent_dominance_fraction"]==1.
                   and len(x["candidate_ecdf_p10_p50_p90"])==3
                   and x["candidate_ecdf_sha256"]
                   for x in trial["independent_null_draws"])
    # Re-running with the exact seeds, GT and main algorithm is bit-for-bit
    # reproducible for all numeric estimates; no randomness is global.
    j2=study(ks=(100,),ideal_sizes=(256,),ideal_seeds=(123,124),
             fit_seeds=(1100,1101),blur_seeds=(1600,1601))
    assert j["results_by_K"]==j2["results_by_K"]


@pytest.mark.parametrize("kwargs",[
    {"ks":(40,40)},
    {"ideal_sizes":(0,)},
    {"ideal_seeds":(12,12)},
    {"fit_seeds":(100,) ,"blur_seeds":(100,)},
    {"ks":()},
    {"ks":(201,)},
])
def test_invalid_protocol_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        study(**kwargs)


def test_distinct_heldout_blur_rng_streams_are_unique_across_K_and_sigma():
    r=study(ks=(40,100),ideal_sizes=(256,),ideal_seeds=(123,),
            fit_seeds=(1100,1101),blur_seeds=(1600,1601),
            independent_noise=True)
    assert r["negative_noise_design"]=="distinct_noise_per_K_fit_and_blur"
    assert r["sigma_fit_seeds_shared_across_K"]
    seeds=[]
    for k,row in r["results_by_K"].items():
        for fit in row["trials"]:
            seeds.extend(x["blur_seed"] for x in fit["independent_null_draws"])
            assert all(x["blur_seed"]!=x["base_blur_seed"]
                       for x in fit["independent_null_draws"])
    assert len(seeds)==8
    assert len(set(seeds))==8
    assert r["no_baseline_or_GT_or_audit_report_modified"]
