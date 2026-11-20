# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""New DPF3 D=2 angular coverage theorem corroboration tests."""
import numpy as np
import pytest
from scripts.audit.gt_dpf3_d2_angular_cover_issue119 import published_curve,study


@pytest.mark.parametrize("m",[3,5,10])
@pytest.mark.parametrize("n",[32,128])
def test_published_d2_angular_curve_has_expected_covering_bound(m,n):
    r=study(m,n,513)
    assert r["published_sample_unique_exact"]==n
    assert 0<=r["measured_probe_max"]<=r["certified_real_arithmetic_upper_bound"]+2e-13
    assert r["real_arithmetic_coverage_proved"]
    assert not r["floating_point_coverage_proved"]
    assert not r["applies_to_general_D"]


def test_curve_is_globally_one_lipschitz_for_sampled_angles():
    eta=np.sort(np.asarray([0.1,0.5,0.9]))
    angles=np.linspace(0,np.pi/2,97)
    points=published_curve(angles,eta)
    distances=np.linalg.norm(np.diff(points,axis=0),axis=1)
    assert np.all(distances<=np.diff(angles)+1e-13)


def test_incorrect_threshold_order_is_not_silently_accepted():
    with pytest.raises(ValueError):
        published_curve(np.asarray([0.1,0.2]),np.asarray([0.8,0.4]))
