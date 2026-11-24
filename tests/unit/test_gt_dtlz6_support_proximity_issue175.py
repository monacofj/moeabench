# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""New independent exact distance-to-certified-original-DTLZ6-PF tests."""
from copy import deepcopy
from fractions import Fraction as Q
from math import cos, pi, sin

import numpy as np
import pytest

from scripts.audit.dtlz56_dtlz6_support_proximity import (
    certify_dtlz6_support_proximity,
    replay_dtlz6_support_proximity,
)
from scripts.audit.dtlz56_dtlz6_radial_enclosure import (
    replay_dtlz6_radial_interval_pf,
)


def _fraction(record):
    return Q(
        int(record["numerator_hex"], 16),
        int(record["denominator_hex"], 16),
    )


def _independent_original(u, g):
    """Independent published M-objective hyperspherical formula."""
    m=len(u)+1
    angles=[pi*float(u[0])/2]+[
        pi*(1+2*g*float(v))/(4*(1+g)) for v in u[1:]
    ]
    out=[]
    for j in range(m):
        value=1+g
        for i in range(m-j-1):
            value*=cos(angles[i])
        if j:
            value*=sin(angles[m-j-1])
        out.append(value)
    return np.asarray(out)


@pytest.mark.parametrize("m", [4,5,7])
@pytest.mark.parametrize("first", [Q(0),Q(1,16),Q(1,4)])
def test_near_boundary_point_has_exact_rational_upper_distance(m,first):
    near_one=Q.from_float(float(np.nextafter(1.,0.)))
    u=[first,near_one,Q(0)]+[Q(1,3)]*(m-4)
    tail=[Q.from_float(float(0.2**10))]
    result=certify_dtlz6_support_proximity(u,tail,k=1)
    assert result.conclusion=="rigorously_within_certified_pf_distance"
    assert result.supporting_objective==m-2
    snapped=[_fraction(x) for x in result.certified_positions]
    assert snapped[:3]==[first,Q(1),Q(0)]
    assert snapped[3:]==u[3:]
    bound=_fraction(result.certified_l2_distance_upper)
    assert Q(0)<bound<Q(1,10**13)
    assert replay_dtlz6_support_proximity(u,tail,result.to_dict(),k=1)
    assert replay_dtlz6_radial_interval_pf(
        snapped,tail,result.supporting_pf_certificate,k=1,
    )
    # Independent numerical spot-check, NOT the proof's basis.
    g=float(tail[0])**0.1
    distance=np.linalg.norm(_independent_original(u,g)-
                            _independent_original(snapped,g))
    assert distance<=float(bound)+1e-14
    assert result.to_dict()["scientific_certification"]=="unverified"


def test_zero_bound_for_exact_known_pf_is_valid():
    u=[Q(1,16),Q(1),Q(0)]
    tail=[Q(1,2)]
    result=certify_dtlz6_support_proximity(u,tail,k=1)
    assert result.conclusion=="rigorously_within_certified_pf_distance"
    assert _fraction(result.certified_l2_distance_upper)==0


def test_distance_is_upper_bound_not_membership_or_nearest_front_proof():
    # A near-PF candidate is *not* classified as globally Pareto optimal.
    u=[Q(1,4),Q(1)-Q(1,2**52),Q(0)]
    tail=[Q.from_float(float(0.2**10))]
    result=certify_dtlz6_support_proximity(u,tail,k=1)
    report=result.to_dict()
    assert report["conclusion"]=="rigorously_within_certified_pf_distance"
    assert report["scientific_certification"]=="unverified"
    assert "globally" not in report["conclusion"]
    assert _fraction(report["certified_l2_distance_upper"])>0


def test_near_support_but_unsafe_first_angle_abstains():
    u=[Q(1),Q(1)-Q(1,2**52),Q(0)]
    result=certify_dtlz6_support_proximity(u,[Q(1,2)],k=1)
    assert result.conclusion=="unverified"
    assert result.supporting_pf_certificate is None
    assert not replay_dtlz6_support_proximity(
        u,[Q(1,2)],result,k=1,
    )


def test_tampering_bound_or_snapped_pf_proof_is_rejected():
    u=[Q(1,4),Q(1)-Q(1,2**52),Q(0)]
    tail=[Q.from_float(float(0.2**10))]
    record=certify_dtlz6_support_proximity(u,tail,k=1).to_dict()
    assert replay_dtlz6_support_proximity(u,tail,record,k=1)
    for property_path in ("certified_l2_distance_upper",
                          "supporting_pf_certificate"):
        forged=deepcopy(record)
        if property_path=="certified_l2_distance_upper":
            forged[property_path]["numerator_hex"]="0x0"
        else:
            forged[property_path]["g_upper"]["numerator_hex"]="0x0"
        assert not replay_dtlz6_support_proximity(u,tail,forged,k=1)


@pytest.mark.parametrize("u,tail,k", [
    ([Q(0),Q(1)],[Q(1,2)],1),
    ([Q(-1),Q(1),Q(0)],[Q(1,2)],1),
    ([Q(0),Q(1),Q(0)],[Q(-1,2)],1),
    ([Q(0),Q(1),Q(0)],[Q(1,2),Q(1,2)],1),
    ([Q(0),Q(1),Q(0)],[Q(1,2)],0),
])
def test_invalid_source_decisions_rejected(u,tail,k):
    with pytest.raises(ValueError):
        certify_dtlz6_support_proximity(u,tail,k=k)
