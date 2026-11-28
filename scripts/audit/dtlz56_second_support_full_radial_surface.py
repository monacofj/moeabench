# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Second-support continuous full-radial SURFACE for original DTLZ5/6 M4K1.

The previously certified x1=0, x2=1, x3=0 second-support curve can
be enlarged to 0<=x1<=1/4 (DTLZ5) or 0<=x1<=3/16 (DTLZ6).
For every exact real radius in the complete original feasible range,
the independent p=2 supporting-objective theorem proves membership
uniformly from the SAME rational strict c² margin as the first-support
face. The j=2 supporting objective is distinct. The conservative
original spherical parameter Lipschitz bounds used by the first
surface remain valid for these boundary angles.

All synthetic-grid nodes have independently replayed exact global
Pareto proofs, including g=0 boundary points. Does not characterize
the whole positive-g angular domain or production GT coverage.
"""
from __future__ import annotations

from fractions import Fraction

from .dtlz56_full_radial_support_face import (
    certify_full_radial_support_face, replay_full_radial_support_face,
)
from .dtlz56_tight_sine_global_pf import (
    certify_tight_sine_global_pf, replay_tight_sine_global_pf,
)
from .dtlz56_dtlz6_radial_enclosure import (
    certify_dtlz6_radial_interval_pf, replay_dtlz6_radial_interval_pf,
)
from .dtlz56_exact_same_ray import _encode

SCHEMA="dtlz56_m4k1_second_support_full_radial_surface_v1"


def certify_second_support_full_radial_surface(
    *,problem,x1_grid=None,radial_grid=None,pi_terms=16,
):
    original=certify_full_radial_support_face(
        problem=problem,x1_grid=x1_grid,
        radial_grid=radial_grid,pi_terms=pi_terms)
    if not replay_full_radial_support_face(
        original,problem=problem,x1_grid=x1_grid,
        radial_grid=radial_grid,pi_terms=pi_terms):
        raise ArithmeticError("Independent first-face geometric certificate failed")

    def value(r):
        return Fraction(int(r["numerator_hex"],16),
                        int(r["denominator_hex"],16))
    nodes=[]
    for old in original["grid_decisions"]:
        source=old["decision"]
        x1=value(source[0])
        tail=value(source[3])
        g=value(old["radial_g"])
        positions=(x1,Fraction(1),Fraction(0))
        decision=(*positions,tail)
        if g==0:
            proof_basis="exact_global_unit_radius"
        elif problem=="DTLZ5":
            proof=certify_tight_sine_global_pf(
                positions,g=g,problem=problem,k=1,pi_terms=pi_terms)
            if (proof.conclusion!="rigorously_global_pf_member"
                or proof.supporting_objective!=2 or proof.sine_power!=2
                or not replay_tight_sine_global_pf(
                    positions,proof.to_dict(),g=g,problem=problem,
                    k=1,pi_terms=pi_terms)):
                raise AssertionError("Second-support DTLZ5 exact PF proof failed")
            proof_basis=proof.proof_id
        else:
            proof=certify_dtlz6_radial_interval_pf(
                positions,(tail,),k=1,root_bits=56,pi_terms=pi_terms)
            if (proof.conclusion!="rigorously_global_pf_member"
                or proof.supporting_objective!=2 or proof.sine_power!=2
                or not replay_dtlz6_radial_interval_pf(
                    positions,(tail,),proof.to_dict(),k=1,
                    root_bits=56,pi_terms=pi_terms)):
                raise AssertionError("Second-support DTLZ6 exact PF proof failed")
            proof_basis=proof.proof_id
        nodes.append({
            "decision":[_encode(x) for x in decision],
            "radial_g":old["radial_g"],
            "supporting_objective":2,
            "proof_basis":proof_basis,
        })
    return {
        "schema":SCHEMA,"problem":problem,"M":4,"K":1,
        "conclusion":"rigorously_global_pf_second_full_radial_surface",
        "second_support_objective":2,
        "sine_power":2,
        "angular_face_x2_x3":[_encode(Fraction(1)),_encode(Fraction(0))],
        "x1_domain":original["x1_domain"],
        "radial_domain":original["radial_domain"],
        "x1_grid":original["x1_grid"],
        "radial_grid":original["radial_grid"],
        "complete_admissible_original_g_range":True,
        "global_pf_uniform_margin":original["global_pf_strict_uniform_margin"],
        "continuous_to_grid_raw_l2_upper":
            original["continuous_to_exact_grid_raw_l2_upper"],
        "grid_decisions":nodes,
        "only_one_supported_angular_surface":True,
        "all_other_pf_angular_regions":"unverified",
        "actual_source_coverage":"unverified",
        "scientific_certification":"unverified",
    }


def replay_second_support_full_radial_surface(
    record,*,problem,x1_grid=None,radial_grid=None,pi_terms=16,
):
    try:
        proof=certify_second_support_full_radial_surface(
            problem=problem,x1_grid=x1_grid,
            radial_grid=radial_grid,pi_terms=pi_terms)
        return (record==proof and proof["conclusion"]==
                "rigorously_global_pf_second_full_radial_surface")
    except (ValueError,TypeError,KeyError,ArithmeticError,
            AssertionError,ZeroDivisionError,OverflowError):
        return False
