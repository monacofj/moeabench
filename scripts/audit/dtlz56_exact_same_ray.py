# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Constructive exact-real same-ray dominance for original DTLZ5/6.

Input is an EXACT rational declaration of a feasible original position
u=(x_1,...,x_{M-1}) and radial distance g. Finite Python floats, if passed,
are interpreted by their exact binary-rational values. No production
evaluator/sampler, transcendental numerical evaluation, or optimization.

If g>g_min, a strictly smaller rational g' is constructed along precisely
the same angle ray. The exact objective inequality follows symbolically:
F(g',u') = ((1+g')/(1+g)) F(g,u), with common positive scale < 1.
All numerical / environmental GT certification gates remain separate.
"""
from dataclasses import dataclass
from fractions import Fraction
from .dtlz56_rational_interval import _as_exact_fraction


def _encode(x: Fraction) -> dict:
    return {"numerator_hex": hex(x.numerator),
            "denominator_hex": hex(x.denominator)}


def _decode(record: dict) -> Fraction:
    return Fraction(int(record["numerator_hex"], 16),
                    int(record["denominator_hex"], 16))


@dataclass(frozen=True)
class SameRayWitness:
    conclusion: str
    original_g: dict
    minimum_ray_g: dict
    witness_g: dict | None
    witness_positions: tuple[dict, ...] | None
    objective_scale: dict | None
    basis: str = (
        "Exact rational inverse angular map and same-ray radial contraction "
        "for original DTLZ5/6; strict norm improvement is symbolic"
    )

    def to_dict(self) -> dict:
        return {
            "conclusion": self.conclusion,
            "original_g": self.original_g,
            "minimum_ray_g": self.minimum_ray_g,
            "witness_g": self.witness_g,
            "witness_positions": (
                list(self.witness_positions)
                if self.witness_positions is not None else None
            ),
            "objective_scale": self.objective_scale,
            "basis": self.basis,
            "scientific_certification": "unverified",
        }


def construct_same_ray_witness(positions, *, g, problem, k):
    """Construct exact g'<g dominator iff current ray admits lower g.

    The result applies to an *originally feasible* decision represented
    by the supplied rational positions and g. A declared pair (u,g)
    is checked against x bounds and the published achievable g range.
    Original decision feasibility beyond these declarations must be
    demonstrated separately when promoting an external GT row.
    """
    if problem not in ("DTLZ5", "DTLZ6") or type(k) is not int or k < 1:
        raise ValueError("Expected original DTLZ5/6 and integer K>=1")
    try:
        u = [_as_exact_fraction(x, name="position") for x in positions]
    except TypeError as exc:
        raise ValueError("Expected position vector") from exc
    if len(u) < 1 or any(x < 0 or x > 1 for x in u):
        raise ValueError("Expected M>=2, position coordinates in [0,1]")
    gy = _as_exact_fraction(g, name="g")
    gmax = Fraction(k, 4) if problem == "DTLZ5" else Fraction(k)
    if not 0 <= gy <= gmax:
        raise ValueError("Radial g outside original feasible range")

    # theta_i = pi/4 * (1 + [g/(1+g)](2u_i-1)), i >= 2.
    # Let q = max_i |[g/(1+g)](2u_i-1)|. Those angles are
    # achievable iff g'/(1+g') >= q.
    deviations = [(gy * (2*v-1))/(1+gy) for v in u[1:]]
    q = max((abs(v) for v in deviations), default=Fraction(0))
    gmin = q/(1-q)
    if gy == 0 or gmin >= gy:
        return SameRayWitness(
            "no_strict_same_ray_reduction", _encode(gy),
            _encode(gmin), None, None, None,
        )
    gp = (gy+gmin)/2
    assert gmin <= gp < gy and gp > 0
    uv = [u[0]]
    uv.extend((Fraction(1,2) + d*(1+gp)/(2*gp)) for d in deviations)
    assert all(Fraction(0) <= v <= Fraction(1) for v in uv)
    scale = (1+gp)/(1+gy)
    assert 0 < scale < 1
    return SameRayWitness(
        "exact_same_ray_dominator", _encode(gy), _encode(gmin),
        _encode(gp), tuple(_encode(v) for v in uv), _encode(scale),
    )


def verify_same_ray_witness(positions, record, *, g, problem, k):
    """Replay the complete exact symbolic certificate; reject tampering."""
    try:
        got = record.to_dict() if isinstance(record, SameRayWitness) else record
        recomputed = construct_same_ray_witness(
            positions, g=g, problem=problem, k=k,
        ).to_dict()
        if got != recomputed:
            return False
        if got["conclusion"] != "exact_same_ray_dominator":
            return False
        gy = _decode(got["original_g"])
        gp = _decode(got["witness_g"])
        witness = [_decode(v) for v in got["witness_positions"]]
        source = [_as_exact_fraction(v, name="position") for v in positions]
        if len(source) != len(witness) or not (0 <= gp < gy):
            return False
        for i in range(1, len(source)):
            before = (1+2*gy*source[i])/(1+gy)
            after = (1+2*gp*witness[i])/(1+gp)
            if before != after:
                return False
        return (witness[0] == source[0]
                and _decode(got["objective_scale"]) == (1+gp)/(1+gy)
                and all(0 <= v <= 1 for v in witness))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
