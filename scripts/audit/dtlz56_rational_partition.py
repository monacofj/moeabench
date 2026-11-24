# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, replayable partition of a closed g interval for DTLZ5/6.

Only exact-rational interval exclusion can mark a subinterval excluded.
The entire [0, upper] is excluded only when EVERY leaf has an independently
replayable proof. Unresolved leaves and budget exhaustion never count as
positive Pareto-membership evidence. No sampling, GT generation or BLAS.

A finite closed subrange [0, upper] with upper < g_y CANNOT prove global
Pareto membership: the open boundary [upper, g_y) is still outstanding.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .dtlz56_rational_interval import (
    _as_exact_fraction, rigorously_exclude_g_interval,
)


def _hex_fraction(f):
    return {"numerator_hex": hex(f.numerator),
            "denominator_hex": hex(f.denominator)}


def _read_fraction(record):
    return Fraction(int(record["numerator_hex"], 16),
                    int(record["denominator_hex"], 16))


@dataclass(frozen=True)
class PartitionAudit:
    outcome: str
    g_lower: dict
    g_upper: dict
    leaves: tuple[dict, ...]
    explored_nodes: int
    excluded_leaves: int
    unresolved_leaves: int
    proof_basis: str = (
        "Exhaustive rational partition of a specified closed g interval; "
        "each excluded leaf independently verifies an exact-rational "
        "strict infeasibility inequality"
    )

    def to_dict(self):
        return {
            "outcome": self.outcome,
            "g_lower": self.g_lower,
            "g_upper": self.g_upper,
            "leaves": list(self.leaves),
            "explored_nodes": self.explored_nodes,
            "excluded_leaves": self.excluded_leaves,
            "unresolved_leaves": self.unresolved_leaves,
            "proof_basis": self.proof_basis,
            "scientific_certification": "unverified",
        }


def partition_excluded_g_range(
    objectives, *, g_lower=0, g_upper, problem, k,
    max_nodes=63, max_depth=5, pi_terms=6, sine_pairs=2,
):
    """Return finite rational partition; certifies only *excluded leaves*.

    A completed partition of excluded leaves proves absence of a
    coordinatewise dominated feasible original DTLZ5/6 point at any
    g in [g_lower, g_upper] for the EXACT supplied input caps.
    """
    if type(max_nodes) is not int or not (1 <= max_nodes <= 4095):
        raise ValueError("max_nodes must be an integer from 1 to 4095")
    if type(max_depth) is not int or not (0 <= max_depth <= 16):
        raise ValueError("max_depth must be an integer from 0 to 16")
    lo = _as_exact_fraction(g_lower, name="g_lower")
    hi = _as_exact_fraction(g_upper, name="g_upper")
    # Validate even when budget is already exhausted.
    _ = rigorously_exclude_g_interval(
        objectives, g_lower=lo, g_upper=hi,
        problem=problem, k=k, pi_terms=pi_terms, sine_pairs=sine_pairs
    )
    stack = [(lo, hi, 0)]
    leaves = []
    explored = 0
    while stack:
        left, right, depth = stack.pop()
        if explored == max_nodes:
            leaves.append({
                "lower": _hex_fraction(left), "upper": _hex_fraction(right),
                "status": "unresolved_budget", "certificate": None
            })
            for a, b, _ in stack:
                leaves.append({
                    "lower": _hex_fraction(a), "upper": _hex_fraction(b),
                    "status": "unresolved_budget", "certificate": None
                })
            break
        examined = rigorously_exclude_g_interval(
            objectives, g_lower=left, g_upper=right,
            problem=problem, k=k, pi_terms=pi_terms, sine_pairs=sine_pairs
        )
        explored += 1
        if examined.conclusion == "rigorously_excluded":
            leaves.append({
                "lower": _hex_fraction(left), "upper": _hex_fraction(right),
                "status": "rigorously_excluded",
                "certificate": examined.to_dict(),
            })
        elif depth >= max_depth or left == right:
            leaves.append({
                "lower": _hex_fraction(left), "upper": _hex_fraction(right),
                "status": "unresolved", "certificate": None,
            })
        else:
            mid = (left + right) / 2
            # Partition both halves; shared midpoint is intentional.
            stack.append((mid, right, depth + 1))
            stack.append((left, mid, depth + 1))
    leaves.sort(key=lambda item: (_read_fraction(item["lower"]),
                                  _read_fraction(item["upper"])))
    excluded = sum(a["status"] == "rigorously_excluded" for a in leaves)
    unresolved = len(leaves) - excluded
    return PartitionAudit(
        "rigorously_excluded_closed_range" if not unresolved else "unverified",
        _hex_fraction(lo), _hex_fraction(hi), tuple(leaves),
        explored, excluded, unresolved,
    )


def replay_partition_proof(
    objectives, audit, *, problem, k, pi_terms=6, sine_pairs=2
):
    """Check continuous coverage and independently recompute excluded leaves.

    Never trust a stored certificate's declared conclusion or margin alone.
    Recompute each strict inequality from inputs and verify exact endpoints,
    adjacency, leaf status, and absence of unproven portions.
    """
    record = audit.to_dict() if isinstance(audit, PartitionAudit) else audit
    leaves = record.get("leaves", ())
    if not leaves:
        return False
    try:
        lower = _read_fraction(record["g_lower"])
        upper = _read_fraction(record["g_upper"])
        current = lower
        for leaf in leaves:
            a = _read_fraction(leaf["lower"])
            b = _read_fraction(leaf["upper"])
            if a != current or b < a or b > upper:
                return False
            if leaf["status"] != "rigorously_excluded":
                return False
            certificate = rigorously_exclude_g_interval(
                objectives, g_lower=a, g_upper=b,
                problem=problem, k=k,
                pi_terms=pi_terms, sine_pairs=sine_pairs
            )
            if certificate.conclusion != "rigorously_excluded":
                return False
            if leaf["certificate"] != certificate.to_dict():
                return False
            current = b
        return (current == upper
                and record["outcome"] == "rigorously_excluded_closed_range")
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return False
