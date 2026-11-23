# SPDX-FileCopyrightText: 2026 Monaco F. J.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only quantitative GT diagnostics (ADR 0059).

Finite geometric probes never constitute continuous-front certificates.
Membership is certified only by explicitly supported independent geometric
oracles, never by calling the MOP's own sampler as its oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GTProperty:
    status: str
    value: Any = None
    basis: str = ""
    threshold: float | None = None
    unit: str = ""

    def to_dict(self):
        return {"status": self.status, "value": self.value,
                "basis": self.basis, "threshold": self.threshold,
                "unit": self.unit}


@dataclass(frozen=True)
class GTAssessment:
    mop_id: str
    objectives: int
    variables: int | None
    provenance: str | None
    requested_count: int | None
    count: int
    unique_count: int
    probe_count: int
    seed: int
    tolerance: float
    validity: GTProperty
    feasibility: GTProperty
    components: GTProperty
    coverage: GTProperty
    regularity: GTProperty
    cardinality: GTProperty
    uniqueness: GTProperty
    replay: GTProperty
    diagnostics: tuple = field(default_factory=tuple)

    def to_dict(self):
        return {
            "mop_id": self.mop_id, "objectives": self.objectives,
            "variables": self.variables, "provenance": self.provenance,
            "requested_count": self.requested_count, "count": self.count,
            "unique_count": self.unique_count, "probe_count": self.probe_count,
            "seed": self.seed, "tolerance": self.tolerance,
            **{k: getattr(self, k).to_dict() for k in (
                "validity", "feasibility", "components", "coverage",
                "regularity", "cardinality", "uniqueness", "replay")},
            "diagnostics": list(self.diagnostics),
        }

    def summary(self):
        lines = [f"GT assessment: {self.mop_id} (M={self.objectives})",
                 f"Reference: {self.count} rows; {self.unique_count} unique; "
                 f"source={self.provenance or 'unknown'}"]
        for name in ("cardinality", "uniqueness", "validity", "feasibility",
                     "components", "coverage", "regularity", "replay"):
            p = getattr(self, name)
            value = "" if p.value is None else f" | {p.value}"
            lines.append(f"  {name:12s} {p.status:14s}{value}"
                         + (f" | {p.unit}" if p.unit else ""))
        if self.diagnostics:
            lines += ["  " + note for note in self.diagnostics]
        return "\n".join(lines)


def _original_problem_kind(mop):
    """Restrict proven oracles to unmodified original built-in classes."""
    from .DTLZ1 import DTLZ1
    from .DTLZ2 import DTLZ2
    from .DTLZ3 import DTLZ3
    from .DTLZ4 import DTLZ4
    if any(name in vars(mop) for name in ("pf", "ps", "evaluation")):
        return None
    kind = type(mop)
    if kind is DTLZ1:
        return "simplex"
    if kind in (DTLZ2, DTLZ3, DTLZ4):
        return "sphere"
    return None


def _independent_oracle(kind, points, tolerance):
    if kind == "simplex":
        residual = np.abs(np.sum(points, axis=1) - 0.5)
        proof = "independent DTLZ1 simplex: f>=0, sum(f)=0.5"
    elif kind == "sphere":
        residual = np.abs(np.linalg.norm(points, axis=1) - 1.0)
        proof = "independent DTLZ2/3/4 positive unit sphere"
    else:
        return GTProperty("unverified", basis="no independent certified oracle")
    bad = np.count_nonzero(np.any(points < -tolerance, axis=1) |
                           (residual > tolerance))
    return GTProperty("fail" if bad else "pass",
                      value={"violations": int(bad),
                             "max_residual": float(np.max(residual))},
                      basis=proof, threshold=tolerance,
                      unit="objective coordinates")


def _independent_probes(kind, m, count, seed):
    """Independent objective geometry, never mop.pf() for validation."""
    rng = np.random.default_rng(seed)
    if kind == "simplex":
        directions = rng.dirichlet(np.ones(m), size=count) * .5
    elif kind == "sphere":
        directions = np.abs(rng.standard_normal((count, m)))
        norms = np.linalg.norm(directions, axis=1)
        directions /= norms[:, None]
    else:
        return None
    return directions


def assess_gt(mop, reference=None, *, n_points=None, probes=128,
              seed=0, tolerance=1e-9):
    """Inspect a GT reference without writing, calibrating or modifying it.

    If neither a reference nor n_points is supplied, an existing in-memory
    calibrated GT is required. An explicit n_points samples one on-demand
    analytical candidate *without* installing that sample as calibrated GT.
    External objective-only references cannot independently establish decision
    feasibility, even if their objective geometry matches a known front.
    """
    if not isinstance(probes, int) or probes < 2:
        raise ValueError("probes must be an integer >= 2")
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if reference is not None and n_points is not None:
        raise ValueError("Supply reference or n_points, not both")
    if n_points is not None:
        if not isinstance(n_points, int) or n_points < 2:
            raise ValueError("n_points must be an integer >= 2")
        points = mop.pf(n_points=n_points)
        provenance = "analytical_candidate"
    elif reference is not None:
        points = reference
        provenance = "external_argument"
    else:
        points = mop.calibrated_gt
        provenance = getattr(mop, "_gt_provenance", None)

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != mop.M or not len(points):
        raise ValueError(f"GT must be a nonempty two-dimensional (n, {mop.M}) array")
    if not np.all(np.isfinite(points)):
        # A nonfinite GT should be measurable, not silently considered valid.
        raise ValueError("GT contains NaN or infinity; finite GT is required")

    n = len(points)
    distinct = np.unique(points, axis=0)
    unique_n = len(distinct)
    cardinality = GTProperty(
        "pass" if n_points is None or n == n_points else "fail",
        value={"requested": n_points, "actual": n},
        basis="exact row count")
    uniqueness = GTProperty(
        "pass" if unique_n == n else "fail",
        value={"distinct": unique_n, "duplicates": n - unique_n},
        basis="exact float64 row equality")
    kind = _original_problem_kind(mop)
    validity = _independent_oracle(kind, points, tolerance)

    # A point in objective space cannot, on its own, establish that a
    # feasible decision generated it. This is intentionally unverified.
    feasibility = GTProperty(
        "unverified", basis="decision-space witness unavailable in GT objectives")

    # For the simplex and sphere, axis extrema can be checked against exact
    # mathematical extrema. This is not an overall covering guarantee.
    if kind in ("simplex", "sphere"):
        extrema = .5 if kind == "simplex" else 1.
        axis_found = [
            bool(np.any(np.max(np.abs(points -
                 extrema * np.eye(mop.M)[i]), axis=1) <= tolerance))
            for i in range(mop.M)
        ]
        components = GTProperty(
            "pass" if all(axis_found) else "fail",
            value={"true_axis_extrema_present": axis_found},
            basis=f"proved {kind} axis extrema", threshold=tolerance)
    else:
        components = GTProperty(
            "unverified", basis="no independent component and extrema oracle")

    independent = _independent_probes(kind, mop.M, probes, int(seed))
    if independent is not None:
        distances = cKDTree(distinct).query(independent, k=1)[0]
        coverage = GTProperty(
            "unverified",
            value={"max_probe_gap": float(np.max(distances)),
                   "mean_probe_gap": float(np.mean(distances))},
            basis="independent finite true-front probes; NOT a continuous bound",
            unit="original objective-space Euclidean distance")
    else:
        coverage = GTProperty("unverified",
                              basis="independent true-front probes unavailable")

    if unique_n >= 3:
        spacing = cKDTree(distinct).query(distinct, k=2)[0][:, 1]
        mean = float(np.mean(spacing))
        regularity = GTProperty(
            "unverified",
            value={"mean_nearest_neighbor": mean,
                   "cv_nearest_neighbor":
                   float(np.std(spacing) / mean) if mean > 0 else None},
            basis="whole-reference Euclidean NN; component grouping not verified",
            unit="original objective-space Euclidean distance")
    else:
        regularity = GTProperty(
            "unverified", basis="at least three distinct points required")

    # Replaying PS/PF without a sound budget bound may be expensive and could
    # consume an opaque custom MOP RNG. Do not claim determinism by inference.
    replay = GTProperty(
        "unverified", basis="no independent replay performed (read-only audit)")
    return GTAssessment(
        mop_id=str(mop.name), objectives=int(mop.M),
        variables=None if mop.N is None else int(mop.N),
        provenance=provenance, requested_count=n_points,
        count=n, unique_count=unique_n, probe_count=probes,
        seed=int(seed), tolerance=float(tolerance),
        validity=validity, feasibility=feasibility,
        components=components, coverage=coverage,
        regularity=regularity, cardinality=cardinality,
        uniqueness=uniqueness, replay=replay,
        diagnostics=("No finite-probe covering bound is a continuous guarantee.",
                     "A GT assessment is not a release certificate."),
    )
