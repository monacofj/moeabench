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
    """Restrict published-geometry oracles to unmodified original built-ins."""
    from .DTLZ1 import DTLZ1
    from .DTLZ2 import DTLZ2
    from .DTLZ3 import DTLZ3
    from .DTLZ4 import DTLZ4
    from .DTLZ7 import DTLZ7
    from .DTLZ8 import DTLZ8
    from .DTLZ9 import DTLZ9
    from .DPF1 import DPF1
    if any(name in vars(mop) for name in ("pf", "ps", "evaluation")):
        return None
    kind = type(mop)
    if kind is DTLZ1:
        return "simplex"
    if kind in (DTLZ2, DTLZ3, DTLZ4):
        return "sphere"
    if kind is DTLZ7:
        return "dtlz7"
    if kind is DTLZ8:
        return "dtlz8"
    if kind is DTLZ9:
        return "dtlz9"
    if kind is DPF1:
        return "dpf1"
    return None


def _dtlz7_record_bounds():
    """Independent derivative and record-crossing roots for a(x)=x(1+sin(3*pi*x))."""
    from scipy.optimize import brentq
    def a(t):
        return t * (1 + np.sin(3 * np.pi * t))
    def derivative(t):
        return 1 + np.sin(3 * np.pi * t) + 3 * np.pi * t * np.cos(3 * np.pi * t)
    first = brentq(derivative, .20, .30)
    last = brentq(derivative, .80, .90)
    reentry = brentq(lambda t: a(t) - a(first), .55, .75)
    return first, reentry, last


def _independent_oracle(kind, points, tolerance, mop):
    m = points.shape[1]
    if kind == "simplex":
        residual = np.abs(np.sum(points, axis=1) - 0.5)
        proof = "independent DTLZ1 simplex: f>=0, sum(f)=0.5"
    elif kind == "sphere":
        residual = np.abs(np.linalg.norm(points, axis=1) - 1.0)
        proof = "independent DTLZ2/3/4 positive unit sphere"
    elif kind == "dpf1":
        d = mop.D
        essential = points[:, :d]
        residual = np.abs(np.sum(essential, axis=1) - 0.5)
        weights = np.asarray(mop._chaos_pool[:(m-d)*d]).reshape(m-d, d)
        predicted = essential @ weights.T
        residual = np.maximum(residual, np.max(np.abs(
            points[:, d:] - predicted), axis=1))
        proof = "independent DPF1 essential simplex and chaotic linear projections"
    elif kind == "dtlz9":
        t = points[:, -1]
        expected = np.sqrt(np.clip(1-t*t, 0, None))
        residual = np.max(np.abs(points[:, :-1] - expected[:, None]), axis=1)
        residual = np.maximum(residual, np.maximum(-t, t-1))
        proof = "independent constrained DTLZ9 circular objective curve"
    elif kind == "dtlz7":
        first, reentry, last = _dtlz7_record_bounds()
        x = points[:, :-1]
        allowed = (((x >= -tolerance) & (x <= first+tolerance)) |
                   ((x >= reentry-tolerance) & (x <= last+tolerance)))
        benefit = np.sum(x * (1 + np.sin(3*np.pi*x)), axis=1)
        expected = 2*m - benefit
        residual = np.abs(points[:, -1] - expected)
        residual = np.where(np.all(allowed, axis=1), residual, np.inf)
        proof = "independent DTLZ7 record-interval roots and final-objective equation"
    elif kind == "dtlz8":
        t = points[:, -1]
        f = points[:, :-1]
        a = ((1-t)/4)[:, None]
        if m == 2:
            residual = np.abs(f[:, 0] - a[:, 0])
        else:
            # Central line for t>=1/3, and M-1 boundary patches below.
            b = (1-2*t)[:, None]
            low = np.min(f, axis=1)[:, None]
            high = b-low
            k = np.argmin(f, axis=1)
            predicted = np.broadcast_to(high, f.shape).copy()
            predicted[np.arange(len(f)), k] = low[:, 0]
            patch_err = np.max(np.abs(f-predicted), axis=1)
            legal_low = ((low[:, 0] >= a[:, 0]-tolerance) &
                         (low[:, 0] <= (b[:, 0]/2)+tolerance))
            patch_err = np.where(legal_low, patch_err, np.inf)
            line_err = np.max(np.abs(f-a), axis=1)
            residual = np.where(t < 1/3, patch_err, line_err)
        residual = np.maximum(residual, np.maximum(-t, t-1))
        proof = "independent constrained DTLZ8 line and M-1 patch equations"
    else:
        return GTProperty("unverified", basis="no independent certified oracle")
    bad = np.count_nonzero(np.any(points < -tolerance, axis=1) |
                           (residual > tolerance))
    return GTProperty("fail" if bad else "pass",
                      value={"violations": int(bad),
                             "max_residual": float(np.max(residual))},
                      basis=proof, threshold=tolerance,
                      unit="objective coordinates")


def _independent_probes(kind, mop, count, seed):
    """True-front objective probes derived from independent geometry."""
    m = mop.M
    rng = np.random.default_rng(seed)
    if kind == "simplex":
        return rng.dirichlet(np.ones(m), size=count) * .5
    if kind == "sphere":
        directions = np.abs(rng.standard_normal((count, m)))
        return directions / np.linalg.norm(directions, axis=1)[:, None]
    if kind == "dpf1":
        d = mop.D
        first = rng.dirichlet(np.ones(d), size=count) * .5
        weights = np.asarray(mop._chaos_pool[:(m-d)*d]).reshape(m-d, d)
        return np.column_stack((first, first @ weights.T))
    if kind == "dtlz9":
        t = rng.uniform(0., np.pi/2, size=count)
        return np.column_stack((
            np.broadcast_to(np.cos(t)[:, None], (count, m-1)),
            np.sin(t)))
    if kind == "dtlz7":
        first, reentry, last = _dtlz7_record_bounds()
        masks = rng.integers(0, 2, size=(count, m-1))
        unit = rng.uniform(size=(count, m-1))
        x = np.where(masks == 0,
                     first * unit,
                     reentry + 1e-8 + (last-reentry-1e-8)*unit)
        f = 2*m - np.sum(x*(1 + np.sin(3*np.pi*x)), axis=1)
        return np.column_stack((x, f))
    if kind == "dtlz8":
        groups = max(2, m)
        labels = np.arange(count) % groups
        rng.shuffle(labels)
        t = rng.uniform(size=count)
        out = np.empty((count, m))
        if m == 2:
            out[:, -1] = t
            out[:, 0] = (1-t)/4
            return out
        # Component index 0 denotes the line; remaining labels the patches.
        on_line = labels == 0
        out[on_line, -1] = 1/3 + (2/3)*t[on_line]
        out[on_line, :-1] = ((1-out[on_line, -1])/4)[:, None]
        patch = ~on_line
        out[patch, -1] = t[patch]/3
        b = 1-2*out[patch, -1]
        a = (1-out[patch, -1])/4
        low = a + rng.uniform(size=np.count_nonzero(patch))*(b/2-a)
        high = b-low
        out[patch, :-1] = high[:, None]
        selected = labels[patch]-1
        out[np.flatnonzero(patch), selected] = low
        return out
    return None


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
    validity = _independent_oracle(kind, points, tolerance, mop)

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
    elif kind == "dtlz7":
        first, reentry, _ = _dtlz7_record_bounds()
        x = points[:, :-1]
        masks = (x >= reentry).astype(np.int64)
        n_components = 1 << (mop.M-1)
        distinct_components = len(set(
            sum(int(bit) << i for i, bit in enumerate(row)) for row in masks))
        coverage_possible = n >= n_components
        components = GTProperty(
            ("pass" if distinct_components == n_components else "fail")
            if coverage_possible else "unverified",
            value={"expected": n_components, "represented": distinct_components},
            basis="independent Cartesian disconnected record-interval identities")
    elif kind == "dtlz8":
        t = points[:, -1]
        x = points[:, :-1]
        if mop.M == 2:
            counts = {"line": int(n)}
            complete = True
        else:
            line = t >= 1/3 - tolerance
            counts = {"line": int(np.count_nonzero(line))}
            for i in range(mop.M-1):
                counts[f"patch_{i}"] = int(np.count_nonzero(
                    (~line) & (np.argmin(x, axis=1) == i)))
            complete = all(v > 0 for v in counts.values())
        components = GTProperty(
            ("pass" if complete else "fail") if n >= mop.M+1
            else "unverified",
            value=counts, basis="independent line/patch index from published geometry")
    elif kind == "dtlz9":
        t = points[:, -1]
        count = sum(bool(np.any(abs(t-v) <= tolerance)) for v in (0., 1.))
        components = GTProperty(
            "pass" if count == 2 else "fail",
            value={"true_curve_endpoints_present": count},
            basis="DTLZ9 true objective arc endpoints")
    else:
        components = GTProperty(
            "unverified", basis="no independent component and extrema oracle")

    independent = _independent_probes(kind, mop, probes, int(seed))
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
