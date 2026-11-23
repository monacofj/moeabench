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


def _immutable(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _immutable(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_immutable(v) for v in value)
    return value


def _plain(value):
    if isinstance(value, MappingProxyType):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class GTProperty:
    status: str
    value: Any = None
    basis: str = ""
    threshold: float | None = None
    unit: str = ""

    def __post_init__(self):
        if self.status not in ("pass", "fail", "unverified", "not_applicable"):
            raise ValueError(f"Unknown GT property status: {self.status}")
        object.__setattr__(self, "value", _immutable(self.value))

    def to_dict(self):
        return {"status": self.status, "value": _plain(self.value),
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
    dominance: GTProperty = field(default_factory=lambda: GTProperty(
        "not_applicable", basis="bounded-margin dominance audit not requested"))
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
                "regularity", "cardinality", "uniqueness", "replay", "dominance")},
            "diagnostics": list(self.diagnostics),
        }

    def summary(self):
        lines = [f"GT assessment: {self.mop_id} (M={self.objectives})",
                 f"Reference: {self.count} rows; {self.unique_count} unique; "
                 f"source={self.provenance or 'unknown'}"]
        for name in ("cardinality", "uniqueness", "validity", "feasibility",
                     "components", "coverage", "regularity", "replay", "dominance"):
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
    from .DTLZ5 import DTLZ5
    from .DTLZ6 import DTLZ6
    from .DTLZ7 import DTLZ7
    from .DTLZ8 import DTLZ8
    from .DTLZ9 import DTLZ9
    from .DPF1 import DPF1
    from .DPF2 import DPF2
    from .DPF3 import DPF3
    from .DPF4 import DPF4
    from .DPF5 import DPF5
    if any(name in vars(mop) for name in ("pf", "ps", "evaluation")):
        return None
    kind = type(mop)
    if kind is DTLZ1:
        return "simplex"
    if kind in (DTLZ2, DTLZ3, DTLZ4):
        return "sphere"
    if kind is DTLZ5:
        return "dtlz5"
    if kind is DTLZ6:
        return "dtlz6"
    if kind is DTLZ7:
        return "dtlz7"
    if kind is DTLZ8:
        return "dtlz8"
    if kind is DTLZ9:
        return "dtlz9"
    if kind is DPF1:
        return "dpf1"
    if kind is DPF2:
        return "dpf2"
    if kind is DPF3:
        return "dpf3"
    if kind is DPF4:
        return "dpf4"
    if kind is DPF5:
        return "dpf5"
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
    elif kind in ("dtlz5", "dtlz6"):
        from ._dtlz56_independent_audit import inspect_dtlz56_geometry
        evidence = inspect_dtlz56_geometry(
            points, m=m, k=mop.N-m+1, problem=kind.upper(),
            tolerance=tolerance)
        if evidence["violations"]:
            status = "fail"
        elif m >= 4:
            # Even exact-g0 points prove their own membership, they do not
            # certify that a finite GT covers the original full PF with
            # positive-g branches. Preserve the established regression gate.
            status = "unverified"
        elif evidence["unverified_positive_g"]:
            status = "unverified"
        else:
            status = "pass"
        return GTProperty(
            status, value=evidence,
            basis="independent original DTLZ5/6 angular image and minimal-g "
                  "radial dominance; exact global PF membership proved at g=0; "
                  "g>0 remains unverified for global Pareto efficiency",
            threshold=tolerance, unit="objective-space geometry")
    elif kind == "dpf1":
        d = mop.D
        essential = points[:, :d]
        residual = np.abs(np.sum(essential, axis=1) - 0.5)
        weights = np.asarray(mop._chaos_pool[:(m-d)*d]).reshape(m-d, d)
        predicted = essential @ weights.T
        residual = np.maximum(residual, np.max(np.abs(
            points[:, d:] - predicted), axis=1))
        proof = "independent DPF1 essential simplex and chaotic linear projections"
    elif kind == "dpf2":
        d = mop.D
        x = points[:, :d-1]
        first, reentry, last = _dtlz7_record_bounds()
        allowed = (((x >= -tolerance) & (x <= first+tolerance)) |
                   ((x >= reentry-tolerance) & (x <= last+tolerance)))
        final_base = 2*d - np.sum(
            x*(1 + np.sin(3*np.pi*x)), axis=1)
        base = np.column_stack((x, final_base))
        weights = np.asarray(mop._chaos_pool[:(m-d)*d]).reshape(m-d, d)
        predicted = (base @ weights.T)**2
        residual = np.maximum(np.abs(points[:, d-1] - final_base),
                              np.max(np.abs(points[:, d:] - predicted), axis=1))
        residual = np.where(np.all(allowed, axis=1), residual, np.inf)
        proof = "independent DPF2 record-domain and squared nonnegative projections"
    elif kind in ("dpf3", "dpf4"):
        d = mop.D
        y = points[:, :d-1] if kind == "dpf3" else np.sqrt(
            np.maximum(points[:, :d-1], 0))
        s = 1-y
        eta = np.sort(np.asarray(mop._chaos_pool[:m-d]))
        observed = (points[:, d-1:] if kind == "dpf3"
                    else np.sqrt(np.maximum(points[:, d-1:], 0)))
        # The published clipped projections partition y_last:
        # sum_j projected_j = y_last + sum_j eta_j.
        # Recover y_last directly instead of subtracting nearly equal
        # squared sphere coordinates and taking an unstable square root.
        y_last = np.sum(observed, axis=1) - np.sum(eta)
        redundant = [np.minimum(y_last, eta[0])]
        for j in range(m-d-1):
            redundant.append(np.minimum(np.maximum(y_last, eta[j]), eta[j+1]))
        redundant.append(np.maximum(y_last, eta[-1]))
        expected = np.column_stack(redundant)
        sphere_residual = np.abs(np.sum(s*s, axis=1) +
                                 (1-y_last)**2 - 1)
        projection_residual = np.max(np.abs(observed-expected), axis=1)
        residual = np.maximum(sphere_residual, projection_residual)
        residual = np.where((y_last >= -tolerance) &
                            (y_last <= 1+tolerance), residual, np.inf)
        proof = ("independent inverted sphere and sum-inverted "
                 "published ordered projections")
    elif kind == "dpf5":
        # On the published Pareto branches g=0 and norm(F)=1.
        # P has first angle < pi/6, equivalent to f_M < 1/2.
        # D has f_M >= 1/2 and M-D+1 repeated first coordinates.
        t = points[:, -1]
        repeats = m-mop.D+1
        d_equal = np.max(
            np.abs(points[:, :repeats]-points[:, :1]), axis=1)
        allowed = (t < .5+tolerance) | (d_equal <= tolerance)
        residual = np.abs(np.linalg.norm(points, axis=1)-1)
        residual = np.where(allowed, residual, np.inf)
        proof = "independent published DPF5 P/D unit-sphere and repeated-coordinate geometry"
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
    maximum = float(np.max(residual))
    return GTProperty("fail" if bad else "pass",
                      value={"violations": int(bad),
                             "max_residual": maximum if np.isfinite(maximum) else None},
                      basis=proof, threshold=tolerance,
                      unit="objective coordinates")


def _spherical_oracle_coordinates(angles):
    """Objective-space positive hypersphere, independent of MOP evaluators."""
    n, q = angles.shape
    result = np.ones((n, q+1))
    for j in range(q+1):
        if j < q:
            result[:, j] *= np.prod(np.cos(angles[:, :q-j]), axis=1)
        if j:
            result[:, j] *= np.sin(angles[:, q-j])
    return result


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
    if kind == "dpf2":
        d = mop.D
        first, reentry, last = _dtlz7_record_bounds()
        mask = rng.integers(0, 2, size=(count, d-1))
        u = rng.uniform(size=(count, d-1))
        x = np.where(mask == 0, first*u,
                     reentry+1e-8+(last-reentry-1e-8)*u)
        f_last = 2*d - np.sum(x*(1+np.sin(3*np.pi*x)), axis=1)
        base = np.column_stack((x, f_last))
        weights = np.asarray(mop._chaos_pool[:(m-d)*d]).reshape(m-d,d)
        return np.column_stack((base, (base @ weights.T)**2))
    if kind in ("dpf3", "dpf4"):
        d = mop.D
        theta = rng.uniform(0, np.pi/2, size=(count, d-1))
        sphere = _spherical_oracle_coordinates(theta)
        y = 1-sphere
        eta = np.sort(np.asarray(mop._chaos_pool[:m-d]))
        last = y[:, -1]
        parts = [np.minimum(last, eta[0])]
        for j in range(m-d-1):
            parts.append(np.minimum(np.maximum(last, eta[j]), eta[j+1]))
        parts.append(np.maximum(last, eta[-1]))
        full = np.column_stack((y[:, :-1], np.column_stack(parts)))
        return full if kind == "dpf3" else full**2
    if kind == "dpf5":
        d = mop.D
        mask = np.arange(count) % 2 == 0
        angles_p = rng.uniform(0, np.pi/2, size=(int(np.count_nonzero(mask)),m-1))
        angles_p[:, 0] *= 1/3
        p = _spherical_oracle_coordinates(angles_p)
        angles_d = rng.uniform(0, np.pi/2, size=(count-len(p),d-1))
        angles_d[:, 0] = np.pi/6 + (np.pi/3)*angles_d[:, 0]/(np.pi/2)
        essential = _spherical_oracle_coordinates(angles_d)
        repeats = m-d+1
        dd = np.column_stack((
            np.repeat(essential[:, :1]/np.sqrt(repeats), repeats, axis=1),
            essential[:, 1:]))
        out = np.empty((count,m))
        out[mask] = p
        out[~mask] = dd
        return out
    return None


def _component_labels(kind, points, mop):
    """Identify verified named Pareto components in objective space."""
    if kind in ("dtlz7", "dpf2"):
        _, reentry, _ = _dtlz7_record_bounds()
        dimension = mop.M-1 if kind == "dtlz7" else mop.D-1
        mask = points[:, :dimension] >= reentry
        return np.array([
            "record_" + "".join(str(int(bit)) for bit in row)
            for row in mask], dtype=str)
    if kind == "dtlz8":
        if mop.M == 2:
            return np.full(len(points), "line", dtype=str)
        t = points[:, -1]
        minimum = np.argmin(points[:, :-1], axis=1)
        return np.array([
            "line" if t[i] >= 1/3 else f"patch_{minimum[i]}"
            for i in range(len(points))], dtype=str)
    if kind == "dpf5":
        return np.where(points[:, -1] < .5, "P", "D")
    return None


def assess_gt(mop, reference=None, *, n_points=None, probes=128,
              seed=0, tolerance=1e-9, dominance_margin=None,
              dominance_max_intervals=4096):
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
    if dominance_margin is not None:
        if not np.isfinite(dominance_margin) or dominance_margin <= 0:
            raise ValueError("dominance_margin must be finite and positive")
        if not isinstance(dominance_max_intervals, int) or dominance_max_intervals < 1:
            raise ValueError("dominance_max_intervals must be a positive integer")
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
        n = len(points)
        invalid = int(np.size(points) - np.count_nonzero(np.isfinite(points)))
        unverified = GTProperty("unverified", basis="nonfinite coordinates prevent geometry tests")
        return GTAssessment(
            mop_id=str(mop.name), objectives=int(mop.M),
            variables=None if mop.N is None else int(mop.N),
            provenance=provenance, requested_count=n_points,
            count=n, unique_count=0, probe_count=probes,
            seed=int(seed), tolerance=float(tolerance),
            validity=GTProperty("fail",
                value={"nonfinite_coordinates": invalid},
                basis="finite float64 coordinates required"),
            feasibility=unverified, components=unverified,
            coverage=unverified, regularity=unverified,
            cardinality=GTProperty(
                "pass" if n_points is None or n==n_points else "fail",
                value={"requested": n_points, "actual": n},
                basis="exact row count"),
            uniqueness=unverified, replay=unverified,
            diagnostics=("Nonfinite GT coordinates prevent scientific use.",)
        )

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
    elif kind in ("dtlz5", "dtlz6"):
        from ._dtlz56_independent_audit import inspect_dtlz56_geometry
        evidence = inspect_dtlz56_geometry(
            points, m=mop.M, k=mop.N-mop.M+1,
            problem=kind.upper(), tolerance=tolerance)
        zero = evidence["g_zero"]
        positive = evidence["g_positive"]
        if mop.M >= 4:
            status = ("pass" if zero and positive else "fail") if n >= 4 else "unverified"
        else:
            status = "pass" if zero == n else "fail"
        components = GTProperty(
            status,
            value={"g_zero": zero, "g_positive": positive,
                   "expected_positive_g": mop.M >= 4},
            basis="independent full-original-front g=0 and g>0 region occupancy; "
                  "presence alone does not certify global membership")
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
    elif kind == "dpf2":
        _, reentry, _ = _dtlz7_record_bounds()
        x = points[:, :mop.D-1]
        masks = (x >= reentry).astype(np.int64)
        expected = 1 << (mop.D-1)
        observed = len(set(
            sum(int(bit) << i for i, bit in enumerate(row)) for row in masks))
        components = GTProperty(
            ("pass" if expected == observed else "fail") if n >= expected
            else "unverified",
            value={"expected": expected, "represented": observed},
            basis="independent DPF2 essential record-component identities")
    elif kind == "dpf5":
        p_count = int(np.count_nonzero(points[:, -1] < .5-tolerance))
        d_count = n-p_count
        components = GTProperty(
            "pass" if p_count and d_count else "fail",
            value={"P": p_count, "D": d_count},
            basis="published DPF5 first-angle P/D branch condition")
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
    labels = _component_labels(kind, distinct, mop)
    if independent is not None:
        distances = cKDTree(distinct).query(independent, k=1)[0]
        gaps = {"max_probe_gap": float(np.max(distances)),
                "mean_probe_gap": float(np.mean(distances))}
        # Do not average measures across disconnected or different-dimensional
        # components. Report each region's independently probed gap.
        if labels is not None:
            probe_labels = _component_labels(kind, independent, mop)
            gaps_by_component = {}
            for label in np.unique(probe_labels):
                chosen = probe_labels == label
                region = distinct[labels == label]
                if len(region):
                    d = cKDTree(region).query(independent[chosen], k=1)[0]
                    gaps_by_component[str(label)] = {
                        "probes": int(np.count_nonzero(chosen)),
                        "points": int(len(region)),
                        "max_probe_gap": float(np.max(d)),
                        "mean_probe_gap": float(np.mean(d)),
                    }
                else:
                    gaps_by_component[str(label)] = {
                        "probes": int(np.count_nonzero(chosen)),
                        "points": 0,
                        "max_probe_gap": None,
                        "mean_probe_gap": None,
                    }
            gaps["by_component"] = gaps_by_component
        coverage = GTProperty(
            "unverified", value=gaps,
            basis="independent finite true-front probes; NOT a continuous bound",
            unit="original objective-space Euclidean distance")
    else:
        coverage = GTProperty(
            "unverified", basis="independent true-front probes unavailable")

    def spacing_diagnostics(rows):
        if len(rows) < 2:
            return {"count": len(rows), "mean_nearest_neighbor": None,
                    "cv_nearest_neighbor": None, "max_nearest_neighbor": None}
        d = cKDTree(rows).query(rows, k=2)[0][:, 1]
        mean = float(np.mean(d))
        return {
            "count": len(rows),
            "mean_nearest_neighbor": mean,
            "cv_nearest_neighbor":
                float(np.std(d)/mean) if mean > 0 else None,
            "max_nearest_neighbor": float(np.max(d))
        }

    if labels is not None:
        parts = {str(label): spacing_diagnostics(distinct[labels == label])
                 for label in np.unique(labels)}
        regularity = GTProperty(
            "unverified", value={"by_component": parts},
            basis="within-component objective-space nearest neighbors; "
                  "no intercomponent probability weighting",
            unit="original objective-space Euclidean distance")
    elif unique_n >= 2:
        regularity = GTProperty(
            "unverified", value=spacing_diagnostics(distinct),
            basis="whole-reference Euclidean nearest-neighbor distances",
            unit="original objective-space Euclidean distance")
    else:
        regularity = GTProperty(
            "unverified", basis="at least two distinct points required")

    dominance = GTProperty(
        "not_applicable", basis="bounded-margin dominance audit not requested")
    if dominance_margin is not None:
        if kind not in ("dtlz5", "dtlz6"):
            raise ValueError(
                "bounded-margin dominance audit supports only original DTLZ5/6")
        from ._dtlz56_dominance_bounds import audit_delta_dominance
        results = [
            audit_delta_dominance(
                point, problem=kind.upper(), k=mop.N-mop.M+1,
                margin=float(dominance_margin),
                max_intervals=dominance_max_intervals)
            for point in points
        ]
        excluded = sum(a.status == "excluded" for a in results)
        dominance = GTProperty(
            "unverified",
            value={
                "margin": float(dominance_margin),
                "numerically_excluded": excluded,
                "unresolved": len(results)-excluded,
                "total_g_intervals": sum(a.intervals_tested for a in results),
                "max_intervals_per_point": dominance_max_intervals,
            },
            basis="independent continuous-g interval exclusion of "
                  "coordinatewise improvement >= margin; numerical slack, "
                  "no directed rounding or full Pareto proof",
            threshold=float(dominance_margin),
            unit="objective coordinates")

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
        uniqueness=uniqueness, replay=replay, dominance=dominance,
        diagnostics=("No finite-probe covering bound is a continuous guarantee.",
                     "A GT assessment is not a release certificate."),
    )
