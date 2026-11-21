# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Published DPF5 evaluation with the full P and degenerate D branches.

This corrects the mathematical objective evaluator without changing the
finite PS sampler or historical GT/calibration policy. Those require
separate geometry-specific validation before a production merge.
"""

import numpy as np

from .base_dpf import BaseDPF


class DPF5(BaseDPF):
    """DPF5: P=(M-1)-dimensional, D=(D-1)-dimensional angular fronts."""

    def __init__(self, **kwargs):
        if 'N' not in kwargs:
            M = kwargs.get('M', 3)
            D = kwargs.get('D', 2)
            if D < 2 or D >= M:
                D = 2
            K = kwargs.get('K', 5)
            kwargs['N'] = max(D + K - 1, M)
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        self._require_supported_dimensions()

    def _require_supported_dimensions(self):
        """The published distance function uses x_M explicitly."""
        if self.N < self.M:
            raise ValueError(
                f"DPF5 requires N >= M because its distance function uses x_M "
                f"(got M={self.M}, N={self.N}). Provide N >= {self.M}. "
                "Other DPF benchmarks can allow M > N."
            )

    @staticmethod
    def _unit_hypersphere(theta):
        """Nonnegative, unit-norm direction with theta.shape[1]+1 objectives.

        Coordinates follow the standard DTLZ2 hyperspherical convention.
        The implementation is vectorized over individuals and loops only
        over the number of objective coordinates.
        """
        n, q = theta.shape
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        direction = np.empty((n, q + 1), dtype=float)

        for j in range(q + 1):
            cos_count = q - j
            coordinate = np.prod(cos_theta[:, :cos_count], axis=1)
            if j > 0:
                coordinate = coordinate * sin_theta[:, cos_count]
            direction[:, j] = coordinate
        return direction

    def evaluation(self, X, n_ieq_constr=0):
        self._require_supported_dimensions()
        X = np.atleast_2d(X)
        M, D = self.M, self.D

        g = (X[:, M-1] - X[:, 0])**2
        g += np.sum((X[:, M:] - 0.5)**2, axis=1)

        theta = X[:, :M-1] * (np.pi / 2)
        P = self._unit_hypersphere(theta)

        # The D direction repeats its first essential spherical component
        # M-D+1 times, scaled so the resulting M-vector retains unit norm.
        essential = self._unit_hypersphere(theta[:, :D-1])
        repeats = M - D + 1
        D_front = np.concatenate(
            (np.repeat(essential[:, :1] / np.sqrt(repeats), repeats, axis=1),
             essential[:, 1:]),
            axis=1,
        )

        # Per the publication: x1 == 1/3 belongs to D, NOT P.
        choose_D = X[:, 0] >= (1 / 3)
        direction = np.where(choose_D[:, None], D_front, P)
        return {"F": direction * (1 + g[:, None])}

    def get_K(self):
        return self.K

    def ps(self, n_points: int = 100):
        """Geometry-specific, reproducible sampling of the *published* DPF5 PS.

        Branch P has M-1 independent angular parameters; degenerate D has
        only D-1. We sample each intrinsic parameter space separately,
        preserve a small set of deterministic boundary/axis landmarks,
        and retain nested full angular tensor grids (when affordable).
        Duplicates are replenished with deterministic Sobol candidates.
        At the requested total budget, branch allocation prioritizes
        the conservative *real-arithmetic* componentwise angular covering
        bound, breaking ties with independent finite-probe distances.

        The tensor-grid bound does not include floating-point rounding;
        the holdout-probe diagnostic is NOT a certified Hausdorff bound
        or a universal mixed-dimensional probability law. Calibration's
        automatic *default budget* remains governed by ADR 0057.
        """
        from itertools import product
        from scipy.spatial import cKDTree
        from scipy.stats import qmc

        self._require_supported_dimensions()
        if not isinstance(n_points, (int, np.integer)) or n_points < 1:
            raise ValueError("n_points must be a positive integer")

        n = int(n_points)
        qP, qD = self.M - 1, self.D - 1

        def angular_decisions(values, label):
            q = qP if label == "P" else qD
            decisions = np.full((len(values), self.N), 0.5)
            decisions[:, :q] = values
            decisions[:, self.M-1] = decisions[:, 0]
            return decisions

        def branch_bounds(label):
            if label == "P":
                return 0.0, np.nextafter(1.0 / 3.0, 0.0)
            return 1.0 / 3.0, 1.0

        def angular_anchors(label):
            q = qP if label == "P" else qD
            lo, hi = branch_bounds(label)
            out = []
            for first in (lo, hi):
                for fill in (0.0, 0.5, 1.0):
                    row = np.full(q, fill)
                    row[0] = first
                    out.append(row.copy())
                for j in range(1, q):
                    for val in (0.0, 1.0):
                        row = np.full(q, 0.5)
                        row[0], row[j] = first, val
                        out.append(row)
            # Angular coordinate-axis extremes that belong to this branch.
            for axis in range(q+1):
                row = np.zeros(q)
                if axis > 0:
                    row[q-axis] = 1.0
                first = lo + (hi-lo)*row[0]
                row[0] = first
                out.append(row)
            return np.asarray(out, dtype=float)

        def candidate_pool(label, target):
            q = qP if label == "P" else qD
            lo, hi = branch_bounds(label)
            seen = set()
            decisions, objectives, completed_grids = [], [], []

            def ingest(angles):
                if len(decisions) >= target:
                    return
                current = angular_decisions(angles, label)
                evaluated = self.evaluation(current)["F"]
                for x, f in zip(current, evaluated):
                    key = tuple(float(v) for v in f)
                    if key not in seen:
                        seen.add(key)
                        decisions.append(x)
                        objectives.append(f)
                        if len(decisions) >= target:
                            break

            # Full nested tensor grids provide a rigorous exact-arithmetic
            # covering certificate when a completed level fits the budget.
            # k=2,3,5,9,... preserves every coarser angular grid.
            k = 2
            while k ** q <= target:
                grid = np.asarray(
                    list(product(np.linspace(0.0, 1.0, k), repeat=q)),
                    dtype=float,
                ).reshape(-1, q)
                grid[:, 0] = lo + (hi-lo)*grid[:, 0]
                ingest(grid)
                completed_grids.append((len(decisions), k))
                k = 2*k - 1

            ingest(angular_anchors(label))
            for attempt in range(4):
                if len(decisions) >= target:
                    break
                remain = target-len(decisions)
                engine = qmc.Sobol(d=q, scramble=True, seed=11950 +
                                   (0 if label == "P" else 1000) + attempt)
                power = int(np.ceil(np.log2(max(2, remain * 2))))
                angles = engine.random_base2(power)
                angles[:, 0] = lo + (hi-lo)*angles[:, 0]
                ingest(angles)
            if len(decisions) != target:
                raise ValueError("Unable to generate enough distinct DPF5 objectives")
            return np.asarray(decisions), np.asarray(objectives), completed_grids

        # For very small budgets both components may not be fully
        # represented. Prefer explicit samples rather than implicit
        # overproduction; for n>=2 always preserve both branches.
        if n == 1:
            return candidate_pool("P", 1)[0]

        # Reserve at least two samples per branch when possible; retain
        # all unique landmarks when the requested budget permits them.
        P_anchor_count = len(np.unique(
            self.evaluation(angular_decisions(angular_anchors("P"), "P"))["F"],
            axis=0))
        D_anchor_count = len(np.unique(
            self.evaluation(angular_decisions(angular_anchors("D"), "D"))["F"],
            axis=0))
        if n >= P_anchor_count + D_anchor_count:
            minP, minD = P_anchor_count, D_anchor_count
        else:
            minP, minD = max(1, n // 2), max(1, n-n//2)
        maxP, maxD = n-minD, n-minP
        p_dec, p_obj, p_grids = candidate_pool("P", maxP)
        d_dec, d_obj, d_grids = candidate_pool("D", maxD)

        # Low-cost, independent deterministic probes; this estimates the
        # worse of the two conditional gaps without assigning a
        # probability mass to either component.
        def branch_probes(label, count=128):
            q = qP if label == "P" else qD
            lo, hi = branch_bounds(label)
            engine = qmc.Sobol(d=q, scramble=True,
                               seed=23471 if label == "P" else 23497)
            unit = engine.random_base2(int(np.log2(count)))
            unit[:, 0] = lo + (hi-lo)*unit[:, 0]
            return self.evaluation(angular_decisions(unit, label))["F"]

        probesP, probesD = branch_probes("P"), branch_probes("D")
        # Include every grid-level completion threshold. These exact
        # integer splits matter because a certified covering bound
        # improves only after a *whole* angular grid is represented.
        possible = set(int(z) for z in
                       np.linspace(minP, maxP, min(17, maxP-minP+1)))
        possible.update(size for size, _ in p_grids if minP <= size <= maxP)
        possible.update(n-size for size, _ in d_grids
                        if minP <= n-size <= maxP)
        possible = sorted(possible)
        def certified_real_bound(label, prefix, completed):
            # Each completed k-grid is a full ideal published-model cover,
            # including all its mathematically unique boundary images.
            # A prefix may include an incomplete finer grid; the last
            # completed level remains a certificate.
            levels = [k for size, k in completed if size <= prefix]
            if not levels:
                return float("inf")
            k = levels[-1]
            q = qP if label == "P" else qD
            lo, hi = branch_bounds(label)
            angular_width = np.sqrt((hi-lo)**2 + (q-1))
            # The P patch is half-open; its omitted exact x1=1/3
            # boundary lies within a float64 nextafter gap of the grid.
            missing_P_edge = ((1 / 3)-hi) * (np.pi / 2) if label == "P" else 0.0
            return angular_width * np.pi / (4 * (k-1)) + missing_P_edge

        best_score, best_p = (float("inf"), float("inf")), possible[0]
        for p_count in possible:
            d_count = n-p_count
            p_distance = cKDTree(p_obj[:p_count]).query(probesP, k=1)[0]
            d_distance = cKDTree(d_obj[:d_count]).query(probesD, k=1)[0]
            # Compare the mathematical covering guarantee first when
            # both branches have completed grids. If a component lacks
            # a full grid, retain the finite-probe diagnostic only.
            certified = max(
                certified_real_bound("P", p_count, p_grids),
                certified_real_bound("D", d_count, d_grids),
            )
            observed = max(float(np.max(p_distance)), float(np.max(d_distance)))
            score = (certified, observed)
            if score < best_score:
                best_score, best_p = score, p_count

        return np.concatenate((p_dec[:best_p], d_dec[:n-best_p]), axis=0)
