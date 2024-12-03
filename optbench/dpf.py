from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

from scipy.spatial import KDTree
from typing_extensions import override

from optbench import Problem
from optbench.dtlz import DTLZ1, DTLZ7


def sigmoid(x: float) -> float:
    """
    The sigmoid transfer function.
    """
    import math

    return 1 / (1 + math.pow(math.e, -x))


class DPF1(DTLZ1):
    """
    The DPF1 family of test problems, defined in Section IV-A.
    """

    def __init__(
        self,
        objectives: int,
        params: int,
        redundancies: int,
        populate: Callable[[int, int], float] = lambda i, j: random.random(),
    ):
        super().__init__(objectives, params)
        self._redundancies = redundancies

        self._vec = [0.0] * (redundancies * objectives)
        for i in range(redundancies):
            for j in range(objectives):
                self._vec[i * objectives + j] = populate(i, j)

    @override
    @property
    def objectives(self) -> int:
        return self._objectives + self._redundancies

    @override
    def value_of(self, objective: int, params: List[float]) -> float:
        if objective >= self._objectives + self._redundancies:
            raise IndexError(
                f"requested objective {objective} is outside the valid objective range 0 <= i < {self._objectives + self._redundancies}"
            )

        if objective >= self._objectives:
            # This is one of the redundancies.
            base = objective - self._objectives
            a = (self.value_of(i, params) for i in range(self._objectives))
            b = (
                self._vec[base * self._objectives + i]
                for i in range(self._objectives)
            )

            return sum((a * b for a, b in zip(a, b)))

        # This is one of the regular objectives.
        return super().value_of(objective, params)

    @override
    def optimum_sdf(self, objs: List[float]) -> float:
        # Lop off the redundant objectives.
        #
        # Theorem 1, in section III, states that the Pareto-optimal front will
        # not change from the one in DTLZ1, as h(x) is non-decreasing with
        # respect to x.
        return super().optimum_sdf(objs[: self._objectives])


class DPF2(DTLZ7):
    """
    The DPF2 family of test problems, defined in Section IV-B.
    """

    def __init__(
        self,
        objectives: int,
        params: int,
        redundancies: int,
        populate: Callable[[int, int], float] = lambda i, j: random.random(),
        transfer: Callable[[float], float] = sigmoid,
    ):
        """
        The value of `transfer` must be a non-decreasing, non-linear function.
        """
        super().__init__(objectives, params)
        self._redundancies = redundancies
        self._transfer = transfer

        self._vec = [0.0] * (redundancies * objectives)
        for i in range(redundancies):
            for j in range(objectives):
                self._vec[i * objectives + j] = populate(i, j)

    @override
    @property
    def objectives(self) -> int:
        return self._objectives + self._redundancies

    @override
    def value_of(self, objective: int, params: List[float]) -> float:
        if objective >= self._objectives + self._redundancies:
            raise IndexError(
                f"requested objective {objective} is outside the valid objective range 0 <= i < {self._objectives + self._redundancies}"
            )

        if objective >= self._objectives:
            # This is one of the redundancies.
            base = objective - self._objectives
            a = (self.value_of(i, params) for i in range(self._objectives))
            b = (
                self._vec[base * self._objectives + i]
                for i in range(self._objectives)
            )

            return self._transfer(sum((a * b for a, b in zip(a, b))))

        # This is one of the regular objectives.
        return super().value_of(objective, params)

    @override
    def optimum_sdf(self, objs: List[float]) -> float:
        # Lop off the redundant objectives.
        #
        # Theorem 1, in section III, states that the Pareto-optimal front will
        # not change from the one in DTLZ7, as h(x) should be non-decreasing
        # with respect to x.
        return super().optimum_sdf(objs[: self._objectives])


class DPF3OrDPF4(Problem):
    def __init__(
        self,
        objectives: int,
        params: int,
        redundancies: int,
        alpha: float,
        samples: int,
        height_quantum: int,
        transfer: Callable[[float], float],
        gen_eta: Callable[[int], float],
    ):
        self._objectives = objectives
        self._params = params
        self._redundancies = redundancies
        self._alpha = alpha
        self._transfer = transfer

        # Generate random numbers in the ]0;1[ range for η.
        self._vec = [max(0.0001, gen_eta(i)) for i in range(self._redundancies)]
        self._vec.sort()

        # Sample random points on the pareto-optimal front.
        assert samples > 1

        points: Dict[int, List[List[float]]] = {}

        for n in range(samples):
            # Generate a random point laying on the front.
            x = [random.random() for _ in range(self._objectives - 1)]
            point = [self._h(i, x) for i in range(self._objectives)]
            height = point[-1]

            # Filter out all the points that have the same approximate height as
            # the point we've just found and that dominate it in the other axes.
            height_key = int(height * height_quantum)

            if height_key not in points:
                points[height_key] = []
            else:
                points[height_key] = [
                    p
                    for p in points[height_key]
                    if not all((a >= b for a, b in zip(p, point)))
                ]

            dominated = False
            for p in points[height_key]:
                if all((a >= b for a, b in zip(point[:-1], p[:-1]))):
                    # This point is dominated, skip it.
                    dominated = True
                    break
            if dominated:
                continue

            points[height_key].append(point)

        print(points[54])

        pts = []
        for item in points.values():
            pts.extend(item)
        self._pts = pts
        self._tree = KDTree(pts, copy_data=True)

    def _h_base(self, objective: int, lo: List[float]) -> float:
        assert len(lo) == self._objectives - 1
        from math import cos, pi, prod, sin

        top = self._objectives - objective - 1
        a = prod((cos(val**self._alpha * pi / 2) for val in lo[:top]))
        b = sin(lo[top] ** self._alpha * pi / 2) if objective > 0 else 1

        return 1.0 - a * b

    def _h(self, objective: int, lo: List[float]) -> float:
        if objective >= self._objectives - 1:
            val = self._h_base(self._objectives - 1, lo)
            if objective == self._objectives - 1:
                return self._transfer(min(val, self._vec[0]))
            if objective == self._objectives + self._redundancies - 1:
                return self._transfer(max(val, self._vec[-1]))

            curr = objective - self._objectives
            nxt = curr + 1

            return self._transfer(min(max(val, curr), nxt))

        return self._transfer(self._h_base(objective, lo))

    def _g(self, x: List[float]) -> float:
        raise NotImplementedError()

    @override
    @property
    def objectives(self) -> int:
        return self._objectives + self._redundancies

    @override
    @property
    def params(self) -> int:
        return self._params

    @override
    def range_of(self, param: int) -> List[Tuple[float, float]]:
        if param >= self._params:
            # Not strictly necessary here, but still good not to fail silently.
            raise IndexError(
                f"requested parameter {param} is outside the valid parameter range 0 <= i < {self._params}"
            )
        return [(0, 1)]

    @override
    def value_of(self, objective: int, params: List[float]) -> float:
        if objective >= self._objectives + self._redundancies:
            raise IndexError(
                f"requested objective {objective} is outside the valid objective range 0 <= i < {self._objectives + self._redundancies}"
            )
        if len(params) != self._params:
            raise ValueError(
                f"parameter list has unexpected length: expected {self._params}, got {len(params)}"
            )

        # Split the parameter vector into the part used by the objective function
        # and the part used by `g(x)`.
        lo = params[: -(self._params - self._objectives + 1)]
        hi = params[len(lo) :]

        g = self._g(hi)
        return (1 + g) * self._h(objective, lo)

    @override
    def optimum_sdf(self, objs: List[float]) -> float:
        nearest = self._tree.query(objs[: self._objectives])[0]

        assert isinstance(nearest, float)
        return nearest


class DPF3(DPF3OrDPF4):
    def __init__(
        self,
        objectives: int,
        params: int,
        redundancies: int,
        samples: int = 5000,
        height_quantum: int = 100,
        gen_eta: Callable[[int], float] = lambda _: random.random(),
    ):
        super().__init__(
            objectives,
            params,
            redundancies,
            100.0,
            samples,
            height_quantum,
            lambda x: x,
            gen_eta,
        )

    @override
    def _g(self, x: List[float]) -> float:
        assert len(x) == self._params - self._objectives + 1
        return sum(((val - 0.5) ** 2 for val in x))


class DPF4(DPF3OrDPF4):
    def __init__(
        self,
        objectives: int,
        params: int,
        redundancies: int,
        transfer: Callable[[float], float] = sigmoid,
        samples: int = 5000,
        height_quantum: int = 100,
        gen_eta: Callable[[int], float] = lambda _: random.random(),
    ):
        super().__init__(
            objectives,
            params,
            redundancies,
            1.0,
            samples,
            height_quantum,
            transfer,
            gen_eta,
        )

    @override
    def _g(self, x: List[float]) -> float:
        assert len(x) == self._params - self._objectives + 1
        from math import cos, pi

        return 100 * (
            len(x)
            + sum(((xi - 0.5) ** 2 - cos(20 * pi * (xi - 0.5)) for xi in x))
        )
