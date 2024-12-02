from __future__ import annotations

import random
from typing import Callable, List

from typing_extensions import override

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
