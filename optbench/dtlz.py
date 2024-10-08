from __future__ import annotations

from typing import Callable, List, Tuple

from typing_extensions import override

from optbench import Problem


class DTLZ1(Problem):
    """
    The DTLZ1 family of test problems, defined in Section 6.7.1.
    """

    def __init__(
        self,
        objectives: int,
        params: int,
        g: Callable[[List[float]], float] | None = None,
    ):
        self._objectives = objectives
        self._params = params
        self._target_g = g if g is not None else self._g

    def _g(self, x: List[float]) -> float:
        # Make sure the length of the vector we've been given corresponds to the
        # value `k`, as required by the definition of DTLZ1.
        assert len(x) == self._params - self._objectives + 1

        from math import cos, pi

        return 100 * (
            len(x)
            + sum([(xi - 0.5) ** 2 - cos(20 * pi * (xi - 0.5)) for xi in x])
        )

    @override
    @property
    def objectives(self) -> int:
        return self._objectives

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

        # All parameters in DTLZ1 are confined to the range 0 <= x <= 1.
        return [(0, 1)]

    @override
    def value_of(self, objective: int, params: List[float]) -> float:
        if objective >= self._objectives:
            raise IndexError(
                f"requested objective {objective} is outside the valid objective range 0 <= i < {self._objectives}"
            )
        if len(params) != self._params:
            raise ValueError(
                f"parameter list has unexpected length: expected {self._params}, got {len(params)}"
            )

        # Split the parameter vector into the part used by the objective function
        # and the part used by `g(x)`.
        lo = params[: -(self._params - self._objectives + 1)]
        hi = params[len(lo) :]
        top = self._objectives - objective - 1

        from math import prod

        a = prod(lo[:top])
        b = (1 - lo[top]) if objective > 0 else 1
        c = 1 + self._target_g(hi)

        return a * b * c / 2

    @override
    def optimum_sdf(self, objs: List[float]) -> float:
        # The Pareto-optimal surface for DTLZ1 is the hyperplane defined by
        # `sum[objs] = 0.5`. We return the signed distance to it.
        return sum(objs) - 0.5


class DTLZ2(Problem):
    """
    The DTLZ2 family of test problems, defined in Section 6.7.2.
    """

    def __init__(self, objectives: int, params: int):
        self._objectives = objectives
        self._params = params

    def _g(self, x: List[float]) -> float:
        assert len(x) == self._params - self._objectives + 1
        return sum(((val - 0.5) ** 2 for val in x))

    @override
    @property
    def objectives(self) -> int:
        return self._objectives

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

        # All parameters in DTLZ2 are confined to the range 0 <= x <= 1.
        return [(0, 1)]

    @override
    def value_of(self, objective: int, params: List[float]) -> float:
        if objective >= self._objectives:
            raise IndexError(
                f"requested objective {objective} is outside the valid objective range 0 <= i < {self._objectives}"
            )
        if len(params) != self._params:
            raise ValueError(
                f"parameter list has unexpected length: expected {self._params}, got {len(params)}"
            )

        # Split the parameter vector into the part used by the objective function
        # and the part used by `g(x)`.
        lo = params[: -(self._params - self._objectives + 1)]
        hi = params[len(lo) :]
        top = self._objectives - objective - 1

        from math import cos, pi, prod, sin

        a = prod((cos(val * pi / 2) for val in lo[:top]))
        b = sin(lo[top] * pi / 2) if objective > 0 else 1
        c = 1 + self._g(hi)

        return a * b * c

    @override
    def optimum_sdf(self, objs: List[float]) -> float:
        # The Pareto-optimal surface for DTLZ2 is the sphere defined by
        # `sum[objs]{x^2} = 1`. We return the signed distance to it.
        return sum((obj**2 for obj in objs)) - 1


class DTLZ3(DTLZ2):
    """
    The DTLZ3 family of test problems, defined in Section 6.7.3.
    """

    @override
    def _g(self, x: List[float]) -> float:
        k = self._params - self._objectives + 1
        assert len(x) == k

        from math import cos, pi

        return 100 * (
            k + sum(((xi - 0.5) ** 2 - cos(20 * pi * (xi - 0.5)) for xi in x))
        )


class DTLZ4(DTLZ2):
    """
    The DTLZ3 family of test problems, defined in Section 6.7.3.
    """

    def __init__(self, objectives: int, params: int, alpha: int):
        super().__init__(objectives, params)
        self._alpha = alpha

    @override
    def value_of(self, objective: int, params: List[float]) -> float:
        if objective >= self._objectives:
            raise IndexError(
                f"requested objective {objective} is outside the valid objective range 0 <= i < {self._objectives}"
            )
        if len(params) != self._params:
            raise ValueError(
                f"parameter list has unexpected length: expected {self._params}, got {len(params)}"
            )

        # Split the parameter vector into the part used by the objective function
        # and the part used by `g(x)`.
        lo = params[: -(self._params - self._objectives + 1)]
        hi = params[len(lo) :]
        top = self._objectives - objective - 1

        from math import cos, pi, prod, sin

        a = prod((cos(val**self._alpha * pi / 2) for val in lo[:top]))
        b = sin(lo[top] ** self._alpha * pi / 2) if objective > 0 else 1
        c = 1 + self._g(hi)

        return a * b * c


class DTLZ5(Problem):
    """
    The DTLZ5 family of test problems, defined in Section 6.7.5.
    """

    def __init__(self, objectives: int, params: int):
        self._objectives = objectives
        self._params = params

    def _g(self, x: List[float]) -> float:
        assert len(x) == self._params - self._objectives + 1
        return sum(((val - 0.5) ** 2 for val in x))

    def _theta(self, xi: float, g: float) -> float:
        from math import pi

        return pi / (4 * (1 + g)) * (1 + 2 * g * xi)

    @override
    @property
    def objectives(self) -> int:
        return self._objectives

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

        # All parameters in DTLZ2 are confined to the range 0 <= x <= 1.
        return [(0, 1)]

    @override
    def value_of(self, objective: int, params: List[float]) -> float:
        if objective >= self._objectives:
            raise IndexError(
                f"requested objective {objective} is outside the valid objective range 0 <= i < {self._objectives}"
            )
        if len(params) != self._params:
            raise ValueError(
                f"parameter list has unexpected length: expected {self._params}, got {len(params)}"
            )

        # Split the parameter vector into the part used by the objective function
        # and the part used by `g(x)`.
        lo = params[: -(self._params - self._objectives + 1)]
        hi = params[len(lo) :]
        top = self._objectives - objective - 1

        from math import cos, pi, prod, sin

        g = self._g(hi)

        a = prod((cos(self._theta(val, g) * pi / 2) for val in lo[:top]))
        b = sin(self._theta(lo[top], g) * pi / 2) if objective > 0 else 1
        c = 1 + g

        return a * b * c

    @override
    def optimum_sdf(self, objs: List[float]) -> float:
        # The Pareto-optimal surface for DTLZ2 is the sphere defined by
        # `sum[objs]{x^2} = 1`. We return the signed distance to it.
        return sum((obj**2 for obj in objs)) - 1


class DTLZ6(DTLZ5):
    """
    The DTLZ6 family of test problems, defined in Section 6.7.6.
    """

    def __init__(self, objectives: int):
        # In DTLZ6, the value of `k` is fixed, and equal to 10.
        super().__init__(objectives, 10 + objectives - 1)

    @override
    def _g(self, x: List[float]) -> float:
        assert len(x) == self._params - self._objectives + 1
        return sum((val**0.1 for val in x))
