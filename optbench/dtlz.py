from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

from scipy.spatial import KDTree
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
        sphere = sum((obj**2 for obj in objs)) - 1
        slices = max((-obj for obj in objs))
        return max(sphere, slices)


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

    def __init__(self, objectives: int, params: int, samples: int = 1000):
        self._objectives = objectives
        self._params = params
        assert samples > 0

        pts: List[List[float]] = []

        # The sampling for DTLZ5 is made in accordance with Tian, Y., Xiang, X.,
        # Zhang, X., Cheng, R., & Jin, Y. (2018). Sampling Reference Points on
        # the Pareto Fronts of Benchmark Multi-Objective Optimization Problems.
        # 2018 IEEE Congress on Evolutionary Computation (CEC).
        # doi:10.1109/cec.2018.8477730
        from math import cos, pi, sin, sqrt

        for n in range(samples):
            x = random.random()
            v = 1 / sqrt(2)
            point = [
                pow(v, objectives - max(i + 1, 2)) * cos(pi / 2 * x)
                for i in range(objectives - 1)
            ]
            point.append(sin(pi / 2 * x))

            pts.append(point)

        self._pts = pts
        self._tree = KDTree(pts, copy_data=True)

    def _g(self, x: List[float]) -> float:
        assert len(x) == self._params - self._objectives + 1
        return sum(((val - 0.5) ** 2 for val in x))

    def _theta(self, xi: float, g: float, index: int) -> float:
        if index == 0:
            return xi
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

        a = prod(
            (
                cos(self._theta(val, g, i) * pi / 2)
                for i, val in enumerate(lo[:top])
            )
        )
        b = sin(self._theta(lo[top], g, top) * pi / 2) if objective > 0 else 1
        c = 1 + g

        return a * b * c

    @override
    def optimum_sdf(self, objs: List[float]) -> float:
        nearest = self._tree.query(objs)[0]

        assert isinstance(nearest, float)
        return nearest


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


class DTLZ7(Problem):
    """
    The DTLZ7 family of test problems, defined in Section 6.7.7.
    """

    def __init__(self, objectives: int, params: int, sampling: int = 100):
        self._objectives = objectives
        self._params = params

        # DTLZ7 is special in that there's no decent analytic version of its
        # signed distance function, and so we have to generate a point cloud and
        # check against it.
        #
        # We know that all points in on the front have Xm=0, and so, we can
        # determine that g* = 1.
        assert sampling > 1

        points: Dict[int, List[List[float]]] = {}

        gap = 1 / (sampling - 1)
        for n in range(sampling ** (objectives - 1)):
            point = [
                ((n // (sampling**i)) % sampling) * gap
                for i in range(objectives - 1)
            ]
            height = self._h(point, 1) * 2
            point.append(height)

            # Filter out all the points that have the same approximate height as
            # the point we've just found and that dominate it in the other axes.
            height_key = int(height * sampling)

            if height_key not in points:
                points[height_key] = []
            else:
                points[height_key] = [
                    p
                    for p in points[height_key]
                    if not all((a >= b for a, b in zip(p[:-1], point[:-1])))
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

        pts = []
        for item in points.values():
            pts.extend(item)
        self._pts = pts
        self.tree = KDTree(pts, copy_data=True)

    def _h(self, params: List[float], g: float) -> float:
        assert len(params) == self._objectives - 1

        # The sum actually happens over coordinates in the objective space, and
        # not over the input vectors. Thankfully, in DTLZ7, all of the input
        # axes that we care about are identity-mapped to the corresponding axes
        # in the objective space.
        from math import pi, sin

        s = sum(((x * (1 + sin(3 * pi * x))) for x in params)) / (1 + g)

        return self._objectives - s

    def _g(self, x: List[float]) -> float:
        assert len(x) == self._params - self._objectives + 1
        return 1 + 9 / len(x) * sum(x)

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

        # All parameters in DTLZ7 are confined to the range 0 <= x <= 1.
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

        if objective < self._objectives - 1:
            # Identity-mapped from 0 to objectives-2
            return params[objective]

        # Split the parameter vector into the part used by the objective function
        # and the part used by `g(x)`.
        lo = params[: -(self._params - self._objectives + 1)]
        hi = params[len(lo) :]

        g = self._g(hi)
        return (1 + g) * self._h(lo, g)

    @override
    def optimum_sdf(self, objs: List[float]) -> float:
        nearest = self.tree.query(objs)[0]

        assert isinstance(nearest, float)
        return nearest
