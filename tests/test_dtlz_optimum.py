from __future__ import annotations

from typing import Callable, Tuple

import pytest

from optbench import Engine, Problem
from optbench.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6

PROBLEMS = [
    lambda objectives, parameters: (
        DTLZ1(objectives, parameters),
        parameters - objectives + 1,
        0.5,
    ),
    lambda objectives, parameters: (
        DTLZ2(objectives, parameters),
        parameters - objectives + 1,
        0.5,
    ),
    lambda objectives, parameters: (
        DTLZ3(objectives, parameters),
        parameters - objectives + 1,
        0.5,
    ),
    lambda objectives, parameters: (
        DTLZ4(objectives, parameters, 100),
        parameters - objectives + 1,
        0.5,
    ),
    lambda objectives, parameters: (
        DTLZ5(objectives, parameters),
        parameters - objectives + 1,
        0.5,
    ),
    lambda objectives, parameters: (DTLZ6(objectives), 10, 0.0),
]
PROBLEM_NAMES = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6"]


@pytest.mark.parametrize(
    "points", [10, 100, 150, 1500, 3000, 6000], ids=lambda x: f"{x}Pts"
)
@pytest.mark.parametrize("objectives", [3], ids=lambda x: f"{x}Objs")
@pytest.mark.parametrize("parameters", [4, 5, 6], ids=lambda x: f"{x}Params")
@pytest.mark.parametrize("problem_gen", PROBLEMS, ids=PROBLEM_NAMES)
def test_dtlz_set_optimum(
    points: int,
    objectives: int,
    parameters: int,
    problem_gen: Callable[[int, int], Tuple[Problem, int, int]],
):
    """
    Tests the DTLZ problem classes and the Engine by setting all values to the
    known-optimal solution, and checking whether we can confirm that our points
    lie on the Pareto-optimal front.
    """
    problem, opt_range_size, opt = problem_gen(objectives, parameters)
    engine = Engine(problem, points)

    # Set the last `k` points (x_M) to 0.5. This moves all of the points to the
    # Pareto-optimal front in objective space.
    for p, point in enumerate(engine.points):
        value = point.value
        for i in range(opt_range_size):
            value[-(i + 1)] = opt
        point.update(value)
    engine.commit()

    # Check if we know that all points are on the front.
    for i in range(points):
        assert abs(engine.optimum_dist(i)) < 0.1
