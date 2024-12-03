from __future__ import annotations

from typing import Callable, Tuple

import pytest

from optbench import Engine, Problem
from optbench.dpf import DPF1

PROBLEMS = [
    lambda objectives, parameters, redundancies: (
        DPF1(objectives, parameters, redundancies),
        parameters - objectives + 1,
        0.5,
    ),
]
PROBLEM_NAMES = ["DPF1"]


@pytest.mark.parametrize(
    "points", [10, 100, 150, 1500, 3000, 6000], ids=lambda x: f"{x}Pts"
)
@pytest.mark.parametrize("objectives", [3], ids=lambda x: f"{x}Objs")
@pytest.mark.parametrize("parameters", [4, 5, 6], ids=lambda x: f"{x}Params")
@pytest.mark.parametrize("redundancies", [4, 5, 6], ids=lambda x: f"{x}Reds")
@pytest.mark.parametrize("problem_gen", PROBLEMS, ids=PROBLEM_NAMES)
def test_dpf1_2_set_optimum(
    points: int,
    objectives: int,
    parameters: int,
    redundancies: int,
    problem_gen: Callable[[int, int, int], Tuple[Problem, int, int]],
):
    """
    Tests the DPF problem classes and the Engine by setting all values to the
    known-optimal solution, and checking whether we can confirm that our points
    lie on the Pareto-optimal front.
    """
    problem, opt_range_size, opt = problem_gen(
        objectives, parameters, redundancies
    )
    engine = Engine(problem, points)

    # Set the last `k` points (x_M) to opt. This moves all of the points to the
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
