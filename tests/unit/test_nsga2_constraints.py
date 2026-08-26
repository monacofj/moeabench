# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import random

import numpy as np

import moeabench as mb


def _moea_with_evaluation(monkeypatch, result):
    moea = mb.moeas.NSGA2deap()
    monkeypatch.setattr(moea, "evaluation_benchmark", lambda _: result)
    return moea


def test_nsga2_feasibility_accepts_unconstrained_individual(monkeypatch):
    moea = _moea_with_evaluation(
        monkeypatch,
        {"F": np.array([[0.1, 0.2]])},
    )

    assert moea._feasible_ind([0.5, 0.5]) is True


def test_nsga2_feasibility_accepts_feasible_individual(monkeypatch):
    moea = _moea_with_evaluation(
        monkeypatch,
        {
            "F": np.array([[0.1, 0.2]]),
            "G": np.array([[-0.1]]),
            "feasible": True,
        },
    )

    assert moea._feasible_ind([0.5, 0.5]) is True


def test_nsga2_feasibility_rejects_infeasible_individual(monkeypatch):
    moea = _moea_with_evaluation(
        monkeypatch,
        {
            "F": np.array([[0.1, 0.2]]),
            "G": np.array([[0.1]]),
            "feasible": False,
        },
    )

    assert moea._feasible_ind([0.5, 0.5]) is False


def test_nsga2_dtlz8_final_front_is_feasible_and_history_is_compatible():
    random.seed(18)
    np.random.seed(18)
    generations = 20
    population = 40
    exp = mb.experiment(
        mop=mb.mops.DTLZ8(M=3),
        moea=mb.moeas.NSGA2deap(
            population=population,
            generations=generations,
            seed=7,
        ),
    )

    exp.run(silent=True)

    final_set = np.asarray(exp.last_run.set())
    assert len(final_set) > 0

    result = exp.mop.evaluation(final_set, exp.mop.get_n_ieq_constr())
    assert np.all(result["feasible"])

    assert len(exp.last_run.history("f")) == generations + 1
    assert len(exp.last_run.history("x")) == generations + 1
    assert len(exp.last_run.history("nd")) == generations + 1
    assert len(exp.last_run.history("nd_x")) == generations + 1
    assert all(front.shape[1] == exp.mop.M for front in exp.last_run.history("f"))
    assert all(decisions.shape[1] == exp.mop.N for decisions in exp.last_run.history("x"))
