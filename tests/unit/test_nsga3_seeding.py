# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import moeabench as mb
from moeabench.core.run import Run
from moeabench.core.seeding import derive_component_seed
from moeabench.moeas._nsga_pymoo import REFERENCE_DIRECTIONS_COMPONENT


EMPTY_PAYLOAD = ([], [], np.empty((0, 3)), [], [], [], [])


@pytest.mark.parametrize(
    ("master_seed", "expected"),
    [(1, 1200648292), (42, 286848756), (10, 1307060349), (11, 1106432143)],
)
def test_component_seed_regression_vectors(master_seed, expected):
    assert derive_component_seed(master_seed, REFERENCE_DIRECTIONS_COMPONENT) == expected


def test_component_seed_is_stable_independent_and_component_specific():
    first = derive_component_seed(42, REFERENCE_DIRECTIONS_COMPONENT)
    assert first == derive_component_seed(42, REFERENCE_DIRECTIONS_COMPONENT)
    assert first != 42
    assert first != derive_component_seed(43, REFERENCE_DIRECTIONS_COMPONENT)
    assert first != derive_component_seed(42, "another.component")


@pytest.mark.parametrize(
    ("master_seed", "component", "exception"),
    [
        (True, REFERENCE_DIRECTIONS_COMPONENT, TypeError),
        (1.0, REFERENCE_DIRECTIONS_COMPONENT, TypeError),
        (-1, REFERENCE_DIRECTIONS_COMPONENT, ValueError),
        (1, None, TypeError),
        (1, "", ValueError),
    ],
)
def test_component_seed_rejects_invalid_inputs(master_seed, component, exception):
    with pytest.raises(exception):
        derive_component_seed(master_seed, component)


def _engine(seed=42, **kwargs):
    exp = mb.experiment(mop=mb.mops.DTLZ3(M=3))
    wrapper = mb.moeas.NSGA3(population=8, generations=1, seed=seed, **kwargs)
    return wrapper(exp, seed=seed), wrapper


def _stub_evaluation(monkeypatch, engine):
    captured = {"ref_dirs": [], "algorithm_kwargs": []}

    def fake_ref_dirs(*args, **kwargs):
        captured["ref_dirs"].append(kwargs["seed"])
        return np.ones((engine.population, engine.M))

    def fake_algorithm(**kwargs):
        captured["algorithm_kwargs"].append(kwargs)
        return object()

    monkeypatch.setattr("moeabench.moeas._nsga_pymoo.get_reference_directions", fake_ref_dirs)
    monkeypatch.setattr("moeabench.moeas._nsga_pymoo.NSGA3", fake_algorithm)
    monkeypatch.setattr(engine, "run_minimize", lambda algorithm: EMPTY_PAYLOAD)
    return captured


@pytest.mark.parametrize("override", [None, 7])
def test_wrapper_consumes_ref_dirs_seed_without_mutating_kwargs(monkeypatch, override):
    engine, _ = _engine(ref_dirs_seed=override)
    captured = _stub_evaluation(monkeypatch, engine)

    engine.evaluation()
    engine.evaluation()

    expected = derive_component_seed(42, REFERENCE_DIRECTIONS_COMPONENT) if override is None else 7
    assert captured["ref_dirs"] == [expected, expected]
    assert all("ref_dirs_seed" not in call for call in captured["algorithm_kwargs"])
    assert engine.kwargs == {"ref_dirs_seed": override}
    assert engine.component_seeds == {REFERENCE_DIRECTIONS_COMPONENT: expected}


def test_ref_dirs_override_does_not_change_seed_passed_to_minimize(monkeypatch):
    evolutionary_seeds = []

    class Result:
        F = np.empty((0, 3))

    def fake_minimize(*args, **kwargs):
        evolutionary_seeds.append(kwargs["seed"])
        return Result()

    monkeypatch.setattr("moeabench.moeas._base_pymoo.minimize", fake_minimize)
    for kwargs in ({}, {"ref_dirs_seed": None}, {"ref_dirs_seed": 7}):
        engine, _ = _engine(seed=42, **kwargs)
        engine.evaluation()

    assert evolutionary_seeds == [42, 42, 42]


@pytest.mark.parametrize("bad_seed", [True, "7", 7.0, -1, 2**32])
def test_ref_dirs_seed_rejects_invalid_values(monkeypatch, bad_seed):
    engine, _ = _engine(ref_dirs_seed=bad_seed)
    _stub_evaluation(monkeypatch, engine)
    with pytest.raises((TypeError, ValueError)):
        engine.evaluation()


def test_run_component_seeds_are_defensive_and_backward_compatible():
    run = Run(component_seeds={REFERENCE_DIRECTIONS_COMPONENT: 7})
    copied = run.component_seeds
    copied[REFERENCE_DIRECTIONS_COMPONENT] = 8
    assert run.component_seeds == {REFERENCE_DIRECTIONS_COMPONENT: 7}

    del run._component_seeds
    assert run.component_seeds == {}


def test_experiment_records_automatic_component_seed_per_run(monkeypatch):
    exp = mb.experiment(
        mop=mb.mops.DTLZ3(M=3),
        moea=mb.moeas.NSGA3(population=8, generations=1, seed=10),
    )
    monkeypatch.setattr("moeabench.moeas._nsga_pymoo.NSGA_pymoo.evaluation", lambda self: (
        setattr(self, "component_seeds", {
            REFERENCE_DIRECTIONS_COMPONENT: derive_component_seed(
                self.seed, REFERENCE_DIRECTIONS_COMPONENT
            )
        }) or EMPTY_PAYLOAD
    ))

    exp.run(repeat=2, silent=True)

    assert exp.seed == [10, 11]
    assert [run.component_seeds[REFERENCE_DIRECTIONS_COMPONENT] for run in exp] == [
        1307060349, 1106432143
    ]
    first = exp[0].component_seeds
    first[REFERENCE_DIRECTIONS_COMPONENT] = 0
    assert exp[1].component_seeds[REFERENCE_DIRECTIONS_COMPONENT] == 1106432143


def test_fixed_component_seed_report_and_persistence(monkeypatch, tmp_path):
    exp = mb.experiment(
        mop=mb.mops.DTLZ3(M=3),
        moea=mb.moeas.NSGA3(
            population=8, generations=1, seed=10, ref_dirs_seed=7
        ),
    )

    def fake_evaluation(self):
        self.component_seeds = {REFERENCE_DIRECTIONS_COMPONENT: 7}
        return EMPTY_PAYLOAD

    monkeypatch.setattr("moeabench.moeas._nsga_pymoo.NSGA_pymoo.evaluation", fake_evaluation)
    exp.run(repeat=2, silent=True)

    assert [run.component_seeds for run in exp] == [
        {REFERENCE_DIRECTIONS_COMPONENT: 7},
        {REFERENCE_DIRECTIONS_COMPONENT: 7},
    ]
    assert "Component seeds" in exp.report(show=False, markdown=False)
    assert "Component seeds" in exp.report(show=False, markdown=True)

    path = exp.save(str(tmp_path / "seeded.zip"))
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    assert metadata["context"]["runs"] == [
        {"index": 1, "seed": 10, "component_seeds": {REFERENCE_DIRECTIONS_COMPONENT: 7}},
        {"index": 2, "seed": 11, "component_seeds": {REFERENCE_DIRECTIONS_COMPONENT: 7}},
    ]

    loaded = mb.experiment()
    loaded.load(path)
    assert [run.component_seeds for run in loaded] == [
        {REFERENCE_DIRECTIONS_COMPONENT: 7},
        {REFERENCE_DIRECTIONS_COMPONENT: 7},
    ]


def test_algorithms_without_components_omit_report_section():
    exp = mb.experiment()
    exp._runs = [Run(EMPTY_PAYLOAD, seed=1, experiment=exp, index=1)]
    assert "Component seeds" not in exp.report(show=False, markdown=False)


def test_append_assigns_unique_run_indices(monkeypatch):
    exp = mb.experiment(
        mop=mb.mops.DTLZ3(M=3),
        moea=mb.moeas.NSGA3(population=8, generations=1, seed=10),
    )
    monkeypatch.setattr(
        "moeabench.moeas._nsga_pymoo.NSGA_pymoo.evaluation",
        lambda self: EMPTY_PAYLOAD,
    )
    exp.run(repeat=1, silent=True)
    exp.run(repeat=1, append=True, silent=True)
    assert [run.index for run in exp] == [1, 2]


@pytest.mark.parametrize("kwargs", [{}, {"ref_dirs_seed": 7}])
def test_short_real_nsga3_run_is_reproducible(kwargs):
    def execute():
        exp = mb.experiment(
            mop=mb.mops.DTLZ3(M=3),
            moea=mb.moeas.NSGA3(
                population=12, generations=2, seed=42, **kwargs
            ),
        )
        exp.run(silent=True)
        return exp

    first = execute()
    second = execute()

    expected = kwargs.get(
        "ref_dirs_seed",
        derive_component_seed(42, REFERENCE_DIRECTIONS_COMPONENT),
    )
    assert first[0].component_seeds == {REFERENCE_DIRECTIONS_COMPONENT: expected}
    assert second[0].component_seeds == first[0].component_seeds
    np.testing.assert_array_equal(first[0].front(), second[0].front())
