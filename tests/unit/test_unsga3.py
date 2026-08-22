# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

import moeabench as mb
from moeabench.core.seeding import derive_component_seed
from moeabench.moeas._unsga_pymoo import REFERENCE_DIRECTIONS_COMPONENT


EMPTY_PAYLOAD = ([], [], np.empty((0, 3)), [], [], [], [])


def _engine(seed=42, **kwargs):
    exp = mb.experiment(mop=mb.mops.DTLZ7(M=3))
    wrapper = mb.moeas.U_NSGA3(
        population=12, generations=1, seed=seed, **kwargs
    )
    return wrapper(exp, seed=seed)


def test_unsga3_supplies_reference_directions_and_consumes_override(monkeypatch):
    engine = _engine(ref_dirs_seed=7)
    captured = {}

    def fake_reference_directions(*args, **kwargs):
        captured["reference_seed"] = kwargs["seed"]
        return np.ones((engine.population, engine.M))

    def fake_unsga3(**kwargs):
        captured["algorithm_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        "moeabench.moeas._unsga_pymoo.get_reference_directions",
        fake_reference_directions,
    )
    monkeypatch.setattr("moeabench.moeas._unsga_pymoo.UNSGA3", fake_unsga3)
    monkeypatch.setattr(engine, "run_minimize", lambda algorithm: EMPTY_PAYLOAD)

    engine.evaluation()

    assert captured["reference_seed"] == 7
    assert captured["algorithm_kwargs"]["ref_dirs"].shape == (12, 3)
    assert "ref_dirs_seed" not in captured["algorithm_kwargs"]
    assert engine.kwargs == {"ref_dirs_seed": 7}
    assert engine.component_seeds == {REFERENCE_DIRECTIONS_COMPONENT: 7}


def test_unsga3_automatically_derives_distinct_reproducible_seed(monkeypatch):
    engine = _engine(seed=42)
    captured = []

    def fake_reference_directions(*args, **kwargs):
        captured.append(kwargs["seed"])
        return np.ones((engine.population, engine.M))

    monkeypatch.setattr(
        "moeabench.moeas._unsga_pymoo.get_reference_directions",
        fake_reference_directions,
    )
    monkeypatch.setattr(
        "moeabench.moeas._unsga_pymoo.UNSGA3", lambda **kwargs: object()
    )
    monkeypatch.setattr(engine, "run_minimize", lambda algorithm: EMPTY_PAYLOAD)

    engine.evaluation()
    engine.evaluation()

    expected = derive_component_seed(42, REFERENCE_DIRECTIONS_COMPONENT)
    assert captured == [expected, expected]
    assert expected != 42


@pytest.mark.parametrize("bad_seed", [True, "7", 7.0, -1, 2**32])
def test_unsga3_rejects_invalid_reference_direction_seed(monkeypatch, bad_seed):
    engine = _engine(ref_dirs_seed=bad_seed)
    monkeypatch.setattr(engine, "run_minimize", lambda algorithm: EMPTY_PAYLOAD)
    with pytest.raises((TypeError, ValueError)):
        engine.evaluation()


@pytest.mark.parametrize("ref_dirs_seed", [None, 7])
def test_short_real_unsga3_dtlz7_run(ref_dirs_seed):
    kwargs = {} if ref_dirs_seed is None else {"ref_dirs_seed": ref_dirs_seed}
    exp = mb.experiment(
        mop=mb.mops.DTLZ7(M=3),
        moea=mb.moeas.U_NSGA3(
            population=12, generations=2, seed=42, **kwargs
        ),
    )

    exp.run(silent=True)

    assert len(exp) == 1
    expected = (
        derive_component_seed(42, REFERENCE_DIRECTIONS_COMPONENT)
        if ref_dirs_seed is None
        else ref_dirs_seed
    )
    assert exp[0].component_seeds == {REFERENCE_DIRECTIONS_COMPONENT: expected}
    assert len(exp[0]) == 2
