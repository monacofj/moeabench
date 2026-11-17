# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Selective persistence mode contracts (ADR 0011) using tiny real runs."""

import json
import zipfile

import numpy as np

import moeabench as mb


def _tiny_experiment(seed=17):
    exp = mb.experiment(
        mop=mb.mops.DTLZ2(M=2),
        moea=mb.moeas.NSGA2deap(population=12, generations=1, seed=seed),
    )
    exp.name = "PersistenceSource"
    exp.run(repeat=1, silent=True)
    return exp


def test_config_archive_preserves_configuration_but_omits_runs(tmp_path):
    original = _tiny_experiment()
    archive = tmp_path / "config.zip"
    original.save(str(archive), mode="config")

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        metadata = json.loads(zf.read("metadata.json"))
    assert "Moeabench.joblib" in names
    assert "result.csv" not in names
    assert "pof.csv" not in names
    assert metadata["context"]["mode"] == "config"
    assert metadata["context"]["runs"] == []

    loaded = mb.experiment()
    loaded.load(str(archive), mode="all")
    assert loaded.name == original.name
    assert loaded.mop.M == original.mop.M
    assert loaded.moea.population == original.moea.population
    assert len(loaded.runs) == 0
    assert len(original.runs) == 1  # Saving must not mutate the source.


def test_data_archive_restores_runs_without_replacing_destination_config(tmp_path):
    original = _tiny_experiment()
    archive = tmp_path / "data.zip"
    original.save(str(archive), mode="data")

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        metadata = json.loads(zf.read("metadata.json"))
    assert "result.csv" in names
    assert metadata["context"]["mode"] == "data"
    assert len(metadata["context"]["runs"]) == 1

    target = mb.experiment(
        mop=mb.mops.DTLZ2(M=2),
        moea=mb.moeas.NSGA2deap(population=8, generations=2, seed=99),
    )
    target.name = "Destination"
    target.load(str(archive), mode="data")

    assert target.name == "Destination"
    assert target.mop.M == 2
    assert target.moea.population == 8
    assert target.moea.seed == 99
    assert len(target.runs) == 1
    np.testing.assert_array_equal(target[0].front(), original[0].front())
