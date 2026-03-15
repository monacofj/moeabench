# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np

import moeabench as mb


def test_persistence_pipeline(paired_experiments, tmp_path):
    exp1, _ = paired_experiments
    path = tmp_path / "integration_exp.zip"

    exp1.save(str(path))
    assert path.exists()

    restored = mb.experiment()
    restored.load(str(path))

    assert restored.name == exp1.name
    assert len(restored.runs) == len(exp1.runs)
    assert restored.mop.name == exp1.mop.name
    assert np.allclose(restored.front(), exp1.front())
