# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent mathematical checks for the genuine-P0 longitudinal pilot."""

import numpy as np
import pytest

from calibration.scripts.audit_longitudinal_pilot import (
    independent_headway, run_pilot,
)


def test_independent_headway_matches_known_perpendicular_displacements():
    gt = np.column_stack([
        np.linspace(0, 1, 21), 1-np.linspace(0, 1, 21), np.zeros(21)
    ])
    p0 = gt[::4] + np.array([0,0,0.1])
    pf = gt[::4] + np.array([0,0,0.025])
    assert independent_headway(p0, pf, gt) == pytest.approx(0.25)


def test_longitudinal_pilot_preserves_genuine_initial_and_repeats_seed():
    first = run_pilot(seed=7, population=16, generations=8)
    second = run_pilot(seed=7, population=16, generations=8)
    assert first == second
    assert first["headway_validated_from_true_P0"] is True
    assert first["sample_sizes"]["P0"] == 16
    assert first["provenance"]["P0_sha256"] != first["provenance"]["Pf_sha256"]
    assert first["physical_metrics"]["HEADWAY"] == pytest.approx(
        first["independent_headway"]
    )
    assert first["status"] == "provisional_longitudinal_pilot_not_certified"
