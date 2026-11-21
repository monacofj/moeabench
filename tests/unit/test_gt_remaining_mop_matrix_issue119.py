# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent read-only checks of remaining built-in GT generators (#119)."""

import numpy as np
import pytest

from scripts.audit.gt_remaining_mop_matrix_issue119 import CLASSES, audit, dtlz8_same_t_pareto_witness


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_actual_mop_generates_deterministic_finite_replay(name):
    row = audit(name, m=3, budget=64)
    assert row["deterministic"] is True
    assert row["PS_rows"] > 0 and row["PF_rows"] > 0
    assert row["unique_objective_rows"] > 0
    assert row["PS_PF_ordered_replay_max_abs"] == 0
    assert row["global_Pareto_membership_proven_by_this_audit"] is False
    assert row["continuous_PF_coverage_certified_by_this_audit"] is False


@pytest.mark.parametrize("name", ("DTLZ1", "DPF1"))
@pytest.mark.parametrize("m", [3, 5])
def test_simplex_geometry_and_finite_cardinality(name, m):
    row = audit(name, m=m, budget=128)
    assert row["simplex_sum_half_max_abs"] < 2e-13
    assert row["PS_rows"] == 128
    assert row["PF_rows"] == 128
    assert row["unique_objective_rows"] == 128
    assert row["unique_objectives_internally_dominated"] == 0


@pytest.mark.parametrize("name", ("DTLZ2", "DTLZ3", "DTLZ4"))
@pytest.mark.parametrize("m", [3, 5])
def test_sphere_geometry_includes_analytical_axis_extrema(name, m):
    row = audit(name, m=m, budget=128)
    assert row["positive_unit_sphere_max_abs"] < 2e-13
    assert row["max_axis_to_reference_distance"] < 1e-12
    assert row["unique_objective_rows"] == 128


@pytest.mark.parametrize("name", ("DTLZ7", "DPF2"))
def test_disconnected_pareto_record_region_sampling(name):
    row = audit(name, m=3, budget=128)
    assert row["record_patches"]["outside_record_domain"] == 0
    assert row["record_patches"]["observed"] == row["record_patches"]["expected"]
    assert row["unique_objectives_internally_dominated"] == 0


@pytest.mark.parametrize("name", ("DTLZ8", "DTLZ9"))
@pytest.mark.parametrize("m", [3, 5])
def test_constrained_pareto_sources_are_feasible_to_numerical_tolerance(name, m):
    row = audit(name, m=m, budget=128)
    assert row["feasibility"]["infeasible_above_1e-12"] == 0
    assert row["unique_objectives_internally_dominated"] == 0


@pytest.mark.parametrize("m", [3, 5])
def test_dtlz9_front_is_a_repeated_coordinate_quarter_circle(m):
    row = audit("DTLZ9", m=m, budget=128)
    assert row["circle_constraint_max_abs"] < 1e-12
    assert row["first_M_minus_1_objectives_equal_max_abs"] < 1e-12


def test_dpf4_records_current_projection_order_without_presuming_paper_fidelity():
    row = audit("DPF4", m=5, budget=128, intrinsic_d=2)
    assert row["published_ascending_thresholds_required"] is True
    assert row["projection_thresholds_in_ascending_order"] is False
    assert row["published_sorted_DPF4_disagreeing_rows_tol_1e_12"] > 0
    assert row["published_sorted_DPF4_max_component_delta"] > 1e-4
    # A published formula mismatch is real evidence, but independently
    # validating the new global PF and reference geometry comes next.
    assert row["global_Pareto_membership_proven_by_this_audit"] is False


@pytest.mark.parametrize("m", [3, 5, 10])
def test_dtlz8_static_source_records_actual_rows_not_requested_rows(m):
    row = audit("DTLZ8", m=m, budget=4096)
    assert row["source"] == "bundled_static_csv"
    assert row["PS_rows"] == row["PF_rows"]
    assert row["PS_rows"] <= 4096
    assert row["unique_objective_rows"] <= row["PF_rows"]


def test_invalid_dpf_dimensions_are_rejected_without_modifying_the_mop():
    with pytest.raises(ValueError, match="intrinsic dimension"):
        audit("DPF1", 3, 32, intrinsic_d=3)


def test_dtlz8_same_t_witness_is_analytically_constructive_and_dominating():
    # At t=0.5 the central point minimizes all earlier objectives.
    # At t=0.1 the lower envelope is the f1+f2=1-2t segment.
    candidate = np.array([
        [0.2, 0.3, 0.5],
        [0.3, 0.6, 0.1],
        [0.25, 0.55, 0.1],
    ])
    witness = dtlz8_same_t_pareto_witness(candidate)
    expected = np.array([
        [0.125, 0.125, 0.5],
        [0.3, 0.5, 0.1],
        [0.25, 0.55, 0.1],
    ])
    np.testing.assert_allclose(witness, expected, rtol=0, atol=3e-16)
    assert np.all(witness <= candidate + 1e-15)
    assert np.any(witness[0] < candidate[0])
    assert np.any(witness[1] < candidate[1])
    np.testing.assert_allclose(witness[2], candidate[2])
    mop=CLASSES["DTLZ8"](M=3)
    assert np.max(np.maximum(-mop._calc_constraints(witness), 0)) < 1e-14


@pytest.mark.parametrize("m", [3, 5, 10])
def test_dtlz8_independent_pareto_witness_is_feasible(m):
    row = audit("DTLZ8", m=m, budget=128)
    check = row["same_fM_independent_pareto_witness"]
    assert check["max_witness_constraint_violation"] < 2e-12
    assert check["strictly_dominated_returned_rows_tol_1e_10"] >= 0
    assert check["global_membership_proven_if_not_dominated"] is False


def test_dpf4_one_threshold_does_not_change_published_projection():
    row = audit("DPF4", m=3, budget=128, intrinsic_d=2)
    assert row["published_ascending_thresholds_required"] is True
    assert row["projection_thresholds_in_ascending_order"] is True
    assert row["published_sorted_DPF4_max_component_delta"] < 1e-12
