# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression for the audit report's required Headway source data."""

from calibration.scripts.audit_report_preflight import inspect_inputs


def test_audit_report_preflight_requires_actual_generation_zero(tmp_path):
    data = tmp_path / "runs"
    gt = tmp_path / "gt"
    data.mkdir()
    gt.mkdir()
    baseline = tmp_path / "baselines.csv"

    (data / "DTLZ1_NSGA2_standard_run00.csv").write_text("f1,f2,f3\n")
    (data / "DTLZ1_NSGA2_standard_run00_gen100.csv").write_text("f1,f2,f3\n")
    (gt / "DTLZ1_3_optimal.csv").write_text("0,0,0\n")

    result = inspect_inputs(data, gt, baseline)
    assert result["headway_data_ready_runs"] == 0
    assert result["report_generation_safe"] is False
    assert result["runs"][0]["earliest_saved_generation"] == 100

    (data / "DTLZ1_NSGA2_standard_run00_gen0.csv").write_text("f1,f2,f3\n")
    baseline.write_text("MOP,Algorithm\n")
    after = inspect_inputs(data, gt, baseline)
    assert after["headway_data_ready_runs"] == 1
    assert after["report_generation_safe"] is True
