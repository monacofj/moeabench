# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from moeabench.metrics.evaluator import MetricMatrix


def test_single_run_report_does_not_infer_stability():
    matrix = MetricMatrix([[0.3], [0.2], [0.1]], metric_name="GD")

    markdown = matrix.report(show=False, markdown=True)
    text = matrix.report(show=False, markdown=False)

    assert "**StdDev**: N/A" in markdown
    assert "**Stability**: Undetermined (requires at least 2 valid runs)" in markdown
    assert "StdDev: N/A" in text
    assert "Stability: Undetermined (requires at least 2 valid runs)" in text
    assert "Stability: High" not in markdown
    assert "Stability: High" not in text


def test_report_requires_two_valid_final_run_values():
    matrix = MetricMatrix([[0.3, 0.4], [0.1, np.nan]], metric_name="GD")

    report = matrix.report(show=False, markdown=False)

    assert "Runs: 2" in report
    assert "StdDev: N/A" in report
    assert "Stability: Undetermined (requires at least 2 valid runs)" in report


def test_multi_run_report_keeps_stability_calculation():
    matrix = MetricMatrix([[0.3, 0.4], [0.10, 0.11]], metric_name="GD")

    report = matrix.report(show=False, markdown=False)

    assert "StdDev: 0.0050" in report
    assert "Stability: High (CV=0.0476 < 0.05)" in report
