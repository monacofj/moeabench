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


def test_generic_report_uses_canonical_markdown_and_plain_projection():
    matrix = MetricMatrix([[0.3, 0.4], [0.10, 0.11]], metric_name="GD")

    markdown = matrix.report(show=False, markdown=True)
    plain = matrix.report(show=False, markdown=False)

    assert "### Metric Report: GD" in markdown
    assert "#### Final Performance (Last Gen)" in markdown
    assert "- **Mean**: 0.1050" in markdown
    assert "```" not in markdown
    assert "###" not in plain
    assert "**" not in plain
    assert "Mean: 0.1050" in plain


def test_empty_and_all_nan_reports_use_the_canonical_format():
    empty = MetricMatrix(np.empty((0, 1)), metric_name="GD")
    all_nan = MetricMatrix([[np.nan]], metric_name="GD")

    empty_markdown = empty.report(show=False, markdown=True)
    empty_plain = empty.report(show=False, markdown=False)
    nan_markdown = all_nan.report(show=False, markdown=True)
    nan_plain = all_nan.report(show=False, markdown=False)

    assert empty_markdown == "### Metric Report: GD\n**Status**: No data available"
    assert empty_plain == "Metric Report: GD\nStatus: No data available"
    assert nan_markdown == "### Metric Report: GD\n**Status**: All values are NaN"
    assert nan_plain == "Metric Report: GD\nStatus: All values are NaN"
