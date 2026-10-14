# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for dependency floors required by MoeaBench APIs."""

from pathlib import Path


def test_plotly_requirement_supports_font_weight():
    """Keep the packaging floor aligned with the Plotly API used by plots.

    MoeaBench interactive figures set Plotly font "weight". That property was
    introduced in Plotly 5.22, so accepting an older release makes installation
    succeed while interactive plotting can still fail at runtime.
    """
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    contents = pyproject.read_text(encoding="utf-8")

    assert '"plotly>=5.22"' in contents
