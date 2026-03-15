# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib
import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import moeabench as mb

from .fixtures import *  # noqa: F401,F403

plt.rcParams["figure.max_open_warning"] = 0
warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The 'labels' parameter of boxplot\\(\\) has been renamed",
    category=matplotlib.MatplotlibDeprecationWarning,
)


def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings",
        "ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:The 'labels' parameter of boxplot\\(\\) has been renamed.*:matplotlib.MatplotlibDeprecationWarning",
    )


@pytest.fixture(autouse=True)
def _headless_plot_defaults():
    """Force deterministic headless plotting during integration tests."""
    old_backend = getattr(mb.defaults, "backend", None)
    old_save_format = getattr(mb.defaults, "save_format", None)
    mb.defaults.backend = "matplotlib"
    mb.defaults.save_format = None
    yield
    mb.defaults.backend = old_backend
    mb.defaults.save_format = old_save_format
    plt.close("all")
