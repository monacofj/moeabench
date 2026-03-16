# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest


SMOKE_UNIT_FILES = {
    "tests/unit/test_api_surface.py",
    "tests/unit/test_compare_aliases.py",
    "tests/unit/test_system.py",
}

SMOKE_INTEGRATION_FILES = {
    "tests/integration/test_topology_gt_protocol.py",
    "tests/integration/test_view_dispatch_pipeline.py",
}

BASIC_UNIT_ROOT_FILES = {
    "tests/test_golden_dtlz6.py",
    "tests/test_light_tier.py",
    "tests/test_mop_calibration.py",
}

STABILITY_BASIC_FILES = {
    "tests/test_stability_basic.py",
}

STABILITY_SMOKE_FILES = {
    "tests/test_stability_smoke.py",
}

STABILITY_DEEP_FILES = {
    "tests/test_stability_deep.py",
}

TEST_RANDOM_SEED = 20260316


def _resolve_scope(rel_path: str) -> str:
    if rel_path.startswith("tests/unit/") or rel_path in BASIC_UNIT_ROOT_FILES:
        return "unit"
    if rel_path.startswith("tests/integration/"):
        return "integration"
    if rel_path in STABILITY_SMOKE_FILES | STABILITY_BASIC_FILES | STABILITY_DEEP_FILES:
        return "stability"
    raise RuntimeError(f"Unclassified test scope for {rel_path}")


def _resolve_level(rel_path: str) -> str:
    if rel_path in STABILITY_DEEP_FILES:
        return "deep"
    if rel_path in SMOKE_UNIT_FILES | SMOKE_INTEGRATION_FILES | STABILITY_SMOKE_FILES:
        return "smoke"
    return "basic"


def pytest_collection_modifyitems(config, items):
    root = Path(config.rootpath)
    for item in items:
        rel_path = str(Path(item.fspath).resolve().relative_to(root))
        scope = _resolve_scope(rel_path)
        level = _resolve_level(rel_path)
        item.add_marker(getattr(pytest.mark, f"scope_{scope}"))
        item.add_marker(getattr(pytest.mark, f"level_{level}"))


@pytest.fixture(autouse=True)
def _deterministic_test_rng():
    """Reset process-global RNG state before each test."""
    random.seed(TEST_RANDOM_SEED)
    np.random.seed(TEST_RANDOM_SEED)
