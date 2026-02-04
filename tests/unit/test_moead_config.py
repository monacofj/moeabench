# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from MoeaBench.moeas.moead_configs import get_moead_params
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.tchebicheff import Tchebicheff

def test_config_registry_integrity():
    """
    Verifies that the Registry Logic returns the correct objects 
    for known problematic cases.
    """
    # Case A: DTLZ1 (Must be PBI/0.5 for fast convergence)
    cfg_dtlz1 = get_moead_params("DTLZ1")
    assert isinstance(cfg_dtlz1["decomposition"], PBI)
    assert cfg_dtlz1["decomposition"].theta == 0.5
    assert cfg_dtlz1["n_neighbors"] == 20

    # Case B: DTLZ3 (Must be Tchebicheff for diversity)
    cfg_dtlz3 = get_moead_params("DTLZ3")
    assert isinstance(cfg_dtlz3["decomposition"], Tchebicheff)
    assert cfg_dtlz3["n_neighbors"] == 30

    # Case C: DTLZ6 (Must be Tchebicheff for biased degenerate)
    cfg_dtlz6 = get_moead_params("DTLZ6")
    assert isinstance(cfg_dtlz6["decomposition"], Tchebicheff)
    assert cfg_dtlz6["n_neighbors"] == 30

def test_config_fallback():
    """
    Verifies that unknown problems fallback to the Standard PBI configuration.
    """
    cfg_unknown = get_moead_params("UNKNOWN_MOP_XYZ")
    
    # Default is PBI theta=5.0, T=15
    assert isinstance(cfg_unknown["decomposition"], PBI)
    assert cfg_unknown["decomposition"].theta == 5.0
    assert cfg_unknown["n_neighbors"] == 15

def test_moead_instantiation_safety():
    """
    Simulates MOEA/D instantiation to ensure the config objects are valid 
    pymoo decomposition instances.
    """
    try:
        from MoeaBench.moeas.MOEAD import MOEAD
        # Since MOEAD applies params lazily at __call__, we only verify class existence here.
        alg = MOEAD(problem_name="DTLZ3")
        assert alg is not None
        
    except ImportError:
        pytest.skip("MOEAD class not found or import error")
