# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Problem-specific tuning registry for MOEA/D.
Maps problem names to optimal decomposition and neighborhood parameters.
"""

from pymoo.decomposition.pbi import PBI

MOEAD_CONFIGS = {
    "DTLZ1": {
        "decomposition": PBI(eps=0.0, theta=0.5),
        "n_neighbors": 20,
    },
    "DTLZ3": {
        "decomposition": PBI(eps=0.0, theta=0.2), # Balanced for convergence and spread
        "n_neighbors": 30, # Increased for diversity
    },
    "DTLZ4": {
        "decomposition": PBI(eps=0.0, theta=5.0), 
        "n_neighbors": 30, 
    },
    "DTLZ5": {
        "decomposition": PBI(eps=0.0, theta=0.5),
        "n_neighbors": 20,
    },
    "DTLZ6": {
        "decomposition": PBI(eps=0.0, theta=0.2), # Balanced for biased degenerate
        "n_neighbors": 30, 
    },
    "DPF1": {
        "decomposition": PBI(eps=0.0, theta=0.5), # Relaxed for curved front
        "n_neighbors": 30,
    },
    "DPF2": {
        "decomposition": PBI(eps=0.0, theta=0.5),
        "n_neighbors": 30,
    },
    "DPF3": {
        "decomposition": PBI(eps=0.0, theta=0.5),
        "n_neighbors": 30,
    },
    "DPF4": {
        "decomposition": PBI(eps=0.0, theta=0.5),
        "n_neighbors": 20,
    },
}

def get_moead_params(problem_name):
    """Returns specialized params or empty dict for default behavior."""
    return MOEAD_CONFIGS.get(problem_name, {})
