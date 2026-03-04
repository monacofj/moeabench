# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.tchebicheff import Tchebicheff

MOEAD_CONFIGS = {
    "DTLZ1": {
        "decomposition": PBI(eps=0.0, theta=0.5),
        "n_neighbors": 20,
    },
    "DTLZ3": {
        "decomposition": Tchebicheff(), # Robust against multimodal diversity collapse
        "n_neighbors": 30, 
    },
    "DTLZ4": {
        "decomposition": Tchebicheff(), # Neutralizes density bias
        "n_neighbors": 30, 
    },
    "DTLZ5": {
        "decomposition": Tchebicheff(), # Stable for degenerate manifolds
        "n_neighbors": 20,
    },
    "DTLZ6": {
        "decomposition": Tchebicheff(), # Forces spread on biased degenerate
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
        "decomposition": Tchebicheff(), # Prevents clustering in curved front
        "n_neighbors": 30,
    },
    "DPF4": {
        "decomposition": PBI(eps=0.0, theta=0.5),
        "n_neighbors": 20,
    },
}

def get_moead_params(mop_name):
    """
    Returns a dictionary of tuned parameters for a given MOP name.
    """
    # Use problem-specific config if available, otherwise fallback to defaults
    if mop_name in MOEAD_CONFIGS:
        return MOEAD_CONFIGS[mop_name]
    
    return {
        "decomposition": PBI(eps=0.0, theta=5.0), # Standard MOEA/D-PBI
        "n_neighbors": 15,
    }
