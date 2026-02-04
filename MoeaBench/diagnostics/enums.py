from enum import Enum, auto

class DiagnosticStatus(Enum):
    """
    Standardized algorithmic pathology classifications.
    """
    OPTIMAL = auto()                 # Balanced convergence and diversity
    DIVERSITY_COLLAPSE = auto()      # Good GD, Poor IGD/H_rel
    CONVERGENCE_FAILURE = auto()     # Poor GD, Poor IGD
    TOPOLOGICAL_DISTORTION = auto()  # High EMD, potentially good H_rel
    SUPER_SATURATION = auto()        # H_rel > 100% (Resolution artifacts)
    UNDEFINED = auto()               # Insufficient data
