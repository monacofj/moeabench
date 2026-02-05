from enum import Enum, auto

class DiagnosticStatus(Enum):
    """
    Standardized algorithmic pathology classifications.
    """
    IDEAL_FRONT = auto()          # Low GD, Low IGD, Low EMD
    BIASED_SPREAD = auto()        # Low GD, Low IGD, High EMD
    GAPPED_COVERAGE = auto()      # Low GD, High IGD, Low EMD
    COLLAPSED_FRONT = auto()      # Low GD, High IGD, High EMD
    NOISY_POPULATION = auto()     # High GD, Low IGD, Low EMD
    DISTORTED_COVERAGE = auto()   # High GD, Low IGD, High EMD
    SHIFTED_FRONT = auto()        # High GD, High IGD, Low EMD
    SEARCH_FAILURE = auto()       # High GD, High IGD, High EMD
    
    SUPER_SATURATION = auto()     # H_rel > 100%
    UNDEFINED = auto()            # Insufficient data
