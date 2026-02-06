from enum import Enum, auto

class DiagnosticStatus(Enum):
    """
    Standardized algorithmic pathology classifications.
    """
    IDEAL_FRONT = auto()          # Low nGD, Low nIGD, Low nEMD
    BIASED_SPREAD = auto()        # Low nGD, Low nIGD, High nEMD
    GAPPED_COVERAGE = auto()      # Low nGD, High nIGD, Low nEMD
    COLLAPSED_FRONT = auto()      # Low nGD, High nIGD, High nEMD
    NOISY_POPULATION = auto()     # High nGD, Low nIGD, Low nEMD
    DISTORTED_COVERAGE = auto()   # High nGD, Low nIGD, High nEMD
    SHIFTED_FRONT = auto()        # High nGD, High nIGD, Low nEMD
    SEARCH_FAILURE = auto()       # High nGD, High nIGD, High nEMD
    
    SUPER_SATURATION = auto()     # H_rel > 100%
    UNDEFINED = auto()            # Insufficient data

class DiagnosticProfile(Enum):
    """
    Precision tiers for algorithmic auditing.
    Values represent thresholds as % of the reference front diameter D.
    """
    EXPLORATORY = 10.0 # Ultrapure tolerant (10% nGD, 20% nIGD, 30% nEMD)
    INDUSTRY = 1.0    # Tolerant (1% nGD, 2% nIGD, 3% nEMD)
    STANDARD = 0.5    # Balanced (0.5% nGD, 1% nIGD, 1.5% nEMD)
    RESEARCH = 0.2    # Rigorous (0.2% nGD, 0.5% nIGD, 0.8% nEMD)
