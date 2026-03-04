# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
    UNDEFINED_BASELINE = auto()   # Missing/mismatched reference package
    UNDEFINED_INPUT = auto()      # Invalid/unsupported input (e.g., dim mismatch)
    UNDEFINED = auto()            # Insufficient data
