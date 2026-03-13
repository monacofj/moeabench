#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 14: Multivariate Diagnostic Visualization and Instrument Suite
---------------------------------------------------------------
This example demonstrates the 4 primary diagnostic instruments:
1. clinic_ecdf: Goal attainment (The Judge)
2. clinic_distribution: Physics of error (The Pathologist)
3. radar: Quality fingerprint (The Validation)
4. clinic_history: Health trajectory (The Monitor)

Scenario: We simulate a "Premature Convergence" search where the algorithm
is close to the front but highly clustered.
"""

import mb_path
import numpy as np
import matplotlib.pyplot as plt
import moeabench as mb

def main():
    mb.system.version()

    # 1. Setup Benchmark and Scenario
    mop = mb.mops.DTLZ2(M=3)
    gt = mop.pf(n_points=500)
    
    exp1 = mb.experiment()
    exp1.mop = mop
    exp1.moea = mb.moeas.NSGA2(population=100, generations=30)
    exp1.name = "Baseline NSGA-II"
    exp1.run()
    
    # 2. Instrument 1: The Radar (Clinical Fingerprint)
    mb.view.radar(exp1, ground_truth=gt)

    # 4. Instrument 2: The ECDF (The Judge)
    mb.view.ecdf(exp1, ground_truth=gt, metric="closeness")
    
    mb.view.ecdf(exp1, ground_truth=gt, metric="coverage")

    # 5. Instrument 3: The Distribution (The Pathologist)
    mb.view.density(exp1, domain='clinic', ground_truth=gt, metric="closeness")
    
    mb.view.density(exp1, domain='clinic', ground_truth=gt, metric="coverage")

    # 6. Instrument 4: The History (The Monitor)
    mb.view.history(exp1, domain='clinic', ground_truth=gt, metric="closeness")
    
    mb.view.history(exp1, domain='clinic', ground_truth=gt, metric="coverage")


if __name__ == "__main__":
    main()
