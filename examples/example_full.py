#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench full analysis: Comprehensive capability exercise
----------------------------------------------------------
This script demonstrates the full capabilities of `moeabench` for algorithmic 
diagnostics, focusing on the comparison between NSGA-II and NSGA-III on a 
many-objective problem (DTLZ2, M=4).
"""
import mb_path
from moeabench import mb
import numpy as np

def main():
    mb.system.version()
    # 1. Setup & Experimentation
    # We use DTLZ2 with M=4 objectives.
    mop = mb.mops.DTLZ2(M=4)

    # Experiment 1: NSGA-II
    exp1 = mb.experiment()
    exp1.name = "NSGA-II"
    exp1.mop = mop
    exp1.moea = mb.moeas.NSGA2(population=120, generations=200)

    # Experiment 2: NSGA-III
    exp2 = mb.experiment()
    exp2.name = "NSGA-III"
    exp2.mop = mop
    exp2.moea = mb.moeas.NSGA3(population=120, generations=200)

    # Execute Multi-run (3 runs each for stability)
    exp1.run(repeat=3)

    exp2.run(repeat=3)

    # Calibrate before plotting so clinical markers are available in topology.
    mop.calibrate(size=120)

    # 2. Visual Analysis (Both fronts + GT)
    # For M=4, topology projects to 3 axes for visualization.
    gt = mop.pf(n_points=3000)
    mb.view.topology(
        exp1,
        exp2,
        gt,
        labels=["NSGA-II", "NSGA-III", "GT"],
        objectives=[0, 1, 2],
        title="DTLZ2 (M=4) - Fronts vs GT (f1,f2,f3 projection)",
        markers=True
    )

    # 3. Physical Metrics & Statistics
    igdp1 = mb.metrics.igdplus(exp1)
    igdp2 = mb.metrics.igdplus(exp2)

    igdp1.report()

    igdp2.report()

    # Performance Probability
    # [mb.stats.perf_win] is equivalent to mb.stats.perf_compare(method='win').
    prob = mb.stats.perf_win(igdp2, igdp1)

    # 4. Stratification Analysis
    strata2 = mb.stats.strata(exp2)

    # EMD (Earth Mover's Distance)
    dist = mb.stats.emd(mb.stats.strata(exp1), mb.stats.strata(exp2))

    # 5. Clinical Report (FAIR Framework)

    # Execute Analytical Multi-run Audit
    audit1 = mb.clinic.audit(exp1)
    audit2 = mb.clinic.audit(exp2)

    audit1.report()

    audit2.report()


if __name__ == "__main__":
    main()
