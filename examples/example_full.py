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
    print("\n1. Setting up Many-Objective Problem (M=4)...")
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

    # Calibrate before plotting so clinical markers are available in topo_shape.
    print(f"Calibrating baselines for {mop.name} (M=4, K=120)...")
    mop.calibrate(size=120)

    # 2. Visual Analysis (Both fronts + GT)
    # For M=4, topo_shape projects to 3 axes for visualization.
    gt = mop.pf(n_points=3000)
    mb.view.topo_shape(
        exp1,
        exp2,
        gt,
        labels=["NSGA-II", "NSGA-III", "GT"],
        objectives=[0, 1, 2],
        title="DTLZ2 (M=4) - Fronts vs GT (f1,f2,f3 projection)",
        markers=True
    )

    # 3. Physical Metrics & Statistics
    print("\n3. Analyzing Physical Metrics...")
    igdp1 = mb.metrics.igdplus(exp1)
    igdp2 = mb.metrics.igdplus(exp2)

    print("NSGA-II IGD+ Summary:")
    igdp1.report()

    print("\nNSGA-III IGD+ Summary:")
    igdp2.report()

    # Performance Probability
    prob = mb.stats.perf_probability(igdp2, igdp1)
    print(f"\nProbability NSGA-III > NSGA-II (based on IGD+): {float(prob):.2%}")

    # 4. Stratification Analysis
    print("\n4. Stratification Analysis...")
    strata2 = mb.stats.strata(exp2)
    print("NSGA-III Stratification (Distribution):")
    for r, freq in strata2.ranks.items():
        print(f" - Rank {r}: {freq*100:.1f}% solutions")

    # EMD (Earth Mover's Distance)
    dist = mb.stats.emd(mb.stats.strata(exp1), mb.stats.strata(exp2))
    print(f"\nGlobal Topographic Displacement (EMD): {float(dist):.4f}")

    # 5. Clinical Report (FAIR Framework)
    print("\n5. Clinical Report (FAIR Framework)...")

    # Execute Analytical Multi-run Audit
    audit1 = mb.diagnostics.audit(exp1)
    audit2 = mb.diagnostics.audit(exp2)

    print("\n--- NSGA-II CLINICAL REPORT ---")
    audit1.report()

    print("\n--- NSGA-III CLINICAL REPORT ---")
    audit2.report()

    print("\n6. Synthesis complete.")

if __name__ == "__main__":
    main()
