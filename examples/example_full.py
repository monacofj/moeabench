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
    print(f"MoeaBench version: {mb.system.version()}")

    # 1. Setup & Experimentation
    # We use DTLZ2 with M=4 objectives.
    print("\n1. Setting up Many-Objective Problem (M=4)...")
    mop = mb.mops.DTLZ2(M=4)

    # Experiment 1: NSGA-II
    exp2 = mb.experiment()
    exp2.name = "NSGA-II"
    exp2.mop = mop
    exp2.moea = mb.moeas.NSGA2(population=120, generations=200)

    # Experiment 2: NSGA-III
    exp3 = mb.experiment()
    exp3.name = "NSGA-III"
    exp3.mop = mop
    exp3.moea = mb.moeas.NSGA3(population=120, generations=200)

    # Execute Multi-run (3 runs each for stability)
    print("Running NSGA-II...")
    exp2.run(repeat=3)

    print("Running NSGA-III...")
    exp3.run(repeat=3)

    # 2. Physical Metrics & Statistics
    print("\n2. Analyzing Physical Metrics...")
    igdp2 = mb.metrics.igdplus(exp2)
    igdp3 = mb.metrics.igdplus(exp3)

    print("NSGA-II IGD+ Summary:")
    igdp2.report()

    print("\nNSGA-III IGD+ Summary:")
    igdp3.report()

    # Performance Probability
    prob = mb.stats.perf_probability(igdp3, igdp2)
    print(f"\nProbability NSGA-III > NSGA-II (based on IGD+): {float(prob):.2%}")

    # 3. Visual Analysis
    # (Visualization requires a display environment; skipping interactive plots in script)
    # But we can still compute them or use non-blocking versions if needed.
    print("\n3. Visual Analysis (Skipping interactive plots in script)...")

    # 4. Stratification Analysis
    print("\n4. Stratification Analysis...")
    strata3 = mb.stats.strata(exp3)
    print("NSGA-III Stratification (Distribution):")
    for r, freq in strata3.ranks.items():
        print(f" - Rank {r}: {freq*100:.1f}% solutions")

    # EMD (Earth Mover's Distance)
    dist = mb.stats.emd(mb.stats.strata(exp2), mb.stats.strata(exp3))
    print(f"\nGlobal Topographic Displacement (EMD): {float(dist):.4f}")

    # 5. Clinical Report (FAIR Framework)
    print("\n5. Clinical Report (FAIR Framework)...")
    
    # Calibrate clinical baselines for M=4 main scenario
    print(f"Calibrating baselines for {mop.name} (M=4, K=120)...")
    mb.calibrate(mop, size=120, force=True)

    # Execute Analytical Multi-run Audit
    audit2 = mb.diagnostics.audit(exp2)
    audit3 = mb.diagnostics.audit(exp3)

    print("\n--- NSGA-II CLINICAL REPORT ---")
    audit2.report()

    print("\n--- NSGA-III CLINICAL REPORT ---")
    audit3.report()

    print("\n6. Synthesis complete.")

if __name__ == "__main__":
    main()
