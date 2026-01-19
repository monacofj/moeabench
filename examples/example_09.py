#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 09: Selective Persistence (Save and Load)
------------------------------------------------
This example demonstrates how to save and load experiments using 
MoeaBench's selective persistence system. We explore the 'all', 
'config', and 'data' modes to manage experimental results efficiently.
"""

import mb_path
from MoeaBench import mb
import os

def main():
    print(f"Version: {mb.system.version()}")
    # 1. Setup and Run a small experiment
    print("--- Phase 1: Creating and Running Experiment ---")
    exp = mb.experiment()
    exp.name = "PersistenceStudy"
    exp.mop = mb.mops.DTLZ2(M=2)
    exp.moea = mb.moeas.NSGA2deap(population=20, generations=10)
    
    print("Executing 3 runs...")
    exp.run(repeat=3)
    
    original_runs = len(exp.runs)
    original_hv = float(mb.metrics.hv(exp.last_pop))
    print(f"Original experiment: {original_runs} runs. Final HV: {original_hv:.4f}")

    # 2. Saving in different modes
    print("\n--- Phase 2: Saving in Different Modes ---")
    
    # Save EVERYTHING
    exp.save("study_full", mode="all")
    print("Saved 'study_full.zip' (Mode: all)")
    
    # Save only the CONFIGURATION (DNA of the study)
    exp.save("study_config", mode="config")
    print("Saved 'study_config.zip' (Mode: config)")

    # 3. Selective Loading
    print("\n--- Phase 3: Selective Loading ---")
    
    # Instance A: Load FULL experiment
    exp_a = mb.experiment()
    exp_a.load("study_full", mode="all")
    print(f"Object A (all): Loaded {len(exp_a.runs)} runs. Name: {exp_a.name}")
    
    # Instance B: Load only CONFIG
    exp_b = mb.experiment()
    exp_b.load("study_config", mode="config")
    print(f"Object B (config): Loaded {len(exp_b.runs)} runs. Name: {exp_b.name}")
    
    # Instance C: Load DATA into a pre-configured object
    # This is common when you have a local script that defines the MOP 
    # but you want to fetch results from a server or previous run.
    exp_c = mb.experiment()
    exp_c.name = "LocalConfig"
    exp_c.mop = mb.mops.DTLZ2(M=2) # Pre-config context
    
    print("Loading data into Object C...")
    exp_c.load("study_full", mode="data")
    print(f"Object C (data): Loaded {len(exp_c.runs)} runs. Name remains: {exp_c.name}")

    # 4. Final Verification Plot
    print("\n--- Phase 4: Visualizing Loaded Data ---")
    mb.view.spaceplot(exp_c, title="Recovered Pareto Front (from Object C)")

    # Note: We are leaving the generated ZIP files (study_full.zip, study_config.zip) 
    # in the directory so you can inspect their internal CSV and manifest files.
    print(f"\nFiles remaining for inspection: \n - {os.path.abspath('study_full.zip')}\n - {os.path.abspath('study_config.zip')}")

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# MoeaBench's persistence system is designed for scientific reproducibility 
# and data manageability.
#
# 1. 'all' mode: Essential for deep analysis. It serializes the entire 
#    Experiment object, including the full generational history of all runs.
#
# 2. 'config' mode: Ideal for version control or sharing methodologies. 
#    It saves the "recipe" (MOP, MOEA, name) without the heavy trajectory data.
#
# 3. 'data' mode: Powerful for distributed workflows. It allows you to 
#    populate a locally configured experiment object with results generated 
#    elsewhere, without risking overwriting the local configuration settings.
