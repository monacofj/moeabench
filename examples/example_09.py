#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 09: Serialization and Persistence of Experimental Data
------------------------------------------------
This example demonstrates how to save and load experiments using 
moeabench's selective persistence system. We explore the 'all', 
'config', and 'data' modes to manage experimental results efficiently.
"""

import mb_path
from moeabench import mb
import os

def main():
    mb.system.version()
    # 1. Setup and Run a small experiment
    exp = mb.experiment()
    exp.name = "PersistenceStudy"
    exp.mop = mb.mops.DTLZ2(M=2)
    exp.moea = mb.moeas.NSGA2deap(population=20, generations=10)
    
    exp.run(repeat=3)
    
    original_runs = len(exp.runs)
    original_hv = float(mb.metrics.hv(exp.last_pop))

    # 2. Saving in different modes
    
    # Save EVERYTHING
    exp.save("study_full", mode="all")
    
    # Save only the CONFIGURATION (DNA of the study)
    exp.save("study_config", mode="config")

    # 3. Selective Loading
    
    # Instance A: Load FULL experiment
    exp_a = mb.experiment()
    exp_a.load("study_full", mode="all")
    
    # Instance B: Load only CONFIG
    exp_b = mb.experiment()
    exp_b.load("study_config", mode="config")
    
    # Instance C: Load DATA into a pre-configured object
    # This is common when you have a local script that defines the MOP 
    # but you want to fetch results from a server or previous run.
    exp_c = mb.experiment()
    exp_c.name = "LocalConfig"
    exp_c.mop = mb.mops.DTLZ2(M=2) # Pre-config context
    
    exp_c.load("study_full", mode="data")

    # 4. Final Verification Plot
    mb.view.topology(exp_c, title="Recovered Pareto Front (from Object C)")

    # Note: We are leaving the generated ZIP files (study_full.zip, study_config.zip) 
    # in the directory so you can inspect their internal CSV and manifest files.

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# moeabench's persistence system is designed for scientific reproducibility 
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
