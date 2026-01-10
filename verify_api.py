# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from MoeaBench import mb
import numpy as np

def verify():
    print("Verifying MoeaBench New API...")
    
    # 1. Instantiation
    print("[1] Instantiating experiment...")
    exp = mb.experiment()
    exp.benchmark = mb.benchmarks.DTLZ2()
    exp.moea = mb.moeas.NSGA3()
    
    # Configure for speed
    exp.moea.population = 10
    exp.moea.generations = 10
    
    # 2. Execution (Single)
    print("\n[2] Execution (Single Run)...")
    exp.run()
    
    print(f"    Runs executed: {len(exp)}")
    print(f"    Last run front shape: {exp.front().shape}")
    
    # 3. Execution (Multiple)
    print("\n[3] Execution (3 Runs)...")
    exp.run(3)
    print(f"    Total runs: {len(exp)}") # Should be 1 + 3 = 4
    
    # 4. Accessors
    print("\n[4] Accessors Check...")
    last_run = exp.last_run
    pop = last_run.pop()
    print(f"    Last Run Pop Size: {len(pop)}")
    print(f"    Non-dominated Front: {exp.front().shape}")
    
    # 5. Metrics
    print("\n[5] Metrics Check...")
    hv = mb.metrics.hypervolume(exp)
    print(f"    Hypervolume Matrix Shape: {hv.gens.shape}")
    
    # 6. Plotting (Dry Run)
    print("\n[6] Plotting Check (Dry Run)...")
    # This might open a window or browser, we just check callability
    try:
        mb.timeplot(hv, mode="static")
        print("    mb.timeplot(mode='static') called successfully")
    except Exception as e:
        print(f"    mb.timeplot() failed: {e}")
        
    try:
        mb.spaceplot(exp.front(), mode="static")
        print("    mb.spaceplot(mode='static') called successfully")
    except Exception as e:
        print(f"    mb.spaceplot() failed: {e}")

if __name__ == "__main__":
    verify()
