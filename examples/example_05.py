#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 05: Custom Benchmark Implementation
------------------------------------------
This example demonstrates how to extend MoeaBench by creating your 
own vectorized multi-objective problem (MOP), allowing you to optimize 
custom domains with the same analytical tools.
"""

import mb_path
from MoeaBench import mb
import numpy as np

# 1. Component: Custom Problem Logic
# Inheriting from BaseMop ensures integration with the framework.
class MyProblem(mb.mops.BaseMop):
    """
    A custom bi-objective problem (ZDT1-like) implemented with NumPy.
    """
    def __init__(self):
        # M=3 Objectives, N=10 Variables, Bounds=[0, 1]
        # Providing a 'name' enables Sidecar Diagnostic generation (v0.9+)
        super().__init__(name="MyCustomProblem_3D_v1", M=3, N=10, xl=0.0, xu=1.0)

    def evaluation(self, X, n_ieq_constr=0):
        # Evaluation MUST be vectorized (X is a population matrix)
        # Objectives are mapped to a spherical surface (DTLZ2-style)
        theta = X[:, :2] * (np.pi / 2)
        g = np.sum((X[:, 2:] - 0.5)**2, axis=1)
        
        f1 = (1 + g) * np.cos(theta[:, 0]) * np.cos(theta[:, 1])
        f2 = (1 + g) * np.cos(theta[:, 0]) * np.sin(theta[:, 1])
        f3 = (1 + g) * np.sin(theta[:, 0])
        
        # Return objectives in dictionary 'F' (pop_size x 3)
        return {'F': np.column_stack([f1, f2, f3])}

    def ps(self, n):
        """Analytical Pareto Set (Decision Variables) used for Ground Truth."""
        # For M=3, the Pareto Set is a 2D manifold (x1, x2 in [0,1], x_others=0.5)
        sqrt_n = int(np.sqrt(n))
        x1, x2 = np.meshgrid(np.linspace(0, 1, sqrt_n), np.linspace(0, 1, sqrt_n))
        x1, x2 = x1.flatten(), x2.flatten()
        
        res = np.zeros((len(x1), self.N))
        res[:, 0] = x1
        res[:, 1] = x2
        res[:, 2:] = 0.5
        return res

def main():
    print(f"MoeaBench v{mb.system.version()}")
    # 2. Setup: Instantiate and use the custom benchmark
    mop1 = MyProblem()
    
    # 2.1 Calibration: "One-Click" Plugin registration
    # This generates/loads a sidecar JSON with baselines for clinical diagnostics.
    # We use the default grid to support varying population sizes.
    mop1.calibrate()

    exp1 = mb.experiment()
    exp1.name = "MyCustom3DProblem"
    exp1.mop = mop1
    exp1.moea = mb.moeas.NSGA2deap(population=200, generations=100)

    # 3. Execution
    print(f"Optimizing {exp1.name}...")
    exp1.run()

    # 4. Visualization: Standard tools work seamlessly with custom mop
    print("Plotting results...")
    mb.view.topo_shape(exp1, title="Final Front of Custom Benchmark")
    
    # 5. Scientific Validation (v0.9+): Clinical Radar
    # This now works because we called mop1.calibrate()
    print("Generating clinical diagnostic...")
    mb.view.clinic_radar(exp1, title="Clinical Performance Profile (Custom MOP)")

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# Creating custom problems is as simple as defining the 'evaluation' function. 
# Because MoeaBench expects a vectorized input (X), you can leverage 
# the full speed of NumPy.
#
# Once defined, your custom mops becomes a "first-class citizen" 
# in the library, compatible with all plotting tools and statistical measures.
