#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
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
# Inheriting from BaseBenchmark ensures integration with the framework.
class MyProblem(mb.benchmarks.BaseBenchmark):
    """
    A custom bi-objective problem (ZDT1-like) implemented with NumPy.
    """
    def __init__(self):
        # M=2 Objectives, N=10 Variables, Bounds=[0, 1]
        super().__init__(M=2, N=10, xl=0.0, xu=1.0)

    def evaluation(self, X, n_ieq_constr=0):
        # Evaluation MUST be vectorized (X is a population matrix)
        # Objective 1: Minimize the value of the first variable
        f1 = X[:, 0]
        
        # Objective 2: Transformation designed for a convex front
        g = 1 + 9 * np.sum(X[:, 1:], axis=1) / (self.N - 1)
        f2 = g * (1 - np.sqrt(abs(f1 / g)))
        
        # Return objectives in dictionary 'F' (pop_size x M_objs)
        return {'F': np.column_stack([f1, f2])}

def main():
    # 2. Setup: Instantiate and use the custom benchmark
    mop1 = MyProblem()

    exp1 = mb.experiment()
    exp1.name = "MyCustomProblem"
    exp1.mop = mop1
    exp1.moea = mb.moeas.NSGA2deap(population=100, generations=50)

    # 3. Execution
    print(f"Optimizing {exp1.name}...")
    exp1.run()

    # 4. Visualization: Standard tools work seamlessly with custom mop
    print("Plotting results...")
    mb.spaceplot(exp1, title="Final Front of Custom Benchmark", mode='static')

if __name__ == "__main__":
    main()

# --- Interpretation ---
#
# Creating custom problems is as simple as defining the 'evaluation' function. 
# Because MoeaBench expects a vectorized input (X), you can leverage 
# the full speed of NumPy.
#
# Once defined, your custom benchmark becomes a "first-class citizen" 
# in the library, compatible with all plotting tools and statistical measures.
