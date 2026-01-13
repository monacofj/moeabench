# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import mb_path
import MoeaBench as mb
import numpy as np

# -------------------------------------------------------------------------
# 1. Define your Custom Benchmark
# -------------------------------------------------------------------------
# To create a new problem, simply inherit from BaseBenchmark.
# You need to define __init__ (dimensions/bounds) and evaluation().

class MyProblem(mb.benchmarks.BaseBenchmark):
    """
    A simple custom bi-objective problem (ZDT1-like).
    """
    def __init__(self):
        # Initialize: 2 Objectives (M), 30 Variables (N)
        # Bounds: 0.0 to 1.0
        super().__init__(M=2, N=30, xl=0.0, xu=1.0)

    def evaluation(self, X, n_ieq_constr=0):
        # Vectorized evaluation logic
        # X shape: (population_size, N_vars)
        
        # Objective 1: Just the first variable
        f1 = X[:, 0]
        
        # Objective 2: Transformation of the others
        g = 1 + 9 * np.sum(X[:, 1:], axis=1) / (self.N - 1)
        f2 = g * (1 - np.sqrt(f1 / g))
        
        # Return dictionary with 'F' (Objectives)
        # Note: Must be formatted as a column-stack (N_samples x M_objs)
        return {'F': np.column_stack([f1, f2])}

# -------------------------------------------------------------------------
# 2. Use it in an Experiment
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running Custom Benchmark Example...")
    
    # Instantiate your custom class
    my_problem = MyProblem()

    # Configure Experiment
    exp = mb.experiment()
    exp.benchmark = my_problem
    exp.moea = mb.moeas.NSGA3(population=100, generations=50)
    exp.name = "MyCustomProblem"

    # Run
    exp.run()

    # Visualize
    # Since it's a 2-objective problem, spaceplot will handle it nicely.
    print("Optimization complete. Plotting results...")
    mb.spaceplot(exp, title="Pareto Front of MyCustomProblem")
