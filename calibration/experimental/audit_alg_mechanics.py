# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Audit: Algorithm Mechanics (Topic C)
====================================

Investigates the failure of standard MOEAs (NSGA2, MOEAD) on degenerate manifolds (e.g., DPF3).
Checks for:
1. Boundary Recession (Gap at min/max extents).
2. Population Clumping (Loss of diversity).
3. Duplicate individuals.

Usage:
    PYTHONPATH=. python3 tests/calibration/audit_alg_mechanics.py
"""

import sys
import os

# Ensure local MoeaBench is importable
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from MoeaBench.mops.DPF3 import DPF3
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

from pymoo.core.problem import Problem

class PymooWrapper(Problem):
    def __init__(self, mb_problem):
        self.mb_problem = mb_problem
        super().__init__(n_var=mb_problem.N,
                         n_obj=mb_problem.M,
                         n_ieq_constr=0,
                         xl=mb_problem.xl,
                         xu=mb_problem.xu,
                         vtype=float)

    def _evaluate(self, X, out, *args, **kwargs):
        res = self.mb_problem.evaluation(X)
        out["F"] = res["F"]
        if "G" in res:
            out["G"] = res["G"]

# --- CONFIG ---
POP_SIZE = 200
N_GEN = 1000

# ... (Callback class remains the same) ...

class AuditCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data["crowding_dist"] = []
        self.data["n_duplicates"] = []
        
    def notify(self, algorithm):
        # Access internal population details
        pop = algorithm.pop
        
        # Crowding Distance (if available)
        if hasattr(pop[0], "cd"):
            cd_vals = set([ind.cd for ind in pop])
            self.data["crowding_dist"].append(list(cd_vals))
            
        # Count Duplicates in Objective Space
        F = pop.get("F")
        u_F = np.unique(F.round(5), axis=0) # Rounding for float tolerance
        n_dups = len(F) - len(u_F)
        self.data["n_duplicates"].append(n_dups)

print(f"--- Algorithm Mechanics Audit: NSGA2 on DPF3 ---")

# 1. Setup Problem
mb_problem = DPF3()
problem = PymooWrapper(mb_problem)
print(f"Problem: DPF3, n_obj={problem.n_obj}, n_var={problem.n_var}")

# 2. Setup Algorithm
algorithm = NSGA2(pop_size=POP_SIZE)

# 3. Run with Callback
res = minimize(problem,
               algorithm,
               ('n_gen', N_GEN),
               callback=AuditCallback(),
               seed=1,
               verbose=False)

# 4. Analyze Results
print("\n--- Analysis ---")
hist = res.algorithm.callback.data

# Duplicate Tracking
dups = hist["n_duplicates"]
print(f"Duplicates (Gen 0): {dups[0]}")
print(f"Duplicates (Final): {dups[-1]}")
if dups[-1] > POP_SIZE * 0.1:
    print("WARNING: High duplicate count! Genetic drift/loss of diversity.")

# Crowding Distance Analysis (Final Gen)
final_pop = res.pop
F = final_pop.get("F")
# Manually re-calc crowding distance potentially?
# Or just check the range of the solution
print("\n[Topology Check]")
print(f"Bounds Min: {np.min(F, axis=0)}")
print(f"Bounds Max: {np.max(F, axis=0)}")
    
# Check coverage vs Ideal/Nadir
ideal = problem.pareto_front().min(axis=0) if problem.pareto_front() is not None else np.zeros(problem.n_obj)
nadir = problem.pareto_front().max(axis=0) if problem.pareto_front() is not None else np.ones(problem.n_obj)

print(f"Ideal GT: {ideal}")
print(f"Nadir GT: {nadir}")

# Metric: Extent Coverage
coverage_min = np.abs(np.min(F, axis=0) - ideal)
coverage_max = np.abs(np.max(F, axis=0) - nadir)
print(f"Gap at Min: {coverage_min}")
print(f"Gap at Max: {coverage_max}")

if np.any(coverage_min > 0.1) or np.any(coverage_max > 0.1):
    print("\nCONCLUSION: Boundary Recession detected. Algorithm failed to anchor corners.")
else:
    print("\nCONCLUSION: Boundaries are fine. Issue is internal gaps (clumping).")
