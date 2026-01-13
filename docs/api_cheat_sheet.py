# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import MoeaBench as mb

## API Cheat Sheet
## This script is for documentation purposes only. Do not execute it.

# ---------------------------------------------------------
# 1. Experiment Configuration
# ---------------------------------------------------------

exp = mb.experiment()

# Assign standard components
exp.mop = mb.mops.DTLZ2(M=3)
exp.moea      = mb.moeas.NSGA3(population=100, generations=200)

# Custom parameters via kwargs
exp.moea      = mb.moeas.NSGA3(n_neighbors=15, seed=42)

# ---------------------------------------------------------
# 2. Execution
# ---------------------------------------------------------

exp.run(repeat=1)        # Single run
exp.run(repeat=30)       # 30 independent runs (seeds auto-incremented)

# ---------------------------------------------------------
# 3. Accessing Data (Selectors)
# ---------------------------------------------------------

# Hierarchy: Experiment -> Run -> Population -> SmartArray

run = exp.runs[4]        # Get the 5th run (Run object)
last = exp.last_run      # Get the last run (shortcut)

# Pop at specific generation
pop = run.pop(100)       # Population at generation 100
final = run.pop()        # Final population (default gen=-1)

# Accessing Matrices (Numpy SmartArrays)
pop.objectives           # (N x M) Objectives
pop.variables            # (N x D) Variables

# Short Aliases
pop.objs                 # Same as pop.objectives
pop.vars                 # Same as pop.variables

# ---------------------------------------------------------
# 4. Filters & Sets (Pareto)
# ---------------------------------------------------------

# Filtering
non_dom = run.non_dominated()  # Population of non-dominated solutions
dom     = run.dominated()      # Population of dominated solutions

# Shortcuts for Fronts/Sets (SmartArrays directly)
run.front()              # Same as run.non_dominated().objectives
run.set()                # Same as run.non_dominated().variables
run.non_front()          # Same as run.dominated().objectives

# Shortcuts from Experiment (delegates to last_run)
exp.front()              # Same as exp.last_run.front()
exp.pop()                # Same as exp.last_run.pop()

# ---------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------

# Calculate metrics for the entire history
hv = mb.hv(exp)          # Hypervolume matrix (Generations x Runs)
igd = mb.igd(exp)        # IGD matrix (requires known true front)

# Access metric data
hv.values                # Raw numpy array
hv.runs[0]               # Trajectory of 1st run

# ---------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------

# Progress over time (Time Plot)
mb.timeplot(hv, title="Hv Convergence")

# Objective Space (Space Plot)
mb.spaceplot(exp)                    # Plot Front of last run
mb.spaceplot(exp.pop(0), mode='static') # Plot Initial Pop (static backend)
mb.spaceplot(exp.dominated(), exp.non_dominated()) # Compare sets

# ---------------------------------------------------------
# 7. Extensions
# ---------------------------------------------------------

# Create custom mops by inheriting from BaseMOP

class MyProblem(mb.mops.BaseMOP):
    def __init__(self):
        super().__init__(M=2, N=10)
    def evaluation(self, X):
        return {'F': ...}

exp.mop = MyProblem()
