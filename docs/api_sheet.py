# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import MoeaBench as mb

## API Sheet
## This script is for documentation purposes only. Do not execute it.

# ---------------------------------------------------------
# 1. Experiment Configuration
# ---------------------------------------------------------

exp1 = mb.experiment()

# Assign benchmarks (MOPs) and algorithms (MOEAs)
exp1.mop = mb.mops.DTLZ2(M=3)
exp1.moea = mb.moeas.NSGA3(population=100, generations=200)

# Custom parameters
exp1.name = "My Experiment"
exp1.moea = mb.moeas.NSGA2(population=100, seed=42)

# ---------------------------------------------------------
# 2. Execution
# ---------------------------------------------------------

exp1.run(repeat=1)        # Single run
exp1.run(repeat=30)       # 30 independent runs (sequential)

# ---------------------------------------------------------
# 3. Accessing Data (Selectors)
# ---------------------------------------------------------

run1 = exp1.runs[0]       # Get the 1st run (Run object)
last = exp1.last_run      # Get the last run (shortcut)

# Accessing Populations at specific generations
pop1 = run1.pop(50)       # Population at generation 50
final = run1.pop()        # Final population (default gen=-1)

# Objectives and Variables (SmartArrays/NumPy)
pop1.objectives           # (N x M) Matrix (Full name)
pop1.objs                 # Alias
pop1.variables            # (N x D) Matrix (Full name)
pop1.vars                 # Alias

# ---------------------------------------------------------
# 4. Indicators and Frontiers
# ---------------------------------------------------------

# Shortcuts from Experiment (Delegates to the Aggregate Cloud/Superfront)
exp1.front()              # ND objectives across ALL runs (Superfront)
exp1.set()                # ND variables across ALL runs (Superset)
exp1.last_run.front()     # Surgical access to the last run only

# Analytical optima
exp1.optimal()            # Known Pareto Front (Population object)
exp1.optimal_front()      # Known Pareto Front (SmartArray)

# ---------------------------------------------------------
# 5. Metrics & "Smart Stats"
# ---------------------------------------------------------

# Metric Evaluators
hv1 = mb.metrics.hv(exp1)         # Hypervolume convergence
igd1 = mb.metrics.igd(exp1)       # Inverted Generational Distance

# Statistical Diagnostics (Rich Results)
# Every result object provides a .report() method and lazy properties.
res1 = mb.stats.mann_whitney(exp1, exp2)
print(res1.report())      # Narrative statistical diagnosis
res1.p_value              # Access programmatic p-value

# Population Strata (Geology)
strat1 = mb.stats.strata(exp1)
strat1.report()           # Maturity and density diagnosis
strat1.selection_pressure  # Lazy-evaluated diagnostic metric

# Comparison via EMD (Earth Mover's Distance)
dist1 = mb.stats.emd(strat1, strat2)

# Competitive Tier Analysis
res_tier = mb.stats.tier(exp1, exp2)
res_tier.dominance_ratio  # Ratio at Tier 1
res_tier.displacement_depth # Depth of infiltration

# Attainment Comparison
diff1 = mb.stats.attainment_diff(exp1, exp2, level=0.5)

# ---------------------------------------------------------
# 6. Visualization (Scientifc Perspectives: mb.view)
# ---------------------------------------------------------

mb.view.timeplot(hv1)                         # Historic: Convergence plot
mb.view.spaceplot(exp1)                       # Spatial: Pareto Front plot (2D/3D)
mb.view.rankplot(exp1)                        # Structural: Selection pressure (counts)
mb.view.casteplot(exp1)                       # Hierarchical: Quality/Density profile
mb.view.tierplot(exp1, exp2)                  # Competitive: Tier duel
mb.view.distplot(exp1, exp2)                  # Distribution: Probability density (KDE)

# ---------------------------------------------------------
# 7. Distribution Matching (v0.6.0)
# ---------------------------------------------------------

res_match = mb.stats.dist_match(exp1, exp2)   # Topological Equivalence
res_match.is_consistent                       # boolean result
res_match.report()                            # Dimensional analysis

# ---------------------------------------------------------
# 8. Custom Extensions
# ---------------------------------------------------------

# Create custom problems using NumPy vectorization
class CustomMOP(mb.mops.BaseMop):
    def __init__(self):
        super().__init__(M=2, N=10)

    def evaluation(self, X):
        # Always use vectorized NumPy operations
        f1 = X[:, 0]
        f2 = 1 - f1**2
        return {'F': np.column_stack([f1, f2])}

exp1.mop = CustomMOP()
