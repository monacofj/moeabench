"""
MoeaBench API Cheat Sheet (v0.10.3)
===================================
Information Interpretation for Multi-objective Evolutionary Algorithms.

This script demonstrates the core capabilities of the framework in a condensed format.
It covers configuration, execution, diagnostics, metrics, and extensibility.

NOTE: This file is for REFERENCE ONLY and is not intended for direct execution.
"""
import sys
import MoeaBench as mb
import numpy as np

# Abort execution if run directly
if __name__ == "__main__":
    print(f"MoeaBench v{mb.system.version()}")
    print("This script is a syntax reference only and cannot be executed directly.")
    sys.exit(0)

# -----------------------------------------------------------------------------
# 1. Global Configuration (mb.defaults)
# -----------------------------------------------------------------------------
# Modify global settings before creating experiments to apply them project-wide.

# Execution Defaults
mb.defaults.population = 200            # Default population size
mb.defaults.generations = 500           # Default generations
mb.defaults.seed = 42                   # Base seed for reproducibility

# Visualization Defaults
mb.defaults.theme = 'moeabench'         # Theme: 'moeabench', 'ggplot', 'seaborn'
mb.defaults.backend = 'auto'            # 'plotly' (interactive) or 'matplotlib' (static)
mb.defaults.save_format = 'pdf'         # Auto-save figures as PDF (None to disable)
mb.defaults.dpi = 300                   # Resolution for static exports
mb.defaults.figsize = (10, 8)           # Figure size (width, height)

# -----------------------------------------------------------------------------
# 2. Experiment Setup
# -----------------------------------------------------------------------------
# The 'experiment' object is the central orchestrator.

# Variant A: Object-oriented assignment (Recommended)
exp = mb.experiment()
exp.mop = mb.mops.DTLZ2(M=3)            # Problem: 3-objective DTLZ2
exp.moea = mb.moeas.NSGA3()             # Algorithm: NSGA-III
exp.name = "My Research Study"          # Label for reports
exp.moea.seed = 100                     # Override global seed

# Variant B: Functional configuration (Concise)
# Note: You can pass instances or classes.
exp2 = mb.experiment(mop=mb.mops.DPF1, moea=mb.moeas.MOEAD)

# Variant C: Kwargs override during execution
# See Section 3.

# Scientific Metadata (for Citation/Persistence)
exp.authors = "Silva, F."
exp.license = "CC-BY-4.0"
exp.year = 2026

# -----------------------------------------------------------------------------
# 3. Execution & Persistence
# -----------------------------------------------------------------------------

# Basic Execution (uses defaults/configured values)
exp.run()   

# Advanced Execution (overrides config for this run only)
exp.run(repeat=30,                      # Perform 30 independent runs
        population=100,                 # Override population size
        generations=1000,               # Override generation count
        stop=('n_evals', 10000))        # Custom stopping criterion

# Persistence (Schema v2)
# Saves full state: populations, random states, and metadata.
exp.save("results/study_v1")            # Creates "results/study_v1.zip"

# Loading
exp_loaded = mb.experiment()
exp_loaded.load("results/study_v1.zip") # Restores exact state

# -----------------------------------------------------------------------------
# 4. Data Access (Selectors)
# -----------------------------------------------------------------------------

# Accessing Runs
run = exp.last_run                      # The most recent run
runs = exp.runs                         # List of all Run objects
first_run = exp[0]                      # Indexing support

# Accessing Populations
# exp.pop() aggregates data from ALL runs (Mental Model: "Cloud of Points")
pop_all = exp.pop()                     # All individuals from all runs (final gen)
pop_50  = exp.pop(gen=50)               # All individuals at generation 50
pop_run = run.last_pop                  # Individuals from a specific run

# Accessing Matrices (SmartArrays)
# SmartArrays are NumPy subclasses with metadata (names, bounds).
F = pop_all.objectives                  # (N_total x M) Objectives Matrix
X = pop_all.variables                   # (N_total x N_vars) Decision Variables

# Property Aliases (Shortcuts)
F = pop_all.objs
X = pop_all.vars

# Theoretical Optima (if available for the MOP)
pf = exp.optimal_front(n_points=500)    # True Pareto Front (N x M)
ps = exp.optimal_set(n_points=500)      # True Pareto Set (N x N_vars)

# -----------------------------------------------------------------------------
# 5. Clinical Diagnostics (System V Integration)
# -----------------------------------------------------------------------------
# The "Clinical" layer compares results against calibrated baselines (v4).
# It issues Verdicts (Pass/Fail) and Quality Scores (0-100).

# A. Full Audit
# Generates a report with Verdicts (Pass/Fail) and Quality Scores (0-100)
mb.diagnostics.audit(exp).report_show()

# B. Calibration (For custom problems)
# Generates a baseline profile relative to a random search.
# This creates a sidecar .json file for the MOP.
# mb.diagnostics.calibrate(exp.mop) 

# C. Baselines Management
# mb.diagnostics.use_baselines('user_baselines.json') # Load custom
# mb.diagnostics.reset_baselines()                    # Revert to system default

# -----------------------------------------------------------------------------
# 6. Visualization (Canonical Names)
# -----------------------------------------------------------------------------

# A. Topography (Shape of the Front)
mb.view.topo_shape(exp)                 # 2D/3D Scatter (was 'spaceplot')
mb.view.topo_bands(exp)                 # Reliability Confidence Bands (2D)
mb.view.topo_density(exp)               # Kernel Density Estimation (Heatmap)

# B. Performance (Time Series)
mb.view.perf_history(exp)               # Convergence Profile (was 'timeplot')
mb.view.perf_spread(exp, metric=mb.metrics.hypervolume) # Distribution Boxplots

# C. Clinical (Q-Score Visualization)
# Requires valid baselines/calibration.
mb.view.clinic_radar(exp)               # 6-Axis Diagnostic Radar
mb.view.clinic_history(exp)             # Quality Evolution over time
mb.view.clinic_distribution(exp)        # Q-Score Histograms

# D. Legacy Support (Supported aliases)
# mb.view.spaceplot(exp)
# mb.view.timeplot(exp)

# -----------------------------------------------------------------------------
# 7. Metrics & Statistics
# -----------------------------------------------------------------------------

# Hypervolume (Tripartite: Exact, Monte Carlo, Slicing)
# Automatically selects 'exact' or 'monte_carlo' based on dimensions (M).
hv_matrix = mb.metrics.hypervolume(exp) # Returns MetricMatrix (Run x Time)
hv_matrix.report_show()                  # Statistics (Mean, Median, Std, IQR)

# Accessing raw values
final_hv = hv_matrix.last               # Array of final values for each run

# Distance Metrics (require optimal front)
igd = mb.metrics.igd(exp)               # Inverted Generational Distance
gd  = mb.metrics.gd(exp)                # Generational Distance

# Statistical Tests
# Non-parametric tests for rigorous comparison between two experiments.
exp_a = exp
exp_b = mb.experiment()                 # Assume another experiment
# ... setup and run exp_b ...

# Mann-Whitney U Test (Evidence of superiority)
res_test = mb.stats.perf_evidence(exp_a, exp_b) 
res_test.report_show()

# Vargha-Delaney A12 (Effect Size / Win Probability)
res_prob = mb.stats.perf_probability(exp_a, exp_b)
res_prob.report_show()

# -----------------------------------------------------------------------------
# 8. Custom Extensions (Plugin Architecture)
# -----------------------------------------------------------------------------
# Hook into the framework by subclassing BaseMop and BaseMoea.

class MyProblem(mb.mops.BaseMop):
    def __init__(self):
        super().__init__(M=2, N=10)     # 2 Objectives, 10 Variables
        self.name = "MyZDT Variant"     # Display name
        self.xl = 0.0                   # Lower bound
        self.xu = 1.0                   # Upper bound
    
    def evaluation(self, X):
        # Vectorized evaluation (X is N_pop x N_vars)
        f1 = X[:, 0]
        f2 = 1 + 9 * X[:, 1:].sum(axis=1) / (self.N - 1)
        h = 1 - np.sqrt(f1 / f2)
        f2 = f2 * h
        return {'F': np.column_stack([f1, f2])}

class MyAlgorithm(mb.moeas.BaseMoea):
    def __init__(self):
        super().__init__()
        self.name = "MyOptimizer"
        
    def run(self):
        # Implementation logic...
        # 1. Initialize population
        # 2. Loop generations
        # 3. Save metrics
        pass
