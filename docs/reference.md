<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench API Reference Guide
**Version 0.7.7 (Unified Diagnostics Edition)**

This document provides the exhaustive technical specification for the MoeaBench Library API.

---

<a name="nomenclature"></a>
## **Nomenclature & Abbreviations**

| Abbreviation | Full Term | Description |
| :--- | :--- | :--- |
| **MOP** | Multi-objective Optimization Problem | The benchmark or real-world problem being solved. |
| **MOEA** | Multi-objective Evolutionary Algorithm | The solver or stochastic search engine. |
| **GT** | Ground Truth | The analytical or high-density discrete Pareto optimal set/front. |
| **PF** | Pareto Front | The image of non-dominated solutions in **Objective Space**. |
| **PS** | Pareto Set | The values of decision variables in **Decision Space**. |
| **H** | Hypervolume | The volume of the objective space dominated by a solution set. |
| **H_raw** | Raw Hypervolume | The absolute numerical value in physical units (e.g., $H_{raw} = 1.15$). |
| **H_ratio** | Hypervolume Ratio | Normalized by the **Reference Box** (Ideal to Nadir+Offset). Measures exploration. |
| **H_rel** | Relative Hypervolume | Normalized by the **Ground Truth** performance. Measures convergence. |
| **IGD** | Inverted Generational Distance | Measure of proximity/convergence to the Ground Truth (Average distance from GT to Solution). |
| **GD** | Generational Distance | Measure of convergence (Average distance from Solution to GT). |
| **SP** | Spacing | Measure of the spread/uniformity of the solution set. |
| **EMD** | Earth Mover's Distance | Wasserstein metric measuring topological/distributional similarity. |
| **ADR** | Architecture Decision Record | Document capturing a significant architectural decision. |
| **EAF** | Empirical Attainment Function | Statistical description of the outcomes of stochastic solvers. |

---

<a name="data-model"></a>
## **1. Data Model**

MoeaBench uses a hierarchical data model: `experiment` $\to$ `Run` $\to$ `Population` $\to$ `SmartArray`. All components are designed to be intuitive and chainable.

<a name="defaults"></a>
### **1.2. Global Defaults (`mb.defaults`)**
The `mb.defaults` object allows centralized control over the library's behavior. These values act as fallback "Honest Defaults"—they are used whenever an explicit value is not provided to a method or constructor.

**Execution Parameters:**
*   `population`: Default population size (default: `150`).
*   `generations`: Default generation count (default: `300`).
*   `seed`: Default base random seed (default: `1`).

**Statistics & Rigor:**
*   `alpha`: Significance level for hypothesis tests (default: `0.05`).
*   `cv_tolerance`: Coef. of Variation threshold for "High" stability (default: `0.05`).
*   `cv_moderate`: Coef. of Variation threshold for "Low" stability (default: `0.15`).
*   `displacement_threshold`: Relative frequency threshold for rank displacement (default: `0.1`, i.e., 10%).
*   `large_gap_threshold`: Number of rank levels considered a "Large Gap" (default: `2`).

**Narrative & Presentation:**
*   `precision`: Float precision in reports (default: `4`).
*   `theme`: Visual identity template (default: `'moeabench'`).
*   `backend`: Graphical backend (default: `'auto'`). Supports `'plotly'` or `'matplotlib'`.
*   `save_format`: Automatic export format for plots (default: `None`). Supports `'pdf'`, `'png'`, `'svg'`.
*   `figsize`: Default dimensions for static plots (default: `(10, 8)`).
*   `plot_width`: Default width for interactive plots (default: `900`).
*   `plot_height`: Default height for interactive plots (default: `800`).

**Usage Example:**
```python
import MoeaBench as mb

mb.defaults.population = 500  # Set global default
exp = mb.experiment()        # Will use population=500
exp.run()                    # Executes with 500
```

### **1.3. Experiment**
The top-level container.

**Properties:**
*   `mop` (*Any*): The problem instance.
*   `moea` (*Any*): The algorithm instance.
*   `stop` (*callable*, optional): Global custom stop criteria function.
*   `repeat` (*int*): Default number of repetitions (default: 1).
*   `runs` (*List[Run]*): Access to all execution results.
*   `last_run` (*Run*): Shortcut to the most recent run (`runs[-1]`).
*   `last_pop` (*Population*): Shortcut to the final population of the last run.
*   `.pop(n=-1)` (*JoinedPopulation*): Access the aggregate cloud at generation `n`.
*   `.optimal(n=500)` (*Population*): Analytical sampling of the true Pareto optimal set and front.
*   `.optimal_front(n=500)` (*SmartArray*): Shortcut to the analytical true PF.
*   `.optimal_set(n=500)` (*SmartArray*): Shortcut to the analytical true PS.

**Cloud-centric Delegation (Aggregate Results):**
These methods operate on the **union of all runs** (The Cloud).
*   `.front(n=-1)` (*SmartArray*): ND objectives from all runs (Superfront).
*   `.set(n=-1)` (*SmartArray*): ND variables from all runs (Superset).
*   `.non_front(n=-1)` (*SmartArray*): Dominated objectives from the cloud.
*   `.non_set(n=-1)` (*SmartArray*): Dominated variables from the cloud.
*   `.non_dominated(n=-1)` (*Population*): Aggregate ND population.
*   `.dominated(n=-1)` (*Population*): Aggregate dominated population.
*   `.all_fronts(n=-1)` (*List[SmartArray]*): List of Pareto fronts from all runs.
*   `.all_sets(n=-1)` (*List[SmartArray]*): List of decision sets from all runs.

**Shortcuts:**
*   `.objectives` (*SmartArray*): Shortcut for the final aggregate objectives (`.pop().objectives`).
*   `.variables` (*SmartArray*): Shortcut for the final aggregate variables (`.pop().variables`).

**Methods:**
*   **`.run(repeat=None, workers=None, diagnose=False, **kwargs)`**: Executes the optimization.
    *   `repeat` (*int*, optional): Number of independent runs. Defaults to `exp.repeat`.
    *   `workers` (*int*): [DEPRECATED] Parallel execution is no longer supported.
    *   `diagnose` (*bool*): If `True`, performs automated algorithmic pathology analysis after execution and prints the rationale using `.report_show()`. Defaults to `False`.
    *   **Reproducibility**: If `repeat > 1`, MoeaBench automatically ensures independence by using `seed + i` for each run `i`. This ensures deterministic results across multiple runs.
    *   `stop` (*callable*, optional): Custom stop criteria function. Receives a reference to the active **solver** as its context. Returns `True` to halt execution.
    *   `**kwargs`: Passed to the MOEA execution engine.
*   **`.save(path, mode='all')`**: Persists the experiment to a compressed ZIP file.
    *   `path` (*str*): Filename or folder.
    *   `mode` (*str*): `'all'`, `'config'`, or `'data'`.
*   **`.load(path, mode='all')`**: Restores the experiment state from a ZIP file.
*   **`.get_M()`**: Returns the number of objectives.
*   **`.get_Nvar()`**: Returns the number of decision variables.
*   **`.get_n_ieq_constr()`**: Returns the number of inequality constraints.

**Usage Example:**
```python
import MoeaBench as mb

exp = mb.experiment()
exp.mop = mb.mops.DTLZ2()
exp.moea = mb.moeas.NSGA3()
exp.run(repeat=1)

print(exp.last_run) # Access result
```

### **1.2. Run (`mb.core.run.Run`)**
Represents a single optimization trajectory (history of one seed).

**History Access:**
*   `.history(type='nd')`: Returns the raw list of arrays for the entire run.
    *   Types: `'f'` (all objectives), `'x'` (all variables), `'nd'` (non-dominated objectives), `'nd_x'` (non-dominated variables), `'dom'` (dominated objectives), `'dom_x'` (dominated variables).

**Filters & Snapshots:**
*   `.pop(gen=-1)`: Returns the **Population** object at generation `gen`.
*   `.non_dominated(gen=-1)`: Returns a **Population** containing only non-dominated solutions at `gen`.
*   `.dominated(gen=-1)`: Returns a **Population** containing only dominated solutions at `gen`.

**Shortcuts (Direct Array Access):**
*   `.front(gen=-1)` $\to$ `SmartArray`
    *   *Equivalent to*: `.non_dominated(gen).objectives`
    *   Returns the **Pareto Front** (objectives).
*   `.set(gen=-1)` $\to$ `SmartArray`
    *   *Equivalent to*: `.non_dominated(gen).variables`
    *   Returns the **Pareto Set** (variables).
*   `.non_front(gen=-1)` $\to$ `SmartArray`
    *   *Equivalent to*: `.dominated(gen).objectives`
*   `.non_set(gen=-1)` $\to$ `SmartArray`
    *   *Equivalent to*: `.dominated(gen).variables`

**Usage Example:**
```python
run = exp.last_run

# Get evolution of Pareto Fronts
history = run.history('nd') 

# Get specific snapshots
initial_pop = run.pop(0)
final_front = run.front() # Last generation
```

<a name="smartarray-and-population"></a>
### **1.3. Population (`mb.core.run.Population`)**
A container for a set of solutions at a specific moment.

**Properties:**
*   `.objectives` (*SmartArray*): Matrix (N x M) of objective values.
*   `.variables` (*SmartArray*): Matrix (N x D) of decision variables.

**Aliases:**
*   `.objs` $\to$ `.objectives`
*   `.vars` $\to$ `.variables`

**Filtering Methods:**
*   `.non_dominated()` $\to$ Returns a new *Population* with only non-dominated individuals.
*   `.dominated()` $\to$ Returns a new *Population* with only dominated individuals.

**Analysis Methods:**
*   **`.stratify()`** $\to$ *np.ndarray*: Performs Non-Dominated Sorting (NDS) and returns the integer rank array (1-based) for all individuals.

**Usage Example:**
```python
pop = run.pop(100) # Population at gen 100

# Separation
elite = pop.non_dominated()
others = pop.dominated()

# Access data
print(elite.objs)
print(others.vars)
```

### **1.4. SmartArray**
A NumPy array subclass (`np.ndarray`) that carries metadata.
*   **Metadata**: `.name`, `.label`, `.gen`, `.source`.
*   **Behavior**: Behaves exactly like a standard NumPy array for all math operations.

---

<a name="view"></a>
## **2. Scientific Perspectives (`mb.view`)**

MoeaBench organizes visualization into **Perspectives**. Every plotter in `mb.view` is polymorphic: it accepts `Experiment`, `Run`, `Population` objects or pre-calculated `Result` objects.

### **2.1. Topography (`mb.view.topo_*`)**

*   **`topo_shape(*args, objectives=None, mode='auto', ...)`**:
    *   **Permanent Alias**: `spaceplot`. Visualizes solutions in Objective Space (2D or 3D). 
    *   **Smart Resolution**: Extracts the non-dominated front from experiments.
*   **`topo_bands(*args, levels=[0.1, 0.5, 0.9], ...)`**:
    *   Visualizes reliability bands using **Empirical Attainment Functions (EAF)**.
*   **`topo_gap(exp1, exp2, level=0.5, ...)`**:
    *   Highlights the **Topologic Gap** (coverage difference) between two algorithms.
*   **`topo_density(*args, axes=None, space='objs', ...)`**:
    *   Plots Spatial Probability Density via KDE.

### **2.2. Performance (`mb.view.perf_*`)**

*   **`perf_history(*args, metric=None, ...)`**:
    *   **Permanent Alias**: `timeplot`. Plots the evolution of a scalar metric over time.
*   **`perf_spread(*args, metric=None, gen=-1, alpha=None, ...)`**:
    *   Visualizes **Performance Contrast**. It uses Boxplots to compare distributions and automatically annotates them with the **A12 Win Probability** and P-values (respecting `defaults.alpha`).
*   **`perf_density(*args, metric=None, gen=-1, ...)`**:
    *   Plots the probability distribution of a performance metric via KDE.

### **2.3. Stratification (`mb.view.strat_*`)**

*   **`strat_ranks(*args, ...)`**:
    *   Permanent Alias: `rankplot`. Shows frequency distribution across dominance ranks.
*   **`strat_caste(*args, metric=None, mode='collective', show_quartiles=True, ...)`**:
    *   Maps Quality vs Density using parametric modes ('collective' vs 'individual').
*   **`strat_caste_deprecated(*args, ...)`**:
    *   [DEPRECATED] Original visualizer. Maps Quality vs Density per dominance layer.
*   **`strat_tiers(exp1, exp2=None, ...)`**:
    *   Competitive Duel: joint dominance proportion per global tier.

### **2.5. Clinical Analysis (`mb.view.clinic_*`)**

Specialized diagnostic instruments for deep-dive pathology analysis. All clinical plotters are polymorphic and adhere to the **Smart Dispatch Protocol**, automatically resolving Ground Truth and Resolution Scale from experiments.

*   **`clinic_radar(target, ground_truth=None, show=True, **kwargs)`**:
    *   **Role**: *The Certification* (Q-Score Spider Plot).
    *   **Logic**: Calculates all 6 Q-Scores (Headway, Closeness, Coverage, Gap, Regularity, Balance) and maps them to a radial chart.
    *   **Grid**: Displays explicit concentric circles at intervals of $0.25$ to indicate quality tiers.
*   **`clinic_ecdf(target, ground_truth=None, metric="closeness", mode='auto', show=True, **kwargs)`**:
    *   **Role**: *The Judge* (Goal Attainment).
    *   **Logic**: Plots the Empirical Cumulative Distribution Function of the specified metric.
    *   **Static Markers**: Automatically inserts dashed drop-lines for the **Median (50%)** and **Robust Max (95%)**.
    *   **Metric Support**: Supports all Layer 1 Fairview metrics.
*   **`clinic_distribution(target, ground_truth=None, metric="closeness", mode='auto', show=True, **kwargs)`**:
    *   **Role**: *The Pathologist* (Point-wise Error Morphology).
    *   **Logic**: Renders a Histogram + KDE (Kernel Density Estimate) of the representative FAIR distribution.
    *   **Markers**: Includes median and 95th percentile vertical lines for scale context.
*   **`clinic_history(target, ground_truth=None, metric="closeness", mode='auto', show=True, **kwargs)`**:
    *   **Role**: *The Monitor* (Temporal Trajectory).
    *   **Logic**: Evolution of the representative scalar fact ($f_{val}$) over generations.
    *   **Cloud Support**: If targeting an `Experiment`, it plots trajectories for all runs provided.

---

## **3. MOPs (`mb.mops.*`)**

### **DTLZ Family**
`DTLZ1` - `DTLZ9`.
*   **Args**: `M` (Objectives), `K` (Distance vars), `**kwargs`.
*   **Derived**: `N = M + K - 1`.

**Usage Example:**
```python
# Standard 3-obj
prob = mb.mops.DTLZ2(M=3)

# Scalable many-obj
prob_many = mb.mops.DTLZ2(M=10, K=20)
```

### **DPF Family**
`DPF1` - `DPF5` (Degenerate Pareto Fronts).
*   **Args**: `M` (Objectives), `D` (Base dims), `K` (Distance vars), `**kwargs`.
*   **Derived**: `N = D + K - 1`.

**Usage Example:**
```python
# Degenerate front in 3D based on 2D manifold
prob = mb.mops.DPF1(M=3, D=2)
```

---

## **4. Algorithms (`mb.moeas.*`)**

Supported: `NSGA3`, `MOEAD`, `SPEA2`, `RVEA`.

**Constructor:**
```python
Algorithm(population=150, generations=300, seed=1, **kwargs)
```
*   **`**kwargs`**: Any additional parameter is propagated directly to the underlying engine (Pymoo).
    *   *Example*: `mutation_rate=0.1`, `n_neighbors=15`.

**Usage Example:**
```python
# Standard Usage
algo = mb.moeas.NSGA3(population=200, generations=500)

# Advanced Tuning (using kwargs)
algo_tuned = mb.moeas.NSGA3(population=200, 
                            generations=500,
                            n_neighbors=15,    # Custom Pymoo arg
                            eliminate_duplicates=True)
```

<a name="solver-state"></a>
### **4.1. Solver Runtime State (Stop Criteria Context)**

When a custom `stop` function is executed, it receives a reference to the active solver instance. The following properties are available for inspection:

*   **`solver.n_gen`** (*int*): The current generation index (starts at 1).
*   **`solver.pop`** (*Pymoo Population*): The current population of the search. Use `.get("F")` for objectives and `.get("X")` for variables.
*   **`solver.problem`** (*Experiment*): Reference back to the parent experiment instance.

**Usage in Stop Criteria:**
```python
# Stop if more than 50% of the population is in the first rank
def custom_stop(solver):
    exp = solver.problem
    return exp.last_pop.non_dominated_count > 50

exp.stop = custom_stop
exp.run()
```

---


<a name="metrics"></a>
## **5. Metrics (`mb.metrics`)**

Standard multi-objective performance metrics. Functions accept `Experiment`, `Run`, or `Population` objects as input.

### **Metric Calculation**
*   **`mb.metrics.hv(data, ref=None, mode='auto', n_samples=100000)`**: Calculates Hypervolume.
    *   **Tripartite Output**: Starting with v0.7.6, HV is reported as three distinct measures:
        1.  **HV Raw**: The physical volume calculated relative to the Reference Point (e.g., $1.15$).
        2.  **HV Ratio**: Percentage of the Reference Box covered ($HV_{raw} / RefBox_{vol}$). Represents **Coverage**.
        3.  **HV Rel**: Relative convergence to the Ground Truth ($HV_{sol} / HV_{GT}$). Represents **Convergence**.
    *   `ref`: Reference Point helper.
        *   `None` (default): Calculates the reference point automatically from the input `data` (Nadir + offset).
        *   `list[Experiment]`: Uses the joint range of all provided experiments to calculate a common reference point (Recommended for comparing algorithms).
        *   `np.ndarray`: Uses the provided array as the explicit reference point.
    *   `mode` (*str*): Calculation strategy. 
        *   `'auto'`: Uses the **Exact** algorithm for $M \leq 6$ and switches to **Monte Carlo** for $M > 6$.
        *   `'exact'`: Forces the Exact (WFG) algorithm (may be slow for high dimensions).
        *   `'fast'`: Forces Monte Carlo approximation.
    *   `n_samples` (*int*): Number of points for Monte Carlo sampling (default: $10^5$).
*   **`mb.metrics.igd(data, ref=None)`**: Calculates IGD (Inverse Generational Distance).
    *   `ref`: True Pareto Front.
        *   `None` (default): Attempts to load the analytical optimal front from the MOP (`exp.optimal_front()`).
        *   `np.ndarray`: Uses the provided array as the True Pareto Front.
*   **`mb.metrics.gd(data, ref=None)`**: Calculates GD (Generational Distance).
    *   `ref`: Usage identical to `mb.metrics.igd`.
*   **`mb.metrics.gdplus(data, ref=None)`**: Calculates GD+ (modified Generational Distance).
    *   `ref`: Usage identical to `mb.metrics.igd`.
*   **`mb.metrics.igdplus(data, ref=None)`**: Calculates IGD+ (modified Inverted Generational Distance).
    *   `ref`: Usage identical to `mb.metrics.igd`.

**Returns**: `MetricMatrix` object.

### **`MetricMatrix` Object**
A matrix of metric values (Generations x Runs).

**Accessors:**
*   **`.values`** (*np.ndarray*): Raw data matrix.
*   **`.runs(idx=-1)`**: Returns the metric trajectory over generations for a specific run index.
    *   *Default*: Last run (`-1`).
*   **`.gens(idx=-1)`**: Returns the metric distribution across all runs for a specific generation index.
    *   *Default*: Last generation (`-1`).

**Convenience Features**:
*   **Float Conversion**: If the matrix contains a single value (e.g., from a single population), it can be cast directly to `float(mat)` or used in f-strings with numeric formatters (e.g., `f"{mat:.4f}"`).

**Example:**
```python
hv = mb.metrics.hv(exp)
final_gen_dist = hv.gens() # Distribution at final generation
first_run_traj = hv.runs(0) # Trajectory of first run

# Single value case
val = mb.metrics.hv(exp.last_pop)
print(f"Final HV: {val:.4f}") 

# Plotting Metrics
mb.metrics.plot_matrix(hv, mode='auto', show_bounds=True)
```

### **Metric Visualization (`mb.metrics.plot_matrix`)**
*   **`plot_matrix(metric_matrices, mode='auto', show_bounds=False, title=None, **kwargs)`**:
    *   Visualizes one or more `MetricMatrix` objects.
    *   `mode`: `'auto'` (default), `'interactive'` (Plotly), or `'static'` (Matplotlib).
    *   `show_bounds`: If `True`, displays min/max bounds as dashed lines.

---

---

<a name="stats"></a>
## **6. Statistics (`mb.stats`)**

Utilities for robust non-parametric statistical analysis. Fully supports **"Smart Stats"** (takes `Experiment` objects, functions, or arrays).

### **The Rich Result Interface (`StatsResult`)**
All statistical functions in MoeaBench return objects inheriting from `StatsResult`. These objects provide:
*   **`.report()` $\to$ `str`**: Returns a detailed narrative string. Useful for logging or file output.
*   **`.report_show()`**: Displays the report appropriately for the environment.
    *   **Terminal**: Automatically calls `print(res.report())`.
    *   **Jupyter**: Renders a formatted **Markdown** block using `display(Markdown(...))`.

---

<a name="reportable"></a>
## **7. The Reporting Contract (`Reportable`)**

MoeaBench enforces a **Universal Reporting Contract**. Every analytical object (`Experiment`, `MetricMatrix`, `StatsResult`) inherits from the `Reportable` mixin.

### **The Interface**
*   **`.report(**kwargs) \to str`**: returns a detailed technical narrative explaining the object's context, data, and scientific meaning.
*   **`.report_show(**kwargs)`**: adaptive display method (Terminal vs. Notebook).

### **Participating Objects**
1.  **`mb.experiment`**: Summarizes the experimental protocol (MOP, MOEA, Status).
2.  **`mb.metrics.MetricMatrix`**: Summarizes mathematical performance, search dynamics, and stochastic stability.
3.  **`mb.stats.StatsResult`**: Summarizes hypothesis tests, rank stratification, and topological matching.

> [!NOTE]
> **Transparency Policy (Explainable Verdicts)**
> As of v0.7.7, all narrative reports explicitly state the decision criteria for qualitative judgments.
> *   **Stability**: Shows Coefficient of Variation (e.g., `CV=0.01 < 0.05`).
> *   **Stratification**: Explicitly states displacement depth trigger (`> 10%`).
> *   **Statistics**: Clarifies significance (`p < 0.05`) and effect size thresholds.

The output is something like:

```text
--- Population Strata Report: NSGA-II on DTLZ2 ---
  Search Depth: 3 non-dominated layers
  Selection Pressure: 0.9412

Rank   | Pop %    | Quality (hypervolume)
----------------------------------------
1      |    85.0% |       0.8200
2      |    12.0% |       0.4500
3      |     3.0% |       0.1200

Diagnosis: High Selection Pressure (Phalanx-like convergence).
```

---

### **`mb.stats.perf_evidence(data1, data2, alternative='two-sided', metric=mb.metrics.hv, gen=-1, **kwargs)`**
Performs the **Mann-Whitney U** rank-sum test (Win Evidence). Returns a `HypothesisTestResult`.

### **`mb.stats.perf_distribution(data1, data2, alternative='two-sided', metric=mb.metrics.hv, gen=-1, **kwargs)`**
Performs the **Kolmogorov-Smirnov (KS)** two-sample test (Performance Distribution). Returns a `HypothesisTestResult`.

### **`mb.stats.perf_probability(data1, data2, metric=mb.metrics.hv, gen=-1, **kwargs)`**
Computes the **Vargha-Delaney $\hat{A}_{12}$** effect size (Win Probability). Returns a `SimpleStatsValue`.

### **`mb.stats.topo_distribution(*args, space='objs', axes=None, method='ks', alpha=0.05, threshold=0.1, **kwargs)`**
Performs multi-axial distribution matching (Topologic Equivalence).
*   **Args**:
    *   `*args`: Two or more datasets (`Experiment`, `Run`, `Population` or `SmartArray`).
    *   `space` (*str*): `'objs'` or `'vars'`. Defaults to `'objs'`.
    *   `axes` (*list*): Specific indices to test.
    *   `method` (*str*): 
        *   `'ks'`: **Kolmogorov-Smirnov** test (Default).
        *   `'anderson'`: **Anderson-Darling k-sample** test.
        *   `'emd'`: **Earth Mover's Distance** (Wasserstein Metric).
*   **Returns**: `DistMatchResult`.

### **`mb.stats.topo_attainment(source, level=0.5)`**
Calculates the attainment surface reached by $k\%$ of the runs.
*   **Methodology**: Grounded in **Empirical Attainment Functions (EAF)**.
*   **Methodology**: Grounded in **Empirical Attainment Functions (EAF)**.
*   **Returns**: `AttainmentSurface` (SmartArray subclass). Can be plotted using `mb.spaceplot`.

### **`mb.stats.topo_gap(exp1, exp2, level=0.5)`**
Calculates the spatial Gap in attainment between two groups.
*   **Methodology**: Based on **EAF Difference** analysis.
*   **Returns**: `AttainmentDiff` object.

### **`mb.stats.strata(data, gen=-1)`**
Performs **Population Strata** (Dominance Layer analysis) based on Pareto dominance.
*   **Returns**: `StratificationResult`.

### **`mb.stats.tier(exp1, exp2, gen=-1)`**
Performs **Joint Stratification** (Tier analysis) between two experiments.
*   **Returns**: `TierResult`.
    *   `.pole` (*np.ndarray*): Proportion of each algorithm in the first rank (Elite).
    *   `.gap` (*int*): Displacement depth (rank where the loser starts to appear significantly).
    *   `.dominance_ratio` (*np.ndarray*): Same as `.pole`.
    *   `.report()`: Generates a competitive narrative ("Pole Position", "Displacement Depth").

### **`mb.stats.emd(strat1, strat2)`**
Computes the **Earth Mover's Distance** between two strata profiles.


---

## **7. System Utilities (`mb.system`)**

The `system` module provides utilities for environmental inspection.

### **`mb.system.check_dependencies()`**
Prints a detailed report of installed optional dependencies (`pymoo`, `deap`, etc.).

### **`mb.system.export_objectives(data, filename=None)`**
Exports objectives to a CSV file.
*   **data**: `Experiment`, `Population`, or raw array.
*   **filename**: Optional custom filename.

### **`mb.system.export_variables(data, filename=None)`**
Exports decision variables to a CSV file.
*   **data**: `Experiment`, `Population`, or raw array.
*   **filename**: Optional custom filename.

### **`mb.system.version()`**
Returns the current library version string.

---

<a name="persistence"></a>
## **8. Persistence & Data Format**

MoeaBench uses a standardized ZIP-based persistence format.

### **Experiment Interface**
*   **`.save(path, mode='all')`**:
    *   `all`: Saves everything.
    *   `config`: Saves only MOP/MOEA setup.
    *   `data`: Saves only execution histories.
*   **`.load(path, mode='all')`**: Reverse of save.

### **File Structure (inside ZIP)**
1.  `Moeabench.joblib`: The binary state of the Python objects.
2.  `result.csv`: The Global Non-Dominated Front.
3.  `problem.txt`: Human-readable summary of the MOP and MOEA parameters.

---

<a name="extensibility"></a>
## **9. Extensibility (Plugin API)**

MoeaBench is designed as a **host framework**. By inheriting from our base classes, your custom logic becomes a "first-class citizen," gaining instant access to the entire analytical suite (metrics, persistence, and specialized plots).

### **9.1. Custom MOPs (`mb.mops.BaseMop`)**
The contract for problems requires implementing a **vectorized** evaluation function. This allows the framework to process entire populations using NumPy's high-performance broadcasting.

**The Contract:**
*   **`__init__(self, M, N, ...)`**: Call `super().__init__` to register the number of objectives ($M$) and variables ($N$).
*   **`evaluation(self, X)`**: Receives a population matrix $X$ ($PopSize \times N$). Must return a dictionary containing the objectives matrix `F` ($PopSize \times M$).

**Example: A Simple Convex Problem**
```python
import numpy as np
from MoeaBench import mb

class MyConvexProblem(mb.mops.BaseMop):
    def __init__(self):
        # 2 Objectives, 10 Variables, variables in range [0, 1]
        super().__init__(M=2, N=10, xl=0.0, xu=1.0)

    def evaluation(self, X):
        # f1: simply the first variable
        f1 = X[:, 0]
        # f2: convex transformation (ZDT1-style)
        g = 1 + 9 * np.mean(X[:, 1:], axis=1)
        f2 = g * (1 - np.sqrt(f1 / g))
        
        return {'F': np.column_stack([f1, f2])}
```

### **9.2. Custom MOEAs (`mb.moeas.BaseMoea`)**
To wrap your own search algorithm, inherit from `BaseMoea`. This ensures that your solver integrates with the `Experiment.run()` lifecycle, including automatic seed management and result persistence.

**The Contract:**
*   **`evaluation(self)`**: This is the entry point for the search. When `exp.run()` is called, it triggers this method.
*   **Data Return**: For full compatibility with MoeaBench's history tools, the method should return the final population and, ideally, the objective/variable trajectories.
*   **Narrative Reporting**: If your algorithm provides internal diagnostics, consider returning a `StatsResult` object (or subclass) to leverage the `.report_show()` system.

**Example: A Random Search Skeleton**
```python
from MoeaBench import mb
import numpy as np

class RandomSearch(mb.moeas.BaseMoea):
    def evaluation(self):
        # Access problem via self.get_problem()
        mop = self.get_problem()
        
        # 1. Generate random solutions
        X = np.random.uniform(mop.xl, mop.xu, (self.population, mop.N))
        
        # 2. Evaluate using the framework's helper
        res = self.evaluation_benchmark(X)
        
        # 3. Return final objectives (F) and variables (X)
        # For simple plugins, returning the final population suffice.
        # More advanced wrapping allows for generational history.
        return res['F'], X, res['F'], None, None, None, None
```
> [!TIP]
> For a detailed walkthrough on implementing and using custom plugins, see **`examples/example_05.py`**.
> To see the **Automated Diagnostics** in action across different algorithmic pathologies, refer to **`examples/example_11.py`**.

---

## **10. References**
For a detailed technical narrative on the implementation history and mathematical nuances of our MOPs, see the **[MOPs Guide](mops.md)**.

*   **[DTLZ]** K. Deb, L. Thiele, M. Laumanns, and E. Zitzler. "[Scalable multi-objective optimization test problems](https://doi.org/10.1109/CEC.2002.1007032)." Proc. IEEE Congress on Evolutionary Computation (CEC), 2002.
*   **[DPF]** L. Zhen, M. Li, R. Cheng, D. Peng, and X. Yao. "[Multiobjective test problems with degenerate Pareto fronts](https://doi.org/10.48550/arXiv.1806.02706)." IEEE Transactions on Evolutionary Computation, vol. 22, no. 5, 2018.

## **11. Legacy Support & Depletion Schedule**

MoeaBench maintains backward compatibility for its evolutionary analytical layer through two tiers of support.

### **11.1. Permanent Aliases**
The following functions have been promoted to permanent status due to their widespread use in the standard optimization literature. They function identically to their taxonomical successors.

| New Full Name | Permanent Alias | Scientific Domain |
| :--- | :--- | :--- |
| `mb.view.topo_shape` | `mb.spaceplot` | Topography |
| `mb.view.perf_history` | `mb.timeplot` | Performance |
| `mb.view.strat_ranks` | `mb.rankplot` | Stratification |

### **11.2. Soft-Deprecated Aliases**
These functions are maintained for compatibility with versions `v0.6.x` but are scheduled for formal deprecation. They currently act as active delegates but will be replaced by informational stubs (warnings without functionality) in future major releases.

| Legacy Alias | Taxonomical Successor | Domain |
| :--- | :--- | :--- |
| `mb.casteplot` | `mb.view.strat_caste_deprecated` | Stratification |
| `mb.tierplot` | `mb.view.strat_tiers` | Stratification |
| `mb.view.topo_dist` | `mb.view.topo_density` | Topography |
| `mb.stats.perf_prob` | `mb.stats.perf_probability` | Performance |
| `mb.stats.perf_dist` | `mb.stats.perf_distribution` | Performance |
| `mb.stats.topo_dist` | `mb.stats.topo_distribution` | Topography |
| `mb.stats.topo_attain` | `mb.stats.topo_attainment` | Topography |

> [!IMPORTANT]
> **Hard Deprecation Policy**: In future versions, the soft-deprecated items above will only produce a `UserWarning` and will no longer execute logic. We strongly recommend updating your research pipelines to the new nomenclature.


<a name="diagnostics"></a>
## **12. Automated Diagnostics (`mb.diagnostics`)**

The `mb.diagnostics` module is the high-level analytical interface for **Algorithmic Pathology**. Following the pattern established in the `mb.stats` module, all diagnostic outputs implement the **Universal Reporting Contract** (`Reportable`), providing narrative insights alongside numerical values.

### **12.1. The Diagnostics Reporting Contract (`Reportable`)**

Instead of returning raw `float` or `ndarray` values, functions return specialized objects that preserve numerical behavior while providing narrative insights.

*   **`DiagnosticValue`**: The base class for single-metric results.
    *   **Numerical Fallback**: Objects can be cast directly to `float()`.
    *   **`.report()`**: Returns a multi-line Markdown string with clinical labels and insights.
    *   **`.report_show()`**: Renders the narrative report with rich formatting.

| Result Class | Returner functions | Characteristics |
| :--- | :--- | :--- |
| **`FairResult`** | `headway`, `coverage`, etc. | Physical facts, normalized by resolution. |
| **`QResult`** | `q_headway`, `q_coverage`, etc. | Clinical scores $[0, 1]$, categorized into 5 quality tiers. |
| **`DiagnosticResult`**| `audit()` | High-level synthesis with Biopsy summary. |

---

### **12.2. The Smart Dispatch Protocol**

> [!NOTE]
> **Design Background**: For the mathematical rationale behind these metrics (including the "Monotonicity Gate" and "Headway" renaming), see **[ADR 0028: Refined Clinical Diagnostics](../docs/adr/0028-refined-clinical-diagnostics-v0.9.1.md)**.

All functions in `mb.diagnostics` use a **Smart Dispatch** system (`_resolve_diagnostic_context`) that automatically interprets input data.

*   **Polymorphic Input**:
    *   `Experiment`: Automatically extracts the **Pareto Front** of the last run, the **Ground Truth** ($GT$) from the MOP, and the **Resolution Scale** ($s_K$).
    *   `Run`: Extracts Front, GT, and Scale from a specific execution.
    *   `Population`: Extracts Front and tries to find MOP references.
    *   `np.ndarray`: Treated as a raw Front. Requires manual `ref` (GT) and `s_k` to be safe.

*   **Context Resolution**:
    *   **$GT$ (Ground Truth)**: If not provided explicitly via `ref=...`, the system looks for `.optimal_front()` or `.mop.pf()`.
    *   **$s_K$ (Resolution Scale)**: The "Physics of Resolution". If not provided via `s_k=...`, it looks for `mop.s_k` or `mop.s_fit`. Defaults to `1.0` if unknown.

---

### **12.2. Physical Metrics (Fair Layer)**

These metrics answer: *"What is the physical state of the population?"*
They are "Fair" because they are divided by $s_K$, making them scale-invariant across different problems.

#### **`headway(data, ref=None, s_k=None) -> FairResult`**
*   **Definition**: $GD_{95}(P \to GT) / s_K$
*   **Rationale**: Measures **Convergence Depth**. It filters out the worst 5% outliers (Headway) and normalizes the distance by the expected separation of optimal points ($s_K$).
*   **Ideal**: $0.0$.
*   **Interpretation**: Values $< 1.0$ indicate the population is "fitting" the manifold better than the discrete resolution limit.

#### **`closeness(data, ref=None, s_k=None) -> np.ndarray`**
*   **Definition**: Vector of point-wise distances: $u_j = \min(\|p_j - GT\|) / s_K$.
*   **Rationale**: Provides the **Distribution of Closeness**. Unlike a single scalar, this array reveals if the population is uniformly close or if parts of it are drifting.
*   **Returns**: 1D Array of size $N$.

#### **`coverage(data, ref=None) -> FairResult`**
*   **Definition**: $IGD_{mean}(GT \to P)$
*   **Rationale**: Measures **Global Coverage**. It is the average distance from *any* point on the True Front to the nearest solution found.
*   **Ideal**: $0.0$.

#### **`gap(data, ref=None) -> FairResult`**
*   **Definition**: $IGD_{95}(GT \to P)$
*   **Rationale**: Measures **worst-case holes** in the coverage. By taking the 95th percentile of the IGD components, it identifies the largest "Gaps" in the manifold approximation.

#### **`regularity(data, ref_distribution=None) -> FairResult`**
*   **Definition**: $W_1(d_{NN}(P), d_{NN}(U_{ref}))$
*   **Rationale**: Measures **Structural Uniformity**. It uses the **Wasserstein-1 Distance** (Earth Mover's Distance) to compare the distribution of Nearest-Neighbor distances in $P$ against a perfectly uniform lattice $U_{ref}$.
*   **Ideal**: $0.0$ (structure matches the lattice).

#### **`balance(data, centroids=None, ref_hist=None) -> FairResult`**
*   **Definition**: $D_{JS}(H_P || H_{ref})$
*   **Rationale**: Measures **Manifold Bias**. It partitions the Ground Truth into $C$ clusters and calculates the **Jensen-Shannon Divergence** between the finding rates of the algorithm and the reference density.
*   **Ideal**: $0.0$ (Balanced occupancy).

---

### **12.3. Clinical Metrics (Q-Scores)**

These metrics answer: *"Is this good or bad?"*
They map physical values to a $[0, 1]$ utility scale using **Offline Baselines**.

*   **Range**: $1.0$ (Ideal) to $0.0$ (Baseline/Random). Negative values indicate pathological failure.
*   **Logic**: $Q = 1 - \text{Correction}(\frac{\text{Fair} - \text{Ideal}}{\text{Random} - \text{Ideal}})$

#### **`q_headway(data, ...) -> QResult`**
*   **Baselines**: Ideal=$0.0$, Random=$Rand_{50}$ (from repository).
*   **Mapping**: **Log-Linear**. Expands resolution near $Q=1.0$ to distinguish "Super-converged" solutions from merely "Good" ones.

#### **`q_closeness(data, ...) -> QResult`**
*   **Baselines**: Ideal=$0.0$, Random=$ECDF(G_{blur})$ (Gaussian-blurred GT).
*   **Mapping**: **Wasserstein-1**. Computes the topological similarity between the finding distribution and the noise model.

#### **`q_coverage`, `q_gap`, `q_regularity`, `q_balance` -> `QResult`**
*   **Baselines**: Ideal=$Uni_{50}$ (Median of FPS subsets), Random=$Rand_{50}$ (Median of Random subsets).
*   **Mapping**: **ECDF**. Uses the full Empirical Cumulative Distribution Function of random sampling to rank the solution.

---

### **12.4. Comprehensive Algorithmic Audit**

#### **`audit(data, ground_truth=None, problem=None, ...) -> DiagnosticResult`**
*   **Rationale**: The primary entry point for automated performance analysis. It runs the full 6-dimensional clinical suite and synthesizes a high-level verdict.
*   **Process**:
    1.  Calculates all 6 Q-Scores.
    2.  Applies the **Performance Auditor** expert system.
    3.  Classifies the algorithm into the **8-State Pathology Ontology**.
*   **Return**: A `DiagnosticResult` object that provides the full narrative biopsy via `.report_show()`.

---

### **12.4. Offline Calibration Method**

MoeaBench uses a strict **Fail-Closed** calibration system (`baselines_v4.json`).

1.  **Discrete Sampling**: Baselines are pre-computed for finite population sizes $K \in \{10..50, 100, 150, 200\}$.
2.  **$Uni_{50}$ (The Anchor)**: Generated using **Farthest Point Sampling (FPS)** on the Ground Truth. Represents the "best possible" distribution for a discrete set of size $K$.
3.  **$Rand_{50}$ & ECDF (The Noise)**: Generated by Monte Carlo sampling (Uniform randomness or Gaussian blur). Represents the "Physics of Failure".
4.  **Snap Policy**: If an experiment uses a $K$ not in the grid, it snaps to the nearest safe floor (e.g., $K=80 \to 50$) to avoid unfair penalization.
