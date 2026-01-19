<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench API Reference Guide
**Version 1.0.0**

This document provides the exhaustive technical specification for the MoeaBench Library API.

---

<a name="data-model"></a>
## **1. Data Model**

MoeaBench uses a hierarchical data model: `experiment` $\to$ `Run` $\to$ `Population` $\to$ `SmartArray`. All components are designed to be intuitive and chainable.

### **1.1. Experiment**
The top-level container.

**Properties:**
*   `mop` (*Any*): The problem instance.
*   `moea` (*Any*): The algorithm instance.
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

**Methods:**
*   **`.run(repeat=1, workers=None, **kwargs)`**: Executes the optimization.
    *   `repeat` (*int*): Number of independent runs. 
    *   `workers` (*int*): [DEPRECATED] Parallel execution is no longer supported. All runs are performed serially for stability and minimal overhead.
    *   **Reproducibility**: If `repeat > 1`, MoeaBench automatically ensures independence by using `seed + i` for each run `i`. This ensures deterministic results across multiple runs.
    *   `**kwargs`: Passed to the MOEA execution engine.
*   **`.save(path, mode='all')`**: Persists the experiment to a compressed ZIP file.
    *   `path` (*str*): Filename or folder.
    *   `mode` (*str*): `'all'`, `'config'`, or `'data'`.
*   **`.load(path, mode='all')`**: Restores the experiment state from a ZIP file.

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
    *   Types: `'f'` (all objectives), `'x'` (all variables), `'nd'` (non-dominated objectives), `'nd_x'` (non-dominated variables).

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

<a name="spaceplot"></a>
<a name="viewspaceplot"></a>
### **2.1. Spatial Perspective (`mb.view.spaceplot`)**
Plots solutions in Objective Space (2D or 3D Scatter).
*   **Args**:
    *   `*args`: experiments, runs, populations, or arrays.
    *   `objectives` (*list*): Indices of objectives to plot (default: `[0, 1]` or `[0, 1, 2]`).
    *   `mode` (*str*): `'auto'` (detects environment), `'static'`, or `'interactive'`.
*   **Smart Resolution**: Automatically extracts the non-dominated front from experiments.

### **2.2. Historic Perspective (`mb.view.timeplot`)**
Plots the evolution of scalar metrics over time.
*   **Args**:
    *   `*args`: experiments or `MetricMatrix` objects.
    *   `metric` (*callable*): The metric to calculate (default: `mb.metrics.hv`).
*   **Smart Resolution**: Automatically calculates the metric for experiments.

### **2.3. Structural Perspective (`mb.view.rankplot`)**
Plots the frequency distribution of dominance ranks (selection pressure).
*   **Args**:
    *   `*args`: experiments, runs, or `StratificationResult` objects.
*   **Smart Resolution**: Calls `mb.stats.strata()` internally for raw objects.

### **2.4. Hierarchical Perspective (`mb.view.casteplot`)**
Plots the **Caste Profile** (Floating bars showing Quality vs. Density per rank).
*   **Geometry**: Center = Quality; Height = Density (Population %).
*   **Args**:
    *   `*args`: experiments or `StratificationResult` objects.
    *   `metric` (*callable*): Quality measure (default: `mb.metrics.hv`).
    *   `height_scale` (*float*): Adjusts the thickness of the bars.

### **2.5. Competitive Perspective (`mb.view.tierplot`)**
Plots the **Competitive Tier Duel** (Stacked bars showing absolute population contribution per tier).
*   **Visualization**: The y-axis represents the **absolute population count** in each tier. Segment heights represent the contribution of each algorithm.
*   **Args**:
    *   `exp1`, `exp2`: The two experiments to compare.
*   **Smart Resolution**: Calls `mb.stats.tier(exp1, exp2)` internally to build the `mb.view.tierplot` input.

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

---

<a name="metrics"></a>
## **5. Metrics (`mb.metrics`)**

Standard multi-objective performance metrics. Functions accept `Experiment`, `Run`, or `Population` objects as input.

### **Metric Calculation**
*   **`mb.metrics.hv(data, ref=None, mode='auto', n_samples=100000)`**: Calculates Hypervolume.
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
```

---

---

<a name="stats"></a>
## **6. Statistics (`mb.stats`)**

Utilities for robust non-parametric statistical analysis. Fully supports **"Smart Stats"** (takes `Experiment` objects, functions, or arrays).

### **`mb.stats.mann_whitney(data1, data2, alternative='two-sided', metric=mb.metrics.hv, gen=-1, **kwargs)`**
Performs the Mann-Whitney U rank test.
*   **Args**:
    *   `data1`, `data2`: Arrays, `Experiment` objects, or `MetricMatrix` objects.
    *   `metric` (*callable*): Metric function to use if experiments are passed (e.g., `mb.metrics.hv`, `mb.metrics.igd`).
    *   `gen` (*int*): Generation index to extract (default is `-1`, the final generation).
    *   `**kwargs`: Arguments passed directly to the `metric` function (e.g., `ref_point`).
*   **Returns**: `HypothesisTestResult` with `.statistic`, `.p_value`, `.a12`, `.significant`, and `.report()`.

### **`mb.stats.ks_test(data1, data2, alternative='two-sided', metric=mb.metrics.hv, gen=-1, **kwargs)`**
Performs the Kolmogorov-Smirnov two-sample test for distribution shape differences.
*   **Args**: Same as `mann_whitney`.
*   **Returns**: `HypothesisTestResult` with `.statistic`, `.p_value`, `.a12`, and `.report()`.

### **`mb.stats.a12(data1, data2, metric=mb.metrics.hv, gen=-1, **kwargs)`**
Computes the **Vargha-Delaney $\hat{A}_{12}$** effect size.
*   **Args**: Same as `mann_whitney`.
*   **Returns**: `SimpleStatsValue` (float-compatible) with `.value` and `.report()`.

### **`mb.stats.strata(data, gen=-1)`**
Performs **Population Strata** (Dominance Layer analysis).
*   **Args**:
    *   `data`: `Experiment`, `Run`, or `Population`.
    *   `gen` (*int*): Generation index.
*   **Returns**: `StratificationResult` object with `.ranks`, `.frequencies()`, `.selection_pressure`, `.quality_by()`, and `.report()`.

### **`mb.stats.tier(exp1, exp2, gen=-1)`**
Performs **Joint Stratification** analysis (Tier analysis) between two groups.
*   **Returns**: `TierResult` object with `.joint_frequencies`, `.pole`, `.gap`, and competitive `.report()`.

### **`mb.stats.emd(strat1, strat2)`**
Computes the **Earth Mover's Distance** (Wasserstein Distance) between two strata profiles. 


---

## **7. System Utilities (`mb.system`)**

The `system` module provides utilities for environmental inspection.

### **`mb.system.check_dependencies()`**
Prints a detailed report of installed optional dependencies (`pymoo`, `deap`, etc.).

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

To implement your own components, follow these base classes:

### **Custom MOPs (`mb.mops.BaseMop`)**
*   **Required**: `evaluation(X) -> {'F': objectives}`.
*   **Optional**: `ps(n) -> Pareto Set` (for IGD calculation).

### **Custom MOEAs (`mb.moeas.MOEA`)**
*   **Required**: `solve(mop, termination) -> Population`. 
*   **Integration**: Inheriting from `MOEA` allows your algorithm to be managed by `Experiment.run()` and automatically handles state persistence.

---

## **10. References**
For a detailed technical narrative on the implementation history and mathematical nuances of our MOPs, see the **[MOPs Guide](mops.md)**.

*   **[DTLZ]** K. Deb, L. Thiele, M. Laumanns, and E. Zitzler. "[Scalable multi-objective optimization test problems](https://doi.org/10.1109/CEC.2002.1007032)." Proc. IEEE Congress on Evolutionary Computation (CEC), 2002.
*   **[DPF]** L. Zhen, M. Li, R. Cheng, D. Peng, and X. Yao. "[Multiobjective test problems with degenerate Pareto fronts](https://doi.org/10.48550/arXiv.1806.02706)." IEEE Transactions on Evolutionary Computation, vol. 22, no. 5, 2018.

# Deprecated API

The following API is deprecated and will be removed in a future release. Use the new API instead.

| Deprecated (Old) | Replacement (New) | Type |
| :--- | :--- | :--- |
| `mb.spaceplot()` | `mb.view.spaceplot()` | Plotting |
| `mb.timeplot()` | `mb.view.timeplot()` | Plotting |
| `mb.rankplot()` | `mb.view.rankplot()` | Plotting |
| `mb.casteplot()` | `mb.view.casteplot()` | Plotting |
| `mb.tierplot()` | `mb.view.tierplot()` | Plotting |
| `mb.hv()` | `mb.metrics.hv()` | Metric |
| `mb.igd()` | `mb.metrics.igd()` | Metric |
| `mb.gd()` | `mb.metrics.gd()` | Metric |
| `mb.strata()` | `mb.stats.strata()` | Stats |
| `mb.tier()` | `mb.stats.tier()` | Stats |
| `mb.emd()` | `mb.stats.emd()` | Stats |
| `mb.attainment()` | `mb.stats.attainment()` | Stats |
| `mb.DTLZ*()` | `mb.mops.DTLZ*()` | Benchmark |
| `mb.NSGA3()` | `mb.moeas.NSGA3()` | Algorithm |
| `exp.superfront()` | `exp.front()` | Delegation |
| `exp.superset()` | `exp.set()` | Delegation |

