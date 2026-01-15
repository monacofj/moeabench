<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench API Reference Guide
**Version 1.0.0**

This document provides the exhaustive technical specification for the MoeaBench Library API.

---

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
*   `.optimal(n=500)` (*Population*): Analytical sampling of the true Pareto optimal set and front.
*   `.optimal_front(n=500)` (*SmartArray*): Shortcut to the analytical true PF.
*   `.optimal_set(n=500)` (*SmartArray*): Shortcut to the analytical true PS.

**Methods:**
*   **`.run(repeat=1, workers=None, **kwargs)`**: Executes the optimization.
    *   `repeat` (*int*): Number of independent runs. 
    *   `workers` (*int*): [DEPRECATED] Parallel execution is no longer supported. All runs are performed serially for stability and minimal overhead.
    *   **Reproducibility**: If `repeat > 1`, MoeaBench automatically ensures independence by using `seed + i` for each run `i`. This ensures deterministic results across multiple runs.
    *   `**kwargs`: Passed to the MOEA execution engine.

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

## **2. Visualization (`mb.spaceplot`, `mb.timeplot`)**

High-level plotting functions with **Smart Input Resolution**.

### **`mb.spaceplot(*args, ...)`**
Plots solutions in Objective Space (3D Scatter).

**Input Resolution Rules:**
The function automatically detects what to plot based on the input type:
1.  **If `args[i]` is an `experiment`**:
    *   Default Behavior: Plots `exp.last_run.front()` (The Pareto front of the last run).
    *   *Note*: To plot all runs, use `*exp.all_fronts()`.
2.  **If `args[i]` is a `Run`**:
    *   Default Behavior: Plots `run.front()` (The final Pareto front of that run).
3.  **If `args[i]` is a `Population`**:
    *   Default Behavior: Plots `pop.objectives` (All solutions in that population).
4.  **If `args[i]` is an Array**:
    *   Plots the array directly.

**Usage Example:**
```python
# Quick plot of last result
mb.spaceplot(exp)               

# Comparing Initial vs Final
mb.spaceplot(exp.pop(0), exp.last_pop, title="Evolution")

# Visualizing Dominance
mb.spaceplot(exp.dominated(), exp.non_dominated())
```

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

## **5. Metrics (`mb.metrics`)**

Standard multi-objective performance metrics. Functions accept `Experiment`, `Run`, or `Population` objects as input.

### **Metric Calculation**
*   **`mb.hv(data, ref=None, mode='auto', n_samples=100000)`**: Calculates Hypervolume.
    *   `mode` (*str*): Calculation strategy. 
        *   `'auto'`: Uses the **Exact** algorithm for $M \leq 6$ and switches to **Monte Carlo** for $M > 6$.
        *   `'exact'`: Forces the Exact (WFG) algorithm (may be slow for high dimensions).
        *   `'fast'`: Forces Monte Carlo approximation.
    *   `n_samples` (*int*): Number of points for Monte Carlo sampling (default: $10^5$).
*   **`mb.igd(data, ref=None)`**: Calculates IGD (Inverse Generational Distance).
*   **`mb.gd(data, ref=None)`**: Calculates GD (Generational Distance).

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
hv = mb.hv(exp)
final_gen_dist = hv.gens() # Distribution at final generation
first_run_traj = hv.runs(0) # Trajectory of first run

# Single value case
val = mb.hv(exp.last_pop)
print(f"Final HV: {val:.4f}") 
```

---

---

## **6. Statistics (`mb.stats`)**

Utilities for robust non-parametric statistical analysis. Fully supports **"Smart Stats"** (takes `Experiment` objects, functions, or arrays).

### **`mb.stats.mann_whitney(data1, data2, alternative='two-sided', metric=mb.hv, gen=-1, **kwargs)`**
Performs the Mann-Whitney U rank test.
*   **Args**:
    *   `data1`, `data2`: Arrays, `Experiment` objects, or `MetricMatrix` objects.
    *   `metric` (*callable*): Metric function to use if experiments are passed (e.g., `mb.hv`, `mb.igd`).
    *   `gen` (*int*): Generation index to extract (default is `-1`, the final generation).
    *   `**kwargs`: Arguments passed directly to the `metric` function (e.g., `ref_point`).
*   **Returns**: `HypothesisTestResult` with `.statistic`, `.p_value`, `.a12`, `.significant`, and `.report()`.

### **`mb.stats.ks_test(data1, data2, alternative='two-sided', metric=mb.hv, gen=-1, **kwargs)`**
Performs the Kolmogorov-Smirnov two-sample test for distribution shape differences.
*   **Args**: Same as `mann_whitney`.
*   **Returns**: `HypothesisTestResult` with `.statistic`, `.p_value`, `.a12`, and `.report()`.

### **`mb.stats.a12(data1, data2, metric=mb.hv, gen=-1, **kwargs)`**
Computes the **Vargha-Delaney $\hat{A}_{12}$** effect size.
*   **Args**: Same as `mann_whitney`.
*   **Returns**: `SimpleStatsValue` (float-compatible) with `.value` and `.report()`.

### **`mb.stats.strata(data, gen=-1)`**
Performs **Population Strata** (Dominance Layer analysis).
*   **Args**:
    *   `data`: `Experiment`, `Run`, or `Population`.
    *   `gen` (*int*): Generation index.
*   **Returns**: `StratificationResult` object with `.ranks`, `.frequencies()`, `.selection_pressure`, `.quality_by()`, and `.report()`.

### **`mb.stats.emd(strat1, strat2)`**
Computes the **Earth Mover's Distance** (Wasserstein Distance) between two strata profiles. 

### **`mb.stats.strataplot(*results, labels=None, title=None)`**
Generates a grouped bar chart comparing multiple strata results.

### **`mb.rankplot(*results, labels=None, title=None, metric=None)`**
Generates a **Floating Rank Quality Profile**.
*   **Geometry**: Each dominance rank is represented by a vertical bar.
*   **Vertical Position (Center)**: Set by the `metric` function (default: `mb.hypervolume`).
*   **Bar Height**: Represents the solution density (normalized count) in that rank.
*   **Args**:
    *   `metric` (*callable*): A function that takes an objective matrix and returns a quality score.
    *   `height_scale` (*float*): Multiplier to adjust the thickness of the floating bars.

---

## **7. System Utilities (`mb.system`)**

The `system` module provides utilities for environmental inspection.

### **`mb.system.check_dependencies()`**
Prints a detailed report of installed optional dependencies (`pymoo`, `deap`, etc.).

### **`mb.system.version()`**
Returns the current library version string.

---

## **8. References**
For a detailed technical narrative on the implementation history and mathematical nuances of our benchmarks, see the **[Benchmarks Guide](file:///home/monaco/Work/moeabench/docs/benchmarks.md)**.

*   **[DTLZ]** K. Deb, L. Thiele, M. Laumanns, and E. Zitzler. "Scalable multi-objective optimization test problems." Proc. IEEE Congress on Evolutionary Computation (CEC), 2002.
*   **[DPF]** L. Zhen, M. Li, R. Cheng, D. Peng, and X. Yao. "Multiobjective test problems with degenerate Pareto fronts." IEEE Transactions on Evolutionary Computation, vol. 22, no. 5, 2018.
