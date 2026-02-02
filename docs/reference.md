<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench API Reference Guide
**Version 0.7.5 (Scientific Domains Edition)**

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

### **1.1. Experiment**
The top-level container.

**Properties:**
*   `mop` (*Any*): The problem instance.
*   `moea` (*Any*): The algorithm instance.
*   `stop` (*callable*, optional): Global custom stop criteria function.
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
    *   `stop` (*callable*, optional): Custom stop criteria function. Receives the MOEA instance as context. Returns `True` to halt execution.
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
*   **`perf_spread(*args, metric=None, gen=-1, ...)`**:
    *   Visualizes **Performance Contrast** using Boxplots with A12/p-value annotations.
*   **`perf_density(*args, metric=None, gen=-1, ...)`**:
    *   Plots the probability distribution of a performance metric via KDE.

### **2.3. Stratification (`mb.view.strat_*`)**

*   **`strat_ranks(*args, ...)`**:
    *   Permanent Alias: `rankplot`. Shows frequency distribution across dominance ranks.
*   **`strat_caste(*args, metric=None, ...)`**:
    *   Maps Quality vs Density per dominance layer.
*   **`strat_tiers(exp1, exp2=None, ...)`**:
    *   Competitive Duel: joint dominance proportion per global tier.

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

### **`mb.stats.perf_evidence(data1, data2, alternative='two-sided', metric=mb.metrics.hv, gen=-1, **kwargs)`**
Performs the **Mann-Whitney U** rank-sum test (Win Evidence). Returns a `HypothesisTestResult`.

### **`mb.stats.perf_distribution(data1, data2, alternative='two-sided', metric=mb.metrics.hv, gen=-1, **kwargs)`**
Performs the **Kolmogorov-Smirnov (KS)** two-sample test (Performance Distribution). Returns a `HypothesisTestResult`.

### **`mb.stats.perf_probability(data1, data2, metric=mb.metrics.hv, gen=-1, **kwargs)`**
Computes the **Vargha-Delaney $\hat{A}_{12}$** effect size (Win Probability). Returns a `SimpleStatsValue`.

### **`mb.stats.topo_distribution(*args, space='objs', axes=None, method='ks', **kwargs)`**
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
*   **Returns**: `AttainmentSurface` (SmartArray subclass).

### **`mb.stats.topo_gap(exp1, exp2, level=0.5)`**
Calculates the spatial Gap in attainment between two groups.
*   **Methodology**: Based on **EAF Difference** analysis.
*   **Returns**: `AttainmentDiff` object.

### **`mb.stats.strata(data, gen=-1)`**
Performs **Population Strata** (Dominance Layer analysis) based on Pareto dominance.
*   **Returns**: `StratificationResult`.

### **`mb.stats.tier(exp1, exp2, gen=-1)`**
Performs **Joint Stratification** analysis (Tier analysis).
*   **Returns**: `TierResult`.

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
| `mb.casteplot` | `mb.view.strat_caste` | Stratification |
| `mb.tierplot` | `mb.view.strat_tiers` | Stratification |
| `mb.view.topo_dist` | `mb.view.topo_density` | Topography |
| `mb.stats.perf_prob` | `mb.stats.perf_probability` | Performance |
| `mb.stats.perf_dist` | `mb.stats.perf_distribution` | Performance |
| `mb.stats.topo_dist` | `mb.stats.topo_distribution` | Topography |
| `mb.stats.topo_attain` | `mb.stats.topo_attainment` | Topography |

> [!IMPORTANT]
> **Hard Deprecation Policy**: In future versions, the soft-deprecated items above will only produce a `UserWarning` and will no longer execute logic. We strongly recommend updating your research pipelines to the new nomenclature.

