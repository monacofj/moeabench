<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench API Reference Guide

This document provides the exhaustive technical specification for the MoeaBench Library API.

## **Summary**
1.  **[Nomenclature & Abbreviations](#nomenclature)**
2.  **[Architectural Patterns](#architectural-patterns)** (Smart Arguments, Cloud Delegation)
3.  **[1. Data Model](#data-model)**
    *   [1.1. Experiment](#experiment)
    *   [1.2. Run](#run)
    *   [1.3. Population](#population)
    *   [1.4. SmartArray](#smartarray)
    *   [1.5. Global Configuration](#defaults)
4.  **[2. Visualization Perspectives](#view)**
    *   [2.1. Topographic Analysis](#topo)
    *   [2.2. Performance Analysis](#perf)
    *   [2.3. Stratification Analysis](#strat)
    *   [2.4. Clinical Diagnostics](#clinic)
5.  **[5. Metrics](#metrics)**
6.  **[6. Statistics](#stats)**
7.  **[12. Diagnostics](#diagnostics)**

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
| **ND-Ratio** | Front Size | Proportion of non-dominated individuals (0.0 to 1.0). See `front_size`. |
| **EMD** | Earth Mover's Distance | Wasserstein metric measuring topological/distributional similarity. |
| **ADR** | Architecture Decision Record | Document capturing a significant architectural decision. |
| **EAF** | Empirical Attainment Function | Statistical description of the outcomes of stochastic solvers. |

---

<a name="architectural-patterns"></a>
## **Architectural Patterns**

MoeaBench is built on two core patterns designed to minimize friction and maximize scientific insight.

### **1. Smart Arguments**
The "Smart Argument" pattern allow functions to be polymorphic and context-aware. Instead of requiring the user to manually extract raw values (e.g., `exp[0].pop().objs`), all analytical and visual functions accept high-level MoeaBench objects (`experiment`, `Run`, `Population`) directly.
*   **Automatic Extraction**: If a metric is requested, the function automatically identifies and extracts the required space (Pareto front for experiments, population objectives for snapshots).
*   **Temporal Slicing (`gens`)**: Numerical arguments passed to `gens` are automatically normalized into slices (e.g., `gens=100` $\to$ `slice(100)`), ensuring intuitive "limit" behavior across the entire API.

*   **Cloud-centric Delegation**: The experiment object aggregates results across multiple runs automatically, providing a statistical "Cloud" perspective of the search.
*   **Scientific Metadata & CC0**: Integrated support for SPDX licenses. If no author is provided, experiments automatically default to **CC0-1.0** to promote scientific common goods.
*   **Introspective Naming**: Experiments attempt to automatically discover their variable name from the caller's scope for clearer reporting.

---

<a name="data-model"></a>
## **1. Data Model**

MoeaBench uses a hierarchical data model: `experiment` $\to$ `Run` $\to$ `Population` $\to$ `SmartArray`. All components are designed to be intuitive and chainable.

<a name="defaults"></a>
### **1.5. Global Configuration (`mb.defaults`)**
The `mb.defaults` object allows centralized control over the library's behavior. These values act as fallback "Standard Configuration"—they are used whenever an explicit value is not provided to a method or constructor.

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

### **1.1. Experiment (`mb.core.experiment.experiment`)**
The top-level container for multi-objective optimization research.

**Properties:**
*   `mop` (*Any*): The problem instance.
*   `moea` (*Any*): The algorithm instance.
*   `stop` (*callable*, optional): Global custom stop criteria function.
*   `repeat` (*int*): Default number of repetitions (default: 1).
*   `runs` (*List[Run]*): Access to all execution results.

**Methods:**

#### **`.run(repeat=None, diagnose=False, stop=None, **kwargs)`**
*   **Description**: Executes the optimization process.
*   **Arguments**:
    *   `repeat` (*int*): Number of independent runs. Defaults to `exp.repeat`.
    *   `diagnose` (*bool*): If `True`, performs automated pathology analysis after execution. Defaults to `False`.
    *   `stop` (*callable*): Custom stop criteria function. Returns `True` to halt.
    *   `**kwargs`: Parameters passed directly to the MOEA engine (e.g., `population`, `generations`, `seed`).
*   **Returns**: `None`.

#### **`.pop(n=-1)`**
*   **Description**: Accesses the aggregate cloud (JoinedPopulation) at generation `n`.
*   **Arguments**:
    *   `n` (*int*): Generation index. Defaults to `-1` (final).
*   **Returns**: `JoinedPopulation`.

#### **`.front(n=-1)`**
*   **Description**: Retrieves non-dominated objectives from the aggregate cloud (Superfront).
*   **Arguments**:
    *   `n` (*int*): Generation index. Defaults to `-1`.
*   **Returns**: `SmartArray` (objectives).

#### **`.optimal(n=500)`**
*   **Description**: Analytical sampling of the true Pareto optimal set and front.
*   **Arguments**:
    *   `n` (*int*): Number of points to sample. Defaults to `500`.
*   **Returns**: `Population`.

#### **`.report(markdown=False)`**
*   **Description**: Narrative report of the experiment configuration and metadata.
*   **Arguments**:
    - `markdown` (*bool*): If `True`, returns rich GitHub-flavored Markdown. Defaults to `False`.
*   **Returns**: `str`.
*   **Scientific Logic**: If no author is provided, automatically assigns the **CC0-1.0** license.

#### **`.save(path, mode='all')`** / **`.load(path, mode='all')`**
*   **Description**: Persists/Restores the experiment state.
*   **Arguments**:
    *   `path` (*str*): Filename or folder.
    *   `mode` (*str*): `'all'`, `'config'`, or `'data'`.

### **1.2. Run (`mb.core.run.Run`)**
Represents a single optimization trajectory.

**Methods:**

#### **`.history(type='nd')`**
*   **Description**: Retrieves the full temporal evolution of the run.
*   **Arguments**:
    *   `type` (*str*): The data type to extract.
        *   `'f'`: All objectives history (Generations x Populations).
        *   `'x'`: All variables history.
        *   `'nd'`: Non-dominated objectives (Pareto Fronts) evolution.
        *   `'nd_x'`: Non-dominated variables (Pareto Sets) evolution.
*   **Returns**: `List[np.ndarray]`.

#### **`.pop(gen=-1)`**
*   **Description**: Returns a snapshot of the population at a specific generation.
*   **Arguments**:
    *   `gen` (*int*): Generation index. Defaults to `-1` (final).
*   **Returns**: `Population`.

#### **`.front(gen=-1)`** / **`.set(gen=-1)`**
*   **Description**: Shortcut for non-dominated objectives/variables.
*   **Arguments**:
    *   `gen` (*int*): Generation index. Defaults to `-1`.
*   **Returns**: `SmartArray`.

<a name="smartarray-and-population"></a>
### **1.3. Population (`mb.core.run.Population`)**
A container for a set of solutions.

**Properties:**
*   `.objectives` (*SmartArray*): Matrix ($N \times M$) of objective values.
*   `.variables` (*SmartArray*): Matrix ($N \times D$) of decision variables.

**Methods:**

#### **`.non_dominated()`** / **`.dominated()`**
*   **Description**: Filters individuals based on Pareto dominance.
*   **Returns**: `Population` (new instance).

#### **`.stratify()`**
*   **Description**: Performs Non-Dominated Sorting (NDS).
*   **Returns**: `np.ndarray` (1-based integer ranks).

### **1.4. SmartArray**
An annotated NumPy array subclass (`np.ndarray`) that encapsulates lineage and usage metadata.
*   **Metadata**: `.name`, `.label`, `.gen`, `.source`.
*   **Behavior**: Behaves exactly like a standard NumPy array for all math operations.

---

<a name="view"></a>
## **2. Visualization Perspectives (`mb.view`)**

MoeaBench organizes visualization into **Perspectives**. Every plotter in `mb.view` is polymorphic: it accepts `Experiment`, `Run`, `Population` objects or pre-calculated `Result` objects.

### **2.1. Topographic Analysis (`mb.view.topo_*`)**

#### **`topo_shape(*args, mode='auto', ...)`**
*   **Permanent Alias**: `spaceplot`.
*   **Description**: Visualizes solutions in Objective Space (2D or 3D).
*   **Arguments**:
    *   `*args`: One or more `experiment`, `Run`, or `Population` objects.
    *   `mode` (*str*): `'auto'` (detects environment), `'interactive'` (Plotly), or `'static'` (Matplotlib).
*   **Returns**: `Figure` (Plotly or Matplotlib).

#### **`topo_bands(*args, levels=[0.1, 0.5, 0.9], ...)`**
*   **Description**: Visualizes reliability bands using Empirical Attainment Functions (EAF).
*   **Arguments**:
    *   `*args`: `experiment` objects to compare.
    *   `levels` (*List[float]*): Attainment levels (probability thresholds) to plot.
*   **Returns**: `Figure`.

#### **`topo_gap(exp1, exp2, level=0.5, ...)`**
*   **Description**: Highlights the Topologic Gap (coverage difference) between two experiments.
*   **Arguments**:
    *   `exp1`, `exp2`: The two `experiment` objects to compare.
    *   `level` (*float*): The attainment level to visualize. Defaults to `0.5` (median).
*   **Returns**: `Figure`.

### **2.2. Performance Analysis (`mb.view.perf_*`)**

#### **`perf_history(*args, metric=hv, ...)`**
*   **Permanent Alias**: `timeplot`.
*   **Description**: Primary convergence perspective (Metric Trajectory).
*   **Arguments**:
    *   `*args`: Datasets to compare.
    *   `metric` (*callable*): Metric to analyze. Defaults to `hv`.
    *   `gens` (*int* or *slice*): Specific generations to plot.
        *   **Standard Metrics**: `mb.metrics.hv`, `mb.metrics.igd`, `mb.metrics.gd`, `mb.metrics.emd`.
        *   **Clinical Metrics**: `mb.diagnostics.headway`, `mb.diagnostics.coverage`, `mb.diagnostics.gap`, `mb.diagnostics.regularity`, `mb.diagnostics.balance`.
    *   `gens` (*int* or *slice*): Limit calculation to specific generation(s). 
        *   Example: `gens=50` (first 50), `gens=slice(20, 80)` (range).
*   **Returns**: `Figure`.

#### **`perf_front_size(*args, mode='run', ...)`**
*   **Description**: Non-Dominated Density Perspective. Tracks how many individuals are on the Pareto front.
*   **Arguments**:
    *   `*args`: Datasets to compare.
    *   `mode` (*str*): `'run'` or `'consensus'`.
*   **Returns**: `Figure`.

#### **`perf_spread(*args, metric=None, gen=-1, ...)`**
*   **Description**: Comparative Boxplot stats with A12 Win Probability.
*   **Arguments**:
    *   `*args`: Datasets to compare.
    *   `metric` (*callable*): Metric to analyze.
    *   `gen` (*int*): Specific generation to snapshot. Defaults to `-1` (final).
*   **Returns**: `Figure`.

### **2.3. Stratification Analysis (`mb.view.strat_*`)**

*   **`strat_ranks(*args, ...)`**:
    *   Permanent Alias: `rankplot`. Shows frequency distribution across dominance ranks.
*   **`strat_caste(*args, metric=None, mode='collective', show_quartiles=True, ...)`**:
    *   Maps Quality vs Density using parametric modes ('collective' vs 'individual').
*   **`strat_caste_deprecated(*args, ...)`**:
    *   [DEPRECATED] Original visualizer. Maps Quality vs Density per dominance layer.
*   **`strat_tiers(exp1, exp2=None, ...)`**:
    *   Competitive Duel: joint dominance proportion per global tier.

### **2.4. Clinical Diagnostics (`mb.view.clinic_*`)**

#### **`clinic_radar(target, ground_truth=None, ...)`**
*   **Description**: Holistic Validation mapping 6 Q-Scores to a radial chart.
*   **Arguments**:
    *   `target`: `experiment` or `Run` object.
    *   `ground_truth` (*np.ndarray*): Optional Pareto Front for reference.
*   **Returns**: `Figure`.

#### **`clinic_ecdf(target, metric="closeness", ...)`**
*   **Description**: Plots the Empirical Cumulative Distribution Function of a clinical metric.
*   **Arguments**:
    *   `target`: Input data.
    *   `metric` (*str*): One of `'closeness'`, `'headway'`, `'coverage'`, `'gap'`, `'regularity'`, or `'balance'`.
*   **Returns**: `Figure`.

#### **`clinic_history(target, metric="closeness", gens=None, ...)`**
*   **Description**: Evolution of a clinical metric over generations.
*   **Arguments**:
    *   `target`: `experiment` or `Run`.
    *   `metric` (*str*): The clinical metric name.
    *   `gens` (*int* or *slice*): Range of generations to plot.
*   **Returns**: `Figure`.

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

### **5.1. Metric Calculation**

#### **`mb.metrics.front_size(data, mode='run', gens=None)`**
*   **Description**: Calculates the proportion of non-dominated individuals (Front Size) relative to the total population size (0.0 to 1.0).
*   **Permanent Alias**: `nd_ratio`.
*   **Arguments**:
    *   `data`: `experiment`, `Run`, or `Population`.
    *   `mode` (*str*):
        - `'run'` (Default): Returns ratio per individual run ($G \times R$).
        - `'consensus'`: Returns the **Consensus Ratio** ($G \times 1$).
    *   `gens`: Temporal limit.
*   **Returns**: `MetricMatrix`.

#### **Scientific Rationale: Consensus Ratio**
The Consensus Ratio ($C_g$) measures algorithmic consistency across $R$ independent runs by evaluating the non-dominance of the aggregated population cloud.

| $C_g$ Value | Interpretation | Scientific Insight |
| :--- | :--- | :--- |
| **High ($\approx 1.0$)** | **Exploratory** | Runs find completely different regions. Potential lack of convergence or vast front. |
| **Low ($\to 1/R$)** | **Redundancy** | Runs converge to the same front. Results are highly reliable and saturated. |
| **Decreasing** | **Competitive Refinement** | Runs are beginning to overlap and dominate each other (Evidence of convergence). |

### `mb.metrics.hv(exp, ref=None, mode='auto', scale='raw', n_samples=100000, gens=None)`

Calculates the Hypervolume for an experiment, run, or population. Always constructs a dynamic Bounding Box (BBox) that encompasses the worst solutions found by all provided experiments (including the reference, if any).

**Parameters:**
*   `exp`: The `Experiment`, `Run`, or `Population` object to evaluate.
*   `ref` (optional): Set of reference experiments used exclusively to expand the Bounding Box.
*   `mode` (str): Algorithmic choice (`'auto'`, `'fast'`, `'exact'`).
*   `scale` (str): Narrative choice for the returned MetricMatrix:
    *   `'raw'` (Default): Returns the absolute physical volume dominated within the BBox. Ensures volumetric invariance for the same BBox, avoiding ratio-induced shifts when worse neighbors are added. Covers the question: *"How much objective space has been physically conquered?"*
    *   `'ratio'` : Divides the physical volume by the maximum volume found in the session. Forces the best experiment to present a `1.0` ceiling. Analyzes competitive efficiency relative to the session's state-of-the-art. 
*   `gens` (optional): Slice or integer to limit the generation scope.

#### **`mb.metrics.igd(data, ref=None, gens=None)`** / **`gd(...)`**
*   **Description**: Calculates Inverted Generational Distance or Generational Distance.
*   **Arguments**:
    *   `data`: Input data.
    *   `ref` (*np.ndarray*): True Pareto Front (automatic if problem-linked).
    *   `gens` (*int* or *slice*): Limit calculation to specific generation(s).
*   **Returns**: `MetricMatrix`.

#### **`mb.metrics.emd(data, ref=None, gens=None)`**
*   **Description**: Earth Mover's Distance (Wasserstein) trajectory.
*   **Arguments**:
    *   `data`: Input data.
    *   `ref` (*np.ndarray*): True Pareto Front.
    *   `gens` (*int* or *slice*): Range of generations.
*   **Returns**: `MetricMatrix`.

### **5.2. Unified Metric Analysis API (`MetricMatrix`)**

Performance functions return a `MetricMatrix`, a multi-dimensional diagnostic object that encapsulates the generational trajectories across all experimental runs. It provides a formal, high-level interface for statistical analysis and temporal navigation.

#### **Indexing and Hierarchy**
The `MetricMatrix` indexing logic is strictly aligned with the `Experiment` hierarchy to preserve conceptual consistency. Every `MetricMatrix` follows a **Run-centric** indexing scheme:
*   **`mm[i]`**: Selects the temporal trajectory of **Run $i$** (consistent with `exp[i]`). This operation returns a new `MetricMatrix` containing all generations for that specific run, preserving the object-oriented diagnostic capabilities.
*   **`len(mm)`**: Returns the number of **Runs** present in the dataset (consistent with `len(exp)`).

#### **Temporal vs. Cross-Sectional Selectors**
To distinguish between temporal trajectories and cross-sectional distributions, the following explicit selectors are provided:
*   **`.gen(n)`**: Returns a NumPy vector representing the distribution of all runs at generation $n$. By default, $n=-1$ selects the final experimental state.
*   **`.run(i)`** (Semantically identical to `mm[i]`): Returns the complete trajectory (all generations) for run $i$ as a NumPy vector.

#### **Statistical Selectors (Academic Reductions)**
These methods reduce the multi-run distribution at a specific generation to a single representative scalar ($float$):
*   **`.mean(n=-1)`**: Computes the arithmetic mean across all runs at generation $n$.
*   **`.std(n=-1)`**: Computes the standard deviation at generation $n$, serving as a measure of algorithmic stability and reliability.
*   **`.best(n=-1)`**: Identifies the optimal performance value at generation $n$. This method is context-aware and automatically handles the minimization/maximization logic intrinsic to the specific metric.
*   **`.last`**: A property shortcut representing the mean performance at the final generation. This is the primary scalar used for peer-reviewed algorithm comparisons.

#### **Advanced Numerical Access**
*   **`.values`**: Provides direct access to the underlying NumPy matrix of shape $(Generations, Runs)$.
*   **Polymorphic Casting**: If the `MetricMatrix` contains a single numerical value (e.g., after a reduction), it can be cast directly to a primitive float using `float(mm)` or formatted using standard numeric specifications (e.g., `f"{mm:.4f}"`).

**Example:**
```python
hv = mb.metrics.hv(exp)
final_gen_dist = hv.gen() # Distribution at final generation
first_run_traj = hv.run(0) # Trajectory of first run

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

Utilities for robust non-parametric statistical analysis. Fully supports **Context-Aware Statistics** (takes `Experiment` objects, functions, or arrays).

### **The Rich Result Interface (`StatsResult`)**
All statistical functions in MoeaBench return objects inheriting from `StatsResult`. These objects provide:
*   **`.report()` $\to$ `str`**: Returns a detailed narrative string. Useful for logging or file output.
*   **`.report_show()`**: Displays the report appropriately for the environment.
    *   **Terminal**: Automatically calls `print(res.report())`.
    *   **Jupyter**: Renders a formatted **Markdown** block using `display(Markdown(...))`.

---

<a name="reportable"></a>
## **7. The Reporting Contract (`Reportable`)**

MoeaBench enforces a **Standardized Reporting Interface**. Every analytical object (`Experiment`, `MetricMatrix`, `StatsResult`) inherits from the `Reportable` mixin.

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

### **6.1. Statistical Contrast**

#### **`mb.stats.perf_evidence(data1, data2, metric=mb.metrics.hv, gen=-1, ...)`**
*   **Description**: Mann-Whitney U rank-sum test (Win Evidence).
*   **Arguments**:
    *   `data1`, `data2`: Datasets to compare.
    *   `metric` (*callable*): Target performance metric.
    *   `gen` (*int*): Specific generation to snapshot.
*   **Returns**: `HypothesisTestResult`.

#### **`mb.stats.perf_probability(data1, data2, metric=mb.metrics.hv, gen=-1, ...)`**
*   **Description**: Vargha-Delaney $\hat{A}_{12}$ effect size (Win Probability).
*   **Arguments**:
    *   `data1`, `data2`: Datasets to compare.
    *   `gen` (*int*): Snapshot generation.
*   **Returns**: `SimpleStatsValue`.

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

MoeaBench is designed as a **host framework**. By inheriting from our base classes, your custom logic becomes a "first-class citizen," gaining instant access to the entire analytical suite (metrics, persistence, and specialized plots). The framework employs a host-guest plugin architecture where custom extensions integrate seamlessly with the core logic.

### **9.1. BaseMop (`mb.mops.BaseMop`)**
The abstract skeleton for all problem plugins.

**Methods to Override:**
*   **`evaluation(X)`**: Must return `{'F': obj_matrix}`.
*   **`ps(n)`**: [MANDATORY for Validation] Returns $n$ Pareto Set decision variable samples.

**API Methods:**
*   **`calibrate(baseline=None, force=False, **kwargs)`**: Performs automated calibration and generates a Sidecar JSON file.
*   **`pf(n)`**: Samples the true Pareto Front (objectives).

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
## **12. Algorithmic Diagnostics and Pathology Detection**

The `mb.diagnostics` module is the high-level analytical interface for **Algorithmic Pathology**. Following the pattern established in the `mb.stats` module, all diagnostic outputs implement the **Standardized Reporting Interface** (`Reportable`), providing narrative insights alongside numerical values.

### **12.1. Diagnostic Reporting Interface (`Reportable`)**

Instead of returning raw `float` or `ndarray` values, functions return specialized objects that preserve numerical behavior while providing narrative insights.

*   **`DiagnosticValue`**: The base class for single-metric results.
    *   **Numerical Fallback**: Objects can be cast directly to `float()`.
    *   **`.report()`**: Returns a multi-line Markdown string with clinical labels and insights.
    *   **`exp.authors`**: `str` Optional author metadata for persistence.
    *   **`exp.license`**: `str` SPDX License ID (e.g., 'MIT').
    *   **`exp.year`**: `int` Publication year.
    *   **`exp.report()`**: `str` Narrative report of configuration.
    *   **`.report_show()`**: Renders the narrative report with rich formatting.

| Result Class | Returner functions | Characteristics |
| :--- | :--- | :--- |
| **`FairResult`** | `headway`, `coverage`, etc. | Physical facts, normalized by resolution. |
| **`QResult`** | `q_headway`, `q_coverage`, etc. | Clinical scores $[0, 1]$, categorized into 5 quality tiers. |

---

### **12.2. Context-Aware Dispatch Protocol**

> [!NOTE]
> **Design Background**: For the mathematical rationale behind these metrics (including the "Monotonicity Gate" and "Headway" renaming), see **[ADR 0028: Refined Clinical Diagnostics](../docs/adr/0028-refined-clinical-diagnostics-v0.9.1.md)**.

All functions in `mb.diagnostics` use a **Context-Aware Dispatch** system (`_resolve_diagnostic_context`) that automatically interprets input data.

*   **Polymorphic Input**:
    *   `Experiment`: Automatically extracts the **Pareto Front** of the last run, the **Ground Truth** ($GT$) from the MOP, and the **Resolution Scale** ($s_K$).
    *   `Run`: Extracts Front, GT, and Scale from a specific execution.
    *   `Population`: Extracts Front and tries to find MOP references.
    *   `np.ndarray`: Treated as a raw Front. Requires manual `ref` (GT) and `s_k` to be safe.

*   **Context Resolution**:
    *   **$GT$ (Ground Truth)**: If not provided explicitly via `ref=...`, the system looks for `.optimal_front()` or `.mop.pf()`.
    *   **$s_K$ (Resolution Scale)**: The "Physics of Resolution". If not provided via `s_k=...`, it looks for `mop.s_k` or `mop.s_fit`. Defaults to `1.0` if unknown.

---

### **12.2. Physical Metrics (Fairview Protocol)**

#### **`headway(data, ref=None, s_k=None)`**
*   **Description**: Measures Convergence Depth ($GD_{95} / s_K$).
*   **Arguments**:
    *   `data`: `experiment`, `Run` or `Population`.
    *   `ref` (*np.ndarray*): Analytical Ground Truth.
    *   `s_k` (*float*): Expected resolution noise floor.
*   **Returns**: `FairResult`.

#### **`closeness(data, ref=None, s_k=None)`**
*   **Description**: Distribution of point-wise distances to the manifold.
*   **Returns**: `np.ndarray` (vector of distances).

#### **`coverage(data, ref=None)`** / **`gap(data, ref=None)`**
*   **Description**: Global coverage ($IGD_{mean}$) and Worst-case holes ($IGD_{95}$).
*   **Returns**: `FairResult`.

#### **`regularity(data, ref_distribution=None)`**
*   **Description**: Structural Uniformity (Wasserstein distance to uniform lattice).
*   **Returns**: `FairResult`.

#### **`balance(data, centroids=None)`**
*   **Description**: Manifold Bias (Jensen-Shannon Divergence of occupancy).
*   **Returns**: `FairResult`.

---

### **12.3. Clinical Normalization (Q-Scores)**

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

### **12.4. Automated Algorithmic Audit**

#### **`audit(data, ground_truth=None) -> DiagnosticResult`**
*   **Rationale**: The primary entry point for automated performance analysis. It runs the full 6-dimensional clinical suite and synthesizes a high-level verdict.
*   **GT Resolution**: If `ground_truth` is a string, it is treated as a file path and loaded automatically (`.npy`, `.npz`, `.csv`).
*   **Process**:
    1.  Calculates all 6 Q-Scores.
    2.  Applies the **Performance Auditor** expert system.
    3.  Classifies the algorithm into the **8-State Pathology Ontology**.
*   **Return**: A `DiagnosticResult` object that provides the full narrative biopsy via `.report_show()`.

### **12.5. Baseline Management**

| Function | Args | Description |
| :--- | :--- | :--- |
| **`mb.diagnostics.register_baselines`** | `source` | Appends a new JSON file or dict to the global baseline registry. |
| **`mb.diagnostics.reset_baselines`** | None | Clears all custom registrations and reverts to library defaults. |
| **`mb.diagnostics.use_baselines`** | `source` | **Context Manager**: Temporarily activates a primary baseline source. |

---

### **12.4. Offline Calibration Method**

MoeaBench uses a strict **Fail-Closed** calibration system (`baselines_v4.json`).

1.  **Discrete Sampling**: Baselines are pre-computed for finite population sizes $K \in \{10..50, 100, 150, 200\}$.
2.  **$Uni_{50}$ (The Anchor)**: Generated using **Farthest Point Sampling (FPS)** on the Ground Truth. Represents the "best possible" distribution for a discrete set of size $K$.
3.  **$Rand_{50}$ & ECDF (The Noise)**: Generated by Monte Carlo sampling (Uniform randomness or Gaussian blur). Represents the "Physics of Failure".
4.  **Snap Policy**: If an experiment uses a $K$ not in the grid, it snaps to the nearest safe floor (e.g., $K=80 \to 50$) to avoid unfair penalization.
