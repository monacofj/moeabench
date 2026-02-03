<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench User Guide

**Welcome to MoeaBench!**

MoeaBench is an **extensible analytical toolkit** that complements multi-objective optimization research by adding a layer of data interpretation and visualization over standard benchmark engines. The framework establishes an intuitive abstraction layer for configuring and executing sophisticated quantitative analysis, transparently handling normalization, numerical reproducibility, and statistical validation. By transforming raw performance metrics into descriptive, narrative-driven results, it facilitates rigorous algorithmic auditing and promotes systematic, reproducible experimental comparisons.

To support this workflow, the package offers high-level facilities for programmatically establishing benchmark protocols and extracting standardized metrics. These features are augmented by advanced graphical capabilities that produce convergence time-series and interactive 3D Pareto front visualizations, bridging the gap between raw numerical data and actionable scientific insight.

For mathematical implementation of built-in MOPs and MOEAS, see the **[MOPs Guide](mops.md)**.

*   **[DTLZ]** K. Deb et al. "[Scalable multi-objective optimization test problems](https://doi.org/10.1109/CEC.2002.1007032)." (2002).
*   **[DPF]** L. Zhen et al. "[Multiobjective test problems with degenerate Pareto fronts](https://doi.org/10.48550/arXiv.1806.02706)." (2018).



This document provides an introductory guide through the framework. For detailed technical specifications of every method and class, please consult the **[API Reference](reference.md)**.


---

## **1. Introduction: The Laboratory Philosophy**

MoeaBench operates on a **Plugin Architecture**. Its purpose is to provide the infrastructure—metrics, statistics, and plots—so you can focus on the core logic of your algorithm.

### **Key Features**
*   **Built-in Benchmark Suite**: Includes state-of-the-art implementations of foundational benchmarks (**DTLZ** and **DPF**), rigorously validated against the original literature and audited as the project's analytical "ground truth".
*   **Built-in Algorithms**: Provides built-in implementations of well-known, literature-referenced MOEAs (e.g., **NSGA-III**, **MOEA/D**, **SPEA2**).
*   **Plugin Architecture**: Seamlessly plug in your own algorithms (MOEAs) and problems (MOPs) without modifying the core library. Your custom code is the guest, MoeaBench is the host.
*   **Many-Objective Readiness**: Full support for Many-Objective Optimization (MaOO) with no artificial limits on the number of objectives ($M$) or variables ($N$).
*   **Performance & Scalability**: Built-in specialized evaluators that automatically switch between exact metrics and efficient approximations (e.g., Monte Carlo) to ensure computability of costly calculations as complexity increases.
*   **Rigor & Reproducibility**: Transparent handling of calibration and statistical validation to ensure robust and reproducible results.
*   **Interpretative Summaries**: Automatically generates interpretative summaries that complement numerical metrics with narrative insights.
*   **Rich Visualizations**: Produces rich spatial (3D fronts), temporal (convergence performance), and stratification (ranking) visualizations.

---

## **2. Quick Start: The "Hello World" Path**

The smallest meaningful unit in MoeaBench is the **Experiment**. An experiment establishes the link between a problem (MOP) and an algorithm (MOEA).

### **Hello MoeaBench!**
Let's solve the DTLZ2 benchmark (3 objectives) using the NSGA-III algorithm:

```python
import MoeaBench as mb

# 1. Configure an experiment
exp = mb.experiment()                   # Instantiate an experiment
exp.mop = mb.mops.DTLZ2()               # Choose a benchmark problem
exp.moea = mb.moeas.NSGA3()             # Choose an optimization algorithm

# 2. Run the experiment
exp.run()

# 3. View instant reward
mb.view.topo_shape(exp)                 # View the resulting Pareto front  
mb.view.perf_history(exp)               # View the hypervomume convergence
```

![Pareto Front](images/hello_space.png)
*Spatial Perspective: Final Population snapshot projected in 3D.*

![Convergence](images/hello_time.png)
*Temporal Perspective: Hypervolume evolution showcasing convergence.*

*Note: In this example, `mb.view.topo_shape(exp)` automatically identifies and projects the **final population snapshot**.*

---

## **3. Scientific Rigor: Multi-Run Experiments**

In evolutionary optimization, a single run is rarely representative. Stochastic algorithms require multiple independent trials to provide statistically significant conclusions.

### **Repeated Executions**
To execute multiple trials (repeating the experiment with different seeds), use the `repeat` argument:

```python
# Execute 10 independent trials
exp.run(repeat=10)
```

### **Cloud Aggregation**
When handling multiple runs, MoeaBench performs **Cloud Aggregation**. This means that high-level analysis tools automatically process the statistical distribution of all runs collectively.

For instance, visualizing a multi-run experiment showing the mean performance and variance:
```python
mb.view.perf_history(exp)
```

![Convergence History](images/timeplot.png)
*Temporal Perspective: Mean Hypervolume trajectory with shaded Variance Cloud across multiple runs.*

To plot a specific stochastic trajectory (e.g., the 5th run) instead of the aggregate cloud, simply index the experiment:
```python
mb.view.perf_history(exp[4])   # Run number is 0-indexed
```

For finer control over specific runs or access to individual trajectories, see **[Section 4: The Data Hierarchy](#4-mastery-the-data-hierarchy)**.

### **Control: Custom Stop Criteria**

Usually, an experiment runs for a given number of generations (default 300)

MoeaBench allows you to inject custom logic to halt the search process based on dynamic conditions (e.g., convergence, time limits, or specific targets). This can be set globally for the experiment or per execution.

The stop function receives a reference to the **Active Solver** as its context, allowing access to the current state of the optimization, including the generation count (`solver.n_gen`), the current population (`solver.pop`), and the problem definition (`solver.problem`). For an exhaustive list of accessible properties, consult the **[API Reference](reference.md#solver-state)**.

```python
# 1. Define a global criteria (applies to all future runs)
# Stop if we reach generation 50 (ignoring the default max)
exp.stop = lambda solver: solver.n_gen >= 50
exp.run()

# 2. Override for a specific run (e.g., debug mode)
# Check if the first objective of the best solution is negative
exp.run(stop=lambda solver: solver.pop.best_obj[0] < 0.0)

# 3. Disable custom criteria (revert to standard generations)
exp.stop = None
```

The `stop` callback is invoked by the solver at the end of every generation. If the function returns `True`, the search process is immediately terminated.

> [!TIP]
> **Efficiency and Periodicity**: Computing complex stop conditions (such as convergence metrics or hypervolume stability) at every iteration can be computationally expensive. It is often advisable to perform the actual evaluation within a periodic conditional block to minimize overhead.

```python
# Evaluate the stop condition only every 10 generations
def periodic_stop(solver):
    if solver.n_gen % 10 == 0:
        # Perform expensive analysis here
        return check_convergence(solver)
    return False

exp.stop = periodic_stop
exp.run()
```


## **4. The Data Hierarchy**

MoeaBench implements a structured data hierarchy designed to facilitate granular access to simulation results. This architecture defines four abstraction layers, ranging from the high-level experiment container down to the raw numerical arrays.

### **The Data Abstraction Layers**
Data access follows a strictly defined path from the management context to the low-level data structures:

```text
  LAYER:     Experiment  -->     Run     -->     Filter     -->    Space
  OBJECT:    [exp]       -->     [run]    -->    [.pop()]   -->    [.objectives]
  ROLE:      Manager             Trajectory      Snapshot          Numbers
```

Using standard indexing and methods, you can navigate these layers:

*   **Layer 1: Experiment (`exp`)**: The root container holding all executions.
*   **Layer 2: Run (`exp[i]`)**: A specific stochastic trajectory (a run) identified by its seed.
*   **Layer 3: Population (`exp.pop(n)`)**: A snapshot of the search at a generation `n`.
*   **Layer 4: Data Space (`exp.pop(n).objs` or `.vars`)**: The raw numerical performance matrix (NumPy arrays).

#### **Topological Consistency Example**
To verify if an optimized version of an algorithm covers the same objective regions as the baseline (Manifold Integrity):

```python
# Check spatial matching with a strict alpha (match if p > 0.01)
res = mb.stats.topo_distribution(exp_baseline, exp_opt, alpha=0.01)

if res.is_consistent:
    print("Optimization preserved topological integrity.")
```

#### **Single-run access Example**
MoeaBench uses **standard 0-based** indexing for generations in `.pop()`, where `0` refers to the first recorded generation (or initial population) and `-1` refers to the final generation.

```python
# A. Get objectives at generation 100 of the first trial
objs_100 = exp[0].pop(100).objectives

# B. Get final decision variables (explicit index -1)
vars_final = exp[0].pop(-1).variables

# C. Deep extraction: ND variables from final gen of the third run
# [Run 2] -> [Last Pop] -> [ND Filter] -> [Space]
nd_vars = exp[2].pop().non_dominated().variables
```

---

## **5. Solution Filters**

To simplify data extraction, MoeaBench implements **Solution Filters**. These methods automatically adjust their scope based on the calling context: when invoked from an **`Experiment`**, they aggregate results from all runs (Cloud context); when invoked from a **`Run`**, they target that specific trajectory (Local context).

```python
# --- Manager Context (Aggregation Cloud) ---
nd   = exp.non_dominated()    # Elite among ALL runs (Superfront)
dom  = exp.dominated()        # Solutions surpassed by at least one in the cloud
ref  = exp.optimal()          # Analytical reference (Truth)

# --- Single-run access (Specific Trajectory) ---
nd_1 = exp[0].non_dominated() # Elite of the first run only
nd_n = exp.pop(50).non_dominated() # Elite of generation 50 across all runs

# --- Visualization (Extracting Space) ---
mb.view.topo_shape(nd.objs, ref.objs)
```

*Note: In the methods above, you can pass an optional generation index `n` (e.g., `exp.non_dominated(50)`); leave it empty to retrieve the **final** state by default.*

---

## **6. Data Delegation and Shortcuts**
 
MoeaBench employs a delegation mechanism to streamline access to nested data. Attributes accessed at the `Experiment` level are automatically resolved to their logical aggregates (e.g., the superfront of all runs) or to the most recent instance.

| Command | Perspective | Technical Equivalent (Structural Access) |
| :--- | :--- | :--- |
| **`exp.last_run`** | The most recent trajectory. | `exp.runs[-1]` |
| **`exp.last_pop`** | Final population of the last run. | `exp.last_run.pop(-1)` |
| **`exp.front()`**  | **Superfront**: ND objectives across *all* runs. | `exp.pop().non_dominated().objs` |
| **`exp.set()`**    | **Superset**: ND variables across *all* runs. | `exp.pop().non_dominated().vars` |
| **`exp.non_front()`**| **Dominance Cloud**: Concatenated dominated objectives. | `exp.pop().dominated().objs` |
| **`exp.non_set()`**  | **Inverse Cloud**: Concatenated dominated variables. | `exp.pop().dominated().vars` |
| **`exp.objectives`**| Raw cloud objectives (all final runs combined). | `exp.pop().objs` |
| **`exp.variables`** | Raw cloud variables (all final runs combined).| `exp.pop().vars` |
| **`exp.optimal_front()`**| The True (Analytical) Pareto Front. | `exp.optimal().objs` |
| **`exp.optimal_set()`**  | The True (Analytical) Pareto Set. | `exp.optimal().vars` |

### **Single-run access**
To access the same metrics for a specific trial, simply navigate to the run level:
*   `exp.last_run.front()` $\to$ Front of the last run only.
*   `exp[i].non_dominated()` $\to$ Elite of the $i$-th run only.

### **Ergonomic Aliases (Layer 4)**
Regardless of the delegation level, you can always use short aliases to access the raw NumPy data:
*   **`.objs`** $\to$ `.objectives`
*   **`.vars`** $\to$ `.variables`

---

## **7. Scientific Perspectives: The Three Domains**

MoeaBench organizes all analytical tools into three fundamental scientific domains. This taxonomy helps you choose the right "lens" to observe your search process.

### **7.1. Topography (Metric Space Analysis)**
This domain analyzes the spatial properties of the solution set within the objective space, focusing on coverage, convergence, and diversity.

*   **`topo_shape`**: Visualizes the geometry of the Pareto front or the entire population cloud in 2D or 3D.
*   **`topo_bands`**: Visualizes **Search Corridors**. It uses Empirical Attainment Functions (EAF) to show the reliability bands (e.g., the region reached by 50% or 90% of the runs).
*   **`topo_gap`**: Highlights the **Topologic Gap**. Identifies exactly which regions of the objective space one algorithm covers that the other does not.
*   **`topo_density`**: Employs Kernel Density Estimation (KDE) to show the spatial probability of solutions along each axis.

```python
# The "Research Standard" View
mb.view.topo_shape(exp.front(), exp.optimal_front())

# Analyzing Search Reliability (50% and 90% bands)
mb.view.topo_bands(exp, levels=[0.5, 0.9])

# Analyzing Spatial Density and Gaps
# 1. Verification of Convergence (Match)
# Do they cover the same Objective Space?
mb.view.topo_density(exp1, exp2, space='objs', title="Pareto Front Topology (Match)")

# 2. Verification of Strategy (Mismatch)
# Do they use the same variables to get there?
mb.view.topo_density(exp1, exp2, space='vars', title="Search Strategy (Mismatch)")
mb.view.topo_gap(exp1, exp2)
```

![Reliability Bands](images/topo_bands.png)
*Figure 1: Search Reliability Corridor showing 50% and 90% attainment bands.*

![Spatial Density](images/topo_density.png)
*Figure 2: Topological Equivalence Analysis (Generated by `examples/example_10.py`). LEFT: A "Match" in objective space indicates similar convergence. RIGHT: A "Mismatch" in decision space reveals distinct search strategies.*

![Topologic Gap](images/topo_gap.png)
*Figure 3: Topologic Gap visualizing the spatial coverage difference between two solvers.*

### **7.2. Performance (Scalar Metrics)**
This domain reduces high-dimensional outcomes into scalar values (Hypervolume, IGD) to facilitate statistical comparison and ranking.

*   **`perf_history`**: Plots the evolution of a metric over generations, showing the mean trajectory and standard deviation cloud.
*   **`perf_spread`**: Visualizes **Performance Contrast**. It uses Boxplots to compare distributions and automatically annotates them with the **A12 Win Probability** and P-values.
*   **`perf_density`**: Shows the "Form of Luck"—the probability distribution of metric values, identifying if an algorithm is stable or outlier-prone.

#### **Metric Rigor and Interpretation**
MoeaBench prioritizes mathematical honesty. When evaluating performance against a **Ground Truth (GT)**, the following protocols apply:

*   **Tripartite Hypervolume Reporting**: Starting with v0.7.6, Hypervolume is reported using three standardized measures:
    1.  **H_raw**: The physical dominated volume.
    2.  **H_ratio**: Search area coverage (normalized to 1.1 reference).
    3.  **H_rel**: Convergence to the truth.
*   **Performance Saturation (H_rel > 100%)**: This occurs when an algorithm's population fills spatial gaps within the discrete reference sampling of the GT. It is a sign of **Convergence Saturation**—the algorithm has reached the maximum precision allowed by the reference discretization.
*   **The EMD Diagnostic**: Proximity metrics like IGD can be deceptive on degenerate fronts (e.g., DPF family). We use **Earth Mover's Distance (EMD)** as our primary indicator of **Topological Integrity**. A high EMD signal takes precedence over IGD, as it identifies clumping and loss of manifold extents that distance-based metrics might overlook.

```python
# Statistical contrast between two methods
mb.view.perf_spread(exp1, exp2, metric=mb.metrics.hv)

# Metric evolution over time
mb.view.perf_history(exp)

# Probability Distribution (Luck Stability)
mb.view.perf_density(exp1, exp2)
```

![Performance Contrast](images/perf_spread.png)
*Figure 4: Performance Contrast using Boxplots with automated A12 and Significance annotations.*

![Performance Density](images/perf_density.png)
*Figure 5: Performance Distribution (KDE) identifying algorithm stability and outlier sensitivity.*

### **7.3. Stratification (Population Structure)**
This domain examines the internal organization of the population, analyzing selection pressure and non-domination levels (Pareto ranks).

*   **`strat_ranks`**: Shows the distribution of individuals across non-domination layers (Ranks).
*   **`strat_caste`**: Maps the relationship between quality and class membership, revealing how the elite differs from the rest of the population.
*   **`strat_tiers`**: A "Duel of Proportions" that merges two algorithms into global tiers to see who dominates whom in direct competition.

```python
# Competitive Tier Analysis
mb.view.strat_tiers(exp1, exp2)
```

![Rank Distribution](images/rankplot.png)
*Figure 6: Global Rank Distribution showing population density across non-domination layers.*

---

## **8. Statistical and Structural Analysis (`mb.stats`)**

The `mb.stats` module is the analytical engine of MoeaBench. It transforms raw stochastic trajectories into structural insights and scientific evidence, maintaining the same polymorphic interface design as the visualization module.

### **8.1 Stratification and Dominance Analysis**

## **8. Statistical Analysis (`mb.stats`)**

The `mb.stats` module transforms raw stochastic trajectories into scientific evidence. It operates on two main axes: **Population Structure** (how solutions are organized) and **Hypothesis Testing** (statistical validation of performance).

### **8.1. Stratification and Structure**
Stratification organizes a population into discrete non-domination layers (ranks). This analysis reveals the selection pressure and internal hierarchy of the search process.

#### **Analyzing Population Layers (`strata`)**
```python
# Analyze the internal structure of a single experiment
res = mb.stats.strata(exp)
print(f"Deepest Rank: {res.max_rank}")
print(f"Selection Pressure: {res.selection_pressure():.2f}") # (0.0 to 1.0)
```

#### **Visualizing the Hierarchy (`strat_caste`)**
The `strat_caste` plot maps the "Caste System" of the population, visualizing the trade-off between **Quantity** (Density) and **Quality** (Performance).

```python
# 1. Individual Merit (Micro View): Diversity distribution within ranks
mb.view.strat_caste(res, mode='individual', title="Population Merit")

# 2. Stochastic Stability (Macro View): Robustness across multiple runs
mb.view.strat_caste(res, mode='collective', title="Stochastic Robustness")
```

![Caste Individual](images/caste_individual.png)
*Figure 7: Micro-view of the population hierarchy. The box reflects quality distribution (Per Capita), while 'n' indicates the average headcount of each rank.*

![Caste Collective](images/caste_collective.png)
*Figure 8: Macro-view (Stochastic Robustness). The vertical stability of the boxes indicates the algorithm's determinism across independent runs.*


#### **Competitive Analysis (`tier`)**
The `tier` function merges two algorithms into a single set to determine global dominance.
```python
# Who dominates whom?
t_res = mb.stats.tier(exp1, exp2)
print(f"Dominance Ratio: {t_res.dominance_ratio}") 
```

### **8.2. Hypothesis Testing & Significance**
These tools answer the critical question: *"Is the difference purely due to luck?"*

#### **Proving Superiority (Performance Analysis)**
To rigorously compare two algorithms (`exp1` vs `exp2`) based on a metric (e.g., Hypervolume):

```python
# 1. Test for Statistical Significance (Mann-Whitney U)
# Answers: "Is the difference real?"
res = mb.stats.perf_evidence(exp1, exp2, metric=mb.metrics.hv)

if res.is_significant(alpha=0.05):
    print(f"Confirmed: {res.winner} outperforms (p={res.p_value:.4f})")
    
    # 2. Measure Effect Size (Vargha-Delaney A12)
    # Answers: "How often does it win?"
    # A12 > 0.5 indicates exp1 wins; A12 < 0.5 indicates exp2 wins.
    prob = mb.stats.perf_probability(exp1, exp2, metric=mb.metrics.hv)
    print(f"Win Probability: {prob:.2f}")
else:
    print("Result Inconclusive: Algorithms are statistically equivalent.")
```

#### **Testing Topological Consistency**
To verify if two algorithms found the same regions of the objective space (e.g., comparing a baseline against an optimized version):

```python
# Check if spatial distributions match (Kolmogorov-Smirnov test per axis)
topo_res = mb.stats.topo_distribution(exp_baseline, exp_optimized)

if topo_res.is_consistent:
    print("Topological Integrity Preserved.")
else:
    print(f"Structural Deviation detected in axes: {topo_res.failed_axes}")
```

> [!NOTE]
> **Further Reading**: MoeaBench supports a wide array of statistical tests, including Anderson-Darling, Earth Mover's Distance (EMD), and Kolmogorov-Smirnov for distribution shapes. For a complete list of available tests and their mathematical definitions, consult the **[API Reference](reference.md#stats)**.

---

## **9. Extensibility: Plugging your Algorithm**

Extensibility is the core reason for MoeaBench's existence. You use the framework to evaluate **your** code.

### **Custom MOP Plugin**
To add a new problem, inherit from `mb.mops.BaseMop` and implement the `evaluation` method:

```python
class MyProblem(mb.mops.BaseMop):
    def __init__(self):
        super().__init__(M=2, N=10) # 2 objectives, 10 variables
        self.xl = np.zeros(10)      # Decision variable lower bounds
        self.xu = np.ones(10)       # Decision variable upper bounds

    def evaluation(self, X):
        # Must return a dictionary with the objectives matrix 'F'
        # Optional: include constraints matrix 'G'
        f1 = ...
        f2 = ...
        return {'F': np.column_stack([f1, f2])}
```

### **Custom MOEA Plugin**
To wrap your own algorithm, inherit from `mb.moeas.MOEA`. By implementing the `solve(mop)` interface, your algorithm gains access to all of MoeaBench's infrastructure.

---

## **10. Precision: Reproducibility & Seeds**

Scientific benchmarking requires absolute control over randomness. MoeaBench treats random seeds as fundamental metadata.

*   **Determinism**: You can set a base seed in the algorithm: `mb.moeas.NSGA3(seed=42)`.
*   **Multi-Run Sequence**: When running `repeat=N`, MoeaBench uses a deterministic increment sequence: `Run i` uses `base_seed + i`.
*   **Traceability**: Every `Run` object stores the exact seed used for its execution, ensuring any result can be perfectly replicated.

---

## **11. Persistence (`save` and `load`)**

MoeaBench allows you to persist experiments to disk as compressed ZIP files. 

```python
# 1. Save the complete state (Config + All Runs)
exp.save("full_study", mode="all")

# 2. Save only the 'Recipe' (No data)
exp.save("protocol", mode="config")

# 3. Load results into an existing setup
exp.load("results", mode="data")
```

#### **Persistence Modes**

*   **`all` (Default)**: Persists the entire state (Problem + Algorithm + Runs).
*   **`config`**: Records only the "experimental protocol" (MOP/MOEA settings).
*   **`data`**: Focuses on the results of the execution (Runs).

---

## **12. Data Export (CSV)**

MoeaBench provides a dedicated **Export API** in the `mb.system` module for raw numerical results.

```python
# 1. Export results from a named experiment
exp.name = "my_study"
mb.system.export_objectives(exp) # Saves to "my_study_objectives.csv"

# 2. Export with a custom name
mb.system.export_variables(exp, "final_vars.csv")

# 3. Export data from a specific population snapshot
pop = exp.last_pop
mb.system.export_objectives(pop, "final_pop_objs.csv")
```

---

## **13. References**

*   **[API Reference](reference.md)**: Total technical mapping of the library.
*   **[Pymoo](https://pymoo.org)**: The optimization engine powering built-in algorithms.
*   **[MOPs Manual](mops.md)**: Detailed history and mathematics of built-in benchmarks.

---

## **14. Architecture Decision Records (ADR)**

This section documents critical architectural decisions and optimization strategies employed by the framework's default configurations.

### ADR 003: Hybrid Decomposition Strategy for MOEA/D
* **Status**: Accepted
* **Context**: The standard Penalty-based Boundary Intersection (PBI) decomposition function in MOEA/D relies heavily on the penalty parameter $\theta$ to balance convergence and diversity. On degenerate (DTLZ5, DTLZ6), disconnected (DPF series), or biased (DTLZ4) manifolds, PBI with low $\theta$ forces convergence but causes "diversity collapse," where the entire population clusters into a few optimal points or a low-dimensional arc. Increasing $\theta$ prevents collapse but hinders convergence on these complex geometries.
* **Decision**: We adopted a **Hybrid Decomposition Architecture**.
    *   **Standard PBI ($\theta=5.0$)**: Used for well-behaved problems (DTLZ1, DTLZ2) where PBI's efficiency is superior.
    *   **Tchebycheff (TCH)**: Used for degenerate, biased, and disconnected problems (DTLZ3, DTLZ4, DTLZ5, DTLZ6, DPF3). Tchebycheff decomposition naturally prioritizes diversity by minimizing the maximum weighted distance, acting as a "hard constraint" on the direction of search and effectively preventing population collapse without sensitive tuning.
* **Consequences**: MOEA/D now achieves robust manifold coverage on degenerate problems (significant improvement in spread/IGD) at the cost of slightly slower convergence speed compared to aggressive PBI.

### ADR 004: Visual Micro-Jitter for Comparative Isomorphism
* **Status**: Accepted
* **Context**: In high-performance calibration scenarios, modern MOEAs (like NSGA-III and MOEA/D) often converge to near-identical locations on the Pareto front. When plotting these solutions in 3D, the points from the last-plotted algorithm perfectly occlude the points of previous algorithms, creating the false impression that the underlying algorithms failed or are invisible (e.g., NSGA-III "disappearing" behind MOEA/D in DTLZ2).
* **Decision**: We implemented a **Gaussian Micro-Jitter** ($\epsilon \sim N(0, 0.003)$) in the visual reporting engine (`generate_visual_report.py`). This jitter is applied only to the visualization coordinates, not the numerical data.
* **Consequences**: Overlapping populations now appear as a mixed "cloud" of colors rather than a single dominant color, allowing visual confirmation of co-existence and competitive dominance without distorting the global shape of the front.
