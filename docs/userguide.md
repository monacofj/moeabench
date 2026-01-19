<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench User Guide

**Welcome to MoeaBench!**

MoeaBench is an **extensible analytical toolkit** designed to host and evaluate multi-objective optimization algorithms. Rather than just a library of pre-built solvers, its central philosophy is to serve as a **scientific laboratory** where users can "plug in" their own MOEAs and MOPs. 

By providing a decoupled architecture, MoeaBench allows you to treat your own algorithm as a modular plugin, benefiting from all built-in metrics, statistical tests, and visualizations without modifying the core framework.

The framework achieves this by organizing stochastic search data into a **structured semantic model**, transforming raw numerical trajectories into **intuitive programmatic access**, narrative-driven results, and professional visualizations.

This guide provides a pedagogical journey through the framework. For exhaustive technical specifications of every method and class, please consult the **[API Reference](reference.md)**.

---

## **1. Introduction: The Laboratory Philosophy**

MoeaBench operates on a **Plugin Architecture**. Its purpose is to provide the infrastructure—metrics, statistics, and plots—so you can focus on the core logic of your algorithm.

### **Key Features**
*   **Central Extensibility**: Seamlessly plug in custom problems or algorithms. Your code is the guest, MoeaBench is the host.
*   **Semantic Data Model**: A strict hierarchy (Experiment → Run → Population) that makes data access surgically precise.
*   **High-Level Delegation**: Powerful shortcuts that bridge the gap between structural data and scientific results.
*   **Polymorphic Visualization**: Plotting tools that "understand" the complex objects you pass to them.
*   **Statistical Rigor**: Built-in support for multi-run aggregation and standard non-parametric tests.

---

## **2. Quick Start: The "Hello World" Path**

The smallest meaningful unit in MoeaBench is the **Experiment**. An experiment establishes the link between a problem (MOP) and an algorithm (MOEA).

### **Hello MoeaBench!**
Let's solve the DTLZ2 benchmark (3 objectives) using the NSGA-III algorithm:

```python
import MoeaBench as mb

# 1. Setup laboratory
exp = mb.experiment()
exp.mop = mb.mops.DTLZ2(M=3) # Standard 3-objective problem
exp.moea = mb.moeas.NSGA3()

# 2. Start search process
exp.run()

# 3. View instant reward
mb.view.spaceplot(exp)
mb.view.timeplot(exp)
```

| Pareto Front (3D) | Convergence History |
| :---: | :---: |
| ![Pareto Front](images/hello_space.png) | ![Convergence](images/hello_time.png) |
| *Spatial Perspective: Final Population* | *Temporal Perspective: Hypervolume Evolution* |

*Note: In this example, `mb.view.spaceplot(exp)` automatically identifies and projects the **final population snapshot** (the state of the search at the last generation).*

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
mb.view.timeplot(exp)
```

| Convergence History |
| :---: |
| ![Convergence History](images/timeplot.png) |
| *Temporal Perspective: Mean Hypervolume with Variance Cloud* |

To plot a specific stochastic trajectory (e.g., the 5th run) instead of the aggregate cloud, simply index the experiment:
```python
mb.view.timeplot(exp[4])
```

For finer control over specific runs or access to individual trajectories, see **[Section 4: The Data Hierarchy](#4-mastery-the-data-hierarchy)**.

---

## **4. Mastery: The Data Hierarchy**

MoeaBench organizes data in a strict hierarchy that mirrors the structure of a scientific study. Understanding this architecture allows for surgical precision in data extraction.

### **The Canonical Selector**
Data access is a journey from the "Manager" down to the raw numbers:

```text
  LAYER:     Experiment  -->     Run     -->     Filter    -->     Space
  OBJECT:      [exp]     -->    [run]    -->    [.pop()]   -->  [.objectives]
  ROLE:      Manager         Trajectory      Snapshot           Numbers
```

Using standard indexing and methods, you can navigate these layers:

*   **Layer 1: Experiment (`exp`)**: The root container holding all executions.
*   **Layer 2: Run (`exp[i]`)**: A specific stochastic trajectory identified by its seed.
*   **Layer 3: Population (`exp.pop(n)`)**: A snapshot of the search at a generation `n`.
*   **Layer 4: Data Space (`exp.pop(n).objs` or `.vars`)**: The raw numerical performance matrix (NumPy arrays).

#### **Single-run access Example**
MoeaBench uses **1-based** indexing for generations in `.pop()`, while `0` refers to the initial population and `-1` refers to the final generation.

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

Instead of manual array slicing, MoeaBench provides **Solution Filters** (Semantic Operators). These filters adjust their scope automatically: when called from an **`Experiment`**, they aggregate results from all runs; when called from a **`Run`**, they target that specific trajectory.

```python
# --- Manager Context (Aggregation Cloud) ---
nd   = exp.non_dominated()    # Elite among ALL runs (Superfront)
dom  = exp.dominated()        # Solutions surpassed by at least one in the cloud
ref  = exp.optimal()          # Analytical reference (Truth)

# --- Single-run access (Specific Trajectory) ---
nd_1 = exp[0].non_dominated() # Elite of the first run only
nd_n = exp.pop(50).non_dominated() # Elite of generation 50 across all runs

# --- Visualization (Extracting Space) ---
mb.view.spaceplot(nd.objs, ref.objs)
```

*Note: In the methods above, you can pass an optional generation index `n` (e.g., `exp.non_dominated(50)`); leave it empty to retrieve the **final** state by default.*

---

## **6. The Power of Delegation: Master Reference**

MoeaBench uses **Delegation** to provide intuitive shortcuts. The `Experiment` manager provides a global perspective by default, while individual `Run` objects provide surgical access.

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

## **6. Visual Perspectives: Plotting Made Easy**

The `mb.view` system is **Polymorphic**. Plotting functions are designed to accept complex objects (like an `Experiment`) and automatically extract the relevant scientific data.

### **Spatial Perspective (`spaceplot`)**
Visualizes the distribution of solutions in the objective space.
```python
mb.view.spaceplot(exp) 
```
*Note: Supports 2D and 3D plotting. Refer to [Reference: spaceplot](reference.md#spaceplot) for advanced options ($interactive$, $title$, etc.).*

### **Temporal Perspective (`timeplot`)**
Visualizes the evolution of metrics over generations.
```python
mb.view.timeplot(exp) # Uses Hypervolume by default
```

### **Structural & Competitive Perspectives**
For deep diagnostics on algorithm behavior:
*   **`rankplot`**: ![Rank Plot](images/rankplot.png) 
    *   Shows how individuals are distributed across dominance ranks.
*   **`casteplot`**: ![Caste Plot](images/casteplot.png)
    *   Visualizes the "Quality vs. Density" profile of population layers.
*   **`tierplot`**: ![Tier Plot](images/tierplot.png)
    *   A comparative "Duel" between two algorithms to see who dominates the elite tiers.

---

## **7. Statistical Analysis ("Smart Stats")**

The `mb.stats` module follows the same polymorphic philosophy. Statistical tests need sample distributions, and MoeaBench extracts them automatically from your experiments.

### **Narrative Reporting**
Statistical tools return **Rich Result Objects** that provide human-readable narratives.

```python
# Compare two algorithms
res = mb.stats.mann_whitney(exp1, exp2)

# Print the narrative report
print(res.report())
```

The report explains significance, provides effect sizes (A12), and offers a diagnostic interpretation. For technical details on the underlying tests, see **[Reference: Stats](reference.md#stats)**.

---

## **8. Advanced Diagnostics: Caste and Tier**

Beyond simple averages, MoeaBench allows you to inspect the "internal health" of the search profile.

### **Stratification & Selection Pressure**
Use `mb.stats.strata(exp)` to analyze how individuals are distributed across non-domination layers.
*   **Selection Pressure**: Quantifies if the algorithm is focusing correctly on the elite.
*   **Caste Profile**: A hierarchical view that reveals the trade-off between **Convergence** and **Search Effort**.

### **The Tier Duel**
The `tierplot` is the definitive way to compare two MOEAs. It reveals which algorithm is truly "infiltrating" the best global ranks and where the rival starts to lose ground.

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
To wrap your own algorithm, inherit from `mb.moeas.MOEA`. By implementing the `solve(mop)` interface, your algorithm gains access to all of MoeaBench's infrastructure, including multi-run management, persistence, and all plotting perspectives.

*For a step-by-step tutorial on building plugins, see the [Reference: Extensibility](reference.md#extensibility).*

---

## **10. Precision: Reproducibility & Seeds**

Scientific benchmarking requires absolute control over randomness. MoeaBench treats random seeds as fundamental metadata.

*   **Determinism**: You can set a base seed in the algorithm: `mb.moeas.NSGA3(seed=42)`.
*   **Multi-Run Sequence**: When running `repeat=N`, MoeaBench uses a deterministic increment sequence: `Run i` uses `base_seed + i`.
*   **Traceability**: Every `Run` object stores the exact seed used for its execution, ensuring any result can be perfectly replicated.

---

## **11. Persistence (`save` and `load`)**

MoeaBench allows you to persist experiments to disk as compressed ZIP files. This is essential for long-running studies and cross-tool analysis.

```python
# Save everything (Trajectories + Config)
exp.save("study_A", mode="all")

# Load only the configuration (to replicate a study with new runs)
new_exp = mb.experiment().load("study_A", mode="config")
```
Supported modes: `all`, `config` (metadata), and `data` (results). For details on the file format, see **[Reference: Persistence](reference.md#persistence)**.

---

## **12. References**

*   **[API Reference](reference.md)**: Total technical mapping of the library.
*   **[Pymoo](https://pymoo.org)**: The optimization engine powering built-in algorithms.
*   **[MOPs Manual](mops.md)**: Detailed history and mathematics of built-in benchmarks.
