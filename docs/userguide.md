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

## **7. Visual Perspectives: Plotting Made Easy**

The `mb.view` system is **Polymorphic**. Plotting functions are designed to accept complex objects and automatically "climb" the data hierarchy to extract the relevant scientific information.

### **Spatial Perspective (`spaceplot`)**
Visualizes the distribution of solutions in the objective space (2D or 3D). The `spaceplot` function is a polymorphic chameleon that understands your intention based on the "state" of the data you pass to it.

#### **Level 1: The Raw Values (The Atom)**
At the most fundamental level, you can pass a raw numerical matrix ($N \times M$ NumPy array). In this state, the graph is a purely mathematical representation without research context.
```python
mb.view.spaceplot(my_numpy_array) 
```

#### **Level 2: Explicit Extraction (Manager Mode)**
You can use the **Manager Mode** to build precise chains that terminate in the numerical space (`.objectives` or `.vars`). This allows for surgical comparisons of specific subsets:
*   `exp.pop().non_dominated().objectives`: A didactic chain representing the total cloud, filtered by elite status, extracted as numbers.
*   `exp.dominated().objectives`: Visualizes the "shadow" of the experiment—those individuals that were surpassed.
*   `exp.non_dominated().objectives`: The aggregated elite among all independent runs.
*   `exp.non_dominated(100).objectives`: A historical snapshot of the elite at generation 100.

```python
# Multiple explicit extracts in one plot
mb.view.spaceplot(
    exp.non_dominated().objs, 
    exp.dominated(50).objs
)
```

#### **Level 3: The Data Container (The Parent Object)**
Here the polimorphism begins. Instead of requesting the numbers explicitly, you pass the object that contains them:
*   `mb.view.spaceplot(exp.non_dominated())`
*   **The Intelligence**: The method detects that it received a `Population` object and automatically infers that the spatial interest lies in its objectives. It performs the access to `.objectives` internally.

```python
# Passing the population object directly
elite = exp.non_dominated()
mb.view.spaceplot(elite)
```

#### **Level 4: Shortcuts and Canonical Equivalents**
For high productivity, MoeaBench provides shortcuts that serve as stable, research-oriented equivalents to the technical chains above:
*   **`exp.front()`**: The canonical equivalent of `exp.non_dominated().objectives`. It is the primary tool for seeing the research outcome.
*   **`exp.optimal_front()`**: The canonical equivalent of `exp.optimal().objectives`, representing the theoretical truth from the literature.

```python
# The standard "research view"
mb.view.spaceplot(exp.front(), exp.optimal_front())
```

#### **Level 5: The Experiment Manager (The Total Abstraction)**
At the highest level of the hierarchy, you pass the `exp` object directly.
*   `mb.view.spaceplot(exp)`
*   **Interpretation**: In this state, the framework interprets your request as a desire to see the **current state of the search**, which is equivalent to calling `exp.pop()`. 
*   **A Critical Distinction**: Since `exp` represents the "total cloud" (all individuals in the state they are), this plot will show the entire population, including dominated points. If you wish to see only the filtered elite, you must be more specific by passing `exp.front()` or `exp.non_dominated()`.

```python
# Quick look at the search cloud
mb.view.spaceplot(exp)
```

#### **Practical Examples with `exp`**
By leveraging these abstractions, you can compose complex scientific comparisons with minimal code:
*   Compare current progress against perfection: `mb.view.spaceplot(exp, exp.optimal())`
*   Highlight the elite over the search mass: `mb.view.spaceplot(exp, exp.front())` (this creates two visual layers, making the Pareto front stand out from the population cloud).

---

### **Temporal Perspective (`timeplot`)**
The `timeplot` follows strictly the same symmetry of abstraction and "intent detection" to visualize the evolution of performance.

1.  **Metric Matrix**: `mb.view.timeplot(my_hv_matrix)` — plots raw performance data.
2.  **Explicit Series Extraction**: `mb.view.timeplot(mb.metrics.hypervolume(exp.non_dominated()))`.
3.  **The Parent Object**: `mb.view.timeplot(exp.non_dominated())` — the function understands it should apply the default metric (Hypervolume) over the generation series of that population.
4.  **The Total Manager**: `mb.view.timeplot(exp)` — the highest abstraction. The function orchestrates the entire statistical analysis: it calculates the performance of every run, computes the mean and variance, and plots the progress of the research across time automatically.

```python
# Abstraction climbing with timeplot
mb.view.timeplot(exp)                    # Aggregate cloud (statistical)
mb.view.timeplot(exp[0])                 # Specific run trajectory
mb.view.timeplot(exp, metric=mb.stats.igd) # Automatic IGD over experiment
```

---

## **8. Statistical and Structural Analysis (`mb.stats`)**

The `mb.stats` module is the analytical engine of MoeaBench. It transforms raw stochastic trajectories into structural insights and scientific evidence, operating with the same polymorphic intelligence as the visualization system.

### **8.1 Stratification and Dominance Analysis**

Stratification is the definitive process of organizing a population into discrete non-domination layers, or Pareto ranks. In MoeaBench, every structural analysis—whether it evaluates the selection pressure of a single solver or the competitive infiltration between two rivals—is rooted in the concept of stratification.

At the core of this analysis are the `strata` and `tier` functions. While `mb.stats.strata(data)` dissects the internal layers of a specific population context, the `mb.stats.tier(exp1, exp2)` function orchestrates a joint-stratification duel. By merging the state of two competitors and ranking them as a unified set, the framework establishes global tiers that reveal the relative dominance in the objective space.

#### **Accessing Structural Data**
The result of a stratification analysis is stored in a `StratificationResult` (or `TierResult` for competitive duels). These objects provide programmatic access to the following properties:

```python
# Perform the analysis
res = mb.stats.strata(exp)
```

The object `res` contains the following metrics and data structures:
*   **`.max_rank`**: The total number of non-domination layers found (search depth).
*   **`.frequencies()`**: A NumPy array representing the distribution percentage of the population across ranks.
*   **`.selection_pressure()`**: A numeric value (0 to 1) estimating the convergence focus based on rank decay.
*   **`.quality_by(metric)`**: A vector of quality values (e.g., IGD or Hypervolume) mapped to each dominance layer.
*   **`.dominance_ratio()`** *(Tier only)*: The relative proportion of each group in the first global rank (the elite).
*   **`.displacement_depth()`** *(Tier only)*: The specific rank index where the rival group begins to appear significantly.

#### **Visualizing the Dominance Hierarchy**
The structural plots are polymorphic lenses that visualize these results. Because stratification metrics typically evaluate the entire search state, they operate over the population cloud.

```python
# 1. Structural Analysis of a single experiment
res = mb.stats.strata(exp)
mb.view.rankplot(res)    # Operates over the cloud (equiv: exp.pop().objs)
mb.view.casteplot(res)   # Operates over the cloud (equiv: exp.pop().objs)

# 2. Competitive Tier Analysis (Joint Strata)
res_tier = mb.stats.tier(exp1, exp2)
mb.view.tierplot(res_tier) # Joint dominance of (exp1.pop().objs + exp2.pop().objs)
```

| Dominance Distribution (`rankplot`) | Relative Efficiency (`casteplot`) |
| :---: | :---: |
| ![Rank Plot](images/rankplot.png) | ![Caste Plot](images/casteplot.png) |

| Competitive Duel (`tierplot`) |
| :---: |
| ![Tier Plot](images/tierplot.png) |

### **8.2 Stochastic Inference (Hypothesis Tests)**

Beyond structural profiles, the laboratory requires rigorous evidence to determine if performance differences are statistically significant across independent trials. The `mb.stats` module provides a suite of stochastic inference tools designed to handle the non-parametric nature of multi-objective performance data.

The `mb.stats.mann_whitney(exp1, exp2)` function serves as the standard rank-sum test for differences in the center of location (median). In addition, the `mb.stats.ks_test(exp1, exp2)` implements the Kolmogorov-Smirnov test to identify disparities in the overall shape of the distributions, such as variance, stability, or bimodality. To quantify the magnitude of these differences, the `mb.stats.a12(exp1, exp2)` function calculates the Vargha-Delaney effect size statistic.

#### **Accessing Inference Meta-data**
Hypothesis test functions return a `HypothesisTestResult` object that encapsulates the scientific findings:

```python
# Perform the inference test
res = mb.stats.ks_test(exp1, exp2)
# (equiv: mb.stats.ks_test(mb.hv(exp1.pop().objs), mb.hv(exp2.pop().objs)))
```

The object `res` contains the following values that can be accessed directly:
*   **`.p_value`**: The calculated probability of the observed results under the null hypothesis.
*   **`.significant`**: A boolean flag that is `True` if the p-value is below the 0.05 significance threshold.
*   **`.a12`**: The effect size statistic (Vargha-Delaney A12), representing the probability of superiority.
*   **`.report()`**: A method that generates a human-readable narrative report ready for research publications.

```python
# Smart Stats: Direct comparison between experiment outcome distributions
res = mb.stats.ks_test(exp1, exp2, metric=mb.metrics.hv)
print(res.report()) 
# (equiv: samples from mb.metrics.hv(exp1.pop().objs) vs mb.metrics.hv(exp2.pop().objs))
```

### **8.3 Topological Matching (`dist_match`)**

While hypothesis tests (Section 8.2) determine "who is better" by comparing scalar performance metrics like Hypervolume, the laboratory often requires a different inquiry: **"Did these algorithms find the same thing?"**.

The `mb.stats.dist_match(*args)` function is designed for this topological inquiry. It performs multi-axial distribution matching to verify if the populations found by different solvers are statistically equivalent in their spatial distribution. This is essential for detecting **multimodality** (different paths to the same result) or verifying the **consistency** of new algorithms against established baselines.

#### **Performance vs. Topology**
It is scientifically possible for two algorithms to have identical Hypervolume (Performance Equivalence) but converge to completely different regions of the objective space (Topological Divergence). Conversely, they might find the same Pareto Front (**`exp.front()`**) but through different sets of decision variables (**`exp.set()`**). `dist_match` allows you to dissect these nuances by comparing distributions axis-by-axis across both spaces.

#### **Cascading Hierarchy in Matching**
Following the MoeaBench abstraction hierarchy, `dist_match` automatically resolves the data context based on the input:

```python
# 1. Level 1: Direct Array Matching
# The most granular form: compares two raw matrices axis-by-axis.
res_raw = mb.stats.dist_match(matrix_a, matrix_b)

# 2. Level 4: Explicit Snapshot Matching
# Uses 'front()' for objective space or 'set()' for decision space.
# It resolves the "Cloud" among all runs before matching.
res_front = mb.stats.dist_match(exp1.front(), exp2.front()) # Objectives
res_set   = mb.stats.dist_match(exp1.set(),   exp2.set())   # Variables

# 3. Level 5: The Experiment Manager (Total Abstraction)
# Automatically extracts exp.front().objs by default (space='objs').
res = mb.stats.dist_match(exp1, exp2) 

# You can change the focus of the Manager comparison via the 'space' selector:
res_vars = mb.stats.dist_match(exp1, exp2, space='vars') # (equiv: exp.set().vars)
```

#### **Analyzing the Match Result**
The function returns a `DistMatchResult` object that provides a dimensional breakdown of the convergence:
*   **`.is_consistent`**: A global boolean indicating if **all** tested axes (objectives or variables) are equivalent.
*   **`.failed_axes`**: A list of indices identifying exactly where the algorithms diverged.
*   **`.p_values`**: A dictionary providing the specific p-value for each dimension.
*   **`.report()`**: A quantitative analysis report that lists each axis and its match status.

```python
res = mb.stats.dist_match(exp1, exp2, method='ks')
print(res.report())

if not res.is_consistent:
    print(f"Divergence detected in dimensions: {res.failed_axes}")
```

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

MoeaBench allows you to persist experiments to disk as compressed ZIP files. This is essential for long-running studies, cross-tool analysis, and sharing experimental protocols. The persistence system supports selective modes to optimize file size and workflow:

```python
# 1. Save the complete state (Config + All Runs)
exp.save("full_study", mode="all")

# 2. Save only the 'Recipe' (No data)
exp.save("protocol", mode="config")

# 3. Load results into an existing setup
exp.load("results", mode="data")
```

#### **Persistence Modes**

*   **`all` (Default)**:
    *   **Behavior**: Persists the entire state, including the problem definition, algorithm configuration, and every execution trajectory (`runs`).
    *   **Utility**: Full project backups or archiving final research results for peer review and total reproducibility.
*   **`config`**:
    *   **Behavior**: Records only the "experimental protocol"—what problem was solved and with which algorithm settings—stripping away all gathered data.
    *   **Utility**: Sharing compact "recipes" of an experiment. On Loading, it updates your experiment's configuration (MOP/MOEA) while **preserving any existing runs**, allowing you to keep your current results while adopting a new protocol.
*   **`data`**:
    *   **Behavior**: Focuses on the results of the execution.
    *   **Utility**: Merging results from different machines or sessions. On Loading, it imports the **execution runs** from the file but **preserves your current configuration**, ensuring you don't overwrite your problem settings with those from the data file.

For details on the underlying file format (CSVs and joblib serialization), see **[Reference: Persistence](reference.md#persistence)**.

---

## **12. References**

*   **[API Reference](reference.md)**: Total technical mapping of the library.
*   **[Pymoo](https://pymoo.org)**: The optimization engine powering built-in algorithms.
*   **[MOPs Manual](mops.md)**: Detailed history and mathematics of built-in benchmarks.
