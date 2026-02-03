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

MoeaBench's analytical integrity is anchored in the native, high-performance implementations of foundational benchmarks

*   **[DTLZ]** K. Deb et al. "[Scalable multi-objective optimization test problems](https://doi.org/10.1109/CEC.2002.1007032)." (2002).
*   **[DPF]** L. Zhen et al. "[Multiobjective test problems with degenerate Pareto fronts](https://doi.org/10.48550/arXiv.1806.02706)." (2018).

For mathematical implementation details, see the **[MOPs Guide](mops.md)**

This guide provides a pedagogical journey through the framework. For exhaustive technical specifications of every method and class, please consult the **[API Reference](reference.md)**.


---

## **1. Introduction: The Laboratory Philosophy**

MoeaBench operates on a **Plugin Architecture**. Its purpose is to provide the infrastructure—metrics, statistics, and plots—so you can focus on the core logic of your algorithm.

### **Key Features**
*   **Central Extensibility**: Seamlessly plug in custom problems or algorithms. Your code is the guest, MoeaBench is the host—zero modifications to the core library required.
*   **Many-Objective Support**: Optimized for high-dimensional objective spaces with no artificial numerical traps on $M$ or $N$.
*   **Hybrid Evaluation**: Intelligent metrics that automatically utilize exact methods for standard problems and efficient fallbacks (like Monte Carlo) for many-objective scenarios.
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
mb.view.topo_shape(exp)     
mb.view.perf_history(exp)   
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
mb.view.perf_history(exp[4])
```

For finer control over specific runs or access to individual trajectories, see **[Section 4: The Data Hierarchy](#4-mastery-the-data-hierarchy)**.

### **Control: Custom Stop Criteria**

MoeaBench allows you to inject custom logic to halt the search process based on dynamic conditions (e.g., convergence, time limits, or specific targets). This can be set globally for the experiment or per execution.

The stop function receives the **Algorithm Instance** as its context, allowing access to the current generation (`algo.n_gen`), population (`algo.pop`), and problem (`algo.problem`).

```python
# 1. Define a global criteria (applies to all future runs)
# Stop if we reach generation 50 (ignoring the default max)
exp.stop = lambda algo: algo.n_gen >= 50
exp.run()

# 2. Override for a specific run (e.g., debug mode)
# Check if the first objective of the best solution is negative
exp.run(stop=lambda algo: algo.pop.best_obj[0] < 0.0)

# 3. Disable custom criteria (revert to standard generations)
exp.stop = None
```

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
mb.view.topo_shape(nd.objs, ref.objs)
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

## **7. Scientific Perspectives: The Three Domains**

MoeaBench organizes all analytical tools into three fundamental scientific domains. This taxonomy helps you choose the right "lens" to observe your search process.

### **7.1 Topography (`topo_`)**
> **Focus: The Geography.** "Where are the solutions in the space of objectives?"

This domain treats the search outcomes as physical coordinates. It is used to analyze coverage, convergence, and spatial diversity.

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
mb.view.topo_density(exp1, exp2)
mb.view.topo_gap(exp1, exp2)
```

![Reliability Bands](images/topo_bands.png)
*Figure 1: Search Reliability Corridor showing 50% and 90% attainment bands.*

![Spatial Density](images/topo_density.png)
*Figure 2: Topologic Match analysis using axis-by-axis Kernel Density Estimation.*

![Topologic Gap](images/topo_gap.png)
*Figure 3: Topologic Gap visualizing the spatial coverage difference between two solvers.*

### **7.2 Performance (`perf_`)**
> **Focus: The Utility.** "How well did the algorithms perform according to scalar metrics?"

This domain reduces high-dimensional outcomes into scalar scores (Hypervolume, IGD) to build leaderboards and verify statistical significance.

*   **`perf_history`**: Plots the evolution of a metric over generations, showing the mean trajectory and variance cloud.
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

### **7.3 Stratification (`strat_`)**
> **Focus: The Geology.** "How is the population organized internally?"

Internal structure analysis looks "under the hood" of the Pareto front to understand selection pressure and dominance hierarchies.

*   **`strat_ranks`**: Shows the distribution of individuals across non-domination layers (Ranks).
*   **`strat_caste`**: Maps the relationship between quality and class membership, revealing how the elite differs from the rest of the population.
*   **`strat_tiers`**: A "Duel of Proportions" that merges two algorithms into global tiers to see who dominates whom in direct competition.

```python
# Competitive Tier Analysis
mb.view.strat_tiers(exp1, exp2)
```

---

## **8. Statistical and Structural Analysis (`mb.stats`)**

The `mb.stats` module is the analytical engine of MoeaBench. It transforms raw stochastic trajectories into structural insights and scientific evidence, operating with the same polymorphic intelligence as the visualization system.

### **8.1 Stratification and Dominance Analysis**

Stratification is the definitive process of organizing a population into discrete non-domination layers, or Pareto ranks. In MoeaBench, every structural analysis—whether it evaluates the selection pressure of a single solver or the competitive infiltration between two rivals—is rooted in the concept of stratification.

At the core of this analysis are the `strata` and `tier` functions. While `mb.stats.strata(data)` dissects the internal layers of a specific population context, the `mb.stats.tier(exp1, exp2)` function orchestrates a joint-stratification duel. By merging the state of two competitors and ranking them as a unified set, the framework establishes global tiers that reveal the relative dominance in the objective space.


### **8.2 Structural Data Access**

The result of a stratification analysis is stored in a `StratificationResult` (or `TierResult` for competitive duels). These objects provide programmatic access to the structural properties of the population:

```python
# Perform the analysis
res = mb.stats.strata(exp)

print(f"Search Depth: {res.max_rank}")
print(f"Selection Pressure: {res.selection_pressure():.2f}")
```

#### **Key Properties:**
*   **`.max_rank`**: The total number of non-domination layers found.
*   **`.frequencies()`**: Distribution percentage of the population across ranks.
*   **`.selection_pressure()`**: Numeric value (0 to 1) estimating convergence focus.
*   **`.dominance_ratio()`** *(Tier only)*: Relative proportion of each group in the elite rank.

#### **Visualizing the Dominance Hierarchy (`strat_caste`)**

The `strat_caste` visualization is the most sophisticated tool in the stratification suite. It maps the biological concept of "caste" to the objective space, visualizing the trade-off between **Quantity** (Density) and **Quality** (Performance).

Starting with version `0.8.0`, this tool offers two parametric modes that answer fundamentally different scientific questions.

```python
# 1. Structural Analysis of a single experiment
res = mb.stats.strata(exp)
mb.view.strat_ranks(res)      

# 2. Caste Analysis (The Hierarchy)
mb.view.strat_caste(res, mode='collective')  # Default: The Macro View (Robustness)
mb.view.strat_caste(res, mode='individual')  # The Micro View (Diversity)

# 3. Competitive Tier Analysis (Joint Strata)
res_tier = mb.stats.tier(exp1, exp2)
mb.view.strat_tiers(res_tier) 
```

### **1. The "Collective" View (`mode='collective'`) (Robustness)**
> *"What is the Gross Domestic Product (GDP) of each caste across history?"*

This mode is designed to measure **Stochastic Robustness**. 
*   **The Data Point**: Each point in the distribution represents the **Total Quality** (e.g., Hypervolume) of an entire rank for a *single independent run*.
*   **The Interpretation**:
    *   **Box Height**: Represents variance between runs. A short, flattened box means the algorithm is highly reliable—it delivers the same total quality every time, even if the individual solutions differ.
    *   **Outliers**: In this mode, an outlier represents a **failed run** (a catastrophic event where the whole population collapsed compared to other seeds).
    *   **Usage**: The primary plot for paper reporting, as it proves your method's stability.

### **2. The "Individual" View (`mode='individual'`) (Diversity)**
> *"What is the Per Capita merit of the citizens?"*

This mode is designed to measure **Internal Diversity**.
*   **The Data Point**: Each point represents a **Single Solution** within the population.
*   **The Interpretation**:
    *   **Box Height**: Represents the diversity of quality within the elite. A tall box means the front includes both "super-solutions" and "marginal solutions".
    *   **Statistics (Example: NSGA3 vs SPEA2)**:
        *   **Top Label ($n \approx 94$)**: The average count. Means the algorithm consistently places 94 solutions in the Rank 1 elite.
        *   **Median ($q \approx 0.37$)**: The "center gravity" of the front. 50% of solutions are better than this merit.
        *   **Whiskers (e.g., $0.10 - 0.57$)**: The effective range of the front. The outliers beyond this range are rare "mutant" checks that extend the front.
    *   **Usage**: Debugging and algorithm design. Helps you understand if your algorithm is producing a uniform front or depending on a few "super-individuals".

### **8.3 Programmatic Access to Quantitative Data**

While the `view` functions provide biological and sociological metaphors (Caste, Tiers), the underlying data consists of standard statistical distributions. You can extract these values directly from the result objects:

#### **Extracting Rank and Caste Data**
```python
res = mb.stats.strata(exp)

# 1. Distribution of frequencies (used in strat_ranks)
freqs = res.frequencies()  # [Rank1%, Rank2%, ...]

# 2. Quality profiles (used in strat_caste)
q_vals = res.quality_profile() # Average HV per rank

# 3. Comprehensive Caste Summary (Method-based API)
# This returns exactly the numbers shown in strat_caste2
summary = res.caste_summary(mode='individual')

print(summary.n(1))          # Population count in Rank 1
print(summary.q(1))          # Median quality (level=50)
print(summary.q(1, level=25))# Q1 quality
print(summary.min(1))        # Lower whisker
print(summary.max(1))        # Upper whisker
```

By default, quality metrics are normalized to an `anchor=1.0`. For comparative studies, you should set the anchor to the maximum HV of the best performing algorithm.

#### **Extracting Competitive Metrics (Tiers)**
```python
t_res = mb.stats.tier(exp1, exp2)

print(f"Prop. in Elite (Pole): {t_res.dominance_ratio}") # [PropA, PropB]
print(f"Stochastic Gap: {t_res.gap}")                   # Rank depth of competition
```

![Rank Plot](images/rankplot.png)
*Figure 6: Dominance Distribution showing the internal stratification of the population.*

![Caste Plot](images/casteplot.png)
*Figure 7: Relative Efficiency (Caste) mapping solution quality to objective space.*

![Tier Plot](images/tierplot.png)
*Figure 8: Competitive Duel visualizing joint-stratification between two rival solvers.*

---

## **9. Advanced Analytics: Significance & Robustness**

MoeaBench provides a rich set of non-parametric statistical tools to transform execution data into scientific evidence. In this library, we distinguish between two fundamental types of inquiry:

1.  **Performance Analysis**: Evaluates "who is better" in terms of objective quality (HV, IGD).
2.  **Topologic Analysis**: Evaluates "what was found" in terms of spatial distribution and coverage.

### **9.1. Performance Analysis (`mb.stats.perf_*`)**

These tools operate on scalar performance metrics (usually from a multi-run experiment) to determine the merit and confidence of an algorithm's results.

#### **Statistical Significance (`mb.stats.perf_evidence`)**
This is the primary tool for hypothesis testing. It performs the **Mann-Whitney U** rank test to answer: *"Is the observed difference in performance statistically significant?"*

```python
# Returns a HypothesisTestResult object
res = mb.stats.perf_evidence(exp1, exp2, metric=mb.metrics.hv)
print(res.report()) 

if res.p_value < 0.05:
    print(f"Confidence confirmed for {res.name}")
```

#### **Win Probability (`mb.stats.perf_probability`)**
Calculates the **Vargha-Delaney $\hat{A}_{12}$** effect size. Mathematically, this captures the "Win Probability": the likelihood that a randomly selected execution of Algorithm A will outperform Algorithm B.

*   **A12 > 0.5**: Algorithm A is likely better.
*   **A12 = 0.5**: They are equivalent.
*   **A12 < 0.5**: Algorithm B is likely better.

```python
prob = mb.stats.perf_probability(exp1, exp2, metric=mb.metrics.hv)
print(f"Probability of Exp1 beating Exp2: {prob:.2f}")
```

#### **Distribution Shape (`mb.stats.perf_distribution`)**
Uses the **Kolmogorov-Smirnov (KS)** test to identify if the *shape* of the performance distributions is different (e.g., detecting if one algorithm is unstable).

#### **Example: Proving Superiority**
The following protocol identifies if an improvement is statistically significant and describes its magnitude:

```python
# 1. Test for Significance (Mann-Whitney U)
res = mb.stats.perf_evidence(exp1, exp2, metric=mb.metrics.hv)

# 2. Interpret the Science
if res.is_significant(alpha=0.05):
    print(f"Evidence supports {res.winner} (p={res.p_value:.4f})")
    
    # 3. Calculate Practical Effect (Win Probability)
    a12 = mb.stats.perf_probability(exp1, exp2)
    print(f"Win Probability (A12): {a12:.2f}")
else:
    print("Null Hypothesis cannot be rejected: Algorithms are equivalent.")
```

### **9.2. Topological Analysis (`mb.stats.topo_*`)**

These tools are designed to answer: *"Did these algorithms find the same thing?"* 

#### **Spatial Distribution Matching (`mb.stats.topo_distribution`)**
Verifies if the clouds of solutions found by different solvers are statistically equivalent in their spatial distribution across each dimension.

*   **`method='ks'` (Default)**: Employs the **Kolmogorov-Smirnov** test per axis.
*   **`method='anderson'`**: Uses the **Anderson-Darling k-sample** test (sensitive to tails).
*   **`method='emd'`**: Calculates the **Earth Mover's Distance** (Purely geometric).

#### **Example: Topological Consistency**
Verify if a new "optimized" version of an algorithm is still converging to the correct manifold regions:

```python
# Verify spatial matching across all objectives
res_topo = mb.stats.topo_distribution(exp_baseline, exp_optimized)

if res_topo.is_consistent:
    print("Optimization preserved topological integrity.")
else:
    print(f"Structural collapse detected in axes: {res_topo.failed_axes}")
    print(res_topo.report())
```

---

## **10. Extensibility: Plugging your Algorithm**

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

## **11. Precision: Reproducibility & Seeds**

Scientific benchmarking requires absolute control over randomness. MoeaBench treats random seeds as fundamental metadata.

*   **Determinism**: You can set a base seed in the algorithm: `mb.moeas.NSGA3(seed=42)`.
*   **Multi-Run Sequence**: When running `repeat=N`, MoeaBench uses a deterministic increment sequence: `Run i` uses `base_seed + i`.
*   **Traceability**: Every `Run` object stores the exact seed used for its execution, ensuring any result can be perfectly replicated.

---

## **12. Persistence (`save` and `load`)**

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

## **13. Data Export (CSV)**

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

## **14. References**

*   **[API Reference](reference.md)**: Total technical mapping of the library.
*   **[Pymoo](https://pymoo.org)**: The optimization engine powering built-in algorithms.
*   **[MOPs Manual](mops.md)**: Detailed history and mathematics of built-in benchmarks.
