<!--
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench User Guide

**Welcome to MoeaBench!**

MoeaBench is an **extensible analytical toolkit** that complements multi-objective optimization research by adding a layer of data interpretation and visualization over standard benchmark engines. It achieves this by organizing stochastic search data into a structured semantic model and transforming raw performance metrics into descriptive, narrative-driven results.

This guide provides an introductory overview for getting started with the framework. For complete documentation of all available methods, classes, and their technical arguments, please refer to the **[API Reference](reference.md)**.

---

## **1. Key Features**

*   **Simple Object-Oriented API**: Intuitive structure using `experiment`, `mop`, and `moea` objects.
*   **Standard Library**: Built-in support for classic mops (**DTLZ**, **DPF**) and state-of-the-art algorithms (**NSGA-III**, **MOEA/D**, **SPEA2**, **RVEA**) powered by Pymoo.
*   **High Performance**: Vectorized metric calculations and non-dominated sorting using NumPy.
*   **Extensible**: Easily plug in your own custom problems or algorithms.
*   **Visual Analysis**: Built-in 3D plotting (`spaceplot`) and convergence tracking (`timeplot`).

### **References & Provenance**
MoeaBench implements standard community benchmarks. For a detailed technical narrative of their implementation, see the **[Benchmarks Guide](benchmarks.md)**:
*   **DTLZ**: Scalable test problems by Deb et al. (2002).
*   **DPF**: Degenerate Pareto Front benchmarks by Zhen et al. (2018).

### **Requirements**
*   Python 3.9+
*   NumPy, SciPy, Matplotlib
*   Pymoo (for core algorithm engines)

---

## **2. Quick Start**

### **Installation**
MoeaBench is a standard Python package. You can install it directly from the source:

```bash
pip install .
```

### **Hello World: Your First Experiment**
Let's solve a simple 3-objective problem (`DTLZ2`) using `NSGA-III`.

```python
import MoeaBench as mb

# 1. Create an Experiment container
exp = mb.experiment()

# 2. Specify the MOP and MOEA to be used, and configure their parameters.
exp.mop = mb.mops.DTLZ2(M=3)  
exp.moea = mb.moeas.NSGA3(population=100, generations=200)

# 3. Run the experiment
# This executes one run and stores the result
exp.run()

# 4. Visualize the result
# Plots the Pareto Front of the last run
mb.spaceplot(exp, title="My First Pareto Front")
```

#### **Reproducibility & Seeds**
To ensure scientific reproducibility, MoeaBench handles random seeds deterministically:
*   **Manual Seed**: If you provide a seed to the MOEA (e.g., `mb.moeas.NSGA3(seed=42)`), it will be used.
- **Automatic Seed**: If no seed is provided, a random one is generated and saved in the results.
- **Multi-run Logic**: When using `exp.run(repeat=N)`, MoeaBench automatically ensures each run is independent but deterministic. It uses the base `seed` for the first run and increments it for subsequent runs (`seed + i`). This ensures that a multi-run experiment is perfectly reproducible if the initial seed is fixed.

---

## **3. Analyzing Results**

MoeaBench provides powerful tools to inspect your optimization data.

### **Calculating Metrics**
Use shortcuts to calculate common metrics like Hypervolume (`hv`) or Inverted Generational Distance (`igd`).

```python
# Calculate Hypervolume for the entire run history (Experiment)
# By default, it uses 'auto' mode (switches to Monte Carlo for M > 6)
hv_matrix = mb.hv(exp)

# Forcing specific modes:
hv_exact = mb.hv(exp, mode='exact')          # Absolute precision (WFG)
hv_fast  = mb.hv(exp, mode='fast', n_samples=10000) # Fast approximation

# Calculate IGD (requires the MOP to have a known Pareto Front)
igd_matrix = mb.igd(exp)
```

### **Visualizing Progress (`timeplot`)**
See how the algorithm converges over time.

```python
# Plot Hypervolume evolution over generations
mb.timeplot(hv_matrix, title="Convergence History")
```

### **Visualizing Solutions (`spaceplot`)**
Inspect the final trade-offs found by the algorithm.

```python
# Compare the Initial Population vs. Final Population
# exp.pop(0) -> Initial
# exp.last_pop -> Final
mb.spaceplot(exp.pop(0), exp.last_pop, title="Evolution")
```

### **Comparing with the Theoretical Limit**
For analytical benchmarks, you can easily plot the **True Pareto Front** to visualize how close your algorithm got to the global optimum.

```python
# exp.optimal() samples the theoretical true PF/PS
mb.spaceplot(exp.optimal(), exp, title="Proximity to Optimal")
```

---

## **4. Advanced Usage**

### **Parameter Tuning (`**kwargs`)**
MoeaBench wrappers are designed to be simple, but they don't block access to power features. You can pass **any** advanced parameter (supported by the underlying engine) directly to the algorithm constructor.

```python
# Tuning Pymoo's NSGA3 internals
algo = mb.moeas.NSGA3(
    population=200, 
    generations=500,
    n_neighbors=15,             # Custom neighbor size
    eliminate_duplicates=True   # Pymoo specific flag
)
```

### **Reproducibility & Seed Management**
Scientific benchmarking requires control over randomness. MoeaBench handles seeds explicitly to ensure reproducibility.

**Single Run**:
When you define an algorithm, you set a **base seed**.
```python
# Base seed = 10
algo = mb.moeas.NSGA3(seed=10) 
exp.run(repeat=1) 
# Result: One run using seed 10.
```

**Multi-Run (Statistical Repeats)**:
When running multiple repetitions for statistical significance, the seed is **automatically incremented** for each run starting from the base seed. This ensures that every run is independent but fully reproducible.

```python
# Base seed = 10
algo = mb.moeas.NSGA3(seed=10)
exp.run(repeat=5)

# Execution Plan:
# Run 1: seed = 10
# Run 2: seed = 11
# Run 3: seed = 12
# ...
# Run 5: seed = 14
```

### **Custom Extensions**
MoeaBench is designed to be easily extensible. You can plug in custom problems and algorithms by following a simple interface contract.

#### **1. Custom MOPs**
Inherit from `mb.mops.BaseMop` and implement the `evaluation` method:

```python
class MyMOP(mb.mops.BaseMop):
    def __init__(self):
        super().__init__(M=2, N=5) # 2 objectives, 5 variables
        self.xl = np.zeros(5)      # Lower bounds
        self.xu = np.ones(5)       # Upper bounds

    def evaluation(self, X):
        # Must return a dictionary with the objectives matrix 'F'
        f1 = np.sum(X**2, axis=1)
        f2 = np.sum((X-1)**2, axis=1)
        return {'F': np.column_stack([f1, f2])}

    # Optional: Analytical Reference Front for IGD/GD
    def ps(self, n_points):
        # Returns decision variables (Pareto Set) to be evaluated as the PF.
        # This is used by mb.igd() and mb.spaceplot() to show the "true" front.
        return np.column_stack([np.linspace(0, 1, n_points)] * self.N)
```

**Guard Mechanisms**: Analytical fronts are sampled **lazily** (only when requested). If a metric like `mb.igd` is called but the MOP does not implement `ps()`, the tool will catch the error and fall back to the best-found front across all runs.

#### **2. Custom MOEAs**
To wrap an external algorithm or implement your own, implement the `solve` interface (which receives the MOP and termination criteria). If you are adapting a Pymoo algorithm, you can use the `BaseMoeaWrapper`:

```python
# Option A: Manual Implementation
class MyAlgorithm(mb.moeas.MOEA):
    def solve(self, mop, termination):
        # mop: Provides .evaluation(X)
        # termination: Stop condition
        ...
        return mb.Population(obj_matrix, var_matrix)

# Option B: Pymoo Wrapper
from pymoo.algorithms.moo.nsga2 import NSGA2

class MyNSGA2(mb.moeas.BaseMoeaWrapper):
    def __init__(self, **kwargs):
        super().__init__(NSGA2, **kwargs)

exp.moea = MyNSGA2(pop_size=50)
```

---

---

## **5. Statistical Analysis ("Smart Stats")**

Comparing algorithms requires systematic testing. MoeaBench provides the **"Smart Stats"** API to perform these comparisons with minimal boilerplate.

### **Functional Comparisons (mann_whitney)**
You can pass `Experiment` or `MetricMatrix` (e.g., returned by `mb.hv`) objects directly to statistical tests. The library automatically handles:
1.  **Metric Calculation**: If an experiment is passed, it uses Hypervolume (`mb.hv`) by default.
2.  **Global Reference**: For experiments, it automatically injects a shared reference point (Global Nadir).
3.  **Extraction**: For both experiments and matrices, it extracts the final generation's distribution for testing.

```python
# The one-liner comparison (Experiments)
res = mb.stats.mann_whitney(exp1, exp2)

# Comparing pre-calculated matrices
hv1 = mb.hv(exp1)
hv2 = mb.hv(exp2)
res = mb.stats.mann_whitney(hv1, hv2) # Automically extracts .gens(-1)
```

> [!TIP]
> **Performance Tip**: Calculating metrics like Hypervolume or IGD for large experiments can be computationally expensive.
> **Smart Stats** are purely functional and do *not* cache results (to ensure accuracy in changing contexts).
> **Best Practice**: Assign metric results to variables (`hv = mb.hv(exp)`) and reuse them, rather than calling `mb.hv(exp)` repeatedly.

### **Detecting Shape Differences (`ks_test`)**
While Mann-Whitney tells you if one algorithm is generally "better," the **Kolmogorov-Smirnov (KS)** test identifies if the distributions have different **shapes**. 

This is useful for spotting:
*   **Stability**: If one algorithm has much higher variance (is less stable).
*   **Bimodality**: If an algorithm has distinct "success" and "failure" modes.

```python
# Check if the performance distributions have different "silhouettes"
res_ks = mb.stats.ks_test(exp1, exp2)
```

### **Customizing the Metric**
You can specify which metric to use by passing the function or a lambda.

```python
# Using IGD (injects common PF automatically)
mb.stats.mann_whitney(exp1, exp2, metric=mb.igd)

# Using a lambda for custom logic
mb.stats.mann_whitney(exp1, exp2, metric=lambda e: mb.hv(e, ref_point=[1.2, 1.2]))

# Passing arguments to the metric directly
mb.stats.mann_whitney(exp1, exp2, metric=mb.gdplus, ref=true_pf)
```

### **Polymorphic Arguments**
"Smart Stats" still supports raw NumPy arrays if you have pre-extracted values.

```python
v1 = [0.81, 0.82, 0.83]
v2 = [0.75, 0.74, 0.76]
res = mb.stats.mann_whitney(v1, v2)
```

### **Rich Results and Narrative Reporting**
All statistical tools in `mb.stats` return **Rich Result Objects**. These objects are designed to be both programmatically powerful and human-centered.

1.  **Lazy Evaluation**: Results are computationally efficient. Metrics (like A12 or Selection Pressure) are only calculated when you actually access the property.
2.  **Narrative Reports**: Every result object has a `.report()` method that prints a formatted summary of the findings, including a diagnosis.
3.  **Programmatic Access**: Every value in the report is available as a property for use in your scripts.

```python
res = mb.stats.mann_whitney(exp1, exp2)

# Programmatic use
if res.significant:
    print(f"Algorithm A is better with effect size {res.a12:.2f}")

# Human-centered report
print(res.report()) 
```

For a full comparison script, see `examples/example-06.py`.

## **6. Advanced Diagnostics**

MoeaBench provides deep insights into the internal "health" of your algorithm's search profile.

### Population Strata
Use `mb.stats.strata` to analyze the distribution of individuals across all dominance ranks (layers). This helps you quantify **Selection Pressure** and detect when an algorithm has stalled or lost diversity.

```python
# Analyze the rank distribution of an experiment
result = mb.stats.strata(exp)

# Acesso programático (Lazy)
pressure = result.selection_pressure # Calculado apenas aqui

# Relatório narrativo (Didático)
print(result.report())

# Visualize the profile
mb.stats.strataplot(result, title="Dominance Layers")
```

### **Advanced Diagnosis: Floating Rank Profile**
Beyond simple distributions, MoeaBench allows you to inspect the **Floating Rank Profile**. Use `mb.rankplot(strat1, strat2)` to visualize the quality and density of each dominance level.

*   **Vertical Position**: Represents the Quality (Default: `mb.hypervolume`).
*   **Bar Height**: Represents the Population Density (how many solutions are in that rank).

This allows you to see both **Convergence** (is the bar high?) and **Search Effort** (is the bar tall?) at a single glance.

### **8. System Utilities (`mb.system`)**

MoeaBench includes a `system` module to monitor your environment and hardware.

```python
# Check library health
mb.system.check_dependencies() # Report on installed solvers
mb.system.version()            # Library version
```

## **7. References**
*   **Full API Documentation**: See `docs/reference.md` for exhaustive details on every class and method.
*   **Pymoo**: The optimization engine powering standard algorithms (https://pymoo.org).
