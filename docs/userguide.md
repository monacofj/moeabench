<!--
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench User Guide

**Welcome to MoeaBench!**

MoeaBench is a comprehensive Python framework designed for moping **Multi-objective Evolutionary Algorithms (MOEAs)**. It simplifies the process of defining optimization problems, running standard algorithms, and analyzing performance through metrics and visualizations.

---

## **1. Key Features**

*   **Simple Object-Oriented API**: Intuitive structure using `experiment`, `mop`, and `moea` objects.
*   **Standard Library**: Built-in support for classic mops (**DTLZ**, **DPF**) and state-of-the-art algorithms (**NSGA-III**, **MOEA/D**, **SPEA2**, **RVEA**) powered by Pymoo.
*   **High Performance**: Vectorized metric calculations and non-dominated sorting using NumPy.
*   **Extensible**: Easily plug in your own custom problems or algorithms.
*   **Visual Analysis**: Built-in 3D plotting (`spaceplot`) and convergence tracking (`timeplot`).

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

# 2. Configure the components
exp.mop = mb.mops.DTLZ2(M=3)  # 3-Objective DTLZ2
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
*   **Automatic Seed**: If no seed is provided, a random one is generated and saved in the results.
*   **Multi-run Logic**: When using `exp.run(repeat=N)`, MoeaBench automatically ensures each run is independent but deterministic. It uses the base `seed` for the first run and increments it for subsequent runs (`seed + i`). This ensures that a multi-run experiment is perfectly reproducible if the initial seed is fixed.

---

## **3. Analyzing Results**

MoeaBench provides powerful tools to inspect your optimization data.

### **Calculating Metrics**
Use shortcuts to calculate common metrics like Hypervolume (`hv`) or Inverted Generational Distance (`igd`).

```python
# Calculate Hypervolume for the entire run history
hv_matrix = mb.hv(exp)

# Calculate IGD (requires the mop to have a known Pareto Front)
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
Scientific moping requires control over randomness. MoeaBench handles seeds explicitly to ensure reproducibility.

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
MoeaBench is fully extensible. You can define your own MOPs and MOEAs using standard object-oriented inheritance.

#### **1. Custom MOPs**
To create a new problem, inherit from `mb.mops.BaseMOP` and implement the `evaluation` method.

```python
import numpy as np
import MoeaBench as mb

class MyParabola(mb.mops.BaseMOP):
    def __init__(self):
        # M=2 objectives, N=1 variable
        super().__init__(M=2, N=1, xl=-10.0, xu=10.0)

    def evaluation(self, X, n_ieq_constr=0):
        # X shape: (pop_size, N_vars)
        x = X[:, 0]
        
        # Define objectives
        f1 = x**2
        f2 = (x - 2)**2
        
        # Return dict with 'F' (objectives stack)
        return {'F': np.column_stack([f1, f2])}

# Use it
exp.mop = MyParabola()
```

#### **2. Custom MOEAs**
To wrap a new algorithm, inherit from `mb.core.BaseMoea`. If using a Pymoo algorithm, you can use `mb.moeas.BaseMoeaWrapper`.

```python
from pymoo.algorithms.soo.nonconvex.ga import GA

class MyGA(mb.moeas.BaseMoeaWrapper):
    def __init__(self, population=100, generations=200, seed=1):
        # Pass the Pymoo class (GA) to the wrapper
        super().__init__(GA, population, generations, seed)

# Use it
exp.moea = MyGA(population=50)
```

---

---

## **5. Statistical Analysis ("Smart Stats")**

Comparing algorithms requires rigorous testing. MoeaBench provides the **"Smart Stats"** API to perform these comparisons with minimal boilerplate.

### **Functional Comparisons**
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

### **Backward Compatibility**
"Smart Stats" still supports raw NumPy arrays if you have pre-extracted values.

```python
v1 = [0.81, 0.82, 0.83]
v2 = [0.75, 0.74, 0.76]
res = mb.stats.mann_whitney(v1, v2)
```

For a full comparison script, see `examples/example-06.py`.

## **6. References**
*   **Full API Documentation**: See `docs/reference.md` for exhaustive details on every class and method.
*   **Pymoo**: The optimization engine powering standard algorithms (https://pymoo.org).
