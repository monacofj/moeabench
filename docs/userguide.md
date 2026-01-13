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

## **5. References**
*   **Full API Documentation**: See `docs/reference.md` for exhaustive details on every class and method.
*   **Pymoo**: The optimization engine powering standard algorithms (https://pymoo.org).
