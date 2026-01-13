<!--
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>

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
`DPF1` - `DPF5` (Disconnected Fronts).
*   **Args**: `M` (Objectives), `D` (Base dims), `K` (Distance vars), `**kwargs`.
*   **Derived**: `N = D + K - 1`.

**Usage Example:**
```python
# Disconnected front in 3D based on 2D manifold
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
