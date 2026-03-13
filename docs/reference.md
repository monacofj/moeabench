<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench API Reference Guide

This document provides the exhaustive technical specification for the MoeaBench Library API.

## **Summary**
1.  **[Nomenclature & Abbreviations](#nomenclature)**
2.  **[1. Data Model](#data-model)**
3.  **[2. Architectural Patterns](#architectural-patterns)**
4.  **[3. Visualization Perspectives](#view)**
5.  **[4. MOPs](#mops)**
6.  **[5. Algorithms](#moeas)**
7.  **[6. Metrics](#metrics)**
8.  **[7. Statistics](#stats)**
9.  **[8. Reporting](#reportable)**
10. **[9. System Utilities](#system)**
11. **[10. Persistence & Data Format](#persistence)**
12. **[11. Extensibility](#extensibility)**
13. **[12. References](#references)**
14. **[13. Legacy Support](#legacy)**
15. **[14. Algorithmic Diagnostics](#diagnostics)**

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
| **H_rel** | Relative Hypervolume | Normalized by the **Session Best**. Measures competitive efficiency. (Pre-v0.11.2: `H_ratio`). |
| **H_abs** | Absolute Hypervolume | Normalized by the **Ground Truth**. Measures proximity to theoretical optimum. |
| **IGD** | Inverted Generational Distance | Measure of proximity/convergence to the Ground Truth (Average distance from GT to Solution). |
| **GD** | Generational Distance | Measure of convergence (Average distance from Solution to GT). |
| **SP** | Spacing | Measure of the spread/uniformity of the solution set. |
| **ND-Ratio** | Front Ratio | Proportion of non-dominated individuals (0.0 to 1.0). See `front_ratio`. |
| **EMD** | Earth Mover's Distance | Wasserstein metric measuring topological/distributional similarity. |
| **ADR** | Architecture Decision Record | Document capturing a significant architectural decision. |
| **EAF** | Empirical Attainment Function | Statistical description of the outcomes of stochastic solvers. |


<a name="data-model"></a>
## **1. Data Model**

MoeaBench uses a hierarchical data model: `experiment` $\to$ `Run` $\to$ `Population` $\to$ `SmartArray`. All components are designed to be intuitive and chainable.

<a name="defaults"></a>
### **1.5. Global Configuration (`mb.defaults`)**
The `mb.defaults` object allows centralized control over the library's behavior. These values act as fallback "Standard Configuration"—they are used whenever an explicit value is not provided to a method or constructor.

**Execution Parameters:**
*   `population`: Default population size (default: `100`).
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
import moeabench as mb

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

#### **`.run(repeat=None, diagnose=False, silent=False, stop=None, **kwargs)`**
*   **Description**: Executes the optimization process.
*   **Arguments**:
    *   `repeat` (*int*): Number of independent runs. Defaults to `exp.repeat`.
    *   `diagnose` (*bool*): If `True`, performs automated pathology analysis after execution. Defaults to `False`.
    *   `silent` (*bool*): If `True`, suppresses runtime output (`Running {exp.name}` banner, progress bars, and run-triggered diagnostic prints). Defaults to `False`.
    *   `stop` (*callable*): Custom stop criteria function. Returns `True` to halt.
    *   `**kwargs`: Parameters passed directly to the MOEA engine (e.g., `population`, `generations`, `seed`).
*   **Returns**: `None`.
*   **Runtime Behavior**: By default, the method prints `Running {exp.name}` at the beginning of execution.

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

#### **`.optimal_front(n=500)`** / **`.optimal_set(n=500)`**
*   **Description**: Shortcuts for extracting the non-dominated objectives (front) or variables (set) from the analytically sampled optimum.
*   **Arguments**:
    *   `n` (*int*): Number of points to sample. Defaults to `500`.
*   **Returns**: `SmartArray`.

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

<a name="architectural-patterns"></a>
## **2. Architectural Patterns**

MoeaBench is built on two core patterns designed to minimize friction and maximize scientific insight.

### **2.1. Smart Arguments**
The "Smart Argument" pattern allows functions to be polymorphic and context-aware. Instead of requiring the user to manually extract raw values (e.g., `exp[0].pop().objs`), all analytical and visual functions accept high-level MoeaBench objects (`experiment`, `Run`, `Population`) directly.
*   **Automatic Extraction**: If a metric is requested, the function automatically identifies and extracts the required space (Pareto front for experiments, population objectives for snapshots).
*   **Temporal Slicing (`gens`)**: Numerical arguments passed to `gens` are automatically normalized into slices (e.g., `gens=100` $\to$ `slice(100)`), ensuring intuitive "limit" behavior across the entire API.

### **2.2. Cloud-centric Delegation**
The experiment object aggregates results across multiple runs automatically, providing a statistical "Cloud" perspective of the search.
*   **Scientific Metadata & CC0**: Integrated support for SPDX licenses. If no author is provided, experiments automatically default to **CC0-1.0** to promote scientific common goods.
*   **Introspective Naming**: Unnamed experiments automatically resolve `exp.name` from the caller's variable identifier (e.g., `exp1`), improving reporting and runtime banners without mandatory manual naming.

---

<a name="view"></a>
## **3. Visualization Perspectives (`mb.view`)**

MoeaBench organizes visualization into **Perspectives**. Every plotter in `mb.view` is polymorphic: it accepts `Experiment`, `Run`, `Population` objects or pre-calculated `Result` objects.
All canonical `mb.view.*` functions support `title=None` with a semantic auto-title fallback; pass `title="..."` only for custom labels.

### **2.1. Topographic Analysis (`mb.view`)**

#### **`mb.view.topology(*args, mode='auto', markers=False, show_gt=True, gt=None, ...)`**
*   **Description**: Visualizes solutions in Objective Space (2D or 3D). 
    *   **GT Display Contract**:
        *   `show_gt=True` and `gt=<array>`: uses explicit GT.
        *   `show_gt=True` and no `gt`: tries GT inference from experiment-like inputs.
        *   `show_gt=False`: does not show GT (and ignores `gt` if provided).
    *   **GT Density**: For dense GT rendering, explicitly pass `gt`: `mb.view.topology(exp, show_gt=True, gt=exp.optimal_front(n=5000))`.
*   **Arguments**:
    *   `*args`: One or more `experiment`, `Run`, or `Population` objects.
    *   `mode` (*str*): `'auto'` (detects environment), `'interactive'` (Plotly), or `'static'` (Matplotlib).
    *   `markers` (*bool*): Enables Clinical Quality Markers (Solid Circle = healthy, Open Circle = near-miss, Open Diamond = severe pathology) based on algorithmic health (`q_closeness`). Defaults to `False`. When `True`, the marker reference uses `gt` when `show_gt=True` (or explicit `ref=` if supplied).
    *   `show_gt` (*bool*): Controls GT visualization/inference. Defaults to `True`.
    *   `gt` (*np.ndarray*): Optional explicit GT array used when `show_gt=True`.
*   **Returns**: `Figure` (Plotly or Matplotlib).

#### **`mb.view.bands(*args, levels=[0.1, 0.5, 0.9], ...)`**
*   **Description**: Visualizes reliability bands using Empirical Attainment Functions (EAF).
*   **Arguments**:
    *   `*args`: `experiment` objects to compare.
    *   `levels` (*List[float]*): Attainment levels (probability thresholds) to plot.
*   **Returns**: `Figure`.

#### **`mb.view.gap(exp1, exp2=None, level=0.5, ...)`**
*   **Description**: Highlights the Topologic Gap (coverage difference) between two experiments.
*   **Arguments**:
    *   `exp1`, `exp2`: The two `experiment` objects to compare.
    *   `level` (*float*): The attainment level to visualize. Defaults to `0.5` (median).
*   **Returns**: `Figure`.

### **2.2. Performance Analysis (`mb.view`)**

#### **`mb.view.history(*args, domain='auto', metric=hv, gens=None, title=None, ...)`**
*   **Description**: Domain-dispatched history view for performance or clinical trajectories.
*   **Arguments**:
    *   `*args`: Datasets to compare.
    *   `domain` (*str*): `'auto'`, `'perf'`, or `'clinic'`. Auto dispatches from metric/result type.
    *   `metric` (*callable* or *str*): Performance metric callable or clinical metric name.
    *   `gens` (*int* or *slice*): Specific generations to plot.
    *   `title` (*str*): Optional explicit title override.
*   **Returns**: `Figure`.

#### **`mb.view.spread(*args, metric=None, gen=-1, title=None, alpha=None, ...)`**
*   **Description**: Comparative boxplot perspective for performance metrics or a precomputed `PerfCompareResult`.
*   **Arguments**:
    *   `*args`: Datasets to compare, or a single `PerfCompareResult`.
    *   `metric` (*callable*): Metric to analyze. Defaults to `hv`.
    *   `gen` (*int*): Specific generation index. Defaults to `-1` (final).
    *   `alpha` (*float*): Significance level used for annotations.
    *   `title` (*str*): Optional explicit title override.
*   **Returns**: `Figure`.

#### **`mb.view.density(*args, domain='auto', metric=None, gen=-1, title=None, ...)`**
*   **Description**: Domain-dispatched density view for performance, topology, or clinic; also accepts canonical result objects directly.
*   **Arguments**:
    *   `*args`: Datasets to compare or precomputed result objects (`PerfCompareResult`, `DistMatchResult`, or clinical result).
    *   `domain` (*str*): `'auto'`, `'perf'`, `'topo'`, or `'clinic'`.
    *   `metric` (*callable* or *str*): Performance metric callable or clinical metric name.
    *   `gen` (*int*): Specific generation index. Defaults to `-1`.
    *   `title` (*str*): Optional explicit title override.
*   **Returns**: `Figure`.
*   **Topological dispatch defaults** (when `domain='topo'` or inferred as topological):
    *   `space='objs'`
    *   `axes=None` (auto-selects available dimensions, up to 4 for grid display)

### **2.3. Stratification Analysis (`mb.view`)**

The underlying structural ontology is **layer**. The public chart-oriented views over that internal layer decomposition are `ranks`, `strata`, and `tiers`.

*   **`mb.view.ranks(*args, title=None, show=True, ...)`**:
    *   Shows frequency distribution across dominance ranks.
*   **`mb.view.strata(*args, metric=None, mode='collective', show_quartiles=True, title=None, show=True, ...)`**:
    *   Maps Quality vs Density using parametric modes ('collective' vs 'individual').
*   **`mb.view.tiers(exp1, exp2=None, title=None, show=True, ...)`**:
    *   Competitive Duel: joint dominance proportion per global tier.

### **2.4. Clinical Diagnostics (`mb.view` canonical)**

#### **`mb.view.radar(*targets, ground_truth=None, mode='auto', show=True, title=None, ...)`**
*   **Description**: Holistic validation view mapping one or more sets of 6 Q-Scores to a radar chart.
*   **Arguments**:
    *   `*targets`: `experiment`, `Run`, `DiagnosticResult`, or Q-score containers.
    *   `ground_truth` (*np.ndarray*): Optional Pareto Front for reference.
    *   `mode` (*str*): `'auto'`, `'interactive'`, or `'static'`.
    *   `show` (*bool*): If `True`, renders immediately.
    *   `title` (*str*): Optional explicit title override.
*   **Returns**: `Figure`.

#### **`mb.view.ecdf(target, ground_truth=None, metric="closeness", mode='auto', show=True, title=None, ...)`**
*   **Description**: Plots the Empirical Cumulative Distribution Function of a clinical metric.
*   **Arguments**:
    *   `target`: Input data.
    *   `ground_truth` (*np.ndarray*): Optional Pareto Front for reference.
    *   `metric` (*str*): One of `'closeness'`, `'headway'`, `'coverage'`, `'gap'`, `'regularity'`, or `'balance'`.
    *   `mode` (*str*): `'auto'`, `'interactive'`, or `'static'`.
    *   `show` (*bool*): If `True`, renders immediately.
    *   `title` (*str*): Optional explicit title override.
*   **Returns**: `Figure`.

#### **Clinical Density Via `mb.view.density(..., domain='clinic')`**
*   **Description**: Histogram/density morphology for one physical clinical metric.
*   **Canonical input**: `DiagnosticValue`, `FairResult`, `QResult`, `Run`, `Population`, or `experiment`.

### **3.5. Aesthetic & Styling System**

MoeaBench maintains a high-precision visual identity designed for academic publications. The system handles colors and sizing consistently across Plotly and Matplotlib backends.

#### **Global Theme Configuration**
Use `mb.view.apply_style()` to configure the global visual environment.

*   **`apply_style(theme='moeabench')`**: 
    - Applies the official **MoeaBench Ocean Palette**.
    - Sets Matplotlib's color cycle and grid defaults.
    - Creates and activates the `moeabench` Plotly template.
    - This is the default behavior at library import.

#### **Per-Trace Customization (`marker_styles`)**
For granular control, plotters accept a `marker_styles` list (one dictionary per dataset).

*   **`color`**: Explicit hex or CSS color (e.g., `{'color': '#FF5733'}`).
*   **`symbol`**: Overrides the marker shape (e.g., `'circle'`, `'diamond'`).
*   **`size`**: Overrides the standard sizing logic.

#### **Standard Sizing Rules (Academic Presets)**
When using `topology` or standard plotters, the following rules apply to maintain diagnostic weight:

| Marker Type | Purpose | Plotly Size | Matplotlib Size ($s$) |
| :--- | :--- | :--- | :--- |
| **Standard / GT / Solid** | Ideal/Healthy Solutions | **6** | **24** |
| **Hollow Circles** | Near-miss pathologies | **10** | **35-40** |
| **Diamonds** | Critical failure pathologies | **9** | **31-36** |

> [!NOTE]
> **Visual Weight Balancing**: In MoeaBench, Diamonds are set to size 9 (instead of 10) to visually compensate for their larger geometric area compared to circles, ensuring they don't appear disproportionately large.

---

## **4. MOPs (`mb.mops.*`)**

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

## **5. Algorithms (`mb.moeas.*`)**

Supported: `NSGA3`, `MOEAD`, `SPEA2`, `RVEA`.

**Constructor:**
```python
Algorithm(population=100, generations=300, seed=1, **kwargs)
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
## **6. Metrics (`mb.metrics`)**

Standard multi-objective performance metrics. Functions accept `Experiment`, `Run`, or `Population` objects as input.

### **6.1. Metric Calculation**

#### **`mb.metrics.front_ratio(data, mode='run', gens=None)`**
*   **Description**: Calculates the proportion of non-dominated individuals (Front Size) relative to the total population size (0.0 to 1.0).
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

### `mb.metrics.hv(exp, ref=None, mode='auto', scale='raw', n_samples=100000, gens=None, joint=True)`

Calculates the Hypervolume for an experiment, run, or population. Always constructs a dynamic Bounding Box (BBox) that encompasses the worst solutions found by all provided experiments (including the reference, if any).

*   **Public alias**: `mb.metrics.hypervolume(...)`

**Parameters:**
*   `exp`: The `Experiment`, `Run`, or `Population` object to evaluate.
*   `ref` (optional): Set of reference experiments used exclusively to expand the Bounding Box.
*   `mode` (str): Algorithmic choice (`'auto'`, `'fast'`, `'exact'`).
*   `scale` (str): Narrative perspective for normalization:
    *   `'raw'` (Default): Returns the absolute physical volume dominated within the Global Bounding Box. Ensures volumetric invariance across comparisons by avoiding ratio-induced shifts. Answers: *"How much objective space has been physically conquered?"*
    *   `'rel'`: Divides the physical volume by the maximum volume found in the current session. Forces the best experiment to present a `1.0` ceiling. Analyzes competitive efficiency relative to the session's state-of-the-art. (Deprecated alias: `'ratio'`).
    *   `'abs'`: Normalizes by the **Ground Truth** of the underlying MOP. Requires pre-calibration (via `mop.calibrate()`). Provides a Cross-Session Absolute Score where `1.0` represents mathematical perfection. Answers: *"What is the absolute proximity to the theoretical optimum?"*
*   `gens` (optional): Slice or integer to limit the generation scope.
*   `joint` (*bool*): If `True` (default), uses the union of `exp` and `ref` to establish the bounding box. If `False`, ignores `ref` for normalization, providing an independent (self-referenced) perspective.

#### **`mb.metrics.igd(data, ref=None, gens=None)`**
*   **Description**: Calculates Inverted Generational Distance.
*   **Arguments**:
    *   `data`: Input data.
    *   `ref` (*np.ndarray*): True Pareto Front (automatic if problem-linked).
    *   `gens` (*int* or *slice*): Limit calculation to specific generation(s).
*   **Returns**: `MetricMatrix`.

#### **`mb.metrics.gd(data, ref=None, gens=None)`**
*   **Description**: Calculates Generational Distance.
*   **Arguments**:
    *   `data`: Input data.
    *   `ref` (*np.ndarray*): True Pareto Front (automatic if problem-linked).
    *   `gens` (*int* or *slice*): Limit calculation to specific generation(s).
*   **Returns**: `MetricMatrix`.

#### **`mb.metrics.igdplus(data, ref=None, gens=None)`**
*   **Description**: Calculates Inverted Generational Distance Plus.
*   **Arguments**:
    *   `data`: Input data.
    *   `ref` (*np.ndarray*): True Pareto Front (automatic if problem-linked).
    *   `gens` (*int* or *slice*): Limit calculation to specific generation(s).
*   **Returns**: `MetricMatrix`.

#### **`mb.metrics.gdplus(data, ref=None, gens=None)`**
*   **Description**: Calculates Generational Distance Plus.
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

### **6.2. Unified Metric Analysis API (`MetricMatrix`)**

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
mb.view.history(hv, mode='auto', show_bounds=True)
```

---

---

<a name="stats"></a>
## **7. Statistics (`mb.stats`)**

Utilities for robust non-parametric statistical analysis. Fully supports **Context-Aware Statistics** (takes `Experiment` objects, functions, or arrays).

### **The Rich Result Interface (`StatsResult`)**
All statistical functions in MoeaBench return objects inheriting from `StatsResult`. These objects provide:
*   **`.report()` $\to$ `str`**: Returns a detailed narrative string. By default, it also displays the report appropriately for the environment.
*   **`.report_show(**kwargs)`**: [DEPRECATED] Use `.report()` instead.

---

<a name="reportable"></a>
## **8. The Reporting Contract (`Reportable`)**

MoeaBench enforces a **Standardized Reporting Interface**. Every analytical object (`Experiment`, `MetricMatrix`, `StatsResult`) inherits from the `Reportable` mixin.

### **The Interface**
*   **`.report(show=True, **kwargs) \to str`**: Returns a detailed technical narrative explaining the object's context, data, and scientific meaning.
    *   **`show=True` (Default)**: Automatically detects the environment (Console vs. Jupyter) and displays the report.
    *   **`show=False`**: Silent mode; only returns the string without printing.
*   **`.report_show(**kwargs)`**: [DEPRECATED] Use `.report()` instead.

### **Participating Objects**
1.  **`mb.experiment`**: Summarizes the experimental protocol (MOP, MOEA, Status).
2.  **`mb.metrics.MetricMatrix`**: Summarizes mathematical performance, search dynamics, and stochastic stability.
3.  **`mb.stats.StatsResult`**: Summarizes hypothesis tests, rank structure, strata distribution, tier duels, and topological matching.

> [!NOTE]
> **Transparency Policy (Explainable Verdicts)**
> As of v0.7.7, all narrative reports explicitly state the decision criteria for qualitative judgments.
> *   **Stability**: Shows Coefficient of Variation (e.g., `CV=0.01 < 0.05`).
> *   **Stratification**: Explicitly states displacement depth trigger (`> 10%`).
> *   **Statistics**: Clarifies significance (`p < 0.05`) and effect size thresholds.

The output is something like:

```text
Rank Structure Report

Data             | Depth | Pressure
--------------------------------------
NSGA-II on DTLZ2 |     3 |   0.9412
```

---

### **7.1. Statistical Contrast**

#### **`mb.stats.perf_compare(data1, data2, method='ks', metric=mb.metrics.hv, gen=-1, ...)`**
*   **Description**: Unified performance comparator.
*   **Methods**:
    *   `method='mannwhitney'`: Mann-Whitney U (location shift).
    *   `method='ks'`: Kolmogorov-Smirnov (distribution match).
    *   `method='a12'`: Vargha-Delaney $\hat{A}_{12}$ (win probability/effect).
*   **Arguments**:
    *   `data1`, `data2`: Datasets to compare.
    *   `metric` (*callable*): Target performance metric.
    *   `gen` (*int*): Specific generation to snapshot.
*   **Returns**: `PerfCompareResult`.
*   **Semantic aliases**:
    *   `mb.stats.perf_shift(...)` == `mb.stats.perf_compare(..., method='mannwhitney')`
    *   `mb.stats.perf_match(...)` == `mb.stats.perf_compare(..., method='ks')`
    *   `mb.stats.perf_win(...)` == `mb.stats.perf_compare(..., method='a12')`

### **`mb.stats.topo_compare(*args, space='objs', axes=None, method='ks', alpha=0.05, threshold=0.1, **kwargs)`**
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
*   **Semantic aliases**:
    *   `mb.stats.topo_match(...)` == `mb.stats.topo_compare(..., method='ks')`
    *   `mb.stats.topo_tail(...)` == `mb.stats.topo_compare(..., method='anderson')`
    *   `mb.stats.topo_shift(...)` == `mb.stats.topo_compare(..., method='emd')`
*   **`topo_shift` default threshold**:
    *   If omitted, `threshold=mb.defaults.displacement_threshold` (current default: `0.1`).

### **`mb.stats.attainment(source, level=0.5)`**
Calculates the attainment surface reached by $k\%$ of the runs.
*   **Methodology**: Grounded in **Empirical Attainment Functions (EAF)**.
*   **Methodology**: Grounded in **Empirical Attainment Functions (EAF)**.
*   **Returns**: `AttainmentSurface` (SmartArray subclass). Can be plotted using `mb.view.topology`.

### **`mb.stats.attainment_gap(exp1, exp2, level=0.5)`**
Calculates the spatial Gap in attainment between two groups.
*   **Methodology**: Based on **EAF Difference** analysis.
*   **Returns**: `AttainmentDiff` object.

### **`mb.stats.ranks(*data, gen=-1)`**
Performs **Rank Structure** analysis based on Pareto dominance.
*   **Returns**: `RankCompareResult`.
*   **Canonical view**: `mb.view.ranks(result)`

### **`mb.stats.strata(*data, metric=mb.metrics.hv, mode='collective', gen=-1)`**
Performs **Strata Distribution** analysis, summarizing rank-wise quality with the chosen metric.
*   **Returns**: `StrataCompareResult`.
*   **Canonical view**: `mb.view.strata(result)`
*   **Ontology note**: uses the internal layer decomposition as its structural base.

### **`mb.stats.tiers(data1, data2, gen=-1)`**
Performs **Tier Duel** analysis between two groups in a shared rank system.
*   **Returns**: `TierResult`.
*   **Canonical view**: `mb.view.tiers(result)`

### **`mb.stats.emd(strat1, strat2)`**
Computes the **Earth Mover's Distance** between two strata profiles.


---

## **9. System Utilities (`mb.system`)**

The `system` module provides utilities for environmental inspection.

### **`mb.system.check_dependencies()`**
Prints a detailed report of installed optional dependencies (`pymoo`, `deap`, etc.).

### **`mb.system.info(show=True)`**
Returns a dictionary with environment metadata used for scientific reproducibility and optionally displays it immediately.
*   **show** (*bool*): If `True` (default), prints/renders the environment summary. If `False`, returns the dictionary silently.
*   **Returns**: `dict` with fields such as MoeaBench version, Python version, NumPy version, platform, and timestamp.

### **`mb.system.export_objectives(data, filename=None)`**
Exports objectives to a CSV file.
*   **data**: `Experiment`, `Population`, or raw array.
*   **filename**: Optional custom filename.

### **`mb.system.export_variables(data, filename=None)`**
Exports decision variables to a CSV file.
*   **data**: `Experiment`, `Population`, or raw array.
*   **filename**: Optional custom filename.

### **`mb.system.version(show=True)`**
Returns the current library version string.
*   **show** (*bool*): If `True` (default), prints the version banner to terminal/notebook and returns the version string. If `False`, returns the version silently.

### **`mb.system.output(text, markdown=None)`**
Environment-aware output helper used by non-report utilities.
*   **text**: Plain-text fallback for terminal output.
*   **markdown**: Optional notebook-specific Markdown rendering.
*   **Returns**: The plain-text string that was emitted.

---

<a name="persistence"></a>
## **10. Persistence & Data Format**

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
## **11. Extensibility (Plugin API)**

MoeaBench is designed as a **host framework**. By inheriting from our base classes, your custom logic becomes a "first-class citizen," gaining instant access to the entire analytical suite (metrics, persistence, and specialized plots). The framework employs a host-guest plugin architecture where custom extensions integrate seamlessly with the core logic.

### **11.1. BaseMop (`mb.mops.BaseMop`)**
The abstract skeleton for all problem plugins. To create a new problem, you must inherit from this class and implement its core mathematical definitions.

**Core Mathematical Contract (Methods to Override):**
*   **`evaluation(self, X)`**: Receives a population matrix $X$ ($PopSize \times N$). Must return a dictionary `{'F': obj_matrix}` where `obj_matrix` is ($PopSize \times M$).
*   **`ps(self, n)`**: [MANDATORY for Diagnostics] Samples the true Pareto Set (decision space). 
    *   **Argument `n`**: An integer dictating the number of points to sample along the continuous mathematical manifold.
    *   **Returns**: A NumPy matrix of shape `(n, N)` containing the ideal decision variables.
    *   **Role**: This is the analytical bedrock (Protocol A). It allows the framework to instantly sample the analytical truth instead of relying on exhaustive empirical searches.

**Diagnostic API (Provided by BaseMop):**
*   **`pf(self, n)`**: Samples the true Pareto Front (objective space). By default, it calls your `ps(n)` and maps it through `evaluation()`.
*   **`calibrate(source_baseline=None, source_gt=None, source_search=None, force=False, **kwargs)`**: The engine that certifies your problem. This method implements the three Ground Truth (GT) protocols defined by the framework:

| Protocol | Strategy | Argument | Description |
| :--- | :--- | :--- | :--- |
| **A** | **Analytical** | *None (Default)* | Calls `ps(n)` to generate the GT using formulaic math. |
| **B** | **Static** | `source_gt` | Loads the GT from an external `.csv` file or NumPy array. |
| **C** | **Empirical** | `source_search` | Runs a high-fidelity MOEA (e.g., `NSGA3`) to find the GT on-the-fly. |

**Arguments:**
*   **`source_baseline`**: Path to the Sidecar JSON file (e.g., `'my_mop_M3.json'`). If `None`, it is auto-resolved as `<ProblemName>_M<M>.json` in the problem's directory. This file "freezes" the GT and ECDFs for future use.
*   **`source_gt`**: A path to a CSV file or a NumPy matrix containing objective-space points to be treated as the Absolute Truth.
*   **`source_search`**: An instance of a MOEA (e.g., `mb.moeas.NSGA3()`) used to Discover the GT. The framework will run this algorithm exaustively and filter the results with `.non_dominated()`.
*   **`force`**: If `True`, ignores existing Sidecar JSON and forces re-calibration.

>**Protocol C (Example)**
>If your MOP has no analytical `ps(n)` and no CSV, you can signal an empirical search during calibration:
>```python
>mop = MyComplexMop()
># MoeaBench will discover the truth, freeze it in '<ProblemName>_M<M>.json', 
># and never run the search again.
>mop.calibrate(source_search=mb.moeas.NSGA3(pop=200, gen=1000))
>```

The contract for problems requires implementing a **vectorized** evaluation function. This allows the framework to process entire populations using NumPy's high-performance broadcasting.

**The Contract:**
*   **`__init__(self, M, N, ...)`**: Call `super().__init__` to register the number of objectives ($M$) and variables ($N$).
*   **`evaluation(self, X)`**: Receives a population matrix $X$ ($PopSize \times N$). Must return a dictionary containing the objectives matrix `F` ($PopSize \times M$).

**Example: A Simple Convex Problem**
```python
import numpy as np
from moeabench import mb

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
*   **Narrative Reporting**: If your algorithm provides internal diagnostics, consider returning a `StatsResult` object (or subclass) to leverage the `.report()` system.

**Example: A Random Search Skeleton**
```python
from moeabench import mb
import numpy as np

class RandomSearch(mb.moeas.BaseMoea):
    def evaluation(self):
        # Access problem via self.get_problem()
        mop = self.get_problem()
        
        # 1. Generate random solutions
        X = np.random.uniform(mop.xl, mop.xu, (self.population, mop.N))
        
        # 2. Evaluate using the framework's helper
        res = self.evaluation_benchmark(X)
        
        # 3. Return generational history (List[np.ndarray])
        # For simple plugins, returning a single-element list suffices.
        return [res['F']], [X], res['F'], [res['F']], [X], [np.array([])], [np.array([])]
```
> [!TIP]
> For a detailed walkthrough on implementing and using custom plugins, see **`examples/example_05.py`**.
> To see the **Automated Diagnostics** in action across different algorithmic pathologies, refer to **`examples/example_11.py`**.

---

## **12. References**
For a detailed technical narrative on the implementation history and mathematical nuances of our MOPs, see the **[MOPs Guide](mops.md)**.

*   **[DTLZ]** K. Deb, L. Thiele, M. Laumanns, and E. Zitzler. "[Scalable multi-objective optimization test problems](https://doi.org/10.1109/CEC.2002.1007032)." Proc. IEEE Congress on Evolutionary Computation (CEC), 2002.
*   **[DPF]** L. Zhen, M. Li, R. Cheng, D. Peng, and X. Yao. "[Multiobjective test problems with degenerate Pareto fronts](https://doi.org/10.48550/arXiv.1806.02706)." IEEE Transactions on Evolutionary Computation, vol. 22, no. 5, 2018.

## **13. API Status**

The documentation above reflects the canonical API for the alpha/beta transition:

- `mb.clinic` is the diagnostics namespace.
- `mb.view` exposes only canonical chart names (`topology`, `bands`, `gap`, `density`, `history`, `spread`, `ranks`, `strata`, `tiers`, `ecdf`, `radar`).
- `mb.stats` uses canonical comparators (`perf_compare`, `topo_compare`) and method aliases (`perf_shift`, `perf_match`, `perf_win`, `topo_match`, `topo_shift`, `topo_tail`).
- `summary()` is removed in favor of `report(show=True, full=False)`.


<a name="diagnostics"></a>
## **14. Algorithmic Diagnostics and Pathology Detection**

The `mb.clinic` module is the high-level analytical interface for **Algorithmic Pathology**. Following the pattern established in the `mb.stats` module, all diagnostic outputs implement the **Standardized Reporting Interface** (`Reportable`), providing narrative insights alongside numerical values.

### **12.1. Diagnostic Reporting Interface (`Reportable`)**

Instead of returning raw `float` or `ndarray` values, functions return specialized objects that preserve numerical behavior while providing narrative insights.

*   **`DiagnosticValue`**: The base class for single-metric results.
    *   **Numerical Fallback**: Objects can be cast directly to `float()`.
    *   **`.report()`**: Returns a multi-line Markdown string with clinical labels and insights. By default, it also displays the report.
    *   **`exp.authors`**: `str` Optional author metadata for persistence.
    *   **`exp.license`**: `str` SPDX License ID (e.g., 'MIT').
    *   **`exp.year`**: `int` Publication year.
    *   **`reproducibility`**: `dict` Environment DNA block (Python/NumPy versions, Baseline version, Platform, Timestamp).

| Result Class | Returner functions | Characteristics |
| :--- | :--- | :--- |
| **`FairResult`** | `headway`, `coverage`, etc. | Physical facts, normalized by resolution. |
| **`QResult`** | `q_headway`, `q_coverage`, etc. | Clinical scores $[0, 1]$, categorized into 5 quality tiers. |

---

### **12.2. Context-Aware Dispatch Protocol**

> [!NOTE]
> **Design Background**: For the mathematical rationale behind these metrics (including the "Monotonicity Gate" and "Headway" renaming), see **[ADR 0028: Refined Clinical Diagnostics](../docs/adr/0028-refined-clinical-diagnostics-v0.9.1.md)**.

All functions in `mb.clinic` use a **Context-Aware Dispatch** system (`_resolve_diagnostic_context`) that automatically interprets input data.

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

For detailed definitions, see the [FAIR Metrics documentation](fair.md).

#### **`mb.clinic.headway(data, ref=None, s_k=None)`**
*   **Description**: Measures Convergence Depth ($GD_{95} / s_K$).
*   **Arguments**:
    *   `data`: `experiment`, `Run` or `Population`.
    *   `ref` (*np.ndarray*): Analytical Ground Truth.
    *   `s_k` (*float*): Expected resolution noise floor.
*   **Returns**: `FairResult`.

#### **`mb.clinic.closeness(data, ref=None, s_k=None)`**
*   **Description**: Distribution of point-wise distances to the manifold. Computed optimally using a `scipy.spatial.KDTree` spatial index for $O(N \log M)$ performance on large manifolds.
*   **Arguments**:
    *   `data`: `experiment`, `Run`, `Population`, or raw front array.
    *   `ref` (*np.ndarray*): Analytical Ground Truth.
    *   `s_k` (*float*): Expected resolution noise floor.
*   **Returns**: `FairResult`.

#### **`mb.clinic.coverage(data, ref=None)`** / **`mb.clinic.gap(data, ref=None)`**
*   **Description**: Global coverage ($IGD_{mean}$) and Worst-case holes ($IGD_{95}$).
*   **Arguments**:
    *   `data`: `experiment`, `Run`, `Population`, or raw front array.
    *   `ref` (*np.ndarray*): Analytical Ground Truth.
*   **Returns**: `FairResult`.

#### **`mb.clinic.regularity(data, ref_distribution=None)`**
*   **Description**: Structural Uniformity (Wasserstein distance to uniform lattice).
*   **Arguments**:
    *   `data`: `experiment`, `Run`, `Population`, or raw front array.
    *   `ref_distribution` (*np.ndarray*): Optional ideal nearest-neighbor spacing distribution.
*   **Returns**: `FairResult`.

#### **`mb.clinic.balance(data, centroids=None, ref_hist=None)`**
*   **Description**: Manifold Bias (Jensen-Shannon Divergence of occupancy).
*   **Arguments**:
    *   `data`: `experiment`, `Run`, `Population`, or raw front array.
    *   `centroids` (*np.ndarray*): Optional region centers for occupancy partitioning.
    *   `ref_hist` (*np.ndarray*): Optional reference occupancy histogram.
*   **Returns**: `FairResult`.

---

### **12.3. Clinical Normalization (Q-Scores)**

These metrics answer: *"Is this good or bad?"*
They map physical values to a $[0, 1]$ utility scale using **Offline Baselines**.

*   **Range**: $1.0$ (Ideal) to $0.0$ (Baseline/Random). Negative values indicate pathological failure.
*   **Logic**: $Q = 1 - \text{Correction}(\frac{\text{Fair} - \text{Ideal}}{\text{Random} - \text{Ideal}})$

#### **`mb.clinic.q_headway(data, ...) -> QResult`**
*   **Baselines**: Ideal=$0.0$, Random=$Rand_{50}$ (from repository).
*   **Mapping**: **Log-Linear**. Expands resolution near $Q=1.0$ to distinguish "Super-converged" solutions from merely "Good" ones.

#### **`mb.clinic.q_closeness(data, ...) -> QResult`**
*   **Baselines**: Ideal=$0.0$, Random=$ECDF(R_d)$ (Half-Normal Projected Error Distribution).
*   **Mapping**: **Wasserstein-1**. Computes the topological similarity between the finding distribution and the noise model.
*   **Note (v0.12.0)**: Random noise assumes a positive-only metric space penalty enforced through Half-Normal absolute validation, prohibiting points from bleeding inside the non-dominated Pareto wall.

#### **`mb.clinic.q_coverage`, `mb.clinic.q_gap`, `mb.clinic.q_regularity`, `mb.clinic.q_balance` -> `QResult`**
*   **Baselines**: Ideal=$Uni_{50}$ (Median of FPS subsets), Random=$Rand_{50}$ (Median of Random subsets).
*   **Mapping**: **ECDF**. Uses the full Empirical Cumulative Distribution Function of random sampling to rank the solution.

#### **`mb.clinic.q_headway_points(data, ...)`** / **`mb.clinic.q_closeness_points(data, ...)`**
*   **Description**: Point-wise Q-score helpers returning arrays suitable for colored topology overlays and per-point diagnostics.
*   **Returns**: `np.ndarray`.

---

### **12.4. Automated Algorithmic Audit**

#### **`mb.clinic.audit(target, ground_truth=None, source_baseline=None, quality=True, **kwargs) -> DiagnosticResult`**
*   **Rationale**: The primary entry point for automated performance analysis. It runs the full 6-dimensional clinical suite and synthesizes a high-level verdict.
*   **Arguments**:
    *   `target`: `experiment`, `Run`, `Population`, or compatible raw/object wrapper.
    *   `ground_truth`: Optional GT array or path (`.npy`, `.npz`, `.csv`).
    *   `source_baseline`: Optional JSON/dict baseline source override.
    *   `quality` (*bool*): If `True` (default), computes Q-score layer in addition to FAIR metrics.
*   **Process**:
    1.  Calculates all 6 Q-Scores.
    2.  Applies the **Performance Auditor** expert system.
    3.  Classifies the algorithm into the **8-State Pathology Ontology**.
*   **Return**: A `DiagnosticResult` object that provides the full narrative biopsy via `.report()`.

### **12.5. Baseline Management**

| Function | Args | Description |
| :--- | :--- | :--- |
| **`mb.clinic.register_baselines`** | `source` | Appends a new JSON file or dict to the global baseline registry. |
| **`mb.clinic.reset_baselines`** | None | Clears all custom registrations and reverts to library defaults. |
| **`mb.clinic.use_baselines`** | `source` | **Context Manager**: Temporarily activates a primary baseline source. |
| **`mb.clinic.calibrate`** | `mop, source_baseline=None, source_gt=None, source_search=None, force=False, ...` | Calibrates or refreshes offline clinical baselines for one problem. |

#### **`ReproducibilityWarning`**
*   **Description**: A custom warning issued during `load_offline_baselines()` when a mismatch is detected between the current environment (Python/NumPy) and the environment that generated the baseline. It supports the library's "Fail-Safe" compatibility protocol.

#### **`.reset_baselines()`**
*   **Description**: Clears any custom or sidecar baseline configurations loaded during the session (via `register_baselines` or auto-discovery). 
*   **Behavior**: Reverts the internal baseline cache (`_CACHE`) entirely back to the canonical library defaults (the internal `baselines.json`). Ensures global state cleanliness when running multiple disparate evaluations in the same kernel.
*   **Arguments**: None.
*   **Returns**: `None`.

---

### **12.4. Offline Calibration Method**

moeabench uses a strict **Fail-Closed** calibration system (`baselines.json`).

1.  **Discrete Sampling**: Baselines are pre-computed for finite population sizes $K \in \{10..50, 100, 150, 200\}$.
2.  **$Uni_{50}$ (The Anchor)**: Generated using **Farthest Point Sampling (FPS)** on the Ground Truth. Represents the "best possible" distribution for a discrete set of size $K$.
3.  **$Rand_{50}$ & ECDF (The Noise)**: Generated by Monte Carlo sampling (Uniform randomness or Gaussian blur). Represents the "Physics of Failure".
4.  **Snap Policy**: If an experiment uses a $K$ not in the grid, it snaps to the nearest safe floor (e.g., $K=80 \to 50$) to avoid unfair penalization.
