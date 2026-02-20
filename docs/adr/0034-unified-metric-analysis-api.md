# ADR 0034: Unified Metric Analysis API and Indexing Consistency

## Status
Proposed/Accepted (v0.11.0)

## Context
In MoeaBench, an `Experiment` is a collection of `Run` objects. Correspondingly, a performance metric (e.g., Hypervolume, Front Size) is a collection of trajectories, one for each run. 

Prior to v0.11.0, the `MetricMatrix` indexed the **Generation** axis by default (axis 0). This created two significant issues:
1. **Conceptual Inconsistency**: `exp[i]` selected a Run, but `metric(exp)[i]` selected a Generation.
2. **Numerical Sensitivity**: Slicing an integer generation from a matrix of $R$ runs resulted in a 1D array of size $R$. The `MetricMatrix` constructor then misinterpreted this 1D array as a single run with $R$ generations, effectively "transposing" the logic and breaking summary statistics like `.mean()`.

## Decision
We formalize the **Unified Metric Analysis API**, establishing the following architectural principles:

### 1. Indexing Consistency
`MetricMatrix` objects must mirror the indexing hierarchy of the `Experiment` object. 
- `mm[i]` now selects the **Run** `i` (axis 1).
- `len(mm)` now returns the number of **Runs** (axis 1).
- Slicing an integer run preserves the 2D "Generations x 1" shape to maintain object integrity.

### 2. Standardized Selector API
To navigate the multi-dimensional data (Generations x Runs), all metrics must support the following standard selectors:

#### A. Temporal Selection (Generations)
- **`.gen(n)`**: Returns the cross-sectional distribution of all runs at generation `n` (default `-1`).
- **Metric Functions**: Statistical reductions often happen on this axis.

#### B. Trajectory Selection (Runs)
- **`.run(i)`** (or `mm[i]`): Returns the temporal trajectory of a specific run.

#### C. Statistical Reductions (Scalar Result)
- **`.mean(n)`**: Average performance across runs at generation `n`.
- **`.std(n)`**: Stability/Variation across runs at generation `n`.
- **`.best(n)`**: The optimal performance found at generation `n` (aware of min/max logic).
- **`.last`**: (Shortcut) The mean performance at the final generation. This is the primary scalar used for algorithm comparison.

### 3. Numerical Polimorphism
`MetricMatrix` objects containing a single scalar (e.g., the result of `mm.mean()`) should implement `__float__` and `__format__`, allowing them to be used directly in mathematical expressions and formatted strings (e.g., `f"{val:.4f}"`).

## Consequences
- Improving developer intuition: `exp[i]` and `metrics(exp)[i]` now refer to the same logical entity (the i-th run).
- Resolving axis-swapping bugs during temporal slicing.
- Establishing a "First-Class Metric" pattern where metrics are rich diagnostic objects rather than raw numpy arrays.
