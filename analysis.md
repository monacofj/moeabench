# MoeaBench Architectural Analysis & Refactoring Proposal

This document summarizes the current state of the MoeaBench library and proposes a transition to a more modern, efficient, and "pythonic" architecture.

## Current State Observations

### 1. Over-Engineered Class Hierarchy
The codebase follows a rigid "Interface -> Implementation" pattern (e.g., `I_MoeaBench` -> `IPL_MoeaBench` -> `MoeaBench`). While robust in static languages, in Python it leads to:
- Excessive boilerplate (`NotImplementedError` stubs).
- Difficult navigation through deep inheritance trees.
- Fragmented logic across many small files.

### 2. Legacy Data Structures (The "CACHE" system)
The core data storage (populations, metrics, trajectories) is handled by legacy classes like `CACHE`, `DATA_conf`, and `RUN`. These structures:
- Use deeply nested lists of lists, making data retrieval complex and prone to errors.
- Force the modern `Run` class to perform expensive and complex "lookups" to extract Numpy arrays.
- Create a heavy dependency on internal legacy logic for simple data access.

### 3. Fragmentation and Redundancy
Utility functions are scattered across dozens of subdirectories (e.g., `MoeaBench/gd/`, `MoeaBench/igd/`, `MoeaBench/objectives/`). Many of these directories contain only a single file, adding unnecessary complexity to the package structure.

### 4. Naming Inconsistency
The library uses a mix of CamelCase, snake_case, and custom prefixes (`I_`, `IPL_`, `GEN_`, `H_`). This leads to a less intuitive developer experience.

---

## Proposed Refactoring Plan

### Phase 1: Flatten the Core Data Model (The "Big Clean")
- **Eliminate Legacy Data Classes**: Replace `CACHE`, `DATA_conf`, and `DATA_arr` with direct storage in the `Run` and `experiment` classes.
- **Direct Data Return**: MOEA engines should return a list of `Population` objects (or a single Numpy tensor `G x P x M`) instead of the opaque `tuple` of legacy objects.
- **Simplify `Run`**: Remove the `_find_data_conf` hunting logic. `Run` should just store a list of `Population` objects.

### Phase 2: Modernize Benchmarks & Algorithms
- **Direct Instantiation**: Remove the `problems()` factory. Calling `mb.benchmarks.DTLZ1()` should directly return the benchmark object.
- **Simplify Inheritance**: Use standard Python `abc.ABC` for bases instead of the custom `I_` files.
- **Registration**: Use a centralized, decorator-based registration system in the `__init__.py` of `benchmarks/` and `moeas/`.

### Phase 3: Structural Reorganization
- **Modular Packaging**: Group functionality into cohesive packages:
  - `MoeaBench.core`: Experiment, Run, Population, SmartArray.
  - `MoeaBench.metrics`: Consolidated metrics (HV, IGD, etc.).
  - `MoeaBench.plotting`: Consolidated plotting backends.
  - `MoeaBench.benchmarks`: Problem definitions.
  - `MoeaBench.moeas`: Algorithm implementations.
- **Snake_case names**: Rename all files and internal methods to follow standard Python conventions.

### Phase 4: Performance & Typings
- **Fast Non-Dominated Sorting**: Implement a more efficient algorithm (like ENS or fast non-dominated sort from NSGA-II) to replace the current $O(N^2)$ loop.
- **Type Hinting**: Add comprehensive type hints across the entire library to improve IDE support and catch bugs.
- **Dataclasses**: Use `dataclasses` for simple data containers like `Population`.

---

## Next Steps
1. **Approve Plan**: User review of this architecture.
2. **Phase 1 Execution**: Start with the core data model to break dependencies on legacy code.
3. **Artifact Cleanup**: Gradually remove the ~40+ redundant files identified.
