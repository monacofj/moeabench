<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0003: NumPy Vectorization and "Near-Native" Performance

**Status**: Accepted  
**Date**: 2026-01-14

## Context
Python is notoriously slow for iterative numerical tasks (like evaluating a population of 1000 individuals across 500 generations). In the legacy codebase, most Multi-Objective Problems (MOPs) and performance metrics were implemented using nested `for` loops. This created a significant performance ceiling, especially when running the high-dimensional benchmarks (e.g., DTLZ with 10+ objectives).

## Decision
We decided to enforce a **Strict Vectorization Policy**. All core mathematical operations in MoeaBench must be implemented using NumPy broadcasting and vectorized functions to bypass Python-level iteration.

### Key Implementation Details
1.  **Vectorized MOPs**: Standard problems (like the DTLZ family) were rewritten to evaluate entire population matrices in a single NumPy call. For example, instead of looping through individuals, we use broadcasting (`x**2` applied to a multi-dimensional array) which executes in optimized C/Fortran code.
2.  **SmartArray Metadata Propagation**: We implemented the `SmartArray` class (a NumPy ndarray subclass) to ensure that metadata (source run, generation index) survives these vectorized operations. Without this, NumPy would strip the metadata, breaking our analysis tools.
3.  **Vectorized Dominance Calculations**: The non-dominated sorting logic was refactored to use chunked broadcasting, balancing the speed of O(N^2) comparisons with memory safety.

## Rationale
"Python is a great way to write C code without writing C." By using NumPy correctly, we offload the heavy lifting to native code while keeping the high-level logic flexible and readable. This allows MoeaBench to perform like a library written in C/C++, provided the operations are sufficiently large to amortize the minor overhead of the Python-to-C bridge.

## Consequences
- **Positive**: Huge performance gains (often 10x-50x faster) for population evaluation and metric calculation.
- **Positive**: Implementation scripts for MOPs are now more concise and closer to their mathematical definitions.
- **Negative**: Vectorized code is often harder to debug for beginners, as "matrix thinking" replaces standard logic.
- **Negative**: Broadcasting can lead to high peak memory usage (e.g., creating large temporary tensors). We mitigated this by implementing "chunked" evaluations in critical paths like `_calc_domination`.
