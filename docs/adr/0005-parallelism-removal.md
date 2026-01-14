# ADR 0005: Suppression of Parallelism for RAM Stability

**Status**: Accepted  
**Date**: 2026-01-14

## Context
MoeaBench initially attempted to implement process-based parallelism (`multiprocessing`) to leverage multi-core CPUs during multi-run experiments. However, we encountered the "Catch-22" of Python concurrency:
- **Threads** are neutered by the Global Interpreter Lock (GIL) and cannot scale CPU-bound search tasks.
- **Processes** avoid the GIL but suffer from massive memory overhead due to process duplication.

On standard hardware (e.g., 8GB RAM), parallel execution of algorithms like SPEA2 or experiments involving complex visualization libraries (Plotly) consistently led to system crashes (Exit Code 9/OOM). 

## Technical Root Cause Analysis
We identified that the `multiprocessing.fork` method (default on Linux) was the primary culprit. While it theoretically uses **Copy-on-Write (COW)** to save memory, the sheer size of the imported libraries (`pymoo`, `plotly`, `scipy`) and their internal global states meant that COW was triggered almost immediately upon worker initialization. 

Spawning even 4 workers would effectively quadruple the RAM footprint of the library's overhead (approx. 300MB-500MB per process just for imports and GUI states), leaving zero room for the actual optimization data (populations/histories). Subsequent attempts using the `spawn` method solved the memory duplication but introduced fatal IPC (Inter-Process Communication) bottlenecks and complexity in sharing progress bar states via Queues.

## Decision
We decided to **completely remove all parallel execution support** from the library core.
1.  **Pure Serial Execution**: `experiment.run` and `attainment_diff` now execute serially, performing one run after another. 
2.  **Suppression of Logic**: We removed all `multiprocessing`, `concurrent.futures`, and `Manager().Queue()` logic to simplify the code.
3.  **Deprecation of `workers`**: The `workers` parameter is preserved in API signatures for backward compatibility but is explicitly marked as deprecated and ignored.

## Rationale: Stability over Peak Throughput
The overhead of process-based parallelism in Python often exceeds the computational gains for typical MOEA runs on standard hardware. By reverting to a serial model:
1.  **Memory Predictability**: The RAM usage remains constant regardless of the number of repetitions. 
2.  **System Responsiveness**: The user's machine remains responsive during the search.
3.  **Code Health**: The complex, bug-prone IPC code was replaced with simple, robust loops.

## Consequences
- **Positive**: 100% stability. Crashes (Killed/Exit Code 9) are eliminated.
- **Positive**: Significantly reduced library footprint and simplified installation.
- **Positive**: Easier debugging—serial stack traces are much clearer than parallel IPC errors.
- **Negative**: Long multi-run experiments (e.g., 100+ runs) will take significantly longer on high-core machines.
- **Neutral**: Users requiring high throughput are encouraged to use MOEAs that implement internal vectorization/parallelization at the C-level (within the evaluation function) rather than at the Python-process level.
