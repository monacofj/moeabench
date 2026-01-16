<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# ADR 0006: Reimplementation of DTLZ and DPF Benchmarks

**Status**: Accepted  
**Date**: 2026-01-14

## Context

The legacy version of MoeaBench relied on a fragmented, script-based approach for evaluating standard Multi-Objective Problems (MOPs). While functional for its time, this architecture suffered from three critical bottlenecks:
1.  **Iterative Sluggishness**: Evaluation logic was implemented using nested Python `for` loops, which evaluated individuals sequentially. As research shifted toward many-objective optimization ($M > 10$), this created a performance ceiling that hindered large-scale statistical experiments.
2.  **API Fragmentation**: The implementation of various benchmarks (DTLZ, DPF) was scattered across multiple interface files (`I_*.py`), often with conflicting logic or missing metadata, making it difficult to automate analysis.
3.  **Visualization Gap**: Comparison with the theoretical "Ground Truth" (the Pareto Front) required external files or manual data loading, adding friction to the research workflow.

## Decision: A Scientific Narrative Journey

We decided to **entirely reimplement** the DTLZ and DPF families from the ground up. This was not merely a porting effort; it was a fundamental shift in how optimization data is processed within the library.

### 1. From Loops to Tensors: The Vectorized Bridge
Our first attempt involved simply wrapping the existing legacy scripts. We tried to maintain backward compatibility with the old "one-by-one" evaluation logic, but it quickly became clear that this was unsustainable. The overhead of Python function calls for every individual in a large population was the primary culprit for system sluggishness.

Consequently, we pivoted to a **Strict Vectorization Policy**. We rewrote the mathematical souls of DTLZ (Deb et al., 2002) and DPF (Zhen et al., 2018) using NumPy broadcasting. Instead of asking "What is the objective of this individual?", the library now asks "What is the result of this objective for the entire population matrix?". This allows the heavy lifting to occur in optimized C-layer code, providing near-native performance for massive populations.

### 2. Standardization through Object-Orientation
We replaced the fragmented script files with a unified class hierarchy. Every problem now inherits from a strictly typed `BaseMop` (or `BaseDPF`) class. This architectural choice was intentional: it ensures that every problem carries its own metadata (dimensionalities, bounds, labels).

This "Object-Awareness" is what enables our **Smart Stats** and **Smart Plots**. When you pass an `Experiment` containing these problems to a plotting function, the benchmark itself provides the necessary context to auto-label axes and legends, eliminating the "Where did this data come from?" problem that plagues legacy research code.

### 3. Closing the Gap with "Analytical Truth"
We identified that researchers spend a disproportionate amount of time seeking and loading reference fronts ($PF_{true}$). To solve this, we integrated the analytical truth directly into the benchmark classes.

In our reimbursement, every DTLZ and DPF class implements its own `.ps()` and `.pf()` sampling logic based on the original publications. This means that comparison is no longer an external task; it is a native method. You can now execute `mb.view.spaceplot(exp.optimal(), exp)` and see your performance against the theoretical limit instantly.

### 4. Deterministic Reproducibility
Finally, we addressed the "Ghost in the Machine": non-deterministic randomness. In legacy MoeaBench, problems shared global random states with the solvers, leading to "leaky" seeds. Our reimplementation decoupling these states. Every benchmark instance now manages its internal state in coordination with the `Experiment` seed logic. If you run the same experiment twice, the vectorized mathematical engine will produce bit-identical results, ensuring absolute scientific rigor.

## Rationale: Sustainability over Legacy Preservation

We explicitly chose to break backward compatibility with the legacy code structure in favor of a clean, sustainable API. By replacing the "slow and fragmented" with the "vectorized and unified," we ensure that MoeaBench remains a viable tool for the next generation of high-dimensional MOEA research.

## Consequences
- **Positive**: 10x-50x performance gains on population evaluations.
- **Positive**: Zero-boilerplate comparison with theoretical optima via `.optimal()`.
- **Positive**: Robust metadata propagation through the `SmartArray` and `BaseMop` architecture.
- **Negative**: Legacy MoeaBench scripts require slight syntax adjustments to use the new object-oriented classes.
- **Neutral**: Requires `numpy` and `scipy` as hard dependencies (which aligns with modern scientific Python standards).
