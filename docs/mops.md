<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Multi-Objective Problems (MOPs): DTLZ and DPF in MoeaBench

In the world of Multi-Objective Evolutionary Algorithms (MOEAs), benchmarks are not just sets of equations—they are the proving grounds where performance distributions are mapped and algorithmic limits are tested.

MoeaBench provides native, high-performance implementations of two critical benchmark families: **DTLZ** and **DPF**. Below, we document our journey in implementing these methods, from their theoretical foundations to our optimized, vectorized codebase.

---

## 1. The DTLZ Family: The Scalability Gold Standard

The **DTLZ** (Deb-Thiele-Laumanns-Zitzler) problem set is arguably the most influential benchmark family in the MOEA community. Introduced in 2002, these problems were designed to be scalable in both the number of objectives ($M$) and decision variables ($N$).

### The Technical Challenge
In many legacy implementations, DTLZ problems were coded using nested loops to evaluate one individual at a time. While this approach is functional, it fails as we scale to many-objective optimization (10+ objectives).

### Our Narrative: The Shift to Vectorization
When we refactored the DTLZ engine for MoeaBench, we prioritized "matrix thinking." We realized that the standard DTLZ evaluation functions (spherical, linear, etc.) could be entirely expressed through linear algebra and broadcasting.

For example, in `DTLZ1`, the $g$ factor calculation:
$$g = 100 \left[ K + \sum_{i \in M} (x_i - 0.5)^2 - \cos(20\pi(x_i - 0.5)) \right]$$

In our `DTLZ1.py`, this is calculated for the entire population in a single NumPy call, bypassing Python's slow iteration layer. This optimization allows researchers to run 30+ repetitions of massive experiments without their hardware grinding to a halt.

**Reference:**
*   K. Deb, L. Thiele, M. Laumanns, and E. Zitzler. "Scalable multi-objective optimization test problems." *Proc. IEEE Congress on Evolutionary Computation (CEC)*, 2002.

---

## 2. The DPF Family: Conquering Degeneracy

While DTLZ problems are excellent for testing search capability, real-world problems often exhibit **Degenerate Pareto Fronts**. A front is degenerate when its dimensionality is lower than $M-1$.

### The Zhen et al. (2018) Approach
The **DPF** (Degenerate Pareto Front) benchmarks, introduced by Zhen et al., address this by projecting a lower-dimensional Pareto front into a higher-dimensional objective space.

### Our Narrative: Implementing the Chaotic Projection
Implementing DPF was significantly more complex than DTLZ. The core logic involves a "Projector" that uses chaotic weights (based on the Logistic Map) to ensure the degenerate front is distributed in a complex, non-trivial way across the $M$-dimensional space.

We spent significant effort ensuring that our `BaseDPF` implementation matched the mathematical nuances of the original paper, specifically the nuances of the chaotic weight generation. We implemented this logic in `MoeaBench/mops/base_dpf.py`, ensuring that even these complex projections are vectorized for performance.

**Reference:**
*   L. Zhen, M. Li, R. Cheng, D. Peng, and X. Yao. "Multiobjective test problems with degenerate Pareto fronts." *IEEE Transactions on Evolutionary Computation*, vol. 22, no. 5, 2018.

---

## 3. The Implementation Journey: From Legacy to Clean API

MoeaBench was born from the need to move away from fragmented script-based benchmarks. In early iterations, we had multiple `I_*.py` files with conflicting evaluation logics. 

We consolidated everything into a clean, object-oriented structure:
1.  **Base Classes**: All problems now inherit from `BaseMop`, ensuring consistent metadata handling.
2.  **Analytical Optimals**: We implemented `.optimal()`, `.optimal_front()`, and `.optimal_set()` for all DTLZ and DPF problems. This allows you to visualize your results against the theoretical truth with a single command: `mb.view.spaceplot(exp.optimal(), exp)`.
3.  **Independence & Seeds**: We ensured that every run is mathematically unique by tying the problem's random states to the Experiment's seed logic.

This documentation serves not just as a manual, but as a record of our commitment to technical rigor and high-performance analysis.
