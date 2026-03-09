<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench: An Extensible Analytical Toolkit for Scientific Auditing of Evolutionary Algorithms

**Abstract**

MoeaBench is a Python-based analytical framework designed to elevate the standard of empirical research in Multi-objective Evolutionary Optimization (MOEA). Unlike traditional libraries that focus primarily on the execution of optimization loops, MoeaBench establishes a semantic abstraction layer dedicated to data interpretation, rigorous calibration, and "Clinical Metrology." This document details the architectural philosophy of the toolkit, explaining how it addresses the challenges of high-dimensional benchmarking through a novel 3-layer diagnostic architecture.

---

## 1. Introduction

In the field of Evolutionary Computation, the complexity of benchmarking has grown significantly with the advent of Many-Objective Optimization (MaOO). As problems scale to 3, 5, or 10 objectives, the geometric properties of the search space change fundamental, rendering traditional performance metrics (like raw IGD or Hypervolume) difficult to interpret in isolation.

A common pitfall in modern research is the "Black Box" approach: algorithms are run, tables of numbers are generated, and bold claims are made based on marginal differences in uncalibrated metrics. This lacks **Scientific Auditing**—the ability to explain *why* an algorithm performed the way it did.

MoeaBench was designed to fill this gap. It is not merely a library for running algorithms; it is an **Instrument of Insight**. Its primary design goal is to transform raw stochastic data into structured scientific evidence, ensuring that every result is reproducible, statistically significant, and physically meaningful.

---

## 2. Key Technical Features

Before discussing its scientific innovations, it is essential to understand the technical foundation that makes MoeaBench a robust research platform. The framework is built upon several core engineering pillars:

### 2.1. Plugin Architecture (Host-Guest Pattern)
MoeaBench employs a "Host-Guest" design pattern. The framework acts as the **Host**, providing infrastructure (metrics, plotting, statistics, seed management), while the user's code (Algorithms and Problems) acts as the **Guest**.
*   **Extensibility**: Users can plug in custom MOEAs and MOPs by inheriting from simple base classes (`BaseMoea`, `BaseMop`).
*   **Decoupling**: The core analysis logic is completely decoupled from the optimization logic, ensuring that diagnostics work identically for built-in algorithms (like NSGA-III) and user-defined experimental heuristics.

### 2.2. Vectorized Engine
To support massive experiments without performance degradation, MoeaBench enforces a **Vectorized Architecture**.
*   **Loop-Free Policy**: Critical paths—including metric calculation (IGD, GD), non-dominated sorting, and reference point generation—are implemented using optimized **NumPy** broadcasting.
*   **Scalability**: This allows the toolkit to handle populations of thousands of individuals in high-dimensional spaces ($M > 10$) with near-native C performance, avoiding the bottlenecks of Python loops.

### 2.3. Determinism and Cloud Aggregation
Scientific reproducibility is non-negotiable.
*   **Seed Management**: MoeaBench treats random seeds as first-class metadata. Every experiment manages a deterministic chain of seeds, ensuring that any specific run can be perfectly reconstructed.
*   **Cloud Aggregation**: The `Experiment` object automatically aggregates data from multiple stochastic trials. Analysis tools operate on this "Cloud" of data, forcing researchers to look at the *distribution* of performance (mean, variance, outliers) rather than cherry-picked "best runs."

### 2.4. Visual Intelligence
Visualization is treated as a form of structural evidence.
*   **Polymorphism**: Plotting functions (`mb.view.*`) are metric-agnostic. A "Timeplot" can visualize the evolution of Hypervolume, IGD, or even Selection Pressure with the same API.
*   **Diagnostic Instruments**: Visuals like the **Clinical Radar** and **ECDF Plots** are designed to reveal algorithmic pathologies, not just display data.

---

## 3. The Challenge of High-Dimensional Benchmarking

To understand the scientific contribution of MoeaBench, we must first address the fundamental problem it solves: the failure of "Naive Metrology" in high-dimensional spaces.

### 3.1. The Sparsity Law (The Curse of Dimensionality)
In a 2-objective problem ($M=2$), a population of $N=100$ individuals can cover the Pareto Front with high density. However, as the number of objectives increases, the manifold expands exponentially.

We define the **Sparsity Law**:
$$ R_{cov} \propto N^{-\frac{1}{M-1}} $$

For a problem with $M=10$ objectives, a population of $N=100$ is exponentially sparse. The "empty space" between solutions becomes vast. In this regime, comparing an algorithm against a fixed threshold (e.g., "IGD must be $< 0.01$") is scientifically invalid. It is physically impossible for a finite population to achieve that density.

### 3.2. The "Phantom Intrusion" Paradox
Another critical issue in benchmarking is establishing a baseline for "Random Performance." Traditionally, researchers generate a random baseline by adding spherical Gaussian noise to the Ground Truth points.

*   **The Problem**: The Pareto Front acts as a "Hard Wall" in objective space—it is the theoretical limit of optimality. A spherical noise distribution (which extends in all directions) will mathematically place some baseline points *behind* this wall (i.e., making them "better" than optimal).
*   **The Consequence**: This creates "Phantom Intrusions"—the baseline effectively cheats. If we compare an algorithm against this flawed baseline, the algorithm appears artificially worse because it is competing against impossible phantom points.

---

## 4. The Clinical Metrology Framework

MoeaBench resolves these challenges by introducing a **Clinical Metrology** architecture. This system shifts the focus from measuring "Distance to Zero" (which is impossible in high dimensions) to measuring "Algorithmic Health" relative to problem difficulty.

This is implemented via a **3-Layer Architecture**:

### Layer 1: The Physical Layer (FAIR Metrics)

For detailed mathematical definitions, see the [FAIR Metrics documentation](fair.md).
This layer handles the raw physics of measurement, corrected for the Sparsity Law.

*   **Resolution Scale ($s_K$)**: Instead of using an absolute ruler, we define a "Macroscopic Ruler." $s_K$ is calculated as the expected nearest-neighbor distance of a finite random population of size $K$.
*   **Normalization**: All physical metrics (Closeness, Coverage, etc.) are divided by $s_K$. This converts absolute error into **Resolution Units**.
    *   *Result*: A score of `1.0` means "The error is equal to the expected resolution of the population." This makes results comparable across different dimensions.

**The Half-Normal Innovation**:
To solve the "Phantom Intrusion" paradox (Section 3.2), MoeaBench models the random baseline using a **Half-Normal Projection**.
*   *Concept*: Instead of a sphere, imagine the noise as a "bouncing ball." It hits the Pareto Wall and bounces back.
*   *Math*: We use the absolute value of a normal distribution ($| \mathcal{N}(0, \sigma^2) |$).
*   *Effect*: This ensures that the baseline noise is strictly *worse* than or equal to the optimum. It respects the geometric integrity of the manifold, providing a scientifically rigorous floor for comparison.

### Layer 2: The Clinical Layer (Q-Scores)
This layer transforms physical measurements into a universal quality grade.

*   **The Need for Grading**: A physical error of `2.5 s_K` might be excellent for a degenerate problem but terrible for a simple linear problem. We need a normalized grade.
*   **ECDF Normalization**: Instead of linear scaling (which is sensitive to outliers), MoeaBench uses the **Empirical Cumulative Distribution Function (ECDF)** of the random baseline.
*   **The Q-Score**:
    $$ Q = 1 - ECDF_{random}(metric) $$
    *   **$Q \approx 1.0$ (Asymptotic)**: The result is indistinguishable from the theoretical optimum.
    *   **$Q \approx 0.5$ (Random)**: The result is equivalent to a random guess.
    *   **$Q \approx 0.0$ (Failure)**: The result is actively worse than random (misguided search).

---

## 5. Diagnostic Ontology

MoeaBench defines a comprehensive ontology of "Algorithmic Pathologies." We do not just measure "Performance"; we diagnose specific symptoms of failure using the Clinical Layers:

1.  **Closeness (Convergence)**
    *   *Symptom*: Stagnation.
    *   *Metric*: Distance from Population to Ground Truth ($P \to GT$).
    *   *Diagnosis*: Is the population pushing against the Pareto Wall?

2.  **Coverage (Completeness)**
    *   *Symptom*: Boundary Recession.
    *   *Metric*: Distance from Ground Truth to Population ($GT \to P$).
    *   *Diagnosis*: Has the algorithm found the *entire* manifold, or just the easy center?

3.  **Gap (Continuity)**
    *   *Symptom*: Topological Fracture.
    *   *Metric*: The 95th percentile of the nearest-neighbor distances.
    *   *Diagnosis*: Are there large holes or discontinuities in the approximation?

4.  **Regularity (Uniformity)**
    *   *Symptom*: Clustering / Genetic Drift.
    *   *Metric*: Wasserstein distance (EMD) to a uniform lattice.
    *   *Diagnosis*: Are the solutions evenly spaced, or are they clumping together?

5.  **Balance (Fairness)**
    *   *Symptom*: Mode Collapse.
    *   *Metric*: Jensen-Shannon divergence across topological clusters.
    *   *Diagnosis*: Is the algorithm favoring one region of the objective space over others?

6.  **Headway (Drive)**
    *   *Symptom*: Lack of Search Pressure.
    *   *Metric*: Log-linear progress from the initial random state.
    *   *Diagnosis*: How much "work" has the algorithm actually done to escape entropy?

---

## 6. Conclusion

MoeaBench represents a paradigm shift in Evolutionary Optimization benchmarking. By moving from **Naive Metrology** (raw numbers) to **Clinical Metrology** (normalized diagnostics), it provides researchers with the vocabulary and tools to conduct rigorous scientific audits.

Through its 3-layer architecture, geometric innovations like the Half-Normal Projection, and comprehensive diagnostic ontology, MoeaBench ensures that the "Why" of algorithmic performance is just as accessible as the "How."
