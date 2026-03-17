
<!--
SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# MoeaBench

MoeaBench is an extensible analytical toolkit for Multi-objective Evolutionary Optimization research that adds an intuitive abstraction layer over standard benchmark engines. By managing the complex and critical steps of configuring and executing sophisticated quantitative experiments, it elevates analysis to the realm of high-level data interpretation and visualization. This transforms raw performance metrics into descriptive, narrative-driven results, enabling rigorous algorithmic auditing and promoting systematic, reproducible experimental comparisons. MoeaBench transparently coordinates normalization, numerical reproducibility, and statistical validation, providing the means for programmatically establishing benchmark protocols and extracting standardized metrics.

## Key Features

- **Built-in Benchmark Problems and Algorithms**: MoeaBench includes a comprehensive suite of foundational benchmark problems (such as DTLZ and DPF), covering a wide range of analytical topographies. It also incorporates a collection of relevant algorithms (e.g., NSGA-III, MOEA/D, SPEA2). These algorithms are fully calibrated and ready to use, integrated with high-density samplings of problem Pareto Fronts and pre-calculated statistical baselines.

- **Many-Objective Scalability**: MoeaBench does not impose hard-coded constraints on the number of objectives a MOP can have. The framework uses a fully vectorized NumPy backend and KDTree spatial indexing to perform distance calculations against massive reference sets instantly. It automatically scales to an unlimited number of objectives by dynamically switching between exact calculations for low dimensions and efficient mathematical approximations (e.g., Monte Carlo sampling for Hypervolume) when exact analytical extraction becomes intractable.

- **Reproducibility**: MoeaBench guarantees experimental reproducibility by strictly tracking and managing the pseudo-random number generator (PRNG) seeds across all execution runs. The exact seed is recorded in the experiment metadata, which can be persistently exported alongside the results to perfectly reconstruct a trial. Moreover, performance evaluation is anchored to pre-computed, verified calibration files: the software provides the Ground Truth of each benchmark problem (the ideal Pareto Front) as a high-density sampling (currently up to 10,000 points) of the manifold—obtained analytically when possible, or through exhaustive numerical optimization. This ensures that algorithmic comparisons are always carried out against a standardized reference. Furthermore, relative performance is evaluated against pre-calculated baselines that represent boundary empirical results according to each problem's geometry and specific algorithmic capabilities.

- **Plug-and-Use Extensibility**: Users can introduce custom MOPs and MOEAs through a decoupled interface, without modifying the underlying source code. When a custom problem is integrated, MoeaBench automatically calibrates it, generating a dense spatial sampling of its Ground Truth and calculating the necessary performance scale baselines. This ensures new models are evaluated across the same rigorous references for fair and valid scientific comparison.

- **Replication and Statistical Confidence**: MoeaBench natively handles experimental replication by merging multiple stochastic runs into a unified analytical record. It seamlessly applies formal statistical hypothesis tests (e.g., Mann-Whitney U, A12 Effect Size), confidence intervals, and density distribution assessments over the aggregated data.

- **Contextual Narrative Reporting**: In addition to extracting quantitative metrics, MoeaBench translates abstract numerical data into contextual interpretation using plain text via the `.report()` method. This functionality provides human-readable summaries detailing the exact configuration, physical bounds, calibration status, and statistical meaning of the results, allowing researchers to perceive complex diagnostics in a rigorous yet intuitive manner.

- **Smart Arguments and Polymorphic Interfaces**: MoeaBench adopts a polymorphic public surface in which the same analytical function can accept experiments, runs, populations, or previously computed result objects. Instead of forcing the user to manually extract arrays and wire intermediate objects, the framework resolves the appropriate data source internally, including temporal slicing over generations when that context is available. This design reduces boilerplate while preserving a consistent scientific contract across metrics, statistics, diagnostics, and visualization.

- **Cloud-Centric Delegation and Multi-Run Aggregation**: MoeaBench treats the experiment as a statistical cloud rather than a mere container of isolated runs. High-level calls automatically aggregate repeated executions into a shared analytical object, exposing the collective geometry, dispersion, and performance trajectory of the search process. This allows the same API to support both per-run inspection and population-level stochastic interpretation without changing the conceptual vocabulary of the workflow.

- **Scientific Common Goods by Default (CC0 Fallback)**: MoeaBench embeds open-science principles directly into the experiment metadata contract. When authorship is not explicitly declared, persisted experiment artifacts fall back to **CC0-1.0**, promoting unrestricted scientific reuse, transparent redistribution, and frictionless archival exchange. This policy makes the absence of authorship metadata scientifically meaningful rather than merely incomplete, aligning computational experimentation with reproducible research practices.

- **Zero-Config Reporting and Metadata Introspection**: Beyond narrative reporting itself, MoeaBench performs introspective metadata recovery to reduce manual bookkeeping. Variable names, experiment names, algorithm settings, licensing context, and execution structure are automatically surfaced whenever possible, allowing the reporting layer to produce immediate scientific narratives with minimal user intervention. In practice, this means that diagnostic context is not an afterthought but a first-class part of the computational object model.

- **Diagnostic-Driven Visualization**: MoeaBench deploys specialized visual instruments categorized into Topography (spatial coverage, geometric bounds, and attainment bands), Performance (temporal convergence trajectories), and Stratification (Pareto-rank distributions). These include holistic tools like the Clinical Radar plot, designed to quickly identify complex structural failures such as diversity collapse or manifold bias.

- **Data Persistence and Portability**: The framework implements native `save()` and `load()` methods to persist complete experiment states—including hyperparameter configurations, resulting population matrices, and mandatory scientific metadata (e.g., authorship, SPDX/CC0 licensing)—into standard ZIP archives, enabling full data portability, independent auditing, and rigorous peer review.

- **Finite Approximation-Inherent Resolution (FAIR) Metrics Framework (FAIR Framework)**: MoeaBench introduces FAIR Framework to address the issue of measuring algorithmic error against a continuous analytical manifold when both the problem's Ground Truth (GT) and the algorithm's population are discrete, finite point sets. It calculates a resolution-scale parameter ($s_K$) based on theoretical spatial expectations to produce **FR physical metrics** that map raw distances into standardized resolution units (e.g., Closeness to GT, Coverage, Gaps, Balance, and Regularity). These physical dimensions are subsequently evaluated via "Clinical" Q-Scores (e.g., `q_closeness`, `q_gap`) which translate numerical discrepancies against problem- and algorithm-specific noise baselines, resulting in a scale-invariant diagnostic assessment of an algorithm's structural integrity.

- **Comparability and Normalization**: When calculating performance indicators such as Hypervolume, MoeaBench can report the numerical output as a raw physical scalar or normalize the outcome against contextual reference thresholds. This unified relative scaling allows for straightforward comparative assertions (e.g., algorithm A performs 20% better than algorithm B) alongside evaluations normalized relative to the analytical ideal front (e.g., reaching 90% of the theoretical maximum performance).





